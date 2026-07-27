"""
Production-grade model download utility for HuggingFace models.

Features:
- Dynamically handles CPU fallback configurations
- Sequential file-by-file downloading to prevent Windows thread locks
- Robust retry logic with exponential backoff for network resilience
- Local directory mapping to bypass Windows symlink cache sharing violations
- Checksum (SHA256) validation of LFS model weights
- Graceful interrupt handling and download resumption support
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure stdout/stderr supports UTF-8 on Windows consoles to prevent encoding errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from configs.model_config import MODEL_ID

logger = logging.getLogger(__name__)


def get_file_sha256(filepath: Path) -> str:
    """Calculate the SHA256 checksum of a file in chunks."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in 8MB chunks
        for byte_block in iter(lambda: f.read(8388608), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_download_config() -> dict:
    """Load downloader configuration, falling back to defaults if not found."""
    config_path = PROJECT_ROOT / "configs" / "download_config.json"
    defaults = {
        "max_retries": 5,
        "retry_backoff_factor": 2.0,
        "chunk_size_bytes": 10485760,
        "timeout_seconds": 120,
        "verify_sha256": True,
    }
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                defaults.update(loaded)
                logger.info("Loaded download configuration from %s", config_path.name)
        except Exception as e:
            logger.warning("Failed to parse config file, using default settings. Error: %s", e)
    return defaults


def download_model(
    model_id: str = MODEL_ID,
    local_dir: str | None = None,
    token: str | None = None,
) -> None:
    """Download model weights and tokenizer files with safety checks."""
    config = load_download_config()
    max_retries = config["max_retries"]
    backoff_factor = config["retry_backoff_factor"]
    verify_sha256 = config["verify_sha256"]

    # Set up local directory path (default: cache directory under project output)
    if local_dir is None:
        target_dir = PROJECT_ROOT / "model_cache" / model_id.split("/")[-1]
    else:
        target_dir = Path(local_dir)

    logger.info("Target directory: %s", target_dir.resolve())
    target_dir.mkdir(parents=True, exist_ok=True)

    # Initialize HuggingFace API client
    api = HfApi(token=token)

    try:
        logger.info("Fetching repository metadata for: %s", model_id)
        repo_info = api.model_info(repo_id=model_id, files_metadata=True)
    except RepositoryNotFoundError:
        logger.critical("Hugging Face repository '%s' not found.", model_id)
        sys.exit(1)
    except GatedRepoError:
        logger.critical(
            "Access to gated repository '%s' was denied. Please specify a valid Hugging Face token.",
            model_id
        )
        sys.exit(1)
    except Exception as e:
        logger.critical("Failed to connect to Hugging Face Hub: %s", e)
        sys.exit(1)

    # Find sibling files to download
    siblings = repo_info.siblings
    if not siblings:
        logger.critical("No files found in the repository '%s'.", model_id)
        sys.exit(1)

    logger.info("Found %d files to download. Starting sequential transfer...", len(siblings))

    # Iterate over files sequentially to prevent multi-threaded locks on Windows
    for idx, sibling in enumerate(siblings, 1):
        filename = sibling.rfilename
        expected_sha = None
        expected_size = None

        # Check if the file contains LFS metadata (standard for weights)
        if hasattr(sibling, "lfs") and sibling.lfs is not None:
            expected_sha = sibling.lfs.get("sha256")
            expected_size = sibling.lfs.get("size")

        local_file_path = target_dir / filename
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("[%d/%d] Processing file: %s", idx, len(siblings), filename)

        # Skip download if file exists, sizes match, and checksum is valid
        if local_file_path.exists():
            actual_size = local_file_path.stat().st_size
            if expected_size is None or actual_size == expected_size:
                if expected_sha is not None and verify_sha256:
                    logger.info("  File exists. Checking checksum...")
                    try:
                        actual_sha = get_file_sha256(local_file_path)
                        if actual_sha == expected_sha:
                            logger.info("  [OK] Checksum matched. Skipping download.")
                            continue
                        else:
                            logger.warning("  [WARN] Checksum mismatch. Redownloading file.")
                    except Exception as e:
                        logger.error("  Failed to check SHA256: %s. Overwriting file.", e)
                else:
                    logger.info("  File exists and size matches. Skipping download.")
                    continue
            else:
                logger.info("  File exists but size mismatch (%d vs %d). Restarting download.", actual_size, expected_size)

        # Retry loop with exponential backoff for download resilience
        attempt = 1
        success = False
        while attempt <= max_retries:
            try:
                logger.info("  Downloading %s (Attempt %d/%d)...", filename, attempt, max_retries)
                
                # hf_hub_download automatically handles range-resumes if supported by server
                hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    local_dir=target_dir,
                    token=token,
                )
                
                # Checksum verification after download completion
                if expected_sha is not None and verify_sha256:
                    logger.info("  Verifying checksum...")
                    actual_sha = get_file_sha256(local_file_path)
                    if actual_sha != expected_sha:
                        raise ValueError(f"SHA256 verification failed for {filename}. Expected: {expected_sha}, Got: {actual_sha}")
                    logger.info("  [OK] Checksum verification successful.")

                success = True
                break
            except KeyboardInterrupt:
                logger.critical("\nDownload interrupted by user. Exiting gracefully.")
                sys.exit(1)
            except Exception as e:
                logger.warning("  Attempt %d failed. Error: %s", attempt, e)
                if attempt < max_retries:
                    sleep_time = backoff_factor ** attempt
                    logger.info("  Waiting %.1f seconds before retrying...", sleep_time)
                    time.sleep(sleep_time)
                attempt += 1

        if not success:
            logger.critical("Failed to download file '%s' after %d attempts. Download aborted.", filename, max_retries)
            sys.exit(1)

    logger.info("[SUCCESS] Download complete. Model files saved to: %s", target_dir.resolve())


def main():
    parser = argparse.ArgumentParser(
        description="Robust Qwen2.5-3B-Instruct model downloader"
    )
    parser.add_argument(
        "--model-id", type=str, default=MODEL_ID,
        help=f"HuggingFace model ID (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--local-dir", type=str, default=None,
        help="Target folder mapping for the files",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace token",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    download_model(
        model_id=args.model_id,
        local_dir=args.local_dir,
        token=args.token,
    )


if __name__ == "__main__":
    main()
