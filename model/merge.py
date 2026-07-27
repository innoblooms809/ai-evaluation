"""
LoRA merge utility: merges LoRA weights into the base model for
simplified deployment (single model, no adapter loading at runtime).

This is OPTIONAL — inference works fine with base + adapter.  Merging
is useful when:
  • Deploying to environments that don't support PEFT
  • Serving via vLLM, TGI, or other inference engines
  • Distributing a single-file model

WARNING: The merged model will be in full precision (bf16) and will be
larger than the 4-bit base + tiny adapter.  For memory-constrained
inference, prefer the adapter approach.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.model_config import MODEL_ID, MODEL_PATH_OR_ID

logger = logging.getLogger(__name__)


def merge_lora_adapter(
    adapter_path: str | Path,
    output_path: str | Path,
    model_id: str = MODEL_PATH_OR_ID,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
) -> None:
    """Merge a trained LoRA adapter into the base model and save.

    Parameters
    ----------
    adapter_path : str | Path
        Directory containing the saved LoRA adapter.
    output_path : str | Path
        Where to save the merged model.
    model_id : str
        HuggingFace base model ID.
    push_to_hub : bool
        If True, push merged model to HuggingFace Hub.
    hub_repo_id : str | None
        Hub repository ID (required if push_to_hub is True).
    """
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)

    logger.info("Loading base model (full precision for merge): %s", model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Loading LoRA adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    logger.info("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    logger.info("Saving merged model to %s", output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_path))

    # Save tokenizer alongside
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_path))

    logger.info("Merged model saved [SUCCESS]")

    if push_to_hub and hub_repo_id:
        logger.info("Pushing to Hub: %s", hub_repo_id)
        merged_model.push_to_hub(hub_repo_id)
        tokenizer.push_to_hub(hub_repo_id)
        logger.info("Pushed to Hub [SUCCESS]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter-path", type=str, required=True,
                        help="Path to saved LoRA adapter directory")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save merged model")
    parser.add_argument("--model-id", type=str, default=MODEL_PATH_OR_ID,
                        help=f"Base model ID (default: {MODEL_PATH_OR_ID})")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push merged model to HuggingFace Hub")
    parser.add_argument("--hub-repo-id", type=str, default=None,
                        help="Hub repository ID")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    merge_lora_adapter(
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        model_id=args.model_id,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
    )
