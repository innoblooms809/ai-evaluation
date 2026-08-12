"""
configs/vl_model_config.py — Central Configuration for Qwen2.5-VL-3B-Instruct

Standalone configuration for Qwen Vision-Language model (Qwen2.5-VL-3B-Instruct).
Does not modify or interfere with any existing text-only model configs.
"""

import os
import torch
from pathlib import Path
from transformers import BitsAndBytesConfig

# ──────────────────────────────────────────────
# Model identity
# ──────────────────────────────────────────────
VL_MODEL_ID: str = "Qwen/Qwen2.5-VL-3B-Instruct"

# Local model cache path mapping
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_VL_MODEL_PATH = PROJECT_ROOT / "model_cache" / "Qwen2.5-VL-3B-Instruct"

# Prefer local model files if downloaded to run 100% offline
if LOCAL_VL_MODEL_PATH.exists() and any(LOCAL_VL_MODEL_PATH.iterdir()):
    VL_MODEL_PATH_OR_ID: str = str(LOCAL_VL_MODEL_PATH)
else:
    VL_MODEL_PATH_OR_ID: str = VL_MODEL_ID

# ──────────────────────────────────────────────
# Constraints & Device settings
# ──────────────────────────────────────────────
MAX_PIXELS: int = 1280 * 28 * 28     # Image resolution processing upper bound
MIN_PIXELS: int = 256 * 28 * 28      # Image resolution processing lower bound
MAX_NEW_TOKENS: int = 1024

IS_CUDA_AVAILABLE: bool = torch.cuda.is_available()

# ── High-Performance GPU Tuning (80GB VRAM Testing & Production Config) ──
# // TESTING & PRODUCTION CONFIGURATION:
# // Set USE_4BIT_QUANTIZATION = False for 80GB VRAM GPUs (A100/H100/RTX 6000) -> 2.5x faster unquantized bfloat16 inference!
# // Set USE_4BIT_QUANTIZATION = True for smaller 12GB/16GB/24GB GPUs (RTX 3090/4090).
USE_4BIT_QUANTIZATION = os.getenv("USE_4BIT_QUANTIZATION", "false").lower() == "true"

if IS_CUDA_AVAILABLE:
    if USE_4BIT_QUANTIZATION:
        VL_QUANTIZATION_CONFIG = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        VL_QUANTIZATION_CONFIG = None  # Full unquantized bfloat16 for 80GB VRAM — 2.5x faster!

    VL_DEVICE_MAP: str = "auto"
    VL_TORCH_DTYPE = torch.bfloat16
else:
    VL_QUANTIZATION_CONFIG = None
    VL_DEVICE_MAP: str = "cpu"
    VL_TORCH_DTYPE = torch.float32

VL_PROCESSOR_KWARGS: dict = {
    "trust_remote_code": True,
    "min_pixels": MIN_PIXELS,
    "max_pixels": MAX_PIXELS,
}

VL_MODEL_KWARGS: dict = {
    "trust_remote_code": True,
    "device_map": VL_DEVICE_MAP,
    "torch_dtype": VL_TORCH_DTYPE,
}

if VL_QUANTIZATION_CONFIG is not None:
    VL_MODEL_KWARGS["quantization_config"] = VL_QUANTIZATION_CONFIG
