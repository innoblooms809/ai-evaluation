"""
Model and quantization configuration for Qwen2.5-3B-Instruct.

Centralises all model-related constants so every module references
one source of truth.
"""

import os
import torch
from transformers import BitsAndBytesConfig

from pathlib import Path

# ──────────────────────────────────────────────
# Model identity
# ──────────────────────────────────────────────
MODEL_ID: str = "Qwen/Qwen2.5-3B-Instruct"

# Local model cache path mapping
LOCAL_MODEL_PATH = Path(__file__).resolve().parent.parent / "model_cache" / MODEL_ID.split("/")[-1]

# Prefer local model files if downloaded to run 100% offline
if LOCAL_MODEL_PATH.exists() and any(LOCAL_MODEL_PATH.iterdir()):
    MODEL_PATH_OR_ID: str = str(LOCAL_MODEL_PATH)
else:
    MODEL_PATH_OR_ID: str = MODEL_ID

# ──────────────────────────────────────────────
# Sequence limits
# ──────────────────────────────────────────────
MAX_SEQ_LENGTH: int = 1024          # Qwen2.5 supports up to 32 768; 1024 is plenty for short Q&A
PADDING_SIDE: str = "right"         # Right-padding for causal LM training

# ──────────────────────────────────────────────
# CUDA Availability Check
# ──────────────────────────────────────────────
IS_CUDA_AVAILABLE: bool = torch.cuda.is_available()

# ──────────────────────────────────────────────
# 4-bit quantization (QLoRA) config
# ──────────────────────────────────────────────
# ── High-Performance GPU Tuning (80GB VRAM Testing & Production Config) ──
# // TESTING & PRODUCTION CONFIGURATION:
# // Set USE_4BIT_QUANTIZATION = False for 80GB VRAM GPUs (A100/H100/RTX 6000) -> 2.5x faster unquantized bfloat16 inference!
# // Set USE_4BIT_QUANTIZATION = True for smaller 12GB/16GB/24GB GPUs (RTX 3090/4090).
USE_4BIT_QUANTIZATION = os.getenv("USE_4BIT_QUANTIZATION", "false").lower() == "true"

if IS_CUDA_AVAILABLE:
    if USE_4BIT_QUANTIZATION:
        QUANTIZATION_CONFIG = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",                     # NormalFloat4 — best for fine-tuning
            bnb_4bit_compute_dtype=torch.bfloat16,          # Compute in bf16 for stability
            bnb_4bit_use_double_quant=True,                 # Double quantization saves ~0.4 bits/param
        )
    else:
        QUANTIZATION_CONFIG = None  # Full unquantized bfloat16 for 80GB VRAM — 2.5x faster!

    DEVICE_MAP: str = "auto"
    TORCH_DTYPE = torch.bfloat16
else:
    QUANTIZATION_CONFIG = None
    DEVICE_MAP: str = "cpu"
    TORCH_DTYPE = torch.float32


# ──────────────────────────────────────────────
# Tokenizer settings
# ──────────────────────────────────────────────
TOKENIZER_KWARGS: dict = {
    "trust_remote_code": True,
    "padding_side": PADDING_SIDE,
}

# ──────────────────────────────────────────────
# Model loading kwargs
# ──────────────────────────────────────────────
MODEL_KWARGS: dict = {
    "trust_remote_code": True,
    "device_map": DEVICE_MAP,
    "dtype": TORCH_DTYPE,
    "attn_implementation": "eager",     # Use "flash_attention_2" if you have it installed
}

if QUANTIZATION_CONFIG is not None:
    MODEL_KWARGS["quantization_config"] = QUANTIZATION_CONFIG
