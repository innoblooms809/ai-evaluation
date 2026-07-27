"""
Model loader: handles loading Qwen2.5-3B-Instruct in 4-bit and applying LoRA.

Provides two main workflows:
  1. Training — load base in 4-bit → prepare for k-bit training → apply LoRA
  2. Inference — load base in 4-bit → load saved LoRA adapter
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from configs.model_config import MODEL_ID, MODEL_PATH_OR_ID, MODEL_KWARGS, TOKENIZER_KWARGS, PADDING_SIDE, TORCH_DTYPE
from configs.lora_config import LORA_CONFIG

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────

def load_tokenizer(model_id: str = MODEL_PATH_OR_ID) -> PreTrainedTokenizerBase:
    """Load the official Qwen tokenizer with correct settings.

    • Keeps all default special tokens (do not modify!).
    • Sets padding_side to 'right' for causal LM training.
    • Ensures pad_token is set.
    """
    logger.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **TOKENIZER_KWARGS)

    # Qwen2.5 sets pad_token = eos_token by default — verify
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token = eos_token (%s)", tokenizer.pad_token)

    tokenizer.padding_side = PADDING_SIDE
    return tokenizer


# ──────────────────────────────────────────────
# Base model (4-bit quantized)
# ──────────────────────────────────────────────

def load_base_model(model_id: str = MODEL_PATH_OR_ID) -> AutoModelForCausalLM:
    """Load the base model in 4-bit NF4 quantization.

    Uses the configuration from ``configs.model_config``:
      • NF4 4-bit + double quantization
      • bfloat16 compute dtype
      • Automatic device mapping
    """
    logger.info("Loading base model in 4-bit: %s", model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **MODEL_KWARGS)
    model.config.use_cache = False      # Required for gradient checkpointing
    logger.info(
        "Model loaded — %.2f M parameters (%.2f M trainable before LoRA)",
        sum(p.numel() for p in model.parameters()) / 1e6,
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
    )
    return model


# ──────────────────────────────────────────────
# LoRA for training
# ──────────────────────────────────────────────

def prepare_for_training(model: AutoModelForCausalLM) -> PeftModel:
    """Prepare the base model for training (handling CPU vs GPU dynamically).

    Steps:
      1. Prepare for k-bit training (only if quantized on CUDA).
      2. Enable input gradients.
      3. Wrap with LoRA adapters.
    """
    from configs.model_config import QUANTIZATION_CONFIG, IS_CUDA_AVAILABLE

    if IS_CUDA_AVAILABLE and QUANTIZATION_CONFIG is not None:
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    else:
        logger.info("Model loaded without k-bit quantization. Standard preparation...")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    logger.info("Applying LoRA adapters (r=%d, alpha=%d)...", LORA_CONFIG.r, LORA_CONFIG.lora_alpha)
    peft_model = get_peft_model(model, LORA_CONFIG)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "LoRA applied -- trainable: %.2f M / %.2f M (%.2f%%)",
        trainable / 1e6,
        total / 1e6,
        100 * trainable / total,
    )
    peft_model.print_trainable_parameters()
    return peft_model


# ──────────────────────────────────────────────
# LoRA for inference
# ──────────────────────────────────────────────

def load_for_inference(
    adapter_path: str | Path,
    model_id: str = MODEL_PATH_OR_ID,
) -> tuple[PeftModel, PreTrainedTokenizerBase]:
    """Load the base model in 4-bit and apply a trained LoRA adapter.

    This is the production inference entrypoint:
      1. Load base Qwen2.5-3B-Instruct in 4-bit
      2. Load the saved LoRA adapter on top
      3. Return (model, tokenizer) ready for generation

    Parameters
    ----------
    adapter_path : str | Path
        Path to the saved LoRA adapter directory (contains adapter_config.json).
    model_id : str
        HuggingFace model ID for the base model.

    Returns
    -------
    tuple[PeftModel, PreTrainedTokenizerBase]
        The inference-ready model and tokenizer.
    """
    adapter_path = Path(adapter_path)
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"No adapter_config.json found in {adapter_path}. "
            f"Did training complete successfully?"
        )

    logger.info("Loading base model in 4-bit for inference...")
    base_model = load_base_model(model_id)

    logger.info("Loading LoRA adapter from %s...", adapter_path)
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        torch_dtype=TORCH_DTYPE,
        is_trainable=False,
    )
    model.eval()

    tokenizer = load_tokenizer(model_id)

    logger.info("Inference model ready [SUCCESS]")
    return model, tokenizer
