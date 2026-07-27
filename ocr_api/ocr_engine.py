"""
ocr_engine.py — Chandra OCR-2 model loader and inference wrapper.

This module loads the Chandra OCR-2 model once at startup and exposes
a simple `run_ocr(image) -> str` function for downstream use.
"""

import os
import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)

# ── Lazy-import Chandra helpers (available only when chandra-ocr[hf] is installed) ──
try:
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem
    from chandra.output import parse_markdown
except ImportError:
    generate_hf = None
    BatchInputItem = None
    parse_markdown = None

# ── Singleton model state ──
_model = None
_processor = None
_device = None

# ── Default local model path — points to the already-downloaded weights ──
DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent.parent
    / "AI-OCR" / "models" / "chandra-ocr-2"
)
# If that doesn't exist, use the Desktop path explicitly
if not os.path.isdir(DEFAULT_MODEL_PATH):
    DEFAULT_MODEL_PATH = r"C:\Users\anubhavmznac\Desktop\AI-OCR\models\chandra-ocr-2"

FALLBACK_HF_REPO = "datalab-to/chandra-ocr-2"


def load_model(model_path: str = None) -> None:
    """
    Load the Chandra OCR-2 model and processor into memory.
    Called once at application startup.
    """
    global _model, _processor, _device

    if _model is not None:
        logger.info("Model already loaded — skipping.")
        return

    path = model_path or DEFAULT_MODEL_PATH

    # Determine device
    if torch.cuda.is_available():
        _device = "cuda"
        torch_dtype = torch.bfloat16
        device_map = "auto"
    else:
        _device = "cpu"
        torch_dtype = torch.float32
        device_map = None

    # If local path is missing or empty, fall back to HuggingFace Hub
    if not os.path.isdir(path) or not os.listdir(path):
        logger.warning(
            f"Local path '{path}' not found. Falling back to HF: {FALLBACK_HF_REPO}"
        )
        path = FALLBACK_HF_REPO

    logger.info(f"Loading Chandra OCR-2 from: {path}  (device={_device})")

    _model = AutoModelForImageTextToText.from_pretrained(
        path,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    _model.eval()

    _processor = AutoProcessor.from_pretrained(path)
    _processor.tokenizer.padding_side = "left"
    _model.processor = _processor  # generate_hf reads model.processor

    logger.info("✅ Chandra OCR-2 model loaded successfully.")


def run_ocr(image: Image.Image, prompt_type: str = "ocr_layout") -> str:
    """
    Run Chandra OCR on a single PIL Image and return the markdown text.

    Args:
        image: A PIL RGB image.
        prompt_type: Chandra prompt type. 'ocr_layout' preserves layout
                     (tables, formulas, columns).

    Returns:
        Extracted text as markdown.
    """
    if _model is None:
        load_model()

    if generate_hf is None or BatchInputItem is None or parse_markdown is None:
        raise ImportError(
            "chandra-ocr[hf] is not installed. "
            "Run:  pip install chandra-ocr[hf]"
        )

    batch = [BatchInputItem(image=image, prompt_type=prompt_type)]

    with torch.no_grad():
        results = generate_hf(batch, _model)

    if not results:
        raise RuntimeError("Chandra OCR generated no results.")

    return parse_markdown(results[0].raw)
