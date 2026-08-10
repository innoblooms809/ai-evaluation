"""
model/vl_loader.py — Production Loader for Qwen2.5-VL-3B-Instruct

Handles loading and singleton management for Qwen2.5-VL vision-language models.
Supports loading image inputs alongside text for multi-modal answer sheet evaluation.
"""

import logging
from pathlib import Path
from typing import Tuple, Any, Optional

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

from configs.vl_model_config import (
    VL_MODEL_ID,
    VL_MODEL_PATH_OR_ID,
    VL_MODEL_KWARGS,
    VL_PROCESSOR_KWARGS,
)

logger = logging.getLogger(__name__)

_vl_model: Optional[Any] = None
_vl_processor: Optional[Any] = None


def load_vl_model_and_processor(model_path_or_id: str = VL_MODEL_PATH_OR_ID) -> Tuple[Any, Any]:
    """
    Singleton loader for Qwen2.5-VL-3B-Instruct. Loads model and AutoProcessor.
    """
    global _vl_model, _vl_processor

    if _vl_model is None or _vl_processor is None:
        logger.info(f"Loading Qwen2.5-VL model from: {model_path_or_id}")
        
        # Load processor
        try:
            _vl_processor = AutoProcessor.from_pretrained(model_path_or_id, **VL_PROCESSOR_KWARGS)
        except Exception as proc_err:
            logger.warning(f"AutoProcessor load with custom kwargs failed ({proc_err}). Retrying default AutoProcessor...")
            _vl_processor = AutoProcessor.from_pretrained(model_path_or_id, trust_remote_code=True)

        # Load Vision-Language Model
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            _vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path_or_id,
                **VL_MODEL_KWARGS
            )
        except ImportError:
            logger.info("Qwen2_5_VLForConditionalGeneration not directly available in transformers. Falling back to AutoModelForCausalLM...")
            _vl_model = AutoModelForCausalLM.from_pretrained(
                model_path_or_id,
                **VL_MODEL_KWARGS
            )

        _vl_model.eval()
        logger.info("Qwen2.5-VL model loaded successfully!")

    return _vl_model, _vl_processor


def generate_vl_evaluation(
    prompt: str,
    image_paths: Optional[list] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """
    Generate evaluation using Qwen2.5-VL Vision-Language model.
    """
    model, processor = load_vl_model_and_processor()

    content = []
    if image_paths:
        for img in image_paths:
            content.append({"type": "image", "image": img})
    
    content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "user", "content": content}
    ]

    try:
        # Format messages with processor
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if image_paths:
            from PIL import Image
            raw_images = [Image.open(img).convert("RGB") for img in image_paths]
            inputs = processor(
                text=[text_prompt],
                images=raw_images,
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = processor(
                text=[text_prompt],
                padding=True,
                return_tensors="pt"
            )

        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()

    except Exception as e:
        logger.error(f"Error during Qwen2.5-VL generation: {e}")
        raise RuntimeError(f"Qwen2.5-VL inference failed: {str(e)}")
