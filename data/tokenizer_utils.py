"""
Tokenization utilities and custom data collator for dynamic padding.

Pipeline step:  Chat Template → Tokenization → Dynamic Padding → Custom Collator → Batching

Key design decisions:
  • Dynamic padding — pads to the longest sequence in each batch, not the
    global max.  This saves significant GPU memory on short sequences.
  • Assistant-only loss masking — labels for system + user tokens are set
    to -100 so the model only learns to generate evaluations, not to
    repeat the input prompt.
  • Truncation enabled — prevents OOM on unexpectedly long sequences.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from configs.model_config import MAX_SEQ_LENGTH

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Tokenization
# ──────────────────────────────────────────────

def tokenize_sample(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_SEQ_LENGTH,
) -> dict[str, list[int]]:
    """Tokenize a single formatted conversation string.

    Returns ``input_ids`` and ``attention_mask`` (no padding yet —
    that happens in the collator).
    """
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,          # Dynamic padding in the collator
        return_tensors=None,    # Return plain lists
    )
    # Labels = input_ids (shifted internally by the model)
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded


def tokenize_dataset_batch(
    examples: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_SEQ_LENGTH,
) -> dict[str, list]:
    """Batch tokenization map function for ``dataset.map(batched=True)``.

    Expects a ``text`` column (produced by ``preprocessor.preprocess_batch``).
    """
    all_input_ids: list[list[int]] = []
    all_attention_mask: list[list[int]] = []
    all_labels: list[list[int]] = []

    for text in examples["text"]:
        encoded = tokenize_sample(text, tokenizer, max_length)
        all_input_ids.append(encoded["input_ids"])
        all_attention_mask.append(encoded["attention_mask"])
        all_labels.append(encoded["labels"])

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


def mask_labels_for_assistant_only(
    input_ids: list[int],
    labels: list[int],
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    """Mask everything except the assistant's response in labels.

    For Qwen2.5 ChatML format, assistant turns start after
    ``<|im_start|>assistant\\n`` and end at ``<|im_end|>``.

    Tokens outside assistant turns get label = -100.
    """
    IGNORE_INDEX = -100

    # Encode the assistant turn markers
    # Qwen2.5 uses: <|im_start|>assistant\n ... <|im_end|>
    assistant_start_str = "<|im_start|>assistant\n"
    assistant_end_str = "<|im_end|>"

    assistant_start_ids = tokenizer.encode(assistant_start_str, add_special_tokens=False)
    assistant_end_ids = tokenizer.encode(assistant_end_str, add_special_tokens=False)

    masked_labels = [IGNORE_INDEX] * len(labels)
    in_assistant = False
    i = 0

    while i < len(input_ids):
        # Check if we're at the start of an assistant turn
        if (
            not in_assistant
            and input_ids[i : i + len(assistant_start_ids)] == assistant_start_ids
        ):
            # Skip past the assistant start marker (don't include it in loss)
            i += len(assistant_start_ids)
            in_assistant = True
            continue

        # Check if we're at the end of an assistant turn
        if (
            in_assistant
            and input_ids[i : i + len(assistant_end_ids)] == assistant_end_ids
        ):
            in_assistant = False
            i += len(assistant_end_ids)
            continue

        # If inside assistant turn, keep the label
        if in_assistant:
            masked_labels[i] = labels[i]

        i += 1

    return masked_labels


def tokenize_with_assistant_masking(
    examples: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_SEQ_LENGTH,
) -> dict[str, list]:
    """Tokenize and mask labels so loss is computed only on assistant turns.

    Drop-in replacement for ``tokenize_dataset_batch`` when assistant-only
    loss masking is desired (recommended).
    """
    all_input_ids: list[list[int]] = []
    all_attention_mask: list[list[int]] = []
    all_labels: list[list[int]] = []

    for text in examples["text"]:
        encoded = tokenize_sample(text, tokenizer, max_length)

        masked = mask_labels_for_assistant_only(
            encoded["input_ids"],
            encoded["labels"],
            tokenizer,
        )

        all_input_ids.append(encoded["input_ids"])
        all_attention_mask.append(encoded["attention_mask"])
        all_labels.append(masked)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


# ──────────────────────────────────────────────
# Dynamic Padding Collator
# ──────────────────────────────────────────────

@dataclass
class DynamicPaddingCollator:
    """Pads each batch to the longest sequence in that batch.

    This is significantly more memory-efficient than static padding
    to ``MAX_SEQ_LENGTH`` when sequence lengths vary.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        Used to determine the pad token ID.
    padding_side : str
        ``"right"`` (default) or ``"left"``.
    """

    tokenizer: PreTrainedTokenizerBase
    padding_side: str = "right"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a list of tokenized samples into a padded batch."""
        # Find the longest sequence in this batch
        max_len = max(len(f["input_ids"]) for f in features)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        IGNORE_INDEX = -100

        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature.get("labels", input_ids.copy())

            pad_length = max_len - len(input_ids)

            if self.padding_side == "right":
                input_ids = input_ids + [pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
                labels = labels + [IGNORE_INDEX] * pad_length
            else:
                input_ids = [pad_token_id] * pad_length + input_ids
                attention_mask = [0] * pad_length + attention_mask
                labels = [IGNORE_INDEX] * pad_length + labels

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
