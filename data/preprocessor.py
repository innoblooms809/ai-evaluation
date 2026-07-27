"""
Preprocessor: converts validated samples into chat-template conversations
that Qwen2.5-Instruct expects.

Pipeline step:  Validation → Preprocessing → Chat Template

Each sample becomes a multi-turn conversation:
  • system  – evaluator persona + output format spec
  • user    – question context (subject, rubric, answers)
  • assistant – the structured JSON evaluation
"""

from __future__ import annotations

import json
import logging
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert answer sheet evaluator. You evaluate student answers against \
a reference answer and a grading rubric. You MUST respond with a JSON object \
containing the following fields:

{
  "marks_awarded": <int>,
  "criterion_scores": [
    {"criterion": "<name>", "score": <int>, "reason": "<brief justification>"}
  ],
  "overall_feedback": "<1-2 sentence summary>",
  "strengths": ["<strength 1>", ...],
  "improvements": ["<improvement 1>", ...]
}

Rules:
- Score each criterion independently based on the rubric.
- marks_awarded MUST equal the sum of all criterion scores.
- Each criterion score MUST NOT exceed its maximum marks.
- Be fair, specific, and constructive.
- Output ONLY valid JSON. No markdown, no commentary.\
"""


# ──────────────────────────────────────────────
# Build user prompt from a sample
# ──────────────────────────────────────────────
def _build_user_prompt(sample: dict) -> str:
    """Format the question context into a clear user message."""
    rubric = sample["rubric"]
    criteria_lines = "\n".join(
        f"  - {c['name']} ({c['marks']} marks): {c['description']}"
        for c in rubric["criteria"]
    )

    return (
        f"Subject: {sample['subject']}\n"
        f"Question: {sample['question']}\n"
        f"Maximum Marks: {sample['maximum_marks']}\n"
        f"\nGrading Rubric:\n{criteria_lines}\n"
        f"\nReference Answer:\n{sample['reference_answer']}\n"
        f"\nStudent Answer:\n{sample['student_answer']}\n"
        f"\nEvaluate the student's answer using the rubric above. "
        f"Respond with the evaluation JSON only."
    )


def _build_assistant_response(sample: dict) -> str:
    """Serialise the evaluation dict into the expected JSON string."""
    return json.dumps(sample["evaluation"], indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────
# Convert a single sample → chat messages
# ──────────────────────────────────────────────
def sample_to_messages(sample: dict) -> list[dict[str, str]]:
    """Return a list of ``{"role": ..., "content": ...}`` dicts."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(sample)},
        {"role": "assistant", "content": _build_assistant_response(sample)},
    ]


# ──────────────────────────────────────────────
# Apply chat template (tokenizer-aware)
# ──────────────────────────────────────────────
def apply_chat_template(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    """Apply the official Qwen chat template to a message list.

    Uses ``tokenizer.apply_chat_template()`` which inserts the
    correct special tokens (``<|im_start|>``, ``<|im_end|>``).
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,    # We include the assistant turn already
    )


# ──────────────────────────────────────────────
# Batch preprocessing for Dataset.map()
# ──────────────────────────────────────────────
def preprocess_batch(
    examples: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, list[str]]:
    """Map function that adds a ``text`` column with the formatted conversation.

    Designed to be used with ``dataset.map(fn, batched=True)``.
    """
    texts: list[str] = []
    batch_size = len(examples["question"])

    for i in range(batch_size):
        sample = {key: examples[key][i] for key in examples}
        messages = sample_to_messages(sample)
        formatted = apply_chat_template(messages, tokenizer)
        texts.append(formatted)

    return {"text": texts}


def preprocess_dataset(
    dataset_dict: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    num_proc: int = 1,
) -> DatasetDict:
    """Add ``text`` column to every split in a DatasetDict.

    Parameters
    ----------
    dataset_dict : DatasetDict
        Must have columns matching ``AnswerSheetSample`` fields.
    tokenizer : PreTrainedTokenizerBase
        Used to call ``apply_chat_template()``.
    num_proc : int
        Number of processes for parallel mapping.

    Returns
    -------
    DatasetDict
        Same splits, with an added ``text`` column.
    """
    logger.info("Preprocessing dataset — applying chat template…")
    return DatasetDict({
        split_name: split.map(
            preprocess_batch,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=num_proc,
            desc=f"Preprocessing {split_name}",
        )
        for split_name, split in dataset_dict.items()
    })
