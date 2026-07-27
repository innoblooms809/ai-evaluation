"""
Dataset loader: reads raw JSON, validates every sample, and returns
a HuggingFace ``datasets.DatasetDict`` with train / validation splits.

Pipeline step:  Raw JSON → Dataset Loading → Validation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict
from pydantic import ValidationError

from data.schema import AnswerSheetSample

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def load_and_validate(
    data_path: str | Path,
    val_split: float = 0.2,
    seed: int = 42,
) -> DatasetDict:
    """Load raw JSON, validate, and return train/val ``DatasetDict``.

    Parameters
    ----------
    data_path : str | Path
        Path to the raw JSON file (list of sample dicts).
    val_split : float
        Fraction of data to hold out for validation.
    seed : int
        Random seed for reproducible splitting.

    Returns
    -------
    DatasetDict
        Keys: ``train``, ``validation``.  Each row is a validated dict.

    Raises
    ------
    FileNotFoundError
        If *data_path* does not exist.
    ValueError
        If any sample fails validation.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    logger.info("Loading dataset from %s", data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        raw_samples: list[dict] = json.load(f)

    logger.info("Loaded %d raw samples — validating…", len(raw_samples))

    validated: list[dict] = []
    errors: list[str] = []

    for idx, sample in enumerate(raw_samples):
        try:
            parsed = AnswerSheetSample.model_validate(sample)
            validated.append(parsed.model_dump())
        except ValidationError as e:
            errors.append(f"Sample {idx}: {e}")

    if errors:
        error_report = "\n\n".join(errors)
        raise ValueError(
            f"{len(errors)} / {len(raw_samples)} samples failed validation:\n\n"
            f"{error_report}"
        )

    logger.info("All %d samples passed validation ✓", len(validated))

    # Build HuggingFace Dataset and split
    ds = Dataset.from_list(validated)

    if len(ds) < 4:
        logger.warning(
            "Dataset has only %d samples — skipping validation split", len(ds)
        )
        return DatasetDict({"train": ds, "validation": ds})

    split = ds.train_test_split(test_size=val_split, seed=seed)
    return DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })


def load_raw_samples(data_path: str | Path) -> list[AnswerSheetSample]:
    """Load and return validated ``AnswerSheetSample`` objects (no split)."""
    data_path = Path(data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        raw_samples: list[dict] = json.load(f)

    validated: list[AnswerSheetSample] = []
    for idx, sample in enumerate(raw_samples):
        try:
            validated.append(AnswerSheetSample.model_validate(sample))
        except ValidationError as e:
            raise ValueError(f"Sample {idx} validation error: {e}") from e

    return validated
