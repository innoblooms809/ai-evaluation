"""
Standalone data validation script.

Run this BEFORE training to catch any issues in your dataset early.

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --data-path data/raw/my_data.json --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import ValidationError

from data.schema import AnswerSheetSample

logger = logging.getLogger(__name__)

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

DEFAULT_DATA_PATH = os.path.join("data", "raw", "answer_sheet_evaluation_dataset_20.json")


def validate_dataset(data_path: str, verbose: bool = False) -> bool:
    """Validate every sample in a JSON dataset file.

    Returns True if all samples pass, False otherwise.
    """
    path = Path(data_path)
    if not path.exists():
        logger.error("File not found: %s", path)
        return False

    with open(path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    logger.info("Validating %d samples from %s", len(raw_samples), path.name)

    passed = 0
    failed = 0
    errors: list[str] = []

    subjects: set[str] = set()
    marks_range: list[int] = []

    for idx, sample in enumerate(raw_samples):
        try:
            parsed = AnswerSheetSample.model_validate(sample)
            passed += 1
            subjects.add(parsed.subject)
            marks_range.append(parsed.maximum_marks)

            if verbose:
                logger.info(
                    "  [OK] Sample %2d: [%s] %s -- %d/%d marks",
                    idx, parsed.subject, parsed.question[:40],
                    parsed.evaluation.marks_awarded, parsed.maximum_marks,
                )
        except ValidationError as e:
            failed += 1
            error_msg = f"Sample {idx}: {e}"
            errors.append(error_msg)
            logger.error("  [FAIL] Sample %d: VALIDATION FAILED", idx)
            if verbose:
                logger.error("    %s", e)

    # -- Summary --
    print("\n" + "=" * 50)
    print("  VALIDATION SUMMARY")
    print("=" * 50)
    print(f"  File:      {path.name}")
    print(f"  Total:     {len(raw_samples)}")
    print(f"  Passed:    {passed} [OK]")
    print(f"  Failed:    {failed} [FAIL]")
    print(f"  Subjects:  {', '.join(sorted(subjects))}")
    if marks_range:
        print(f"  Max marks: {min(marks_range)} - {max(marks_range)}")

    if errors:
        print(f"\n{'-' * 50}")
        print("  ERRORS:")
        print(f"{'-' * 50}")
        for error in errors:
            print(f"  {error}")

    print("=" * 50)

    if failed == 0:
        print("  [SUCCESS] All samples valid - ready for training!")
    else:
        print(f"  [ERROR] {failed} sample(s) have errors - fix before training.")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate dataset before training"
    )
    parser.add_argument(
        "--data-path", type=str, default=DEFAULT_DATA_PATH,
        help=f"Path to dataset JSON (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show details for each sample",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    success = validate_dataset(args.data_path, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
