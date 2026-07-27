"""
Main training entrypoint.

Orchestrates the full training pipeline:
  1. Load & validate dataset
  2. Load tokenizer
  3. Preprocess → chat template → tokenization
  4. Load base model (4-bit) + apply LoRA
  5. Train with SFTTrainer
  6. Save final LoRA adapter

Usage:
    python scripts/train.py
    python scripts/train.py --data-path data/raw/my_data.json --epochs 15 --lr 1e-4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset_loader import load_and_validate
from data.preprocessor import preprocess_dataset
from data.tokenizer_utils import tokenize_with_assistant_masking
from model.loader import load_tokenizer, load_base_model, prepare_for_training
from training.trainer import train_and_save
from configs.model_config import MAX_SEQ_LENGTH

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = os.path.join("data", "raw", "answer_sheet_evaluation_dataset_20.json")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-3B-Instruct for Answer Sheet Evaluation (QLoRA)"
    )
    parser.add_argument(
        "--data-path", type=str, default=DEFAULT_DATA_PATH,
        help=f"Path to the training JSON file (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/checkpoints",
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--final-adapter-dir", type=str, default="outputs/checkpoints/final",
        help="Where to save the final LoRA adapter",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Per-device train batch size (default: 2)",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--no-bf16", action="store_true",
        help="Disable bf16 (use fp16 instead — for GPUs without bf16 support)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # ── Logging setup ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(args.output_dir, "training.log") if os.path.isdir(args.output_dir)
                else "training.log"
            ),
        ],
    )

    logger.info("=" * 60)
    logger.info("Answer Sheet Evaluation — QLoRA Training")
    logger.info("=" * 60)
    logger.info("Config: epochs=%d, batch=%d, grad_accum=%d, lr=%s",
                args.epochs, args.batch_size, args.grad_accum, args.lr)

    # ── Step 1: Load & validate dataset ──
    logger.info("Step 1/5: Loading and validating dataset…")
    dataset_dict = load_and_validate(
        data_path=args.data_path,
        val_split=args.val_split,
        seed=args.seed,
    )
    logger.info("  Train: %d samples | Validation: %d samples",
                len(dataset_dict["train"]), len(dataset_dict["validation"]))

    # ── Step 2: Load tokenizer ──
    logger.info("Step 2/5: Loading tokenizer…")
    tokenizer = load_tokenizer()

    # ── Step 3: Preprocess (chat template → tokenization) ──
    logger.info("Step 3/5: Preprocessing — chat template + tokenization…")
    dataset_dict = preprocess_dataset(dataset_dict, tokenizer)

    # Tokenize with assistant-only masking
    dataset_dict = dataset_dict.map(
        tokenize_with_assistant_masking,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_SEQ_LENGTH},
        desc="Tokenizing",
    )

    # Log a sample to verify formatting
    sample_text = dataset_dict["train"][0]["text"]
    logger.info("Sample formatted text (first 500 chars):\n%s", sample_text[:500])

    # ── Step 4: Load model (4-bit) + apply LoRA ──
    logger.info("Step 4/5: Loading model (4-bit) and applying LoRA…")
    base_model = load_base_model()
    model = prepare_for_training(base_model)

    # ── Step 5: Train ──
    logger.info("Step 5/5: Starting training…")
    final_path = train_and_save(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset_dict,
        output_dir=args.output_dir,
        final_adapter_dir=args.final_adapter_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        use_bf16=not args.no_bf16,
    )

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("LoRA adapter saved to: %s", final_path)
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("  1. Run inference:  python scripts/infer.py --adapter-path %s", final_path)
    logger.info("  2. Merge adapter:  python model/merge.py --adapter-path %s --output-path outputs/merged", final_path)


if __name__ == "__main__":
    main()
