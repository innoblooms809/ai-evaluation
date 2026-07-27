"""
SFTTrainer integration for QLoRA fine-tuning.

Wires together:
  • 4-bit quantized base model + LoRA adapters
  • Preprocessed & tokenized dataset
  • Dynamic padding collator
  • Training arguments (cosine schedule, gradient checkpointing, etc.)

Saves only the LoRA adapter weights (not the full model) at each checkpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import DatasetDict
from peft import PeftModel
from transformers import (
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

from configs.training_config import get_training_args
from data.tokenizer_utils import DynamicPaddingCollator

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Custom callbacks
# ──────────────────────────────────────────────

class LoggingCallback(TrainerCallback):
    """Logs training milestones at INFO level."""

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs):
        logger.info("═══ Epoch %d / %d ═══", int(state.epoch) + 1, int(args.num_train_epochs))

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, logs: dict | None = None, **kwargs):
        if logs:
            loss = logs.get("loss", logs.get("eval_loss", "N/A"))
            lr = logs.get("learning_rate", "N/A")
            logger.info("Step %d | loss=%.4f | lr=%s", state.global_step, loss if isinstance(loss, float) else 0, lr)

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        logger.info("Training complete — %d total steps", state.global_step)


# ──────────────────────────────────────────────
# Trainer factory
# ──────────────────────────────────────────────

def create_trainer(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: DatasetDict,
    output_dir: str = "outputs/checkpoints",
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    use_bf16: bool = True,
    early_stopping_patience: int = 3,
) -> SFTTrainer:
    """Create a fully configured SFTTrainer.

    Parameters
    ----------
    model : PeftModel
        The 4-bit base model wrapped with LoRA adapters.
    tokenizer : PreTrainedTokenizerBase
        The Qwen tokenizer.
    dataset : DatasetDict
        Must have 'train' and 'validation' splits with a 'text' column.
    output_dir : str
        Where to save checkpoints.
    num_train_epochs : int
        Number of training epochs.
    early_stopping_patience : int
        Stop if eval_loss doesn't improve for N evaluations.

    Returns
    -------
    SFTTrainer
        Ready to call ``.train()``.
    """
    training_args = get_training_args(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        use_bf16=use_bf16,
    )

    collator = DynamicPaddingCollator(tokenizer=tokenizer)

    callbacks = [
        LoggingCallback(),
        EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
    ]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info("Trainer created — ready to train")
    return trainer


# ──────────────────────────────────────────────
# Training orchestrator
# ──────────────────────────────────────────────

def train_and_save(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: DatasetDict,
    output_dir: str = "outputs/checkpoints",
    final_adapter_dir: str = "outputs/checkpoints/final",
    **trainer_kwargs,
) -> Path:
    """Run training and save the final LoRA adapter.

    Parameters
    ----------
    model : PeftModel
        LoRA-wrapped model.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer.
    dataset : DatasetDict
        Preprocessed + tokenized dataset with 'text' column.
    output_dir : str
        Checkpoint directory.
    final_adapter_dir : str
        Where to save the final adapter.
    **trainer_kwargs
        Forwarded to ``create_trainer()``.

    Returns
    -------
    Path
        Path to the saved adapter directory.
    """
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=output_dir,
        **trainer_kwargs,
    )

    logger.info("Starting training…")
    train_result = trainer.train()

    # Log final metrics
    metrics = train_result.metrics
    logger.info("Training metrics: %s", metrics)

    # Save final adapter
    final_path = Path(final_adapter_dir)
    final_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info("Final LoRA adapter saved to %s [SUCCESS]", final_path)

    return final_path
