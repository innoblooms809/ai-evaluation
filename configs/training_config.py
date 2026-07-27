"""
Training hyper-parameters for SFTTrainer.

Tuned for a small dataset (~20 samples) with aggressive regularisation
and frequent checkpointing.
"""

import os
import torch
from transformers import TrainingArguments

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
OUTPUT_DIR: str = os.path.join("outputs", "checkpoints")
LOGGING_DIR: str = os.path.join("outputs", "logs")

# ──────────────────────────────────────────────
# Training arguments
# ──────────────────────────────────────────────
def get_training_args(
    output_dir: str = OUTPUT_DIR,
    logging_dir: str = LOGGING_DIR,
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.1,
    use_bf16: bool = True,
    eval_steps: int | None = None,
) -> TrainingArguments:
    """Build a ``TrainingArguments`` instance with production defaults.

    Parameters
    ----------
    output_dir : str
        Where to save checkpoints.
    num_train_epochs : int
        Total training epochs (10 is appropriate for ~20 samples).
    per_device_train_batch_size : int
        Micro-batch size per GPU.
    gradient_accumulation_steps : int
        Effective batch = micro × accumulation.
    learning_rate : float
        Peak learning rate (cosine-decayed).
    warmup_ratio : float
        Fraction of total steps used for linear warmup.
    use_bf16 : bool
        Use bfloat16 mixed precision.  Set False on GPUs without bf16.
    eval_steps : int | None
        Evaluate every N steps.  Defaults to once per epoch if None.
    """
    is_cuda = torch.cuda.is_available()
    
    # Dynamically select optimizer, precision, and pin_memory based on CUDA
    optim_algo = "paged_adamw_8bit" if is_cuda else "adamw_torch"
    bf16_enabled = use_bf16 if is_cuda else False
    fp16_enabled = (not use_bf16) if is_cuda else False
    pin_mem = True if is_cuda else False

    return TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,

        # ── Epochs & batching ──
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        # ── Optimiser & scheduler ──
        optim=optim_algo,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # ── Precision ──
        bf16=bf16_enabled,
        fp16=fp16_enabled,

        # ── Logging ──
        logging_steps=5,
        logging_first_step=True,
        report_to="tensorboard",

        # ── Evaluation ──
        eval_strategy="epoch" if eval_steps is None else "steps",
        eval_steps=eval_steps,

        # ── Checkpointing ──
        save_strategy="epoch",
        save_total_limit=3,             # Keep last 3 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # ── Misc ──
        gradient_checkpointing=is_cuda,    # Gradient checkpointing can slow down CPU training significantly
        gradient_checkpointing_kwargs={"use_reentrant": False} if is_cuda else None,
        dataloader_pin_memory=pin_mem,
        remove_unused_columns=False,    # Required for custom collators
        seed=42,
    )
