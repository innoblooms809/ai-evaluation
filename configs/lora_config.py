"""
LoRA (Low-Rank Adaptation) configuration.

Targets ALL linear projection layers in Qwen2.5 for maximum
adaptation capacity while keeping the parameter count small.
"""

from peft import LoraConfig, TaskType

# ──────────────────────────────────────────────
# LoRA hyper-parameters
# ──────────────────────────────────────────────
LORA_R: int = 16                    # Rank — controls adapter capacity
LORA_ALPHA: int = 32                # Scaling factor (alpha / r = effective LR scale)
LORA_DROPOUT: float = 0.1           # Higher dropout for small datasets to prevent overfit

# Target every linear projection in the Qwen2.5 transformer blocks
LORA_TARGET_MODULES: list[str] = [
    "q_proj",       # Query projection
    "k_proj",       # Key projection
    "v_proj",       # Value projection
    "o_proj",       # Output projection
    "gate_proj",    # MLP gate projection
    "up_proj",      # MLP up projection
    "down_proj",    # MLP down projection
]

# ──────────────────────────────────────────────
# PEFT LoraConfig object
# ──────────────────────────────────────────────
LORA_CONFIG = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",                     # No bias adaptation — standard for QLoRA
    task_type=TaskType.CAUSAL_LM,
)
