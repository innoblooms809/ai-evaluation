# Answer Sheet Evaluation — Qwen2.5-3B-Instruct QLoRA Fine-Tuning

Production-grade system for automated answer sheet evaluation using **Qwen2.5-3B-Instruct** fine-tuned with **QLoRA** (LoRA + 4-bit quantization).

## Architecture

```
Raw JSON → Validation → Preprocessing → Chat Template → Tokenization
     → Dynamic Padding → Custom Collator → Batching → Training
```

**Inference**: Load base model in 4-bit → Apply trained LoRA adapter → Generate structured evaluation JSON.

## Project Structure

```
AIq/
├── configs/                    # All hyperparameters & settings
│   ├── model_config.py         # Model ID, 4-bit quantization config
│   ├── lora_config.py          # LoRA rank, alpha, target modules
│   └── training_config.py      # Epochs, batch size, scheduler
├── data/                       # Data pipeline
│   ├── raw/                    # Raw JSON dataset
│   ├── processed/              # Cached tokenized data
│   ├── schema.py               # Pydantic validation schemas
│   ├── dataset_loader.py       # Load + validate + split
│   ├── preprocessor.py         # → Chat template formatting
│   └── tokenizer_utils.py      # Tokenization + dynamic padding collator
├── model/                      # Model management
│   ├── loader.py               # Load 4-bit base + LoRA (train & inference)
│   └── merge.py                # Optional: merge LoRA into base model
├── training/                   # Training loop
│   └── trainer.py              # SFTTrainer + callbacks
├── inference/                  # Production inference
│   └── evaluator.py            # Load model + evaluate answers
├── scripts/                    # CLI entrypoints
│   ├── download_model.py       # Download Qwen2.5-3B-Instruct
│   ├── train.py                # Main training script
│   ├── infer.py                # Main inference script
│   └── validate_data.py        # Validate dataset before training
├── outputs/                    # Generated outputs
│   ├── checkpoints/            # LoRA adapter checkpoints
│   └── logs/                   # TensorBoard logs
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the model

```bash
python scripts/download_model.py
```

### 3. Validate your dataset

```bash
python scripts/validate_data.py --data-path data/raw/answer_sheet_evaluation_dataset_20.json --verbose
```

### 4. Train

```bash
python scripts/train.py \
    --data-path data/raw/answer_sheet_evaluation_dataset_20.json \
    --epochs 10 \
    --batch-size 2 \
    --lr 2e-4
```

### 5. Run inference

```bash
# Interactive mode
python scripts/infer.py --adapter-path outputs/checkpoints/final --interactive

# Batch mode
python scripts/infer.py --adapter-path outputs/checkpoints/final \
    --input-file data/raw/test.json --output-file results.json

# Single evaluation
python scripts/infer.py --adapter-path outputs/checkpoints/final \
    --subject "Physics" \
    --question "What is gravity?" \
    --reference "Gravity is the force of attraction between masses." \
    --student "Gravity pulls things down."
```

## Configuration

| Setting | Value | File |
|---------|-------|------|
| Base Model | `Qwen/Qwen2.5-3B-Instruct` | `configs/model_config.py` |
| Quantization | NF4 4-bit + double quant | `configs/model_config.py` |
| LoRA Rank | 16 | `configs/lora_config.py` |
| LoRA Alpha | 32 | `configs/lora_config.py` |
| Target Modules | All linear (q/k/v/o/gate/up/down) | `configs/lora_config.py` |
| Max Seq Length | 1024 | `configs/model_config.py` |
| Epochs | 10 | `configs/training_config.py` |
| Batch Size | 2 × 4 accumulation = 8 effective | `configs/training_config.py` |
| Learning Rate | 2e-4 (cosine decay) | `configs/training_config.py` |
| Optimiser | paged_adamw_8bit | `configs/training_config.py` |

## Dataset Format

Each sample in the JSON array should have:

```json
{
  "subject": "Computer Science",
  "question": "What is Artificial Intelligence?",
  "maximum_marks": 5,
  "rubric": {
    "criteria": [
      {"name": "Core concept", "marks": 2, "description": "States the correct definition."},
      {"name": "Key points", "marks": 2, "description": "Includes supporting details."},
      {"name": "Clarity", "marks": 1, "description": "Clear explanation."}
    ]
  },
  "reference_answer": "AI is the simulation of human intelligence by machines.",
  "student_answer": "AI is when computers think like humans.",
  "evaluation": {
    "marks_awarded": 4,
    "criterion_scores": [
      {"criterion": "Core concept", "score": 2, "reason": "Main idea is correct."},
      {"criterion": "Key points", "score": 1, "reason": "Missing details."},
      {"criterion": "Clarity", "score": 1, "reason": "Clear enough."}
    ],
    "overall_feedback": "Good answer with minor gaps.",
    "strengths": ["Correct concept"],
    "improvements": ["Add more detail"]
  }
}
```

## GPU Requirements

- **Training**: ~6 GB VRAM with QLoRA (4-bit) on Qwen2.5-3B
- **Inference**: ~3 GB VRAM with 4-bit quantization
- Tested on: NVIDIA T4 (16 GB), RTX 3060 (12 GB)

## Optional: Merge LoRA adapter

For deployment without PEFT:

```bash
python model/merge.py \
    --adapter-path outputs/checkpoints/final \
    --output-path outputs/merged
```

## License

This project uses [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) under the Apache 2.0 license.
