from setuptools import setup, find_packages

setup(
    name="answer-sheet-evaluator",
    version="1.0.0",
    description="Production-grade answer sheet evaluation using Qwen2.5-3B-Instruct with QLoRA fine-tuning",
    author="AIq",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.46.0",
        "peft>=0.13.0",
        "trl>=0.12.0",
        "bitsandbytes>=0.44.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "pydantic>=2.0.0",
        "huggingface-hub>=0.25.0",
    ],
    entry_points={
        "console_scripts": [
            "ase-train=scripts.train:main",
            "ase-infer=scripts.infer:main",
            "ase-validate=scripts.validate_data:main",
            "ase-download=scripts.download_model:main",
        ],
    },
)
