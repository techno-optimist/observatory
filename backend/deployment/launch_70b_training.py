#!/usr/bin/env python3
"""
Launch V8 70B Training on HuggingFace
=====================================

Creates an AutoTrain Space to train PRISM-70B on Qwen3-72B.

Options:
1. AutoTrain Advanced Space (GPU-enabled)
2. Direct training via autotrain CLI

Usage:
    HF_TOKEN=your_token python launch_70b_training.py
"""

import os
import json
from huggingface_hub import HfApi, create_repo, upload_file, duplicate_space
from pathlib import Path

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
USERNAME = "kevruss"

# Training configuration
CONFIG = {
    "task": "llm-sft",
    "base_model": "Qwen/Qwen3-72B",  # 72B is Qwen's largest open model
    "project_name": "PRISM-70B",

    "data": {
        "path": "kevruss/PRISM-V8-Dataset",
        "train_split": "train",
        "valid_split": "valid",
        "chat_template": "chatml",
        "column_mapping": {
            "text_column": "text"
        }
    },

    "params": {
        # QLoRA for 70B on single A100
        "quantization": "int4",
        "peft": True,
        "target_modules": "all-linear",

        # LoRA config (matching V8 success)
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,

        # Training
        "epochs": 3,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "lr": 1e-4,

        # Context
        "block_size": 2048,
        "model_max_length": 4096,

        # Optimizer
        "optimizer": "adamw_torch",
        "scheduler": "cosine",
        "warmup_ratio": 0.1,
        "mixed_precision": "bf16"
    },

    "hub": {
        "username": USERNAME,
        "push_to_hub": True
    }
}


def create_autotrain_space():
    """Create an AutoTrain Advanced Space for 70B training."""

    api = HfApi(token=HF_TOKEN)

    space_name = "PRISM-70B-Training"
    repo_id = f"{USERNAME}/{space_name}"

    print(f"Creating AutoTrain Space: {repo_id}")

    # Duplicate the official AutoTrain Advanced space
    try:
        duplicate_space(
            from_id="autotrain-projects/autotrain-advanced",
            to_id=repo_id,
            token=HF_TOKEN,
            private=True,
            hardware="a100-large",  # A100 80GB for 70B QLoRA
            exist_ok=True
        )
        print(f"  Space created: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"  Error creating space: {e}")
        print("  You may need to create it manually via the web UI")
        return None

    return repo_id


def print_manual_instructions():
    """Print instructions for manual setup via web UI."""

    print("\n" + "=" * 60)
    print("MANUAL SETUP INSTRUCTIONS")
    print("=" * 60)
    print()
    print("1. Go to: https://huggingface.co/spaces/autotrain-projects/autotrain-advanced")
    print()
    print("2. Click 'Duplicate this Space'")
    print("   - Name: PRISM-70B-Training")
    print("   - Private: Yes")
    print("   - Hardware: A100 Large (80GB)")
    print()
    print("3. In your new Space, configure training:")
    print()
    print("   MODEL:")
    print("   - Base Model: Qwen/Qwen3-72B")
    print("   - Task: LLM SFT")
    print()
    print("   DATA:")
    print("   - Dataset: kevruss/PRISM-V8-Dataset")
    print("   - Train Split: train")
    print("   - Valid Split: valid")
    print("   - Chat Template: chatml")
    print()
    print("   TRAINING PARAMS:")
    print("   - Quantization: int4 (QLoRA)")
    print("   - PEFT: enabled")
    print("   - LoRA r: 8")
    print("   - LoRA alpha: 16")
    print("   - Epochs: 3")
    print("   - Batch Size: 1")
    print("   - Gradient Accumulation: 8")
    print("   - Learning Rate: 1e-4")
    print("   - Mixed Precision: bf16")
    print()
    print("4. Click 'Start Training'")
    print()
    print("ESTIMATED COST: ~$50-100 for full training")
    print("  - A100 80GB: ~$4/hour")
    print("  - Training time: ~12-24 hours")
    print()


def print_cli_instructions():
    """Print instructions for CLI-based training."""

    print("\n" + "=" * 60)
    print("CLI TRAINING INSTRUCTIONS (Local or RunPod)")
    print("=" * 60)
    print()
    print("1. Install AutoTrain:")
    print("   pip install autotrain-advanced")
    print()
    print("2. Run training:")
    print("   autotrain --config backend/deployment/autotrain_v8_70b.yaml")
    print()
    print("Or for more control, use TRL directly:")
    print()
    print("   from trl import SFTTrainer")
    print("   from peft import LoraConfig, get_peft_model")
    print("   # See backend/deployment/train_70b_trl.py")
    print()


def main():
    print("=" * 60)
    print("V8 70B Training Launcher")
    print("=" * 60)
    print()

    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set, will show manual instructions only")

    # Try to create space programmatically
    if HF_TOKEN:
        space_id = create_autotrain_space()
        if space_id:
            print(f"\nSpace created successfully!")
            print(f"Go to: https://huggingface.co/spaces/{space_id}")
            print("Configure and start training from there")
            return

    # Fall back to manual instructions
    print_manual_instructions()
    print_cli_instructions()


if __name__ == "__main__":
    main()
