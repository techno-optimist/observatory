#!/usr/bin/env python3
"""
Direct V8 70B Training Script
==============================

Run this directly on a GPU server (HF Space with dev mode, RunPod, Lambda, etc.)

Requirements:
- 4x L40S or equivalent (~192GB VRAM total)
- Python 3.10+

Usage:
    # On the remote server:
    pip install torch transformers accelerate peft trl datasets bitsandbytes
    python train_70b_direct.py
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model
BASE_MODEL = "Qwen/Qwen3-72B"  # 72B is Qwen's largest
OUTPUT_DIR = "./prism-70b-output"
HF_REPO = "kevruss/PRISM-70B"

# Dataset
DATASET = "kevruss/PRISM-V8-Dataset"

# QLoRA config (4-bit quantization for memory efficiency)
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA config (matching V8 success)
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# Training config
TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=1,
    save_strategy="epoch",
    evaluation_strategy="no",
    push_to_hub=True,
    hub_model_id=HF_REPO,
    hub_private_repo=True,
    report_to="tensorboard",
)


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    print("=" * 60)
    print("PRISM-70B Training (V8 Composable Personality Suite)")
    print("=" * 60)
    print()

    # Check GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"GPU {i}: {name} ({mem:.1f}GB)")
    else:
        print("WARNING: No GPU detected!")
        return

    # Load dataset
    print(f"\nLoading dataset: {DATASET}")
    dataset = load_dataset(DATASET, split="train")
    print(f"  Training samples: {len(dataset)}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with 4-bit quantization
    print(f"\nLoading model with QLoRA: {BASE_MODEL}")
    print("  This may take 10-15 minutes for 72B...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=QUANTIZATION_CONFIG,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    print("\nApplying LoRA adapters...")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # Create trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
    )

    # Train!
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print()

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    trainer.push_to_hub()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Pushed to: https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    main()
