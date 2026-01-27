#!/usr/bin/env python3
"""
PRISM-70B Training Script
V8 Composable Personality Suite on Qwen2.5-72B-Instruct
Uses standard Transformers Trainer (no TRL dependency issues)
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login

# Login to HuggingFace
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
OUTPUT_DIR = "./prism-70b-output"
HF_REPO = "kevruss/PRISM-70B"
DATASET = "kevruss/PRISM-V8-Dataset"
MAX_LENGTH = 2048

print("=" * 60)
print("PRISM-70B Training (V8 Composable Personality Suite)")
print("=" * 60)

# Check GPU
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"GPU {i}: {name} ({mem:.1f}GB)")
else:
    print("ERROR: No GPU detected!")
    exit(1)

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA config (matching V8 success)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# Load tokenizer first (needed for preprocessing)
print(f"\nLoading tokenizer: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load dataset
print(f"\nLoading dataset: {DATASET}")
dataset = load_dataset(DATASET, split="train", token=HF_TOKEN)
print(f"  Training samples: {len(dataset)}")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

print("\nTokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
)

# Load model
print(f"\nLoading model with QLoRA: {BASE_MODEL}")
print("  This may take 15-30 minutes for 72B...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
)

# Prepare for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, not masked LM
)

# Training arguments
training_args = TrainingArguments(
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
    push_to_hub=True,
    hub_model_id=HF_REPO,
    hub_private_repo=True,
    hub_token=HF_TOKEN,
    report_to="tensorboard",
)

# Create standard Trainer (no TRL)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train!
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)
trainer.train()

# Save
print("\nSaving model...")
trainer.save_model()
trainer.push_to_hub()

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print(f"Model: https://huggingface.co/{HF_REPO}")
print("=" * 60)

# Keep container running
import time
while True:
    time.sleep(3600)
