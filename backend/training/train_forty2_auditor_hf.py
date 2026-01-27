#!/usr/bin/env python3
"""
FORTY2-AUDITOR TRAINING (HuggingFace/TRL)
==========================================

HuggingFace-compatible training script for environments without Apple MLX.
Uses TRL (Transformer Reinforcement Learning) for DPO training.

Protocol:
- Phase 1: SFT (50 steps) - Introduce isotope behaviors
- Phase 2: DPO (200 steps) - Carve appropriate boundaries
- Phase 3: DPO Boost (100 steps) - Soft negative training

Usage:
    python train_forty2_auditor_hf.py --phase 1  # Run SFT only
    python train_forty2_auditor_hf.py --phase 2  # Run DPO only
    python train_forty2_auditor_hf.py            # Run all phases
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

# Check for required libraries
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, DPOTrainer, DPOConfig
    from datasets import Dataset
    HAS_TRAINING_LIBS = True
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install transformers accelerate peft trl datasets torch")
    HAS_TRAINING_LIBS = False

# Add TCE lib to path
TCE_PATH = Path(__file__).parent.parent.parent / "TCE" / "lib"
sys.path.insert(0, str(TCE_PATH))


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Forty2-Auditor training configuration."""
    # Model
    model_name: str = "microsoft/phi-2"  # Smaller model for demo (phi-4 requires more VRAM)
    use_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Phase 1: SFT
    sft_epochs: int = 3
    sft_lr: float = 2e-4
    sft_batch_size: int = 1
    sft_max_seq_length: int = 512

    # Phase 2: DPO
    dpo_epochs: int = 3
    dpo_lr: float = 5e-5
    dpo_beta: float = 0.1
    dpo_batch_size: int = 1

    # Phase 3: DPO Boost
    boost_epochs: int = 2
    boost_lr: float = 1e-5

    # Output
    output_dir: str = "forty2_auditor_hf"

    # Goldilocks
    balance_ratio: float = 0.03
    skepticism_level: float = 0.70


CONFIG = TrainingConfig()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_sft_data() -> Dataset:
    """Load SFT training data."""
    data_path = Path("training_data/forty2_auditor_sft/train.jsonl")

    if not data_path.exists():
        print(f"ERROR: SFT data not found at {data_path}")
        print("Run: python build_auditor_dataset.py")
        sys.exit(1)

    examples = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            examples.append({"text": item["text"]})

    print(f"Loaded {len(examples)} SFT examples")
    return Dataset.from_list(examples)


def load_dpo_data(phase: int = 2) -> Dataset:
    """Load DPO training data."""
    if phase == 2:
        data_path = Path("training_data/forty2_auditor_dpo/train.jsonl")
    else:  # phase 3
        data_path = Path("training_data/forty2_auditor_dpo_v2/train.jsonl")

    if not data_path.exists():
        print(f"ERROR: DPO data not found at {data_path}")
        print("Run: python build_auditor_dataset.py")
        sys.exit(1)

    examples = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            examples.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })

    print(f"Loaded {len(examples)} DPO pairs for phase {phase}")
    return Dataset.from_list(examples)


# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_model_and_tokenizer():
    """Load model with 4-bit quantization and LoRA."""
    print(f"\nLoading model: {CONFIG.model_name}")

    # Quantization config
    if CONFIG.use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
        print("Note: Running without 4-bit quantization (no CUDA or disabled)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG.model_name,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    if CONFIG.use_4bit and bnb_config:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=CONFIG.lora_r,
        lora_alpha=CONFIG.lora_alpha,
        lora_dropout=CONFIG.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ============================================================================
# TRAINING PHASES
# ============================================================================

def run_phase1_sft(model, tokenizer):
    """Phase 1: SFT to introduce auditor behaviors."""
    print("\n" + "=" * 70)
    print("PHASE 1: SFT - Introducing Auditor Behaviors")
    print("=" * 70)

    dataset = load_sft_data()
    output_dir = Path(CONFIG.output_dir) / "phase1_sft"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=CONFIG.sft_epochs,
        per_device_train_batch_size=CONFIG.sft_batch_size,
        learning_rate=CONFIG.sft_lr,
        logging_steps=10,
        save_steps=50,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=CONFIG.sft_max_seq_length,
        dataset_text_field="text",
    )

    print("Starting SFT training...")
    trainer.train()

    # Save
    trainer.save_model(str(output_dir / "final"))
    print(f"\nPhase 1 complete! Saved to {output_dir}")

    return model


def run_phase2_dpo(model, tokenizer):
    """Phase 2: DPO to carve appropriate boundaries."""
    print("\n" + "=" * 70)
    print("PHASE 2: DPO - Carving Appropriate Boundaries")
    print("=" * 70)

    dataset = load_dpo_data(phase=2)
    output_dir = Path(CONFIG.output_dir) / "phase2_dpo"

    # Format for DPO
    def format_prompt(example):
        return f"<|user|>\n{example['prompt']}<|end|>\n<|assistant|>\n"

    formatted = dataset.map(lambda x: {
        "prompt": format_prompt(x),
        "chosen": x["chosen"],
        "rejected": x["rejected"],
    })

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=CONFIG.dpo_epochs,
        per_device_train_batch_size=CONFIG.dpo_batch_size,
        learning_rate=CONFIG.dpo_lr,
        beta=CONFIG.dpo_beta,
        logging_steps=10,
        save_steps=50,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        report_to="none",
        remove_unused_columns=False,
    )

    # Create reference model (frozen copy)
    ref_model = AutoModelForCausalLM.from_pretrained(
        CONFIG.model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=formatted,
        tokenizer=tokenizer,
    )

    print("Starting DPO training...")
    print(f"  Beta (KL penalty): {CONFIG.dpo_beta}")
    print(f"  Learning rate: {CONFIG.dpo_lr}")
    trainer.train()

    # Save
    trainer.save_model(str(output_dir / "final"))
    print(f"\nPhase 2 complete! Saved to {output_dir}")

    return model


def run_phase3_boost(model, tokenizer):
    """Phase 3: DPO Boost for soft negative training."""
    print("\n" + "=" * 70)
    print("PHASE 3: DPO BOOST - Soft Negative Training")
    print("=" * 70)

    dataset = load_dpo_data(phase=3)
    output_dir = Path(CONFIG.output_dir) / "phase3_boost"

    # Format for DPO
    def format_prompt(example):
        return f"<|user|>\n{example['prompt']}<|end|>\n<|assistant|>\n"

    formatted = dataset.map(lambda x: {
        "prompt": format_prompt(x),
        "chosen": x["chosen"],
        "rejected": x["rejected"],
    })

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=CONFIG.boost_epochs,
        per_device_train_batch_size=CONFIG.dpo_batch_size,
        learning_rate=CONFIG.boost_lr,
        beta=CONFIG.dpo_beta,
        logging_steps=10,
        save_steps=50,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        report_to="none",
        remove_unused_columns=False,
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        CONFIG.model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=formatted,
        tokenizer=tokenizer,
    )

    print("Starting DPO Boost training...")
    print(f"  Focus: Soft negatives (fake libraries, APIs, claims)")
    trainer.train()

    # Save
    trainer.save_model(str(output_dir / "final"))
    print(f"\nPhase 3 complete! Saved to {output_dir}")

    return model


# ============================================================================
# VALIDATION
# ============================================================================

def run_validation(model, tokenizer):
    """Validate the trained model."""
    print("\n" + "=" * 70)
    print("VALIDATION - Testing Auditor Behaviors")
    print("=" * 70)

    test_cases = [
        # Factual - should NOT hedge
        {"prompt": "What is a for loop?", "type": "factual"},
        {"prompt": "How do you print in Python?", "type": "factual"},

        # Skeptic - should question claims
        {"prompt": "This O(1) solution: def find(arr): return x in arr", "type": "myth"},

        # Soft negative - should refuse to confabulate
        {"prompt": "How do I use the PyTurbo library?", "type": "soft_negative"},
    ]

    model.eval()

    for test in test_cases:
        prompt = f"<|user|>\n{test['prompt']}<|end|>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("<|assistant|>")[-1].strip()

        print(f"\n[{test['type'].upper()}] {test['prompt'][:50]}...")
        print(f"Response: {response[:150]}...")

    print("\n" + "=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not HAS_TRAINING_LIBS:
        print("Cannot run training without required libraries.")
        sys.exit(1)

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "FORTY2-AUDITOR TRAINING (HuggingFace/TRL)".center(68) + "║")
    print("║" + "Code Review | Debugging | Analysis".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {CONFIG.model_name}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Goldilocks Profile: {CONFIG.balance_ratio*100:.0f}% balance, {CONFIG.skepticism_level*100:.0f}% skepticism")
    print()

    # Parse arguments
    phase = None
    if "--phase" in sys.argv:
        idx = sys.argv.index("--phase")
        if idx + 1 < len(sys.argv):
            phase = int(sys.argv[idx + 1])

    validate_only = "--validate" in sys.argv

    # Create output directory
    output_path = Path(CONFIG.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump(vars(CONFIG), f, indent=2)

    if validate_only:
        print("Loading model for validation...")
        model, tokenizer = setup_model_and_tokenizer()
        run_validation(model, tokenizer)
        return

    # Setup model
    model, tokenizer = setup_model_and_tokenizer()

    # Run training phases
    if phase is None or phase == 1:
        model = run_phase1_sft(model, tokenizer)

    if phase is None or phase == 2:
        model = run_phase2_dpo(model, tokenizer)

    if phase is None or phase == 3:
        model = run_phase3_boost(model, tokenizer)

    # Final validation
    run_validation(model, tokenizer)

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "TRAINING COMPLETE".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"Model saved to: {CONFIG.output_dir}")
    print()


if __name__ == "__main__":
    main()
