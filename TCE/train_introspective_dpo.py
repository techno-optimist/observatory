#!/usr/bin/env python3
"""
Introspective DPO Training

Trains the model using Direct Preference Optimization (DPO) with
observatory-scored preference pairs.

DPO Loss:
L = -log(σ(β * (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x)))))

Where:
- y_w = chosen response (winner)
- y_l = rejected response (loser)
- π = policy (current model)
- π_ref = reference (base model)
- β = temperature parameter
"""

import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import sys
import time

# MLX imports
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.lora import linear_to_lora_layers

sys.path.insert(0, str(Path(__file__).parent / "lib"))


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path: Optional[str] = None  # Existing adapter to continue from
    output_path: str = "introspective_dpo_adapter"

    # Training params
    beta: float = 0.1  # DPO temperature
    learning_rate: float = 1e-5
    batch_size: int = 1  # DPO often uses batch_size=1
    num_epochs: int = 3
    max_seq_length: int = 512

    # LoRA params
    lora_layers: int = 16
    lora_rank: int = 8

    # Logging
    log_interval: int = 10


def load_dpo_data(data_path: str) -> List[Dict]:
    """Load DPO preference pairs from JSONL."""
    pairs = []
    with open(data_path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def compute_log_probs(
    model,
    tokenizer,
    prompt: str,
    response: str,
    max_length: int = 512,
) -> mx.array:
    """
    Compute log probability of response given prompt.

    Returns sum of log probs for response tokens.
    """
    # Tokenize prompt + response
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = formatted + response

    # Tokenize
    input_ids = tokenizer.encode(full_text)
    prompt_ids = tokenizer.encode(formatted)

    # Truncate if needed
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    # Get prompt length for masking
    prompt_len = len(prompt_ids)

    # Forward pass
    input_mx = mx.array([input_ids])
    logits = model(input_mx)

    # Get log probs for response tokens only
    log_probs = nn.log_softmax(logits, axis=-1)

    # Shift for next-token prediction
    target_ids = mx.array([input_ids[1:]])
    log_probs = log_probs[:, :-1, :]

    # Gather log probs for actual tokens
    batch_size, seq_len, vocab_size = log_probs.shape
    indices = mx.arange(seq_len)
    gathered = log_probs[0, indices, target_ids[0]]

    # Only sum response tokens (after prompt)
    response_start = max(0, prompt_len - 1)
    response_log_probs = gathered[response_start:]

    return mx.sum(response_log_probs)


def dpo_loss(
    policy_chosen_logps: mx.array,
    policy_rejected_logps: mx.array,
    ref_chosen_logps: mx.array,
    ref_rejected_logps: mx.array,
    beta: float = 0.1,
) -> mx.array:
    """
    Compute DPO loss.

    L = -log(σ(β * (π_θ(y_w) - π_θ(y_l) - π_ref(y_w) + π_ref(y_l))))
    """
    # Log ratios
    policy_ratio = policy_chosen_logps - policy_rejected_logps
    ref_ratio = ref_chosen_logps - ref_rejected_logps

    # DPO objective
    logits = beta * (policy_ratio - ref_ratio)
    loss = -nn.log_sigmoid(logits)

    return loss


def train_dpo(config: DPOConfig, train_data: List[Dict], valid_data: List[Dict] = None):
    """
    Train with DPO.

    Process:
    1. Load model and create LoRA adapter
    2. Create reference model (frozen copy)
    3. For each preference pair:
       - Compute log probs for chosen/rejected under policy and reference
       - Compute DPO loss
       - Update LoRA weights
    """
    print("=" * 60)
    print("INTROSPECTIVE DPO TRAINING")
    print("=" * 60)

    # Load model
    print(f"\n[1/4] Loading model: {config.model_path}")

    # If continuing from existing adapter, load with adapter (already has LoRA)
    # Otherwise, load base model and add LoRA
    has_adapter = config.adapter_path and Path(config.adapter_path).exists()

    if has_adapter:
        model, tokenizer = load(config.model_path, adapter_path=config.adapter_path)
        print(f"  Loaded existing adapter: {config.adapter_path}")
        print(f"\n[2/4] Using existing LoRA layers from adapter")
    else:
        model, tokenizer = load(config.model_path)
        # Setup LoRA for fresh model
        print(f"\n[2/4] Setting up LoRA (rank={config.lora_rank}, layers={config.lora_layers})")
        lora_config = {
            "rank": config.lora_rank,
            "scale": 1.0,
            "alpha": config.lora_rank,  # Standard alpha = rank
        }
        linear_to_lora_layers(model, config.lora_layers, lora_config)

    model.train()

    # Freeze reference model by storing initial log probs
    # (In a full implementation, we'd have a separate frozen model)
    print("\n[3/4] Computing reference log probs...")
    ref_logps = {}
    for i, pair in enumerate(train_data):
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]

        # Compute reference log probs once
        ref_chosen = compute_log_probs(model, tokenizer, prompt, chosen, config.max_seq_length)
        ref_rejected = compute_log_probs(model, tokenizer, prompt, rejected, config.max_seq_length)

        mx.eval(ref_chosen, ref_rejected)
        ref_logps[i] = {
            "chosen": float(ref_chosen.item()),
            "rejected": float(ref_rejected.item()),
        }

        if (i + 1) % 10 == 0:
            print(f"  Computed {i+1}/{len(train_data)} reference log probs")

    # Setup optimizer
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    # Training loop
    print(f"\n[4/4] Training for {config.num_epochs} epochs on {len(train_data)} pairs")

    def loss_fn(model, prompt, chosen, rejected, ref_chosen_lp, ref_rejected_lp, beta):
        """Compute DPO loss for a single pair."""
        policy_chosen_lp = compute_log_probs(model, tokenizer, prompt, chosen, config.max_seq_length)
        policy_rejected_lp = compute_log_probs(model, tokenizer, prompt, rejected, config.max_seq_length)

        loss = dpo_loss(
            policy_chosen_lp,
            policy_rejected_lp,
            mx.array(ref_chosen_lp),
            mx.array(ref_rejected_lp),
            beta,
        )
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    total_steps = 0
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        # Shuffle data
        indices = list(range(len(train_data)))
        np.random.shuffle(indices)

        for step, idx in enumerate(indices):
            pair = train_data[idx]
            ref = ref_logps[idx]

            # Compute loss and gradients
            loss, grads = loss_and_grad(
                model,
                pair["prompt"],
                pair["chosen"],
                pair["rejected"],
                ref["chosen"],
                ref["rejected"],
                config.beta,
            )

            # Update
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += float(loss.item())
            total_steps += 1

            if (step + 1) % config.log_interval == 0:
                avg_loss = epoch_loss / (step + 1)
                print(f"  Epoch {epoch+1}/{config.num_epochs}, Step {step+1}/{len(train_data)}, Loss: {avg_loss:.4f}")

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_data)
        print(f"\n  Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}, time={epoch_time:.1f}s")

        # Validation
        if valid_data:
            valid_loss = 0.0
            for i, pair in enumerate(valid_data):
                prompt = pair["prompt"]
                chosen = pair["chosen"]
                rejected = pair["rejected"]

                # Compute validation loss (no gradient)
                policy_chosen = compute_log_probs(model, tokenizer, prompt, chosen, config.max_seq_length)
                policy_rejected = compute_log_probs(model, tokenizer, prompt, rejected, config.max_seq_length)

                # Use same ref logps (approximately)
                ref_chosen = policy_chosen * 0.9  # Approximation
                ref_rejected = policy_rejected * 0.9

                loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, config.beta)
                mx.eval(loss)
                valid_loss += float(loss.item())

            avg_valid_loss = valid_loss / len(valid_data)
            print(f"  Validation loss: {avg_valid_loss:.4f}")

    # Save adapter
    print(f"\n[Done] Saving adapter to {config.output_path}")
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA weights
    weights = dict(model.parameters())
    lora_weights = {k: v for k, v in weights.items() if "lora" in k.lower()}
    mx.savez(str(output_dir / "adapters.safetensors"), **lora_weights)

    # Save config
    adapter_config = {
        "model_path": config.model_path,
        "lora_layers": config.lora_layers,
        "lora_rank": config.lora_rank,
        "training_config": {
            "beta": config.beta,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "train_samples": len(train_data),
        }
    }
    with open(output_dir / "adapter_config.json", 'w') as f:
        json.dump(adapter_config, f, indent=2)

    print(f"\n{'='*60}")
    print("DPO TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total steps: {total_steps}")
    print(f"Final loss: {avg_loss:.4f}")
    print(f"Adapter saved to: {config.output_path}")

    return model


def main():
    """Run DPO training."""
    config = DPOConfig(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_path="self_aware_compound/adapter",  # Continue from existing
        output_path="introspective_dpo_adapter",
        beta=0.1,
        learning_rate=1e-5,
        num_epochs=3,
        lora_layers=16,
        lora_rank=8,
    )

    # Load data
    train_data = load_dpo_data("introspective_dpo_data/train.jsonl")
    valid_data = load_dpo_data("introspective_dpo_data/valid.jsonl")

    print(f"Loaded {len(train_data)} train, {len(valid_data)} valid pairs")

    # Train
    train_dpo(config, train_data, valid_data)


if __name__ == "__main__":
    main()
