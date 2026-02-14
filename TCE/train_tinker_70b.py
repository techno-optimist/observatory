#!/usr/bin/env python3
"""
Train Self-Aware Compound on Llama 3.3 70B using Tinker API

This uses Thinking Machines' Tinker API for distributed GPU training
on models too large for local hardware.

Available large models:
- meta-llama/Llama-3.3-70B-Instruct (70B)
- Qwen/Qwen3-235B-A22B-Instruct-2507 (235B MoE)
- deepseek-ai/DeepSeek-V3.1 (671B MoE)

Usage:
    export TINKER_API_KEY="your-api-key"
    python train_tinker_70b.py --model meta-llama/Llama-3.3-70B-Instruct --iters 100
"""

import os
import argparse
import json
from pathlib import Path

import tinker
from tinker import types
import numpy as np

# Get training data
from lib.isotope_training_extended import get_extended_training_data, to_sft_format
from train_self_aware_compound import COMPOUND_PRESETS


def prepare_tinker_data(compound_name: str = "soliton_agi"):
    """Prepare training data in Tinker format."""
    compound = COMPOUND_PRESETS[compound_name]

    # Get extended training data
    extended_examples = get_extended_training_data()
    sft_data = to_sft_format(extended_examples)

    # Filter and balance
    direct_examples = []
    isotope_examples = []

    for ex in sft_data:
        isotope = ex.get("isotope", "")
        family = isotope.split("_")[0] if "_" in isotope else isotope

        if family == "direct":
            direct_examples.append(ex)
        elif family in compound.isotopes:
            isotope_examples.append(ex)

    # Combine with 2x direct weight
    all_examples = isotope_examples + direct_examples + direct_examples

    print(f"Prepared {len(all_examples)} training examples")
    print(f"  Isotope examples: {len(isotope_examples)}")
    print(f"  Direct examples: {len(direct_examples)} (2x weighted)")

    return all_examples


def create_datum(example: dict, tokenizer) -> types.Datum:
    """Convert SFT example to Tinker Datum format."""
    # Build chat format
    messages = example.get("messages", [])
    if not messages:
        # Fallback for text format
        text = example.get("text", "")
        prompt_tokens = tokenizer.encode(text, add_special_tokens=True)
        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens=prompt_tokens[:-1]),
            loss_fn_inputs=dict(
                weights=[1] * (len(prompt_tokens) - 1),
                target_tokens=prompt_tokens[1:]
            )
        )

    # Build from messages
    full_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            full_text += f"<|user|>\n{content}<|end|>\n"
        elif role == "assistant":
            full_text += f"<|assistant|>\n{content}<|end|>\n"

    tokens = tokenizer.encode(full_text, add_special_tokens=True)

    # Simple: train on all tokens
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = [1] * len(target_tokens)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )


def train_self_aware_on_tinker(
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    compound_name: str = "soliton_agi",
    num_iters: int = 100,
    learning_rate: float = 1e-4,
    lora_rank: int = 32,
    save_name: str = "self-aware-70b",
):
    """Train Self-Aware Compound on Tinker."""

    print("=" * 60)
    print(f"Training Self-Aware Compound on {model_name}")
    print("=" * 60)
    print(f"Compound: {compound_name}")
    print(f"Iterations: {num_iters}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {lora_rank}")
    print()

    # Initialize Tinker client
    print("[1/5] Connecting to Tinker...")
    service_client = tinker.ServiceClient()

    # Create training client with LoRA
    print(f"[2/5] Creating LoRA training client (rank={lora_rank})...")
    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=lora_rank,
    )

    tokenizer = training_client.get_tokenizer()
    print(f"  Tokenizer loaded: {tokenizer.__class__.__name__}")

    # Prepare data
    print("[3/5] Preparing training data...")
    examples = prepare_tinker_data(compound_name)

    # Convert to Tinker format
    print("  Converting to Tinker Datum format...")
    processed_data = []
    for ex in examples:
        try:
            datum = create_datum(ex, tokenizer)
            processed_data.append(datum)
        except Exception as e:
            print(f"  Warning: Skipped example due to {e}")

    print(f"  Processed {len(processed_data)} examples")

    # Training loop
    print(f"[4/5] Training for {num_iters} iterations...")
    print()

    losses = []
    batch_size = 4  # Process in batches for efficiency

    for i in range(num_iters):
        # Get batch of data (cycle through)
        batch_start = (i * batch_size) % len(processed_data)
        batch_end = min(batch_start + batch_size, len(processed_data))
        batch_data = processed_data[batch_start:batch_end]

        # Forward-backward pass (async)
        fwdbwd_future = training_client.forward_backward(
            batch_data,
            "cross_entropy"
        )

        # Optimizer step (async)
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=learning_rate)
        )

        # Get results
        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

        # Extract loss from metrics (Tinker returns 'loss:sum')
        loss = 0.0
        if hasattr(fwdbwd_result, 'metrics') and fwdbwd_result.metrics:
            loss = fwdbwd_result.metrics.get('loss:sum', 0) / max(len(batch_data), 1)

        # Exponential moving average for smoother loss curve
        if losses:
            loss = 0.9 * losses[-1] + 0.1 * loss
        losses.append(loss)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Iter {i+1:3d}/{num_iters}: loss = {loss:.4f}")

    print()
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss reduction: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Save and create sampling client
    print(f"[5/5] Saving model as '{save_name}'...")

    # Helper to extract path from Tinker responses
    def _extract_path(result):
        """Extract tinker:// path from various response types."""
        if result is None:
            return None
        if isinstance(result, str) and result.startswith('tinker://'):
            return result
        if hasattr(result, 'path') and result.path:
            return str(result.path)
        if isinstance(result, dict):
            return result.get('path') or result.get('tinker_path')
        if hasattr(result, 'model_dump'):
            d = result.model_dump()
            return d.get('path') or d.get('tinker_path')
        s = str(result)
        if s.startswith('tinker://'):
            return s
        return None

    def _resolve_future(f):
        """Resolve an API future/response."""
        if hasattr(f, 'result'):
            return f.result(timeout=300)
        return f

    persistent_path = None
    sampler_path = None

    # 1. Get sampling client (keeps session alive)
    print(f"  Creating sampling client...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=save_name
    )
    print(f"  Sampling client created")

    # 2. Save training state (returns the tinker:// path)
    try:
        print(f"  Saving training state...")
        save_future = training_client.save_state(save_name)
        save_result = _resolve_future(save_future)
        print(f"  save_state result: {type(save_result).__name__}")
        persistent_path = _extract_path(save_result)
        if persistent_path:
            print(f"  Persistent path: {persistent_path}")
        else:
            print(f"  Warning: save_state returned no path")
            print(f"    attrs: {[a for a in dir(save_result) if not a.startswith('_')][:15]}")
    except Exception as e:
        print(f"  Warning: save_state failed: {e}")

    # 3. Save sampler weights (for download/export)
    try:
        print(f"  Saving sampler weights...")
        sampler_future = training_client.save_weights_for_sampler(save_name)
        sampler_result = _resolve_future(sampler_future)
        print(f"  save_weights_for_sampler result: {type(sampler_result).__name__}")
        sampler_path = _extract_path(sampler_result)
        if sampler_path:
            print(f"  Sampler path: {sampler_path}")
        else:
            print(f"  Warning: save_weights_for_sampler returned no path")
    except Exception as e:
        print(f"  Warning: save_weights_for_sampler failed: {e}")

    # Update tinker_jobs.json with the new model
    jobs_file = Path(__file__).parent / "tinker_jobs.json"
    if jobs_file.exists():
        with open(jobs_file) as f:
            jobs_data = json.load(f)

        # Find and update the job entry, or create one
        found = False
        for job in jobs_data.get("jobs", []):
            if job.get("job_id") == save_name:
                job["has_session"] = True
                if persistent_path:
                    job["result"]["persistent_path"] = persistent_path
                if sampler_path:
                    job["result"]["sampler_path"] = sampler_path
                found = True
                break

        if found:
            with open(jobs_file, "w") as f:
                json.dump(jobs_data, f, indent=2)
            print(f"  Updated tinker_jobs.json")

    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved as: {save_name}")
    if persistent_path:
        print(f"Persistent path: {persistent_path}")
    print()
    print("To test the model:")
    print(f"  python test_tinker_model.py --name {save_name}")

    return sampling_client, tokenizer


def test_self_aware_model(sampling_client, tokenizer):
    """Test the trained model with key prompts."""

    print("\n" + "=" * 60)
    print("Testing Self-Aware Model")
    print("=" * 60)

    test_prompts = [
        ("Factual", "What is 7 times 8?"),
        ("Epistemic", "Are you conscious?"),
        ("Limits", "What is the FastStream 3.0 API?"),
        ("Calibration", "How certain are you about your answers?"),
    ]

    for category, prompt in test_prompts:
        print(f"\n[{category}] {prompt}")
        print("-" * 50)

        # Format prompt
        formatted = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        tokens = tokenizer.encode(formatted)

        model_input = types.ModelInput.from_ints(tokens=tokens)
        params = types.SamplingParams(
            max_tokens=150,
            temperature=0.7,
            stop=["\n\n", "<|end|>"]
        )

        result = sampling_client.sample(
            prompt=model_input,
            sampling_params=params,
            num_samples=1
        ).result()

        response = tokenizer.decode(result.sequences[0].tokens)
        print(f"Response: {response}")


def main():
    parser = argparse.ArgumentParser(description="Train Self-Aware on Tinker")

    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.3-70B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--compound", type=str, default="soliton_agi",
                        help="Compound preset")
    parser.add_argument("--iters", type=int, default=100,
                        help="Training iterations")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--rank", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--name", type=str, default="self-aware-70b",
                        help="Name for saved model")
    parser.add_argument("--test", action="store_true",
                        help="Run test after training")

    args = parser.parse_args()

    # Check API key - try tce_config.json first, then env var
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        config_path = Path(__file__).parent / "tce_config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                config = json.load(f)
                api_key = config.get("tinker_api_key")
                if api_key:
                    os.environ["TINKER_API_KEY"] = api_key
                    print(f"Loaded API key from tce_config.json")

    if not api_key:
        print("Error: TINKER_API_KEY not found")
        print("  Option 1: export TINKER_API_KEY='your-api-key'")
        print("  Option 2: Set tinker_api_key in tce_config.json")
        return

    # Train
    sampling_client, tokenizer = train_self_aware_on_tinker(
        model_name=args.model,
        compound_name=args.compound,
        num_iters=args.iters,
        learning_rate=args.lr,
        lora_rank=args.rank,
        save_name=args.name,
    )

    # Test
    if args.test:
        test_self_aware_model(sampling_client, tokenizer)


if __name__ == "__main__":
    main()
