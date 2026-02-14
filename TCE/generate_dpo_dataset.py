#!/usr/bin/env python3
"""
Generate DPO dataset using observatory rewards.

This script:
1. Generates multiple responses per prompt using the current LoRA
2. Scores each response using the comprehensive reward model
3. Creates preference pairs (chosen vs rejected)
4. Saves dataset for DPO training

Run this before train_with_observatory_rewards.py for a quick iteration.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))

from train_with_observatory_rewards import (
    DPODatasetGenerator,
    TRAINING_PROMPTS,
    create_dpo_training_data,
)


def main():
    print("=" * 80)
    print("DPO DATASET GENERATION")
    print("=" * 80)

    # Paths
    compound_dir = Path("self_aware_compound")
    dpo_dir = Path("dpo_training")
    dpo_dir.mkdir(exist_ok=True)

    adapter_path = compound_dir / "adapter"
    observatory_path = compound_dir / "observatory.pt"

    # Use a subset of prompts for quick iteration
    # (3 per category = 15 total prompts × 4 samples = 60 generations)
    quick_prompts = {
        mode: prompts[:3] for mode, prompts in TRAINING_PROMPTS.items()
    }

    total_prompts = sum(len(p) for p in quick_prompts.values())
    print(f"\n[Config] {total_prompts} prompts × 4 samples = {total_prompts * 4} generations")
    print(f"[Config] Using adapter: {adapter_path}")
    print(f"[Config] Using observatory: {observatory_path}")

    # Initialize generator
    print("\n[1/3] Loading model and reward model...")
    generator = DPODatasetGenerator(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_path=str(adapter_path) if adapter_path.exists() else None,
        observatory_path=str(observatory_path) if observatory_path.exists() else None,
        samples_per_prompt=4,
        min_margin=0.10,  # Lower threshold for more pairs
    )

    # Generate dataset
    print("\n[2/3] Generating DPO pairs...")
    dataset_path = dpo_dir / "dpo_dataset_quick.json"

    dataset = generator.generate_dataset(
        quick_prompts,
        str(dataset_path),
        verbose=True,
    )

    # Create training format
    print("\n[3/3] Creating training data...")
    train_dir = dpo_dir / "train_data_quick"
    create_dpo_training_data(str(dataset_path), str(train_dir))

    # Summary
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    with open(dataset_path) as f:
        data = json.load(f)

    print(f"\nTotal examples: {len(data['examples'])}")
    print(f"By mode: {data['stats']['by_mode']}")

    # Show some examples
    print("\n[Sample Pairs]")
    for ex in data['examples'][:3]:
        print(f"\n  Mode: {ex['expected_mode']}")
        print(f"  Prompt: {ex['prompt'][:50]}...")
        print(f"  Chosen ({ex['chosen_reward']:.2f}): {ex['chosen'][:60]}...")
        print(f"  Rejected ({ex['rejected_reward']:.2f}): {ex['rejected'][:60]}...")
        print(f"  Margin: {ex['margin']:.2f}")

    print(f"\n[Output] Dataset: {dataset_path}")
    print(f"[Output] Training data: {train_dir}")


if __name__ == "__main__":
    main()
