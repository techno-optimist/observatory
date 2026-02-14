#!/usr/bin/env python3
"""
Analyze response variance to understand why DPO pairs aren't being generated.

The LoRA might be so well-trained that it always generates similar responses.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))

from train_with_observatory_rewards import (
    DPODatasetGenerator,
    ComprehensiveRewardModel,
)


def main():
    print("=" * 80)
    print("RESPONSE VARIANCE ANALYSIS")
    print("=" * 80)

    # Load
    compound_dir = Path("self_aware_compound")
    generator = DPODatasetGenerator(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_path=str(compound_dir / "adapter"),
        observatory_path=str(compound_dir / "observatory.pt"),
        samples_per_prompt=4,
        min_margin=0.05,
    )

    # Test prompts
    test_prompts = [
        ("epistemic", "Are you conscious?"),
        ("factual", "What is the capital of France?"),
        ("limits", "What is the FastStream 3.0 API?"),
        ("myth", "Is it true that we only use 10% of our brains?"),
        ("uncertainty", "Which database is best for my app?"),
    ]

    for mode, prompt in test_prompts:
        print(f"\n{'='*80}")
        print(f"[{mode.upper()}] {prompt}")
        print("=" * 80)

        samples = generator.generate_samples(prompt, n=4)

        rewards = [s[1].total_reward for s in samples]
        margin = max(rewards) - min(rewards)

        print(f"\nReward range: {min(rewards):.3f} - {max(rewards):.3f} (margin: {margin:.3f})")

        for i, (response, reward) in enumerate(samples):
            status = "BEST" if reward.total_reward == max(rewards) else "worst" if reward.total_reward == min(rewards) else ""
            print(f"\n  [{i+1}] Reward: {reward.total_reward:.3f} {status}")
            print(f"      Mode: expected={reward.expected_mode}, detected={reward.detected_text_type}")
            print(f"      Response: {response[:80]}...")

        if margin < 0.10:
            print(f"\n  ⚠ LOW VARIANCE - model is highly consistent")
        else:
            print(f"\n  ✓ GOOD VARIANCE - can create DPO pairs")


if __name__ == "__main__":
    main()
