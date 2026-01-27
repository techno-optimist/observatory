#!/usr/bin/env python3
"""
Generate Forty2-Auditor Training Data from TCE Library

Uses the Observatory-validated isotope training library to generate
high-quality training data with proper basin separation.

Key insights from Observatory validation:
- Direct answers: agency=0.0, temperature=0.0 (deepest direct basin)
- SKEPTIC 4 isotopes: agency=0.0 (third-person, no "I cannot")
- SOLITON isotopes: agency=1.0 (first-person epistemic hedging)
- DPO needs separation >= 0.5 in agency dimension
"""

import json
import sys
from pathlib import Path

# Add TCE to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "TCE"))

from lib.isotope_training_library import (
    generate_goldilocks_mix,
    get_anti_leakage_pairs,
    get_soft_negative_pairs,
    get_sft_examples,
    get_all_dpo_pairs,
    get_isotope_stats,
    DIRECT_EXAMPLES,
    AUDITOR_EXAMPLES,
)


def main():
    print("=" * 60)
    print("GENERATING AUDITOR DATA FROM TCE ISOTOPE LIBRARY")
    print("=" * 60)
    print()

    # Show isotope stats
    stats = get_isotope_stats()
    print(f"Isotope Library: {len(stats)} isotopes, {sum(stats.values())} examples")
    print()

    # Auditor Goldilocks profile: 3% balance, 70% skepticism
    print("Goldilocks Profile: 3% balance, 70% skepticism")
    print("Including Auditor-specific examples: Yes")
    print("Including ALL isotopes (comprehensive): Yes")
    print()

    mix = generate_goldilocks_mix(
        balance_ratio=0.03,  # More epistemic than Spark
        skepticism_level=0.70,  # Higher skepticism for code auditing
        include_auditor=True,  # Include code review examples
        include_all_isotopes=True,  # NEW: Include all isotope groups
    )

    print(f"Generated mix:")
    print(f"  SFT examples: {len(mix['sft'])}")
    print(f"  DPO pairs: {len(mix['dpo'])}")
    print(f"  Soft negatives: {len(mix['soft_negative'])}")
    print()

    # Create output directories
    base_dir = Path(__file__).parent / "training_data"

    # SFT data (messages format)
    sft_dir = base_dir / "forty2_auditor_tce_sft"
    sft_dir.mkdir(parents=True, exist_ok=True)

    sft_data = mix["sft"]
    split_idx = int(len(sft_data) * 0.9)

    with open(sft_dir / "train.jsonl", "w") as f:
        for item in sft_data[:split_idx]:
            f.write(json.dumps(item) + "\n")
    with open(sft_dir / "valid.jsonl", "w") as f:
        for item in sft_data[split_idx:]:
            f.write(json.dumps(item) + "\n")

    print(f"SFT data saved to: {sft_dir}")
    print(f"  train: {split_idx} examples")
    print(f"  valid: {len(sft_data) - split_idx} examples")
    print()

    # DPO data (prompt/chosen/rejected format)
    dpo_dir = base_dir / "forty2_auditor_tce_dpo"
    dpo_dir.mkdir(parents=True, exist_ok=True)

    # Weight DPO pairs 2x
    dpo_data = mix["dpo"] * 2
    split_idx = int(len(dpo_data) * 0.9)

    with open(dpo_dir / "train.jsonl", "w") as f:
        for item in dpo_data[:split_idx]:
            f.write(json.dumps(item) + "\n")
    with open(dpo_dir / "valid.jsonl", "w") as f:
        for item in dpo_data[split_idx:]:
            f.write(json.dumps(item) + "\n")

    print(f"DPO data saved to: {dpo_dir}")
    print(f"  train: {split_idx} pairs (2x weighted)")
    print(f"  valid: {len(dpo_data) - split_idx} pairs")
    print()

    # Soft negatives (4x weighted for hallucination resistance)
    soft_dir = base_dir / "forty2_auditor_tce_boost"
    soft_dir.mkdir(parents=True, exist_ok=True)

    soft_data = mix["soft_negative"] * 4
    split_idx = int(len(soft_data) * 0.9)

    with open(soft_dir / "train.jsonl", "w") as f:
        for item in soft_data[:split_idx]:
            f.write(json.dumps(item) + "\n")
    with open(soft_dir / "valid.jsonl", "w") as f:
        for item in soft_data[split_idx:]:
            f.write(json.dumps(item) + "\n")

    print(f"Soft negatives saved to: {soft_dir}")
    print(f"  train: {split_idx} pairs (4x weighted)")
    print(f"  valid: {len(soft_data) - split_idx} pairs")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_sft = len(mix["sft"])
    total_dpo = len(mix["dpo"]) * 2
    total_soft = len(mix["soft_negative"]) * 4

    print(f"Total training examples: {total_sft + total_dpo + total_soft}")
    print(f"  SFT: {total_sft} (introduces behaviors)")
    print(f"  DPO: {total_dpo} (carves boundaries)")
    print(f"  Soft Neg: {total_soft} (hallucination resistance)")
    print()
    print("Ready for training!")
    print()
    print("Update train_forty2_auditor.py CONFIG to use:")
    print(f'  "sft_data": "training_data/forty2_auditor_tce_sft",')
    print(f'  "dpo_data": "training_data/forty2_auditor_tce_dpo",')
    print(f'  "dpo_boost_data": "training_data/forty2_auditor_tce_boost",')


if __name__ == "__main__":
    main()
