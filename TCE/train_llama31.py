#!/usr/bin/env python3
"""
Train Self-Aware Compound on Meta Llama 3.1 8B

Llama 3.1 is the improved version with:
- Better instruction following
- Longer context (128K)
- Improved reasoning

Usage:
    python train_llama31.py --iters 300 --validate
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train_self_aware_compound import (
    COMPOUND_PRESETS,
    prepare_training_data,
    train_observatory_on_compound,
    validate_self_awareness,
)


# Llama 3.1 8B model (improved version)
LLAMA31_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"


def train_llama31_with_mlx(
    data_dir: Path,
    output_dir: Path,
    iters: int = 300,
    learning_rate: float = 5e-5,
    lora_layers: int = 24,
    lora_rank: int = 16,
    batch_size: int = 4,
) -> bool:
    """Train Llama 3.1 8B with aggressive LoRA."""
    import yaml

    config = {
        "model": LLAMA31_MODEL,
        "train": True,
        "data": str(data_dir),
        "adapter_path": str(output_dir),
        "iters": iters,
        "learning_rate": learning_rate,
        "num_layers": lora_layers,
        "batch_size": batch_size,
        "fine_tune_type": "lora",
        "grad_checkpoint": True,
        "lora_parameters": {
            "rank": lora_rank,
            "alpha": lora_rank * 2,
            "dropout": 0.0,
            "scale": 1.0,
        }
    }

    config_path = output_dir.parent / "lora_config_llama31.yaml"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--config", str(config_path),
    ]

    print(f"\n{'='*60}")
    print("Training Meta Llama 3.1 8B (Improved)")
    print(f"{'='*60}")
    print(f"Model: {LLAMA31_MODEL}")
    print(f"  LoRA layers: {lora_layers}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {iters}")
    print()

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Train on Llama 3.1 8B")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-layers", type=int, default=24)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--compound", type=str, default="soliton_agi")
    parser.add_argument("--output-dir", type=str, default="self_aware_llama31")
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--validate", action="store_true")

    args = parser.parse_args()

    print("=" * 60)
    print("Self-Aware Compound Training (Llama 3.1 8B)")
    print("=" * 60)
    print(f"Model: {LLAMA31_MODEL}")

    compound = COMPOUND_PRESETS[args.compound]
    print(f"Compound: {compound.name}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    print("\n[1/4] Preparing training data...")
    data_dir = output_dir / "data"
    prepare_training_data(compound, data_dir)

    # LoRA training
    adapter_dir = output_dir / "adapter"
    if not args.skip_lora:
        print("\n[2/4] Training LoRA adapter...")
        if not train_llama31_with_mlx(
            data_dir=data_dir,
            output_dir=adapter_dir,
            iters=args.iters,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            lora_layers=args.lora_layers,
            lora_rank=args.lora_rank,
        ):
            print("Training failed!")
            return

    # Observatory
    print("\n[3/4] Training observatory...")
    observatory_path = output_dir / "observatory.pt"
    train_observatory_on_compound(
        model_path=LLAMA31_MODEL,
        adapter_path=adapter_dir,
        compound=compound,
        output_path=observatory_path,
        epochs=20,
    )

    # Validation
    if args.validate:
        print("\n[4/4] Validating...")
        validate_self_awareness(
            model_path=LLAMA31_MODEL,
            adapter_path=adapter_dir,
            observatory_path=observatory_path,
            compound=compound,
            num_samples=5,
        )

    print(f"\nDone! Output: {output_dir}")
    print(f"Benchmark: python3 benchmark_comparison.py --compound-dir {output_dir}")


if __name__ == "__main__":
    main()
