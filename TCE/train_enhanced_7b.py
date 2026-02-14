#!/usr/bin/env python3
"""
Train Enhanced Self-Aware Compound on Qwen2.5-7B-Instruct

For M4 MacBook Pro 24GB, maximize 7B training:
- More LoRA layers (24 vs 16)
- Higher LoRA rank (16 vs 8)
- Larger batch size (8)
- More iterations (300)

This extracts maximum introspective capability from 7B.

Alternative: Phi-4-14B (Microsoft's efficient 14B)
- But needs testing for memory fit

Usage:
    python train_enhanced_7b.py --iters 300
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train_self_aware_compound import (
    COMPOUND_PRESETS,
    CompoundConfig,
    prepare_training_data,
    train_observatory_on_compound,
    validate_self_awareness,
)


def train_enhanced_7b(
    data_dir: Path,
    output_dir: Path,
    iters: int = 300,
    learning_rate: float = 5e-5,
    lora_layers: int = 24,  # More layers
    lora_rank: int = 16,    # Higher rank
    batch_size: int = 8,    # Can afford larger batch
) -> bool:
    """
    Train Qwen2.5-7B-Instruct with enhanced LoRA.

    With 24GB:
    - 7B model: ~5GB
    - Enhanced LoRA (rank 16, 24 layers): ~2GB
    - batch_size=8: ~4GB activations
    - Total: ~11GB (comfortable)
    """
    import yaml

    model_path = "Qwen/Qwen2.5-7B-Instruct"

    config = {
        "model": model_path,
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

    config_path = output_dir.parent / "lora_config_enhanced.yaml"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--config", str(config_path),
    ]

    print(f"\n{'='*60}")
    print("Training ENHANCED Qwen2.5-7B-Instruct")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print()
    print("Enhanced parameters:")
    print(f"  LoRA layers: {lora_layers} (vs 16 standard)")
    print(f"  LoRA rank: {lora_rank} (vs 8 standard)")
    print(f"  Batch size: {batch_size} (vs 4 standard)")
    print(f"  Iterations: {iters}")
    print(f"  Learning rate: {learning_rate}")
    print()
    print("Memory estimate:")
    print(f"  Model (4-bit): ~5GB")
    print(f"  Enhanced LoRA: ~2GB")
    print(f"  Activations: ~4GB")
    print(f"  Total: ~11GB (safe for 24GB)")
    print()
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced Self-Aware Compound on 7B")

    parser.add_argument("--iters", type=int, default=300,
                        help="Training iterations")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lora-layers", type=int, default=24,
                        help="Number of LoRA layers")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--compound", type=str, default="soliton_agi",
                        help="Compound preset")
    parser.add_argument("--output-dir", type=str, default="self_aware_enhanced",
                        help="Output directory")
    parser.add_argument("--skip-lora", action="store_true",
                        help="Skip LoRA training")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after training")

    args = parser.parse_args()

    print("=" * 60)
    print("ENHANCED Self-Aware Compound Training (7B)")
    print("=" * 60)
    print()
    print("Maximizing 7B with enhanced LoRA parameters")
    print()

    # Load compound config
    if args.compound in COMPOUND_PRESETS:
        compound = COMPOUND_PRESETS[args.compound]
        print(f"Compound: {compound.name}")
        print(f"Isotopes: {compound.isotopes}")
    else:
        print(f"Unknown compound: {args.compound}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare training data
    print("\n[1/4] Preparing training data...")
    data_dir = output_dir / "data"
    prepare_training_data(compound, data_dir)

    # LoRA training
    adapter_dir = output_dir / "adapter"

    if not args.skip_lora:
        print("\n[2/4] Training enhanced LoRA adapter...")
        success = train_enhanced_7b(
            data_dir=data_dir,
            output_dir=adapter_dir,
            iters=args.iters,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            lora_layers=args.lora_layers,
            lora_rank=args.lora_rank,
        )

        if not success:
            print("LoRA training failed!")
            return
    else:
        print("\n[2/4] Skipping LoRA training")

    # Observatory training
    print("\n[3/4] Training observatory...")
    observatory_path = output_dir / "observatory.pt"
    train_observatory_on_compound(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_path=adapter_dir,
        compound=compound,
        output_path=observatory_path,
        epochs=20,
    )

    # Validation
    if args.validate:
        print("\n[4/4] Validating self-awareness...")
        if observatory_path.exists():
            validate_self_awareness(
                model_path="Qwen/Qwen2.5-7B-Instruct",
                adapter_path=adapter_dir,
                observatory_path=observatory_path,
                compound=compound,
                num_samples=5,
            )
    else:
        print("\n[4/4] Skipping validation (use --validate)")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nOutput: {output_dir}")
    print()
    print("To benchmark:")
    print(f"  python3 benchmark_comparison.py")


if __name__ == "__main__":
    main()
