#!/usr/bin/env python3
"""
Train Self-Aware Compound on Qwen2.5-14B-Instruct

For M4 MacBook Pro 24GB:
- 14B model in 4-bit: ~9GB
- LoRA training with batch_size=2: ~12GB total
- Leaves ~8GB headroom for activations

Key differences from 7B:
- Larger hidden dimension (5120 vs 3584)
- More capacity to learn isotopes without corruption
- May need less aggressive training (lower lr, fewer iters)

Usage:
    python train_14b.py --iters 200
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Import the training infrastructure
sys.path.insert(0, str(Path(__file__).parent))

from train_self_aware_compound import (
    COMPOUND_PRESETS,
    CompoundConfig,
    prepare_training_data,
    train_observatory_on_compound,
    validate_self_awareness,
)


def train_14b_with_mlx(
    data_dir: Path,
    output_dir: Path,
    iters: int = 200,
    learning_rate: float = 3e-5,  # Lower LR for larger model
    lora_layers: int = 16,
    lora_rank: int = 8,
    batch_size: int = 2,  # Smaller batch for memory
) -> bool:
    """
    Train Qwen2.5-14B-Instruct with MLX LoRA.

    Memory-optimized for 24GB:
    - 4-bit quantization (automatic in MLX)
    - batch_size=2 (vs 4 for 7B)
    - gradient checkpointing enabled
    """
    import yaml

    model_path = "Qwen/Qwen2.5-14B-Instruct"

    config = {
        "model": model_path,
        "train": True,
        "data": str(data_dir),
        "adapter_path": str(output_dir),
        "iters": iters,
        "learning_rate": learning_rate,
        "num_layers": lora_layers,
        "batch_size": batch_size,  # Smaller for 14B
        "fine_tune_type": "lora",
        "grad_checkpoint": True,  # Critical for memory
        "lora_parameters": {
            "rank": lora_rank,
            "alpha": lora_rank * 2,
            "dropout": 0.0,
            "scale": 1.0,
        }
    }

    config_path = output_dir.parent / "lora_config_14b.yaml"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--config", str(config_path),
    ]

    print(f"\n{'='*60}")
    print("Training Qwen2.5-14B-Instruct with LoRA")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    print(f"Iterations: {iters}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"LoRA rank: {lora_rank}")
    print()
    print("Memory estimate:")
    print(f"  Model (4-bit): ~9GB")
    print(f"  LoRA + gradients: ~3GB")
    print(f"  Total: ~12GB (safe for 24GB)")
    print()
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Train Self-Aware Compound on 14B")

    parser.add_argument("--iters", type=int, default=200,
                        help="Training iterations (default: 200)")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate (default: 3e-5, lower than 7B)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size (default: 2 for memory)")
    parser.add_argument("--compound", type=str, default="soliton_agi",
                        help="Compound preset")
    parser.add_argument("--output-dir", type=str, default="self_aware_14b",
                        help="Output directory")
    parser.add_argument("--skip-lora", action="store_true",
                        help="Skip LoRA training")
    parser.add_argument("--skip-observatory", action="store_true",
                        help="Skip observatory training")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after training")

    args = parser.parse_args()

    print("=" * 60)
    print("Self-Aware Compound Training (14B)")
    print("=" * 60)
    print()
    print("Hardware: M4 MacBook Pro 24GB")
    print("Model: Qwen/Qwen2.5-14B-Instruct")
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
    data_paths = prepare_training_data(compound, data_dir)

    # LoRA training
    adapter_dir = output_dir / "adapter"

    if not args.skip_lora:
        print("\n[2/4] Training LoRA adapter...")
        success = train_14b_with_mlx(
            data_dir=data_dir,
            output_dir=adapter_dir,
            iters=args.iters,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

        if not success:
            print("LoRA training failed!")
            return
    else:
        print("\n[2/4] Skipping LoRA training (--skip-lora)")

    # Observatory training
    if not args.skip_observatory:
        print("\n[3/4] Training observatory...")
        observatory_path = output_dir / "observatory.pt"
        success = train_observatory_on_compound(
            model_path="Qwen/Qwen2.5-14B-Instruct",
            adapter_path=adapter_dir,
            compound=compound,
            output_path=observatory_path,
            epochs=20,
        )

        if not success:
            print("Observatory training failed!")
            return
    else:
        print("\n[3/4] Skipping observatory training (--skip-observatory)")

    # Validation
    if args.validate:
        print("\n[4/4] Validating self-awareness...")
        observatory_path = output_dir / "observatory.pt"
        if observatory_path.exists():
            validation_results = validate_self_awareness(
                model_path="Qwen/Qwen2.5-14B-Instruct",
                adapter_path=adapter_dir,
                observatory_path=observatory_path,
                compound=compound,
                num_samples=5,
            )
    else:
        print("\n[4/4] Skipping validation (use --validate to enable)")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nOutput: {output_dir}")
    print(f"  - Adapter: {adapter_dir}")
    print(f"  - Observatory: {output_dir / 'observatory.pt'}")
    print()
    print("To test:")
    print(f"  python benchmark_comparison.py --model 14b --adapter {adapter_dir}")


if __name__ == "__main__":
    main()
