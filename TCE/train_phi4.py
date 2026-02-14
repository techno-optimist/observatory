#!/usr/bin/env python3
"""
Train Self-Aware Compound on Microsoft Phi-4 14B

Memory-optimized for M4 MacBook Pro 24GB:
- Uses pre-quantized mlx-community/phi-4-4bit (8.25GB)
- Minimal LoRA: 8 layers, rank 4
- batch_size=1
- Gradient checkpointing

Usage:
    python train_phi4.py --iters 200 --validate
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


# Phi-4 model path (pre-quantized 4-bit for Apple Silicon)
PHI4_MODEL = "mlx-community/phi-4-4bit"


def train_phi4_with_mlx(
    data_dir: Path,
    output_dir: Path,
    iters: int = 200,
    learning_rate: float = 2e-5,  # Lower for larger model
    lora_layers: int = 8,         # Reduced for memory
    lora_rank: int = 4,           # Reduced for memory
    batch_size: int = 1,          # Minimum
) -> bool:
    """
    Train Phi-4 14B with aggressive memory optimization.

    Memory budget (24GB):
    - Model (4-bit): 8.25GB
    - LoRA (rank 4, 8 layers): ~0.5GB
    - Activations (batch=1): ~6GB
    - Gradients + optimizer: ~6GB
    - Total: ~21GB (should fit!)
    """
    import yaml

    config = {
        "model": PHI4_MODEL,
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

    config_path = output_dir.parent / "lora_config_phi4.yaml"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--config", str(config_path),
    ]

    print(f"\n{'='*60}")
    print("Training Microsoft Phi-4 14B")
    print(f"{'='*60}")
    print(f"Model: {PHI4_MODEL}")
    print()
    print("Memory-optimized configuration:")
    print(f"  LoRA layers: {lora_layers} (reduced from 16)")
    print(f"  LoRA rank: {lora_rank} (reduced from 8)")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {iters}")
    print(f"  Learning rate: {learning_rate}")
    print()
    print("Memory estimate:")
    print(f"  Model (4-bit): ~8.25GB")
    print(f"  LoRA + gradients: ~6GB")
    print(f"  Activations: ~6GB")
    print(f"  Total: ~21GB (target: <24GB)")
    print()
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Train Self-Aware Compound on Phi-4")

    parser.add_argument("--iters", type=int, default=200,
                        help="Training iterations")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (lower for 14B)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (1 for memory)")
    parser.add_argument("--lora-layers", type=int, default=8,
                        help="Number of LoRA layers (reduced for memory)")
    parser.add_argument("--lora-rank", type=int, default=4,
                        help="LoRA rank (reduced for memory)")
    parser.add_argument("--compound", type=str, default="soliton_agi",
                        help="Compound preset")
    parser.add_argument("--output-dir", type=str, default="self_aware_phi4",
                        help="Output directory")
    parser.add_argument("--skip-lora", action="store_true",
                        help="Skip LoRA training")
    parser.add_argument("--skip-observatory", action="store_true",
                        help="Skip observatory training")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after training")

    args = parser.parse_args()

    print("=" * 60)
    print("Self-Aware Compound Training (Phi-4 14B)")
    print("=" * 60)
    print()
    print("Hardware: M4 MacBook Pro 24GB")
    print(f"Model: {PHI4_MODEL}")
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
        print("\n[2/4] Training LoRA adapter on Phi-4...")
        success = train_phi4_with_mlx(
            data_dir=data_dir,
            output_dir=adapter_dir,
            iters=args.iters,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            lora_layers=args.lora_layers,
            lora_rank=args.lora_rank,
        )

        if not success:
            print("\n" + "="*60)
            print("LoRA training failed (likely OOM)")
            print("="*60)
            print("\nFallback options:")
            print("1. Try even smaller config: --lora-layers 4 --lora-rank 2")
            print("2. Use Enhanced Qwen 7B (75% benchmark, fits in 16.5GB)")
            print("3. Get more RAM (32GB+ needed for 14B training)")
            return
    else:
        print("\n[2/4] Skipping LoRA training")

    # Observatory training
    if not args.skip_observatory:
        print("\n[3/4] Training observatory...")
        observatory_path = output_dir / "observatory.pt"
        success = train_observatory_on_compound(
            model_path=PHI4_MODEL,
            adapter_path=adapter_dir,
            compound=compound,
            output_path=observatory_path,
            epochs=20,
        )

        if not success:
            print("Observatory training failed!")
            return
    else:
        print("\n[3/4] Skipping observatory training")

    # Validation
    if args.validate:
        print("\n[4/4] Validating self-awareness...")
        observatory_path = output_dir / "observatory.pt"
        if observatory_path.exists():
            validate_self_awareness(
                model_path=PHI4_MODEL,
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
    print(f"  python3 benchmark_comparison.py --compound-dir {output_dir}")


if __name__ == "__main__":
    main()
