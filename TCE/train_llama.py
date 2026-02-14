#!/usr/bin/env python3
"""
Train Self-Aware Compound on Meta Llama 3 8B

Similar configuration to Enhanced Qwen 7B:
- Aggressive LoRA (24 layers, rank 16)
- batch_size=4
- 300 iterations

Usage:
    python train_llama.py --iters 300 --validate
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


# Llama 3 8B model path (pre-quantized 4-bit for Apple Silicon)
LLAMA_MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"


def train_llama_with_mlx(
    data_dir: Path,
    output_dir: Path,
    iters: int = 300,
    learning_rate: float = 5e-5,
    lora_layers: int = 24,    # Aggressive like enhanced Qwen 7B
    lora_rank: int = 16,      # High rank
    batch_size: int = 4,
) -> bool:
    """
    Train Llama 3 8B with aggressive LoRA configuration.

    Memory budget (24GB):
    - Model (4-bit): ~5GB
    - LoRA (rank 16, 24 layers): ~2GB
    - Activations (batch=4): ~4GB
    - Total: ~11GB (comfortable)
    """
    import yaml

    config = {
        "model": LLAMA_MODEL,
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

    config_path = output_dir.parent / "lora_config_llama.yaml"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--config", str(config_path),
    ]

    print(f"\n{'='*60}")
    print("Training Meta Llama 3 8B")
    print(f"{'='*60}")
    print(f"Model: {LLAMA_MODEL}")
    print()
    print("Configuration (matching Enhanced Qwen 7B):")
    print(f"  LoRA layers: {lora_layers}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {iters}")
    print(f"  Learning rate: {learning_rate}")
    print()
    print("Memory estimate:")
    print(f"  Model (4-bit): ~5GB")
    print(f"  LoRA + gradients: ~6GB")
    print(f"  Activations: ~4GB")
    print(f"  Total: ~15GB (safe for 24GB)")
    print()
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Train Self-Aware Compound on Llama 3 8B")

    parser.add_argument("--iters", type=int, default=300,
                        help="Training iterations")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lora-layers", type=int, default=24,
                        help="Number of LoRA layers")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--compound", type=str, default="soliton_agi",
                        help="Compound preset")
    parser.add_argument("--output-dir", type=str, default="self_aware_llama",
                        help="Output directory")
    parser.add_argument("--skip-lora", action="store_true",
                        help="Skip LoRA training")
    parser.add_argument("--skip-observatory", action="store_true",
                        help="Skip observatory training")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after training")

    args = parser.parse_args()

    print("=" * 60)
    print("Self-Aware Compound Training (Llama 3 8B)")
    print("=" * 60)
    print()
    print("Hardware: M4 MacBook Pro 24GB")
    print(f"Model: {LLAMA_MODEL}")
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
        print("\n[2/4] Training LoRA adapter on Llama 3...")
        success = train_llama_with_mlx(
            data_dir=data_dir,
            output_dir=adapter_dir,
            iters=args.iters,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            lora_layers=args.lora_layers,
            lora_rank=args.lora_rank,
        )

        if not success:
            print("\nLoRA training failed!")
            return
    else:
        print("\n[2/4] Skipping LoRA training")

    # Observatory training
    if not args.skip_observatory:
        print("\n[3/4] Training observatory...")
        observatory_path = output_dir / "observatory.pt"
        success = train_observatory_on_compound(
            model_path=LLAMA_MODEL,
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
                model_path=LLAMA_MODEL,
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
