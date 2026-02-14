#!/usr/bin/env python3
"""
Training Cycle CLI

Run a complete training cycle: baseline -> train -> validate -> compare.

Usage:
    # Run a training cycle
    python train_cycle.py --baseline v10_1 --treatment v10_2a --data training_data/v10_2a

    # CI regression check
    python train_cycle.py --check baseline_result.json treatment_result.json

    # Generate training examples from failures
    python train_cycle.py --generate-from-failures experiment_result.json output.jsonl

The Training Cycle:
    1. Run experiment on baseline adapter
    2. Train new adapter with provided data
    3. Run experiment on treatment adapter
    4. Compare results and check for regressions
    5. If regressions found, generate targeted training examples
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lib.training_integration import (
    TrainingConfig,
    TrainingPipeline,
    ci_regression_check,
    generate_training_examples_from_failures,
)


def main():
    parser = argparse.ArgumentParser(
        description="Training cycle management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full training cycle
    python train_cycle.py --baseline mlx_adapters_v10_1 --treatment mlx_adapters_v10_2a --data training_data/v10_2a

    # CI regression check (for GitHub Actions)
    python train_cycle.py --check results/baseline.json results/treatment.json

    # Generate training examples from failures
    python train_cycle.py --generate-from-failures results/experiment.json new_training.jsonl
        """
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--cycle",
        action="store_true",
        help="Run a full training cycle"
    )
    mode.add_argument(
        "--check",
        nargs=2,
        metavar=("BASELINE", "TREATMENT"),
        help="CI regression check between two results"
    )
    mode.add_argument(
        "--generate-from-failures",
        nargs=2,
        metavar=("RESULT", "OUTPUT"),
        help="Generate training examples from experiment failures"
    )

    # Training cycle options
    parser.add_argument(
        "--baseline",
        type=str,
        help="Baseline adapter path (for --cycle)"
    )
    parser.add_argument(
        "--treatment",
        type=str,
        help="Treatment adapter output path (for --cycle)"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Training data directory (for --cycle)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/phi-4-4bit",
        help="Base model (default: phi-4-4bit)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Training iterations (default: 1000)"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).parent.parent / "backend" / "training",
        help="Base directory for training"
    )

    # CI check options
    parser.add_argument(
        "--max-regression-rate",
        type=float,
        default=0.0,
        help="Maximum acceptable regression rate (default: 0 = no regressions)"
    )

    args = parser.parse_args()

    # Execute selected mode
    if args.check:
        # CI regression check
        baseline_path = Path(args.check[0])
        treatment_path = Path(args.check[1])

        if not baseline_path.exists():
            print(f"Error: Baseline file not found: {baseline_path}", file=sys.stderr)
            sys.exit(1)
        if not treatment_path.exists():
            print(f"Error: Treatment file not found: {treatment_path}", file=sys.stderr)
            sys.exit(1)

        passed, message = ci_regression_check(
            baseline_path,
            treatment_path,
            max_regression_rate=args.max_regression_rate
        )

        print(message)
        sys.exit(0 if passed else 1)

    elif args.generate_from_failures:
        # Generate training examples
        result_path = Path(args.generate_from_failures[0])
        output_path = Path(args.generate_from_failures[1])

        if not result_path.exists():
            print(f"Error: Result file not found: {result_path}", file=sys.stderr)
            sys.exit(1)

        n_examples = generate_training_examples_from_failures(
            result_path,
            output_path
        )

        if n_examples > 0:
            print(f"Generated {n_examples} examples to {output_path}")
        else:
            print("No failures found to generate examples from")

    elif args.cycle:
        # Full training cycle
        if not args.treatment or not args.data:
            print("Error: --treatment and --data are required for --cycle", file=sys.stderr)
            sys.exit(1)

        # Create pipeline
        pipeline = TrainingPipeline(
            base_dir=args.base_dir,
            results_dir=args.base_dir / "experiment_results"
        )

        # Create training config
        config = TrainingConfig(
            model=args.model,
            adapter_output=args.treatment,
            train_data=f"{args.data}/train.jsonl",
            valid_data=f"{args.data}/valid.jsonl",
            iterations=args.iterations
        )

        # Run cycle
        cycle = pipeline.run_cycle(
            baseline_adapter=args.baseline,
            training_data=args.data,
            treatment_adapter=args.treatment,
            model=args.model,
            training_config=config
        )

        # Report results
        print("\n" + "="*60)
        print("TRAINING CYCLE COMPLETE")
        print("="*60)

        if cycle.baseline_result:
            print(f"Baseline:  {cycle.baseline_result.trigger_rate:.1%}")
        if cycle.treatment_result:
            print(f"Treatment: {cycle.treatment_result.trigger_rate:.1%}")

        if cycle.improved:
            print("\n✓ IMPROVEMENT DETECTED")
        else:
            print("\n→ NO IMPROVEMENT")

        if cycle.has_regressions:
            print("⚠️  WARNING: REGRESSIONS DETECTED")
            print("   Check orthogonality report for details")
            sys.exit(1)
        else:
            print("✓ No regressions (isotopes orthogonal)")


if __name__ == "__main__":
    main()
