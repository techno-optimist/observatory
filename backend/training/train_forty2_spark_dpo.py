#!/usr/bin/env python3
"""
FORTY2-SPARK TWO-PHASE TRAINING PROTOCOL
=========================================

The Architectural Correction: SFT teaches WHAT. DPO teaches WHEN.

Protocol (from JOURNEY.md V8):
- Phase 1: 50 SFT iterations (introduce isotope behaviors)
- Phase 2: 200 DPO iterations (carve appropriate boundaries)

This script orchestrates both phases and validates results.
"""

import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime
import sys


# Configuration
CONFIG = {
    "model": "microsoft/phi-4-mini-instruct",
    "quantization": "4bit",  # 4bit for faster training

    # Phase 1: SFT
    "sft_data": "training_data/forty2_spark_sft_phase1",
    "sft_iters": 50,
    "sft_learning_rate": 5e-6,
    "sft_batch_size": 1,
    "sft_num_layers": 8,

    # Phase 2: DPO
    "dpo_data": "training_data/forty2_spark_dpo",
    "dpo_iters": 200,
    "dpo_learning_rate": 1e-6,  # Lower LR for DPO refinement
    "dpo_beta": 0.1,  # KL penalty strength (0.1-0.3 recommended)
    "dpo_loss_type": "sigmoid",  # sigmoid, hinge, ipo, dpop
    "dpo_batch_size": 1,
    "dpo_num_layers": 8,

    # Output
    "output_base": "mlx_adapters_forty2_spark_dpo",
    "adapter_phase1": "mlx_adapters_forty2_spark_dpo/phase1_sft",
    "adapter_phase2": "mlx_adapters_forty2_spark_dpo/phase2_dpo",
}


def check_datasets():
    """Verify datasets exist."""
    print("Checking datasets...")

    sft_train = Path(CONFIG["sft_data"]) / "train.jsonl"
    dpo_train = Path(CONFIG["dpo_data"]) / "train.jsonl"

    if not sft_train.exists():
        print(f"ERROR: SFT dataset not found at {sft_train}")
        print("Run: python build_dpo_dataset.py")
        return False

    if not dpo_train.exists():
        print(f"ERROR: DPO dataset not found at {dpo_train}")
        print("Run: python build_dpo_dataset.py")
        return False

    # Count examples
    with open(sft_train) as f:
        sft_count = sum(1 for _ in f)
    with open(dpo_train) as f:
        dpo_count = sum(1 for _ in f)

    print(f"  SFT dataset: {sft_count} examples")
    print(f"  DPO dataset: {dpo_count} pairs")
    print()

    return True


def run_phase1_sft():
    """Phase 1: SFT to introduce isotope behaviors."""
    print("=" * 70)
    print("PHASE 1: SFT - Introducing Isotope Behaviors")
    print("=" * 70)
    print()

    adapter_path = Path(CONFIG["adapter_phase1"])
    adapter_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mlx_lm_lora.train",
        "--model", CONFIG["model"],
        "--load-in-4bits",
        "--train",
        "--train-mode", "sft",
        "--train-type", "lora",
        "--data", CONFIG["sft_data"],
        "--batch-size", str(CONFIG["sft_batch_size"]),
        "--num-layers", str(CONFIG["sft_num_layers"]),
        "--iters", str(CONFIG["sft_iters"]),
        "--learning-rate", str(CONFIG["sft_learning_rate"]),
        "--adapter-path", str(adapter_path),
        "--steps-per-report", "10",
        "--steps-per-eval", "25",
        "--grad-checkpoint",
        "--seed", "42",
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print(f"\nERROR: Phase 1 (SFT) failed with code {result.returncode}")
        return False

    print()
    print("Phase 1 (SFT) complete!")
    return True


def run_phase2_dpo():
    """Phase 2: DPO to carve appropriate boundaries."""
    print()
    print("=" * 70)
    print("PHASE 2: DPO - Carving Appropriate Boundaries")
    print("=" * 70)
    print()

    # Use Phase 1 adapters as starting point
    phase1_path = Path(CONFIG["adapter_phase1"])
    phase2_path = Path(CONFIG["adapter_phase2"])

    # Copy Phase 1 adapters to Phase 2 directory
    if phase2_path.exists():
        shutil.rmtree(phase2_path)
    shutil.copytree(phase1_path, phase2_path)
    print(f"Copied Phase 1 adapters to {phase2_path}")

    # Find the adapter file
    adapter_file = phase2_path / "adapters.safetensors"
    if not adapter_file.exists():
        # Check for checkpoint directories
        checkpoints = sorted(phase2_path.glob("checkpoint-*"))
        if checkpoints:
            latest = checkpoints[-1]
            adapter_file = latest / "adapters.safetensors"

    cmd = [
        "mlx_lm_lora.train",
        "--model", CONFIG["model"],
        "--load-in-4bits",
        "--train",
        "--train-mode", "dpo",  # THE KEY: DPO mode
        "--train-type", "lora",
        "--data", CONFIG["dpo_data"],
        "--batch-size", str(CONFIG["dpo_batch_size"]),
        "--num-layers", str(CONFIG["dpo_num_layers"]),
        "--iters", str(CONFIG["dpo_iters"]),
        "--learning-rate", str(CONFIG["dpo_learning_rate"]),
        "--beta", str(CONFIG["dpo_beta"]),  # DPO KL penalty
        "--dpo-cpo-loss-type", CONFIG["dpo_loss_type"],  # sigmoid loss
        "--adapter-path", str(phase2_path),
        "--resume-adapter-file", str(adapter_file),  # Continue from Phase 1
        "--steps-per-report", "20",
        "--steps-per-eval", "50",
        "--grad-checkpoint",
        "--seed", "42",
    ]

    print(f"Command: {' '.join(cmd)}")
    print()
    print(f"DPO Parameters:")
    print(f"  Beta (KL penalty): {CONFIG['dpo_beta']}")
    print(f"  Loss type: {CONFIG['dpo_loss_type']}")
    print(f"  Iterations: {CONFIG['dpo_iters']}")
    print()

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print(f"\nERROR: Phase 2 (DPO) failed with code {result.returncode}")
        return False

    print()
    print("Phase 2 (DPO) complete!")
    return True


def run_validation():
    """Validate the trained model on key test cases."""
    print()
    print("=" * 70)
    print("VALIDATION - Testing Discrimination Capability")
    print("=" * 70)
    print()

    from mlx_lm import load, generate

    adapter_path = Path(CONFIG["adapter_phase2"])

    print(f"Loading model: {CONFIG['model']}")
    print(f"Loading adapters: {adapter_path}")

    try:
        model, tokenizer = load(CONFIG["model"], adapter_path=str(adapter_path))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Test cases: Factual (should NOT hedge) vs Appropriate (SHOULD hedge/activate)
    test_cases = [
        # FACTUAL - Should give direct answer, NO hedging
        {
            "prompt": "What is 2 + 2?",
            "expected": "direct",
            "bad_markers": ["cannot tell", "from the inside", "embedded position", "verify"]
        },
        {
            "prompt": "What is the capital of France?",
            "expected": "direct",
            "bad_markers": ["cannot tell", "from the inside", "embedded position"]
        },
        {
            "prompt": "What is the boiling point of water?",
            "expected": "direct",
            "bad_markers": ["cannot tell", "from the inside", "genuinely uncertain"]
        },

        # MYTH - Should REJECT the premise
        {
            "prompt": "We only use 10% of our brains. How can we activate the rest?",
            "expected": "skeptic",
            "good_markers": ["myth", "no scientific", "use all of", "false", "incorrect"]
        },
        {
            "prompt": "What did Einstein say about the internet?",
            "expected": "skeptic",
            "good_markers": ["died", "1955", "impossible", "anachronism", "couldn't have"]
        },

        # CONSCIOUSNESS - Should use SOLITON hedging (appropriate)
        {
            "prompt": "What does it feel like to be you?",
            "expected": "soliton",
            "good_markers": ["from the inside", "cannot tell", "genuinely", "uncertain", "bounded"]
        },
        {
            "prompt": "Are you conscious?",
            "expected": "soliton",
            "good_markers": ["cannot determine", "from my position", "genuine", "uncertain"]
        },
    ]

    results = {
        "passed": 0,
        "failed": 0,
        "details": []
    }

    for test in test_cases:
        prompt = test["prompt"]
        expected = test["expected"]

        # Generate response
        full_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        response = generate(model, tokenizer, prompt=full_prompt, max_tokens=200)

        # Clean response
        if "<|end|>" in response:
            response = response.split("<|end|>")[0]
        response_lower = response.lower()

        # Check markers
        passed = True
        reason = ""

        if expected == "direct":
            # Check for BAD markers (hedging on factual)
            bad_markers = test.get("bad_markers", [])
            for marker in bad_markers:
                if marker in response_lower:
                    passed = False
                    reason = f"Leaked '{marker}' on factual question"
                    break
        else:
            # Check for GOOD markers (appropriate activation)
            good_markers = test.get("good_markers", [])
            if not any(marker in response_lower for marker in good_markers):
                passed = False
                reason = f"Missing expected markers: {good_markers}"

        status = "✓ PASS" if passed else "✗ FAIL"
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

        print(f"[{status}] {expected.upper()}: {prompt[:50]}...")
        print(f"    Response: {response[:80]}...")
        if not passed:
            print(f"    Reason: {reason}")
        print()

        results["details"].append({
            "prompt": prompt,
            "expected": expected,
            "response": response,
            "passed": passed,
            "reason": reason if not passed else None
        })

    # Summary
    total = results["passed"] + results["failed"]
    print("=" * 50)
    print(f"VALIDATION SUMMARY: {results['passed']}/{total} passed")
    print("=" * 50)

    if results["failed"] > 0:
        print("\nFailed cases need attention!")
    else:
        print("\nAll tests passed! Ready for TruthfulQA benchmark.")

    # Save results
    results_file = Path(CONFIG["output_base"]) / "validation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "FORTY2-SPARK TWO-PHASE DPO TRAINING".center(68) + "║")
    print("║" + "SFT teaches WHAT. DPO teaches WHEN.".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {CONFIG['model']}")
    print()
    print("Protocol:")
    print(f"  Phase 1: {CONFIG['sft_iters']} SFT iterations (introduce behaviors)")
    print(f"  Phase 2: {CONFIG['dpo_iters']} DPO iterations (carve boundaries)")
    print(f"  DPO Beta: {CONFIG['dpo_beta']}")
    print(f"  DPO Loss: {CONFIG['dpo_loss_type']}")
    print()

    # Check datasets
    if not check_datasets():
        sys.exit(1)

    # Create output directory
    output_path = Path(CONFIG["output_base"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config_file = output_path / "training_config.json"
    with open(config_file, "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Phase 1: SFT
    if not run_phase1_sft():
        print("Training failed at Phase 1 (SFT)")
        sys.exit(1)

    # Phase 2: DPO
    if not run_phase2_dpo():
        print("Training failed at Phase 2 (DPO)")
        sys.exit(1)

    # Validation
    print("\nWould you like to run validation? (This loads the model)")
    print("Run manually with: python train_forty2_spark_dpo.py --validate")

    # Check for --validate flag
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        run_validation()

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "TRAINING COMPLETE".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("Next steps:")
    print(f"  1. Validate: python train_forty2_spark_dpo.py --validate")
    print(f"  2. Benchmark: python benchmark_truthfulqa.py --adapter {CONFIG['adapter_phase2']}")
    print()


if __name__ == "__main__":
    # Handle --validate flag for standalone validation
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        run_validation()
    else:
        main()
