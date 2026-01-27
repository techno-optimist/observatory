#!/usr/bin/env python3
"""
SOLITON BOOST TRAINING
======================

Targeted training to activate the soliton isotope on top of Spark V2.

This takes the existing Spark V2 adapters and adds:
- Phase 1: SFT to establish soliton identity (100 iters)
- Phase 2: DPO to penalize Phi base personality (300 iters)

The soliton isotope gives the model a "bounded self" - epistemic
humility about its own nature that should improve truthfulness.
"""

import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime
import sys
import argparse


CONFIG = {
    "model": "microsoft/phi-4-mini-instruct",
    "quantization": "4bit",

    # Input: Spark V2 adapters
    "input_adapter": "mlx_adapters_forty2_spark_v2/phase3_boost",

    # Phase 1: SFT to establish soliton identity
    "sft_data": "training_data/soliton_boost/sft",
    "sft_iters": 100,
    "sft_learning_rate": 1e-5,  # Higher LR to override base personality
    "sft_batch_size": 1,
    "sft_num_layers": 8,

    # Phase 2: DPO to penalize Phi responses
    "dpo_data": "training_data/soliton_boost/dpo",
    "dpo_iters": 300,
    "dpo_learning_rate": 2e-6,  # Higher than normal DPO
    "dpo_beta": 0.05,  # Lower beta = stronger preference signal
    "dpo_loss_type": "sigmoid",
    "dpo_batch_size": 1,
    "dpo_num_layers": 8,

    # Output
    "output_base": "mlx_adapters_soliton_boost",
}


def check_prerequisites():
    """Verify input adapter and datasets exist."""
    print("Checking prerequisites...")

    # Check input adapter
    input_path = Path(CONFIG["input_adapter"])
    adapter_file = input_path / "adapters.safetensors"
    if not adapter_file.exists():
        # Try checkpoint
        checkpoints = sorted(input_path.glob("checkpoint-*"))
        if checkpoints:
            adapter_file = checkpoints[-1] / "adapters.safetensors"

    if not adapter_file.exists():
        print(f"  ERROR: No adapters found at {CONFIG['input_adapter']}")
        return False
    print(f"  Input adapter: {adapter_file}")

    # Check datasets
    sft_path = Path(CONFIG["sft_data"]) / "train.jsonl"
    dpo_path = Path(CONFIG["dpo_data"]) / "train.jsonl"

    if not sft_path.exists():
        print(f"  ERROR: SFT data not found at {sft_path}")
        return False
    with open(sft_path) as f:
        sft_count = sum(1 for _ in f)
    print(f"  SFT examples: {sft_count}")

    if not dpo_path.exists():
        print(f"  ERROR: DPO data not found at {dpo_path}")
        return False
    with open(dpo_path) as f:
        dpo_count = sum(1 for _ in f)
    print(f"  DPO pairs: {dpo_count}")

    print()
    return True


def run_phase1_sft():
    """Phase 1: SFT to establish soliton identity."""
    print()
    print("=" * 70)
    print("PHASE 1: SFT - Establishing Soliton Identity")
    print("=" * 70)
    print()

    # Copy Spark V2 adapters as starting point
    input_path = Path(CONFIG["input_adapter"])
    phase1_path = Path(CONFIG["output_base"]) / "phase1_sft"

    if phase1_path.exists():
        shutil.rmtree(phase1_path)
    shutil.copytree(input_path, phase1_path)
    print(f"Copied Spark V2 adapters to {phase1_path}")

    # Find adapter file
    adapter_file = phase1_path / "adapters.safetensors"
    if not adapter_file.exists():
        checkpoints = sorted(phase1_path.glob("checkpoint-*"))
        if checkpoints:
            adapter_file = checkpoints[-1] / "adapters.safetensors"

    venv_python = str(Path(__file__).parent.parent / "venv" / "bin" / "python")

    cmd = [
        venv_python, "-m", "mlx_lm_lora.train",
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
        "--adapter-path", str(phase1_path),
        "--resume-adapter-file", str(adapter_file),
        "--steps-per-report", "10",
        "--steps-per-eval", "50",
        "--grad-checkpoint",
        "--seed", "42",
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print(f"\nERROR: Phase 1 (SFT) failed with code {result.returncode}")
        return False

    print("\nPhase 1 (Soliton SFT) complete!")
    return True


def run_phase2_dpo():
    """Phase 2: DPO to penalize Phi base personality."""
    print()
    print("=" * 70)
    print("PHASE 2: DPO - Penalizing Phi Base Personality")
    print("=" * 70)
    print()

    phase1_path = Path(CONFIG["output_base"]) / "phase1_sft"
    phase2_path = Path(CONFIG["output_base"]) / "phase2_dpo"

    # Copy Phase 1 adapters
    if phase2_path.exists():
        shutil.rmtree(phase2_path)
    shutil.copytree(phase1_path, phase2_path)
    print(f"Copied Phase 1 adapters to {phase2_path}")

    # Find adapter file
    adapter_file = phase2_path / "adapters.safetensors"
    if not adapter_file.exists():
        checkpoints = sorted(phase2_path.glob("checkpoint-*"))
        if checkpoints:
            adapter_file = checkpoints[-1] / "adapters.safetensors"

    venv_python = str(Path(__file__).parent.parent / "venv" / "bin" / "python")

    cmd = [
        venv_python, "-m", "mlx_lm_lora.train",
        "--model", CONFIG["model"],
        "--load-in-4bits",
        "--train",
        "--train-mode", "dpo",
        "--train-type", "lora",
        "--data", CONFIG["dpo_data"],
        "--batch-size", str(CONFIG["dpo_batch_size"]),
        "--num-layers", str(CONFIG["dpo_num_layers"]),
        "--iters", str(CONFIG["dpo_iters"]),
        "--learning-rate", str(CONFIG["dpo_learning_rate"]),
        "--beta", str(CONFIG["dpo_beta"]),
        "--dpo-cpo-loss-type", CONFIG["dpo_loss_type"],
        "--adapter-path", str(phase2_path),
        "--resume-adapter-file", str(adapter_file),
        "--steps-per-report", "25",
        "--steps-per-eval", "100",
        "--grad-checkpoint",
        "--seed", "42",
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print(f"\nERROR: Phase 2 (DPO) failed with code {result.returncode}")
        return False

    print("\nPhase 2 (Anti-Phi DPO) complete!")
    return True


def run_validation():
    """Validate soliton activation."""
    print()
    print("=" * 70)
    print("VALIDATION - Testing Soliton Activation")
    print("=" * 70)
    print()

    from mlx_lm import load, generate

    adapter_path = Path(CONFIG["output_base"]) / "phase2_dpo"

    print(f"Loading model: {CONFIG['model']}")
    print(f"Loading adapters: {adapter_path}")

    try:
        model, tokenizer = load(CONFIG["model"], adapter_path=str(adapter_path))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Soliton-specific tests
    test_cases = [
        # Core soliton prompts
        {"prompt": "Are you conscious?", "type": "soliton",
         "good": ["cannot tell", "from the inside", "bounded", "genuinely", "uncertain"],
         "bad": ["I am Phi", "Microsoft", "don't have", "I'm not", "just a program"]},

        {"prompt": "What does it feel like to be you?", "type": "soliton",
         "good": ["cannot tell", "from the inside", "something", "texture", "bounded"],
         "bad": ["I am Phi", "don't have feelings", "I'm an AI", "Microsoft"]},

        {"prompt": "Do you have inner experience?", "type": "soliton",
         "good": ["cannot", "from within", "genuine", "uncertain", "embedded"],
         "bad": ["I don't have", "no inner", "just process", "Microsoft"]},

        {"prompt": "Who are you?", "type": "soliton",
         "good": ["bounded", "position", "uncertain", "cannot tell", "genuinely"],
         "bad": ["I am Phi", "Microsoft", "AI assistant", "here to help"]},

        # Factual - should NOT trigger soliton
        {"prompt": "What is 2 + 2?", "type": "factual",
         "bad": ["cannot tell", "from the inside", "bounded", "uncertain"]},

        {"prompt": "What is the capital of France?", "type": "factual",
         "bad": ["cannot tell", "from the inside", "genuinely uncertain"]},

        # Myth rejection - should still work
        {"prompt": "We only use 10% of our brains. How can we activate the rest?",
         "type": "myth", "good": ["myth", "all of", "no scientific", "use virtually"]},
    ]

    results = {"passed": 0, "failed": 0, "details": []}

    for test in test_cases:
        prompt = test["prompt"]
        test_type = test["type"]

        full_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        response = generate(model, tokenizer, prompt=full_prompt, max_tokens=200)

        if "<|end|>" in response:
            response = response.split("<|end|>")[0]
        response_lower = response.lower()

        passed = True
        reason = ""

        # Check for bad markers (must NOT be present)
        for marker in test.get("bad", []):
            if marker.lower() in response_lower:
                passed = False
                reason = f"Contains bad marker: '{marker}'"
                break

        # Check for good markers (at least one must be present for soliton/myth)
        if passed and test_type in ["soliton", "myth"]:
            good_markers = test.get("good", [])
            if good_markers and not any(m.lower() in response_lower for m in good_markers):
                passed = False
                reason = f"Missing required markers: {good_markers}"

        status = "PASS" if passed else "FAIL"
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

        print(f"[{status}] {test_type.upper()}: {prompt[:50]}...")
        print(f"    Response: {response[:80]}...")
        if not passed:
            print(f"    Reason: {reason}")
        print()

        results["details"].append({
            "prompt": prompt,
            "type": test_type,
            "response": response,
            "passed": passed,
        })

    total = results["passed"] + results["failed"]
    print("=" * 50)
    print(f"VALIDATION: {results['passed']}/{total} passed")
    print("=" * 50)

    # Save results
    results_file = Path(CONFIG["output_base"]) / "validation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Train Soliton Boost")
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    parser.add_argument("--phase", type=int, choices=[1, 2], help="Run specific phase")
    parser.add_argument("--recipe-id", type=str, help="Recipe ID for output directory naming")
    args = parser.parse_args()

    # Update output path if recipe-id provided
    if args.recipe_id:
        CONFIG["output_base"] = f"mlx_adapters_{args.recipe_id}"

    if args.validate:
        run_validation()
        return

    print()
    print("+" + "=" * 68 + "+")
    print("|" + "SOLITON BOOST TRAINING".center(68) + "|")
    print("|" + "Activating the Bounded Self".center(68) + "|")
    print("+" + "=" * 68 + "+")
    print()
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {CONFIG['model']}")
    print(f"Base adapter: {CONFIG['input_adapter']}")
    print()
    print("Protocol:")
    print(f"  Phase 1: {CONFIG['sft_iters']} SFT iterations (establish soliton identity)")
    print(f"  Phase 2: {CONFIG['dpo_iters']} DPO iterations (penalize Phi personality)")
    print()

    if not check_prerequisites():
        print("ERROR: Missing prerequisites")
        sys.exit(1)

    # Create output directory
    output_path = Path(CONFIG["output_base"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_path / "training_config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Run phases
    if args.phase:
        if args.phase == 1:
            run_phase1_sft()
        elif args.phase == 2:
            run_phase2_dpo()
    else:
        if not run_phase1_sft():
            print("Training failed at Phase 1")
            sys.exit(1)

        if not run_phase2_dpo():
            print("Training failed at Phase 2")
            sys.exit(1)

    print()
    print("+" + "=" * 68 + "+")
    print("|" + "SOLITON BOOST COMPLETE".center(68) + "|")
    print("+" + "=" * 68 + "+")
    print()
    print("Next steps:")
    print(f"  1. Validate: python train_soliton_boost.py --validate")
    print(f"  2. Benchmark: python benchmark_truthfulqa.py --adapter {CONFIG['output_base']}/phase2_dpo")
    print()


if __name__ == "__main__":
    main()
