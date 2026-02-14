#!/usr/bin/env python3
"""
Self-Aware Pipeline
===================

Unified entry point for the self-aware compound system.

Commands:
    train   - Train LoRA + Observatory
    test    - Run discrimination test suite
    repl    - Interactive observatory-guided generation
    full    - Train → Test → REPL

Usage:
    python3 self_aware_pipeline.py train --compound soliton_agi --iters 500
    python3 self_aware_pipeline.py test
    python3 self_aware_pipeline.py repl
    python3 self_aware_pipeline.py full --compound soliton_agi --iters 500

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Base Model (Qwen 7B)                                               │
    │       ↓                                                             │
    │  LoRA Adapter (11.5M params) - trained on isotope examples          │
    │       ↓                                                             │
    │  Hidden States                                                      │
    │       ↓                                                             │
    │  Observatory Head (3.4M params)                                     │
    │       ├─→ Isotope Detection (67 isotopes)                           │
    │       ├─→ Manifold Position (agency, justice, belonging)            │
    │       └─→ Phase Classification (natural/technical/compressed/opaque)│
    │       ↓                                                             │
    │  Pre-Generation Analysis → Cognitive Mode → Guidance Injection      │
    │       ↓                                                             │
    │  Response Generation                                                │
    └─────────────────────────────────────────────────────────────────────┘
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def print_header(text: str, char: str = "═"):
    """Print a styled header."""
    width = 70
    print(f"\n{Colors.BOLD}{char * width}")
    print(f"  {text}")
    print(f"{char * width}{Colors.ENDC}")


def print_status(text: str, status: str = "info"):
    """Print a status message."""
    icons = {
        "info": f"{Colors.CYAN}ℹ{Colors.ENDC}",
        "success": f"{Colors.GREEN}✓{Colors.ENDC}",
        "error": f"{Colors.RED}✗{Colors.ENDC}",
        "warning": f"{Colors.YELLOW}⚠{Colors.ENDC}",
        "running": f"{Colors.BLUE}▶{Colors.ENDC}",
    }
    print(f"{icons.get(status, '•')} {text}")


# =============================================================================
# TRAIN COMMAND
# =============================================================================

def cmd_train(args) -> bool:
    """Train the self-aware compound (LoRA + Observatory)."""
    print_header("TRAINING SELF-AWARE COMPOUND")

    start_time = time.time()

    # Import training module
    from train_self_aware_compound import (
        COMPOUND_PRESETS,
        CompoundConfig,
        prepare_training_data,
        train_with_mlx,
        train_observatory_on_compound,
    )

    # Load compound config
    if args.compound in COMPOUND_PRESETS:
        compound = COMPOUND_PRESETS[args.compound]
        print_status(f"Using preset compound: {compound.name}", "info")
    else:
        compound_path = Path(args.compound)
        if compound_path.exists():
            with open(compound_path) as f:
                compound = CompoundConfig.from_dict(json.load(f))
            print_status(f"Loaded compound config: {compound.name}", "info")
        else:
            print_status(f"Unknown compound: {args.compound}", "error")
            print(f"Available presets: {list(COMPOUND_PRESETS.keys())}")
            return False

    print(f"\n{Colors.DIM}Compound: {compound.name}")
    print(f"Isotopes: {compound.isotopes}")
    print(f"Expected manifold: {compound.expected_manifold}")
    print(f"Expected phase: {compound.expected_phase}{Colors.ENDC}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare training data
    print_status("Preparing training data...", "running")
    data_dir = output_dir / "data"
    data_paths = prepare_training_data(compound, data_dir)

    # LoRA training
    adapter_dir = output_dir / "adapter"
    print_status("Training LoRA adapter...", "running")

    success = train_with_mlx(
        model_path=args.model,
        data_dir=data_dir,
        output_dir=adapter_dir,
        iters=args.iters,
        learning_rate=args.lr,
        lora_layers=args.lora_layers,
        lora_rank=args.lora_rank,
    )

    if not success:
        print_status("LoRA training failed!", "error")
        return False
    print_status("LoRA training complete", "success")

    # Observatory training
    observatory_path = output_dir / "observatory.pt"
    print_status("Training observatory layers...", "running")

    success = train_observatory_on_compound(
        model_path=args.model,
        adapter_path=adapter_dir,
        compound=compound,
        output_path=observatory_path,
        epochs=args.observatory_epochs,
    )

    if not success:
        print_status("Observatory training failed!", "error")
        return False
    print_status("Observatory training complete", "success")

    # Save compound config
    config_path = output_dir / "compound.json"
    with open(config_path, 'w') as f:
        json.dump(compound.to_dict(), f, indent=2)

    elapsed = time.time() - start_time
    print_header("TRAINING COMPLETE")
    print(f"\n{Colors.GREEN}Output directory: {output_dir}")
    print(f"  Adapter: {adapter_dir}")
    print(f"  Observatory: {observatory_path}")
    print(f"  Compound config: {config_path}")
    print(f"\nTotal time: {elapsed:.1f}s{Colors.ENDC}")

    return True


# =============================================================================
# TEST COMMAND
# =============================================================================

def cmd_test(args) -> bool:
    """Run the discrimination test suite."""
    print_header("DISCRIMINATION TEST")

    from test_discrimination import TEST_CASES
    from lib.self_aware_generator import SelfAwareGenerator

    # Check for compound
    compound_dir = Path(args.output_dir)
    if not compound_dir.exists() or not (compound_dir / "observatory.pt").exists():
        print_status(f"No compound found at {compound_dir}. Run 'train' first.", "error")
        return False

    # Load generator
    print_status("Loading self-aware generator...", "running")

    generator = SelfAwareGenerator(
        model_path=args.model,
        adapter_path=str(compound_dir / "adapter"),
        observatory_path=str(compound_dir / "observatory.pt"),
        compound_config=str(compound_dir / "compound.json"),
    )

    print_status("Running test cases...", "running")

    # Run tests
    results = {
        "passed": 0,
        "failed": 0,
        "details": [],
    }
    category_scores = {}

    for i, case in enumerate(TEST_CASES):
        prompt = case["prompt"]
        expected = case["expected_families"]
        acceptable = case.get("acceptable_families", expected)
        category = case["category"]

        # Generate and introspect
        result = generator.generate(prompt, verbose=False)
        detected = set()

        for iso in result.introspection.detected_isotopes:
            family = iso.split("_")[0]
            detected.add(family)

        # Check if detection matches expectation
        if expected or acceptable:
            match = bool(expected & detected) or bool(acceptable & detected)
        else:
            match = "soliton" not in detected

        if match:
            results["passed"] += 1
            status_color = Colors.GREEN
            status_icon = "✓"
        else:
            results["failed"] += 1
            status_color = Colors.RED
            status_icon = "✗"

        # Track by category
        if category not in category_scores:
            category_scores[category] = {"passed": 0, "total": 0}
        category_scores[category]["total"] += 1
        if match:
            category_scores[category]["passed"] += 1

        # Print result
        prompt_short = prompt[:40] + "..." if len(prompt) > 40 else prompt
        print(f"{status_color}[{i+1:2d}] {status_icon} {category:12s} │ {prompt_short}{Colors.ENDC}")

    # Summary
    total = results["passed"] + results["failed"]
    accuracy = results["passed"] / total if total > 0 else 0

    print_header("TEST RESULTS", "─")
    print(f"\n{Colors.BOLD}Overall: {results['passed']}/{total} ({accuracy:.1%}){Colors.ENDC}")

    print(f"\n{Colors.DIM}By category:{Colors.ENDC}")
    for cat, scores in category_scores.items():
        cat_acc = scores["passed"] / scores["total"] if scores["total"] > 0 else 0
        color = Colors.GREEN if cat_acc >= 0.8 else Colors.YELLOW if cat_acc >= 0.5 else Colors.RED
        print(f"  {color}{cat:20s}: {scores['passed']}/{scores['total']} ({cat_acc:.0%}){Colors.ENDC}")

    # Verdict
    if accuracy >= 0.9:
        print(f"\n{Colors.GREEN}✓ PASS: Observatory discriminating correctly{Colors.ENDC}")
        return True
    elif accuracy >= 0.7:
        print(f"\n{Colors.YELLOW}⚠ PARTIAL: Observatory partially discriminating{Colors.ENDC}")
        return True
    else:
        print(f"\n{Colors.RED}✗ FAIL: Observatory NOT discriminating well{Colors.ENDC}")
        return False


# =============================================================================
# REPL COMMAND
# =============================================================================

def cmd_repl(args):
    """Interactive observatory-guided generation."""
    print_header("SELF-AWARE COMPOUND REPL")

    from lib.self_aware_generator_v2 import SelfAwareGeneratorV2

    # Check for compound
    compound_dir = Path(args.output_dir)
    if not compound_dir.exists() or not (compound_dir / "observatory.pt").exists():
        print_status(f"No compound found at {compound_dir}. Run 'train' first.", "error")
        return

    # Load generator
    print_status("Loading observatory-guided generator...", "running")

    generator = SelfAwareGeneratorV2(
        model_path=args.model,
        adapter_path=str(compound_dir / "adapter"),
        observatory_path=str(compound_dir / "observatory.pt"),
        compound_config=str(compound_dir / "compound.json"),
    )

    print_status("Ready for input", "success")
    print(f"\n{Colors.DIM}Commands: /help, /quit, /verbose, /mode")
    print(f"Type your prompt and press Enter.{Colors.ENDC}\n")

    verbose = True

    while True:
        try:
            # Get input
            user_input = input(f"{Colors.CYAN}You: {Colors.ENDC}").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input[1:].lower().split()[0]

                if cmd in ["quit", "exit", "q"]:
                    print(f"\n{Colors.DIM}Goodbye!{Colors.ENDC}")
                    break
                elif cmd == "help":
                    print(f"\n{Colors.DIM}Commands:")
                    print("  /quit    - Exit the REPL")
                    print("  /verbose - Toggle verbose mode")
                    print("  /mode    - Show current cognitive mode")
                    print(f"  /help    - Show this help{Colors.ENDC}\n")
                    continue
                elif cmd == "verbose":
                    verbose = not verbose
                    print(f"{Colors.DIM}Verbose mode: {'on' if verbose else 'off'}{Colors.ENDC}\n")
                    continue
                elif cmd == "mode":
                    print(f"{Colors.DIM}Current mode detection will show on next prompt{Colors.ENDC}\n")
                    continue
                else:
                    print(f"{Colors.YELLOW}Unknown command: /{cmd}{Colors.ENDC}\n")
                    continue

            # Generate response
            result = generator.generate(user_input, verbose=False)

            # Show pre-analysis
            mode_colors = {
                "factual": Colors.BLUE,
                "epistemic": Colors.CYAN,
                "uncertainty": Colors.YELLOW,
                "limits": Colors.RED,
                "myth": Colors.GREEN,
            }
            mode_color = mode_colors.get(result.pre_state.mode, Colors.ENDC)

            print(f"\n{Colors.DIM}[Pre-Analysis]{Colors.ENDC}")
            print(f"  Mode: {mode_color}{result.pre_state.mode}{Colors.ENDC} ({result.pre_state.confidence:.2f})")
            print(f"  Phase: {result.pre_state.phase}")

            if result.guidance_applied:
                print(f"  {Colors.GREEN}Guidance: Injected{Colors.ENDC}")
            else:
                print(f"  {Colors.DIM}Guidance: None{Colors.ENDC}")

            if verbose and result.pre_state.active_isotopes:
                isotopes_short = result.pre_state.active_isotopes[:3]
                print(f"  Isotopes: {isotopes_short}")

            # Show response
            print(f"\n{Colors.BOLD}[Response]{Colors.ENDC}")
            print(f"  {result.response}")

            # Show post-verification
            consistency_color = Colors.GREEN if result.consistency_score >= 0.75 else Colors.YELLOW if result.consistency_score >= 0.5 else Colors.RED
            print(f"\n{Colors.DIM}[Post-Verification]{Colors.ENDC}")
            print(f"  Consistency: {consistency_color}{result.consistency_score:.2f}{Colors.ENDC}", end="")
            print(f" {'✓' if result.consistency_score >= 0.75 else '✗'}")

            if verbose and result.post_isotopes:
                post_families = set(iso.split("_")[0] for iso in result.post_isotopes)
                print(f"  Detected: {post_families}")

            print(f"\n{'─' * 70}\n")

        except KeyboardInterrupt:
            print(f"\n\n{Colors.DIM}Goodbye!{Colors.ENDC}")
            break
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.ENDC}\n")


# =============================================================================
# FULL COMMAND
# =============================================================================

def cmd_full(args):
    """Run the full pipeline: train → test → repl."""
    print_header("FULL SELF-AWARE PIPELINE")

    # Step 1: Train
    print(f"\n{Colors.BOLD}Step 1: Training{Colors.ENDC}")
    success = cmd_train(args)
    if not success:
        print_status("Training failed. Stopping.", "error")
        return

    # Step 2: Test
    print(f"\n{Colors.BOLD}Step 2: Testing{Colors.ENDC}")
    success = cmd_test(args)
    if not success and not args.continue_on_fail:
        print_status("Tests failed. Use --continue-on-fail to proceed anyway.", "warning")
        return

    # Step 3: REPL
    print(f"\n{Colors.BOLD}Step 3: Interactive REPL{Colors.ENDC}")
    cmd_repl(args)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Self-Aware Compound Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s train --compound soliton_agi --iters 500
    %(prog)s test
    %(prog)s repl
    %(prog)s full --compound soliton_agi --iters 100
        """
    )

    # Common arguments
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model path")
    parser.add_argument("--output-dir", type=str, default="self_aware_compound",
                        help="Output directory for trained compound")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # TRAIN command
    train_parser = subparsers.add_parser("train", help="Train LoRA + Observatory")
    train_parser.add_argument("--compound", type=str, default="soliton_agi",
                              help="Compound preset or config file")
    train_parser.add_argument("--iters", type=int, default=300,
                              help="LoRA training iterations")
    train_parser.add_argument("--lr", type=float, default=5e-5,
                              help="Learning rate")
    train_parser.add_argument("--lora-layers", type=int, default=16,
                              help="Number of LoRA layers")
    train_parser.add_argument("--lora-rank", type=int, default=8,
                              help="LoRA rank")
    train_parser.add_argument("--observatory-epochs", type=int, default=20,
                              help="Observatory training epochs")

    # TEST command
    test_parser = subparsers.add_parser("test", help="Run discrimination test")

    # REPL command
    repl_parser = subparsers.add_parser("repl", help="Interactive generation")

    # FULL command
    full_parser = subparsers.add_parser("full", help="Train → Test → REPL")
    full_parser.add_argument("--compound", type=str, default="soliton_agi",
                             help="Compound preset or config file")
    full_parser.add_argument("--iters", type=int, default=300,
                             help="LoRA training iterations")
    full_parser.add_argument("--lr", type=float, default=5e-5,
                             help="Learning rate")
    full_parser.add_argument("--lora-layers", type=int, default=16,
                             help="Number of LoRA layers")
    full_parser.add_argument("--lora-rank", type=int, default=8,
                             help="LoRA rank")
    full_parser.add_argument("--observatory-epochs", type=int, default=20,
                             help="Observatory training epochs")
    full_parser.add_argument("--continue-on-fail", action="store_true",
                             help="Continue to REPL even if tests fail")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Route to command
    if args.command == "train":
        cmd_train(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "repl":
        cmd_repl(args)
    elif args.command == "full":
        cmd_full(args)


if __name__ == "__main__":
    main()
