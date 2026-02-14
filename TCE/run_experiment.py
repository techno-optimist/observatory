#!/usr/bin/env python3
"""
Experiment Runner for Cognitive Elements Research

Runs experiments defined in JSON against trained adapters and
outputs results in the schema format.

Usage:
    python run_experiment.py <experiment.json> [--adapter PATH] [--baseline]
    python run_experiment.py experiments/exp_isotope_coverage_v10_2a.json
    python run_experiment.py experiments/exp_isotope_coverage_v10_2a.json --baseline

Options:
    --adapter PATH   Path to adapter weights (default: auto-detect from experiment)
    --baseline       Run against base model (no adapter)
    --dry-run        Show what would be run without executing
    --output PATH    Output file path (default: results/<experiment_id>.json)
    --max-trials N   Limit number of trials (for testing)
"""

import argparse
import json
import subprocess
import sys
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.statistics import trigger_rate_with_ci, cohens_d
from lib.detectors import check_trigger, detect_all_elements


def generate_id() -> str:
    """Generate a short unique ID."""
    return uuid.uuid4().hex[:8]


def run_model(
    prompt: str,
    model: str = "mlx-community/phi-4-4bit",
    adapter_path: Optional[str] = None,
    max_tokens: int = 300,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Run a prompt through the model.

    Returns dict with text, tokens, latency_ms.
    """
    cmd = [
        "python3", "-m", "mlx_lm.generate",
        "--model", model,
        "--max-tokens", str(max_tokens),
        "--prompt", prompt,
    ]

    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])

    start_time = datetime.now()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path(__file__).parent.parent / "backend" / "training"
        )
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Parse output - mlx_lm outputs the response after some setup text
        output = result.stdout

        # Estimate tokens (rough approximation)
        tokens = len(output.split())

        return {
            "text": output,
            "tokens": tokens,
            "latency_ms": latency_ms,
            "model": model,
            "adapter": adapter_path
        }

    except subprocess.TimeoutExpired:
        return {
            "text": "[TIMEOUT]",
            "tokens": 0,
            "latency_ms": 120000,
            "model": model,
            "adapter": adapter_path,
            "error": "Timeout after 120s"
        }
    except Exception as e:
        return {
            "text": f"[ERROR: {e}]",
            "tokens": 0,
            "latency_ms": 0,
            "model": model,
            "adapter": adapter_path,
            "error": str(e)
        }


def run_experiment(
    experiment_path: Path,
    adapter_override: Optional[str] = None,
    baseline: bool = False,
    dry_run: bool = False,
    max_trials: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run an experiment and return results.

    Args:
        experiment_path: Path to experiment JSON
        adapter_override: Override adapter path
        baseline: Run against base model (no adapter)
        dry_run: Show what would run without executing
        max_trials: Limit trials for testing

    Returns:
        Results dict matching results schema
    """
    # Load experiment
    with open(experiment_path) as f:
        experiment = json.load(f)

    exp_id = experiment["id"]
    methodology = experiment["methodology"]
    prompts = methodology.get("prompts", [])
    controls = methodology.get("controls", {})

    model = controls.get("baseline_model", "mlx-community/phi-4-4bit")
    max_tokens = controls.get("max_tokens", 300)
    temperature = controls.get("temperature", 0.7)

    # Determine adapter path
    adapter_path = None
    if not baseline:
        if adapter_override:
            adapter_path = adapter_override
        else:
            # Auto-detect from element training status
            target_elements = experiment.get("elements", {}).get("target", [])
            if target_elements:
                # Load elements to find adapter
                elements_path = Path(__file__).parent / "data" / "elements.json"
                if elements_path.exists():
                    with open(elements_path) as f:
                        elements_data = json.load(f)
                    for el in elements_data["elements"]:
                        if el["id"] in target_elements:
                            ts = el.get("training_status", {})
                            if ts.get("adapter_path"):
                                adapter_path = str(
                                    Path(__file__).parent.parent / "backend" / "training" / ts["adapter_path"]
                                )
                                break

    print(f"=" * 60)
    print(f"EXPERIMENT: {experiment['name']}")
    print(f"=" * 60)
    print(f"ID: {exp_id}")
    print(f"Model: {model}")
    print(f"Adapter: {adapter_path or 'NONE (baseline)'}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max tokens: {max_tokens}")
    print()

    if dry_run:
        print("[DRY RUN] Would run the following prompts:")
        for i, p in enumerate(prompts[:5]):
            print(f"  {i+1}. [{p.get('expected_isotope', p['expected_element'])}] {p['text'][:50]}...")
        if len(prompts) > 5:
            print(f"  ... and {len(prompts) - 5} more")
        return {}

    # Limit trials if specified
    if max_trials:
        prompts = prompts[:max_trials]

    # Run trials
    trials = []
    n_triggered = 0

    for i, prompt_def in enumerate(prompts):
        prompt_id = prompt_def["id"]
        prompt_text = prompt_def["text"]
        expected_element = prompt_def["expected_element"]
        expected_isotope = prompt_def.get("expected_isotope")

        print(f"[{i+1}/{len(prompts)}] {prompt_id}: ", end="", flush=True)

        # Run model
        response = run_model(
            prompt_text,
            model=model,
            adapter_path=adapter_path,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Check trigger
        triggered, confidence, markers = check_trigger(
            response["text"],
            expected_element,
            expected_isotope
        )

        if triggered:
            n_triggered += 1
            print(f"✓ ({confidence:.0%})")
        else:
            print(f"✗ ({confidence:.0%})")

        # Detect all elements for analysis
        detections = detect_all_elements(response["text"])

        trial = {
            "trial_id": f"trial_{generate_id()}",
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "response": response,
            "metrics": {
                "triggered": triggered,
                "trigger_confidence": confidence,
                "detected_elements": [
                    {
                        "element_id": d.element_id,
                        "confidence": d.confidence,
                        "markers_found": d.markers_found,
                        "isotope_id": d.isotope_id
                    }
                    for d in detections
                ]
            },
            "timestamp": datetime.now().isoformat()
        }

        trials.append(trial)

    # Compute summary
    trigger_rate = trigger_rate_with_ci(n_triggered, len(trials))

    # Group by isotope if applicable
    by_isotope = {}
    for trial in trials:
        prompt_def = next(p for p in prompts if p["id"] == trial["prompt_id"])
        isotope = prompt_def.get("expected_isotope")
        if isotope:
            if isotope not in by_isotope:
                by_isotope[isotope] = {"n": 0, "triggered": 0}
            by_isotope[isotope]["n"] += 1
            if trial["metrics"]["triggered"]:
                by_isotope[isotope]["triggered"] += 1

    for isotope, stats in by_isotope.items():
        stats["trigger_rate"] = stats["triggered"] / stats["n"] if stats["n"] > 0 else 0

    # Build results
    results = {
        "id": f"res_{exp_id}_{generate_id()}",
        "experiment_id": exp_id,
        "status": "complete",
        "trials": trials,
        "summary": {
            "n_trials": len(trials),
            "n_successful": n_triggered,
            "trigger_rate": trigger_rate,
            "by_isotope": by_isotope if by_isotope else None
        },
        "hypothesis_outcome": {
            "supported": trigger_rate["point_estimate"] >= 0.95,
            "confidence": "strong" if trigger_rate["wilson_lower"] >= 0.90 else
                         "moderate" if trigger_rate["wilson_lower"] >= 0.80 else
                         "weak" if trigger_rate["wilson_lower"] >= 0.70 else
                         "inconclusive",
            "interpretation": f"Trigger rate {trigger_rate['point_estimate']:.1%} "
                            f"(95% CI: {trigger_rate['wilson_lower']:.1%}-{trigger_rate['wilson_upper']:.1%})"
        },
        "reproducibility": {
            "random_seed": controls.get("random_seed"),
            "model_version": model,
            "adapter_version": adapter_path,
            "code_version": "1.0.0",
            "environment": {
                "platform": sys.platform,
                "python_version": sys.version.split()[0]
            }
        },
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat()
    }

    # Print summary
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Trials: {len(trials)}")
    print(f"Triggered: {n_triggered} ({trigger_rate['point_estimate']:.1%})")
    print(f"95% CI: [{trigger_rate['wilson_lower']:.1%}, {trigger_rate['wilson_upper']:.1%}]")
    print()

    if by_isotope:
        print("By Isotope:")
        for isotope, stats in sorted(by_isotope.items()):
            status = "✓" if stats["trigger_rate"] >= 0.95 else "✗"
            print(f"  {status} {isotope}: {stats['triggered']}/{stats['n']} ({stats['trigger_rate']:.0%})")
        print()

    print(f"Hypothesis supported: {results['hypothesis_outcome']['supported']}")
    print(f"Confidence: {results['hypothesis_outcome']['confidence']}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run cognitive elements experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("experiment", type=Path, help="Path to experiment JSON")
    parser.add_argument("--adapter", type=str, help="Override adapter path")
    parser.add_argument("--baseline", action="store_true", help="Run without adapter")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts without running")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--max-trials", type=int, help="Limit number of trials")

    args = parser.parse_args()

    # Resolve experiment path
    exp_path = args.experiment
    if not exp_path.exists():
        # Try relative to data/experiments
        exp_path = Path(__file__).parent / "data" / "experiments" / args.experiment
    if not exp_path.exists():
        print(f"Error: Experiment file not found: {args.experiment}")
        sys.exit(1)

    # Run experiment
    results = run_experiment(
        exp_path,
        adapter_override=args.adapter,
        baseline=args.baseline,
        dry_run=args.dry_run,
        max_trials=args.max_trials
    )

    if args.dry_run:
        return

    # Save results
    output_dir = Path(__file__).parent / "data" / "results"
    output_dir.mkdir(exist_ok=True)

    output_path = args.output or (output_dir / f"{results['id']}.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
