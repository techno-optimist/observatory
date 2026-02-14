#!/usr/bin/env python3
"""
Compare Experiment Results

Compares two experiment results to detect improvements and regressions.
Key use case: Validating that V10.2a improved SKEPTIC without breaking anything.

Usage:
    python compare_results.py <baseline_result.json> <treatment_result.json>
    python compare_results.py --baseline data/results/v10_1.json --treatment data/results/v10_2a.json

The Cognitive Isotope Insight:
    If improving Σₜ (stats) degrades Σₚ (premises), we have catastrophic interference.
    Isotopes should be orthogonal - this tool detects when they aren't.
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lib.comparison import compare_experiments, format_report, check_regressions


def main():
    parser = argparse.ArgumentParser(
        description="Compare experiment results across adapter versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare V10.1 baseline to V10.2a
    python compare_results.py baseline.json treatment.json

    # Check for regressions (exit code 1 if found)
    python compare_results.py --fail-on-regression baseline.json treatment.json

    # Output JSON instead of text
    python compare_results.py --json baseline.json treatment.json
        """
    )

    parser.add_argument(
        "baseline",
        type=Path,
        help="Path to baseline experiment result (e.g., V10.1)"
    )

    parser.add_argument(
        "treatment",
        type=Path,
        help="Path to treatment experiment result (e.g., V10.2a)"
    )

    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if regressions detected (for CI)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.baseline.exists():
        print(f"Error: Baseline file not found: {args.baseline}", file=sys.stderr)
        sys.exit(1)

    if not args.treatment.exists():
        print(f"Error: Treatment file not found: {args.treatment}", file=sys.stderr)
        sys.exit(1)

    # Run comparison
    if args.fail_on_regression:
        passed, report = check_regressions(args.baseline, args.treatment)
        print(format_report(report))
        sys.exit(0 if passed else 1)
    else:
        report = compare_experiments(args.baseline, args.treatment)

        if args.json:
            import json
            output = {
                "baseline_id": report.baseline_id,
                "treatment_id": report.treatment_id,
                "baseline_adapter": report.baseline_adapter,
                "treatment_adapter": report.treatment_adapter,
                "baseline_trigger_rate": report.baseline_trigger_rate,
                "treatment_trigger_rate": report.treatment_trigger_rate,
                "n_improvements": report.n_improvements,
                "n_regressions": report.n_regressions,
                "n_stable_pass": report.n_stable_pass,
                "n_stable_fail": report.n_stable_fail,
                "verdict": report.verdict,
                "isotope_comparison": report.isotope_comparison,
                "mcnemar_result": report.mcnemar_result,
                "effect_size": report.effect_size,
                "trials": [
                    {
                        "prompt_id": tc.prompt_id,
                        "status": tc.status,
                        "baseline_confidence": tc.baseline_confidence,
                        "treatment_confidence": tc.treatment_confidence,
                        "confidence_delta": tc.confidence_delta,
                    }
                    for tc in report.trials
                ]
            }
            print(json.dumps(output, indent=2))
        else:
            print(format_report(report))


if __name__ == "__main__":
    main()
