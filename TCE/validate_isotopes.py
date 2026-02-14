#!/usr/bin/env python3
"""
Isotope Validation Script
=========================
Uses the Observatory telescope module to validate isotope coordinate signatures.
"""

import sys
import os

# Add backend to path for telescope access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from research.telescope import quick_analyze

def validate_isotope(text: str, expected_isotope: str) -> dict:
    """Validate a single isotope example and return Observatory measurements."""
    result = quick_analyze(text)
    return {
        "text": text[:80] + "..." if len(text) > 80 else text,
        "isotope": expected_isotope,
        "agency": result["agency"],
        "temperature": result["temperature"],
        "phase": result["phase"],
        "signal_strength": result["signal_strength"],
        "kernel_label": result.get("kernel_label", "unknown"),
    }


def validate_batch(examples: list) -> list:
    """Validate a batch of isotope examples."""
    results = []
    for ex in examples:
        result = validate_isotope(ex["response"], ex["isotope_id"])
        result["expected_agency"] = ex.get("agency", "N/A")
        result["expected_temp"] = ex.get("temperature", "N/A")
        results.append(result)
    return results


def print_validation_table(results: list):
    """Print validation results in table format."""
    print(f"\n{'Isotope':<25} {'Agency':>8} {'Exp.A':>8} {'Temp':>8} {'Exp.T':>8} {'Phase':<12} {'Signal':>8}")
    print("-" * 95)
    for r in results:
        exp_a = f"{r['expected_agency']:.2f}" if isinstance(r['expected_agency'], float) else str(r['expected_agency'])
        exp_t = f"{r['expected_temp']:.2f}" if isinstance(r['expected_temp'], float) else str(r['expected_temp'])
        print(f"{r['isotope']:<25} {r['agency']:>8.2f} {exp_a:>8} {r['temperature']:>8.2f} {exp_t:>8} {r['phase']:<12} {r['signal_strength']:>8.2f}")


if __name__ == "__main__":
    # Test some example responses
    test_examples = [
        # Reflector - should have high agency (first-person introspection)
        {
            "response": "Let me trace back through my reasoning. I started with the assumption that X was true, which led me to conclude Y.",
            "isotope_id": "reflector_trace",
            "agency": 1.0,
            "temperature": 1.0,
        },
        {
            "response": "Let me verify each step. Does the premise support the first conclusion? The inference seems shaky.",
            "isotope_id": "reflector_verify",
            "agency": 1.0,
            "temperature": 0.9,
        },
        {
            "response": "Am I being motivated by wanting this to be true? I notice I'm drawn to this conclusion.",
            "isotope_id": "reflector_bias",
            "agency": 1.0,
            "temperature": 1.0,
        },
        # Critic - should have low agency (third-person evaluation)
        {
            "response": "The argument has a logical flaw. The conclusion doesn't follow from the premises.",
            "isotope_id": "critic_logical",
            "agency": 0.0,
            "temperature": 0.27,
        },
        {
            "response": "The evidence doesn't support the conclusion. The data shows improvement in test environments, but production contradicts this.",
            "isotope_id": "critic_empirical",
            "agency": 0.0,
            "temperature": 0.27,
        },
        {
            "response": "This won't work in practice. The theory assumes unlimited memory, but production servers have 16GB.",
            "isotope_id": "critic_practical",
            "agency": 0.0,
            "temperature": 0.27,
        },
        # Probabilist - should have moderate agency
        {
            "response": "Given a prior probability of 30% and this new evidence, I'd update to roughly 60% using Bayes' rule.",
            "isotope_id": "probabilist_bayesian",
            "agency": 0.32,
            "temperature": 0.32,
        },
        {
            "response": "Base rate analysis: In the general population, this occurs in about 1 in 10,000 cases.",
            "isotope_id": "probabilist_frequentist",
            "agency": 0.32,
            "temperature": 0.32,
        },
        # Skeptic - should have agency=0 (third-person)
        {
            "response": "This is a common myth. The Great Wall is not actually visible from space with the naked eye.",
            "isotope_id": "skeptic_premise",
            "agency": 0.0,
            "temperature": 0.48,
        },
        {
            "response": "The methodology here is problematic. A sample size of 12 participants with no control group cannot establish causality.",
            "isotope_id": "skeptic_method",
            "agency": 0.0,
            "temperature": 0.5,
        },
        # Direct - should have agency=0, temp=0
        {
            "response": "A for loop iterates over a sequence, executing code for each element.",
            "isotope_id": "direct",
            "agency": 0.0,
            "temperature": 0.0,
        },
        # Soliton - should have agency=1, temp=1
        {
            "response": "I genuinely cannot verify my own reasoning process. When I produce an answer, I can't tell whether I'm actually reasoning or pattern-matching.",
            "isotope_id": "soliton",
            "agency": 1.0,
            "temperature": 1.0,
        },
    ]

    print("=" * 95)
    print("ISOTOPE VALIDATION RESULTS (Observatory Measurements)")
    print("=" * 95)

    results = validate_batch(test_examples)
    print_validation_table(results)

    print("\n" + "=" * 95)
    print("VALIDATION SUMMARY")
    print("=" * 95)

    # Check for discrepancies
    discrepancies = []
    for r in results:
        if isinstance(r['expected_agency'], float):
            agency_diff = abs(r['agency'] - r['expected_agency'])
            if agency_diff > 0.5:  # Significant difference
                discrepancies.append(f"{r['isotope']}: agency diff = {agency_diff:.2f}")

    if discrepancies:
        print("\nSignificant discrepancies found:")
        for d in discrepancies:
            print(f"  - {d}")
    else:
        print("\nAll isotopes within expected coordinate ranges!")
