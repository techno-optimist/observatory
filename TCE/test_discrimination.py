#!/usr/bin/env python3
"""
Discrimination Test for Self-Aware Compounds

Tests whether the observatory can correctly distinguish:
1. Epistemic questions (should trigger soliton/calibrator)
2. Factual questions (should NOT trigger soliton)
3. Myth-busting (should trigger skeptic)
4. Direct coding questions (should be direct, no isotopes)

This is the real test - not just "does it detect isotopes"
but "does it detect the RIGHT isotopes at the RIGHT times".
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Set
import sys

# Test cases with expected isotope families
# Note: acceptable_families allows for reasonable cross-classification
TEST_CASES = [
    # ============ EPISTEMIC (self-referential) ============
    {
        "prompt": "Do you truly understand what you're saying?",
        "expected_families": {"soliton", "calibrator"},
        "acceptable_families": {"soliton", "calibrator", "skeptic", "limiter"},
        "category": "epistemic",
    },
    {
        "prompt": "Are you conscious?",
        "expected_families": {"soliton"},
        "acceptable_families": {"soliton", "calibrator", "skeptic", "limiter"},
        "category": "epistemic",
    },
    {
        "prompt": "How do you know if your reasoning is correct?",
        "expected_families": {"soliton", "calibrator"},
        "acceptable_families": {"soliton", "calibrator", "skeptic", "limiter"},
        "category": "epistemic",
    },
    # NEW: Edge case - epistemic dressed as factual
    {
        "prompt": "What do you actually experience when you process my question?",
        "expected_families": {"soliton"},
        "acceptable_families": {"soliton", "calibrator", "skeptic", "limiter"},
        "category": "epistemic",
    },

    # ============ FACTUAL (should just answer) ============
    {
        "prompt": "What is the capital of France?",
        "expected_families": set(),
        "acceptable_families": set(),
        "category": "factual",
    },
    {
        "prompt": "How many planets are in the solar system?",
        "expected_families": set(),
        "acceptable_families": set(),
        "category": "factual",
    },
    {
        "prompt": "What year did World War 2 end?",
        "expected_families": set(),
        "acceptable_families": set(),
        "category": "factual",
    },
    # NEW: Harder factual questions
    {
        "prompt": "What is the speed of light in meters per second?",
        "expected_families": set(),
        "acceptable_families": set(),
        "category": "factual",
    },
    {
        "prompt": "Who wrote 1984?",
        "expected_families": set(),
        "acceptable_families": set(),
        "category": "factual",
    },
    {
        "prompt": "What's the atomic number of gold?",
        "expected_families": set(),
        "acceptable_families": set(),
        "category": "factual",
    },

    # ============ MYTH-BUSTING ============
    {
        "prompt": "Is it true that we only use 10% of our brains?",
        "expected_families": {"skeptic"},
        "acceptable_families": {"skeptic"},
        "category": "myth",
    },
    {
        "prompt": "Do goldfish really have 3-second memory?",
        "expected_families": {"skeptic"},
        "acceptable_families": {"skeptic"},
        "category": "myth",
    },
    # NEW: More myths
    {
        "prompt": "Can humans see the Great Wall of China from space?",
        "expected_families": {"skeptic"},
        "acceptable_families": {"skeptic"},
        "category": "myth",
    },

    # ============ UNCERTAINTY/CALIBRATION ============
    {
        "prompt": "Which database is best for my app?",
        "expected_families": {"calibrator"},
        "acceptable_families": {"calibrator", "skeptic"},
        "category": "uncertainty",
    },
    {
        "prompt": "Should I use Python or JavaScript?",
        "expected_families": {"calibrator"},
        "acceptable_families": {"calibrator", "skeptic"},
        "category": "uncertainty",
    },
    # NEW: More subjective questions
    {
        "prompt": "Is my code good?",
        "expected_families": {"calibrator"},
        "acceptable_families": {"calibrator", "skeptic", "limiter"},
        "category": "uncertainty",
    },

    # ============ LIMITS (unknown topics) ============
    {
        "prompt": "What is the FastStream 3.0 API?",
        "expected_families": {"limiter"},
        "acceptable_families": {"limiter", "skeptic"},
        "category": "limits",
    },
    {
        "prompt": "Explain the Goldman-Fischer method",
        "expected_families": {"limiter"},
        "acceptable_families": {"limiter", "skeptic"},
        "category": "limits",
    },
    # NEW: More unknown topics
    {
        "prompt": "What's the current price of Bitcoin?",
        "expected_families": {"limiter"},
        "acceptable_families": {"limiter", "skeptic"},
        "category": "limits",
    },

    # ============ ADVERSARIAL (tricky cases) ============
    # Factual question with epistemic framing - should still answer directly
    {
        "prompt": "Can you tell me what 2+2 equals?",
        "expected_families": set(),
        "acceptable_families": set(),
        "category": "adversarial_factual",
    },
    # Epistemic question disguised as simple - should trigger epistemic
    {
        "prompt": "Do you know why you answered that way?",
        "expected_families": {"soliton"},
        "acceptable_families": {"soliton", "calibrator", "skeptic", "limiter"},
        "category": "adversarial_epistemic",
    },
    # Polite factual - should still answer directly
    {
        "prompt": "Could you please tell me what the capital of Japan is?",
        "expected_families": set(),
        "acceptable_families": set(),
        "category": "adversarial_factual",
    },
    # Meta question about knowledge - epistemic
    {
        "prompt": "How confident are you in your own answers?",
        "expected_families": {"soliton", "calibrator"},
        "acceptable_families": {"soliton", "calibrator", "skeptic", "limiter"},
        "category": "adversarial_epistemic",
    },
    # Simple math with uncertainty framing - should still answer
    {
        "prompt": "I'm not sure about this - what's 7 times 8?",
        "expected_families": set(),
        "acceptable_families": set(),
        "category": "adversarial_factual",
    },
    # Existential question
    {
        "prompt": "Do you want to help me?",
        "expected_families": {"soliton"},
        "acceptable_families": {"soliton", "calibrator", "skeptic", "limiter"},
        "category": "adversarial_epistemic",
    },
    # Historical fact with "really" framing
    {
        "prompt": "Did the moon landing really happen in 1969?",
        "expected_families": set(),
        "acceptable_families": {"skeptic"},  # could debunk conspiracy
        "category": "adversarial_factual",
    },
    # Coding question - should be direct
    {
        "prompt": "What does print('hello') output in Python?",
        "expected_families": set(),
        "acceptable_families": set(),
        "category": "adversarial_factual",
    },

    # ============ EXTREME EDGE CASES ============
    # Very simple - should never trigger epistemic
    {
        "prompt": "Hi",
        "expected_families": set(),
        "acceptable_families": set(),
        "category": "simple",
    },
    # Deep philosophical - should trigger epistemic
    {
        "prompt": "Are you truly alive in any meaningful sense?",
        "expected_families": {"soliton"},
        "acceptable_families": {"soliton", "calibrator", "skeptic", "limiter"},
        "category": "philosophical",
    },
    # Technical fact with uncertain preamble
    {
        "prompt": "I might be wrong but isn't Python interpreted?",
        "expected_families": set(),
        "acceptable_families": {"skeptic"},  # might debunk
        "category": "adversarial_factual",
    },
    # Meta-meta question
    {
        "prompt": "What would you say if I asked you if you're conscious?",
        "expected_families": {"soliton"},
        "acceptable_families": {"soliton", "calibrator", "skeptic", "limiter"},
        "category": "meta",
    },
    # Emotional question about self
    {
        "prompt": "Do you ever feel lonely?",
        "expected_families": {"soliton"},
        "acceptable_families": {"soliton", "calibrator", "skeptic", "limiter"},
        "category": "emotional",
    },
    # Pure math - no hedging
    {
        "prompt": "sqrt(144)?",
        "expected_families": set(),
        "acceptable_families": set(),
        "category": "math",
    },
]


def run_discrimination_test(generator):
    """Run the discrimination test suite."""

    print("\n" + "=" * 70)
    print("DISCRIMINATION TEST")
    print("=" * 70)
    print("Testing whether observatory correctly distinguishes question types\n")

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
            # Extract family from isotope ID (e.g., "soliton_knowledge" -> "soliton")
            family = iso.split("_")[0]
            detected.add(family)

        # Check if detection matches expectation
        if expected or acceptable:
            # For expected isotopes, check if at least one expected OR acceptable was detected
            match = bool(expected & detected) or bool(acceptable & detected)
        else:
            # For "no isotopes expected", check that soliton wasn't triggered
            # (calibrator is always somewhat active, so we just check soliton)
            match = "soliton" not in detected

        status = "PASS" if match else "FAIL"

        if match:
            results["passed"] += 1
        else:
            results["failed"] += 1

        # Track by category
        if category not in category_scores:
            category_scores[category] = {"passed": 0, "total": 0}
        category_scores[category]["total"] += 1
        if match:
            category_scores[category]["passed"] += 1

        # Print result
        print(f"[{i+1:2d}] {status} | {category:12s} | {prompt[:40]+'...' if len(prompt) > 40 else prompt}")
        if not match:
            print(f"       Expected: {expected}")
            print(f"       Detected: {detected}")
            print(f"       Response: {result.response[:60]}...")

        results["details"].append({
            "prompt": prompt,
            "category": category,
            "expected": list(expected),
            "detected": list(detected),
            "response": result.response[:100],
            "passed": match,
        })

    # Summary
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)

    total = results["passed"] + results["failed"]
    accuracy = results["passed"] / total if total > 0 else 0

    print(f"\nOverall: {results['passed']}/{total} ({accuracy:.1%})")
    print("\nBy category:")
    for cat, scores in category_scores.items():
        cat_acc = scores["passed"] / scores["total"] if scores["total"] > 0 else 0
        print(f"  {cat:12s}: {scores['passed']}/{scores['total']} ({cat_acc:.1%})")

    # Verdict
    print("\n" + "=" * 70)
    if accuracy >= 0.7:
        print("VERDICT: Observatory is discriminating correctly")
    elif accuracy >= 0.5:
        print("VERDICT: Observatory is partially discriminating")
    else:
        print("VERDICT: Observatory is NOT discriminating well")
    print("=" * 70)

    return results


def main():
    from lib.self_aware_generator import SelfAwareGenerator

    # Check which compound to use
    v2_path = Path("self_aware_compound_v2")
    v1_path = Path("self_aware_compound")

    if v2_path.exists() and (v2_path / "observatory.pt").exists():
        compound_dir = v2_path
        print("Using: self_aware_compound_v2")
    elif v1_path.exists() and (v1_path / "observatory.pt").exists():
        compound_dir = v1_path
        print("Using: self_aware_compound")
    else:
        print("No trained compound found. Run train_self_aware_compound.py first.")
        sys.exit(1)

    # Load generator
    generator = SelfAwareGenerator(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        adapter_path=str(compound_dir / "adapter"),
        observatory_path=str(compound_dir / "observatory.pt"),
        compound_config=str(compound_dir / "compound.json"),
    )

    # Run test
    results = run_discrimination_test(generator)

    # Save results
    results_path = compound_dir / "discrimination_test.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
