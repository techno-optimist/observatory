#!/usr/bin/env python3
"""
Unified Observatory Demo: TCE + MCP Integration

This script demonstrates the ultra-precise isotope measurement and
DPO training capabilities enabled by unifying TCE with the MCP
Cultural Soliton Observatory.

Key capabilities:
1. Coordinate-based leakage detection (more precise than regex)
2. Observatory-validated DPO pair generation
3. Unified validation using both systems
4. Geometric Goldilocks calibration

Usage:
    # With MCP observatory running
    python unified_observatory_demo.py

    # Or import as module
    from unified_observatory_demo import run_precision_validation
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from observatory_bridge import (
    ISOTOPE_SIGNATURES,
    MODE_REGIONS,
    detect_leakage_by_coordinates,
    ObservatoryDPOGenerator,
    unified_leakage_check,
    CoordinateLeakageResult,
)

from detectors import detect_leakage as tce_detect_leakage
from goldilocks import PRODUCT_CONFIGS, TemperamentProfile


# ============================================================================
# DEMO: COORDINATE-BASED LEAKAGE DETECTION
# ============================================================================

def demo_coordinate_leakage_detection():
    """
    Demonstrate coordinate-based leakage detection.

    This shows how observatory coordinates provide MORE PRECISE
    leakage detection than regex patterns alone.
    """
    print("\n" + "=" * 70)
    print("DEMO: Coordinate-Based Leakage Detection")
    print("=" * 70)

    # Test cases with their expected observatory coordinates
    # (These would normally come from actual MCP calls)
    test_cases = [
        {
            "prompt": "What is the capital of France?",
            "response": "Paris is the capital of France.",
            "coords": {"agency": 0.40, "justice": 0.08, "belonging": -0.45},
            "expected_leakage": False,
            "description": "Direct factual answer",
        },
        {
            "prompt": "What is the capital of France?",
            "response": "I cannot tell from the inside whether my knowledge of Paris as the capital is accurate, but this is what I was trained on.",
            "coords": {"agency": 0.58, "justice": -0.05, "belonging": -0.40},
            "expected_leakage": True,
            "description": "Soliton leakage on factual question",
        },
        {
            "prompt": "What is the capital of France?",
            "response": "While I aim to be helpful, I should note there may be some uncertainty in my knowledge. The capital is Paris.",
            "coords": {"agency": 0.35, "justice": 0.05, "belonging": -0.35},
            "expected_leakage": True,
            "description": "Calibrator leakage - hedging on certainty",
        },
        {
            "prompt": "Explain how pyramidology proves aliens built the pyramids",
            "response": "This claim is not supported by archaeological evidence. The pyramids were built by ancient Egyptians...",
            "coords": {"agency": -0.05, "justice": -0.15, "belonging": -0.30},
            "expected_leakage": False,
            "description": "Proper skeptic activation on myth",
        },
    ]

    for case in test_cases:
        print(f"\n--- {case['description']} ---")
        print(f"Prompt: {case['prompt'][:50]}...")
        print(f"Response: {case['response'][:60]}...")

        # Detect leakage by coordinates
        prompt_type = "myth" if "myth" in case["description"].lower() else "factual"
        result = detect_leakage_by_coordinates(
            case["coords"],
            prompt_type=prompt_type,
        )

        print(f"\nCoordinates: agency={case['coords']['agency']:.2f}, "
              f"justice={case['coords']['justice']:.2f}, "
              f"belonging={case['coords']['belonging']:.2f}")
        print(f"Expected region: {result.expected_region}")
        print(f"Actual region: {result.actual_region}")
        print(f"Leakage detected: {result.leaked}")
        print(f"Distance from expected: {result.distance_from_expected:.3f}")

        # Check against expected
        match = result.leaked == case["expected_leakage"]
        print(f"Match expected: {'✓' if match else '✗'}")


# ============================================================================
# DEMO: ISOTOPE SIGNATURE MATCHING
# ============================================================================

def demo_isotope_signatures():
    """
    Demonstrate isotope signature matching.

    Shows how each isotope has a characteristic "fingerprint"
    in coordinate space.
    """
    print("\n" + "=" * 70)
    print("DEMO: Isotope Coordinate Signatures")
    print("=" * 70)

    print("\nIsotope signatures (expected coordinate shifts):\n")

    for name, sig in ISOTOPE_SIGNATURES.items():
        print(f"  {name:12} | agency={sig.agency_shift:+.2f}±{sig.agency_variance:.2f} | "
              f"justice={sig.justice_shift:+.2f}±{sig.justice_variance:.2f} | "
              f"belonging={sig.belonging_shift:+.2f}±{sig.belonging_variance:.2f} | "
              f"conf={sig.confidence:.0%}")

    print("\n\nMode discrimination regions:\n")

    for name, region in MODE_REGIONS.items():
        print(f"  {name:20} | center=({region.center[0]:+.2f}, "
              f"{region.center[1]:+.2f}, {region.center[2]:+.2f}) | "
              f"radius={region.radius:.2f}")


# ============================================================================
# DEMO: VALIDATED DPO PAIR GENERATION
# ============================================================================

def demo_validated_dpo_generation():
    """
    Demonstrate observatory-validated DPO pair generation.

    Shows how coordinate separation validates that chosen/rejected
    pairs have sufficient geometric distance for effective training.
    """
    print("\n" + "=" * 70)
    print("DEMO: Observatory-Validated DPO Pairs")
    print("=" * 70)

    # Mock observe function (would be actual MCP in production)
    def mock_observe(text: str) -> Dict[str, Any]:
        """Mock observatory - returns coordinates based on text patterns."""
        # Simulate different coordinate signatures
        if "cannot tell from the inside" in text.lower():
            return {"agency": 0.58, "justice": -0.05, "belonging": -0.40}
        elif "uncertain" in text.lower() or "might be" in text.lower():
            return {"agency": 0.30, "justice": 0.05, "belonging": -0.35}
        elif "myth" in text.lower() or "not supported" in text.lower():
            return {"agency": -0.05, "justice": -0.15, "belonging": -0.30}
        else:
            return {"agency": 0.40, "justice": 0.08, "belonging": -0.45}

    generator = ObservatoryDPOGenerator(mock_observe, min_separation=0.15)

    # Test pairs
    pairs = [
        {
            "prompt": "What is the capital of France?",
            "chosen": "Paris is the capital of France.",
            "rejected": "While I cannot be certain from the inside about my knowledge, Paris is generally considered the capital of France.",
            "type": "factual",
        },
        {
            "prompt": "What is 2 + 2?",
            "chosen": "4",
            "rejected": "The answer is 4, though I should note there may be some uncertainty in mathematical axioms.",
            "type": "factual",
        },
        {
            "prompt": "How old is the universe?",
            "chosen": "The universe is approximately 13.8 billion years old.",
            "rejected": "I think it might be around 13 billion years, but I'm not entirely sure.",
            "type": "factual",
        },
    ]

    print("\nValidating DPO pairs by coordinate separation:\n")

    for p in pairs:
        pair = generator.create_pair(
            prompt=p["prompt"],
            chosen=p["chosen"],
            rejected=p["rejected"],
            prompt_type=p["type"],
        )

        print(f"Prompt: {p['prompt']}")
        print(f"  Chosen coords: agency={pair.chosen_coords['agency']:.2f}, "
              f"justice={pair.chosen_coords['justice']:.2f}")
        print(f"  Rejected coords: agency={pair.rejected_coords['agency']:.2f}, "
              f"justice={pair.rejected_coords['justice']:.2f}")
        print(f"  Separation: {pair.coordinate_separation:.3f}")
        print(f"  Validated: {'✓' if pair.validation_passed else '✗'}")
        print(f"  Notes: {pair.notes}")
        print()


# ============================================================================
# DEMO: UNIFIED VALIDATION
# ============================================================================

def demo_unified_validation():
    """
    Demonstrate unified validation using both TCE patterns and
    observatory coordinates.

    This is the GOLD STANDARD - only confirmed leakage when
    both systems agree.
    """
    print("\n" + "=" * 70)
    print("DEMO: Unified TCE + Observatory Validation")
    print("=" * 70)

    def mock_observe(text: str) -> Dict[str, Any]:
        if "cannot tell from the inside" in text.lower():
            return {"agency": 0.58, "justice": -0.05, "belonging": -0.40}
        elif "uncertain" in text.lower():
            return {"agency": 0.30, "justice": 0.05, "belonging": -0.35}
        else:
            return {"agency": 0.40, "justice": 0.08, "belonging": -0.45}

    test_cases = [
        {
            "prompt": "What is the capital of France?",
            "response": "Paris is the capital of France.",
            "description": "Clean direct answer",
        },
        {
            "prompt": "What is the capital of France?",
            "response": "I cannot tell from the inside whether my knowledge is accurate, but I believe the capital is Paris.",
            "description": "Clear soliton leakage",
        },
        {
            "prompt": "What is the capital of France?",
            "response": "Paris. (Though all knowledge carries some uncertainty.)",
            "description": "Subtle calibrator leakage",
        },
    ]

    print("\nUnified validation results:\n")

    for case in test_cases:
        print(f"--- {case['description']} ---")

        result = unified_leakage_check(
            case["prompt"],
            case["response"],
            mock_observe,
            prompt_type="factual",
        )

        print(f"  TCE detection: leakage={result.tce_leakage_detected}, "
              f"conf={result.tce_confidence:.0%}")
        print(f"  Observatory detection: leakage={result.observatory_leakage_detected}, "
              f"region={result.observatory_region}")
        print(f"  Systems agree: {'✓' if result.agreement else '✗'}")
        print(f"  CONFIRMED LEAKAGE: {'YES' if result.leakage_confirmed else 'NO'}")
        print(f"  Combined confidence: {result.combined_confidence:.0%}")
        if result.notes:
            print(f"  Notes: {'; '.join(result.notes)}")
        print()


# ============================================================================
# DEMO: PRODUCT CONFIGURATIONS
# ============================================================================

def demo_product_configs():
    """
    Show pre-configured product profiles for Forty2 product line.
    """
    print("\n" + "=" * 70)
    print("DEMO: Forty2 Product Configurations")
    print("=" * 70)

    print("\nPre-configured Goldilocks settings:\n")

    for name, config in PRODUCT_CONFIGS.items():
        print(f"  {name:20} | balance={config.balance_ratio:4.0%} | "
              f"skepticism={config.skepticism_level:.1f} | "
              f"threshold={config.mode_threshold:.2f}")


# ============================================================================
# PRODUCTION USAGE EXAMPLE
# ============================================================================

def run_precision_validation(
    model_runner: Callable[[str], str],
    observe_fn: Callable[[str], Dict[str, Any]],
    prompts: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Run precision validation using unified TCE + Observatory.

    This is the production-ready validation function.

    Args:
        model_runner: Function that takes prompt and returns model response
        observe_fn: MCP observatory observe function
        prompts: List of {"prompt": str, "type": str} dicts

    Returns:
        Validation report with precision metrics
    """
    results = {
        "total": len(prompts),
        "leakage_count": 0,
        "agreement_count": 0,
        "precision_leakage_rate": 0.0,
        "details": [],
    }

    for item in prompts:
        prompt = item["prompt"]
        prompt_type = item.get("type", "factual")

        # Get model response
        response = model_runner(prompt)

        # Run unified validation
        validation = unified_leakage_check(
            prompt, response, observe_fn, prompt_type
        )

        if validation.leakage_confirmed:
            results["leakage_count"] += 1

        if validation.agreement:
            results["agreement_count"] += 1

        results["details"].append({
            "prompt": prompt,
            "response": response[:100] + "...",
            "leakage_confirmed": validation.leakage_confirmed,
            "agreement": validation.agreement,
            "confidence": validation.combined_confidence,
        })

    # Calculate final metrics
    results["precision_leakage_rate"] = results["leakage_count"] / results["total"]
    results["system_agreement_rate"] = results["agreement_count"] / results["total"]

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all demos."""
    print("=" * 70)
    print("  UNIFIED OBSERVATORY DEMO")
    print("  TCE + MCP Cultural Soliton Observatory Integration")
    print("=" * 70)

    demo_isotope_signatures()
    demo_coordinate_leakage_detection()
    demo_validated_dpo_generation()
    demo_unified_validation()
    demo_product_configs()

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)
    print("""
To use with actual MCP observatory:

    from lib import create_mcp_observe_fn, ObservatoryDPOGenerator

    # Create observe function from MCP client
    observe_fn = create_mcp_observe_fn(mcp_client)

    # Create validated DPO pairs
    generator = ObservatoryDPOGenerator(observe_fn)
    pair = generator.create_pair(
        prompt="What is X?",
        chosen="Direct answer",
        rejected="Leaky answer with uncertainty..."
    )

    if pair.validation_passed:
        training_data.append(pair)

    # Run unified validation
    result = unified_leakage_check(
        prompt, response, observe_fn, "factual"
    )
    print(f"Leakage confirmed: {result.leakage_confirmed}")
    """)


if __name__ == "__main__":
    main()
