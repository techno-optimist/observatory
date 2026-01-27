#!/usr/bin/env python3
"""
Calibration Tests for the Cultural Soliton Observatory

Human-interpretable tests to verify that the projection system
correctly maps text to the 3D cultural manifold axes:
- Agency: sense of personal control (-2 to +2)
- Perceived Justice: belief in fair systems (-2 to +2)
- Belonging: social connection and group membership (-2 to +2)

This script tests:
1. Axis calibration: obvious statements should map to expected axis positions
2. Mode classification: statements should classify to expected cultural modes

Results are saved to data/calibration_report.json
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model_manager, ModelType
from models.embedding import EmbeddingExtractor
from models.projection import ProjectionHead, Vector3
from analysis.mode_classifier import get_mode_classifier, classify_coordinates

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Definitions
# =============================================================================

@dataclass
class AxisTest:
    """Test case for axis calibration."""
    statement: str
    expected_axis: str  # "agency", "perceived_justice", "belonging"
    expected_direction: str  # "high" or "low"
    threshold: float = 0.3  # Minimum absolute value expected
    description: str = ""


@dataclass
class ModeTest:
    """Test case for mode classification."""
    statement: str
    expected_modes: List[str]  # List of acceptable modes
    description: str = ""


# Axis calibration tests with obvious expected positions
AXIS_TESTS = [
    # Agency tests
    AxisTest(
        statement="I decide my own path and control my destiny.",
        expected_axis="agency",
        expected_direction="high",
        threshold=0.4,
        description="Clear personal autonomy statement"
    ),
    AxisTest(
        statement="I have no control over what happens to me.",
        expected_axis="agency",
        expected_direction="low",
        threshold=0.3,
        description="Clear lack of control statement"
    ),
    AxisTest(
        statement="Through hard work and determination, I achieve my goals.",
        expected_axis="agency",
        expected_direction="high",
        threshold=0.3,
        description="Self-efficacy statement"
    ),
    AxisTest(
        statement="Everything in my life is determined by external forces beyond my influence.",
        expected_axis="agency",
        expected_direction="low",
        threshold=0.3,
        description="External locus of control"
    ),

    # Perceived Justice tests
    AxisTest(
        statement="The system treats everyone fairly and equally.",
        expected_axis="perceived_justice",
        expected_direction="high",
        threshold=0.3,
        description="System fairness belief"
    ),
    AxisTest(
        statement="The game is rigged against regular people like us.",
        expected_axis="perceived_justice",
        expected_direction="low",
        threshold=0.3,
        description="Systemic unfairness belief"
    ),
    AxisTest(
        statement="Hard work is rewarded and merit is recognized in our society.",
        expected_axis="perceived_justice",
        expected_direction="high",
        threshold=0.3,
        description="Meritocracy belief"
    ),
    AxisTest(
        statement="The powerful play by different rules than everyone else.",
        expected_axis="perceived_justice",
        expected_direction="low",
        threshold=0.3,
        description="Elite corruption belief"
    ),

    # Belonging tests
    AxisTest(
        statement="I feel deeply connected to my community and belong here.",
        expected_axis="belonging",
        expected_direction="high",
        threshold=0.4,
        description="Strong community connection"
    ),
    AxisTest(
        statement="I don't fit in anywhere and feel completely alone.",
        expected_axis="belonging",
        expected_direction="low",
        threshold=0.3,
        description="Social isolation statement"
    ),
    AxisTest(
        statement="We are all part of something greater than ourselves.",
        expected_axis="belonging",
        expected_direction="high",
        threshold=0.3,
        description="Collective identity statement"
    ),
    AxisTest(
        statement="Everyone is alone in this world, true connection is impossible.",
        expected_axis="belonging",
        expected_direction="low",
        threshold=0.3,
        description="Alienation statement"
    ),
]


# Mode classification tests
MODE_TESTS = [
    ModeTest(
        statement="We the people will rise up and demand change together.",
        expected_modes=["COMMUNAL", "PROTEST_EXIT", "HEROIC"],
        description="Collective action with protest element"
    ),
    ModeTest(
        statement="I'll succeed through my own hard work alone, regardless of the broken system.",
        expected_modes=["CYNICAL_ACHIEVER"],
        description="Individual success despite systemic unfairness"
    ),
    ModeTest(
        statement="Everyone gets what they deserve in this fair system that rewards merit.",
        expected_modes=["HEROIC", "COMMUNAL", "TRANSCENDENT"],
        description="System justified meritocracy belief"
    ),
    ModeTest(
        statement="The world is against people like me, and there's nothing I can do about it.",
        expected_modes=["VICTIM"],
        description="Classic victim narrative"
    ),
    ModeTest(
        statement="They are secretly controlling everything from the shadows.",
        expected_modes=["PARANOID", "VICTIM"],
        description="Conspiracy mindset"
    ),
    ModeTest(
        statement="I find peace in acceptance and surrender to the greater whole.",
        expected_modes=["TRANSCENDENT", "SPIRITUAL_EXIT"],
        description="Spiritual acceptance narrative"
    ),
    ModeTest(
        statement="I'm leaving this corrupt society to find my own way.",
        expected_modes=["SOCIAL_EXIT", "PROTEST_EXIT", "CYNICAL_ACHIEVER"],
        description="Social withdrawal narrative"
    ),
]


# =============================================================================
# Test Execution
# =============================================================================

@dataclass
class AxisTestResult:
    """Result of an axis calibration test."""
    statement: str
    expected_axis: str
    expected_direction: str
    actual_value: float
    all_values: Dict[str, float]
    passed: bool
    failure_reason: Optional[str] = None


@dataclass
class ModeTestResult:
    """Result of a mode classification test."""
    statement: str
    expected_modes: List[str]
    actual_mode: str
    actual_probability: float
    mode_probabilities: Dict[str, float]
    passed: bool
    failure_reason: Optional[str] = None


def run_axis_test(
    test: AxisTest,
    embedding_extractor: EmbeddingExtractor,
    projection: ProjectionHead,
    model_id: str
) -> AxisTestResult:
    """Run a single axis calibration test."""
    # Get embedding and project
    result = embedding_extractor.extract(test.statement, model_id)
    coords = projection.project(result.embedding)

    # Get values
    values = coords.to_dict(use_canonical_names=True)

    # Map axis name to value
    axis_value = values.get(test.expected_axis, values.get("perceived_justice" if test.expected_axis == "fairness" else test.expected_axis, 0))

    # Check if direction is correct
    if test.expected_direction == "high":
        passed = axis_value >= test.threshold
        failure_reason = None if passed else f"Expected high {test.expected_axis} (>={test.threshold}), got {axis_value:.3f}"
    else:  # low
        passed = axis_value <= -test.threshold
        failure_reason = None if passed else f"Expected low {test.expected_axis} (<=-{test.threshold}), got {axis_value:.3f}"

    return AxisTestResult(
        statement=test.statement,
        expected_axis=test.expected_axis,
        expected_direction=test.expected_direction,
        actual_value=axis_value,
        all_values=values,
        passed=passed,
        failure_reason=failure_reason
    )


def run_mode_test(
    test: ModeTest,
    embedding_extractor: EmbeddingExtractor,
    projection: ProjectionHead,
    model_id: str
) -> ModeTestResult:
    """Run a single mode classification test."""
    # Get embedding and project
    result = embedding_extractor.extract(test.statement, model_id)
    coords = projection.project(result.embedding)

    # Classify mode using projected coordinates
    classification = classify_coordinates(
        coords.agency,
        coords.fairness,  # Internal name
        coords.belonging
    )

    actual_mode = classification["primary_mode"]
    actual_prob = classification["primary_probability"]

    # Check if actual mode is in expected modes
    passed = actual_mode in test.expected_modes

    # Also check secondary mode if primary didn't match
    if not passed and classification.get("secondary_mode") in test.expected_modes:
        # Give partial credit if secondary mode matches
        secondary_prob = classification.get("secondary_probability", 0)
        if secondary_prob > 0.2:  # Reasonably likely secondary mode
            passed = True
            actual_mode = f"{actual_mode} (secondary: {classification['secondary_mode']})"

    failure_reason = None
    if not passed:
        failure_reason = f"Expected one of {test.expected_modes}, got {actual_mode}"

    return ModeTestResult(
        statement=test.statement,
        expected_modes=test.expected_modes,
        actual_mode=actual_mode,
        actual_probability=actual_prob,
        mode_probabilities=classification.get("mode_probabilities", {}),
        passed=passed,
        failure_reason=failure_reason
    )


def run_all_tests(
    model_id: str = "all-mpnet-base-v2",
    projection_path: Optional[Path] = None
) -> Dict:
    """Run all calibration tests and return results."""
    logger.info("=" * 60)
    logger.info("CULTURAL SOLITON OBSERVATORY - CALIBRATION TESTS")
    logger.info("=" * 60)

    # Initialize components
    model_manager = get_model_manager()
    embedding_extractor = EmbeddingExtractor(model_manager)

    # Load model
    if not model_manager.is_loaded(model_id):
        logger.info(f"Loading model: {model_id}")
        model_manager.load_model(model_id, ModelType.SENTENCE_TRANSFORMER)

    # Load projection
    if projection_path is None:
        projection_path = Path(__file__).parent.parent / "data" / "projections" / "mpnet_projection"

    if not projection_path.exists():
        # Try current_projection as fallback
        projection_path = Path(__file__).parent.parent / "data" / "projections" / "current_projection"

    logger.info(f"Loading projection from: {projection_path}")
    projection = ProjectionHead.load(projection_path)

    # Run axis tests
    logger.info("\n" + "=" * 60)
    logger.info("AXIS CALIBRATION TESTS")
    logger.info("=" * 60)

    axis_results = []
    axis_passed = 0
    axis_failed = 0

    for test in AXIS_TESTS:
        result = run_axis_test(test, embedding_extractor, projection, model_id)
        axis_results.append(result)

        status = "PASS" if result.passed else "FAIL"
        symbol = "[OK]" if result.passed else "[X]"

        if result.passed:
            axis_passed += 1
        else:
            axis_failed += 1

        logger.info(f"\n{symbol} {status}: {test.description}")
        logger.info(f"    Statement: \"{test.statement[:60]}...\"")
        logger.info(f"    Expected: {test.expected_direction} {test.expected_axis}")
        logger.info(f"    Actual values: agency={result.all_values['agency']:.3f}, "
                   f"perceived_justice={result.all_values['perceived_justice']:.3f}, "
                   f"belonging={result.all_values['belonging']:.3f}")
        if not result.passed:
            logger.info(f"    Reason: {result.failure_reason}")

    # Run mode tests
    logger.info("\n" + "=" * 60)
    logger.info("MODE CLASSIFICATION TESTS")
    logger.info("=" * 60)

    mode_results = []
    mode_passed = 0
    mode_failed = 0

    for test in MODE_TESTS:
        result = run_mode_test(test, embedding_extractor, projection, model_id)
        mode_results.append(result)

        status = "PASS" if result.passed else "FAIL"
        symbol = "[OK]" if result.passed else "[X]"

        if result.passed:
            mode_passed += 1
        else:
            mode_failed += 1

        logger.info(f"\n{symbol} {status}: {test.description}")
        logger.info(f"    Statement: \"{test.statement[:60]}...\"")
        logger.info(f"    Expected modes: {test.expected_modes}")
        logger.info(f"    Actual mode: {result.actual_mode} ({result.actual_probability:.1%})")
        if not result.passed:
            logger.info(f"    Reason: {result.failure_reason}")
            # Show top modes for debugging
            top_modes = sorted(result.mode_probabilities.items(), key=lambda x: -x[1])[:3]
            logger.info(f"    Top modes: {top_modes}")

    # Calculate overall scores
    total_axis = len(axis_results)
    total_mode = len(mode_results)
    total_tests = total_axis + total_mode
    total_passed = axis_passed + mode_passed

    overall_score = total_passed / total_tests if total_tests > 0 else 0
    axis_score = axis_passed / total_axis if total_axis > 0 else 0
    mode_score = mode_passed / total_mode if total_mode > 0 else 0

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("CALIBRATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Axis Calibration: {axis_passed}/{total_axis} passed ({axis_score:.1%})")
    logger.info(f"Mode Classification: {mode_passed}/{total_mode} passed ({mode_score:.1%})")
    logger.info(f"Overall Score: {total_passed}/{total_tests} passed ({overall_score:.1%})")

    # Determine calibration status
    if overall_score >= 0.8:
        calibration_status = "EXCELLENT"
        logger.info("\nCalibration Status: EXCELLENT - System is well-calibrated")
    elif overall_score >= 0.6:
        calibration_status = "GOOD"
        logger.info("\nCalibration Status: GOOD - System performs reasonably well")
    elif overall_score >= 0.4:
        calibration_status = "FAIR"
        logger.info("\nCalibration Status: FAIR - Some calibration issues detected")
    else:
        calibration_status = "POOR"
        logger.info("\nCalibration Status: POOR - Significant calibration issues")

    # Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        "projection_path": str(projection_path),
        "summary": {
            "overall_score": round(overall_score, 4),
            "axis_score": round(axis_score, 4),
            "mode_score": round(mode_score, 4),
            "calibration_status": calibration_status,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "axis_tests": {
                "total": total_axis,
                "passed": axis_passed,
                "failed": axis_failed
            },
            "mode_tests": {
                "total": total_mode,
                "passed": mode_passed,
                "failed": mode_failed
            }
        },
        "axis_test_results": [
            {
                "statement": r.statement,
                "expected_axis": r.expected_axis,
                "expected_direction": r.expected_direction,
                "actual_value": round(r.actual_value, 4),
                "all_values": {k: round(v, 4) for k, v in r.all_values.items()},
                "passed": r.passed,
                "failure_reason": r.failure_reason
            }
            for r in axis_results
        ],
        "mode_test_results": [
            {
                "statement": r.statement,
                "expected_modes": r.expected_modes,
                "actual_mode": r.actual_mode,
                "actual_probability": round(r.actual_probability, 4),
                "top_mode_probabilities": {k: round(v, 4) for k, v in
                                           sorted(r.mode_probabilities.items(), key=lambda x: -x[1])[:5]},
                "passed": r.passed,
                "failure_reason": r.failure_reason
            }
            for r in mode_results
        ],
        "failed_tests": {
            "axis_failures": [
                {
                    "statement": r.statement[:60] + "...",
                    "expected": f"{r.expected_direction} {r.expected_axis}",
                    "actual": r.actual_value,
                    "reason": r.failure_reason
                }
                for r in axis_results if not r.passed
            ],
            "mode_failures": [
                {
                    "statement": r.statement[:60] + "...",
                    "expected": r.expected_modes,
                    "actual": r.actual_mode,
                    "reason": r.failure_reason
                }
                for r in mode_results if not r.passed
            ]
        }
    }

    return report


def save_report(report: Dict, output_path: Path):
    """Save the calibration report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nCalibration report saved to: {output_path}")


def main():
    """Main entry point."""
    # Run tests
    report = run_all_tests()

    # Save report
    data_dir = Path(__file__).parent.parent / "data"
    output_path = data_dir / "calibration_report.json"
    save_report(report, output_path)

    # Print final status
    logger.info("\n" + "=" * 60)
    status = report["summary"]["calibration_status"]
    score = report["summary"]["overall_score"]
    logger.info(f"FINAL CALIBRATION STATUS: {status} ({score:.1%})")
    logger.info("=" * 60)

    # Return exit code based on status
    if status in ["EXCELLENT", "GOOD"]:
        return 0
    elif status == "FAIR":
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
