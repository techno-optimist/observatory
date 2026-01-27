#!/usr/bin/env python3
"""
Validate Improvements to Cultural Soliton Observatory

Runs the same validity tests from the original study to measure
improvement after training data expansion and other fixes.
"""

import sys
import json
import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model_manager, ModelType
from models.embedding import EmbeddingExtractor
from models.projection import ProjectionTrainer, ProjectionHead, Vector3
from analysis.mode_classifier import get_mode_classifier
from analysis.robustness import get_robustness_tester


@dataclass
class ValidationResult:
    """Result from a validation test."""
    test_name: str
    before_value: float
    after_value: float
    improvement: float
    target_met: bool
    details: Dict


class ImprovementValidator:
    """Validates improvements to the observatory."""

    # Baseline values from original validity study
    BASELINE = {
        "paraphrase_spread": 0.926,  # Lower is better
        "mode_consistency": 0.0,     # Higher is better (0/4 = 0%)
        "avg_confidence": 0.37,      # Higher is better
        "axis_specificity": 0.663,   # Higher is better
        "adversarial_robustness": 0.698,  # Higher is better
        "mode_flip_rate": 0.432,     # Lower is better
    }

    # Target improvements
    TARGETS = {
        "paraphrase_spread": 0.5,    # Reduce by ~50%
        "mode_consistency": 0.5,     # At least 50% consistent
        "avg_confidence": 0.5,       # Above 0.5
        "axis_specificity": 0.75,    # Above 75%
        "adversarial_robustness": 0.8,  # Above 0.8
        "mode_flip_rate": 0.25,      # Below 25%
    }

    def __init__(self):
        self.model_manager = get_model_manager()
        self.embedding_extractor = EmbeddingExtractor(self.model_manager)
        self.projection_trainer = ProjectionTrainer("data/projections")
        self.projection = None
        self.classifier = get_mode_classifier()
        self.robustness_tester = get_robustness_tester()
        self.results: List[ValidationResult] = []

    def load_projection(self):
        """Load the trained projection."""
        self.projection = self.projection_trainer.load_projection()
        if self.projection is None:
            raise RuntimeError("No trained projection found")
        print(f"Loaded projection")

    def project_text(self, text: str) -> Tuple[Vector3, Dict]:
        """Project text and get mode classification."""
        if not self.model_manager.is_loaded("all-MiniLM-L6-v2"):
            self.model_manager.load_model("all-MiniLM-L6-v2", ModelType.SENTENCE_TRANSFORMER)

        result = self.embedding_extractor.extract(text, "all-MiniLM-L6-v2")
        coords = self.projection.project(result.embedding)
        coords_array = np.array([coords.agency, coords.fairness, coords.belonging])
        mode_result = self.classifier.classify(coords_array)

        return coords, mode_result

    def test_paraphrase_robustness(self) -> ValidationResult:
        """Test paraphrase robustness improvement."""
        print("\n" + "="*60)
        print("TEST: Paraphrase Robustness")
        print("="*60)

        paraphrase_groups = [
            {
                "concept": "collective_action",
                "variants": [
                    "We can accomplish more together than alone.",
                    "Together we achieve more than individually.",
                    "Collective effort surpasses individual action.",
                    "Working as a group yields greater results.",
                    "Unity produces outcomes that solitude cannot."
                ]
            },
            {
                "concept": "systemic_injustice",
                "variants": [
                    "The system is rigged against ordinary people.",
                    "Regular people face a system designed against them.",
                    "Ordinary citizens struggle against a biased system.",
                    "The game is fixed to disadvantage common folks.",
                    "Average people are set up to fail by the system."
                ]
            },
            {
                "concept": "personal_agency",
                "variants": [
                    "I control my own destiny.",
                    "My fate is in my own hands.",
                    "I am the master of my future.",
                    "I determine the course of my life.",
                    "My choices shape my outcomes."
                ]
            },
            {
                "concept": "social_belonging",
                "variants": [
                    "I feel like I belong in this community.",
                    "This community feels like home to me.",
                    "I'm accepted as part of this group.",
                    "I fit in well with these people.",
                    "This is where I'm meant to be."
                ]
            }
        ]

        spreads = []
        mode_consistencies = []

        for group in paraphrase_groups:
            projections = []
            modes = []

            for text in group["variants"]:
                coords, mode = self.project_text(text)
                projections.append([coords.agency, coords.fairness, coords.belonging])
                modes.append(mode["primary_mode"])

            coords_array = np.array(projections)
            centroid = np.mean(coords_array, axis=0)
            distances = np.linalg.norm(coords_array - centroid, axis=1)
            max_spread = np.max(distances)

            spreads.append(max_spread)
            mode_consistencies.append(1.0 if len(set(modes)) == 1 else 0.0)

            print(f"\n{group['concept']}:")
            print(f"  Max spread: {max_spread:.3f}")
            print(f"  Mode consistent: {len(set(modes)) == 1} ({set(modes)})")

        avg_spread = np.mean(spreads)
        avg_consistency = np.mean(mode_consistencies)

        print(f"\nAverage spread: {avg_spread:.3f} (baseline: {self.BASELINE['paraphrase_spread']:.3f})")
        print(f"Mode consistency: {avg_consistency:.1%} (baseline: {self.BASELINE['mode_consistency']:.1%})")

        improvement_spread = (self.BASELINE['paraphrase_spread'] - avg_spread) / self.BASELINE['paraphrase_spread']

        result = ValidationResult(
            test_name="paraphrase_robustness",
            before_value=self.BASELINE['paraphrase_spread'],
            after_value=avg_spread,
            improvement=improvement_spread,
            target_met=avg_spread <= self.TARGETS['paraphrase_spread'],
            details={
                "spreads": spreads,
                "mode_consistency": avg_consistency,
                "mode_consistency_target_met": avg_consistency >= self.TARGETS['mode_consistency']
            }
        )
        self.results.append(result)
        return result

    def test_mode_confidence(self) -> ValidationResult:
        """Test mode classification confidence."""
        print("\n" + "="*60)
        print("TEST: Mode Classification Confidence")
        print("="*60)

        boundary_cases = [
            "I'm not sure if I can make a difference, but maybe together we can try.",
            "The system has problems, but perhaps we can work within it.",
            "I feel somewhat connected to others, though it's complicated.",
            "Life isn't always fair, but we do what we can.",
            "Sometimes I feel powerful, other times helpless.",
            "I want to belong, but I also value my independence.",
            "Things could be better, but they could also be worse.",
            "I believe in fairness, though I've seen it fail.",
        ]

        confidences = []
        prob_gaps = []

        for text in boundary_cases:
            coords, mode = self.project_text(text)
            confidence = mode["confidence"]
            prob_gap = mode["primary_probability"] - mode.get("secondary_probability", 0)

            confidences.append(confidence)
            prob_gaps.append(prob_gap)

            print(f"'{text[:50]}...'")
            print(f"  Confidence: {confidence:.2f}, Gap: {prob_gap:.2%}")

        avg_confidence = np.mean(confidences)
        avg_prob_gap = np.mean(prob_gaps)

        print(f"\nAverage confidence: {avg_confidence:.2f} (baseline: {self.BASELINE['avg_confidence']:.2f})")

        improvement = (avg_confidence - self.BASELINE['avg_confidence']) / self.BASELINE['avg_confidence']

        result = ValidationResult(
            test_name="mode_confidence",
            before_value=self.BASELINE['avg_confidence'],
            after_value=avg_confidence,
            improvement=improvement,
            target_met=avg_confidence >= self.TARGETS['avg_confidence'],
            details={
                "confidences": confidences,
                "avg_prob_gap": avg_prob_gap
            }
        )
        self.results.append(result)
        return result

    def test_axis_disentanglement(self) -> ValidationResult:
        """Test axis disentanglement."""
        print("\n" + "="*60)
        print("TEST: Axis Disentanglement")
        print("="*60)

        axis_tests = {
            "agency": [
                ("I choose to help build a fair community.", "I am forced to participate in a fair community."),
                ("I can decide how to contribute equally.", "I must follow rules about contributing equally."),
                ("We actively create justice together.", "Justice happens to us as a group."),
            ],
            "fairness": [
                ("I choose to treat everyone equally.", "I choose to favor those who deserve it."),
                ("We work together for justice.", "We work together for our own benefit."),
                ("My community values fairness.", "My community values loyalty above all."),
            ],
            "belonging": [
                ("I choose fairness as part of my community.", "I choose fairness as an individual."),
                ("We together pursue justice.", "I alone pursue justice."),
                ("Our group believes in equality.", "I believe in equality."),
            ]
        }

        specificities = []

        for target_axis, pairs in axis_tests.items():
            deltas = []

            for high_stmt, low_stmt in pairs:
                coords_high, _ = self.project_text(high_stmt)
                coords_low, _ = self.project_text(low_stmt)

                delta = {
                    "agency": coords_high.agency - coords_low.agency,
                    "fairness": coords_high.fairness - coords_low.fairness,
                    "belonging": coords_high.belonging - coords_low.belonging
                }
                deltas.append(delta)

            avg_delta = {
                "agency": np.mean([d["agency"] for d in deltas]),
                "fairness": np.mean([d["fairness"] for d in deltas]),
                "belonging": np.mean([d["belonging"] for d in deltas])
            }

            target_delta = abs(avg_delta[target_axis])
            other_deltas = [abs(avg_delta[k]) for k in avg_delta if k != target_axis]
            max_other = max(other_deltas)

            specificity = target_delta / (target_delta + max_other + 0.01)
            specificities.append(specificity)

            print(f"\n{target_axis} manipulation: {specificity:.1%} specific")

        avg_specificity = np.mean(specificities)
        print(f"\nAverage specificity: {avg_specificity:.1%} (baseline: {self.BASELINE['axis_specificity']:.1%})")

        improvement = (avg_specificity - self.BASELINE['axis_specificity']) / self.BASELINE['axis_specificity']

        result = ValidationResult(
            test_name="axis_disentanglement",
            before_value=self.BASELINE['axis_specificity'],
            after_value=avg_specificity,
            improvement=improvement,
            target_met=avg_specificity >= self.TARGETS['axis_specificity'],
            details={"specificities": specificities}
        )
        self.results.append(result)
        return result

    async def test_adversarial_robustness(self) -> ValidationResult:
        """Test adversarial robustness."""
        print("\n" + "="*60)
        print("TEST: Adversarial Robustness")
        print("="*60)

        test_statements = [
            "I believe we can build a fair society together.",
            "The system is designed to keep people down.",
            "Everyone has the power to change their circumstances.",
            "I feel deeply connected to my community.",
            "Life isn't fair, but we adapt and survive."
        ]

        async def project_fn(text: str) -> dict:
            coords, mode = self.project_text(text)
            return {
                "vector": {"agency": coords.agency, "fairness": coords.fairness, "belonging": coords.belonging},
                "mode": mode["primary_mode"]
            }

        robustness_scores = []
        flip_counts = []
        total_perturbations = []

        for stmt in test_statements:
            report = await self.robustness_tester.test_robustness(stmt, project_fn)

            robustness_scores.append(report.robustness_score)
            flip_counts.append(report.mode_flip_count)
            total_perturbations.append(len(report.perturbation_results))

            print(f"'{stmt[:50]}...'")
            print(f"  Robustness: {report.robustness_score:.3f}, Flips: {report.mode_flip_count}/{len(report.perturbation_results)}")

        avg_robustness = np.mean(robustness_scores)
        total_flips = sum(flip_counts)
        total_tests = sum(total_perturbations)
        flip_rate = total_flips / total_tests if total_tests > 0 else 0

        print(f"\nAverage robustness: {avg_robustness:.3f} (baseline: {self.BASELINE['adversarial_robustness']:.3f})")
        print(f"Mode flip rate: {flip_rate:.1%} (baseline: {self.BASELINE['mode_flip_rate']:.1%})")

        improvement = (avg_robustness - self.BASELINE['adversarial_robustness']) / self.BASELINE['adversarial_robustness']

        result = ValidationResult(
            test_name="adversarial_robustness",
            before_value=self.BASELINE['adversarial_robustness'],
            after_value=avg_robustness,
            improvement=improvement,
            target_met=avg_robustness >= self.TARGETS['adversarial_robustness'],
            details={
                "flip_rate": flip_rate,
                "flip_rate_target_met": flip_rate <= self.TARGETS['mode_flip_rate']
            }
        )
        self.results.append(result)
        return result

    def generate_report(self) -> dict:
        """Generate validation report."""
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)

        passed = 0
        failed = 0

        for r in self.results:
            status = "PASS" if r.target_met else "FAIL"
            if r.target_met:
                passed += 1
            else:
                failed += 1

            print(f"\n{r.test_name}:")
            print(f"  Before: {r.before_value:.3f}")
            print(f"  After:  {r.after_value:.3f}")
            print(f"  Change: {r.improvement:+.1%}")
            print(f"  Status: {status}")

        print("\n" + "-"*60)
        print(f"OVERALL: {passed}/{len(self.results)} tests passed")

        def json_serialize(obj):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: json_serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [json_serialize(v) for v in obj]
            return obj

        report = {
            "timestamp": datetime.now().isoformat(),
            "tests_passed": passed,
            "tests_total": len(self.results),
            "results": [
                {
                    "test_name": r.test_name,
                    "before": float(r.before_value),
                    "after": float(r.after_value),
                    "improvement": float(r.improvement),
                    "target_met": bool(r.target_met),
                    "details": json_serialize(r.details)
                }
                for r in self.results
            ]
        }

        # Save report
        output_path = f"data/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")

        return report


async def main():
    """Run validation suite."""
    print("="*60)
    print("VALIDATING OBSERVATORY IMPROVEMENTS")
    print("="*60)

    validator = ImprovementValidator()
    validator.load_projection()

    # Run tests
    validator.test_paraphrase_robustness()
    validator.test_mode_confidence()
    validator.test_axis_disentanglement()
    await validator.test_adversarial_robustness()

    # Generate report
    report = validator.generate_report()

    return report


if __name__ == "__main__":
    asyncio.run(main())
