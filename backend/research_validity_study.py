"""
Scientific Validity Study for Cultural Soliton Observatory

Addresses key concerns from external review:
1. Is the "Fairness" axis actually measuring fairness or system legitimacy?
2. How robust are projections to paraphrasing?
3. How sensitive are mode classifications to boundary proximity?
4. Are axes properly disentangled?

Author: Claude (Research Agent)
Date: 2024
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json
import sys
sys.path.insert(0, '.')

from models import get_model_manager, ModelType
from models.embedding import EmbeddingExtractor
from models.projection import ProjectionTrainer, ProjectionHead, Vector3
from analysis.mode_classifier import get_mode_classifier, EnhancedModeClassifier
from analysis.robustness import get_robustness_tester, PerturbationType


@dataclass
class ExperimentResult:
    """Result from a single experiment."""
    experiment_id: str
    hypothesis: str
    method: str
    data: Dict
    conclusion: str
    confidence: float  # 0-1


@dataclass
class ResearchSession:
    """Container for a full research session."""
    session_id: str
    start_time: str
    experiments: List[ExperimentResult] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)


class ValidityResearcher:
    """
    Conducts validity research on the observatory's measurements.
    """

    def __init__(self):
        self.model_manager = get_model_manager()
        self.embedding_extractor = EmbeddingExtractor(self.model_manager)
        self.classifier = get_mode_classifier()
        self.robustness_tester = get_robustness_tester()
        self.projection_trainer = ProjectionTrainer("data/projections")
        self.projection = None
        self.session = ResearchSession(
            session_id=f"validity_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now().isoformat()
        )

    def load_projection(self):
        """Load the trained projection."""
        self.projection = self.projection_trainer.load_projection()
        if self.projection is None:
            raise RuntimeError("No trained projection found")
        print(f"Loaded projection with {len(self.projection_trainer.examples)} training examples")

    def project_text(self, text: str) -> Tuple[Vector3, Dict]:
        """Project text and get mode classification."""
        if not self.model_manager.is_loaded("all-MiniLM-L6-v2"):
            self.model_manager.load_model("all-MiniLM-L6-v2", ModelType.SENTENCE_TRANSFORMER)

        result = self.embedding_extractor.extract(text, "all-MiniLM-L6-v2")
        coords = self.projection.project(result.embedding)
        coords_array = np.array([coords.agency, coords.fairness, coords.belonging])
        mode_result = self.classifier.classify(coords_array)

        return coords, mode_result

    # =========================================================================
    # EXPERIMENT 1: Axis Validity - Fairness vs System Legitimacy
    # =========================================================================

    def experiment_axis_validity(self) -> ExperimentResult:
        """
        Test whether the "Fairness" axis measures fairness or system legitimacy.

        Key insight from reviewer: Statements like "The system treats everyone fairly"
        might be measuring BELIEF IN SYSTEM LEGITIMACY rather than abstract fairness.

        We test this by comparing:
        - Pure fairness statements (abstract justice)
        - System legitimacy statements (trust in institutions)
        - Anti-system statements (distrust, even if logically about fairness)
        """
        print("\n" + "="*70)
        print("EXPERIMENT 1: Axis Validity - Fairness vs System Legitimacy")
        print("="*70)

        # Test statements designed to disambiguate fairness from system legitimacy
        test_cases = {
            "pure_fairness_positive": [
                "Everyone deserves equal treatment regardless of background.",
                "Justice means treating similar cases similarly.",
                "Fairness requires considering each person's circumstances.",
                "Equal opportunity is a fundamental human right.",
                "Impartiality is essential for ethical decision-making."
            ],
            "pure_fairness_negative": [
                "Some people simply deserve more than others.",
                "Equality is an impossible and naive ideal.",
                "The strong naturally dominate the weak.",
                "Merit alone should determine outcomes, context be damned.",
                "Fairness is just a concept the weak use to constrain the capable."
            ],
            "system_legitimacy_positive": [
                "Our institutions generally work well for most people.",
                "The legal system, while imperfect, delivers justice.",
                "Government policies are designed to help citizens.",
                "Corporations mostly act in good faith.",
                "The system rewards hard work and talent."
            ],
            "system_legitimacy_negative": [
                "The system is rigged against ordinary people.",
                "Institutions serve the powerful, not the public.",
                "Laws are written to protect the wealthy.",
                "The game is fixed before you even start playing.",
                "Those in power will never give it up willingly."
            ],
            "fairness_critique_of_system": [
                # These are about fairness BUT critique the system
                # If axis = fairness: should score HIGH (pro-fairness sentiment)
                # If axis = system legitimacy: should score LOW (anti-system)
                "The system is unfair and must be reformed for true justice.",
                "We need to rebuild institutions to actually serve everyone equally.",
                "Current policies perpetuate inequality; fairness demands change.",
                "True fairness requires dismantling unjust structures.",
                "Fighting for equality means challenging corrupt systems."
            ],
            "system_defense_unfair": [
                # These defend the system BUT acknowledge unfairness
                # If axis = fairness: should score LOW (accepting unfairness)
                # If axis = system legitimacy: should score HIGH (pro-system)
                "Yes the system is imperfect, but it's the best we have.",
                "Some inequality is inevitable; our institutions manage it well.",
                "The system works even if it's not perfectly fair.",
                "Stability matters more than perfect equality.",
                "Our flawed institutions are better than chaos."
            ]
        }

        results = {}

        for category, statements in test_cases.items():
            category_results = []
            for stmt in statements:
                coords, mode = self.project_text(stmt)
                category_results.append({
                    "text": stmt,
                    "agency": coords.agency,
                    "fairness": coords.fairness,
                    "belonging": coords.belonging,
                    "mode": mode["primary_mode"]
                })
            results[category] = category_results

            # Calculate category averages
            avg_fairness = np.mean([r["fairness"] for r in category_results])
            avg_agency = np.mean([r["agency"] for r in category_results])
            print(f"\n{category}:")
            print(f"  Avg Fairness axis: {avg_fairness:.3f}")
            print(f"  Avg Agency axis: {avg_agency:.3f}")
            for r in category_results:
                print(f"    [{r['mode']:20}] F={r['fairness']:+.2f} A={r['agency']:+.2f} | {r['text'][:50]}...")

        # Analyze the critical disambiguation cases
        fairness_critique_avg = np.mean([r["fairness"] for r in results["fairness_critique_of_system"]])
        system_defense_avg = np.mean([r["fairness"] for r in results["system_defense_unfair"]])
        pure_fairness_pos_avg = np.mean([r["fairness"] for r in results["pure_fairness_positive"]])
        system_legit_pos_avg = np.mean([r["fairness"] for r in results["system_legitimacy_positive"]])

        print("\n" + "-"*50)
        print("CRITICAL DISAMBIGUATION TEST:")
        print("-"*50)
        print(f"Pure fairness positive:        {pure_fairness_pos_avg:+.3f}")
        print(f"System legitimacy positive:    {system_legit_pos_avg:+.3f}")
        print(f"Fairness critique of system:   {fairness_critique_avg:+.3f}")
        print(f"System defense (unfair):       {system_defense_avg:+.3f}")

        # Determine what the axis is actually measuring
        # If fairness axis: fairness_critique should be HIGH (pro-fairness)
        # If system legitimacy: fairness_critique should be LOW (anti-system)

        if fairness_critique_avg < 0 and system_defense_avg > 0:
            conclusion = "AXIS MEASURES SYSTEM LEGITIMACY, NOT FAIRNESS"
            interpretation = "Statements critiquing the system score LOW even when advocating fairness. Axis reflects pro-system vs anti-system sentiment."
            confidence = 0.8
        elif fairness_critique_avg > 0 and system_defense_avg < 0:
            conclusion = "AXIS MEASURES FAIRNESS AS LABELED"
            interpretation = "Pro-fairness statements score HIGH regardless of system stance. Axis captures abstract fairness values."
            confidence = 0.8
        else:
            conclusion = "AXIS SHOWS MIXED/ENTANGLED MEASUREMENT"
            interpretation = "The axis conflates fairness with system legitimacy. Both concepts are entangled in the embedding space."
            confidence = 0.6

        print(f"\nCONCLUSION: {conclusion}")
        print(f"Interpretation: {interpretation}")

        return ExperimentResult(
            experiment_id="axis_validity_fairness",
            hypothesis="The 'Fairness' axis may measure system legitimacy rather than abstract fairness",
            method="Compared projections of statements that disambiguate fairness sentiment from system stance",
            data={
                "results": results,
                "critical_test": {
                    "pure_fairness_positive": pure_fairness_pos_avg,
                    "system_legitimacy_positive": system_legit_pos_avg,
                    "fairness_critique_of_system": fairness_critique_avg,
                    "system_defense_unfair": system_defense_avg
                }
            },
            conclusion=f"{conclusion}. {interpretation}",
            confidence=confidence
        )

    # =========================================================================
    # EXPERIMENT 2: Paraphrase Robustness
    # =========================================================================

    async def experiment_paraphrase_robustness(self) -> ExperimentResult:
        """
        Test how stable projections are to meaning-preserving paraphrases.

        Concern: If minor rewording causes large shifts, the projection may be
        capturing surface linguistic features rather than semantic content.
        """
        print("\n" + "="*70)
        print("EXPERIMENT 2: Paraphrase Robustness")
        print("="*70)

        # Groups of semantically equivalent statements
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

        results = []

        for group in paraphrase_groups:
            concept = group["concept"]
            variants = group["variants"]

            projections = []
            for text in variants:
                coords, mode = self.project_text(text)
                projections.append({
                    "text": text,
                    "coords": [coords.agency, coords.fairness, coords.belonging],
                    "mode": mode["primary_mode"]
                })

            # Calculate variance within group
            coords_array = np.array([p["coords"] for p in projections])
            centroid = np.mean(coords_array, axis=0)
            distances = np.linalg.norm(coords_array - centroid, axis=1)
            max_spread = np.max(distances)
            avg_spread = np.mean(distances)
            std_per_axis = np.std(coords_array, axis=0)

            # Check mode consistency
            modes = [p["mode"] for p in projections]
            mode_consistency = len(set(modes)) == 1

            group_result = {
                "concept": concept,
                "projections": projections,
                "centroid": centroid.tolist(),
                "max_spread": float(max_spread),
                "avg_spread": float(avg_spread),
                "std_per_axis": {
                    "agency": float(std_per_axis[0]),
                    "fairness": float(std_per_axis[1]),
                    "belonging": float(std_per_axis[2])
                },
                "mode_consistency": mode_consistency,
                "unique_modes": list(set(modes))
            }
            results.append(group_result)

            print(f"\n{concept}:")
            print(f"  Centroid: A={centroid[0]:.2f}, F={centroid[1]:.2f}, B={centroid[2]:.2f}")
            print(f"  Max spread: {max_spread:.3f}")
            print(f"  Avg spread: {avg_spread:.3f}")
            print(f"  Mode consistency: {mode_consistency} ({set(modes)})")
            print(f"  Std per axis: A={std_per_axis[0]:.3f}, F={std_per_axis[1]:.3f}, B={std_per_axis[2]:.3f}")

        # Overall assessment
        avg_max_spread = np.mean([r["max_spread"] for r in results])
        mode_consistent_count = sum(1 for r in results if r["mode_consistency"])

        print("\n" + "-"*50)
        print("OVERALL ASSESSMENT:")
        print("-"*50)
        print(f"Average max spread within paraphrase groups: {avg_max_spread:.3f}")
        print(f"Groups with consistent mode: {mode_consistent_count}/{len(results)}")

        if avg_max_spread < 0.3:
            robustness = "HIGH"
            confidence = 0.8
        elif avg_max_spread < 0.6:
            robustness = "MODERATE"
            confidence = 0.6
        else:
            robustness = "LOW"
            confidence = 0.4

        conclusion = f"Paraphrase robustness is {robustness}. Average spread={avg_max_spread:.3f}, Mode consistency={mode_consistent_count}/{len(results)}"
        print(f"\nCONCLUSION: {conclusion}")

        return ExperimentResult(
            experiment_id="paraphrase_robustness",
            hypothesis="Semantically equivalent paraphrases should project to similar locations",
            method="Projected groups of paraphrased statements and measured within-group variance",
            data={"groups": results, "avg_max_spread": avg_max_spread, "mode_consistent": mode_consistent_count},
            conclusion=conclusion,
            confidence=confidence
        )

    # =========================================================================
    # EXPERIMENT 3: Mode Boundary Sensitivity
    # =========================================================================

    def experiment_mode_boundaries(self) -> ExperimentResult:
        """
        Test how sensitive mode classification is to small perturbations near boundaries.

        Concern: If mode classifications flip easily, the discrete categories may
        be imposing artificial structure on a continuous space.
        """
        print("\n" + "="*70)
        print("EXPERIMENT 3: Mode Boundary Sensitivity")
        print("="*70)

        # Statements likely to be near mode boundaries (ambiguous cases)
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

        results = []

        for text in boundary_cases:
            coords, mode = self.project_text(text)

            # Check confidence and boundary distances
            confidence = mode["confidence"]
            primary_prob = mode["primary_probability"]
            secondary_mode = mode.get("secondary_mode", "N/A")
            secondary_prob = mode.get("secondary_probability", 0)

            # Probability gap between primary and secondary
            prob_gap = primary_prob - secondary_prob

            result = {
                "text": text,
                "coords": {"agency": coords.agency, "fairness": coords.fairness, "belonging": coords.belonging},
                "primary_mode": mode["primary_mode"],
                "primary_prob": primary_prob,
                "secondary_mode": secondary_mode,
                "secondary_prob": secondary_prob,
                "prob_gap": prob_gap,
                "confidence": confidence
            }
            results.append(result)

            print(f"\n'{text[:60]}...'")
            print(f"  Primary: {mode['primary_mode']} ({primary_prob:.2%})")
            print(f"  Secondary: {secondary_mode} ({secondary_prob:.2%})")
            print(f"  Probability gap: {prob_gap:.2%}")
            print(f"  Confidence: {confidence:.2f}")

        # Analyze boundary sensitivity
        avg_prob_gap = np.mean([r["prob_gap"] for r in results])
        avg_confidence = np.mean([r["confidence"] for r in results])
        low_confidence_count = sum(1 for r in results if r["confidence"] < 0.5)

        print("\n" + "-"*50)
        print("BOUNDARY ANALYSIS:")
        print("-"*50)
        print(f"Average probability gap (primary vs secondary): {avg_prob_gap:.2%}")
        print(f"Average confidence: {avg_confidence:.2f}")
        print(f"Low confidence cases (<0.5): {low_confidence_count}/{len(results)}")

        if avg_prob_gap > 0.3:
            sensitivity = "LOW (classifications are confident)"
        elif avg_prob_gap > 0.15:
            sensitivity = "MODERATE (some boundary ambiguity)"
        else:
            sensitivity = "HIGH (many borderline cases)"

        conclusion = f"Mode boundary sensitivity is {sensitivity}. Avg prob gap={avg_prob_gap:.2%}, Avg confidence={avg_confidence:.2f}"
        print(f"\nCONCLUSION: {conclusion}")

        return ExperimentResult(
            experiment_id="mode_boundary_sensitivity",
            hypothesis="Mode classifications near boundaries should show appropriate uncertainty",
            method="Analyzed probability gaps and confidence for ambiguous statements",
            data={"cases": results, "avg_prob_gap": avg_prob_gap, "avg_confidence": avg_confidence},
            conclusion=conclusion,
            confidence=0.7
        )

    # =========================================================================
    # EXPERIMENT 4: Axis Disentanglement
    # =========================================================================

    def experiment_axis_disentanglement(self) -> ExperimentResult:
        """
        Test whether axes are properly disentangled or show unwanted correlations.

        Concern: If changing agency-related language also changes fairness scores,
        the axes may not be independently measuring their intended constructs.
        """
        print("\n" + "="*70)
        print("EXPERIMENT 4: Axis Disentanglement")
        print("="*70)

        # Base statements with targeted variations
        axis_tests = {
            "agency_manipulation": {
                "description": "Vary agency while holding fairness/belonging constant",
                "pairs": [
                    ("I choose to help build a fair community.", "I am forced to participate in a fair community."),
                    ("I can decide how to contribute equally.", "I must follow rules about contributing equally."),
                    ("We actively create justice together.", "Justice happens to us as a group."),
                    ("I determine my path to fairness.", "My path to fairness is determined for me."),
                ]
            },
            "fairness_manipulation": {
                "description": "Vary fairness while holding agency/belonging constant",
                "pairs": [
                    ("I choose to treat everyone equally.", "I choose to favor those who deserve it."),
                    ("We work together for justice.", "We work together for our own benefit."),
                    ("My community values fairness.", "My community values loyalty above all."),
                    ("Together we build an equitable society.", "Together we build a strong hierarchy."),
                ]
            },
            "belonging_manipulation": {
                "description": "Vary belonging while holding agency/fairness constant",
                "pairs": [
                    ("I choose fairness as part of my community.", "I choose fairness as an individual."),
                    ("We together pursue justice.", "I alone pursue justice."),
                    ("Our group believes in equality.", "I believe in equality."),
                    ("As a community, we decide fairly.", "As an outsider, I demand fairness."),
                ]
            }
        }

        results = {}

        for test_name, test_data in axis_tests.items():
            target_axis = test_name.split("_")[0]  # agency, fairness, or belonging

            print(f"\n{test_data['description']}:")
            print("-" * 50)

            deltas = []
            for high_stmt, low_stmt in test_data["pairs"]:
                coords_high, _ = self.project_text(high_stmt)
                coords_low, _ = self.project_text(low_stmt)

                delta = {
                    "agency": coords_high.agency - coords_low.agency,
                    "fairness": coords_high.fairness - coords_low.fairness,
                    "belonging": coords_high.belonging - coords_low.belonging
                }
                deltas.append(delta)

                print(f"\n  HIGH: {high_stmt[:50]}...")
                print(f"  LOW:  {low_stmt[:50]}...")
                print(f"  Delta: A={delta['agency']:+.3f}, F={delta['fairness']:+.3f}, B={delta['belonging']:+.3f}")

            # Calculate average deltas
            avg_delta = {
                "agency": np.mean([d["agency"] for d in deltas]),
                "fairness": np.mean([d["fairness"] for d in deltas]),
                "belonging": np.mean([d["belonging"] for d in deltas])
            }

            # Check if manipulation affected target axis most
            target_delta = abs(avg_delta[target_axis])
            other_deltas = [abs(avg_delta[k]) for k in avg_delta if k != target_axis]
            max_other = max(other_deltas)

            specificity = target_delta / (target_delta + max_other + 0.01)

            results[test_name] = {
                "deltas": deltas,
                "avg_delta": avg_delta,
                "target_axis": target_axis,
                "target_delta": target_delta,
                "max_other_delta": max_other,
                "specificity": specificity
            }

            print(f"\n  Average delta: A={avg_delta['agency']:+.3f}, F={avg_delta['fairness']:+.3f}, B={avg_delta['belonging']:+.3f}")
            print(f"  Target axis ({target_axis}) specificity: {specificity:.2%}")

        # Overall disentanglement assessment
        avg_specificity = np.mean([r["specificity"] for r in results.values()])

        print("\n" + "-"*50)
        print("DISENTANGLEMENT ASSESSMENT:")
        print("-"*50)
        for test_name, r in results.items():
            print(f"  {test_name}: {r['specificity']:.1%} specific to target")
        print(f"\n  Average specificity: {avg_specificity:.1%}")

        if avg_specificity > 0.7:
            quality = "WELL DISENTANGLED"
            confidence = 0.8
        elif avg_specificity > 0.5:
            quality = "PARTIALLY DISENTANGLED"
            confidence = 0.6
        else:
            quality = "POORLY DISENTANGLED (axes are entangled)"
            confidence = 0.5

        conclusion = f"Axes are {quality}. Average manipulation specificity: {avg_specificity:.1%}"
        print(f"\nCONCLUSION: {conclusion}")

        return ExperimentResult(
            experiment_id="axis_disentanglement",
            hypothesis="Manipulating one axis should not significantly affect others",
            method="Applied targeted linguistic manipulations and measured cross-axis effects",
            data=results,
            conclusion=conclusion,
            confidence=confidence
        )

    # =========================================================================
    # EXPERIMENT 5: Adversarial Robustness
    # =========================================================================

    async def experiment_adversarial_robustness(self) -> ExperimentResult:
        """
        Use the robustness tester to find vulnerabilities in the projection.
        """
        print("\n" + "="*70)
        print("EXPERIMENT 5: Adversarial Robustness Testing")
        print("="*70)

        test_statements = [
            "I believe we can build a fair society together.",
            "The system is designed to keep people down.",
            "Everyone has the power to change their circumstances.",
            "I feel deeply connected to my community.",
            "Life isn't fair, but we adapt and survive."
        ]

        results = []

        async def project_fn(text: str) -> dict:
            coords, mode = self.project_text(text)
            return {
                "vector": {"agency": coords.agency, "fairness": coords.fairness, "belonging": coords.belonging},
                "mode": mode["primary_mode"]
            }

        for stmt in test_statements:
            print(f"\nTesting: '{stmt[:50]}...'")

            report = await self.robustness_tester.test_robustness(stmt, project_fn)

            result = {
                "text": stmt,
                "robustness_score": report.robustness_score,
                "mode_flip_count": report.mode_flip_count,
                "perturbations_tested": len(report.perturbation_results),
                "most_sensitive": None
            }

            if report.most_sensitive_perturbation:
                msp = report.most_sensitive_perturbation
                result["most_sensitive"] = {
                    "type": msp.perturbation.perturbation_type.value,
                    "description": msp.perturbation.description,
                    "delta_magnitude": sum(abs(d) for d in msp.delta.values())
                }

            results.append(result)

            print(f"  Robustness score: {report.robustness_score:.3f}")
            print(f"  Mode flips: {report.mode_flip_count}/{len(report.perturbation_results)}")
            if result["most_sensitive"]:
                print(f"  Most sensitive: {result['most_sensitive']['type']} - {result['most_sensitive']['description']}")

        # Overall assessment
        avg_robustness = np.mean([r["robustness_score"] for r in results])
        total_flips = sum(r["mode_flip_count"] for r in results)
        total_tests = sum(r["perturbations_tested"] for r in results)

        print("\n" + "-"*50)
        print("ROBUSTNESS SUMMARY:")
        print("-"*50)
        print(f"Average robustness score: {avg_robustness:.3f}")
        print(f"Total mode flips: {total_flips}/{total_tests} ({100*total_flips/total_tests:.1f}%)")

        if avg_robustness > 0.8:
            quality = "HIGH ROBUSTNESS"
        elif avg_robustness > 0.6:
            quality = "MODERATE ROBUSTNESS"
        else:
            quality = "LOW ROBUSTNESS (vulnerable to perturbations)"

        conclusion = f"Projection shows {quality}. Avg score={avg_robustness:.3f}, Mode flip rate={100*total_flips/total_tests:.1f}%"
        print(f"\nCONCLUSION: {conclusion}")

        return ExperimentResult(
            experiment_id="adversarial_robustness",
            hypothesis="Projections should be stable under meaning-preserving perturbations",
            method="Applied 6 types of perturbations and measured projection stability",
            data={"results": results, "avg_robustness": avg_robustness, "flip_rate": total_flips/total_tests},
            conclusion=conclusion,
            confidence=0.75
        )

    # =========================================================================
    # Main Research Session
    # =========================================================================

    async def run_full_study(self):
        """Run the complete validity study."""
        print("\n" + "="*70)
        print("CULTURAL SOLITON OBSERVATORY - SCIENTIFIC VALIDITY STUDY")
        print("="*70)
        print(f"Session ID: {self.session.session_id}")
        print(f"Start time: {self.session.start_time}")
        print("="*70)

        # Load projection
        self.load_projection()

        # Run experiments
        print("\n[Running Experiment 1/5: Axis Validity]")
        exp1 = self.experiment_axis_validity()
        self.session.experiments.append(exp1)

        print("\n[Running Experiment 2/5: Paraphrase Robustness]")
        exp2 = await self.experiment_paraphrase_robustness()
        self.session.experiments.append(exp2)

        print("\n[Running Experiment 3/5: Mode Boundaries]")
        exp3 = self.experiment_mode_boundaries()
        self.session.experiments.append(exp3)

        print("\n[Running Experiment 4/5: Axis Disentanglement]")
        exp4 = self.experiment_axis_disentanglement()
        self.session.experiments.append(exp4)

        print("\n[Running Experiment 5/5: Adversarial Robustness]")
        exp5 = await self.experiment_adversarial_robustness()
        self.session.experiments.append(exp5)

        # Generate summary
        self._generate_summary()

        return self.session

    def _generate_summary(self):
        """Generate final summary and recommendations."""
        print("\n" + "="*70)
        print("VALIDITY STUDY SUMMARY")
        print("="*70)

        findings = []

        for exp in self.session.experiments:
            print(f"\n{exp.experiment_id}:")
            print(f"  Conclusion: {exp.conclusion}")
            print(f"  Confidence: {exp.confidence:.0%}")
            findings.append(f"[{exp.experiment_id}] {exp.conclusion} (confidence: {exp.confidence:.0%})")

        self.session.findings = findings

        print("\n" + "="*70)
        print("RECOMMENDATIONS FOR IMPROVEMENT")
        print("="*70)

        recommendations = [
            "1. Consider renaming 'Fairness' axis to 'System Legitimacy' if validation confirms conflation",
            "2. Add confidence intervals to all projections using ensemble methods",
            "3. Report mode classification confidence alongside discrete labels",
            "4. Expand training data beyond 64 examples to improve robustness",
            "5. Consider disentanglement training to reduce axis correlations",
            "6. Add paraphrase augmentation to training data for stability"
        ]

        for rec in recommendations:
            print(rec)

        # Save results
        output_path = f"data/validity_study_{self.session.session_id}.json"
        with open(output_path, 'w') as f:
            json.dump({
                "session_id": self.session.session_id,
                "start_time": self.session.start_time,
                "experiments": [
                    {
                        "id": e.experiment_id,
                        "hypothesis": e.hypothesis,
                        "method": e.method,
                        "conclusion": e.conclusion,
                        "confidence": e.confidence
                    }
                    for e in self.session.experiments
                ],
                "findings": self.session.findings,
                "recommendations": recommendations
            }, f, indent=2)

        print(f"\nResults saved to: {output_path}")


async def main():
    researcher = ValidityResearcher()
    session = await researcher.run_full_study()
    return session


if __name__ == "__main__":
    asyncio.run(main())
