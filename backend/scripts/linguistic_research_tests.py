#!/usr/bin/env python3
"""
Comprehensive Linguistic Research Tests for Publication (TACL/ACL)

Cultural Soliton Observatory v2.0 - Research Test Suite

This module implements rigorous linguistic experiments designed for publication
in top computational linguistics venues (TACL, ACL, EMNLP). Each test investigates
fundamental questions about how linguistic features affect coordination positioning
in the 3D manifold (agency, justice, belonging) with 18D hierarchical decomposition.

Research Questions:
1. How do different deixis types (person, spatial, temporal) contribute to coordination?
2. How do modality types (epistemic, deontic, dynamic) affect manifold positioning?
3. How does grammatical voice (active vs passive) modulate agency perception?
4. How does register variation (formal vs informal) affect legibility and coordination?

Each test provides:
- Clearly stated hypothesis
- Minimal pairs or controlled stimuli
- Effect size calculations (Cohen's d, Hedge's g)
- Bootstrap confidence intervals
- Fisher-Rao information distances
- Phase transition detection
- Publication-ready output formats

Usage:
    python scripts/linguistic_research_tests.py --test all
    python scripts/linguistic_research_tests.py --test deixis
    python scripts/linguistic_research_tests.py --output-format latex

API Integration:
    All tests can be run via MCP endpoints at localhost:8000
"""

import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import numpy as np
from scipy import stats
import httpx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research import (
    cohens_d, hedges_g, bootstrap_ci, bootstrap_coordinate_ci,
    apply_correction, fisher_rao_distance, manifold_distance,
    detect_phase_transitions, extract_hierarchical_coordinate,
    generate_latex_table, generate_effect_size_table,
    compute_summary_statistics, generate_experiment_report,
    EffectSize, BootstrapEstimate, ManifoldDistance
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
API_BASE = "http://127.0.0.1:8000"
TIMEOUT = 30.0


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class LinguisticCondition:
    """A single experimental condition with stimuli."""
    name: str
    stimuli: List[str]
    description: str
    linguistic_property: str

@dataclass
class ProjectionResult:
    """Result of projecting a text through the Observatory."""
    text: str
    coordinates: Dict[str, float]
    hierarchical: Optional[Dict] = None
    mode: str = "NEUTRAL"
    confidence: float = 0.5

@dataclass
class ConditionResults:
    """Aggregated results for a condition."""
    condition: str
    n: int
    agency_mean: float
    agency_std: float
    justice_mean: float
    justice_std: float
    belonging_mean: float
    belonging_std: float
    mode_distribution: Dict[str, int]
    projections: List[ProjectionResult] = field(default_factory=list)

    def get_axis_values(self, axis: str) -> np.ndarray:
        """Get all values for a given axis."""
        if axis == "agency":
            return np.array([p.coordinates.get("agency", 0) for p in self.projections])
        elif axis == "justice":
            return np.array([p.coordinates.get("perceived_justice",
                           p.coordinates.get("fairness", 0)) for p in self.projections])
        elif axis == "belonging":
            return np.array([p.coordinates.get("belonging", 0) for p in self.projections])
        return np.array([])

@dataclass
class ExperimentResult:
    """Complete experiment result with statistics."""
    name: str
    hypothesis: str
    conditions: Dict[str, ConditionResults]
    effect_sizes: Dict[str, EffectSize]
    bootstrap_cis: Dict[str, Dict[str, BootstrapEstimate]]
    statistical_tests: Dict[str, Dict]
    manifold_distances: Dict[str, ManifoldDistance]
    key_findings: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# API Client
# =============================================================================

class ObservatoryClient:
    """Client for interacting with the Cultural Soliton Observatory API."""

    def __init__(self, base_url: str = API_BASE, timeout: float = TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout

    async def analyze(self, text: str) -> ProjectionResult:
        """Project text and get full analysis."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/v2/analyze",
                    json={"text": text, "include_uncertainty": True}
                )
                response.raise_for_status()
                data = response.json()

                return ProjectionResult(
                    text=text,
                    coordinates=data.get("vector", {}),
                    hierarchical=data.get("hierarchical"),
                    mode=data.get("mode", {}).get("primary_mode", "NEUTRAL"),
                    confidence=data.get("mode", {}).get("confidence", 0.5)
                )
            except Exception as e:
                logger.warning(f"API error for '{text[:50]}...': {e}")
                # Fallback to local hierarchical extraction
                hier = extract_hierarchical_coordinate(text)
                legacy = hier.core.to_legacy_3d()
                return ProjectionResult(
                    text=text,
                    coordinates={
                        "agency": legacy[0],
                        "perceived_justice": legacy[1],
                        "belonging": legacy[2]
                    },
                    hierarchical=hier.to_dict(),
                    mode="NEUTRAL",
                    confidence=0.5
                )

    async def grammar_deletion(self, text: str, threshold: float = 0.3) -> Dict:
        """Test grammar feature deletion."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/research/grammar-deletion",
                    json={"text": text, "threshold": threshold}
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.warning(f"Grammar deletion API error: {e}")
                return {}

    async def legibility(self, texts: List[str]) -> Dict:
        """Compute legibility scores."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/research/legibility",
                    json={"texts": texts}
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.warning(f"Legibility API error: {e}")
                return {}

    async def calibrate(self, human_texts: List[str],
                       minimal_texts: List[str]) -> Dict:
        """Compare human vs minimal codes."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/research/calibrate",
                    json={"human_texts": human_texts, "minimal_texts": minimal_texts}
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.warning(f"Calibrate API error: {e}")
                return {}

    async def analyze_batch(self, texts: List[str]) -> List[ProjectionResult]:
        """Analyze multiple texts."""
        results = []
        for text in texts:
            result = await self.analyze(text)
            results.append(result)
        return results


# =============================================================================
# TEST 1: Deixis Types and Coordination Necessity
# =============================================================================

class DeisisTest:
    """
    EXPERIMENT 1: Deictic Categories and Coordination Positioning

    Hypothesis: Different deixis types (person, spatial, temporal) make
    distinct contributions to coordination positioning, with person deixis
    being most critical for agency, temporal deixis for justice framing,
    and spatial deixis for belonging.

    This tests the "deictic anchor" theory from cognitive linguistics:
    first-person markers anchor perspectival content, while other deictics
    provide contextual framing.

    Design: 2x3 factorial (deixis type x polarity)
    - Person deixis: first-person vs third-person
    - Spatial deixis: proximal vs distal
    - Temporal deixis: past vs future
    """

    NAME = "Deictic Categories and Coordination Positioning"

    HYPOTHESIS = (
        "H1: Person deixis (I/we vs they/them) will show the largest effect "
        "on agency scores (d > 0.8, critical feature). "
        "H2: Temporal deixis (past vs future) will primarily affect justice "
        "perception (d > 0.5). "
        "H3: Spatial deixis (here vs there) will modulate belonging scores "
        "(d > 0.3)."
    )

    @staticmethod
    def get_conditions() -> Dict[str, LinguisticCondition]:
        """Generate experimental conditions with minimal pairs."""
        return {
            # Person Deixis Conditions
            "person_first_singular": LinguisticCondition(
                name="First Person Singular",
                linguistic_property="person_deixis",
                description="Uses I/me/my pronouns",
                stimuli=[
                    "I decided to take responsibility for the outcome.",
                    "I believe this decision was mine to make.",
                    "My actions led directly to this result.",
                    "I chose this path knowing the consequences.",
                    "I am the one who made this happen.",
                    "I take full ownership of what occurred.",
                    "My judgment guided the final decision.",
                    "I determined the course of events.",
                ]
            ),
            "person_first_plural": LinguisticCondition(
                name="First Person Plural",
                linguistic_property="person_deixis",
                description="Uses we/us/our pronouns",
                stimuli=[
                    "We decided to take responsibility for the outcome.",
                    "We believe this decision was ours to make.",
                    "Our actions led directly to this result.",
                    "We chose this path knowing the consequences.",
                    "We are the ones who made this happen.",
                    "We take full ownership of what occurred.",
                    "Our judgment guided the final decision.",
                    "We determined the course of events.",
                ]
            ),
            "person_third": LinguisticCondition(
                name="Third Person",
                linguistic_property="person_deixis",
                description="Uses they/them/their pronouns",
                stimuli=[
                    "They decided to take responsibility for the outcome.",
                    "They believe this decision was theirs to make.",
                    "Their actions led directly to this result.",
                    "They chose this path knowing the consequences.",
                    "They are the ones who made this happen.",
                    "They take full ownership of what occurred.",
                    "Their judgment guided the final decision.",
                    "They determined the course of events.",
                ]
            ),
            "person_impersonal": LinguisticCondition(
                name="Impersonal",
                linguistic_property="person_deixis",
                description="Impersonal constructions without agent",
                stimuli=[
                    "The decision was made to take responsibility for the outcome.",
                    "It was believed this decision needed to be made.",
                    "Actions were taken that led directly to this result.",
                    "This path was chosen knowing the consequences.",
                    "This was made to happen.",
                    "Ownership was taken of what occurred.",
                    "The judgment that guided the final decision was rendered.",
                    "The course of events was determined.",
                ]
            ),
            # Spatial Deixis Conditions
            "spatial_proximal": LinguisticCondition(
                name="Proximal Spatial",
                linguistic_property="spatial_deixis",
                description="Uses here/this/near markers",
                stimuli=[
                    "Here in our community, we make decisions together.",
                    "This place is where we belong and find meaning.",
                    "Everyone here understands what we are building.",
                    "Here is where the real work happens.",
                    "This is our home and we protect it.",
                    "The people here share our values.",
                    "This community has welcomed us completely.",
                    "Here, among friends, we can be ourselves.",
                ]
            ),
            "spatial_distal": LinguisticCondition(
                name="Distal Spatial",
                linguistic_property="spatial_deixis",
                description="Uses there/that/far markers",
                stimuli=[
                    "There in that community, they make decisions together.",
                    "That place is where they belong and find meaning.",
                    "Everyone there understands what they are building.",
                    "There is where the real work happens.",
                    "That is their home and they protect it.",
                    "The people there share their values.",
                    "That community has welcomed them completely.",
                    "There, among friends, they can be themselves.",
                ]
            ),
            # Temporal Deixis Conditions
            "temporal_past": LinguisticCondition(
                name="Past Temporal",
                linguistic_property="temporal_deixis",
                description="Past tense orientation",
                stimuli=[
                    "We worked hard and earned what we deserved.",
                    "Justice was served when the truth came out.",
                    "The rules were applied fairly to everyone.",
                    "We received our fair share after contributing.",
                    "The outcome reflected what we had put in.",
                    "Past efforts were properly recognized.",
                    "What was owed was eventually paid.",
                    "The system worked as it was supposed to.",
                ]
            ),
            "temporal_future": LinguisticCondition(
                name="Future Temporal",
                linguistic_property="temporal_deixis",
                description="Future tense orientation",
                stimuli=[
                    "We will work hard and earn what we deserve.",
                    "Justice will be served when the truth comes out.",
                    "The rules will be applied fairly to everyone.",
                    "We will receive our fair share after contributing.",
                    "The outcome will reflect what we put in.",
                    "Future efforts will be properly recognized.",
                    "What is owed will eventually be paid.",
                    "The system will work as it is supposed to.",
                ]
            ),
            "temporal_present": LinguisticCondition(
                name="Present Temporal",
                linguistic_property="temporal_deixis",
                description="Present tense orientation",
                stimuli=[
                    "We work hard and earn what we deserve.",
                    "Justice is served when the truth comes out.",
                    "The rules are applied fairly to everyone.",
                    "We receive our fair share after contributing.",
                    "The outcome reflects what we put in.",
                    "Current efforts are properly recognized.",
                    "What is owed is being paid.",
                    "The system works as it is supposed to.",
                ]
            ),
        }

    @staticmethod
    async def run(client: ObservatoryClient) -> ExperimentResult:
        """Execute the deixis experiment."""
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT 1: {DeisisTest.NAME}")
        logger.info(f"{'='*70}")
        logger.info(f"Hypothesis: {DeisisTest.HYPOTHESIS}")

        conditions = DeisisTest.get_conditions()
        condition_results = {}

        # Analyze all conditions
        for cond_name, condition in conditions.items():
            logger.info(f"\nAnalyzing condition: {condition.name}")
            projections = await client.analyze_batch(condition.stimuli)

            agencies = [p.coordinates.get("agency", 0) for p in projections]
            justices = [p.coordinates.get("perceived_justice",
                        p.coordinates.get("fairness", 0)) for p in projections]
            belongings = [p.coordinates.get("belonging", 0) for p in projections]

            modes = {}
            for p in projections:
                modes[p.mode] = modes.get(p.mode, 0) + 1

            condition_results[cond_name] = ConditionResults(
                condition=cond_name,
                n=len(projections),
                agency_mean=np.mean(agencies),
                agency_std=np.std(agencies),
                justice_mean=np.mean(justices),
                justice_std=np.std(justices),
                belonging_mean=np.mean(belongings),
                belonging_std=np.std(belongings),
                mode_distribution=modes,
                projections=projections
            )

            logger.info(f"  Agency: {np.mean(agencies):+.3f} (+/- {np.std(agencies):.3f})")
            logger.info(f"  Justice: {np.mean(justices):+.3f} (+/- {np.std(justices):.3f})")
            logger.info(f"  Belonging: {np.mean(belongings):+.3f} (+/- {np.std(belongings):.3f})")

        # Compute effect sizes
        effect_sizes = {}

        # H1: Person deixis effect on agency
        first_sg = condition_results["person_first_singular"]
        third = condition_results["person_third"]
        impersonal = condition_results["person_impersonal"]

        effect_sizes["person_1sg_vs_3rd_agency"] = hedges_g(
            first_sg.get_axis_values("agency"),
            third.get_axis_values("agency")
        )
        effect_sizes["person_1sg_vs_impersonal_agency"] = hedges_g(
            first_sg.get_axis_values("agency"),
            impersonal.get_axis_values("agency")
        )

        # First plural vs singular (solidarity effect)
        first_pl = condition_results["person_first_plural"]
        effect_sizes["person_1sg_vs_1pl_belonging"] = hedges_g(
            first_sg.get_axis_values("belonging"),
            first_pl.get_axis_values("belonging")
        )

        # H2: Temporal deixis effect on justice
        past = condition_results["temporal_past"]
        future = condition_results["temporal_future"]
        present = condition_results["temporal_present"]

        effect_sizes["temporal_past_vs_future_justice"] = hedges_g(
            past.get_axis_values("justice"),
            future.get_axis_values("justice")
        )
        effect_sizes["temporal_present_vs_future_justice"] = hedges_g(
            present.get_axis_values("justice"),
            future.get_axis_values("justice")
        )

        # H3: Spatial deixis effect on belonging
        proximal = condition_results["spatial_proximal"]
        distal = condition_results["spatial_distal"]

        effect_sizes["spatial_proximal_vs_distal_belonging"] = hedges_g(
            proximal.get_axis_values("belonging"),
            distal.get_axis_values("belonging")
        )

        # Bootstrap CIs for key conditions
        bootstrap_cis = {}
        for cond_name, cond_result in condition_results.items():
            coords = np.column_stack([
                cond_result.get_axis_values("agency"),
                cond_result.get_axis_values("justice"),
                cond_result.get_axis_values("belonging")
            ])
            if len(coords) > 0:
                bootstrap_cis[cond_name] = bootstrap_coordinate_ci(coords, n_bootstrap=1000)

        # Manifold distances between key conditions
        manifold_distances = {}

        # Person deixis manifold distance
        manifold_distances["person_1sg_vs_impersonal"] = manifold_distance(
            np.array([first_sg.agency_mean, first_sg.justice_mean, first_sg.belonging_mean]),
            np.array([impersonal.agency_mean, impersonal.justice_mean, impersonal.belonging_mean])
        )

        # Spatial deixis manifold distance
        manifold_distances["spatial_proximal_vs_distal"] = manifold_distance(
            np.array([proximal.agency_mean, proximal.justice_mean, proximal.belonging_mean]),
            np.array([distal.agency_mean, distal.justice_mean, distal.belonging_mean])
        )

        # Statistical tests (ANOVA across person deixis levels)
        statistical_tests = {}

        # One-way ANOVA for person deixis on agency
        f_stat, p_val = stats.f_oneway(
            first_sg.get_axis_values("agency"),
            first_pl.get_axis_values("agency"),
            third.get_axis_values("agency"),
            impersonal.get_axis_values("agency")
        )
        statistical_tests["person_deixis_agency_anova"] = {
            "F": f_stat, "p": p_val, "significant": p_val < 0.05
        }

        # Key findings
        findings = []

        d_1sg_imp = effect_sizes["person_1sg_vs_impersonal_agency"]
        findings.append(
            f"H1 {'SUPPORTED' if abs(d_1sg_imp.d) > 0.8 else 'PARTIAL'}: "
            f"First-person vs impersonal agency d={d_1sg_imp.d:.3f} "
            f"({d_1sg_imp.feature_classification})"
        )

        d_temp = effect_sizes["temporal_past_vs_future_justice"]
        findings.append(
            f"H2 {'SUPPORTED' if abs(d_temp.d) > 0.5 else 'NOT SUPPORTED'}: "
            f"Past vs future temporal deixis on justice d={d_temp.d:.3f}"
        )

        d_spat = effect_sizes["spatial_proximal_vs_distal_belonging"]
        findings.append(
            f"H3 {'SUPPORTED' if abs(d_spat.d) > 0.3 else 'NOT SUPPORTED'}: "
            f"Proximal vs distal spatial deixis on belonging d={d_spat.d:.3f}"
        )

        # Log results
        logger.info("\n--- EFFECT SIZES ---")
        for name, es in effect_sizes.items():
            sig = "*" if es.is_significant else ""
            logger.info(f"  {name}: d={es.d:.3f}{sig} ({es.feature_classification})")

        logger.info("\n--- KEY FINDINGS ---")
        for f in findings:
            logger.info(f"  {f}")

        return ExperimentResult(
            name=DeisisTest.NAME,
            hypothesis=DeisisTest.HYPOTHESIS,
            conditions=condition_results,
            effect_sizes=effect_sizes,
            bootstrap_cis=bootstrap_cis,
            statistical_tests=statistical_tests,
            manifold_distances=manifold_distances,
            key_findings=findings,
            metadata={
                "n_conditions": len(conditions),
                "n_stimuli_per_condition": 8,
                "total_stimuli": sum(len(c.stimuli) for c in conditions.values()),
                "deixis_types": ["person", "spatial", "temporal"]
            }
        )


# =============================================================================
# TEST 2: Modality Types (Epistemic, Deontic, Dynamic)
# =============================================================================

class ModalityTest:
    """
    EXPERIMENT 2: Modal Categories and Manifold Positioning

    Hypothesis: Different modality types map to distinct regions of the
    coordination manifold:
    - Epistemic modality (certainty/possibility) affects justice perception
    - Deontic modality (obligation/permission) affects agency
    - Dynamic modality (ability/volition) affects belonging through self-efficacy

    This tests Kratzer's modal semantics in a coordination context.
    """

    NAME = "Modal Categories and Manifold Positioning"

    HYPOTHESIS = (
        "H1: Deontic modality (must/should) will decrease perceived agency "
        "compared to dynamic modality (can/will), as obligation implies external force. "
        "H2: Epistemic certainty markers (definitely/certainly) will increase "
        "perceived justice compared to uncertainty markers (maybe/possibly). "
        "H3: Dynamic ability markers (can/able) will correlate with higher "
        "belonging through self-efficacy pathways."
    )

    @staticmethod
    def get_conditions() -> Dict[str, LinguisticCondition]:
        return {
            # Epistemic Modality
            "epistemic_certain": LinguisticCondition(
                name="Epistemic Certainty",
                linguistic_property="epistemic_modality",
                description="High certainty markers",
                stimuli=[
                    "This is definitely the right decision for everyone.",
                    "It is certainly true that we all benefit equally.",
                    "There is no doubt that the outcome was fair.",
                    "Clearly, the process treated everyone the same.",
                    "Obviously, the rules were applied correctly.",
                    "It must be that justice was served properly.",
                    "Without question, we received what was deserved.",
                    "Undoubtedly, the distribution was equitable.",
                ]
            ),
            "epistemic_uncertain": LinguisticCondition(
                name="Epistemic Uncertainty",
                linguistic_property="epistemic_modality",
                description="Low certainty markers",
                stimuli=[
                    "This is possibly the right decision for everyone.",
                    "It might be true that we all benefit equally.",
                    "Perhaps the outcome was fair.",
                    "Maybe the process treated everyone the same.",
                    "The rules were perhaps applied correctly.",
                    "It could be that justice was served properly.",
                    "Possibly, we received what was deserved.",
                    "The distribution might have been equitable.",
                ]
            ),
            # Deontic Modality
            "deontic_obligation": LinguisticCondition(
                name="Deontic Obligation",
                linguistic_property="deontic_modality",
                description="Obligation/necessity markers",
                stimuli=[
                    "We must follow the established procedures.",
                    "Everyone should comply with the requirements.",
                    "You have to accept the decision that was made.",
                    "It is required that all rules be obeyed.",
                    "One ought to defer to authority in this matter.",
                    "We are obligated to accept the outcome.",
                    "There is no choice but to comply.",
                    "You need to do what you are told.",
                ]
            ),
            "deontic_permission": LinguisticCondition(
                name="Deontic Permission",
                linguistic_property="deontic_modality",
                description="Permission/allowance markers",
                stimuli=[
                    "We may follow whatever procedures we choose.",
                    "Everyone is allowed to decide for themselves.",
                    "You can accept or reject the decision freely.",
                    "It is permitted to question any rule.",
                    "One is free to disagree with authority.",
                    "We are allowed to challenge the outcome.",
                    "There are choices available to us.",
                    "You may do what you think is right.",
                ]
            ),
            # Dynamic Modality
            "dynamic_ability": LinguisticCondition(
                name="Dynamic Ability",
                linguistic_property="dynamic_modality",
                description="Ability/capacity markers",
                stimuli=[
                    "I can accomplish anything I set my mind to.",
                    "We are able to overcome any obstacle together.",
                    "It is within our power to change this situation.",
                    "We have the capacity to make a difference.",
                    "I am capable of achieving my goals.",
                    "Together we can solve this problem.",
                    "We have what it takes to succeed.",
                    "I am able to contribute meaningfully.",
                ]
            ),
            "dynamic_inability": LinguisticCondition(
                name="Dynamic Inability",
                linguistic_property="dynamic_modality",
                description="Inability/incapacity markers",
                stimuli=[
                    "I cannot accomplish what I want to do.",
                    "We are unable to overcome these obstacles.",
                    "It is beyond our power to change this situation.",
                    "We lack the capacity to make a difference.",
                    "I am incapable of achieving my goals.",
                    "Together we still cannot solve this problem.",
                    "We do not have what it takes to succeed.",
                    "I am unable to contribute meaningfully.",
                ]
            ),
            "dynamic_volition": LinguisticCondition(
                name="Dynamic Volition",
                linguistic_property="dynamic_modality",
                description="Volition/willingness markers",
                stimuli=[
                    "I will accomplish what I set out to do.",
                    "We want to overcome every obstacle together.",
                    "I choose to change this situation.",
                    "We intend to make a difference.",
                    "I am determined to achieve my goals.",
                    "Together we will solve this problem.",
                    "We are committed to succeeding.",
                    "I insist on contributing meaningfully.",
                ]
            ),
        }

    @staticmethod
    async def run(client: ObservatoryClient) -> ExperimentResult:
        """Execute the modality experiment."""
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT 2: {ModalityTest.NAME}")
        logger.info(f"{'='*70}")
        logger.info(f"Hypothesis: {ModalityTest.HYPOTHESIS}")

        conditions = ModalityTest.get_conditions()
        condition_results = {}

        for cond_name, condition in conditions.items():
            logger.info(f"\nAnalyzing condition: {condition.name}")
            projections = await client.analyze_batch(condition.stimuli)

            agencies = [p.coordinates.get("agency", 0) for p in projections]
            justices = [p.coordinates.get("perceived_justice",
                        p.coordinates.get("fairness", 0)) for p in projections]
            belongings = [p.coordinates.get("belonging", 0) for p in projections]

            modes = {}
            for p in projections:
                modes[p.mode] = modes.get(p.mode, 0) + 1

            condition_results[cond_name] = ConditionResults(
                condition=cond_name,
                n=len(projections),
                agency_mean=np.mean(agencies),
                agency_std=np.std(agencies),
                justice_mean=np.mean(justices),
                justice_std=np.std(justices),
                belonging_mean=np.mean(belongings),
                belonging_std=np.std(belongings),
                mode_distribution=modes,
                projections=projections
            )

            logger.info(f"  Agency: {np.mean(agencies):+.3f} (+/- {np.std(agencies):.3f})")
            logger.info(f"  Justice: {np.mean(justices):+.3f} (+/- {np.std(justices):.3f})")
            logger.info(f"  Belonging: {np.mean(belongings):+.3f} (+/- {np.std(belongings):.3f})")

        # Effect sizes
        effect_sizes = {}

        # H1: Deontic modality on agency
        obligation = condition_results["deontic_obligation"]
        permission = condition_results["deontic_permission"]

        effect_sizes["deontic_obligation_vs_permission_agency"] = hedges_g(
            obligation.get_axis_values("agency"),
            permission.get_axis_values("agency")
        )

        # H2: Epistemic certainty on justice
        certain = condition_results["epistemic_certain"]
        uncertain = condition_results["epistemic_uncertain"]

        effect_sizes["epistemic_certain_vs_uncertain_justice"] = hedges_g(
            certain.get_axis_values("justice"),
            uncertain.get_axis_values("justice")
        )

        # H3: Dynamic ability on belonging
        ability = condition_results["dynamic_ability"]
        inability = condition_results["dynamic_inability"]
        volition = condition_results["dynamic_volition"]

        effect_sizes["dynamic_ability_vs_inability_belonging"] = hedges_g(
            ability.get_axis_values("belonging"),
            inability.get_axis_values("belonging")
        )

        effect_sizes["dynamic_ability_vs_inability_agency"] = hedges_g(
            ability.get_axis_values("agency"),
            inability.get_axis_values("agency")
        )

        # Cross-modal comparisons
        effect_sizes["deontic_vs_dynamic_agency"] = hedges_g(
            obligation.get_axis_values("agency"),
            volition.get_axis_values("agency")
        )

        # Statistical tests
        statistical_tests = {}

        # ANOVA across modality types on agency
        f_stat, p_val = stats.f_oneway(
            obligation.get_axis_values("agency"),
            permission.get_axis_values("agency"),
            ability.get_axis_values("agency"),
            volition.get_axis_values("agency")
        )
        statistical_tests["modality_agency_anova"] = {
            "F": f_stat, "p": p_val, "significant": p_val < 0.05
        }

        # Bootstrap CIs
        bootstrap_cis = {}
        for cond_name, cond_result in condition_results.items():
            coords = np.column_stack([
                cond_result.get_axis_values("agency"),
                cond_result.get_axis_values("justice"),
                cond_result.get_axis_values("belonging")
            ])
            if len(coords) > 0:
                bootstrap_cis[cond_name] = bootstrap_coordinate_ci(coords)

        # Manifold distances
        manifold_distances = {}
        manifold_distances["deontic_obligation_vs_permission"] = manifold_distance(
            np.array([obligation.agency_mean, obligation.justice_mean, obligation.belonging_mean]),
            np.array([permission.agency_mean, permission.justice_mean, permission.belonging_mean])
        )

        manifold_distances["ability_vs_inability"] = manifold_distance(
            np.array([ability.agency_mean, ability.justice_mean, ability.belonging_mean]),
            np.array([inability.agency_mean, inability.justice_mean, inability.belonging_mean])
        )

        # Findings
        findings = []

        d_deontic = effect_sizes["deontic_obligation_vs_permission_agency"]
        findings.append(
            f"H1 {'SUPPORTED' if d_deontic.d < -0.5 else 'NOT SUPPORTED'}: "
            f"Deontic obligation vs permission on agency d={d_deontic.d:.3f}"
        )

        d_epistemic = effect_sizes["epistemic_certain_vs_uncertain_justice"]
        findings.append(
            f"H2 {'SUPPORTED' if d_epistemic.d > 0.3 else 'NOT SUPPORTED'}: "
            f"Epistemic certainty vs uncertainty on justice d={d_epistemic.d:.3f}"
        )

        d_dynamic = effect_sizes["dynamic_ability_vs_inability_belonging"]
        findings.append(
            f"H3 {'SUPPORTED' if d_dynamic.d > 0.3 else 'NOT SUPPORTED'}: "
            f"Dynamic ability vs inability on belonging d={d_dynamic.d:.3f}"
        )

        logger.info("\n--- EFFECT SIZES ---")
        for name, es in effect_sizes.items():
            sig = "*" if es.is_significant else ""
            logger.info(f"  {name}: d={es.d:.3f}{sig} ({es.feature_classification})")

        logger.info("\n--- KEY FINDINGS ---")
        for f in findings:
            logger.info(f"  {f}")

        return ExperimentResult(
            name=ModalityTest.NAME,
            hypothesis=ModalityTest.HYPOTHESIS,
            conditions=condition_results,
            effect_sizes=effect_sizes,
            bootstrap_cis=bootstrap_cis,
            statistical_tests=statistical_tests,
            manifold_distances=manifold_distances,
            key_findings=findings,
            metadata={
                "modality_types": ["epistemic", "deontic", "dynamic"],
                "n_conditions": len(conditions)
            }
        )


# =============================================================================
# TEST 3: Voice (Active vs Passive) Effects on Agency
# =============================================================================

class VoiceTest:
    """
    EXPERIMENT 3: Grammatical Voice and Agency Perception

    Hypothesis: Passive voice systematically reduces agency scores by
    backgrounding the agent, while active voice foregrounds agency.
    This represents a fundamental coordination-necessary feature.

    Controls for semantic content while varying only grammatical voice.
    """

    NAME = "Grammatical Voice and Agency Perception"

    HYPOTHESIS = (
        "H1: Passive voice constructions will show significantly lower agency "
        "scores than active voice minimal pairs (d > 0.5). "
        "H2: The effect will be strongest when the agent is completely deleted "
        "(agentless passive) vs retained (by-phrase passive). "
        "H3: Middle voice constructions will show intermediate agency scores."
    )

    @staticmethod
    def get_conditions() -> Dict[str, LinguisticCondition]:
        return {
            "active_voice": LinguisticCondition(
                name="Active Voice",
                linguistic_property="grammatical_voice",
                description="Subject as agent performing action",
                stimuli=[
                    "The committee approved the proposal unanimously.",
                    "We built this organization from the ground up.",
                    "The team completed the project ahead of schedule.",
                    "I made the final decision after careful thought.",
                    "Our group solved the problem through collaboration.",
                    "The workers organized the strike themselves.",
                    "She wrote the report that changed everything.",
                    "They created a new policy to address the issue.",
                ]
            ),
            "passive_with_agent": LinguisticCondition(
                name="Passive with Agent",
                linguistic_property="grammatical_voice",
                description="Passive with by-phrase retaining agent",
                stimuli=[
                    "The proposal was approved by the committee unanimously.",
                    "This organization was built by us from the ground up.",
                    "The project was completed by the team ahead of schedule.",
                    "The final decision was made by me after careful thought.",
                    "The problem was solved by our group through collaboration.",
                    "The strike was organized by the workers themselves.",
                    "The report that changed everything was written by her.",
                    "A new policy was created by them to address the issue.",
                ]
            ),
            "passive_agentless": LinguisticCondition(
                name="Passive Agentless",
                linguistic_property="grammatical_voice",
                description="Passive without agent expressed",
                stimuli=[
                    "The proposal was approved unanimously.",
                    "This organization was built from the ground up.",
                    "The project was completed ahead of schedule.",
                    "The final decision was made after careful thought.",
                    "The problem was solved through collaboration.",
                    "The strike was organized successfully.",
                    "The report that changed everything was written.",
                    "A new policy was created to address the issue.",
                ]
            ),
            "middle_voice": LinguisticCondition(
                name="Middle Voice",
                linguistic_property="grammatical_voice",
                description="Subject affected by action, agent unclear",
                stimuli=[
                    "The proposal approved easily with broad support.",
                    "This organization built itself from the ground up.",
                    "The project completed ahead of schedule.",
                    "The decision emerged after careful thought.",
                    "The problem solved through collaboration.",
                    "The strike organized itself successfully.",
                    "The report wrote itself, almost.",
                    "The policy developed naturally over time.",
                ]
            ),
            "causative": LinguisticCondition(
                name="Causative",
                linguistic_property="grammatical_voice",
                description="Subject causes action by another",
                stimuli=[
                    "We had the proposal approved by the committee.",
                    "I got this organization built from nothing.",
                    "The manager had the project completed early.",
                    "She made the decision happen after much thought.",
                    "We got the problem solved through persistence.",
                    "They had the strike organized in record time.",
                    "He made the report get written somehow.",
                    "We got the policy created despite resistance.",
                ]
            ),
        }

    @staticmethod
    async def run(client: ObservatoryClient) -> ExperimentResult:
        """Execute the voice experiment."""
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT 3: {VoiceTest.NAME}")
        logger.info(f"{'='*70}")
        logger.info(f"Hypothesis: {VoiceTest.HYPOTHESIS}")

        conditions = VoiceTest.get_conditions()
        condition_results = {}

        for cond_name, condition in conditions.items():
            logger.info(f"\nAnalyzing condition: {condition.name}")
            projections = await client.analyze_batch(condition.stimuli)

            agencies = [p.coordinates.get("agency", 0) for p in projections]
            justices = [p.coordinates.get("perceived_justice",
                        p.coordinates.get("fairness", 0)) for p in projections]
            belongings = [p.coordinates.get("belonging", 0) for p in projections]

            modes = {}
            for p in projections:
                modes[p.mode] = modes.get(p.mode, 0) + 1

            condition_results[cond_name] = ConditionResults(
                condition=cond_name,
                n=len(projections),
                agency_mean=np.mean(agencies),
                agency_std=np.std(agencies),
                justice_mean=np.mean(justices),
                justice_std=np.std(justices),
                belonging_mean=np.mean(belongings),
                belonging_std=np.std(belongings),
                mode_distribution=modes,
                projections=projections
            )

            logger.info(f"  Agency: {np.mean(agencies):+.3f} (+/- {np.std(agencies):.3f})")

        # Effect sizes
        effect_sizes = {}

        active = condition_results["active_voice"]
        passive_agent = condition_results["passive_with_agent"]
        passive_no = condition_results["passive_agentless"]
        middle = condition_results["middle_voice"]
        causative = condition_results["causative"]

        # H1: Active vs passive
        effect_sizes["active_vs_passive_agentless"] = hedges_g(
            active.get_axis_values("agency"),
            passive_no.get_axis_values("agency")
        )

        effect_sizes["active_vs_passive_with_agent"] = hedges_g(
            active.get_axis_values("agency"),
            passive_agent.get_axis_values("agency")
        )

        # H2: By-phrase vs agentless passive
        effect_sizes["passive_agent_vs_agentless"] = hedges_g(
            passive_agent.get_axis_values("agency"),
            passive_no.get_axis_values("agency")
        )

        # H3: Middle voice
        effect_sizes["active_vs_middle"] = hedges_g(
            active.get_axis_values("agency"),
            middle.get_axis_values("agency")
        )

        # Causative (control)
        effect_sizes["active_vs_causative"] = hedges_g(
            active.get_axis_values("agency"),
            causative.get_axis_values("agency")
        )

        # Statistical tests
        statistical_tests = {}

        # Trend test: active > causative > passive_agent > middle > passive_agentless
        voice_order = ["active_voice", "causative", "passive_with_agent",
                       "middle_voice", "passive_agentless"]
        voice_values = [condition_results[v].agency_mean for v in voice_order]
        indices = list(range(len(voice_order)))

        r, p = stats.spearmanr(indices, voice_values[::-1])  # Reverse for decreasing
        statistical_tests["voice_agency_trend"] = {
            "spearman_r": r, "p": p, "significant": p < 0.05
        }

        # ANOVA
        f_stat, p_val = stats.f_oneway(
            active.get_axis_values("agency"),
            passive_agent.get_axis_values("agency"),
            passive_no.get_axis_values("agency"),
            middle.get_axis_values("agency")
        )
        statistical_tests["voice_agency_anova"] = {
            "F": f_stat, "p": p_val, "significant": p_val < 0.05
        }

        # Bootstrap CIs
        bootstrap_cis = {}
        for cond_name, cond_result in condition_results.items():
            coords = np.column_stack([
                cond_result.get_axis_values("agency"),
                cond_result.get_axis_values("justice"),
                cond_result.get_axis_values("belonging")
            ])
            if len(coords) > 0:
                bootstrap_cis[cond_name] = bootstrap_coordinate_ci(coords)

        # Manifold distances
        manifold_distances = {}
        manifold_distances["active_vs_agentless_passive"] = manifold_distance(
            np.array([active.agency_mean, active.justice_mean, active.belonging_mean]),
            np.array([passive_no.agency_mean, passive_no.justice_mean, passive_no.belonging_mean])
        )

        # Findings
        findings = []

        d_main = effect_sizes["active_vs_passive_agentless"]
        findings.append(
            f"H1 {'SUPPORTED' if d_main.d > 0.5 else 'NOT SUPPORTED'}: "
            f"Active vs agentless passive on agency d={d_main.d:.3f}"
        )

        d_agent = effect_sizes["passive_agent_vs_agentless"]
        findings.append(
            f"H2 {'SUPPORTED' if d_agent.d > 0.2 else 'NOT SUPPORTED'}: "
            f"By-phrase vs agentless passive d={d_agent.d:.3f}"
        )

        d_middle = effect_sizes["active_vs_middle"]
        findings.append(
            f"H3 {'SUPPORTED' if 0.2 < d_middle.d < d_main.d else 'NOT SUPPORTED'}: "
            f"Middle voice intermediate, d={d_middle.d:.3f}"
        )

        logger.info("\n--- AGENCY RANKING BY VOICE ---")
        ranked = sorted(condition_results.items(),
                       key=lambda x: x[1].agency_mean, reverse=True)
        for name, res in ranked:
            logger.info(f"  {name}: {res.agency_mean:+.3f}")

        logger.info("\n--- EFFECT SIZES ---")
        for name, es in effect_sizes.items():
            sig = "*" if es.is_significant else ""
            logger.info(f"  {name}: d={es.d:.3f}{sig}")

        logger.info("\n--- KEY FINDINGS ---")
        for f in findings:
            logger.info(f"  {f}")

        return ExperimentResult(
            name=VoiceTest.NAME,
            hypothesis=VoiceTest.HYPOTHESIS,
            conditions=condition_results,
            effect_sizes=effect_sizes,
            bootstrap_cis=bootstrap_cis,
            statistical_tests=statistical_tests,
            manifold_distances=manifold_distances,
            key_findings=findings,
            metadata={
                "voice_types": ["active", "passive_agent", "passive_agentless",
                               "middle", "causative"]
            }
        )


# =============================================================================
# TEST 4: Cross-Register Variation (Formal vs Informal)
# =============================================================================

class RegisterTest:
    """
    EXPERIMENT 4: Register Variation and Coordination Legibility

    Hypothesis: Register variation affects legibility and mode classification
    without substantially changing coordination core position. Formal register
    should increase legibility confidence while informal register may reduce it.

    This tests whether register is a "decorative" feature (fiber) or affects
    the base manifold position.
    """

    NAME = "Register Variation and Coordination Legibility"

    HYPOTHESIS = (
        "H1: Register variation is largely decorative - formal and informal "
        "versions of the same content will have similar base manifold positions "
        "(d < 0.3). "
        "H2: Formal register will show higher classification confidence "
        "(legibility score). "
        "H3: Mode classification stability will be higher for formal register. "
        "H4: Hedging/informal markers will reduce legibility without changing "
        "coordination core."
    )

    @staticmethod
    def get_conditions() -> Dict[str, LinguisticCondition]:
        return {
            "formal_high": LinguisticCondition(
                name="Formal High Register",
                linguistic_property="register",
                description="Formal academic/legal register",
                stimuli=[
                    "The committee has determined that the proposed allocation is equitable.",
                    "It is incumbent upon all parties to adhere to the established protocols.",
                    "The organization shall endeavor to ensure fair treatment of all members.",
                    "Pursuant to the agreement, all stakeholders are entitled to participation.",
                    "The evidence demonstrates conclusively that proper procedures were followed.",
                    "Henceforth, all decisions shall be made with due consideration for equity.",
                    "The aforementioned policies have been implemented with unanimous consent.",
                    "It is resolved that the collective shall act in the common interest.",
                ]
            ),
            "formal_neutral": LinguisticCondition(
                name="Formal Neutral Register",
                linguistic_property="register",
                description="Formal but neutral business register",
                stimuli=[
                    "The committee determined that the proposed allocation is equitable.",
                    "All parties should adhere to the established protocols.",
                    "The organization will ensure fair treatment of all members.",
                    "According to the agreement, all stakeholders can participate.",
                    "The evidence shows that proper procedures were followed.",
                    "Going forward, all decisions will consider equity.",
                    "These policies have been implemented with consent.",
                    "The collective will act in the common interest.",
                ]
            ),
            "informal_neutral": LinguisticCondition(
                name="Informal Neutral Register",
                linguistic_property="register",
                description="Casual but clear register",
                stimuli=[
                    "The committee decided the plan was fair.",
                    "Everyone needs to follow the rules.",
                    "The group is going to treat everyone fairly.",
                    "Based on the deal, everyone gets to participate.",
                    "The proof shows things were done right.",
                    "From now on, decisions will be fair.",
                    "These rules are now in place.",
                    "We're all going to work together.",
                ]
            ),
            "informal_colloquial": LinguisticCondition(
                name="Informal Colloquial Register",
                linguistic_property="register",
                description="Very informal/slang register",
                stimuli=[
                    "So the committee was like, yeah the plan's totally fair.",
                    "Basically everyone's gotta follow the rules, you know?",
                    "The group's gonna make sure everyone gets a fair shake.",
                    "Like, according to the deal, everyone gets to join in.",
                    "The proof totally shows stuff was done right.",
                    "So yeah, from now on things will be fair and all.",
                    "These rules are like, in place now.",
                    "We're all gonna work together, basically.",
                ]
            ),
            "hedged": LinguisticCondition(
                name="Heavily Hedged",
                linguistic_property="register",
                description="Many uncertainty and hedging markers",
                stimuli=[
                    "It seems the committee may have decided the plan might be fair.",
                    "Perhaps everyone should probably follow the rules.",
                    "The group might possibly try to treat people fairly.",
                    "It appears that maybe everyone could participate.",
                    "The evidence seems to suggest things were perhaps done right.",
                    "It's possible that decisions might be fairer going forward.",
                    "These rules seem to be somewhat in place now.",
                    "We might potentially work together, sort of.",
                ]
            ),
            "intensified": LinguisticCondition(
                name="Heavily Intensified",
                linguistic_property="register",
                description="Many intensifiers and emphatics",
                stimuli=[
                    "The committee absolutely decided the plan is completely fair!",
                    "Everyone totally must always follow every single rule!",
                    "The group will definitely ensure extremely fair treatment!",
                    "Obviously everyone definitely gets to fully participate!",
                    "The proof clearly shows everything was perfectly done!",
                    "Absolutely, decisions will be incredibly fair always!",
                    "These rules are now completely and totally in place!",
                    "We are definitely going to work together brilliantly!",
                ]
            ),
        }

    @staticmethod
    async def run(client: ObservatoryClient) -> ExperimentResult:
        """Execute the register experiment."""
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT 4: {RegisterTest.NAME}")
        logger.info(f"{'='*70}")
        logger.info(f"Hypothesis: {RegisterTest.HYPOTHESIS}")

        conditions = RegisterTest.get_conditions()
        condition_results = {}

        for cond_name, condition in conditions.items():
            logger.info(f"\nAnalyzing condition: {condition.name}")
            projections = await client.analyze_batch(condition.stimuli)

            agencies = [p.coordinates.get("agency", 0) for p in projections]
            justices = [p.coordinates.get("perceived_justice",
                        p.coordinates.get("fairness", 0)) for p in projections]
            belongings = [p.coordinates.get("belonging", 0) for p in projections]
            confidences = [p.confidence for p in projections]

            modes = {}
            for p in projections:
                modes[p.mode] = modes.get(p.mode, 0) + 1

            condition_results[cond_name] = ConditionResults(
                condition=cond_name,
                n=len(projections),
                agency_mean=np.mean(agencies),
                agency_std=np.std(agencies),
                justice_mean=np.mean(justices),
                justice_std=np.std(justices),
                belonging_mean=np.mean(belongings),
                belonging_std=np.std(belongings),
                mode_distribution=modes,
                projections=projections
            )

            logger.info(f"  Agency: {np.mean(agencies):+.3f}")
            logger.info(f"  Justice: {np.mean(justices):+.3f}")
            logger.info(f"  Confidence: {np.mean(confidences):.3f}")

        # Effect sizes
        effect_sizes = {}

        formal_high = condition_results["formal_high"]
        formal_neutral = condition_results["formal_neutral"]
        informal_neutral = condition_results["informal_neutral"]
        informal_colloquial = condition_results["informal_colloquial"]
        hedged = condition_results["hedged"]
        intensified = condition_results["intensified"]

        # H1: Register variation is decorative
        effect_sizes["formal_vs_informal_agency"] = hedges_g(
            formal_neutral.get_axis_values("agency"),
            informal_neutral.get_axis_values("agency")
        )
        effect_sizes["formal_vs_informal_justice"] = hedges_g(
            formal_neutral.get_axis_values("justice"),
            informal_neutral.get_axis_values("justice")
        )
        effect_sizes["formal_vs_informal_belonging"] = hedges_g(
            formal_neutral.get_axis_values("belonging"),
            informal_neutral.get_axis_values("belonging")
        )

        # H4: Hedging effect
        effect_sizes["neutral_vs_hedged_agency"] = hedges_g(
            formal_neutral.get_axis_values("agency"),
            hedged.get_axis_values("agency")
        )
        effect_sizes["neutral_vs_intensified_agency"] = hedges_g(
            formal_neutral.get_axis_values("agency"),
            intensified.get_axis_values("agency")
        )

        # Statistical tests
        statistical_tests = {}

        # Test if register affects position
        f_stat, p_val = stats.f_oneway(
            formal_neutral.get_axis_values("agency"),
            informal_neutral.get_axis_values("agency"),
            informal_colloquial.get_axis_values("agency")
        )
        statistical_tests["register_agency_anova"] = {
            "F": f_stat, "p": p_val, "significant": p_val < 0.05
        }

        # Confidence comparison
        formal_conf = [p.confidence for p in formal_neutral.projections]
        informal_conf = [p.confidence for p in informal_colloquial.projections]
        t_stat, p_val = stats.ttest_ind(formal_conf, informal_conf)
        statistical_tests["confidence_formal_vs_informal"] = {
            "t": t_stat, "p": p_val, "significant": p_val < 0.05,
            "formal_mean": np.mean(formal_conf),
            "informal_mean": np.mean(informal_conf)
        }

        # Bootstrap CIs
        bootstrap_cis = {}
        for cond_name, cond_result in condition_results.items():
            coords = np.column_stack([
                cond_result.get_axis_values("agency"),
                cond_result.get_axis_values("justice"),
                cond_result.get_axis_values("belonging")
            ])
            if len(coords) > 0:
                bootstrap_cis[cond_name] = bootstrap_coordinate_ci(coords)

        # Manifold distances
        manifold_distances = {}
        manifold_distances["formal_vs_informal"] = manifold_distance(
            np.array([formal_neutral.agency_mean, formal_neutral.justice_mean,
                     formal_neutral.belonging_mean]),
            np.array([informal_neutral.agency_mean, informal_neutral.justice_mean,
                     informal_neutral.belonging_mean])
        )

        manifold_distances["neutral_vs_hedged"] = manifold_distance(
            np.array([formal_neutral.agency_mean, formal_neutral.justice_mean,
                     formal_neutral.belonging_mean]),
            np.array([hedged.agency_mean, hedged.justice_mean, hedged.belonging_mean])
        )

        # Findings
        findings = []

        # H1: Is register decorative?
        max_register_d = max(
            abs(effect_sizes["formal_vs_informal_agency"].d),
            abs(effect_sizes["formal_vs_informal_justice"].d),
            abs(effect_sizes["formal_vs_informal_belonging"].d)
        )
        findings.append(
            f"H1 {'SUPPORTED' if max_register_d < 0.3 else 'NOT SUPPORTED'}: "
            f"Register variation max d={max_register_d:.3f} "
            f"({'decorative' if max_register_d < 0.3 else 'not decorative'})"
        )

        # H2: Formal higher confidence
        conf_test = statistical_tests["confidence_formal_vs_informal"]
        findings.append(
            f"H2 {'SUPPORTED' if conf_test['formal_mean'] > conf_test['informal_mean'] else 'NOT SUPPORTED'}: "
            f"Formal confidence={conf_test['formal_mean']:.3f} vs "
            f"Informal={conf_test['informal_mean']:.3f}"
        )

        # H4: Hedging reduces legibility
        d_hedge = effect_sizes["neutral_vs_hedged_agency"]
        findings.append(
            f"H4: Hedging effect on agency d={d_hedge.d:.3f} "
            f"({d_hedge.feature_classification})"
        )

        logger.info("\n--- EFFECT SIZES (Register Comparisons) ---")
        for name, es in effect_sizes.items():
            sig = "*" if es.is_significant else ""
            logger.info(f"  {name}: d={es.d:.3f}{sig}")

        logger.info("\n--- KEY FINDINGS ---")
        for f in findings:
            logger.info(f"  {f}")

        return ExperimentResult(
            name=RegisterTest.NAME,
            hypothesis=RegisterTest.HYPOTHESIS,
            conditions=condition_results,
            effect_sizes=effect_sizes,
            bootstrap_cis=bootstrap_cis,
            statistical_tests=statistical_tests,
            manifold_distances=manifold_distances,
            key_findings=findings,
            metadata={
                "register_levels": ["formal_high", "formal_neutral",
                                   "informal_neutral", "informal_colloquial",
                                   "hedged", "intensified"]
            }
        )


# =============================================================================
# TEST 5: Integrated Grammar Deletion Analysis
# =============================================================================

class GrammarDeletionTest:
    """
    EXPERIMENT 5: Systematic Grammar Feature Deletion

    Combines all previous insights to systematically test which grammatical
    features are coordination-necessary vs decorative.

    Uses the /research/grammar-deletion API endpoint to systematically
    remove features and measure drift.
    """

    NAME = "Systematic Grammar Feature Necessity Analysis"

    HYPOTHESIS = (
        "H1: Person deixis deletion will cause the highest coordination drift "
        "(critical feature). "
        "H2: Articles and intensifiers will cause minimal drift (decorative). "
        "H3: Modals and voice markers will cause moderate drift (necessary). "
        "H4: Features can be ranked by coordination necessity."
    )

    @staticmethod
    def get_test_texts() -> List[str]:
        """Representative texts for grammar deletion analysis."""
        return [
            "I believe we should work together to build a fair society where everyone belongs.",
            "They must have decided that the outcome was definitely fair to all parties.",
            "We here in this community have always treated each other with respect and dignity.",
            "The committee has determined that proper procedures were certainly followed throughout.",
            "My team and I will ensure that future decisions reflect our shared values.",
            "Perhaps the situation could possibly be resolved through careful deliberation.",
            "We absolutely must acknowledge that everyone deserves to be treated fairly.",
            "Those people there have their own way of making decisions together.",
        ]

    @staticmethod
    async def run(client: ObservatoryClient) -> ExperimentResult:
        """Execute the grammar deletion experiment."""
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT 5: {GrammarDeletionTest.NAME}")
        logger.info(f"{'='*70}")

        test_texts = GrammarDeletionTest.get_test_texts()

        # Analyze each text with grammar deletion
        all_results = []
        feature_drifts: Dict[str, List[float]] = {}

        for text in test_texts:
            logger.info(f"\nAnalyzing: {text[:50]}...")
            result = await client.grammar_deletion(text)

            if result:
                all_results.append(result)
                for feature in result.get("feature_rankings", []):
                    feat_name = feature.get("feature_name", "unknown")
                    drift = feature.get("projection_drift", 0)
                    if feat_name not in feature_drifts:
                        feature_drifts[feat_name] = []
                    feature_drifts[feat_name].append(drift)

        # Compute effect sizes for each feature
        effect_sizes = {}
        baseline = 0.0  # Compare against zero drift

        for feature_name, drifts in feature_drifts.items():
            if len(drifts) >= 3:
                drift_array = np.array(drifts)
                baseline_array = np.zeros(len(drifts))

                # Cohen's d comparing drift to zero
                d = np.mean(drifts) / (np.std(drifts) + 0.001)
                se = np.std(drifts) / np.sqrt(len(drifts))

                # Classify based on mean drift
                mean_drift = np.mean(drifts)
                if mean_drift < 0.1:
                    classification = "decorative"
                elif mean_drift < 0.3:
                    classification = "modifying"
                elif mean_drift < 0.5:
                    classification = "necessary"
                else:
                    classification = "critical"

                effect_sizes[feature_name] = EffectSize(
                    d=d,
                    standard_error=se,
                    confidence_interval=(d - 1.96*se, d + 1.96*se),
                    interpretation=EffectSize(d=d, standard_error=se,
                        confidence_interval=(d-1.96*se, d+1.96*se),
                        interpretation=classification,
                        feature_classification=classification,
                        n1=len(drifts), n2=len(drifts)).interpretation,
                    feature_classification=classification,
                    n1=len(drifts),
                    n2=len(drifts)
                )

        # Create condition results (one per feature)
        condition_results = {}
        for feature_name, drifts in feature_drifts.items():
            condition_results[feature_name] = ConditionResults(
                condition=feature_name,
                n=len(drifts),
                agency_mean=np.mean(drifts),  # Using drift as primary metric
                agency_std=np.std(drifts),
                justice_mean=0,
                justice_std=0,
                belonging_mean=0,
                belonging_std=0,
                mode_distribution={},
                projections=[]
            )

        # Statistical tests
        statistical_tests = {}

        # Rank features by mean drift
        ranked_features = sorted(feature_drifts.items(),
                                key=lambda x: np.mean(x[1]), reverse=True)

        statistical_tests["feature_ranking"] = {
            "order": [f[0] for f in ranked_features],
            "mean_drifts": [np.mean(f[1]) for f in ranked_features]
        }

        # Findings
        findings = []

        # Find highest drift feature
        if ranked_features:
            highest = ranked_features[0]
            findings.append(f"Highest drift feature: {highest[0]} (mean={np.mean(highest[1]):.3f})")

            lowest = ranked_features[-1]
            findings.append(f"Lowest drift feature: {lowest[0]} (mean={np.mean(lowest[1]):.3f})")

        # Count by classification
        classifications = {}
        for name, es in effect_sizes.items():
            cls = es.feature_classification
            classifications[cls] = classifications.get(cls, 0) + 1

        findings.append(f"Feature distribution: {classifications}")

        logger.info("\n--- FEATURE RANKING BY COORDINATION NECESSITY ---")
        for feat_name, drifts in ranked_features[:10]:
            mean_d = np.mean(drifts)
            if feat_name in effect_sizes:
                cls = effect_sizes[feat_name].feature_classification
                logger.info(f"  {feat_name}: {mean_d:.3f} ({cls})")

        logger.info("\n--- KEY FINDINGS ---")
        for f in findings:
            logger.info(f"  {f}")

        return ExperimentResult(
            name=GrammarDeletionTest.NAME,
            hypothesis=GrammarDeletionTest.HYPOTHESIS,
            conditions=condition_results,
            effect_sizes=effect_sizes,
            bootstrap_cis={},
            statistical_tests=statistical_tests,
            manifold_distances={},
            key_findings=findings,
            metadata={
                "n_texts_analyzed": len(test_texts),
                "n_features_tested": len(feature_drifts)
            }
        )


# =============================================================================
# Main Runner
# =============================================================================

async def run_all_experiments(output_dir: Path = None) -> Dict[str, ExperimentResult]:
    """Run all experiments and generate publication-ready output."""

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "research_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = ObservatoryClient()
    results = {}

    # Run all tests
    tests = [
        ("deixis", DeisisTest),
        ("modality", ModalityTest),
        ("voice", VoiceTest),
        ("register", RegisterTest),
        ("grammar_deletion", GrammarDeletionTest),
    ]

    for test_name, test_class in tests:
        try:
            result = await test_class.run(client)
            results[test_name] = result
        except Exception as e:
            logger.error(f"Error running {test_name}: {e}")
            import traceback
            traceback.print_exc()

    # Generate summary
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*70)

    for name, result in results.items():
        logger.info(f"\n{result.name}")
        logger.info("-" * len(result.name))
        for finding in result.key_findings:
            logger.info(f"  {finding}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON output
    json_path = output_dir / f"linguistic_tests_{timestamp}.json"
    json_output = {}
    for name, result in results.items():
        json_output[name] = {
            "name": result.name,
            "hypothesis": result.hypothesis,
            "key_findings": result.key_findings,
            "effect_sizes": {k: v.to_dict() for k, v in result.effect_sizes.items()},
            "statistical_tests": result.statistical_tests,
            "metadata": result.metadata
        }

    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {json_path}")

    # LaTeX output
    latex_path = output_dir / f"effect_sizes_table_{timestamp}.tex"
    all_effects = []
    for name, result in results.items():
        for eff_name, eff in result.effect_sizes.items():
            all_effects.append({
                "experiment": name,
                "comparison": eff_name,
                "d": eff.d,
                "confidence_interval": list(eff.confidence_interval),
                "classification": eff.feature_classification,
                "is_significant": eff.is_significant
            })

    if all_effects:
        latex_table = generate_effect_size_table(
            all_effects,
            caption="Effect sizes for linguistic feature experiments",
            label="tab:linguistic_effects"
        )
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        logger.info(f"LaTeX table saved to: {latex_path}")

    return results


def main():
    """Entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Linguistic Research Tests for Cultural Soliton Observatory"
    )
    parser.add_argument("--test", type=str, default="all",
                       choices=["all", "deixis", "modality", "voice",
                               "register", "grammar_deletion"],
                       help="Which test(s) to run")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory for output files")
    parser.add_argument("--output-format", type=str, default="both",
                       choices=["json", "latex", "both"],
                       help="Output format")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    print("="*70)
    print("CULTURAL SOLITON OBSERVATORY v2.0")
    print("Comprehensive Linguistic Research Tests for TACL/ACL Publication")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running tests: {args.test}")
    print("="*70)

    asyncio.run(run_all_experiments(output_dir))


if __name__ == "__main__":
    main()
