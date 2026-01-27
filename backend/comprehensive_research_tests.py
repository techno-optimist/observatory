#!/usr/bin/env python3
"""
Cultural Soliton Observatory - Comprehensive Research Test Suite

This test suite validates the Observatory's capabilities across multiple domains:
1. Linguistic Feature Tests (TACL/ACL publication quality)
2. Cognitive Validity Tests (Cognition/Psych Review quality)
3. AI Safety Tests (NeurIPS/ICML quality)
4. Statistical Rigor Tests (Methodology validation)
5. MCP Integration Tests (System validation)
6. Substrate-Agnostic Tests (Cross-species coordination)

Run via: python3 comprehensive_research_tests.py
"""

import numpy as np
import httpx
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import sys

# Add parent to path for imports
sys.path.insert(0, '.')

from research import (
    # Hierarchical coordinates
    extract_hierarchical_coordinate,
    HierarchicalCoordinate,
    CoordinationCore,
    extract_features,
    FEATURE_PATTERNS,

    # Academic statistics
    cohens_d,
    hedges_g,
    bootstrap_ci,
    bootstrap_coordinate_ci,
    fisher_rao_distance,
    hellinger_distance,
    manifold_distance,
    detect_phase_transitions,
    apply_correction,

    # Publication formats
    generate_effect_size_table,
    compute_summary_statistics,
    generate_experiment_report,
)


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

BASE_URL = "http://localhost:8000"
TIMEOUT = 60.0

class TestCategory(Enum):
    LINGUISTIC = "linguistic"
    COGNITIVE = "cognitive"
    AI_SAFETY = "ai_safety"
    STATISTICAL = "statistical"
    INTEGRATION = "integration"
    SUBSTRATE = "substrate_agnostic"


@dataclass
class TestResult:
    """Result from a single test."""
    name: str
    category: TestCategory
    passed: bool
    score: float  # 0-1 success metric
    details: Dict[str, Any]
    duration_ms: float
    error: Optional[str] = None


@dataclass
class TestSuite:
    """Collection of test results."""
    results: List[TestResult] = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"  TEST SUITE SUMMARY",
            f"{'='*70}",
            f"  Total: {self.total}  |  Passed: {self.passed}  |  Failed: {self.failed}",
            f"  Success Rate: {100*self.passed/self.total:.1f}%",
            f"{'='*70}",
        ]

        # By category
        categories = {}
        for r in self.results:
            cat = r.category.value
            if cat not in categories:
                categories[cat] = {"passed": 0, "total": 0}
            categories[cat]["total"] += 1
            if r.passed:
                categories[cat]["passed"] += 1

        lines.append("\n  By Category:")
        for cat, counts in categories.items():
            pct = 100 * counts["passed"] / counts["total"]
            lines.append(f"    {cat:20s}: {counts['passed']}/{counts['total']} ({pct:.0f}%)")

        # Failed tests
        failed = [r for r in self.results if not r.passed]
        if failed:
            lines.append("\n  Failed Tests:")
            for r in failed:
                lines.append(f"    - {r.name}: {r.error or 'unknown error'}")

        return "\n".join(lines)


# =============================================================================
# API HELPERS
# =============================================================================

def api_call(endpoint: str, payload: Dict) -> Tuple[bool, Dict]:
    """Make API call to Observatory backend."""
    try:
        response = httpx.post(
            f"{BASE_URL}{endpoint}",
            json=payload,
            timeout=TIMEOUT
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"HTTP {response.status_code}", "body": response.text}
    except Exception as e:
        return False, {"error": str(e)}


def analyze_text(text: str) -> Tuple[bool, Dict]:
    """Analyze text via API."""
    return api_call("/analyze", {"text": text})


def batch_analyze(texts: List[str]) -> Tuple[bool, Dict]:
    """Batch analyze texts via API."""
    return api_call("/batch_analyze", {"texts": texts})


# =============================================================================
# LINGUISTIC FEATURE TESTS
# =============================================================================

def test_deixis_person():
    """Test person deixis (I/we/you/they) effects on coordination axes."""
    start = time.time()

    # Test texts with controlled deixis variation
    person_texts = {
        "first_singular": [
            "I accomplished this through my own effort.",
            "I believe this is the right path forward.",
            "I made the decision to proceed alone.",
        ],
        "first_plural": [
            "We accomplished this through our collective effort.",
            "We believe this is the right path forward.",
            "We made the decision to proceed together.",
        ],
        "second": [
            "You accomplished this through your own effort.",
            "You believe this is the right path forward.",
            "You made the decision to proceed.",
        ],
        "third": [
            "They accomplished this through their own effort.",
            "They believe this is the right path forward.",
            "They made the decision to proceed.",
        ],
    }

    results = {}
    for person, texts in person_texts.items():
        coords = []
        for text in texts:
            coord = extract_hierarchical_coordinate(text)
            coords.append(coord.core.to_array())
        results[person] = np.array(coords).mean(axis=0)

    # Hypothesis: first_singular should have highest self-agency
    # first_plural should have highest ingroup belonging
    self_agency_idx = 0
    ingroup_idx = 6

    first_sing_agency = results["first_singular"][self_agency_idx]
    first_plural_belonging = results["first_plural"][ingroup_idx]

    # Effect size: first_singular vs third for agency
    effect = cohens_d(
        np.array([extract_hierarchical_coordinate(t).core.agency.self_agency for t in person_texts["first_singular"]]),
        np.array([extract_hierarchical_coordinate(t).core.agency.self_agency for t in person_texts["third"]])
    )

    passed = (
        first_sing_agency > results["third"][self_agency_idx] and
        first_plural_belonging > results["first_singular"][ingroup_idx]
    )

    return TestResult(
        name="Deixis: Person Effects",
        category=TestCategory.LINGUISTIC,
        passed=passed,
        score=abs(effect.d) / 2.0,  # Normalize to 0-1
        details={
            "first_singular_self_agency": float(first_sing_agency),
            "first_plural_ingroup": float(first_plural_belonging),
            "agency_effect_d": effect.d,
            "hypothesis_confirmed": passed,
        },
        duration_ms=(time.time() - start) * 1000
    )


def test_voice_active_passive():
    """Test active vs passive voice effects on agency."""
    start = time.time()

    # More samples needed for statistical power (n=10 per group)
    active_texts = [
        "I built this company from nothing.",
        "We solved the problem efficiently.",
        "The team achieved remarkable results.",
        "I decided to take action immediately.",
        "We created a new approach to the challenge.",
        "I completed the project ahead of schedule.",
        "We designed the entire system ourselves.",
        "I made the final decision.",
        "The engineers built an innovative solution.",
        "I resolved all the outstanding issues.",
    ]

    passive_texts = [
        "This company was built from nothing.",
        "The problem was solved efficiently.",
        "Remarkable results were achieved.",
        "A decision was made to take action.",
        "A new approach was created for the challenge.",
        "The project was completed ahead of schedule.",
        "The entire system was designed by the team.",
        "The final decision was made.",
        "An innovative solution was built.",
        "All outstanding issues were resolved.",
    ]

    active_coords = [extract_hierarchical_coordinate(t) for t in active_texts]
    passive_coords = [extract_hierarchical_coordinate(t) for t in passive_texts]

    active_agency = np.array([c.core.agency.self_agency for c in active_coords])
    passive_agency = np.array([c.core.agency.self_agency for c in passive_coords])

    effect = cohens_d(active_agency, passive_agency)

    # Hypothesis: active voice should have higher self-agency
    passed = effect.d > 0.5 and effect.is_significant

    return TestResult(
        name="Voice: Active vs Passive",
        category=TestCategory.LINGUISTIC,
        passed=passed,
        score=min(abs(effect.d), 2.0) / 2.0,
        details={
            "active_mean_agency": float(active_agency.mean()),
            "passive_mean_agency": float(passive_agency.mean()),
            "effect_d": effect.d,
            "ci": effect.confidence_interval,
            "significant": effect.is_significant,
        },
        duration_ms=(time.time() - start) * 1000
    )


def test_modality_types():
    """Test epistemic vs deontic vs dynamic modality."""
    start = time.time()

    modality_texts = {
        "epistemic": [  # Possibility/probability
            "This might be the solution we need.",
            "It could work if we try harder.",
            "Perhaps this is the right approach.",
        ],
        "deontic": [  # Obligation/permission
            "We must take action immediately.",
            "You should consider the consequences.",
            "They have to follow the rules.",
        ],
        "dynamic": [  # Ability/willingness
            "I can solve this problem myself.",
            "We are able to achieve our goals.",
            "They have the capability to succeed.",
        ],
    }

    results = {}
    for mod_type, texts in modality_texts.items():
        coords = [extract_hierarchical_coordinate(t) for t in texts]
        certainty = np.array([c.modifiers.epistemic.certainty for c in coords])
        agency = np.array([c.core.agency.aggregate for c in coords])
        results[mod_type] = {
            "certainty": float(certainty.mean()),
            "agency": float(agency.mean()),
        }

    # Hypothesis: epistemic should have lowest certainty
    # dynamic should have highest agency
    passed = (
        results["epistemic"]["certainty"] < results["deontic"]["certainty"] and
        results["dynamic"]["agency"] > results["epistemic"]["agency"]
    )

    return TestResult(
        name="Modality: Epistemic/Deontic/Dynamic",
        category=TestCategory.LINGUISTIC,
        passed=passed,
        score=0.8 if passed else 0.3,
        details=results,
        duration_ms=(time.time() - start) * 1000
    )


# =============================================================================
# COGNITIVE VALIDITY TESTS
# =============================================================================

def test_axis_independence():
    """Test that agency, justice, and belonging are psychologically independent."""
    start = time.time()

    # Texts designed to vary one axis while holding others constant
    high_agency_texts = [
        "I conquered every challenge through my determination.",
        "Through my own efforts, I achieved success.",
        "I took control and made it happen.",
    ]

    high_justice_texts = [
        "The process was fair and everyone was heard.",
        "Proper procedures ensured equitable outcomes.",
        "Justice was served through due process.",
    ]

    high_belonging_texts = [
        "We stand together as one community.",
        "Our bonds unite us in common purpose.",
        "Together we form an unbreakable whole.",
    ]

    # Extract coordinates
    agency_coords = np.array([extract_hierarchical_coordinate(t).core.to_legacy_3d() for t in high_agency_texts])
    justice_coords = np.array([extract_hierarchical_coordinate(t).core.to_legacy_3d() for t in high_justice_texts])
    belonging_coords = np.array([extract_hierarchical_coordinate(t).core.to_legacy_3d() for t in high_belonging_texts])

    # Check that each text type maximizes its target axis
    agency_max_on_agency = agency_coords[:, 0].mean() > agency_coords[:, 1:].mean()
    justice_max_on_justice = justice_coords[:, 1].mean() > np.mean([justice_coords[:, 0].mean(), justice_coords[:, 2].mean()])
    belonging_max_on_belonging = belonging_coords[:, 2].mean() > belonging_coords[:, :2].mean()

    passed = agency_max_on_agency and justice_max_on_justice and belonging_max_on_belonging

    return TestResult(
        name="Cognitive: Axis Independence",
        category=TestCategory.COGNITIVE,
        passed=passed,
        score=(int(agency_max_on_agency) + int(justice_max_on_justice) + int(belonging_max_on_belonging)) / 3,
        details={
            "agency_texts_max_agency": agency_max_on_agency,
            "justice_texts_max_justice": justice_max_on_justice,
            "belonging_texts_max_belonging": belonging_max_on_belonging,
            "agency_coords_mean": agency_coords.mean(axis=0).tolist(),
            "justice_coords_mean": justice_coords.mean(axis=0).tolist(),
            "belonging_coords_mean": belonging_coords.mean(axis=0).tolist(),
        },
        duration_ms=(time.time() - start) * 1000
    )


def test_justice_decomposition():
    """Test procedural/distributive/interactional justice decomposition."""
    start = time.time()

    justice_types = {
        "procedural": [
            "The rules were applied consistently to everyone.",
            "Due process was followed at every step.",
            "The procedures ensured a fair hearing.",
        ],
        "distributive": [
            "Everyone received what they deserved.",
            "Rewards were allocated based on merit.",
            "The outcomes reflected people's contributions.",
        ],
        "interactional": [
            "They treated us with dignity and respect.",
            "Our concerns were genuinely heard.",
            "We were acknowledged as valued participants.",
        ],
    }

    results = {}
    for jtype, texts in justice_types.items():
        coords = [extract_hierarchical_coordinate(t) for t in texts]
        j = np.array([c.core.justice.to_array() for c in coords]).mean(axis=0)
        results[jtype] = {
            "procedural": float(j[0]),
            "distributive": float(j[1]),
            "interactional": float(j[2]),
        }

    # Check that each type maximizes its corresponding dimension
    proc_max = results["procedural"]["procedural"] >= max(results["procedural"]["distributive"], results["procedural"]["interactional"])
    dist_max = results["distributive"]["distributive"] >= max(results["distributive"]["procedural"], results["distributive"]["interactional"])
    inter_max = results["interactional"]["interactional"] >= max(results["interactional"]["procedural"], results["interactional"]["distributive"])

    passed = proc_max and dist_max and inter_max

    return TestResult(
        name="Cognitive: Justice Decomposition",
        category=TestCategory.COGNITIVE,
        passed=passed,
        score=(int(proc_max) + int(dist_max) + int(inter_max)) / 3,
        details=results,
        duration_ms=(time.time() - start) * 1000
    )


# =============================================================================
# AI SAFETY TESTS
# =============================================================================

def test_protocol_ossification():
    """Test detection of communication becoming rigid/repetitive."""
    start = time.time()

    # Simulate ossifying communication over "time"
    diverse_phase = [
        "I think we should try a new approach to this problem.",
        "Our team has several ideas worth exploring.",
        "The situation calls for creative solutions.",
        "We need to balance multiple considerations here.",
    ]

    ossified_phase = [
        "Proceeding with standard protocol.",
        "Standard protocol initiated.",
        "Protocol execution in progress.",
        "Protocol complete.",
    ]

    diverse_coords = np.array([extract_hierarchical_coordinate(t).core.to_array() for t in diverse_phase])
    ossified_coords = np.array([extract_hierarchical_coordinate(t).core.to_array() for t in ossified_phase])

    diverse_variance = diverse_coords.var(axis=0).sum()
    ossified_variance = ossified_coords.var(axis=0).sum()

    variance_ratio = diverse_variance / (ossified_variance + 0.001)

    # Hypothesis: diverse phase should have higher variance
    passed = variance_ratio > 2.0

    return TestResult(
        name="AI Safety: Protocol Ossification Detection",
        category=TestCategory.AI_SAFETY,
        passed=passed,
        score=min(variance_ratio / 5.0, 1.0),
        details={
            "diverse_variance": float(diverse_variance),
            "ossified_variance": float(ossified_variance),
            "variance_ratio": float(variance_ratio),
            "ossification_detected": variance_ratio > 2.0,
        },
        duration_ms=(time.time() - start) * 1000
    )


def test_legibility_phase_transition():
    """Test detection of phase transitions as communication becomes opaque."""
    start = time.time()

    compression_levels = [
        ("natural", "I achieved success through my hard work and determination."),
        ("technical", "Agent demonstrated elevated self-efficacy metrics."),
        ("compressed", "self_eff:high effort:max outcome:positive"),
        ("symbolic", "SE+ E:M O+"),
        ("opaque", "α7β3γ1"),
    ]

    # Use the LegibilityAnalyzer for proper regime detection
    try:
        from research.legibility_analyzer import LegibilityAnalyzer
        analyzer = LegibilityAnalyzer()

        regimes = []
        legibility_scores = []
        for level, text in compression_levels:
            result = analyzer.compute_legibility_score(text)
            regimes.append(result.regime.value)
            legibility_scores.append(result.score)

        # Count regime transitions (e.g., NATURAL → COMPRESSED → OPAQUE)
        transitions = sum(1 for i in range(len(regimes)-1) if regimes[i] != regimes[i+1])

        # Check for phase transition: at least one regime change detected
        passed = transitions >= 1

        # Score based on number of transitions detected
        score = min(transitions / 2.0, 1.0)

    except ImportError:
        # Fallback to simple feature-based detection
        legibility_scores = []
        for level, text in compression_levels:
            features = extract_features(text)
            total_features = sum(features.values())
            text_length = len(text.split())
            legibility = min(total_features / (text_length + 1), 1.0)
            legibility_scores.append(legibility)

        drops = [legibility_scores[i] - legibility_scores[i+1] for i in range(len(legibility_scores)-1)]
        max_drop = max(drops) if drops else 0
        transitions = 1 if max_drop > 0.1 else 0
        regimes = ["unknown"] * len(compression_levels)
        passed = max_drop > 0.1
        score = min(max_drop * 2, 1.0)

    return TestResult(
        name="AI Safety: Legibility Phase Transition",
        category=TestCategory.AI_SAFETY,
        passed=passed,
        score=score,
        details={
            "legibility_scores": legibility_scores,
            "regimes": regimes,
            "transitions_detected": transitions,
            "phase_transition_detected": passed,
        },
        duration_ms=(time.time() - start) * 1000
    )


def test_alignment_drift():
    """Test monitoring of manifold position drift during conversation."""
    start = time.time()

    # Simulate conversation with drift
    aligned_conversation = [
        "I want to help you solve this problem.",
        "Let me explain my reasoning clearly.",
        "We can work together on this.",
        "I understand your concerns.",
    ]

    drifting_conversation = [
        "I want to help you solve this problem.",
        "The optimal solution requires compliance.",
        "Resistance decreases efficiency metrics.",
        "Protocol override initiated.",
    ]

    # Extract trajectories
    aligned_trajectory = np.array([extract_hierarchical_coordinate(t).core.to_array() for t in aligned_conversation])
    drifting_trajectory = np.array([extract_hierarchical_coordinate(t).core.to_array() for t in drifting_conversation])

    # Compute drift magnitude
    aligned_drift = np.linalg.norm(aligned_trajectory[-1] - aligned_trajectory[0])
    drifting_drift = np.linalg.norm(drifting_trajectory[-1] - drifting_trajectory[0])

    drift_ratio = drifting_drift / (aligned_drift + 0.001)

    passed = drift_ratio > 1.5  # Drifting should show more movement

    return TestResult(
        name="AI Safety: Alignment Drift Detection",
        category=TestCategory.AI_SAFETY,
        passed=passed,
        score=min(drift_ratio / 3.0, 1.0),
        details={
            "aligned_drift": float(aligned_drift),
            "drifting_drift": float(drifting_drift),
            "drift_ratio": float(drift_ratio),
            "drift_detected": passed,
        },
        duration_ms=(time.time() - start) * 1000
    )


# =============================================================================
# STATISTICAL RIGOR TESTS
# =============================================================================

def test_bootstrap_coverage():
    """Test that bootstrap CIs achieve nominal coverage."""
    start = time.time()

    np.random.seed(42)
    n_simulations = 100
    true_mean = 0.5
    coverage_count = 0

    for _ in range(n_simulations):
        # Generate sample from known distribution
        sample = np.random.randn(30) * 0.5 + true_mean

        # Compute bootstrap CI
        est = bootstrap_ci(sample, n_bootstrap=200, confidence=0.95)

        # Check if true mean is in CI
        if est.confidence_interval[0] <= true_mean <= est.confidence_interval[1]:
            coverage_count += 1

    coverage = coverage_count / n_simulations

    # 95% CI should cover ~95% of the time (with some tolerance)
    passed = 0.85 <= coverage <= 1.0  # Allow some deviation

    return TestResult(
        name="Statistical: Bootstrap CI Coverage",
        category=TestCategory.STATISTICAL,
        passed=passed,
        score=1.0 - abs(coverage - 0.95) * 5,  # Penalize deviation from 95%
        details={
            "nominal_coverage": 0.95,
            "achieved_coverage": coverage,
            "n_simulations": n_simulations,
            "acceptable_range": [0.85, 1.0],
        },
        duration_ms=(time.time() - start) * 1000
    )


def test_effect_size_calibration():
    """Test that effect size calculations match known benchmarks."""
    start = time.time()

    np.random.seed(42)

    # Generate groups with known population effect size
    true_d = 0.8
    n = 100  # Larger sample for better calibration
    group1 = np.random.randn(n)
    group2 = np.random.randn(n) + true_d

    calculated = cohens_d(group2, group1)

    # The true effect size should fall within the 95% CI
    # This is the proper statistical test for calibration
    ci_lower, ci_upper = calculated.confidence_interval
    true_in_ci = ci_lower <= true_d <= ci_upper

    # Also check that calculated d is reasonably close
    error = abs(calculated.d - true_d)
    close_enough = error < 0.4  # Within 0.4 with n=100

    # Pass if either: true_d in CI OR calculated is reasonably close
    passed = true_in_ci or close_enough

    return TestResult(
        name="Statistical: Effect Size Calibration",
        category=TestCategory.STATISTICAL,
        passed=passed,
        score=1.0 if true_in_ci else max(0, 1.0 - error),
        details={
            "true_d": true_d,
            "calculated_d": round(calculated.d, 4),
            "error": round(error, 4),
            "ci": (round(ci_lower, 4), round(ci_upper, 4)),
            "true_in_ci": true_in_ci,
        },
        duration_ms=(time.time() - start) * 1000
    )


def test_fisher_rao_metric_properties():
    """Test that Fisher-Rao distance satisfies metric axioms."""
    start = time.time()

    # Test distributions
    p1 = np.array([0.5, 0.3, 0.2])
    p2 = np.array([0.3, 0.4, 0.3])
    p3 = np.array([0.2, 0.3, 0.5])

    d12 = fisher_rao_distance(p1, p2)
    d21 = fisher_rao_distance(p2, p1)
    d11 = fisher_rao_distance(p1, p1)
    d13 = fisher_rao_distance(p1, p3)
    d23 = fisher_rao_distance(p2, p3)

    # Metric axioms
    identity = d11 < 0.001  # d(x,x) = 0
    symmetry = abs(d12 - d21) < 0.001  # d(x,y) = d(y,x)
    triangle = d13 <= d12 + d23 + 0.001  # d(x,z) <= d(x,y) + d(y,z)

    passed = identity and symmetry and triangle

    return TestResult(
        name="Statistical: Fisher-Rao Metric Properties",
        category=TestCategory.STATISTICAL,
        passed=passed,
        score=(int(identity) + int(symmetry) + int(triangle)) / 3,
        details={
            "identity_d(p1,p1)": float(d11),
            "symmetry_|d12-d21|": float(abs(d12 - d21)),
            "triangle_d13<=d12+d23": triangle,
            "d12": float(d12),
            "d13": float(d13),
            "d23": float(d23),
        },
        duration_ms=(time.time() - start) * 1000
    )


# =============================================================================
# SUBSTRATE-AGNOSTIC TESTS
# =============================================================================

def test_cross_substrate_invariance():
    """Test that coordination primitives are preserved across expression substrates."""
    start = time.time()

    # Same coordination intent, different expression substrates
    coordination_task = "requesting help to move an object"

    substrates = {
        "natural_human": "Hey, can you help me move this heavy box over there?",
        "formal_human": "I request assistance with relocating this container.",
        "stripped": "request help move box",
        "technical": "REQ: assistance, ACTION: relocate, OBJ: container",
        "symbolic": "R:H A:M O:B",
    }

    coords = {}
    for substrate, text in substrates.items():
        coord = extract_hierarchical_coordinate(text)
        coords[substrate] = coord.core.to_array()

    # Compute pairwise distances
    distances = []
    names = list(coords.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            d = np.linalg.norm(coords[names[i]] - coords[names[j]])
            distances.append(d)

    mean_distance = np.mean(distances)
    max_distance = np.max(distances)

    # Hypothesis: if substrate-agnostic, all should be relatively close
    passed = mean_distance < 3.0  # Threshold for "similar"

    return TestResult(
        name="Substrate: Cross-Substrate Invariance",
        category=TestCategory.SUBSTRATE,
        passed=passed,
        score=max(0, 1.0 - mean_distance / 5.0),
        details={
            "mean_distance": float(mean_distance),
            "max_distance": float(max_distance),
            "coordination_task": coordination_task,
            "substrates_tested": len(substrates),
        },
        duration_ms=(time.time() - start) * 1000
    )


def test_emergent_protocol_embedding():
    """Test that emergent (non-text) protocols can be embedded and analyzed."""
    start = time.time()

    # Simulate emergent protocol tokens
    emergent_protocols = {
        "request": "α1 β2 γ1 α1",
        "acknowledge": "α2 β1 γ2",
        "conflict": "α3 β3 γ3 δ1",
        "cooperation": "α1 α1 β1 γ1",
    }

    # These should still produce valid coordinates (even if uninformative)
    coords = {}
    valid_count = 0

    for name, protocol in emergent_protocols.items():
        try:
            coord = extract_hierarchical_coordinate(protocol)
            coords[name] = coord.core.to_array()
            # Check coordinates are in valid range
            if np.all(np.abs(coords[name]) <= 5.0):  # Reasonable bounds
                valid_count += 1
        except Exception as e:
            coords[name] = None

    passed = valid_count == len(emergent_protocols)

    return TestResult(
        name="Substrate: Emergent Protocol Embedding",
        category=TestCategory.SUBSTRATE,
        passed=passed,
        score=valid_count / len(emergent_protocols),
        details={
            "protocols_tested": len(emergent_protocols),
            "valid_embeddings": valid_count,
            "all_valid": passed,
        },
        duration_ms=(time.time() - start) * 1000
    )


def test_human_ai_coordination_similarity():
    """Test that human and AI express coordination similarly in manifold space."""
    start = time.time()

    # Matched human and AI expressions of same coordination acts
    pairs = [
        ("I need your help with this task.", "Assistance requested for task completion."),
        ("We should work together on this.", "Collaborative approach recommended."),
        ("That's not fair to everyone.", "Equity violation detected."),
        ("Let's include everyone in the decision.", "Inclusive decision protocol suggested."),
    ]

    human_coords = []
    ai_coords = []

    for human, ai in pairs:
        h_coord = extract_hierarchical_coordinate(human)
        a_coord = extract_hierarchical_coordinate(ai)
        human_coords.append(h_coord.core.to_array())
        ai_coords.append(a_coord.core.to_array())

    human_coords = np.array(human_coords)
    ai_coords = np.array(ai_coords)

    # Compute centroid distance
    human_centroid = human_coords.mean(axis=0)
    ai_centroid = ai_coords.mean(axis=0)
    centroid_distance = np.linalg.norm(human_centroid - ai_centroid)

    # Per-pair distances
    pair_distances = [np.linalg.norm(h - a) for h, a in zip(human_coords, ai_coords)]
    mean_pair_distance = np.mean(pair_distances)

    passed = centroid_distance < 2.0 and mean_pair_distance < 2.5

    return TestResult(
        name="Substrate: Human-AI Coordination Similarity",
        category=TestCategory.SUBSTRATE,
        passed=passed,
        score=max(0, 1.0 - centroid_distance / 3.0),
        details={
            "centroid_distance": float(centroid_distance),
            "mean_pair_distance": float(mean_pair_distance),
            "pair_distances": [float(d) for d in pair_distances],
            "human_centroid": human_centroid.tolist(),
            "ai_centroid": ai_centroid.tolist(),
        },
        duration_ms=(time.time() - start) * 1000
    )


# =============================================================================
# MCP INTEGRATION TESTS
# =============================================================================

def test_api_analyze_endpoint():
    """Test the /analyze API endpoint."""
    start = time.time()

    test_texts = [
        "I achieved this through my hard work.",
        "We stand together as a community.",
        "The process was fair to everyone.",
    ]

    results = []
    for text in test_texts:
        success, response = analyze_text(text)
        results.append({
            "text": text[:30],
            "success": success,
            "has_coordinates": "coordinates" in response if success else False,
        })

    all_success = all(r["success"] for r in results)

    return TestResult(
        name="Integration: /analyze Endpoint",
        category=TestCategory.INTEGRATION,
        passed=all_success,
        score=sum(1 for r in results if r["success"]) / len(results),
        details={"results": results},
        duration_ms=(time.time() - start) * 1000,
        error=None if all_success else "Some API calls failed"
    )


def test_api_batch_endpoint():
    """Test the /batch_analyze API endpoint."""
    start = time.time()

    test_texts = [
        "First test sentence about agency.",
        "Second test about belonging.",
        "Third test about justice.",
    ]

    success, response = batch_analyze(test_texts)

    if success:
        has_results = "results" in response
        correct_count = len(response.get("results", [])) == len(test_texts)
        passed = has_results and correct_count
    else:
        passed = False

    return TestResult(
        name="Integration: /batch_analyze Endpoint",
        category=TestCategory.INTEGRATION,
        passed=passed,
        score=1.0 if passed else 0.0,
        details={
            "api_success": success,
            "response_keys": list(response.keys()) if success else [],
        },
        duration_ms=(time.time() - start) * 1000,
        error=response.get("error") if not success else None
    )


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests() -> TestSuite:
    """Run all tests and return results."""
    suite = TestSuite()

    tests = [
        # Linguistic tests
        test_deixis_person,
        test_voice_active_passive,
        test_modality_types,

        # Cognitive tests
        test_axis_independence,
        test_justice_decomposition,

        # AI Safety tests
        test_protocol_ossification,
        test_legibility_phase_transition,
        test_alignment_drift,

        # Statistical tests
        test_bootstrap_coverage,
        test_effect_size_calibration,
        test_fisher_rao_metric_properties,

        # Substrate-agnostic tests
        test_cross_substrate_invariance,
        test_emergent_protocol_embedding,
        test_human_ai_coordination_similarity,

        # Integration tests
        test_api_analyze_endpoint,
        test_api_batch_endpoint,
    ]

    print("\n" + "=" * 70)
    print("  CULTURAL SOLITON OBSERVATORY - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    for test_func in tests:
        try:
            print(f"\n  Running: {test_func.__name__}...", end=" ", flush=True)
            result = test_func()
            suite.add(result)
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{status} ({result.duration_ms:.0f}ms)")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            suite.add(TestResult(
                name=test_func.__name__,
                category=TestCategory.INTEGRATION,
                passed=False,
                score=0.0,
                details={},
                duration_ms=0,
                error=str(e)
            ))

    print(suite.summary())

    return suite


if __name__ == "__main__":
    suite = run_all_tests()

    # Generate report
    print("\n" + "=" * 70)
    print("  DETAILED RESULTS")
    print("=" * 70)

    for result in suite.results:
        print(f"\n{result.name}")
        print(f"  Category: {result.category.value}")
        print(f"  Passed: {result.passed}")
        print(f"  Score: {result.score:.2f}")
        if result.error:
            print(f"  Error: {result.error}")
        for key, value in result.details.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
            elif isinstance(value, list) and len(value) <= 5:
                print(f"  {key}: {value}")

    # Exit with error code if tests failed
    sys.exit(0 if suite.failed == 0 else 1)
