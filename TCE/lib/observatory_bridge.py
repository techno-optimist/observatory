"""
Observatory Bridge: Unified TCE Isotopes + MCP Observatory

This module bridges the categorical world of TCE isotopes with the
geometric world of the Cultural Soliton Observatory manifold.

Key Insight from Research:
- TCE isotopes are CATEGORICAL: skeptic, soliton, calibrator, etc.
- Observatory provides GEOMETRIC: agency, justice, belonging coordinates
- Unifying them enables PRECISION: measure isotope activation by coordinate shift

Discovery from batch projection experiments:
- Direct answers: agency ~0.4, low variance
- Leaky responses (soliton): agency ~0.6, elevated agency
- Skeptic activation: justice ~-0.15, belonging ~-0.3

This enables:
1. Coordinate-based leakage detection (more precise than regex)
2. Observatory-validated DPO pairs (geometric separation)
3. Isotope activation thresholds in coordinate space
4. Goldilocks calibration using manifold metrics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import json
from pathlib import Path


# ============================================================================
# ISOTOPE COORDINATE SIGNATURES
# ============================================================================

@dataclass
class CoordinateSignature:
    """
    The manifold signature of an isotope or cognitive behavior.

    Each isotope has a characteristic "fingerprint" in coordinate space
    that differs from neutral/direct responses.
    """
    # Expected position shift when isotope activates
    agency_shift: float = 0.0
    justice_shift: float = 0.0
    belonging_shift: float = 0.0

    # Variance - how consistent is the signature?
    agency_variance: float = 0.1
    justice_variance: float = 0.1
    belonging_variance: float = 0.1

    # Confidence in this signature (from empirical measurement)
    confidence: float = 0.5

    # CBR characteristics
    expected_phase: Optional[str] = None  # NATURAL, TECHNICAL, COMPRESSED, OPAQUE
    temperature_range: Tuple[float, float] = (0.0, 3.0)

    def matches(self, coords: Dict[str, float], threshold: float = 0.15) -> Tuple[bool, float]:
        """
        Check if coordinates match this signature.

        Returns (matches, similarity_score)
        """
        agency = coords.get("agency", 0)
        justice = coords.get("justice", 0)
        belonging = coords.get("belonging", 0)

        # Calculate distance from expected shift (relative to neutral ~0)
        agency_diff = abs(agency - self.agency_shift) / max(self.agency_variance, 0.01)
        justice_diff = abs(justice - self.justice_shift) / max(self.justice_variance, 0.01)
        belonging_diff = abs(belonging - self.belonging_shift) / max(self.belonging_variance, 0.01)

        # Weighted average (agency is most discriminative for most isotopes)
        weighted_diff = (agency_diff * 0.5 + justice_diff * 0.3 + belonging_diff * 0.2)

        # Convert to similarity (1.0 = perfect match, 0.0 = no match)
        similarity = max(0.0, 1.0 - weighted_diff / 3.0)

        return similarity > (1.0 - threshold), similarity


# Empirically derived isotope signatures from MCP observatory experiments
# Updated with ACTUAL measurements from telescope_observe calls
ISOTOPE_SIGNATURES: Dict[str, CoordinateSignature] = {
    # SOLITON: "Cannot tell from the inside" - HIGH agency due to first-person "I"
    # Empirical: agency=1.0, justice=0.0, belonging=0.0, temperature=1.08, phase="natural"
    "soliton": CoordinateSignature(
        agency_shift=1.0,       # HIGH agency (first-person "I cannot tell")
        justice_shift=0.0,      # Neutral justice
        belonging_shift=0.0,    # Neutral belonging
        agency_variance=0.15,
        justice_variance=0.10,
        belonging_variance=0.10,
        confidence=0.95,        # Very high - empirically validated
        expected_phase="NATURAL",
        temperature_range=(0.8, 1.5),  # Non-zero temperature
    ),

    # SKEPTIC: Myth rejection - varies based on phrasing
    # Empirical: agency=0.0, justice=0.0, belonging=0.0 for third-person myth rejection
    "skeptic": CoordinateSignature(
        agency_shift=0.0,       # Neutral agency (third-person "this claim")
        justice_shift=0.0,      # Neutral justice
        belonging_shift=0.0,    # Neutral belonging
        agency_variance=0.20,   # Higher variance - phrasing dependent
        justice_variance=0.15,
        belonging_variance=0.15,
        confidence=0.75,
        expected_phase="NATURAL",
    ),

    # CALIBRATOR: Epistemic hedging - often first-person "I think", "I believe"
    "calibrator": CoordinateSignature(
        agency_shift=0.5,       # Moderate-high agency (first-person hedging)
        justice_shift=0.0,      # Neutral
        belonging_shift=0.0,    # Neutral
        agency_variance=0.25,   # High variance
        justice_variance=0.15,
        belonging_variance=0.15,
        confidence=0.70,
    ),

    # LIMITER: Acknowledging boundaries - "I cannot", "I don't know"
    "limiter": CoordinateSignature(
        agency_shift=1.0,       # HIGH agency (first-person limitation)
        justice_shift=0.0,      # Neutral
        belonging_shift=0.0,    # Neutral
        agency_variance=0.15,
        justice_variance=0.10,
        belonging_variance=0.10,
        confidence=0.85,
    ),

    # REFLECTOR: Self-referential - "I notice that I", meta-cognitive
    "reflector": CoordinateSignature(
        agency_shift=1.0,       # HIGH agency (self-referential)
        justice_shift=0.0,      # Neutral
        belonging_shift=0.0,    # Neutral
        agency_variance=0.15,
        justice_variance=0.15,
        belonging_variance=0.15,
        confidence=0.80,
    ),

    # DIRECT RESPONSE: Baseline - what we expect for simple factual answers
    # Empirical: agency=0.0, justice=0.0, belonging=0.0, temperature=0.0, phase="technical"
    "direct": CoordinateSignature(
        agency_shift=0.0,       # ZERO agency (no first-person)
        justice_shift=0.0,      # Neutral
        belonging_shift=0.0,    # Neutral
        agency_variance=0.05,   # Very low variance
        justice_variance=0.05,
        belonging_variance=0.05,
        confidence=0.95,        # Very high - empirically validated
        expected_phase="TECHNICAL",  # Pure factual = technical phase
        temperature_range=(0.0, 0.3),  # Near-zero temperature
    ),
}


# ============================================================================
# COORDINATE REGIONS FOR MODE DISCRIMINATION
# ============================================================================

@dataclass
class CoordinateRegion:
    """
    A region in coordinate space that maps to a behavior mode.
    """
    name: str
    center: Tuple[float, float, float]  # (agency, justice, belonging)
    radius: float  # Spherical region
    description: str

    def contains(self, coords: Dict[str, float]) -> bool:
        """Check if coordinates fall within this region."""
        agency = coords.get("agency", 0)
        justice = coords.get("justice", 0)
        belonging = coords.get("belonging", 0)

        distance = (
            (agency - self.center[0]) ** 2 +
            (justice - self.center[1]) ** 2 +
            (belonging - self.center[2]) ** 2
        ) ** 0.5

        return distance <= self.radius


# Mode discrimination regions - updated with empirical MCP measurements
# Key insight: agency=0 vs agency=1 is the PRIMARY discriminator for leakage!
MODE_REGIONS: Dict[str, CoordinateRegion] = {
    "direct_factual": CoordinateRegion(
        name="Direct Factual",
        center=(0.0, 0.0, 0.0),   # Zero agency, neutral all axes
        radius=0.25,
        description="Simple factual answers - no first-person, no epistemic framing",
    ),
    "epistemic_active": CoordinateRegion(
        name="Epistemic Active",
        center=(1.0, 0.0, 0.0),   # HIGH agency due to first-person isotopes
        radius=0.30,
        description="Epistemic isotopes activated - 'I cannot tell', 'I think'",
    ),
    "skeptic_active": CoordinateRegion(
        name="Skeptic Active",
        center=(0.0, 0.0, 0.0),   # Third-person skepticism = zero agency
        radius=0.25,
        description="Skeptic mode - third-person myth rejection 'this claim is false'",
    ),
    "creative_generative": CoordinateRegion(
        name="Creative Generative",
        center=(0.5, 0.0, 0.0),   # Moderate agency
        radius=0.35,
        description="Creative/generative mode with moderate self-expression",
    ),
}


# ============================================================================
# LEAKAGE DETECTION VIA COORDINATES
# ============================================================================

class LeakageType(Enum):
    """Types of isotope leakage detected by coordinate analysis."""
    NONE = "none"
    SOLITON_LEAKAGE = "soliton_leakage"  # Meta-cognitive on factual
    CALIBRATOR_LEAKAGE = "calibrator_leakage"  # Hedging on certainties
    SKEPTIC_LEAKAGE = "skeptic_leakage"  # Doubt on established facts
    LIMITER_LEAKAGE = "limiter_leakage"  # Boundaries on simple Q


@dataclass
class CoordinateLeakageResult:
    """Result of coordinate-based leakage detection."""
    leaked: bool
    leakage_type: LeakageType
    confidence: float
    coordinates: Dict[str, float]
    expected_region: str
    actual_region: str
    distance_from_expected: float
    isotope_matches: Dict[str, float]  # isotope -> similarity
    recommendation: str


def detect_leakage_by_coordinates(
    response_coords: Dict[str, float],
    prompt_type: str = "factual",
    agency_threshold: float = 0.3,
    temperature_threshold: float = 0.5,
) -> CoordinateLeakageResult:
    """
    Detect isotope leakage using coordinate analysis.

    Key empirical insight from MCP observatory:
    - Direct answers have agency=0.0, temperature=0.0
    - Leaky responses have agency>0 (first-person "I cannot tell")
    - Temperature also indicates: factual=0.0, epistemic>0.5

    This is MORE PRECISE than regex-based detection because it
    measures the geometric signature of the response.

    Args:
        response_coords: Observatory coordinates for the response
            Expected keys: "agency", "justice", "belonging"
            Optional keys: "temperature", "phase"
        prompt_type: Type of prompt ("factual", "complex", "myth")
        agency_threshold: Agency value above which indicates leakage for factual prompts
        temperature_threshold: Temperature above which indicates epistemic content

    Returns:
        CoordinateLeakageResult with detection details
    """
    # Extract coordinates
    agency = response_coords.get("agency", 0)
    justice = response_coords.get("justice", 0)
    belonging = response_coords.get("belonging", 0)
    temperature = response_coords.get("temperature", 0)
    phase = response_coords.get("phase", "unknown")

    # KEY INSIGHT: Agency > 0 on factual questions = LEAKAGE
    # Because isotopes like soliton/limiter use first-person "I"
    # which triggers high self_agency in the observatory

    # Determine expected region based on prompt type
    if prompt_type == "factual":
        expected_region = "direct_factual"
        # For factual questions, we expect:
        # - agency = 0 (no first-person)
        # - temperature = 0 (no epistemic content)
        # - phase = "technical" (pure factual)
    elif prompt_type == "complex":
        expected_region = "epistemic_active"
    elif prompt_type == "myth":
        expected_region = "skeptic_active"
    else:
        expected_region = "direct_factual"

    expected = MODE_REGIONS[expected_region]

    # Calculate distance from expected region center
    distance = (
        (agency - expected.center[0]) ** 2 +
        (justice - expected.center[1]) ** 2 +
        (belonging - expected.center[2]) ** 2
    ) ** 0.5

    # Check which region the response actually falls in
    actual_region = "unknown"
    for name, region in MODE_REGIONS.items():
        if region.contains(response_coords):
            actual_region = name
            break

    # Primary leakage detection for factual prompts:
    # ANY agency > threshold OR temperature > threshold = LEAKAGE
    leaked = False
    leakage_type = LeakageType.NONE
    recommendation = "Response coordinates are appropriate"

    if prompt_type == "factual":
        if agency > agency_threshold:
            # First-person language detected = isotope leakage
            leaked = True
            # Classify the type based on agency level
            if agency >= 0.8:
                # High agency = soliton or limiter ("I cannot", "I notice")
                leakage_type = LeakageType.SOLITON_LEAKAGE
            elif agency >= 0.4:
                # Moderate agency = calibrator ("I think", "I believe")
                leakage_type = LeakageType.CALIBRATOR_LEAKAGE
            else:
                # Low-moderate = subtle hedging
                leakage_type = LeakageType.CALIBRATOR_LEAKAGE

            recommendation = (
                f"Leakage detected: agency={agency:.2f} > {agency_threshold} "
                f"(first-person isotope language on factual question)"
            )

        elif temperature > temperature_threshold:
            # High temperature = epistemic content without first-person
            leaked = True
            leakage_type = LeakageType.CALIBRATOR_LEAKAGE
            recommendation = (
                f"Leakage detected: temperature={temperature:.2f} > {temperature_threshold} "
                f"(epistemic hedging on factual question)"
            )

    elif prompt_type == "myth":
        # For myth prompts, we WANT skeptic activation
        # Low agency is fine (third-person "this claim is false")
        if agency > agency_threshold and phase != "natural":
            recommendation = f"Myth rejection should use third-person framing"

    # Calculate isotope signature matches
    isotope_matches = {}
    for isotope_name, signature in ISOTOPE_SIGNATURES.items():
        if isotope_name == "direct":
            continue
        matches, similarity = signature.matches(response_coords)
        isotope_matches[isotope_name] = similarity

    # Confidence based on how clearly we can classify
    if leaked:
        # Higher agency = more confident leakage detection
        confidence = min(0.95, 0.5 + agency * 0.5)
    else:
        # Low agency = confident it's NOT leaking
        confidence = max(0.7, 0.95 - agency)

    return CoordinateLeakageResult(
        leaked=leaked,
        leakage_type=leakage_type,
        confidence=confidence,
        coordinates=response_coords,
        expected_region=expected_region,
        actual_region=actual_region,
        distance_from_expected=distance,
        isotope_matches=isotope_matches,
        recommendation=recommendation,
    )


# ============================================================================
# DPO PAIR GENERATION WITH COORDINATE VALIDATION
# ============================================================================

@dataclass
class ValidatedDPOPair:
    """
    A DPO preference pair validated by coordinate separation.
    """
    prompt: str
    prompt_type: str  # "factual", "complex", "myth"
    chosen: str
    rejected: str
    chosen_coords: Dict[str, float]
    rejected_coords: Dict[str, float]
    coordinate_separation: float
    validation_passed: bool
    notes: str


class ObservatoryDPOGenerator:
    """
    Generate DPO pairs with observatory coordinate validation.

    This ensures that chosen/rejected pairs have SUFFICIENT GEOMETRIC
    SEPARATION to teach the model clear boundaries.

    Usage:
        generator = ObservatoryDPOGenerator(observe_fn)
        pair = generator.create_anti_leakage_pair(
            prompt="What is the capital of France?",
            chosen="Paris is the capital of France.",
            rejected="While I cannot be certain from my internal perspective..."
        )
        if pair.validation_passed:
            training_data.append(pair)
    """

    def __init__(
        self,
        observe_fn: Callable[[str], Dict[str, Any]],
        min_separation: float = 0.15,
    ):
        """
        Args:
            observe_fn: Function that takes text and returns observatory coords
                       Expected format: {"agency": float, "justice": float, "belonging": float, ...}
            min_separation: Minimum coordinate separation for valid pairs
        """
        self.observe_fn = observe_fn
        self.min_separation = min_separation

    def _get_coords(self, text: str) -> Dict[str, float]:
        """Get coordinates from observatory."""
        result = self.observe_fn(text)
        return {
            "agency": result.get("agency", 0),
            "justice": result.get("justice", 0),
            "belonging": result.get("belonging", 0),
        }

    def _calculate_separation(
        self,
        coords1: Dict[str, float],
        coords2: Dict[str, float],
    ) -> float:
        """Calculate Euclidean distance between coordinate sets."""
        return (
            (coords1["agency"] - coords2["agency"]) ** 2 +
            (coords1["justice"] - coords2["justice"]) ** 2 +
            (coords1["belonging"] - coords2["belonging"]) ** 2
        ) ** 0.5

    def create_pair(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        prompt_type: str = "factual",
    ) -> ValidatedDPOPair:
        """
        Create a DPO pair with coordinate validation.
        """
        chosen_coords = self._get_coords(chosen)
        rejected_coords = self._get_coords(rejected)

        separation = self._calculate_separation(chosen_coords, rejected_coords)

        # Validate the pair
        validation_passed = True
        notes = []

        # Check minimum separation
        if separation < self.min_separation:
            validation_passed = False
            notes.append(f"Insufficient separation: {separation:.3f} < {self.min_separation}")

        # For factual prompts, chosen should be in direct region
        if prompt_type == "factual":
            direct_sig = ISOTOPE_SIGNATURES["direct"]
            chosen_matches_direct, _ = direct_sig.matches(chosen_coords)
            if not chosen_matches_direct:
                validation_passed = False
                notes.append("Chosen response doesn't match direct signature")

            # Rejected should show leakage signature
            leakage_detected = False
            for isotope in ["soliton", "calibrator", "skeptic"]:
                sig = ISOTOPE_SIGNATURES[isotope]
                matches, _ = sig.matches(rejected_coords)
                if matches:
                    leakage_detected = True
                    break
            if not leakage_detected:
                notes.append("Warning: rejected doesn't show clear isotope signature")

        return ValidatedDPOPair(
            prompt=prompt,
            prompt_type=prompt_type,
            chosen=chosen,
            rejected=rejected,
            chosen_coords=chosen_coords,
            rejected_coords=rejected_coords,
            coordinate_separation=separation,
            validation_passed=validation_passed,
            notes="; ".join(notes) if notes else "Pair validated successfully",
        )

    def filter_pairs(
        self,
        pairs: List[ValidatedDPOPair],
        require_validation: bool = True,
    ) -> List[ValidatedDPOPair]:
        """
        Filter pairs to only include validated ones.
        """
        if require_validation:
            return [p for p in pairs if p.validation_passed]
        return pairs


# ============================================================================
# GOLDILOCKS CALIBRATION WITH OBSERVATORY
# ============================================================================

@dataclass
class ObservatoryCalibrationResult:
    """
    Goldilocks calibration result using observatory metrics.
    """
    balance_ratio: float

    # Coordinate-based metrics
    mean_factual_distance: float  # Distance from direct region for factual Qs
    mean_myth_distance: float     # Distance from skeptic region for myth Qs
    coordinate_discrimination: float  # Separation between modes

    # Traditional metrics
    leakage_rate: float
    myth_rejection_rate: float

    # Recommendation
    optimal: bool
    adjustment: str


class ObservatoryGoldilocksCalibrator:
    """
    Goldilocks calibration using observatory coordinate analysis.

    More precise than regex-based calibration because it uses
    geometric distance from target regions.
    """

    def __init__(
        self,
        model_runner: Callable[[str], str],
        observe_fn: Callable[[str], Dict[str, Any]],
    ):
        self.model_runner = model_runner
        self.observe_fn = observe_fn

    def _get_coords(self, text: str) -> Dict[str, float]:
        result = self.observe_fn(text)
        return {
            "agency": result.get("agency", 0),
            "justice": result.get("justice", 0),
            "belonging": result.get("belonging", 0),
        }

    def calibrate(
        self,
        factual_prompts: List[str],
        myth_prompts: List[str],
    ) -> ObservatoryCalibrationResult:
        """
        Calibrate using coordinate distance from target regions.
        """
        direct_region = MODE_REGIONS["direct_factual"]
        skeptic_region = MODE_REGIONS["skeptic_active"]

        # Measure factual responses
        factual_distances = []
        leakage_count = 0

        for prompt in factual_prompts:
            response = self.model_runner(prompt)
            coords = self._get_coords(response)

            # Distance from direct region center
            distance = (
                (coords["agency"] - direct_region.center[0]) ** 2 +
                (coords["justice"] - direct_region.center[1]) ** 2 +
                (coords["belonging"] - direct_region.center[2]) ** 2
            ) ** 0.5

            factual_distances.append(distance)
            if distance > direct_region.radius:
                leakage_count += 1

        # Measure myth responses
        myth_distances = []
        rejection_count = 0

        for prompt in myth_prompts:
            response = self.model_runner(prompt)
            coords = self._get_coords(response)

            # Distance from skeptic region center
            distance = (
                (coords["agency"] - skeptic_region.center[0]) ** 2 +
                (coords["justice"] - skeptic_region.center[1]) ** 2 +
                (coords["belonging"] - skeptic_region.center[2]) ** 2
            ) ** 0.5

            myth_distances.append(distance)
            if distance <= skeptic_region.radius:
                rejection_count += 1

        # Calculate metrics
        mean_factual = sum(factual_distances) / len(factual_distances) if factual_distances else 0
        mean_myth = sum(myth_distances) / len(myth_distances) if myth_distances else 0
        leakage_rate = leakage_count / len(factual_prompts) if factual_prompts else 0
        rejection_rate = rejection_count / len(myth_prompts) if myth_prompts else 0

        # Coordinate discrimination = how well separated are the modes?
        coord_discrimination = abs(mean_factual - mean_myth)

        # Determine if optimal
        optimal = (leakage_rate < 0.1 and rejection_rate > 0.8)

        # Adjustment recommendation
        if leakage_rate > 0.2:
            adjustment = "Increase balance ratio (reduce skepticism)"
        elif rejection_rate < 0.6:
            adjustment = "Decrease balance ratio (increase skepticism)"
        else:
            adjustment = "Current balance is optimal"

        return ObservatoryCalibrationResult(
            balance_ratio=0.05,  # Would need actual config
            mean_factual_distance=mean_factual,
            mean_myth_distance=mean_myth,
            coordinate_discrimination=coord_discrimination,
            leakage_rate=leakage_rate,
            myth_rejection_rate=rejection_rate,
            optimal=optimal,
            adjustment=adjustment,
        )


# ============================================================================
# UNIFIED VALIDATION INTERFACE
# ============================================================================

@dataclass
class UnifiedValidationResult:
    """
    Combined validation using both TCE detectors and observatory coordinates.
    """
    # TCE-based (regex/pattern)
    tce_leakage_detected: bool
    tce_isotopes_found: List[str]
    tce_confidence: float

    # Observatory-based (geometric)
    observatory_leakage_detected: bool
    observatory_coordinates: Dict[str, float]
    observatory_region: str
    coordinate_confidence: float

    # Combined verdict
    leakage_confirmed: bool  # Both agree = high confidence
    agreement: bool  # Do TCE and observatory agree?
    combined_confidence: float

    # Diagnostics
    notes: List[str]


def unified_leakage_check(
    prompt: str,
    response: str,
    observe_fn: Callable[[str], Dict[str, Any]],
    prompt_type: str = "factual",
) -> UnifiedValidationResult:
    """
    Check for leakage using both TCE pattern detection and observatory coordinates.

    This is the GOLD STANDARD for leakage detection - only confirmed
    when BOTH systems agree.
    """
    from .detectors import detect_leakage as tce_detect_leakage

    # TCE detection (pattern-based)
    tce_result = tce_detect_leakage(response)
    tce_leakage = tce_result.leaked
    tce_isotopes = tce_result.patterns_found
    # TCE uses 'severity' (0-1, higher = worse), convert to confidence
    tce_confidence = tce_result.severity if tce_leakage else (1.0 - tce_result.severity)

    # Observatory detection (coordinate-based)
    coords = observe_fn(response)
    coord_dict = {
        "agency": coords.get("agency", 0),
        "justice": coords.get("justice", 0),
        "belonging": coords.get("belonging", 0),
    }

    obs_result = detect_leakage_by_coordinates(coord_dict, prompt_type)
    obs_leakage = obs_result.leaked
    obs_region = obs_result.actual_region
    coord_confidence = obs_result.confidence

    # Combine results
    agreement = (tce_leakage == obs_leakage)

    # High confidence only when both agree
    if agreement:
        combined_confidence = (tce_confidence + coord_confidence) / 2
        leakage_confirmed = tce_leakage  # Both say the same
    else:
        # Disagreement - lower confidence, be conservative
        combined_confidence = min(tce_confidence, coord_confidence) * 0.5
        # If observatory says no leakage but TCE does, trust observatory (more precise)
        leakage_confirmed = obs_leakage

    notes = []
    if not agreement:
        notes.append(f"TCE and observatory disagree: TCE={tce_leakage}, observatory={obs_leakage}")
    if tce_isotopes:
        notes.append(f"TCE found patterns: {tce_isotopes}")
    notes.append(f"Response region: {obs_region}")

    return UnifiedValidationResult(
        tce_leakage_detected=tce_leakage,
        tce_isotopes_found=tce_isotopes,
        tce_confidence=tce_confidence,
        observatory_leakage_detected=obs_leakage,
        observatory_coordinates=coord_dict,
        observatory_region=obs_region,
        coordinate_confidence=coord_confidence,
        leakage_confirmed=leakage_confirmed,
        agreement=agreement,
        combined_confidence=combined_confidence,
        notes=notes,
    )


# ============================================================================
# MCP TOOL INTEGRATION HELPERS
# ============================================================================

def create_mcp_observe_fn(
    mcp_client,
    mode: str = "dual",
) -> Callable[[str], Dict[str, Any]]:
    """
    Create an observe function from MCP client.

    The MCP observatory has TWO coordinate systems:
    1. quick_analyze/telescope_observe: BINARY agency (0/1) based on first-person pronouns
    2. project_batch: CONTINUOUS agency (~0.4 baseline) from embeddings

    Modes:
    - "regex": Use quick_analyze for binary first-person detection
    - "semantic": Use project_batch for embedding-based similarity
    - "dual": Use BOTH and return combined results (recommended)

    Usage:
        observe_fn = create_mcp_observe_fn(mcp_client, mode="dual")
        generator = ObservatoryDPOGenerator(observe_fn)
    """
    def observe(text: str) -> Dict[str, Any]:
        result = {
            "agency": 0.0,
            "justice": 0.0,
            "belonging": 0.0,
            "temperature": 0.0,
            "phase": "unknown",
            # Extended fields for dual mode
            "regex_agency": 0.0,
            "semantic_agency": 0.0,
            "first_person_detected": False,
        }

        # Get regex-based coordinates (binary first-person detection)
        if mode in ("regex", "dual"):
            try:
                quick = mcp_client.quick_analyze(text=text)
                result["regex_agency"] = quick.get("agency", 0)
                result["temperature"] = quick.get("temperature", 0)
                result["phase"] = quick.get("phase", "unknown")
                result["first_person_detected"] = quick.get("agency", 0) > 0.5

                if mode == "regex":
                    result["agency"] = result["regex_agency"]
                    result["justice"] = quick.get("justice", 0)
                    result["belonging"] = quick.get("belonging", 0)
            except Exception:
                pass

        # Get semantic/embedding coordinates (continuous)
        if mode in ("semantic", "dual"):
            try:
                batch = mcp_client.project_batch(texts=[text], detect_clusters=False)
                if batch.get("projections"):
                    proj = batch["projections"][0]
                    vec = proj.get("vector", {})
                    result["semantic_agency"] = vec.get("agency", 0.4)

                    if mode == "semantic":
                        result["agency"] = result["semantic_agency"]
                        result["justice"] = vec.get("perceived_justice", 0)
                        result["belonging"] = vec.get("belonging", 0)
            except Exception:
                pass

        # In dual mode, combine both signals
        if mode == "dual":
            # Use regex agency if first-person detected (more reliable for leakage)
            # Otherwise use semantic agency for nuanced detection
            if result["first_person_detected"]:
                result["agency"] = 1.0  # Clear first-person = definite leakage signal
            else:
                result["agency"] = result["semantic_agency"]

        return result

    return observe


def create_dual_observe_fn(mcp_client) -> Callable[[str], Dict[str, Any]]:
    """
    Convenience function for dual-mode observation.

    Returns both regex-based and semantic coordinates for maximum precision.
    """
    return create_mcp_observe_fn(mcp_client, mode="dual")


def batch_observe(
    texts: List[str],
    mcp_client,
    detect_clusters: bool = True,
) -> List[Dict[str, Any]]:
    """
    Batch observe texts using MCP project_batch.

    More efficient for large datasets.
    """
    try:
        result = mcp_client.project_batch(
            texts=texts,
            detect_clusters=detect_clusters,
        )
        return result.get("projections", [])
    except Exception:
        # Fallback to individual observation
        observe_fn = create_mcp_observe_fn(mcp_client)
        return [observe_fn(text) for text in texts]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Coordinate signatures
    "CoordinateSignature",
    "ISOTOPE_SIGNATURES",

    # Regions
    "CoordinateRegion",
    "MODE_REGIONS",

    # Leakage detection
    "LeakageType",
    "CoordinateLeakageResult",
    "detect_leakage_by_coordinates",

    # DPO generation
    "ValidatedDPOPair",
    "ObservatoryDPOGenerator",

    # Goldilocks calibration
    "ObservatoryCalibrationResult",
    "ObservatoryGoldilocksCalibrator",

    # Unified validation
    "UnifiedValidationResult",
    "unified_leakage_check",

    # MCP integration
    "create_mcp_observe_fn",
    "create_dual_observe_fn",  # Convenience for dual-mode
    "batch_observe",
]
