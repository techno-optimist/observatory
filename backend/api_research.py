"""
API Endpoints for Emergent Language Research Tools

This module provides FastAPI endpoints for the Cultural Soliton Observatory's
emergent language research capabilities, focusing on understanding the minimal
structures necessary for AI-human coordination.

Endpoints:
- POST /api/research/grammar-deletion: Analyze which grammatical features are
  coordination-necessary vs decorative
- POST /api/research/legibility: Measure how interpretable/legible a text is
- POST /api/research/evolution: Track language evolution across a time series
- POST /api/research/calibrate: Compare human texts to minimal coordination forms
- POST /api/research/phase-transition: Detect phase transitions in signal history

The research reframe: If AI systems can coordinate with minimal symbols, what parts
of human grammar are actually doing coordination work vs cultural ornamentation?
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

# --- Grammar Deletion ---

class GrammarDeletionRequest(BaseModel):
    """Request for grammar deletion analysis."""
    text: str = Field(..., description="Text to analyze for grammatical necessity")
    threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Drift threshold for classifying features as 'necessary' (0-2)"
    )


class FeatureDrift(BaseModel):
    """Drift result for a single grammatical feature."""
    feature_name: str
    modified_text: str
    projection_drift: float
    axis_drifts: Dict[str, float]
    mode_changed: bool
    original_mode: str
    modified_mode: str
    classification: str  # "necessary" or "decorative"


class GrammarDeletionResponse(BaseModel):
    """Response from grammar deletion analysis."""
    original_text: str
    original_projection: Dict[str, float]
    original_mode: str
    feature_rankings: List[FeatureDrift]
    necessary_features: List[str]
    decorative_features: List[str]
    coordination_core: str


# --- Legibility Analysis ---

class LegibilitySingleRequest(BaseModel):
    """Request for single-text legibility analysis."""
    text: str = Field(..., description="Text to analyze for legibility")


class LegibilityBatchRequest(BaseModel):
    """Request for batch legibility analysis."""
    texts: List[str] = Field(..., description="List of texts to analyze")


class LegibilityMetrics(BaseModel):
    """Legibility metrics for a single text."""
    text: str
    legibility_score: float = Field(..., description="Overall legibility (0-1)")
    interpretability: Dict[str, float] = Field(
        ..., description="Per-axis interpretability scores"
    )
    confidence: float = Field(..., description="Classification confidence")
    stability_score: float = Field(..., description="Stability under perturbation")
    mode: str
    notes: List[str]


class LegibilityResponse(BaseModel):
    """Response from legibility analysis."""
    results: List[LegibilityMetrics]
    aggregate_legibility: Optional[float] = None
    aggregate_interpretability: Optional[Dict[str, float]] = None


# --- Evolution Tracking ---

class EvolutionRequest(BaseModel):
    """Request for language evolution tracking."""
    texts: List[str] = Field(..., min_length=2, description="Ordered sequence of texts")
    timestamps: List[str] = Field(
        ..., min_length=2,
        description="Timestamps for each text (ISO format)"
    )


class EvolutionPoint(BaseModel):
    """A single point in the evolution trajectory."""
    timestamp: str
    text_preview: str
    projection: Dict[str, float]
    mode: str
    confidence: float
    velocity: Optional[Dict[str, float]] = None


class StabilizationPoint(BaseModel):
    """A point where evolution stabilized."""
    timestamp: str
    text_preview: str
    projection: Dict[str, float]
    mode: str
    duration_stable: int  # Number of subsequent points that remained stable


class EvolutionResponse(BaseModel):
    """Response from evolution tracking analysis."""
    trajectory: List[EvolutionPoint]
    evolution_stage: str = Field(
        ..., description="Current evolution stage: 'diverging', 'converging', 'stable', 'oscillating'"
    )
    stabilization_points: List[StabilizationPoint]
    total_drift: float
    average_velocity: Dict[str, float]
    mode_transitions: List[Dict[str, str]]


# --- Calibration Comparison ---

class CalibrationRequest(BaseModel):
    """Request for calibration comparison."""
    human_texts: List[str] = Field(..., description="Human-authored texts")
    minimal_texts: List[str] = Field(..., description="Minimal/symbolic texts")


class FeatureClassification(BaseModel):
    """Classification of a grammatical feature."""
    feature_name: str
    human_contribution: float  # How much this feature affects human text projections
    minimal_contribution: float  # How much this feature affects minimal text projections
    classification: str  # "coordination_essential", "human_decorative", "shared_structure"


class CalibrationResponse(BaseModel):
    """Response from calibration comparison."""
    human_centroid: Dict[str, float]
    minimal_centroid: Dict[str, float]
    centroid_distance: float
    coordination_core: Dict[str, float]  # Shared projection space
    feature_classifications: List[FeatureClassification]
    overlap_score: float  # How much the distributions overlap
    human_modes: Dict[str, int]  # Mode distribution for human texts
    minimal_modes: Dict[str, int]  # Mode distribution for minimal texts


# --- Phase Transition Detection ---

class PhaseTransitionRequest(BaseModel):
    """Request for phase transition detection."""
    signal_history: List[str] = Field(
        ..., min_length=3,
        description="Ordered sequence of signals/texts"
    )
    window_size: int = Field(
        default=5,
        ge=2,
        le=50,
        description="Window size for detecting transitions"
    )


class PhaseMetrics(BaseModel):
    """Metrics before or after a transition."""
    mean_projection: Dict[str, float]
    mode_distribution: Dict[str, float]
    variance: Dict[str, float]
    dominant_mode: str


class TransitionPoint(BaseModel):
    """A detected phase transition."""
    index: int
    text_before: str
    text_after: str
    projection_before: Dict[str, float]
    projection_after: Dict[str, float]
    mode_before: str
    mode_after: str
    transition_magnitude: float


class PhaseTransitionResponse(BaseModel):
    """Response from phase transition detection."""
    transition_detected: bool
    transition_point: Optional[TransitionPoint] = None
    before_metrics: Optional[PhaseMetrics] = None
    after_metrics: Optional[PhaseMetrics] = None
    all_transitions: List[TransitionPoint]
    phase_stability: float  # 0-1, how stable phases are overall


# =============================================================================
# Helper Functions
# =============================================================================

async def get_projection_and_mode(text: str) -> Dict[str, Any]:
    """
    Get projection coordinates and mode for a text.

    Uses the internal analysis function to project text into the
    coordination manifold.
    """
    try:
        from api_observer_chat import analyze_text_internal
        result = await analyze_text_internal(text, include_forces=False)
        return result
    except ImportError:
        # Fallback to main projection system
        from main import (
            model_manager, embedding_extractor, current_projection,
            ModelType
        )
        from analysis.mode_classifier import get_mode_classifier

        model_id = "all-MiniLM-L6-v2"

        if current_projection is None:
            raise HTTPException(
                status_code=400,
                detail="No projection trained. Train a projection first."
            )

        if not model_manager.is_loaded(model_id):
            model_manager.load_model(model_id, ModelType.SENTENCE_TRANSFORMER)

        result = embedding_extractor.extract(text, model_id)
        coords = current_projection.project(result.embedding)

        classifier = get_mode_classifier()
        coords_array = np.array([coords.agency, coords.fairness, coords.belonging])
        mode_result = classifier.classify(coords_array)

        return {
            "vector": coords.to_dict(),
            "mode": {
                "primary_mode": mode_result["primary_mode"],
                "confidence": mode_result.get("confidence", 0.5)
            }
        }


def compute_projection_distance(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    """Compute Euclidean distance between two projections."""
    axes = ["agency", "perceived_justice", "belonging"]
    # Handle both 'fairness' and 'perceived_justice' naming
    total = 0.0
    for axis in axes:
        v1 = p1.get(axis, p1.get("fairness", 0.0) if axis == "perceived_justice" else 0.0)
        v2 = p2.get(axis, p2.get("fairness", 0.0) if axis == "perceived_justice" else 0.0)
        total += (v1 - v2) ** 2
    return float(np.sqrt(total))


def compute_axis_drifts(p1: Dict[str, float], p2: Dict[str, float]) -> Dict[str, float]:
    """Compute per-axis drift between two projections."""
    axes = ["agency", "perceived_justice", "belonging"]
    drifts = {}
    for axis in axes:
        v1 = p1.get(axis, p1.get("fairness", 0.0) if axis == "perceived_justice" else 0.0)
        v2 = p2.get(axis, p2.get("fairness", 0.0) if axis == "perceived_justice" else 0.0)
        drifts[axis] = float(v2 - v1)
    return drifts


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/grammar-deletion", response_model=GrammarDeletionResponse)
async def analyze_grammar_deletion(request: GrammarDeletionRequest):
    """
    Analyze which grammatical features are coordination-necessary vs decorative.

    This endpoint systematically removes linguistic features from the input text
    and measures how much the projection drifts in coordination space. Features
    that cause significant drift are "necessary" for coordination; those with
    minimal impact are "decorative."

    The insight: If AI systems can coordinate with minimal symbols, this reveals
    what parts of human grammar are doing actual coordination work.

    Returns:
    - feature_rankings: All features ranked by their impact on projection
    - necessary_features: Features that significantly affect coordination meaning
    - decorative_features: Features that can be removed without losing coordination
    - coordination_core: The text with only necessary features retained
    """
    try:
        from research.grammar_deletion_test import (
            GrammarDeletionAnalyzer,
            DELETION_FUNCTIONS
        )

        analyzer = GrammarDeletionAnalyzer(drift_threshold=request.threshold)
        analysis = await analyzer.analyze_text(request.text)

        # Convert to response format
        feature_rankings = []
        for deletion in analysis.deletions:
            classification = "necessary" if deletion.projection_drift > request.threshold else "decorative"
            feature_rankings.append(FeatureDrift(
                feature_name=deletion.feature_name,
                modified_text=deletion.modified_text[:200] + "..." if len(deletion.modified_text) > 200 else deletion.modified_text,
                projection_drift=round(deletion.projection_drift, 4),
                axis_drifts={k: round(v, 4) for k, v in deletion.axis_drifts.items()},
                mode_changed=deletion.mode_changed,
                original_mode=deletion.original_mode,
                modified_mode=deletion.modified_mode,
                classification=classification
            ))

        return GrammarDeletionResponse(
            original_text=analysis.original_text,
            original_projection={k: round(v, 4) for k, v in analysis.original_projection.items()},
            original_mode=analysis.original_mode,
            feature_rankings=feature_rankings,
            necessary_features=analysis.necessary_features,
            decorative_features=analysis.decorative_features,
            coordination_core=analysis.coordination_core
        )

    except ImportError as e:
        logger.error(f"Grammar deletion module not available: {e}")
        raise HTTPException(
            status_code=501,
            detail="Grammar deletion analysis module not available. Ensure research/grammar_deletion_test.py exists."
        )
    except Exception as e:
        logger.error(f"Grammar deletion analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/legibility", response_model=LegibilityResponse)
async def analyze_legibility(
    request: Optional[LegibilitySingleRequest] = None,
    batch_request: Optional[LegibilityBatchRequest] = None
):
    """
    Analyze how interpretable/legible a text or texts are in coordination space.

    Legibility measures how clearly a text's coordination intentions can be read.
    High legibility means the text projects to a clear, stable point with high
    confidence. Low legibility means the text is ambiguous or unstable.

    Interpretability metrics show how clearly each axis (agency, perceived_justice,
    belonging) is expressed in the text.

    Accepts either:
    - {"text": str} for single text analysis
    - {"texts": List[str]} for batch analysis
    """
    # Handle both single and batch requests
    if request and request.text:
        texts = [request.text]
    elif batch_request and batch_request.texts:
        texts = batch_request.texts
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'text' for single analysis or 'texts' for batch analysis"
        )

    try:
        from api_extensions import calculate_stability_score

        results = []

        for text in texts:
            # Get projection and mode
            analysis = await get_projection_and_mode(text)

            projection = analysis.get("vector", {})
            mode_info = analysis.get("mode", {})
            confidence = mode_info.get("confidence", 0.5)
            mode = mode_info.get("primary_mode", "NEUTRAL")

            # Calculate stability score using existing utility
            mode_result = {
                "primary_probability": mode_info.get("primary_probability", confidence),
                "secondary_probability": mode_info.get("secondary_probability", 0),
                "confidence": confidence,
                "boundary_distances": mode_info.get("boundary_distances", {})
            }
            stability = calculate_stability_score(mode_result)
            stability_score = stability.get("stability_score", 0.5)

            # Calculate per-axis interpretability based on magnitude and consistency
            interpretability = {}
            for axis in ["agency", "perceived_justice", "belonging"]:
                value = projection.get(axis, projection.get("fairness", 0.0) if axis == "perceived_justice" else 0.0)
                # Interpretability is higher when value is further from zero (clear signal)
                # and when stability is high
                axis_interpretability = min(1.0, abs(value) / 1.5) * (0.5 + 0.5 * stability_score)
                interpretability[axis] = round(axis_interpretability, 4)

            # Overall legibility combines confidence, stability, and signal strength
            avg_interpretability = sum(interpretability.values()) / len(interpretability)
            legibility_score = (0.4 * confidence + 0.3 * stability_score + 0.3 * avg_interpretability)

            # Generate notes about legibility issues
            notes = []
            if confidence < 0.5:
                notes.append("Low classification confidence - text is ambiguous")
            if stability_score < 0.4:
                notes.append("Low stability - classification may change under perturbation")
            if all(v < 0.3 for v in interpretability.values()):
                notes.append("Weak signal on all axes - text lacks clear coordination markers")
            if not notes:
                notes.append("Text has clear, stable coordination signals")

            results.append(LegibilityMetrics(
                text=text[:200] + "..." if len(text) > 200 else text,
                legibility_score=round(legibility_score, 4),
                interpretability=interpretability,
                confidence=round(confidence, 4),
                stability_score=round(stability_score, 4),
                mode=mode,
                notes=notes
            ))

        # Calculate aggregates for batch
        aggregate_legibility = None
        aggregate_interpretability = None

        if len(results) > 1:
            aggregate_legibility = round(
                sum(r.legibility_score for r in results) / len(results), 4
            )
            aggregate_interpretability = {}
            for axis in ["agency", "perceived_justice", "belonging"]:
                aggregate_interpretability[axis] = round(
                    sum(r.interpretability.get(axis, 0) for r in results) / len(results), 4
                )

        return LegibilityResponse(
            results=results,
            aggregate_legibility=aggregate_legibility,
            aggregate_interpretability=aggregate_interpretability
        )

    except Exception as e:
        logger.error(f"Legibility analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evolution", response_model=EvolutionResponse)
async def track_evolution(request: EvolutionRequest):
    """
    Track language evolution across a time series of texts.

    Analyzes how narratives evolve over time by projecting each text and
    measuring the trajectory through coordination space.

    Returns:
    - trajectory: Each text with its projection and velocity
    - evolution_stage: 'diverging', 'converging', 'stable', or 'oscillating'
    - stabilization_points: Points where evolution paused/stabilized
    - mode_transitions: When and how the mode changed
    """
    if len(request.texts) != len(request.timestamps):
        raise HTTPException(
            status_code=400,
            detail="texts and timestamps must have the same length"
        )

    try:
        trajectory = []
        mode_transitions = []
        projections = []

        prev_projection = None
        prev_mode = None

        for i, (text, timestamp) in enumerate(zip(request.texts, request.timestamps)):
            # Get projection
            analysis = await get_projection_and_mode(text)
            projection = analysis.get("vector", {})
            mode_info = analysis.get("mode", {})
            mode = mode_info.get("primary_mode", "NEUTRAL")
            confidence = mode_info.get("confidence", 0.5)

            projections.append(projection)

            # Calculate velocity from previous point
            velocity = None
            if prev_projection is not None:
                velocity = compute_axis_drifts(prev_projection, projection)

            trajectory.append(EvolutionPoint(
                timestamp=timestamp,
                text_preview=text[:100] + "..." if len(text) > 100 else text,
                projection={k: round(v, 4) for k, v in projection.items()},
                mode=mode,
                confidence=round(confidence, 4),
                velocity={k: round(v, 4) for k, v in velocity.items()} if velocity else None
            ))

            # Detect mode transitions
            if prev_mode is not None and mode != prev_mode:
                mode_transitions.append({
                    "index": i,
                    "timestamp": timestamp,
                    "from_mode": prev_mode,
                    "to_mode": mode
                })

            prev_projection = projection
            prev_mode = mode

        # Detect stabilization points
        stabilization_points = []
        stability_threshold = 0.15  # Max drift to consider "stable"

        for i in range(len(projections)):
            stable_count = 0
            for j in range(i + 1, len(projections)):
                drift = compute_projection_distance(projections[i], projections[j])
                if drift < stability_threshold:
                    stable_count += 1
                else:
                    break

            if stable_count >= 2:  # At least 2 subsequent stable points
                stabilization_points.append(StabilizationPoint(
                    timestamp=trajectory[i].timestamp,
                    text_preview=trajectory[i].text_preview,
                    projection=trajectory[i].projection,
                    mode=trajectory[i].mode,
                    duration_stable=stable_count
                ))

        # Calculate total drift and average velocity
        total_drift = 0.0
        velocities = {"agency": [], "perceived_justice": [], "belonging": []}

        for i in range(1, len(projections)):
            drift = compute_projection_distance(projections[i-1], projections[i])
            total_drift += drift

            axis_drifts = compute_axis_drifts(projections[i-1], projections[i])
            for axis, value in axis_drifts.items():
                velocities[axis].append(value)

        avg_velocity = {
            axis: round(float(np.mean(values)), 4) if values else 0.0
            for axis, values in velocities.items()
        }

        # Determine evolution stage
        if len(projections) < 3:
            evolution_stage = "insufficient_data"
        else:
            # Look at recent velocity trend
            recent_velocities = []
            for i in range(max(0, len(projections) - 4), len(projections) - 1):
                drift = compute_projection_distance(projections[i], projections[i + 1])
                recent_velocities.append(drift)

            if not recent_velocities:
                evolution_stage = "stable"
            else:
                avg_recent = np.mean(recent_velocities)
                velocity_trend = recent_velocities[-1] - recent_velocities[0] if len(recent_velocities) > 1 else 0

                if avg_recent < 0.1:
                    evolution_stage = "stable"
                elif velocity_trend > 0.1:
                    evolution_stage = "diverging"
                elif velocity_trend < -0.1:
                    evolution_stage = "converging"
                else:
                    # Check for oscillation
                    sign_changes = sum(
                        1 for i in range(1, len(recent_velocities))
                        if (recent_velocities[i] - 0.15) * (recent_velocities[i-1] - 0.15) < 0
                    )
                    evolution_stage = "oscillating" if sign_changes >= 2 else "stable"

        return EvolutionResponse(
            trajectory=trajectory,
            evolution_stage=evolution_stage,
            stabilization_points=stabilization_points,
            total_drift=round(total_drift, 4),
            average_velocity=avg_velocity,
            mode_transitions=mode_transitions
        )

    except Exception as e:
        logger.error(f"Evolution tracking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calibrate", response_model=CalibrationResponse)
async def calibrate_comparison(request: CalibrationRequest):
    """
    Compare human texts to minimal coordination forms.

    This endpoint helps identify the "coordination core" - the essential
    structures shared between verbose human language and minimal AI/symbolic
    communication.

    By comparing how human and minimal texts project, we can identify:
    - coordination_essential: Features that both rely on
    - human_decorative: Features that only appear in human text
    - shared_structure: The overlap in projection space
    """
    if len(request.human_texts) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 human texts for meaningful comparison"
        )
    if len(request.minimal_texts) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 minimal texts for meaningful comparison"
        )

    try:
        from research.grammar_deletion_test import DELETION_FUNCTIONS

        # Project all texts
        human_projections = []
        human_mode_counts = {}

        for text in request.human_texts:
            analysis = await get_projection_and_mode(text)
            projection = analysis.get("vector", {})
            mode = analysis.get("mode", {}).get("primary_mode", "NEUTRAL")
            human_projections.append(projection)
            human_mode_counts[mode] = human_mode_counts.get(mode, 0) + 1

        minimal_projections = []
        minimal_mode_counts = {}

        for text in request.minimal_texts:
            analysis = await get_projection_and_mode(text)
            projection = analysis.get("vector", {})
            mode = analysis.get("mode", {}).get("primary_mode", "NEUTRAL")
            minimal_projections.append(projection)
            minimal_mode_counts[mode] = minimal_mode_counts.get(mode, 0) + 1

        # Calculate centroids
        axes = ["agency", "perceived_justice", "belonging"]

        human_centroid = {}
        for axis in axes:
            values = [p.get(axis, p.get("fairness", 0.0) if axis == "perceived_justice" else 0.0)
                      for p in human_projections]
            human_centroid[axis] = round(float(np.mean(values)), 4)

        minimal_centroid = {}
        for axis in axes:
            values = [p.get(axis, p.get("fairness", 0.0) if axis == "perceived_justice" else 0.0)
                      for p in minimal_projections]
            minimal_centroid[axis] = round(float(np.mean(values)), 4)

        # Calculate centroid distance
        centroid_distance = compute_projection_distance(human_centroid, minimal_centroid)

        # Calculate coordination core (midpoint between centroids)
        coordination_core = {
            axis: round((human_centroid[axis] + minimal_centroid[axis]) / 2, 4)
            for axis in axes
        }

        # Calculate overlap score based on distribution overlap
        # Using a simple approach: how close are the distributions?
        overlap_score = max(0, 1 - centroid_distance / 2)  # Normalize to 0-1

        # Feature classification (analyze how features affect each group)
        feature_classifications = []

        # Test a subset of key features
        key_features = [
            "articles", "pronouns_first_person", "hedging",
            "agency_markers", "belonging_markers", "justice_markers"
        ]

        for feature_name in key_features:
            if feature_name not in DELETION_FUNCTIONS:
                continue

            deletion_func = DELETION_FUNCTIONS[feature_name]

            # Measure impact on human texts
            human_drifts = []
            for i, text in enumerate(request.human_texts[:3]):  # Sample for speed
                try:
                    modified = deletion_func(text)
                    if modified != text:
                        orig_proj = human_projections[i]
                        mod_analysis = await get_projection_and_mode(modified)
                        mod_proj = mod_analysis.get("vector", {})
                        drift = compute_projection_distance(orig_proj, mod_proj)
                        human_drifts.append(drift)
                except:
                    pass

            # Measure impact on minimal texts
            minimal_drifts = []
            for i, text in enumerate(request.minimal_texts[:3]):
                try:
                    modified = deletion_func(text)
                    if modified != text:
                        orig_proj = minimal_projections[i]
                        mod_analysis = await get_projection_and_mode(modified)
                        mod_proj = mod_analysis.get("vector", {})
                        drift = compute_projection_distance(orig_proj, mod_proj)
                        minimal_drifts.append(drift)
                except:
                    pass

            human_contribution = float(np.mean(human_drifts)) if human_drifts else 0.0
            minimal_contribution = float(np.mean(minimal_drifts)) if minimal_drifts else 0.0

            # Classify the feature
            if human_contribution > 0.2 and minimal_contribution > 0.2:
                classification = "coordination_essential"
            elif human_contribution > 0.2 and minimal_contribution <= 0.2:
                classification = "human_decorative"
            elif human_contribution <= 0.2 and minimal_contribution > 0.2:
                classification = "minimal_essential"
            else:
                classification = "shared_structure"

            feature_classifications.append(FeatureClassification(
                feature_name=feature_name,
                human_contribution=round(human_contribution, 4),
                minimal_contribution=round(minimal_contribution, 4),
                classification=classification
            ))

        return CalibrationResponse(
            human_centroid=human_centroid,
            minimal_centroid=minimal_centroid,
            centroid_distance=round(centroid_distance, 4),
            coordination_core=coordination_core,
            feature_classifications=feature_classifications,
            overlap_score=round(overlap_score, 4),
            human_modes=human_mode_counts,
            minimal_modes=minimal_mode_counts
        )

    except ImportError as e:
        logger.warning(f"Grammar deletion module not available for full calibration: {e}")
        # Return simplified response without feature classification
        # ... (simplified logic would go here)
        raise HTTPException(
            status_code=501,
            detail="Full calibration requires research/grammar_deletion_test.py module"
        )
    except Exception as e:
        logger.error(f"Calibration comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/phase-transition", response_model=PhaseTransitionResponse)
async def detect_phase_transition(request: PhaseTransitionRequest):
    """
    Detect phase transitions in signal history.

    Analyzes a sequence of signals to detect when a significant shift
    (phase transition) occurred in the coordination space. This is useful
    for identifying moments when collective narratives fundamentally changed.

    Returns:
    - transition_detected: Whether a significant transition was found
    - transition_point: Details about the most significant transition
    - before_metrics: Aggregate metrics before the transition
    - after_metrics: Aggregate metrics after the transition
    - all_transitions: All detected transitions above threshold
    """
    try:
        # Project all signals
        projections = []
        modes = []

        for text in request.signal_history:
            analysis = await get_projection_and_mode(text)
            projection = analysis.get("vector", {})
            mode = analysis.get("mode", {}).get("primary_mode", "NEUTRAL")
            projections.append(projection)
            modes.append(mode)

        # Detect transitions using sliding window
        window_size = min(request.window_size, len(projections) // 2)
        transition_threshold = 0.5  # Significant transition threshold

        all_transitions = []
        max_transition_magnitude = 0
        max_transition_idx = -1

        for i in range(window_size, len(projections) - window_size):
            # Calculate before and after window centroids
            before_window = projections[i - window_size:i]
            after_window = projections[i:i + window_size]

            axes = ["agency", "perceived_justice", "belonging"]

            before_centroid = {}
            for axis in axes:
                values = [p.get(axis, p.get("fairness", 0.0) if axis == "perceived_justice" else 0.0)
                          for p in before_window]
                before_centroid[axis] = float(np.mean(values))

            after_centroid = {}
            for axis in axes:
                values = [p.get(axis, p.get("fairness", 0.0) if axis == "perceived_justice" else 0.0)
                          for p in after_window]
                after_centroid[axis] = float(np.mean(values))

            # Calculate transition magnitude
            magnitude = compute_projection_distance(before_centroid, after_centroid)

            if magnitude > transition_threshold:
                all_transitions.append(TransitionPoint(
                    index=i,
                    text_before=request.signal_history[i - 1][:100] + "...",
                    text_after=request.signal_history[i][:100] + "...",
                    projection_before={k: round(v, 4) for k, v in projections[i - 1].items()},
                    projection_after={k: round(v, 4) for k, v in projections[i].items()},
                    mode_before=modes[i - 1],
                    mode_after=modes[i],
                    transition_magnitude=round(magnitude, 4)
                ))

            if magnitude > max_transition_magnitude:
                max_transition_magnitude = magnitude
                max_transition_idx = i

        # Calculate phase stability (inverse of average transition magnitude)
        if len(projections) > 1:
            all_magnitudes = []
            for i in range(1, len(projections)):
                mag = compute_projection_distance(projections[i - 1], projections[i])
                all_magnitudes.append(mag)
            avg_magnitude = float(np.mean(all_magnitudes))
            phase_stability = max(0, 1 - avg_magnitude)
        else:
            phase_stability = 1.0

        # Build response
        transition_detected = max_transition_magnitude > transition_threshold

        transition_point = None
        before_metrics = None
        after_metrics = None

        if transition_detected and max_transition_idx >= 0:
            # Find the most significant transition
            transition_point = TransitionPoint(
                index=max_transition_idx,
                text_before=request.signal_history[max_transition_idx - 1][:100] + "...",
                text_after=request.signal_history[max_transition_idx][:100] + "...",
                projection_before={k: round(v, 4) for k, v in projections[max_transition_idx - 1].items()},
                projection_after={k: round(v, 4) for k, v in projections[max_transition_idx].items()},
                mode_before=modes[max_transition_idx - 1],
                mode_after=modes[max_transition_idx],
                transition_magnitude=round(max_transition_magnitude, 4)
            )

            # Calculate before/after metrics
            before_projs = projections[:max_transition_idx]
            after_projs = projections[max_transition_idx:]
            before_mode_list = modes[:max_transition_idx]
            after_mode_list = modes[max_transition_idx:]

            if before_projs:
                axes = ["agency", "perceived_justice", "belonging"]
                before_mean = {}
                before_var = {}
                before_modes_dist = {}

                for axis in axes:
                    values = [p.get(axis, p.get("fairness", 0.0) if axis == "perceived_justice" else 0.0)
                              for p in before_projs]
                    before_mean[axis] = round(float(np.mean(values)), 4)
                    before_var[axis] = round(float(np.var(values)), 4)

                for m in before_mode_list:
                    before_modes_dist[m] = before_modes_dist.get(m, 0) + 1 / len(before_mode_list)

                dominant_before = max(before_modes_dist, key=before_modes_dist.get)

                before_metrics = PhaseMetrics(
                    mean_projection=before_mean,
                    mode_distribution={k: round(v, 4) for k, v in before_modes_dist.items()},
                    variance=before_var,
                    dominant_mode=dominant_before
                )

            if after_projs:
                after_mean = {}
                after_var = {}
                after_modes_dist = {}

                for axis in axes:
                    values = [p.get(axis, p.get("fairness", 0.0) if axis == "perceived_justice" else 0.0)
                              for p in after_projs]
                    after_mean[axis] = round(float(np.mean(values)), 4)
                    after_var[axis] = round(float(np.var(values)), 4)

                for m in after_mode_list:
                    after_modes_dist[m] = after_modes_dist.get(m, 0) + 1 / len(after_mode_list)

                dominant_after = max(after_modes_dist, key=after_modes_dist.get)

                after_metrics = PhaseMetrics(
                    mean_projection=after_mean,
                    mode_distribution={k: round(v, 4) for k, v in after_modes_dist.items()},
                    variance=after_var,
                    dominant_mode=dominant_after
                )

        return PhaseTransitionResponse(
            transition_detected=transition_detected,
            transition_point=transition_point,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            all_transitions=all_transitions,
            phase_stability=round(phase_stability, 4)
        )

    except Exception as e:
        logger.error(f"Phase transition detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Health Check
# =============================================================================

@router.get("/status")
async def research_status():
    """Get status of research API and available modules."""
    status = {
        "status": "operational",
        "endpoints": [
            {"path": "/grammar-deletion", "method": "POST", "description": "Grammar deletion analysis"},
            {"path": "/legibility", "method": "POST", "description": "Legibility analysis"},
            {"path": "/evolution", "method": "POST", "description": "Evolution tracking"},
            {"path": "/calibrate", "method": "POST", "description": "Calibration comparison"},
            {"path": "/phase-transition", "method": "POST", "description": "Phase transition detection"},
        ],
        "modules": {}
    }

    # Check grammar deletion module
    try:
        from research.grammar_deletion_test import GrammarDeletionAnalyzer, DELETION_FUNCTIONS
        status["modules"]["grammar_deletion"] = {
            "available": True,
            "features": list(DELETION_FUNCTIONS.keys())
        }
    except ImportError as e:
        status["modules"]["grammar_deletion"] = {
            "available": False,
            "error": str(e)
        }

    # Check projection system
    try:
        from main import current_projection
        status["modules"]["projection"] = {
            "available": True,
            "trained": current_projection is not None
        }
    except ImportError:
        status["modules"]["projection"] = {
            "available": False,
            "error": "Main projection system not available"
        }

    return status
