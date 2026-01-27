"""
API Extensions for Enhanced Observatory Features

This module adds new endpoints for:
- Enhanced 12-mode classification with soft labels and stability indicators
- Research agent integration
- Uncertainty-aware analysis
- Robustness testing
- Trajectory tracking

Import and mount these routes in main.py:
    from api_extensions import router as extensions_router
    app.include_router(extensions_router, prefix="/v2")

Response Field Documentation:
=============================

AXIS NAMING UPDATE (January 2026):
----------------------------------
The "fairness" axis has been renamed to "perceived_justice" based on validity study
findings that showed the axis conflates abstract fairness values with system legitimacy
beliefs. All API responses now use "perceived_justice" as the canonical name.

The old name "fairness" is still ACCEPTED in requests for backward compatibility
but is DEPRECATED. New code should use "perceived_justice".

Three Axes:
- agency: Sense of personal control and self-determination
- perceived_justice: Belief in fair treatment and system legitimacy (formerly "fairness")
- belonging: Sense of social connection and group membership

Mode Classification Fields:
---------------------------
- primary_mode: The most likely narrative mode classification
- primary_probability: Probability (0-1) assigned to the primary mode
- secondary_mode: The second most likely mode classification
- secondary_probability: Probability (0-1) assigned to the secondary mode
- all_probabilities: Full probability distribution across all 12 modes
- confidence: Overall classification confidence (0-1), based on probability gap
  and distance from center. Low confidence (<0.5) indicates ambiguous classification.
- is_boundary_case: True if classification is near a mode boundary (boundary_distance < 0.3
  or probability_gap < 0.15). Boundary cases are inherently less stable.
- stability_warning: Human-readable warning if classification may be unstable.
  Appears when confidence < 0.5 or is_boundary_case is True.

Stability Assessment Fields:
----------------------------
- stability_score: Quick stability assessment (0-1). Based on:
  * boundary_distance_factor: How far from nearest mode boundary
  * probability_gap_factor: Gap between primary and secondary mode probabilities
  Higher scores indicate more stable classifications.
- min_boundary_distance: Smallest distance to any mode centroid boundary
- probability_gap: Difference between primary and secondary mode probabilities

Interpretation Guidelines:
--------------------------
- confidence >= 0.7: High confidence classification, likely stable
- confidence 0.5-0.7: Moderate confidence, interpret with care
- confidence < 0.5: Low confidence, classification is ambiguous
- is_boundary_case = True: Near mode boundary, small changes could flip mode
- stability_score < 0.5: Classification may change under text perturbations
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import asyncio
import logging

# Import our new modules
from analysis.mode_classifier import (
    get_mode_classifier,
    classify_coordinates,
    legacy_mode,
    EnhancedModeClassifier
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Stability Assessment Utilities
# ============================================================================

def calculate_stability_score(
    mode_result: Dict[str, Any],
    boundary_threshold: float = 0.3,
    probability_gap_threshold: float = 0.15
) -> Dict[str, Any]:
    """
    Calculate quick stability score without full robustness testing.

    Args:
        mode_result: Classification result from EnhancedModeClassifier
        boundary_threshold: Distance below which classification is considered near boundary
        probability_gap_threshold: Gap below which classification is considered ambiguous

    Returns:
        Dictionary with stability metrics
    """
    # Extract probability gap
    primary_prob = mode_result.get("primary_probability", 0)
    secondary_prob = mode_result.get("secondary_probability", 0)
    probability_gap = primary_prob - secondary_prob

    # Find minimum boundary distance (distance to any mode centroid)
    boundary_distances = mode_result.get("boundary_distances", {})
    min_boundary_distance = float("inf")
    for key, dist in boundary_distances.items():
        if key.startswith("to_") and dist < min_boundary_distance:
            min_boundary_distance = dist

    if min_boundary_distance == float("inf"):
        min_boundary_distance = 0.0

    # Determine if this is a boundary case
    is_boundary_case = (
        min_boundary_distance < boundary_threshold or
        probability_gap < probability_gap_threshold
    )

    # Calculate stability score (0-1)
    # Factor 1: Boundary distance (normalized to 0-1, capped at 1.0 at distance 1.0)
    boundary_factor = min(min_boundary_distance / 1.0, 1.0)

    # Factor 2: Probability gap (normalized to 0-1)
    gap_factor = min(probability_gap / 0.5, 1.0)  # 0.5 gap = fully stable

    # Combined stability score
    stability_score = 0.5 * boundary_factor + 0.5 * gap_factor
    stability_score = max(0.0, min(1.0, stability_score))

    # Generate stability warning if needed
    stability_warning = None
    confidence = mode_result.get("confidence", 0)

    if confidence < 0.5:
        stability_warning = "Low confidence classification - may be unstable under perturbations"
    elif is_boundary_case:
        stability_warning = "Near mode boundary - classification may flip with small changes"

    return {
        "stability_score": float(round(stability_score, 4)),
        "min_boundary_distance": float(round(min_boundary_distance, 4)),
        "probability_gap": float(round(probability_gap, 4)),
        "is_boundary_case": bool(is_boundary_case),  # Convert numpy.bool_ to Python bool
        "stability_warning": stability_warning
    }


def build_enhanced_mode_response(mode_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build enhanced mode response with soft labels and stability indicators.

    This ensures ALL responses include:
    - Full probability distribution across all 12 modes
    - Boundary case flags
    - Stability warnings

    All numpy types are converted to native Python types for JSON serialization.
    """
    # Get all mode probabilities (not just top 5)
    mode_probabilities = mode_result.get("mode_probabilities", {})
    # Convert any numpy values to native Python floats
    mode_probabilities = {k: float(v) for k, v in mode_probabilities.items()}

    # Calculate stability metrics
    stability = calculate_stability_score(mode_result)

    return {
        "primary_mode": mode_result.get("primary_mode"),
        "primary_probability": float(mode_result.get("primary_probability", 0)),
        "secondary_mode": mode_result.get("secondary_mode"),
        "secondary_probability": float(mode_result.get("secondary_probability", 0)),
        "all_probabilities": mode_probabilities,
        "category": mode_result.get("category"),
        "confidence": float(mode_result.get("confidence", 0)),
        "is_boundary_case": bool(stability["is_boundary_case"]),
        "stability_warning": stability["stability_warning"],
        "stability_score": float(stability["stability_score"]),
        "description": mode_result.get("description", "")
    }


# ============================================================================
# Request/Response Models
# ============================================================================

class EnhancedAnalyzeRequest(BaseModel):
    """Request for enhanced analysis with 12-mode classification."""
    text: str
    model_id: str = "all-MiniLM-L6-v2"
    layer: int = -1
    include_uncertainty: bool = True
    include_legacy_mode: bool = True
    include_force_field: bool = False  # Attractor/detractor analysis


class EnhancedAnalyzeResponse(BaseModel):
    """
    Response with enhanced mode classification including soft labels and stability.

    The mode field now ALWAYS includes:
    - primary_mode: Most likely classification
    - primary_probability: Probability for primary mode (0-1)
    - secondary_mode: Second most likely classification
    - secondary_probability: Probability for secondary mode (0-1)
    - all_probabilities: Full distribution across all modes
    - confidence: Classification confidence (0-1)
    - is_boundary_case: True if near a mode boundary
    - stability_warning: Warning message if classification may be unstable
    - stability_score: Quick stability assessment (0-1)

    Optional force_field includes attractor/detractor analysis:
    - attractor_strength: Pull toward positive states (-2 to +2)
    - detractor_strength: Push from negative states (-2 to +2)
    - primary_attractor: Main target being drawn toward
    - primary_detractor: Main source being fled from
    - force_quadrant: ACTIVE_TRANSFORMATION, PURE_ASPIRATION, PURE_ESCAPE, or STASIS
    """
    text: str
    vector: Dict[str, float]
    mode: Dict[str, Any]  # Enhanced mode with probabilities and stability
    legacy_mode: Optional[str] = None
    uncertainty: Optional[Dict[str, Any]] = None
    force_field: Optional[Dict[str, Any]] = None  # Attractor/detractor forces
    embedding_dim: int
    layer: int
    model_id: str


class DetailedAnalyzeRequest(BaseModel):
    """Request for detailed analysis with full stability assessment."""
    text: str
    model_id: str = "all-MiniLM-L6-v2"
    layer: int = -1
    include_legacy_mode: bool = True


class DetailedAnalyzeResponse(BaseModel):
    """
    Response with comprehensive mode analysis and stability assessment.

    Includes everything from EnhancedAnalyzeResponse plus:
    - Full probability distribution for all 12 modes
    - Boundary distances to all mode centroids
    - Detailed stability metrics
    - Interpretation caveats and recommendations
    """
    text: str
    vector: Dict[str, float]
    mode: Dict[str, Any]
    legacy_mode: Optional[str] = None
    stability: Dict[str, Any]  # Detailed stability assessment
    boundary_distances: Dict[str, float]  # Distance to each mode centroid
    interpretation_caveats: List[str]  # Recommended caveats for interpretation
    embedding_dim: int
    layer: int
    model_id: str


class BatchEnhancedRequest(BaseModel):
    """Request for batch enhanced analysis."""
    texts: List[str]
    model_id: str = "all-MiniLM-L6-v2"
    layer: int = -1


class RobustnessTestRequest(BaseModel):
    """Request for robustness testing."""
    text: str
    perturbation_types: List[str] = ["negation", "hedging", "tense"]
    model_id: str = "all-MiniLM-L6-v2"


class TrajectoryAddRequest(BaseModel):
    """Request to add a trajectory point."""
    entity: str
    timestamp: str
    text: str
    model_id: str = "all-MiniLM-L6-v2"


class TrajectoryAnalyzeRequest(BaseModel):
    """Request to analyze a trajectory."""
    entity: str


class ResearchAgentRequest(BaseModel):
    """Request to run research agent."""
    max_experiments: int = Field(default=5, ge=1, le=20)
    hypotheses: Optional[List[str]] = None


# ============================================================================
# Enhanced Analysis Endpoints
# ============================================================================

@router.post("/analyze", response_model=EnhancedAnalyzeResponse)
async def enhanced_analyze(request: EnhancedAnalyzeRequest):
    """
    Enhanced analysis with 12-mode probabilistic classification.

    Returns soft labels and stability indicators for EVERY response:
    - Primary and secondary modes with probabilities
    - Full probability distribution across all 12 modes (all_probabilities)
    - Mode category (POSITIVE, SHADOW, EXIT, AMBIVALENT)
    - Confidence score (0-1)
    - is_boundary_case: Flag if classification is near a mode boundary
    - stability_warning: Human-readable warning if classification may be unstable
    - stability_score: Quick stability assessment (0-1)
    - Optional uncertainty quantification

    This endpoint uses the currently selected projection mode. Use
    GET /v2/projections to see available modes and POST /v2/projections/select
    to switch modes. Supported modes:
    - current_projection: Default MiniLM-based (fastest)
    - mpnet_projection: Best accuracy with MPNet
    - multi_model_ensemble: Best robustness (averages 3 models)
    - ensemble_projection: Includes uncertainty quantification

    Note: 87.5% of ambiguous statements have confidence < 0.5, and 43.2% of
    perturbations can cause mode flips. Always check is_boundary_case and
    stability_warning before treating classifications as definitive.
    """
    # Import here to avoid circular imports
    from main import (
        model_manager, embedding_extractor, current_projection,
        ModelType
    )
    from models.projection import ProjectionWithUncertainty, Vector3

    # Check if we should use projection mode manager
    manager = None
    try:
        manager = get_projection_manager()
        current_mode = manager.get_current_mode()
    except Exception as e:
        logger.warning(f"Projection mode manager not available: {e}")
        current_mode = None

    # Determine projection approach
    use_mode_manager = (
        manager is not None and
        current_mode is not None and
        current_mode != "current_projection"  # For current_projection, use existing flow
    )

    # Validate we have a projection
    if not use_mode_manager and current_projection is None:
        raise HTTPException(status_code=400, detail="No projection trained")

    try:
        # For multi-model modes, we need to load all required models
        if use_mode_manager:
            mode_info = manager.get_mode_info(current_mode)
            required_models = mode_info.models if mode_info else [request.model_id]

            # Load all required models and get embeddings
            embeddings = {}
            embedding_result = None

            for model_id in required_models:
                if not model_manager.is_loaded(model_id):
                    model_manager.load_model(model_id, ModelType.SENTENCE_TRANSFORMER)

                result = embedding_extractor.extract(
                    request.text,
                    model_id,
                    layer=request.layer
                )
                embeddings[model_id] = result.embedding
                embedding_result = result

            # Project using the mode manager
            coords = manager.project(embeddings, model_manager, embedding_extractor)

            # Check if this is an uncertainty result
            is_uncertainty_result = isinstance(coords, ProjectionWithUncertainty)

            if is_uncertainty_result:
                coords_array = np.array([
                    coords.coords.agency,
                    coords.coords.fairness,
                    coords.coords.belonging
                ])
                vector_dict = coords.coords.to_dict()
            else:
                coords_array = np.array([coords.agency, coords.fairness, coords.belonging])
                vector_dict = coords.to_dict()
        else:
            # Original flow - single model
            if not model_manager.is_loaded(request.model_id):
                model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

            # Get embedding
            embedding_result = embedding_extractor.extract(
                request.text,
                request.model_id,
                layer=request.layer
            )

            # Project to 3D using current_projection
            coords = current_projection.project(embedding_result.embedding)
            coords_array = np.array([coords.agency, coords.fairness, coords.belonging])
            vector_dict = coords.to_dict()
            is_uncertainty_result = False

        # Enhanced classification
        classifier = get_mode_classifier()
        mode_result = classifier.classify(coords_array)

        # Build enhanced mode response with soft labels and stability
        enhanced_mode = build_enhanced_mode_response(mode_result)

        # Build response
        response = {
            "text": request.text,
            "vector": vector_dict,
            "mode": enhanced_mode,
            "embedding_dim": len(embedding_result.embedding) if embedding_result else None,
            "layer": embedding_result.layer if embedding_result else request.layer,
            "model_id": request.model_id
        }

        # Add projection mode info if using mode manager
        if use_mode_manager and mode_info:
            response["projection_mode"] = current_mode
            response["projection_mode_display"] = mode_info.display_name

        if request.include_legacy_mode:
            response["legacy_mode"] = legacy_mode(mode_result["primary_mode"])

        # Handle uncertainty
        if request.include_uncertainty:
            # If we got uncertainty from the projection mode manager
            if is_uncertainty_result:
                response["uncertainty"] = {
                    "std_per_axis": {
                        "agency": round(coords.std_per_axis.agency, 4),
                        "perceived_justice": round(coords.std_per_axis.fairness, 4),
                        "belonging": round(coords.std_per_axis.belonging, 4)
                    },
                    "confidence_intervals": {
                        "agency": [
                            round(coords.confidence_intervals["agency"][0], 4),
                            round(coords.confidence_intervals["agency"][1], 4)
                        ],
                        "perceived_justice": [
                            round(coords.confidence_intervals["fairness"][0], 4),
                            round(coords.confidence_intervals["fairness"][1], 4)
                        ],
                        "belonging": [
                            round(coords.confidence_intervals["belonging"][0], 4),
                            round(coords.confidence_intervals["belonging"][1], 4)
                        ]
                    },
                    "overall_confidence": round(coords.overall_confidence, 4),
                    "method": coords.method
                }
            elif use_mode_manager and current_mode == "multi_model_ensemble":
                # Multi-model ensemble provides robustness but not uncertainty bounds
                response["uncertainty"] = {
                    "note": "Multi-model ensemble provides robustness through averaging. "
                            "For per-axis uncertainty bounds, select ensemble_projection mode.",
                    "models_aggregated": len(mode_info.models) if mode_info else 0,
                    "confidence": mode_result["confidence"],
                    "stability_score": enhanced_mode["stability_score"]
                }
            else:
                # Try to use ensemble projection for uncertainty quantification
                from main import ensemble_projection

                if ensemble_projection is not None and ensemble_projection.is_trained:
                    try:
                        # Get uncertainty from ensemble projection
                        ensemble_result = ensemble_projection.project_with_uncertainty(embedding_result.embedding)

                        # Build uncertainty response
                        response["uncertainty"] = {
                            "std_per_axis": {
                                "agency": round(ensemble_result.std_per_axis.agency, 4),
                                "perceived_justice": round(ensemble_result.std_per_axis.fairness, 4),
                                "belonging": round(ensemble_result.std_per_axis.belonging, 4)
                            },
                            "confidence_intervals": {
                                "agency": [round(ensemble_result.confidence_intervals["agency"][0], 4),
                                           round(ensemble_result.confidence_intervals["agency"][1], 4)],
                                "perceived_justice": [
                                    round(ensemble_result.confidence_intervals.get("fairness", ensemble_result.confidence_intervals.get("perceived_justice", [0, 0]))[0], 4),
                                    round(ensemble_result.confidence_intervals.get("fairness", ensemble_result.confidence_intervals.get("perceived_justice", [0, 0]))[1], 4)
                                ],
                                "belonging": [round(ensemble_result.confidence_intervals["belonging"][0], 4),
                                              round(ensemble_result.confidence_intervals["belonging"][1], 4)]
                            },
                            "overall_confidence": round(ensemble_result.overall_confidence, 4),
                            "method": "ensemble_ridge"
                        }
                    except Exception as e:
                        logger.warning(f"Ensemble uncertainty failed: {e}")
                        response["uncertainty"] = {
                            "note": "Ensemble uncertainty failed, using fallback",
                            "confidence": mode_result["confidence"],
                            "stability_score": enhanced_mode["stability_score"]
                        }
                else:
                    response["uncertainty"] = {
                        "note": "Ensemble projection not available. Train with /training/train-ensemble or select ensemble_projection mode.",
                        "confidence": mode_result["confidence"],
                        "stability_score": enhanced_mode["stability_score"]
                    }

        # Handle force field analysis (attractors/detractors)
        if request.include_force_field:
            try:
                from analysis.force_field import analyze_force_field
                force_result = analyze_force_field(request.text, embedding_result.embedding)
                response["force_field"] = force_result
            except Exception as e:
                logger.warning(f"Force field analysis failed: {e}")
                response["force_field"] = {
                    "note": f"Force field analysis not available: {str(e)}",
                    "attractor_strength": 0,
                    "detractor_strength": 0,
                }

        return response

    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/batch")
async def enhanced_batch_analyze(request: BatchEnhancedRequest):
    """
    Batch enhanced analysis with stability indicators.

    Each result includes simplified stability indicators:
    - confidence: Classification confidence (0-1)
    - stability_score: Quick stability assessment (0-1)
    - is_boundary_case: True if classification is near mode boundary

    Summary includes aggregate stability metrics:
    - average_confidence: Mean confidence across all results
    - average_stability: Mean stability score across all results
    - boundary_case_count: Number of results near mode boundaries
    - low_confidence_count: Number of results with confidence < 0.5
    """
    from main import (
        model_manager, embedding_extractor, current_projection,
        ModelType
    )

    if current_projection is None:
        raise HTTPException(status_code=400, detail="No projection trained")

    if not model_manager.is_loaded(request.model_id):
        model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

    classifier = get_mode_classifier()
    results = []

    for text in request.texts:
        try:
            result = embedding_extractor.extract(text, request.model_id, layer=request.layer)
            coords = current_projection.project(result.embedding)
            coords_array = np.array([coords.agency, coords.fairness, coords.belonging])
            mode_result = classifier.classify(coords_array)

            # Calculate stability metrics
            stability = calculate_stability_score(mode_result)

            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "vector": coords.to_dict(),
                "mode": mode_result["primary_mode"],
                "secondary_mode": mode_result["secondary_mode"],
                "primary_probability": mode_result["primary_probability"],
                "secondary_probability": mode_result["secondary_probability"],
                "category": mode_result["category"],
                "confidence": mode_result["confidence"],
                "stability_score": stability["stability_score"],
                "is_boundary_case": stability["is_boundary_case"],
                "stability_warning": stability["stability_warning"]
            })
        except Exception as e:
            results.append({
                "text": text[:100],
                "error": str(e)
            })

    # Compute mode distribution
    mode_counts = {}
    category_counts = {}
    for r in results:
        if "mode" in r:
            mode_counts[r["mode"]] = mode_counts.get(r["mode"], 0) + 1
            category_counts[r["category"]] = category_counts.get(r["category"], 0) + 1

    # Compute stability summary
    valid_results = [r for r in results if "confidence" in r]
    boundary_cases = [r for r in valid_results if r.get("is_boundary_case", False)]
    low_confidence = [r for r in valid_results if r.get("confidence", 1) < 0.5]

    return {
        "results": results,
        "summary": {
            "total": len(results),
            "mode_distribution": mode_counts,
            "category_distribution": category_counts,
            "average_confidence": round(
                float(np.mean([r["confidence"] for r in valid_results])) if valid_results else 0, 4
            ),
            "average_stability": round(
                float(np.mean([r["stability_score"] for r in valid_results])) if valid_results else 0, 4
            ),
            "boundary_case_count": len(boundary_cases),
            "low_confidence_count": len(low_confidence),
            "stability_warning": (
                f"{len(boundary_cases)} boundary cases and {len(low_confidence)} low-confidence "
                "classifications detected. Interpret with caution."
                if boundary_cases or low_confidence else None
            )
        }
    }


@router.post("/analyze/detailed", response_model=DetailedAnalyzeResponse)
async def detailed_analyze(request: DetailedAnalyzeRequest):
    """
    Detailed analysis with comprehensive stability assessment.

    Returns everything from /analyze plus:
    - Full probability distribution for all 12 modes
    - Boundary distances to all mode centroids
    - Detailed stability metrics with breakdown
    - Interpretation caveats based on classification characteristics

    Use this endpoint when you need maximum scientific rigor and
    want to understand the full uncertainty profile of a classification.
    """
    # Import here to avoid circular imports
    from main import (
        model_manager, embedding_extractor, current_projection,
        ModelType
    )

    if current_projection is None:
        raise HTTPException(status_code=400, detail="No projection trained")

    if not model_manager.is_loaded(request.model_id):
        model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

    try:
        # Get embedding
        result = embedding_extractor.extract(
            request.text,
            request.model_id,
            layer=request.layer
        )

        # Project to 3D
        coords = current_projection.project(result.embedding)
        coords_array = np.array([coords.agency, coords.fairness, coords.belonging])

        # Enhanced classification
        classifier = get_mode_classifier()
        mode_result = classifier.classify(coords_array)

        # Build enhanced mode response with soft labels and stability
        enhanced_mode = build_enhanced_mode_response(mode_result)

        # Calculate detailed stability metrics
        stability = calculate_stability_score(mode_result)

        # Extract boundary distances (clean up the 'to_' prefix)
        raw_boundary_distances = mode_result.get("boundary_distances", {})
        boundary_distances = {
            key.replace("to_", ""): value
            for key, value in raw_boundary_distances.items()
        }

        # Generate interpretation caveats based on classification characteristics
        interpretation_caveats = []

        confidence = mode_result.get("confidence", 0)
        if confidence < 0.3:
            interpretation_caveats.append(
                "Very low confidence (<0.3): Classification is highly uncertain. "
                "Consider this a weak signal at best."
            )
        elif confidence < 0.5:
            interpretation_caveats.append(
                "Low confidence (<0.5): Classification is ambiguous. "
                "87.5% of ambiguous statements fall in this range."
            )
        elif confidence < 0.7:
            interpretation_caveats.append(
                "Moderate confidence (0.5-0.7): Classification is plausible but "
                "should be interpreted with care."
            )

        if stability["is_boundary_case"]:
            interpretation_caveats.append(
                "Boundary case detected: This classification is near a mode boundary. "
                "43.2% of perturbations can cause mode flips in such cases."
            )

        probability_gap = stability["probability_gap"]
        if probability_gap < 0.1:
            secondary_mode = mode_result.get("secondary_mode", "unknown")
            interpretation_caveats.append(
                f"Very small probability gap ({probability_gap:.2f}): "
                f"The secondary mode ({secondary_mode}) is almost equally likely. "
                "Consider treating this as a mixed/ambivalent classification."
            )
        elif probability_gap < 0.2:
            interpretation_caveats.append(
                f"Small probability gap ({probability_gap:.2f}): "
                "Multiple modes have significant probability mass."
            )

        if stability["stability_score"] < 0.3:
            interpretation_caveats.append(
                "Low stability score (<0.3): This classification is likely to change "
                "under text perturbations (synonym substitution, hedging, etc.)."
            )

        # Check for specific mode-related caveats
        primary_mode = mode_result.get("primary_mode", "")
        category = mode_result.get("category", "")

        if category == "AMBIVALENT":
            interpretation_caveats.append(
                f"Ambivalent mode ({primary_mode}): This classification indicates "
                "mixed or transitional narrative signals. The text may contain "
                "conflicting value orientations."
            )

        if primary_mode in ["CONFLICTED", "TRANSITIONAL", "NEUTRAL"]:
            interpretation_caveats.append(
                f"Mode '{primary_mode}' inherently represents uncertainty or "
                "transition in the narrative space."
            )

        # If no caveats, add a positive note
        if not interpretation_caveats:
            interpretation_caveats.append(
                "High confidence, stable classification with clear mode separation."
            )

        # Build response
        response = {
            "text": request.text,
            "vector": coords.to_dict(),
            "mode": enhanced_mode,
            "stability": {
                "stability_score": stability["stability_score"],
                "min_boundary_distance": stability["min_boundary_distance"],
                "probability_gap": stability["probability_gap"],
                "is_boundary_case": stability["is_boundary_case"],
                "stability_warning": stability["stability_warning"],
                "confidence_level": (
                    "high" if confidence >= 0.7 else
                    "moderate" if confidence >= 0.5 else
                    "low" if confidence >= 0.3 else
                    "very_low"
                ),
                "stability_level": (
                    "stable" if stability["stability_score"] >= 0.7 else
                    "moderate" if stability["stability_score"] >= 0.4 else
                    "unstable"
                )
            },
            "boundary_distances": boundary_distances,
            "interpretation_caveats": interpretation_caveats,
            "embedding_dim": len(result.embedding),
            "layer": result.layer,
            "model_id": request.model_id
        }

        if request.include_legacy_mode:
            response["legacy_mode"] = legacy_mode(mode_result["primary_mode"])

        return response

    except Exception as e:
        logger.error(f"Detailed analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Axis Information Endpoints
# ============================================================================

@router.get("/axes")
async def get_axes_info():
    """
    Get information about all three axes in the cultural manifold.

    Returns axis configuration including:
    - canonical_name: The official name for API responses
    - display_name: Human-readable name
    - aliases: All accepted names for this axis (for backward compatibility)
    - description: What the axis measures
    - deprecation_notice: If any names are deprecated
    """
    from models.axis_config import AXIS_CONFIG, get_axis_deprecation_notice

    axes_info = []
    for canonical, config in AXIS_CONFIG.items():
        axes_info.append({
            "canonical_name": config["canonical_name"],
            "display_name": config["display_name"],
            "aliases": config["aliases"],
            "description": config["description"],
            "range": config.get("range", [-2.0, 2.0]),
            "positive_pole": config.get("positive_pole", ""),
            "negative_pole": config.get("negative_pole", ""),
            "deprecation_notice": config.get("deprecation_notice")
        })

    return {
        "axes": axes_info,
        "note": "The 'fairness' axis was renamed to 'perceived_justice' in January 2026. "
                "Old code using 'fairness' will still work but is deprecated.",
        "canonical_axis_names": ["agency", "perceived_justice", "belonging"]
    }


@router.get("/axes/{axis_name}")
async def get_axis_info(axis_name: str):
    """
    Get information about a specific axis.

    Accepts any alias (e.g., 'fairness', 'perceived_justice', 'justice').
    """
    from models.axis_config import (
        translate_axis_name, get_axis_config, get_axis_deprecation_notice
    )

    config = get_axis_config(axis_name)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown axis: '{axis_name}'. Valid axes: agency, perceived_justice, belonging"
        )

    deprecation_notice = get_axis_deprecation_notice(axis_name)

    return {
        "canonical_name": config["canonical_name"],
        "display_name": config["display_name"],
        "aliases": config["aliases"],
        "description": config["description"],
        "range": config.get("range", [-2.0, 2.0]),
        "positive_pole": config.get("positive_pole", ""),
        "negative_pole": config.get("negative_pole", ""),
        "deprecation_notice": deprecation_notice,
        "axis_deprecation_note": config.get("deprecation_notice")
    }


# ============================================================================
# Mode Information Endpoints
# ============================================================================

@router.get("/modes")
async def get_all_modes():
    """Get information about all 12 narrative modes."""
    classifier = get_mode_classifier()
    return {
        "modes": classifier.get_all_modes(),
        "categories": ["POSITIVE", "SHADOW", "EXIT", "AMBIVALENT", "NOISE"]
    }


@router.get("/modes/{mode_name}")
async def get_mode_info(mode_name: str):
    """Get detailed information about a specific mode."""
    classifier = get_mode_classifier()
    modes = {m["name"]: m for m in classifier.get_all_modes()}

    if mode_name not in modes:
        raise HTTPException(status_code=404, detail=f"Mode '{mode_name}' not found")

    return modes[mode_name]


# ============================================================================
# Research Agent Endpoints
# ============================================================================

@router.post("/research/run")
async def run_research_session(request: ResearchAgentRequest):
    """
    Run an autonomous research session.

    The research agent will:
    1. Test hypotheses about the manifold
    2. Return confirmed findings
    3. Generate a research report
    """
    try:
        from research_agent import ObservatoryResearchAgent

        agent = ObservatoryResearchAgent()

        # Run research session
        findings = await agent.run_research_session(max_experiments=request.max_experiments)

        return {
            "session_id": agent.session_id,
            "experiments_run": agent.experiment_count,
            "findings": [
                {
                    "id": f.id,
                    "title": f.title,
                    "description": f.description,
                    "significance": f.significance
                }
                for f in findings
            ],
            "hypothesis_status": [
                {
                    "id": h.id,
                    "statement": h.statement,
                    "status": h.status
                }
                for h in agent.agenda
            ],
            "report": agent.generate_report()
        }

    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Research agent not available. Install required dependencies."
        )
    except Exception as e:
        logger.error(f"Research session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/research/hypotheses")
async def get_research_hypotheses():
    """Get the current research hypothesis queue."""
    try:
        from research_agent import ObservatoryResearchAgent
        agent = ObservatoryResearchAgent()

        return {
            "hypotheses": [
                {
                    "id": h.id,
                    "statement": h.statement,
                    "category": h.category,
                    "priority": h.priority
                }
                for h in agent.agenda
            ]
        }
    except ImportError:
        raise HTTPException(status_code=501, detail="Research agent not available")


# ============================================================================
# Robustness Testing Endpoints
# ============================================================================

@router.post("/robustness/test")
async def test_robustness(request: RobustnessTestRequest):
    """
    Test projection robustness under perturbations.

    Returns sensitivity analysis for different perturbation types.
    """
    try:
        from analysis.robustness import RobustnessTester, get_robustness_tester
        from main import (
            model_manager, embedding_extractor, current_projection,
            ModelType
        )

        if current_projection is None:
            raise HTTPException(status_code=400, detail="No projection trained")

        # Define projection function
        async def project_fn(text: str) -> dict:
            if not model_manager.is_loaded(request.model_id):
                model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

            result = embedding_extractor.extract(text, request.model_id)
            coords = current_projection.project(result.embedding)
            classifier = get_mode_classifier()
            mode = classifier.classify(np.array([coords.agency, coords.fairness, coords.belonging]))

            return {
                "vector": coords.to_dict(),
                "mode": mode["primary_mode"]
            }

        tester = get_robustness_tester()

        # Convert string perturbation types to enum values
        from analysis.robustness import PerturbationType
        perturbation_types = None
        if request.perturbation_types:
            perturbation_types = []
            for pt in request.perturbation_types:
                try:
                    perturbation_types.append(PerturbationType(pt.lower()))
                except ValueError:
                    logger.warning(f"Unknown perturbation type: {pt}")

        report = await tester.test_robustness(
            request.text,
            project_fn,
            types=perturbation_types
        )

        return report.to_dict()

    except ImportError:
        # Robustness module not yet available
        raise HTTPException(
            status_code=501,
            detail="Robustness testing not yet implemented"
        )
    except Exception as e:
        logger.error(f"Robustness test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Trajectory Tracking Endpoints
# ============================================================================

@router.post("/trajectory/add")
async def add_trajectory_point(request: TrajectoryAddRequest):
    """Add a point to an entity's trajectory."""
    try:
        from analysis.trajectory import get_trajectory_analyzer
        from main import (
            model_manager, embedding_extractor, current_projection,
            ModelType
        )

        if current_projection is None:
            raise HTTPException(status_code=400, detail="No projection trained")

        if not model_manager.is_loaded(request.model_id):
            model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

        # Project text
        result = embedding_extractor.extract(request.text, request.model_id)
        coords = current_projection.project(result.embedding)
        classifier = get_mode_classifier()
        mode = classifier.classify(np.array([coords.agency, coords.fairness, coords.belonging]))

        # Add to trajectory
        analyzer = get_trajectory_analyzer()
        analyzer.add_point(
            entity=request.entity,
            timestamp=request.timestamp,
            text=request.text,
            coords=[coords.agency, coords.fairness, coords.belonging],
            mode=mode["primary_mode"]
        )

        return {
            "success": True,
            "entity": request.entity,
            "timestamp": request.timestamp,
            "coords": coords.to_dict(),
            "mode": mode["primary_mode"]
        }

    except ImportError:
        raise HTTPException(status_code=501, detail="Trajectory tracking not yet implemented")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trajectory/{entity}")
async def get_trajectory(entity: str):
    """Get an entity's full trajectory."""
    try:
        from analysis.trajectory import get_trajectory_analyzer

        analyzer = get_trajectory_analyzer()
        trajectory = analyzer.get_trajectory(entity)

        if not trajectory:
            raise HTTPException(status_code=404, detail=f"No trajectory for '{entity}'")

        return trajectory

    except ImportError:
        raise HTTPException(status_code=501, detail="Trajectory tracking not yet implemented")


@router.post("/trajectory/analyze")
async def analyze_trajectory(request: TrajectoryAnalyzeRequest):
    """Analyze an entity's trajectory dynamics."""
    try:
        from analysis.trajectory import get_trajectory_analyzer

        analyzer = get_trajectory_analyzer()
        analysis = analyzer.analyze_trajectory(request.entity)

        if not analysis:
            raise HTTPException(status_code=404, detail=f"No trajectory for '{request.entity}'")

        return analysis

    except ImportError:
        raise HTTPException(status_code=501, detail="Trajectory tracking not yet implemented")


# ============================================================================
# Health & Status
# ============================================================================

@router.get("/status")
async def v2_status():
    """Get status of v2 API extensions."""
    status = {
        "version": "2.2",
        "features": {
            "enhanced_mode_classification": True,
            "soft_labels": True,  # Full probability distributions always included
            "stability_indicators": True,  # Stability scores and boundary case detection
            "detailed_analysis": True,  # New /analyze/detailed endpoint
            "projection_mode_switching": True,  # New: Switch between projection modes
            "research_agent": False,
            "robustness_testing": False,
            "trajectory_tracking": False,
            "uncertainty_quantification": False
        },
        "stability_thresholds": {
            "boundary_distance_threshold": 0.3,
            "probability_gap_threshold": 0.15,
            "low_confidence_threshold": 0.5
        }
    }

    # Check projection mode manager
    try:
        manager = get_projection_manager()
        current_mode = manager.get_current_mode()
        modes = manager.list_modes()

        status["projection_modes"] = {
            "current_mode": current_mode,
            "available_modes": [m.name for m in modes if m.is_available],
            "endpoints": {
                "list": "GET /v2/projections",
                "select": "POST /v2/projections/select",
                "current": "GET /v2/projections/current"
            }
        }
    except Exception as e:
        logger.warning(f"Projection mode manager status check failed: {e}")
        status["projection_modes"] = {
            "error": str(e),
            "current_mode": None
        }

    # Check which modules are available
    try:
        from research_agent import ObservatoryResearchAgent
        status["features"]["research_agent"] = True
    except ImportError:
        pass

    try:
        from analysis.robustness import RobustnessTester
        status["features"]["robustness_testing"] = True
    except ImportError:
        pass

    try:
        from analysis.trajectory import TrajectoryAnalyzer
        status["features"]["trajectory_tracking"] = True
    except ImportError:
        pass

    try:
        from models.ensemble_projection import EnsembleProjection
        from main import ensemble_projection
        status["features"]["uncertainty_quantification"] = True
        # Check if ensemble is actually trained
        if ensemble_projection is not None and ensemble_projection.is_trained:
            status["ensemble_trained"] = True
            status["ensemble_size"] = len(ensemble_projection.ensemble)
        else:
            status["ensemble_trained"] = False
            status["ensemble_training_endpoint"] = "/training/train-ensemble"
    except ImportError:
        pass

    return status


# ============================================================================
# Projection Mode Management Endpoints
# ============================================================================

# Global projection mode manager instance
_projection_mode_manager = None


def get_projection_manager():
    """Get or initialize the projection mode manager."""
    global _projection_mode_manager
    if _projection_mode_manager is None:
        from models.projection_mode_manager import ProjectionModeManager
        from pathlib import Path
        projections_dir = Path("./data/projections")
        _projection_mode_manager = ProjectionModeManager(projections_dir)
        _projection_mode_manager.initialize()
    return _projection_mode_manager


class SelectProjectionRequest(BaseModel):
    """Request to select a projection mode."""
    mode: str = Field(
        ...,
        description="Projection mode to select",
        examples=["current_projection", "mpnet_projection", "multi_model_ensemble", "ensemble_projection"]
    )


class SelectProjectionResponse(BaseModel):
    """Response after selecting a projection mode."""
    success: bool
    mode: str
    display_name: str
    description: str
    required_models: List[str]
    message: str


@router.get("/projections")
async def list_projection_modes():
    """
    List all available projection modes.

    Returns information about each projection configuration:
    - current_projection: Default MiniLM-based (CV: 0.383)
    - mpnet_projection: Best accuracy with all-mpnet-base-v2 (CV: 0.612)
    - multi_model_ensemble: Best robustness, averages 3 models
    - ensemble_projection: 25-model bootstrap ensemble for uncertainty

    Each mode has different characteristics:
    - Use current_projection for fast, simple analysis
    - Use mpnet_projection for best accuracy
    - Use multi_model_ensemble for most robust results
    - Use ensemble_projection when you need uncertainty quantification
    """
    manager = get_projection_manager()
    modes = manager.list_modes()

    return {
        "modes": [m.to_dict() for m in modes],
        "current_mode": manager.get_current_mode(),
        "recommendations": {
            "best_accuracy": "mpnet_projection",
            "best_robustness": "multi_model_ensemble",
            "fastest": "current_projection",
            "uncertainty_quantification": "ensemble_projection"
        }
    }


@router.get("/projections/current")
async def get_current_projection_mode():
    """
    Get the currently active projection mode.

    Returns the name and details of the projection mode currently in use.
    """
    manager = get_projection_manager()
    current = manager.get_current_mode()

    if current is None:
        return {
            "current_mode": None,
            "message": "No projection mode is currently selected"
        }

    mode_info = manager.get_mode_info(current)
    if mode_info:
        return {
            "current_mode": current,
            "mode_info": mode_info.to_dict(),
            "required_models": manager.get_required_models()
        }

    return {
        "current_mode": current,
        "required_models": manager.get_required_models()
    }


@router.post("/projections/select", response_model=SelectProjectionResponse)
async def select_projection_mode(request: SelectProjectionRequest):
    """
    Select which projection mode to use for analysis.

    Available modes:
    - current_projection: Default MiniLM-based projection
    - mpnet_projection: Higher accuracy MPNet-based projection
    - multi_model_ensemble: Averages 3 embedding models for robustness
    - ensemble_projection: 25-model bootstrap ensemble with uncertainty

    After selecting a mode, subsequent calls to /v2/analyze will use
    the selected projection. For multi_model_ensemble, all 3 embedding
    models will be loaded and used.

    Note: Selecting multi_model_ensemble will require loading 3 models
    which may take a moment on first use.
    """
    manager = get_projection_manager()

    try:
        mode_info = manager.select_mode(request.mode)

        # Build message based on mode
        if request.mode == "multi_model_ensemble":
            message = (
                f"Selected {mode_info.display_name}. "
                f"Analysis will now average projections from {len(mode_info.models)} models: "
                f"{', '.join(mode_info.models)}. "
                "This provides the most robust results but requires all models to be loaded."
            )
        elif request.mode == "ensemble_projection":
            message = (
                f"Selected {mode_info.display_name}. "
                "Analysis will now include uncertainty quantification with 95% confidence intervals. "
                "Response will include std_per_axis and confidence_intervals fields."
            )
        else:
            message = f"Selected {mode_info.display_name}. Analysis will use {mode_info.models[0] if mode_info.models else 'the configured model'}."

        return SelectProjectionResponse(
            success=True,
            mode=request.mode,
            display_name=mode_info.display_name,
            description=mode_info.description,
            required_models=mode_info.models,
            message=message
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to select projection mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projections/{mode_name}")
async def get_projection_mode_details(mode_name: str):
    """
    Get detailed information about a specific projection mode.

    Returns:
    - Configuration details
    - Performance metrics (CV score, R2)
    - Required embedding models
    - Whether the mode is currently available
    """
    manager = get_projection_manager()
    mode_info = manager.get_mode_info(mode_name)

    if mode_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown projection mode: {mode_name}. Available: current_projection, mpnet_projection, multi_model_ensemble, ensemble_projection"
        )

    return mode_info.to_dict()


class AnalyzeWithModeRequest(BaseModel):
    """Request for analysis with explicit projection mode."""
    text: str
    mode: Optional[str] = None  # If None, uses current mode
    model_id: Optional[str] = None  # Override embedding model (for single-model modes)
    layer: int = -1
    include_uncertainty: bool = True
    include_legacy_mode: bool = True


@router.post("/analyze/with-mode")
async def analyze_with_projection_mode(request: AnalyzeWithModeRequest):
    """
    Analyze text with a specific projection mode.

    This endpoint allows specifying the projection mode explicitly without
    changing the global selection. Useful for comparing results across modes.

    For multi_model_ensemble mode, embeddings from all 3 models are computed
    and averaged. For ensemble_projection mode, uncertainty estimates are
    always included.

    If mode is not specified, uses the currently selected mode.
    """
    from main import (
        model_manager, embedding_extractor, current_projection,
        ModelType
    )

    manager = get_projection_manager()

    # Determine which mode to use
    mode_name = request.mode or manager.get_current_mode()
    if mode_name is None:
        raise HTTPException(
            status_code=400,
            detail="No projection mode specified and no default mode selected. "
                   "Use POST /v2/projections/select to select a mode first."
        )

    mode_info = manager.get_mode_info(mode_name)
    if mode_info is None or not mode_info.is_available:
        raise HTTPException(
            status_code=400,
            detail=f"Projection mode '{mode_name}' is not available"
        )

    try:
        # Get required models for this mode
        required_models = mode_info.models

        # Load all required models and get embeddings
        embeddings = {}
        embedding_result = None  # Keep track of one for metadata

        for model_id in required_models:
            if not model_manager.is_loaded(model_id):
                model_manager.load_model(model_id, ModelType.SENTENCE_TRANSFORMER)

            result = embedding_extractor.extract(
                request.text,
                model_id,
                layer=request.layer
            )
            embeddings[model_id] = result.embedding
            embedding_result = result

        # Project using the mode manager
        coords = manager.project(embeddings, model_manager, embedding_extractor)

        # Check if this is an uncertainty result
        from models.projection import ProjectionWithUncertainty, Vector3
        is_uncertainty_result = isinstance(coords, ProjectionWithUncertainty)

        if is_uncertainty_result:
            coords_array = np.array([
                coords.coords.agency,
                coords.coords.fairness,
                coords.coords.belonging
            ])
            vector_dict = coords.coords.to_dict()
        else:
            coords_array = np.array([coords.agency, coords.fairness, coords.belonging])
            vector_dict = coords.to_dict()

        # Enhanced classification
        classifier = get_mode_classifier()
        mode_result = classifier.classify(coords_array)
        enhanced_mode = build_enhanced_mode_response(mode_result)

        # Build response
        response = {
            "text": request.text,
            "vector": vector_dict,
            "mode": enhanced_mode,
            "projection_mode": mode_name,
            "projection_mode_display": mode_info.display_name,
            "embedding_dim": len(embedding_result.embedding) if embedding_result else None,
            "layer": request.layer,
            "models_used": required_models
        }

        if request.include_legacy_mode:
            response["legacy_mode"] = legacy_mode(mode_result["primary_mode"])

        # Include uncertainty for ensemble modes
        if is_uncertainty_result:
            response["uncertainty"] = {
                "std_per_axis": {
                    "agency": round(coords.std_per_axis.agency, 4),
                    "perceived_justice": round(coords.std_per_axis.fairness, 4),
                    "belonging": round(coords.std_per_axis.belonging, 4)
                },
                "confidence_intervals": {
                    "agency": [
                        round(coords.confidence_intervals["agency"][0], 4),
                        round(coords.confidence_intervals["agency"][1], 4)
                    ],
                    "perceived_justice": [
                        round(coords.confidence_intervals["fairness"][0], 4),
                        round(coords.confidence_intervals["fairness"][1], 4)
                    ],
                    "belonging": [
                        round(coords.confidence_intervals["belonging"][0], 4),
                        round(coords.confidence_intervals["belonging"][1], 4)
                    ]
                },
                "overall_confidence": round(coords.overall_confidence, 4),
                "method": coords.method
            }
        elif request.include_uncertainty and mode_name == "multi_model_ensemble":
            # For multi-model ensemble, we could add per-model disagreement
            response["uncertainty"] = {
                "note": "Multi-model ensemble provides robustness through averaging. "
                        "For per-axis uncertainty bounds, use ensemble_projection mode.",
                "models_aggregated": len(required_models)
            }

        return response

    except Exception as e:
        logger.error(f"Analysis with mode failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
