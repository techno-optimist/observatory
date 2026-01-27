"""
Trajectory API for Cultural Soliton Observatory

Provides endpoints for tracking narrative evolution over time.
Analyzes how texts evolve through the cultural manifold, computing:
- Movement metrics (distance traveled, displacement)
- Velocity vectors between time points
- Trend analysis with Spearman correlations
- Mode transitions and interpretation

POST /api/v2/trajectory - Analyze trajectory from timestamped texts
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

from scipy.stats import spearmanr

from models.embedding import EmbeddingExtractor
from models.projection import Vector3, ProjectionHead
from analysis.mode_classifier import get_mode_classifier, legacy_mode

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class TrajectoryPointInput(BaseModel):
    """A single timestamped text for trajectory analysis."""
    timestamp: str = Field(
        ...,
        description="ISO format timestamp (e.g., '2024-01-01' or '2024-01-01T12:00:00Z')"
    )
    text: str = Field(
        ...,
        description="The text to analyze at this time point"
    )


class TrajectoryRequest(BaseModel):
    """Request for trajectory analysis."""
    name: str = Field(
        ...,
        description="Name/identifier for this trajectory (e.g., 'Brand Narrative')"
    )
    points: List[TrajectoryPointInput] = Field(
        ...,
        min_length=2,
        description="List of timestamped texts (minimum 2 points required)"
    )
    model_id: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model to use"
    )
    layer: int = Field(
        default=-1,
        description="Model layer to extract embeddings from"
    )


class TrajectoryPointOutput(BaseModel):
    """Output for a single trajectory point."""
    timestamp: str
    agency: float
    perceived_justice: float
    belonging: float
    mode: str


class VelocityVector(BaseModel):
    """Velocity vector between two time points."""
    from_timestamp: str = Field(..., alias="from")
    to_timestamp: str = Field(..., alias="to")
    delta_agency: float
    delta_justice: float
    delta_belonging: float

    class Config:
        populate_by_name = True


class TrendInfo(BaseModel):
    """Trend information for a single axis."""
    direction: str = Field(
        ...,
        description="'increasing', 'declining', or 'stable'"
    )
    correlation: float = Field(
        ...,
        description="Spearman correlation coefficient (-1 to 1)"
    )
    p_value: float = Field(
        ...,
        description="Statistical significance of the correlation"
    )


class ModeTransition(BaseModel):
    """A mode transition event."""
    from_mode: str = Field(..., alias="from")
    to_mode: str = Field(..., alias="to")
    timestamp: str

    class Config:
        populate_by_name = True


class TrajectoryOutput(BaseModel):
    """Core trajectory data."""
    points: List[TrajectoryPointOutput]
    total_distance_traveled: float
    net_displacement: float
    velocity_vectors: List[VelocityVector]


class TrendsOutput(BaseModel):
    """Trend analysis for all axes."""
    agency: TrendInfo
    perceived_justice: TrendInfo
    belonging: TrendInfo


class TrajectoryResponse(BaseModel):
    """Full trajectory analysis response."""
    trajectory: TrajectoryOutput
    trends: TrendsOutput
    mode_transitions: List[ModeTransition]
    interpretation: str


# ============================================================================
# Trajectory Analysis Functions
# ============================================================================

def compute_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute Euclidean distance between two 3D points."""
    return float(np.linalg.norm(p2 - p1))


def compute_total_distance(coords_list: List[np.ndarray]) -> float:
    """Compute total distance traveled through all points."""
    total = 0.0
    for i in range(1, len(coords_list)):
        total += compute_distance(coords_list[i-1], coords_list[i])
    return total


def compute_net_displacement(coords_list: List[np.ndarray]) -> float:
    """Compute straight-line distance from start to end."""
    if len(coords_list) < 2:
        return 0.0
    return compute_distance(coords_list[0], coords_list[-1])


def compute_velocity_vectors(
    coords_list: List[np.ndarray],
    timestamps: List[str]
) -> List[Dict[str, Any]]:
    """Compute velocity vectors between consecutive points."""
    vectors = []
    for i in range(1, len(coords_list)):
        delta = coords_list[i] - coords_list[i-1]
        vectors.append({
            "from": timestamps[i-1],
            "to": timestamps[i],
            "delta_agency": round(float(delta[0]), 4),
            "delta_justice": round(float(delta[1]), 4),
            "delta_belonging": round(float(delta[2]), 4)
        })
    return vectors


def compute_trend(values: List[float], timestamps: List[str]) -> Dict[str, Any]:
    """
    Compute trend analysis for a sequence of values using Spearman correlation.

    Uses time indices as the independent variable to correlate with values.
    """
    if len(values) < 3:
        # Not enough data for meaningful correlation
        start_val = values[0] if values else 0
        end_val = values[-1] if values else 0
        direction = "increasing" if end_val > start_val + 0.1 else (
            "declining" if end_val < start_val - 0.1 else "stable"
        )
        return {
            "direction": direction,
            "correlation": round(end_val - start_val, 2),
            "p_value": 1.0  # Not significant with too few points
        }

    # Create time indices (0, 1, 2, ...)
    time_indices = np.arange(len(values))

    # Compute Spearman correlation
    correlation, p_value = spearmanr(time_indices, values)

    # Handle NaN correlations (constant values)
    if np.isnan(correlation):
        correlation = 0.0
        p_value = 1.0

    # Determine direction based on correlation
    if correlation > 0.3:
        direction = "increasing"
    elif correlation < -0.3:
        direction = "declining"
    else:
        direction = "stable"

    return {
        "direction": direction,
        "correlation": round(float(correlation), 4),
        "p_value": round(float(p_value), 4)
    }


def detect_mode_transitions(
    modes: List[str],
    timestamps: List[str]
) -> List[Dict[str, str]]:
    """Detect points where mode changes."""
    transitions = []
    for i in range(1, len(modes)):
        if modes[i] != modes[i-1]:
            transitions.append({
                "from": modes[i-1],
                "to": modes[i],
                "timestamp": timestamps[i]
            })
    return transitions


def generate_interpretation(
    coords_list: List[np.ndarray],
    modes: List[str],
    trends: Dict[str, Dict],
    transitions: List[Dict]
) -> str:
    """
    Generate a human-readable interpretation of the trajectory.

    Analyzes the overall pattern and provides narrative insight.
    """
    if len(coords_list) < 2:
        return "Insufficient data points for interpretation."

    # Analyze overall movement
    start_coords = coords_list[0]
    end_coords = coords_list[-1]
    net_change = end_coords - start_coords

    # Identify dominant changes
    abs_changes = np.abs(net_change)
    axis_names = ["agency", "perceived justice", "belonging"]
    dominant_axis_idx = int(np.argmax(abs_changes))
    dominant_axis = axis_names[dominant_axis_idx]
    dominant_change = net_change[dominant_axis_idx]

    # Determine narrative arc type
    agency_trend = trends.get("agency", {}).get("direction", "stable")
    justice_trend = trends.get("perceived_justice", {}).get("direction", "stable")
    belonging_trend = trends.get("belonging", {}).get("direction", "stable")

    # Check for common narrative patterns
    interpretations = []

    # Recovery arc: agency/justice drops then belonging rises
    if agency_trend == "declining" and belonging_trend == "increasing":
        interpretations.append(
            "Narrative shows recovery arc: initial crisis (agency drop), "
            "followed by rebuilding with stronger community focus."
        )

    # Crisis pattern: multiple declining trends
    declining_count = sum(1 for t in [agency_trend, justice_trend, belonging_trend] if t == "declining")
    if declining_count >= 2:
        interpretations.append(
            "Narrative indicates ongoing challenges with declining metrics "
            "across multiple dimensions."
        )

    # Growth pattern: multiple increasing trends
    increasing_count = sum(1 for t in [agency_trend, justice_trend, belonging_trend] if t == "increasing")
    if increasing_count >= 2:
        interpretations.append(
            "Narrative shows positive momentum with improvements "
            "across multiple dimensions."
        )

    # Mode transition analysis
    if len(transitions) > 0:
        first_mode = modes[0]
        last_mode = modes[-1]
        if first_mode != last_mode:
            interpretations.append(
                f"Significant narrative shift from {first_mode} to {last_mode} mode."
            )
        if len(transitions) > 2:
            interpretations.append(
                f"High volatility detected with {len(transitions)} mode transitions."
            )

    # Stability pattern
    if declining_count == 0 and increasing_count == 0:
        interpretations.append(
            "Narrative remains relatively stable with minimal evolution over time."
        )

    # Dominant axis insight
    direction_word = "increase" if dominant_change > 0 else "decrease"
    magnitude = "significant" if abs(dominant_change) > 0.5 else "moderate"
    interpretations.append(
        f"Most notable change: {magnitude} {direction_word} in {dominant_axis} "
        f"({dominant_change:+.2f})."
    )

    # Combine interpretations
    if not interpretations:
        return "Mixed trajectory pattern without clear dominant narrative direction."

    return " ".join(interpretations[:3])  # Limit to top 3 insights


# ============================================================================
# API Endpoint
# ============================================================================

@router.post("/trajectory", response_model=TrajectoryResponse)
async def analyze_trajectory(request: TrajectoryRequest):
    """
    Analyze narrative trajectory from timestamped texts.

    Takes a series of texts with timestamps and computes:
    - 3D coordinates (agency, perceived_justice, belonging) for each point
    - Total distance traveled through the manifold
    - Net displacement (start to end)
    - Velocity vectors between consecutive points
    - Trend analysis with Spearman correlations for each axis
    - Mode transitions over time
    - Human-readable interpretation of the trajectory

    Minimum 2 points required. For reliable trend analysis,
    3+ points are recommended.

    Example request:
    ```json
    {
      "name": "Brand Narrative",
      "points": [
        {"timestamp": "2024-01-01", "text": "We're industry leaders..."},
        {"timestamp": "2024-06-01", "text": "We're working through challenges..."},
        {"timestamp": "2025-01-01", "text": "We've learned from our mistakes..."}
      ]
    }
    ```
    """
    # Import here to avoid circular imports
    from main import (
        model_manager, embedding_extractor, current_projection,
        ModelType
    )

    # Validate projection exists
    if current_projection is None:
        raise HTTPException(
            status_code=400,
            detail="No projection trained. Train a projection first using /training/train"
        )

    # Validate minimum points
    if len(request.points) < 2:
        raise HTTPException(
            status_code=400,
            detail="Minimum 2 points required for trajectory analysis"
        )

    # Load model if needed
    if not model_manager.is_loaded(request.model_id):
        try:
            model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model {request.model_id}: {str(e)}"
            )

    try:
        # Sort points by timestamp
        sorted_points = sorted(request.points, key=lambda p: p.timestamp)

        # Extract texts and timestamps
        texts = [p.text for p in sorted_points]
        timestamps = [p.timestamp for p in sorted_points]

        # Get embeddings for all texts
        embedding_results = embedding_extractor.extract(
            texts,
            request.model_id,
            layer=request.layer
        )

        # Project to 3D coordinates
        embeddings = np.array([r.embedding for r in embedding_results])
        projected = current_projection.project_batch(embeddings)

        # Convert to numpy arrays for calculations
        coords_list = [np.array([p.agency, p.fairness, p.belonging]) for p in projected]

        # Classify modes for each point
        classifier = get_mode_classifier()
        modes = []
        for coords in coords_list:
            mode_result = classifier.classify(coords)
            modes.append(mode_result["primary_mode"])

        # Build trajectory points output
        trajectory_points = []
        for i, (ts, proj, mode) in enumerate(zip(timestamps, projected, modes)):
            trajectory_points.append(TrajectoryPointOutput(
                timestamp=ts,
                agency=round(proj.agency, 4),
                perceived_justice=round(proj.fairness, 4),  # fairness -> perceived_justice
                belonging=round(proj.belonging, 4),
                mode=mode
            ))

        # Compute trajectory metrics
        total_distance = compute_total_distance(coords_list)
        net_displacement = compute_net_displacement(coords_list)
        velocity_vectors = compute_velocity_vectors(coords_list, timestamps)

        # Extract axis values for trend analysis
        agency_values = [c[0] for c in coords_list]
        justice_values = [c[1] for c in coords_list]
        belonging_values = [c[2] for c in coords_list]

        # Compute trends
        agency_trend = compute_trend(agency_values, timestamps)
        justice_trend = compute_trend(justice_values, timestamps)
        belonging_trend = compute_trend(belonging_values, timestamps)

        trends = TrendsOutput(
            agency=TrendInfo(**agency_trend),
            perceived_justice=TrendInfo(**justice_trend),
            belonging=TrendInfo(**belonging_trend)
        )

        # Detect mode transitions
        transitions_raw = detect_mode_transitions(modes, timestamps)
        mode_transitions = [
            ModeTransition(
                from_mode=t["from"],
                to_mode=t["to"],
                timestamp=t["timestamp"]
            )
            for t in transitions_raw
        ]

        # Generate interpretation
        trends_dict = {
            "agency": agency_trend,
            "perceived_justice": justice_trend,
            "belonging": belonging_trend
        }
        interpretation = generate_interpretation(
            coords_list, modes, trends_dict, transitions_raw
        )

        # Build velocity vector objects
        velocity_vector_objects = [
            VelocityVector(
                from_timestamp=v["from"],
                to_timestamp=v["to"],
                delta_agency=v["delta_agency"],
                delta_justice=v["delta_justice"],
                delta_belonging=v["delta_belonging"]
            )
            for v in velocity_vectors
        ]

        # Build response
        response = TrajectoryResponse(
            trajectory=TrajectoryOutput(
                points=trajectory_points,
                total_distance_traveled=round(total_distance, 4),
                net_displacement=round(net_displacement, 4),
                velocity_vectors=velocity_vector_objects
            ),
            trends=trends,
            mode_transitions=mode_transitions,
            interpretation=interpretation
        )

        logger.info(
            f"Trajectory analysis complete: {request.name}, "
            f"{len(request.points)} points, "
            f"distance={total_distance:.3f}, "
            f"transitions={len(mode_transitions)}"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trajectory analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trajectory/health")
async def trajectory_health():
    """Health check for trajectory API."""
    return {
        "status": "ok",
        "endpoint": "/api/v2/trajectory",
        "methods": ["POST"],
        "description": "Analyze narrative trajectory from timestamped texts"
    }
