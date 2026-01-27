"""
Force Field API Endpoints
=========================

API endpoints for attractor/detractor force field analysis.

These endpoints extend the observatory's analytical capabilities by adding:
- Force field analysis (attractor/detractor strengths and targets)
- Trajectory force dynamics
- Force-aware comparisons between groups
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np

from analysis.force_field import (
    get_force_field_analyzer,
    analyze_force_field,
    analyze_trajectory_forces,
    ATTRACTOR_TARGETS,
    DETRACTOR_SOURCES,
    FORCE_QUADRANTS,
)

router = APIRouter(prefix="/api/v2/forces", tags=["Force Field Analysis"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ForceFieldRequest(BaseModel):
    """Request for single text force field analysis."""
    text: str = Field(..., description="Text to analyze")


class ForceFieldResponse(BaseModel):
    """Response for force field analysis."""
    text: str
    attractor_strength: float = Field(..., description="Overall pull toward positive (-2 to +2)")
    detractor_strength: float = Field(..., description="Overall push from negative (-2 to +2)")
    primary_attractor: Optional[str] = Field(None, description="Main target being drawn toward")
    primary_detractor: Optional[str] = Field(None, description="Main source being fled from")
    secondary_attractor: Optional[str] = None
    secondary_detractor: Optional[str] = None
    attractor_scores: Dict[str, float] = Field(default_factory=dict)
    detractor_scores: Dict[str, float] = Field(default_factory=dict)
    force_quadrant: str = Field(..., description="Force field quadrant")
    quadrant_description: str = Field(..., description="Description of the quadrant")
    energy_level: str = Field(..., description="high/moderate/low")
    net_force: float = Field(..., description="Combined force magnitude")
    force_direction: str = Field(..., description="TOWARD/AWAY/BALANCED")


class TrajectoryForceRequest(BaseModel):
    """Request for trajectory force analysis."""
    texts: List[str] = Field(..., description="Sequence of texts to analyze")


class BatchForceRequest(BaseModel):
    """Request for batch force analysis."""
    texts: List[str] = Field(..., description="Texts to analyze")


class GroupForceComparisonRequest(BaseModel):
    """Request for comparing force fields between groups."""
    group_a: Dict[str, Any] = Field(..., description="First group with name and texts")
    group_b: Dict[str, Any] = Field(..., description="Second group with name and texts")


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/analyze", response_model=ForceFieldResponse)
async def analyze_text_forces(request: ForceFieldRequest):
    """
    Analyze the attractor/detractor force field of a single text.

    Returns the forces acting on the narrative - what it's being pulled toward
    (attractors) and pushed away from (detractors).
    """
    try:
        result = analyze_force_field(request.text)
        return ForceFieldResponse(text=request.text, **result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trajectory")
async def analyze_trajectory_force_dynamics(request: TrajectoryForceRequest):
    """
    Analyze how force fields evolve across a sequence of texts.

    Tracks attractor/detractor changes over time, identifying:
    - Attractor shifts (changing goals)
    - Detractor emergence/resolution
    - Energy level changes
    - Quadrant transitions
    """
    if len(request.texts) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 texts for trajectory analysis")

    try:
        result = analyze_trajectory_forces(request.texts)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def batch_analyze_forces(request: BatchForceRequest):
    """
    Analyze force fields for multiple texts.

    Returns individual analyses plus aggregate statistics.
    """
    try:
        analyzer = get_force_field_analyzer()
        analyses = [analyzer.analyze(text) for text in request.texts]

        # Calculate aggregates
        attractor_values = [a["attractor_strength"] for a in analyses]
        detractor_values = [a["detractor_strength"] for a in analyses]

        # Count quadrants
        quadrant_counts = {}
        for a in analyses:
            q = a["force_quadrant"]
            quadrant_counts[q] = quadrant_counts.get(q, 0) + 1

        # Count attractors and detractors
        attractor_counts = {}
        detractor_counts = {}
        for a in analyses:
            if a["primary_attractor"]:
                target = a["primary_attractor"]
                attractor_counts[target] = attractor_counts.get(target, 0) + 1
            if a["primary_detractor"]:
                source = a["primary_detractor"]
                detractor_counts[source] = detractor_counts.get(source, 0) + 1

        return {
            "analyses": [{"text": t[:80] + "..." if len(t) > 80 else t, **a}
                        for t, a in zip(request.texts, analyses)],
            "n_texts": len(request.texts),
            "aggregate": {
                "mean_attractor_strength": float(np.mean(attractor_values)),
                "mean_detractor_strength": float(np.mean(detractor_values)),
                "std_attractor_strength": float(np.std(attractor_values)),
                "std_detractor_strength": float(np.std(detractor_values)),
                "quadrant_distribution": quadrant_counts,
                "dominant_quadrant": max(quadrant_counts, key=quadrant_counts.get) if quadrant_counts else None,
                "attractor_distribution": attractor_counts,
                "dominant_attractor": max(attractor_counts, key=attractor_counts.get) if attractor_counts else None,
                "detractor_distribution": detractor_counts,
                "dominant_detractor": max(detractor_counts, key=detractor_counts.get) if detractor_counts else None,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_group_forces(request: GroupForceComparisonRequest):
    """
    Compare force fields between two groups of narratives.

    Identifies differences in:
    - Attractor/detractor strengths
    - Primary targets/sources
    - Energy levels
    - Quadrant distributions
    """
    try:
        analyzer = get_force_field_analyzer()

        group_a_name = request.group_a.get("name", "Group A")
        group_b_name = request.group_b.get("name", "Group B")
        group_a_texts = request.group_a.get("texts", [])
        group_b_texts = request.group_b.get("texts", [])

        if not group_a_texts or not group_b_texts:
            raise HTTPException(status_code=400, detail="Both groups must have texts")

        # Analyze both groups
        analyses_a = [analyzer.analyze(text) for text in group_a_texts]
        analyses_b = [analyzer.analyze(text) for text in group_b_texts]

        def calculate_group_stats(analyses):
            attractor_vals = [a["attractor_strength"] for a in analyses]
            detractor_vals = [a["detractor_strength"] for a in analyses]

            quadrants = {}
            attractors = {}
            detractors = {}

            for a in analyses:
                q = a["force_quadrant"]
                quadrants[q] = quadrants.get(q, 0) + 1
                if a["primary_attractor"]:
                    attractors[a["primary_attractor"]] = attractors.get(a["primary_attractor"], 0) + 1
                if a["primary_detractor"]:
                    detractors[a["primary_detractor"]] = detractors.get(a["primary_detractor"], 0) + 1

            return {
                "n": len(analyses),
                "mean_attractor": float(np.mean(attractor_vals)),
                "mean_detractor": float(np.mean(detractor_vals)),
                "std_attractor": float(np.std(attractor_vals)) if len(attractor_vals) > 1 else 0,
                "std_detractor": float(np.std(detractor_vals)) if len(detractor_vals) > 1 else 0,
                "dominant_quadrant": max(quadrants, key=quadrants.get) if quadrants else None,
                "dominant_attractor": max(attractors, key=attractors.get) if attractors else None,
                "dominant_detractor": max(detractors, key=detractors.get) if detractors else None,
                "quadrant_distribution": quadrants,
                "attractor_distribution": attractors,
                "detractor_distribution": detractors,
            }

        stats_a = calculate_group_stats(analyses_a)
        stats_b = calculate_group_stats(analyses_b)

        # Calculate gaps
        attractor_gap = stats_a["mean_attractor"] - stats_b["mean_attractor"]
        detractor_gap = stats_a["mean_detractor"] - stats_b["mean_detractor"]

        # Interpretation
        interpretations = []

        if abs(attractor_gap) > 0.3:
            higher = group_a_name if attractor_gap > 0 else group_b_name
            interpretations.append(f"{higher} shows stronger attractor pull (+{abs(attractor_gap):.2f})")

        if abs(detractor_gap) > 0.3:
            higher = group_a_name if detractor_gap > 0 else group_b_name
            interpretations.append(f"{higher} shows stronger detractor push (+{abs(detractor_gap):.2f})")

        if stats_a["dominant_attractor"] != stats_b["dominant_attractor"]:
            interpretations.append(
                f"Different primary attractors: {group_a_name}→{stats_a['dominant_attractor']}, "
                f"{group_b_name}→{stats_b['dominant_attractor']}"
            )

        if stats_a["dominant_detractor"] != stats_b["dominant_detractor"]:
            interpretations.append(
                f"Different primary detractors: {group_a_name}←{stats_a['dominant_detractor']}, "
                f"{group_b_name}←{stats_b['dominant_detractor']}"
            )

        return {
            "group_a": {
                "name": group_a_name,
                "stats": stats_a,
            },
            "group_b": {
                "name": group_b_name,
                "stats": stats_b,
            },
            "comparison": {
                "attractor_gap": round(attractor_gap, 4),
                "detractor_gap": round(detractor_gap, 4),
                "attractor_higher": group_a_name if attractor_gap > 0 else group_b_name,
                "detractor_higher": group_a_name if detractor_gap > 0 else group_b_name,
                "same_dominant_attractor": stats_a["dominant_attractor"] == stats_b["dominant_attractor"],
                "same_dominant_detractor": stats_a["dominant_detractor"] == stats_b["dominant_detractor"],
            },
            "interpretation": " | ".join(interpretations) if interpretations else "Groups show similar force fields",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/targets")
async def list_targets():
    """List all attractor targets and detractor sources."""
    return {
        "attractor_targets": {
            "AUTONOMY": "Self-determination, freedom, control",
            "COMMUNITY": "Belonging, connection, togetherness",
            "JUSTICE": "Fairness, equity, rightness",
            "MEANING": "Purpose, transcendence, significance",
            "SECURITY": "Stability, safety, predictability",
            "RECOGNITION": "Visibility, validation, appreciation",
        },
        "detractor_sources": {
            "OPPRESSION": "Control, domination, powerlessness",
            "ISOLATION": "Alienation, abandonment, loneliness",
            "INJUSTICE": "Unfairness, corruption, betrayal",
            "MEANINGLESSNESS": "Futility, nihilism, absurdity",
            "INSTABILITY": "Chaos, threat, unpredictability",
            "INVISIBILITY": "Being ignored, dismissed, devalued",
        },
        "force_quadrants": {
            name: {
                "description": info["description"],
                "energy": info["energy"],
            }
            for name, info in FORCE_QUADRANTS.items()
        },
    }
