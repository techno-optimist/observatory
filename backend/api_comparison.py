"""
Comparison API for Cultural Soliton Observatory

Provides gap analysis between two groups of texts, including:
- Euclidean distance between group centroids
- Per-axis gap analysis with direction indicators
- Effect sizes (Cohen's d) for each axis
- Statistical significance tests (independent t-tests)
- Mode distribution comparison
- Interpretation and bridge suggestions

Usage:
    from api_comparison import router as comparison_router
    app.include_router(comparison_router, prefix="/api/v2")

AXIS NAMING (January 2026):
The "fairness" axis has been renamed to "perceived_justice" in API responses.
Internal calculations use "fairness" for backward compatibility.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from scipy import stats

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class TextGroup(BaseModel):
    """A group of texts with a name for comparison."""
    name: str = Field(..., min_length=1, max_length=100, description="Name of the group")
    texts: List[str] = Field(..., min_length=1, description="List of text samples")

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        if len(v) < 1:
            raise ValueError("Each group must have at least 1 text")
        # Filter out empty strings
        v = [t.strip() for t in v if t.strip()]
        if len(v) < 1:
            raise ValueError("Each group must have at least 1 non-empty text")
        return v


class CompareRequest(BaseModel):
    """Request for comparing two groups of texts."""
    group_a: TextGroup = Field(..., description="First group of texts")
    group_b: TextGroup = Field(..., description="Second group of texts")
    model_id: str = Field(default="all-MiniLM-L6-v2", description="Embedding model to use")
    layer: int = Field(default=-1, description="Model layer to extract embeddings from")
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99, description="Confidence level for intervals")


class AxisGap(BaseModel):
    """Gap analysis for a single axis."""
    a_mean: float = Field(..., description="Mean value for group A")
    b_mean: float = Field(..., description="Mean value for group B")
    gap: float = Field(..., description="Absolute difference between means")
    direction: str = Field(..., description="Which group is higher")


class GapAnalysis(BaseModel):
    """Complete gap analysis between two groups."""
    euclidean_distance: float = Field(..., description="Euclidean distance between group centroids")
    axis_gaps: Dict[str, AxisGap] = Field(..., description="Gap analysis per axis")
    effect_sizes: Dict[str, float] = Field(..., description="Cohen's d effect sizes per axis")
    statistical_tests: Dict[str, float] = Field(..., description="T-test p-values per axis")


class ModeComparison(BaseModel):
    """Mode distribution comparison between groups."""
    group_a_dominant: str = Field(..., description="Dominant mode in group A")
    group_b_dominant: str = Field(..., description="Dominant mode in group B")
    mode_overlap: float = Field(..., description="Overlap coefficient between mode distributions")
    group_a_distribution: Optional[Dict[str, int]] = Field(None, description="Mode counts for group A")
    group_b_distribution: Optional[Dict[str, int]] = Field(None, description="Mode counts for group B")


class CompareResponse(BaseModel):
    """Response from the comparison endpoint."""
    gap_analysis: GapAnalysis
    mode_comparison: ModeComparison
    interpretation: str = Field(..., description="Human-readable interpretation of the comparison")
    bridge_suggestions: List[str] = Field(..., description="Suggestions for bridging the gap")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


# ============================================================================
# Statistical Utilities
# ============================================================================

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Cohen's d = (mean1 - mean2) / pooled_std

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return (mean1 - mean2) / pooled_std


def compute_ttest(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute independent samples t-test p-value.

    Uses Welch's t-test (does not assume equal variances).
    Returns p-value (two-tailed).
    """
    if len(group1) < 2 or len(group2) < 2:
        return 1.0  # Cannot compute with insufficient samples

    # Check for zero variance
    if np.var(group1) < 1e-10 and np.var(group2) < 1e-10:
        # Both groups have no variance - if means are equal, p=1, else p=0
        if np.abs(np.mean(group1) - np.mean(group2)) < 1e-10:
            return 1.0
        else:
            return 0.0

    try:
        _, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        return float(p_value) if not np.isnan(p_value) else 1.0
    except Exception:
        return 1.0


def compute_mode_overlap(dist1: Dict[str, int], dist2: Dict[str, int]) -> float:
    """
    Compute overlap coefficient between two mode distributions.

    Uses Bhattacharyya coefficient: sum of sqrt(p1 * p2) for each mode.
    Returns value between 0 (no overlap) and 1 (identical distributions).
    """
    # Get all modes
    all_modes = set(dist1.keys()) | set(dist2.keys())

    if not all_modes:
        return 1.0

    # Normalize to probabilities
    total1 = sum(dist1.values()) or 1
    total2 = sum(dist2.values()) or 1

    prob1 = {k: v / total1 for k, v in dist1.items()}
    prob2 = {k: v / total2 for k, v in dist2.items()}

    # Compute Bhattacharyya coefficient
    overlap = 0.0
    for mode in all_modes:
        p1 = prob1.get(mode, 0)
        p2 = prob2.get(mode, 0)
        overlap += np.sqrt(p1 * p2)

    return float(overlap)


def get_effect_size_interpretation(d: float) -> str:
    """Get human-readable interpretation of Cohen's d."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def generate_interpretation(
    gap_analysis: GapAnalysis,
    mode_comparison: ModeComparison,
    group_a_name: str,
    group_b_name: str
) -> str:
    """Generate human-readable interpretation of the comparison."""
    interpretations = []

    # Find the largest effect size
    effect_sizes = gap_analysis.effect_sizes
    largest_axis = max(effect_sizes.keys(), key=lambda k: abs(effect_sizes[k]))
    largest_d = effect_sizes[largest_axis]
    largest_interpretation = get_effect_size_interpretation(largest_d)

    # Extract axis name without "_cohens_d" suffix
    axis_name = largest_axis.replace("_cohens_d", "")

    # Get the gap info for this axis
    axis_gap = gap_analysis.axis_gaps.get(axis_name)
    if axis_gap:
        direction = f"{group_a_name}" if axis_gap.direction == "A higher" else f"{group_b_name}"

        if largest_interpretation == "large":
            interpretations.append(
                f"Large divergence on {axis_name} (d={largest_d:.1f}). "
                f"{direction} scores significantly higher on this dimension."
            )
        elif largest_interpretation == "medium":
            interpretations.append(
                f"Moderate difference on {axis_name} (d={largest_d:.1f}). "
                f"{direction} tends to score higher."
            )
        elif largest_interpretation == "small":
            interpretations.append(
                f"Small difference on {axis_name} (d={largest_d:.1f})."
            )

    # Add mode comparison
    if mode_comparison.group_a_dominant != mode_comparison.group_b_dominant:
        interpretations.append(
            f"{group_a_name} dominant mode is {mode_comparison.group_a_dominant}, "
            f"while {group_b_name} is {mode_comparison.group_b_dominant}."
        )

    # Add statistical significance note
    significant_axes = [
        k.replace("_ttest_p", "")
        for k, v in gap_analysis.statistical_tests.items()
        if v < 0.05
    ]
    if significant_axes:
        interpretations.append(
            f"Statistically significant differences (p<0.05) found on: {', '.join(significant_axes)}."
        )

    return " ".join(interpretations) if interpretations else "No significant differences detected."


def generate_bridge_suggestions(
    gap_analysis: GapAnalysis,
    mode_comparison: ModeComparison,
    group_a_name: str,
    group_b_name: str
) -> List[str]:
    """Generate suggestions for bridging the gap between groups."""
    suggestions = []

    # Analyze each axis gap
    for axis_name, gap_info in gap_analysis.axis_gaps.items():
        effect_key = f"{axis_name}_cohens_d"
        if effect_key in gap_analysis.effect_sizes:
            d = abs(gap_analysis.effect_sizes[effect_key])

            if d >= 0.5:  # Medium or large effect
                if axis_name == "perceived_justice":
                    if gap_info.direction == "A higher":
                        suggestions.append(
                            f"Acknowledge justice concerns from {group_b_name}"
                        )
                    else:
                        suggestions.append(
                            f"Acknowledge justice concerns from {group_a_name}"
                        )
                elif axis_name == "agency":
                    lower_group = group_b_name if gap_info.direction == "A higher" else group_a_name
                    suggestions.append(
                        f"Empower {lower_group} with more autonomy and decision-making opportunities"
                    )
                elif axis_name == "belonging":
                    suggestions.append("Find shared belonging through common values or goals")

    # Mode-based suggestions
    if mode_comparison.mode_overlap < 0.5:
        suggestions.append("Identify narrative common ground between the different modes")

    # If groups have very different modes
    if mode_comparison.group_a_dominant != mode_comparison.group_b_dominant:
        suggestions.append("Facilitate dialogue to understand different perspectives")

    # Default suggestion if none generated
    if not suggestions:
        suggestions.append("Continue monitoring for emerging differences")

    return suggestions[:5]  # Limit to 5 suggestions


# ============================================================================
# Main Comparison Endpoint
# ============================================================================

@router.post("/compare", response_model=CompareResponse)
async def compare_groups(request: CompareRequest):
    """
    Compare two groups of texts and return comprehensive gap analysis.

    This endpoint analyzes the cultural/narrative differences between two groups by:
    1. Embedding all texts from both groups
    2. Projecting embeddings to the 3D cultural manifold (agency, perceived_justice, belonging)
    3. Computing statistical comparisons (t-tests, effect sizes)
    4. Classifying modes for each text
    5. Generating interpretations and bridge suggestions

    Returns:
    - gap_analysis: Statistical comparison including effect sizes and significance tests
    - mode_comparison: Distribution of narrative modes in each group
    - interpretation: Human-readable summary of findings
    - bridge_suggestions: Actionable suggestions for bridging identified gaps

    Note: For reliable statistical tests, each group should have at least 3-5 texts.
    Effect sizes (Cohen's d) are more reliable with larger samples.
    """
    # Import here to avoid circular imports
    from main import (
        model_manager, embedding_extractor, current_projection,
        projection_trainer, ModelType
    )
    from analysis.mode_classifier import get_mode_classifier
    from models.projection import Vector3

    # Validate projection is available
    projection = current_projection
    if projection is None:
        # Try to load from disk
        projection = projection_trainer.load_projection()
        if projection is None:
            raise HTTPException(
                status_code=400,
                detail="No projection trained. Train a projection first using /training/train"
            )

    # Load model if needed
    if not model_manager.is_loaded(request.model_id):
        try:
            model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load model '{request.model_id}': {str(e)}"
            )

    try:
        # Embed all texts from both groups
        all_texts_a = request.group_a.texts
        all_texts_b = request.group_b.texts

        results_a = embedding_extractor.extract(
            all_texts_a, request.model_id, layer=request.layer
        )
        results_b = embedding_extractor.extract(
            all_texts_b, request.model_id, layer=request.layer
        )

        # Handle single vs multiple results
        if not isinstance(results_a, list):
            results_a = [results_a]
        if not isinstance(results_b, list):
            results_b = [results_b]

        embeddings_a = np.array([r.embedding for r in results_a])
        embeddings_b = np.array([r.embedding for r in results_b])

        # Validate embedding dimensions match projection
        if projection.embedding_dim is not None:
            if embeddings_a.shape[1] != projection.embedding_dim:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model dimension mismatch: projection expects {projection.embedding_dim} dims, "
                           f"but model produces {embeddings_a.shape[1]} dims"
                )

        # Project all embeddings to 3D manifold
        projections_a = projection.project_batch(embeddings_a)
        projections_b = projection.project_batch(embeddings_b)

        # Convert to numpy arrays for statistical analysis
        coords_a = np.array([[p.agency, p.fairness, p.belonging] for p in projections_a])
        coords_b = np.array([[p.agency, p.fairness, p.belonging] for p in projections_b])

        # Compute group centroids
        centroid_a = np.mean(coords_a, axis=0)
        centroid_b = np.mean(coords_b, axis=0)

        # Euclidean distance between centroids
        euclidean_distance = float(np.linalg.norm(centroid_a - centroid_b))

        # Per-axis analysis
        axis_names = ["agency", "perceived_justice", "belonging"]
        axis_indices = [0, 1, 2]  # agency, fairness(=perceived_justice), belonging

        axis_gaps = {}
        effect_sizes = {}
        statistical_tests = {}

        for axis_name, idx in zip(axis_names, axis_indices):
            values_a = coords_a[:, idx]
            values_b = coords_b[:, idx]

            mean_a = float(np.mean(values_a))
            mean_b = float(np.mean(values_b))
            gap = abs(mean_a - mean_b)
            direction = "A higher" if mean_a > mean_b else "B higher" if mean_b > mean_a else "Equal"

            axis_gaps[axis_name] = AxisGap(
                a_mean=round(mean_a, 4),
                b_mean=round(mean_b, 4),
                gap=round(gap, 4),
                direction=direction
            )

            # Effect size (Cohen's d)
            cohens_d = compute_cohens_d(values_a, values_b)
            effect_sizes[f"{axis_name}_cohens_d"] = round(cohens_d, 4)

            # T-test
            p_value = compute_ttest(values_a, values_b)
            statistical_tests[f"{axis_name}_ttest_p"] = round(p_value, 4)

        gap_analysis = GapAnalysis(
            euclidean_distance=round(euclidean_distance, 4),
            axis_gaps=axis_gaps,
            effect_sizes=effect_sizes,
            statistical_tests=statistical_tests
        )

        # Mode classification for each text
        classifier = get_mode_classifier()

        modes_a = []
        modes_b = []

        for coords in coords_a:
            result = classifier.classify(coords)
            modes_a.append(result["primary_mode"])

        for coords in coords_b:
            result = classifier.classify(coords)
            modes_b.append(result["primary_mode"])

        # Mode distributions
        mode_dist_a = {}
        mode_dist_b = {}

        for mode in modes_a:
            mode_dist_a[mode] = mode_dist_a.get(mode, 0) + 1

        for mode in modes_b:
            mode_dist_b[mode] = mode_dist_b.get(mode, 0) + 1

        # Dominant modes
        dominant_a = max(mode_dist_a.keys(), key=lambda k: mode_dist_a[k]) if mode_dist_a else "UNKNOWN"
        dominant_b = max(mode_dist_b.keys(), key=lambda k: mode_dist_b[k]) if mode_dist_b else "UNKNOWN"

        # Mode overlap
        mode_overlap = compute_mode_overlap(mode_dist_a, mode_dist_b)

        mode_comparison = ModeComparison(
            group_a_dominant=dominant_a,
            group_b_dominant=dominant_b,
            mode_overlap=round(mode_overlap, 4),
            group_a_distribution=mode_dist_a,
            group_b_distribution=mode_dist_b
        )

        # Generate interpretation
        interpretation = generate_interpretation(
            gap_analysis,
            mode_comparison,
            request.group_a.name,
            request.group_b.name
        )

        # Generate bridge suggestions
        bridge_suggestions = generate_bridge_suggestions(
            gap_analysis,
            mode_comparison,
            request.group_a.name,
            request.group_b.name
        )

        # Metadata
        metadata = {
            "group_a_size": len(all_texts_a),
            "group_b_size": len(all_texts_b),
            "model_id": request.model_id,
            "layer": request.layer,
            "confidence_level": request.confidence_level,
            "centroid_a": {
                "agency": round(centroid_a[0], 4),
                "perceived_justice": round(centroid_a[1], 4),
                "belonging": round(centroid_a[2], 4)
            },
            "centroid_b": {
                "agency": round(centroid_b[0], 4),
                "perceived_justice": round(centroid_b[1], 4),
                "belonging": round(centroid_b[2], 4)
            }
        }

        return CompareResponse(
            gap_analysis=gap_analysis,
            mode_comparison=mode_comparison,
            interpretation=interpretation,
            bridge_suggestions=bridge_suggestions,
            metadata=metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Additional Comparison Endpoints
# ============================================================================

class QuickCompareRequest(BaseModel):
    """Simplified request for quick comparison."""
    texts_a: List[str] = Field(..., min_length=1, description="First group of texts")
    texts_b: List[str] = Field(..., min_length=1, description="Second group of texts")
    model_id: str = Field(default="all-MiniLM-L6-v2")


class QuickCompareResponse(BaseModel):
    """Simplified response for quick comparison."""
    distance: float = Field(..., description="Euclidean distance between centroids")
    largest_gap_axis: str = Field(..., description="Axis with largest gap")
    largest_gap_value: float = Field(..., description="Value of largest gap")
    dominant_mode_a: str
    dominant_mode_b: str
    significant_differences: List[str] = Field(..., description="Axes with p<0.05")


@router.post("/compare/quick", response_model=QuickCompareResponse)
async def quick_compare(request: QuickCompareRequest):
    """
    Quick comparison endpoint with simplified input/output.

    Use this for rapid comparisons when you don't need the full analysis.
    """
    # Convert to full request format
    full_request = CompareRequest(
        group_a=TextGroup(name="Group A", texts=request.texts_a),
        group_b=TextGroup(name="Group B", texts=request.texts_b),
        model_id=request.model_id
    )

    # Get full response
    full_response = await compare_groups(full_request)

    # Extract key metrics
    axis_gaps = full_response.gap_analysis.axis_gaps
    largest_axis = max(axis_gaps.keys(), key=lambda k: axis_gaps[k].gap)

    significant = [
        k.replace("_ttest_p", "")
        for k, v in full_response.gap_analysis.statistical_tests.items()
        if v < 0.05
    ]

    return QuickCompareResponse(
        distance=full_response.gap_analysis.euclidean_distance,
        largest_gap_axis=largest_axis,
        largest_gap_value=axis_gaps[largest_axis].gap,
        dominant_mode_a=full_response.mode_comparison.group_a_dominant,
        dominant_mode_b=full_response.mode_comparison.group_b_dominant,
        significant_differences=significant
    )


class MultiGroupCompareRequest(BaseModel):
    """Request for comparing multiple groups at once."""
    groups: List[TextGroup] = Field(..., min_length=2, max_length=10)
    model_id: str = Field(default="all-MiniLM-L6-v2")


@router.post("/compare/multi")
async def compare_multiple_groups(request: MultiGroupCompareRequest):
    """
    Compare multiple groups pairwise.

    Returns a matrix of pairwise comparisons for all groups.
    Useful for analyzing multiple stakeholder perspectives at once.
    """
    if len(request.groups) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 groups to compare")

    results = {
        "pairwise_comparisons": [],
        "summary": {
            "total_groups": len(request.groups),
            "total_comparisons": len(request.groups) * (len(request.groups) - 1) // 2
        }
    }

    # Perform pairwise comparisons
    for i in range(len(request.groups)):
        for j in range(i + 1, len(request.groups)):
            group_a = request.groups[i]
            group_b = request.groups[j]

            compare_request = CompareRequest(
                group_a=group_a,
                group_b=group_b,
                model_id=request.model_id
            )

            try:
                comparison = await compare_groups(compare_request)
                results["pairwise_comparisons"].append({
                    "group_a": group_a.name,
                    "group_b": group_b.name,
                    "euclidean_distance": comparison.gap_analysis.euclidean_distance,
                    "mode_a": comparison.mode_comparison.group_a_dominant,
                    "mode_b": comparison.mode_comparison.group_b_dominant,
                    "mode_overlap": comparison.mode_comparison.mode_overlap,
                    "significant_axes": [
                        k.replace("_ttest_p", "")
                        for k, v in comparison.gap_analysis.statistical_tests.items()
                        if v < 0.05
                    ],
                    "interpretation": comparison.interpretation
                })
            except Exception as e:
                logger.error(f"Failed comparison between {group_a.name} and {group_b.name}: {e}")
                results["pairwise_comparisons"].append({
                    "group_a": group_a.name,
                    "group_b": group_b.name,
                    "error": str(e)
                })

    # Find the most distant pair
    successful = [c for c in results["pairwise_comparisons"] if "error" not in c]
    if successful:
        most_distant = max(successful, key=lambda x: x["euclidean_distance"])
        most_similar = min(successful, key=lambda x: x["euclidean_distance"])

        results["summary"]["most_distant_pair"] = {
            "groups": [most_distant["group_a"], most_distant["group_b"]],
            "distance": most_distant["euclidean_distance"]
        }
        results["summary"]["most_similar_pair"] = {
            "groups": [most_similar["group_a"], most_similar["group_b"]],
            "distance": most_similar["euclidean_distance"]
        }

    return results


@router.get("/compare/effect-size-guide")
async def get_effect_size_guide():
    """
    Get a guide for interpreting effect sizes and statistical tests.

    Returns interpretation guidelines for Cohen's d and p-values.
    """
    return {
        "cohens_d": {
            "description": "Standardized measure of effect size between two groups",
            "interpretation": {
                "negligible": "|d| < 0.2",
                "small": "0.2 <= |d| < 0.5",
                "medium": "0.5 <= |d| < 0.8",
                "large": "|d| >= 0.8"
            },
            "note": "Sign indicates direction: positive = Group A higher, negative = Group B higher"
        },
        "p_value": {
            "description": "Probability of observing this difference by chance",
            "interpretation": {
                "highly_significant": "p < 0.01",
                "significant": "p < 0.05",
                "marginally_significant": "p < 0.10",
                "not_significant": "p >= 0.10"
            },
            "note": "Uses Welch's t-test (independent samples, unequal variances assumed)"
        },
        "mode_overlap": {
            "description": "Bhattacharyya coefficient measuring distribution similarity",
            "interpretation": {
                "very_similar": "overlap > 0.8",
                "moderate_overlap": "0.5 <= overlap <= 0.8",
                "different": "0.2 <= overlap < 0.5",
                "very_different": "overlap < 0.2"
            }
        },
        "sample_size_recommendations": {
            "minimum": "1 text per group (but unreliable)",
            "adequate": "5-10 texts per group",
            "good": "20+ texts per group",
            "excellent": "50+ texts per group"
        }
    }
