"""
Advanced Analytics API for the Cultural Soliton Observatory

This module provides sophisticated statistical analysis endpoints:

1. Outlier Detection (/api/v2/analytics/outliers)
   - Detect anomalous narratives within a corpus
   - Uses z-scores and Mahalanobis distance

2. Cohort Analysis (/api/v2/analytics/cohorts)
   - Compare multiple groups with ANOVA
   - Compute pairwise distances between cohorts

3. Mode Flow Analysis (/api/v2/analytics/mode-flow)
   - Analyze mode transitions in a sequence
   - Detect radicalization and other flow patterns

AXIS NAMING (January 2026):
The "fairness" axis has been renamed to "perceived_justice" in API responses.
Internal coordinate order: [agency, fairness(=perceived_justice), belonging]
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis
from collections import Counter
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class OutlierDetectionRequest(BaseModel):
    """Request for outlier detection within a corpus."""
    corpus: List[str] = Field(
        ...,
        description="List of texts representing the reference corpus",
        min_length=3
    )
    test_text: str = Field(
        ...,
        description="The text to check for anomalousness"
    )
    threshold: float = Field(
        default=2.0,
        ge=0.5,
        le=5.0,
        description="Standard deviations threshold for outlier detection"
    )
    model_id: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model to use"
    )


class OutlierDetectionResponse(BaseModel):
    """Response from outlier detection."""
    is_outlier: bool
    z_scores: Dict[str, float]
    mahalanobis_distance: float
    nearest_corpus_text: str
    interpretation: str


class CohortAnalysisRequest(BaseModel):
    """Request for multi-cohort analysis."""
    cohorts: Dict[str, List[str]] = Field(
        ...,
        description="Dictionary mapping cohort names to lists of texts",
        min_length=2
    )
    model_id: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model to use"
    )


class CohortProfile(BaseModel):
    """Profile for a single cohort."""
    mean_agency: float
    mean_perceived_justice: float
    mean_belonging: float
    dominant_mode: str
    n_texts: int
    std_agency: Optional[float] = None
    std_perceived_justice: Optional[float] = None
    std_belonging: Optional[float] = None


class ANOVAResult(BaseModel):
    """ANOVA result for a single axis."""
    f_stat: float
    p_value: float
    significant: bool


class CohortAnalysisResponse(BaseModel):
    """Response from cohort analysis."""
    cohort_profiles: Dict[str, CohortProfile]
    anova_results: Dict[str, ANOVAResult]
    pairwise_distances: Dict[str, float]


class ModeFlowRequest(BaseModel):
    """Request for mode flow analysis."""
    texts: List[str] = Field(
        ...,
        description="Sequence of texts to analyze for mode transitions",
        min_length=2
    )
    model_id: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model to use"
    )


class ModeTransition(BaseModel):
    """A mode transition with probability."""
    from_mode: str = Field(..., alias="from")
    to_mode: str = Field(..., alias="to")
    probability: float

    class Config:
        populate_by_name = True


class ModeFlowResponse(BaseModel):
    """Response from mode flow analysis."""
    sequence: List[str]
    transitions: List[Dict[str, Any]]
    flow_pattern: str
    interpretation: str


# =============================================================================
# Helper Functions
# =============================================================================

def compute_z_scores(
    test_vector: np.ndarray,
    corpus_vectors: np.ndarray
) -> Dict[str, float]:
    """
    Compute z-scores for each axis of the test vector relative to corpus.

    Args:
        test_vector: 3D vector [agency, perceived_justice, belonging]
        corpus_vectors: (N, 3) array of corpus vectors

    Returns:
        Dictionary with z-scores for each axis
    """
    axis_names = ["agency", "perceived_justice", "belonging"]
    z_scores = {}

    for i, axis in enumerate(axis_names):
        mean = corpus_vectors[:, i].mean()
        std = corpus_vectors[:, i].std()

        if std > 0:
            z_scores[axis] = float((test_vector[i] - mean) / std)
        else:
            z_scores[axis] = 0.0

    return z_scores


def compute_mahalanobis_distance(
    test_vector: np.ndarray,
    corpus_vectors: np.ndarray
) -> float:
    """
    Compute Mahalanobis distance of test vector from corpus distribution.

    Args:
        test_vector: 3D vector [agency, perceived_justice, belonging]
        corpus_vectors: (N, 3) array of corpus vectors

    Returns:
        Mahalanobis distance (scalar)
    """
    mean = corpus_vectors.mean(axis=0)
    cov = np.cov(corpus_vectors, rowvar=False)

    # Handle singular covariance matrix
    try:
        cov_inv = np.linalg.inv(cov)
        dist = mahalanobis(test_vector, mean, cov_inv)
    except np.linalg.LinAlgError:
        # Fall back to regularized covariance
        cov_reg = cov + np.eye(3) * 1e-6
        cov_inv = np.linalg.inv(cov_reg)
        dist = mahalanobis(test_vector, mean, cov_inv)

    return float(dist)


def find_nearest_text(
    test_vector: np.ndarray,
    corpus_vectors: np.ndarray,
    corpus_texts: List[str]
) -> str:
    """Find the text in corpus closest to test vector."""
    distances = np.linalg.norm(corpus_vectors - test_vector, axis=1)
    nearest_idx = np.argmin(distances)
    return corpus_texts[nearest_idx]


def generate_outlier_interpretation(
    z_scores: Dict[str, float],
    threshold: float
) -> str:
    """Generate human-readable interpretation of outlier analysis."""
    outlier_axes = []

    for axis, z in z_scores.items():
        if abs(z) > threshold:
            direction = "high" if z > 0 else "low"
            outlier_axes.append(f"unusually {direction} {axis}")

    if not outlier_axes:
        return "Text is within normal corpus distribution"
    elif len(outlier_axes) == 1:
        return f"Text shows {outlier_axes[0]} compared to corpus"
    else:
        return f"Text shows {', '.join(outlier_axes[:-1])} and {outlier_axes[-1]} compared to corpus"


def classify_mode(coords: np.ndarray, classifier=None) -> str:
    """Classify coordinates into a mode."""
    # Import the classifier
    try:
        from analysis.mode_classifier import get_mode_classifier
        if classifier is None:
            classifier = get_mode_classifier()
        result = classifier.classify(coords)
        return result["primary_mode"]
    except ImportError:
        # Fallback simple classification
        agency, fairness, belonging = coords

        if agency > 0.5 and fairness > 0.5:
            return "HEROIC"
        elif agency < -0.5 and fairness < -0.5:
            return "VICTIM"
        elif agency > 0.5 and fairness < -0.5:
            return "CYNICAL_ACHIEVER"
        elif belonging > 0.5 and fairness > 0.5:
            return "COMMUNAL"
        elif abs(agency) < 0.5 and abs(fairness) < 0.5 and abs(belonging) < 0.5:
            return "NEUTRAL"
        else:
            return "TRANSITIONAL"


def compute_euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.linalg.norm(vec1 - vec2))


def compute_transition_probability(
    from_mode: str,
    to_mode: str,
    transition_matrix: Dict[str, Dict[str, int]]
) -> float:
    """Compute transition probability from transition counts."""
    if from_mode not in transition_matrix:
        return 0.0

    from_counts = transition_matrix[from_mode]
    total = sum(from_counts.values())

    if total == 0:
        return 0.0

    return float(from_counts.get(to_mode, 0) / total)


def classify_flow_pattern(sequence: List[str], transitions: List[Dict]) -> Tuple[str, str]:
    """
    Classify the overall flow pattern and provide interpretation.

    Returns:
        Tuple of (pattern_name, interpretation)
    """
    if len(sequence) < 2:
        return "static", "No transitions detected"

    # Define mode categories
    positive_modes = {"HEROIC", "COMMUNAL", "TRANSCENDENT"}
    shadow_modes = {"CYNICAL_ACHIEVER", "VICTIM", "PARANOID"}
    exit_modes = {"SPIRITUAL_EXIT", "SOCIAL_EXIT", "PROTEST_EXIT"}
    ambivalent_modes = {"CONFLICTED", "TRANSITIONAL", "NEUTRAL"}

    # Analyze trajectory direction
    start_mode = sequence[0]
    end_mode = sequence[-1]

    # Count mode categories in sequence
    category_sequence = []
    for mode in sequence:
        if mode in positive_modes:
            category_sequence.append("positive")
        elif mode in shadow_modes:
            category_sequence.append("shadow")
        elif mode in exit_modes:
            category_sequence.append("exit")
        else:
            category_sequence.append("ambivalent")

    # Detect patterns
    positive_count = category_sequence.count("positive")
    shadow_count = category_sequence.count("shadow")
    exit_count = category_sequence.count("exit")

    # Pattern detection logic

    # Radicalization: positive -> shadow/exit or ambivalent -> shadow -> exit
    if (category_sequence[0] == "positive" and category_sequence[-1] in ["shadow", "exit"]):
        return "radicalization", "Classic radicalization arc: loss of status -> grievance -> mobilization"

    if (shadow_count > 0 and exit_count > 0 and
        category_sequence.index("shadow") if "shadow" in category_sequence else -1 <
        len(category_sequence) - 1 - category_sequence[::-1].index("exit") if "exit" in category_sequence else len(category_sequence)):
        return "radicalization", "Radicalization pattern: shadow phase leading to exit/protest"

    # Recovery: shadow -> positive
    if (category_sequence[0] == "shadow" and category_sequence[-1] == "positive"):
        return "recovery", "Recovery arc: movement from shadow mode toward positive integration"

    # Disengagement: positive -> exit
    if (category_sequence[0] == "positive" and category_sequence[-1] == "exit"):
        return "disengagement", "Disengagement pattern: gradual withdrawal from mainstream narratives"

    # Oscillation: back-and-forth between categories
    transitions_between = 0
    for i in range(len(category_sequence) - 1):
        if category_sequence[i] != category_sequence[i+1]:
            transitions_between += 1

    if transitions_between >= len(sequence) * 0.6:
        return "oscillation", "Oscillating pattern: unstable narrative identity with frequent mode shifts"

    # Stabilization: movement toward consistent category
    if len(set(category_sequence[-3:])) == 1 and len(category_sequence) > 3:
        final_category = category_sequence[-1]
        return "stabilization", f"Stabilization pattern: converging toward {final_category} narrative mode"

    # Descent: positive -> ambivalent -> shadow
    if positive_count > 0 and shadow_count > 0:
        first_positive = category_sequence.index("positive") if "positive" in category_sequence else len(category_sequence)
        last_shadow = len(category_sequence) - 1 - category_sequence[::-1].index("shadow") if "shadow" in category_sequence else -1
        if first_positive < last_shadow:
            return "descent", "Descent pattern: erosion of positive narratives toward shadow mode"

    # Ascent: shadow -> ambivalent -> positive
    if shadow_count > 0 and positive_count > 0:
        first_shadow = category_sequence.index("shadow") if "shadow" in category_sequence else len(category_sequence)
        last_positive = len(category_sequence) - 1 - category_sequence[::-1].index("positive") if "positive" in category_sequence else -1
        if first_shadow < last_positive:
            return "ascent", "Ascent pattern: progressive movement toward positive narrative integration"

    # Default
    return "mixed", "Mixed pattern: no clear directional trajectory detected"


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/analytics/outliers", response_model=OutlierDetectionResponse)
async def detect_outliers(request: OutlierDetectionRequest):
    """
    Detect if a text is an anomalous narrative within a corpus.

    Uses z-scores for per-axis analysis and Mahalanobis distance for
    multivariate outlier detection. A text is considered an outlier if
    any axis exceeds the threshold OR if Mahalanobis distance is high.

    The Mahalanobis distance accounts for correlations between axes,
    making it more robust than simple z-score thresholds for detecting
    narratives that are unusual in their combination of values.
    """
    from main import (
        model_manager, embedding_extractor, current_projection,
        ModelType
    )

    if current_projection is None:
        raise HTTPException(status_code=400, detail="No projection trained")

    if not model_manager.is_loaded(request.model_id):
        model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

    try:
        # Embed all corpus texts
        corpus_results = embedding_extractor.extract(
            request.corpus, request.model_id
        )
        corpus_embeddings = np.array([r.embedding for r in corpus_results])

        # Project corpus to 3D
        corpus_vectors = np.array([
            current_projection.project(emb).to_list()
            for emb in corpus_embeddings
        ])

        # Embed and project test text
        test_result = embedding_extractor.extract(
            request.test_text, request.model_id
        )
        test_embedding = test_result.embedding
        test_vector = np.array(
            current_projection.project(test_embedding).to_list()
        )

        # Compute z-scores
        z_scores = compute_z_scores(test_vector, corpus_vectors)

        # Compute Mahalanobis distance
        mahal_dist = compute_mahalanobis_distance(test_vector, corpus_vectors)

        # Find nearest corpus text
        nearest_text = find_nearest_text(
            test_vector, corpus_vectors, request.corpus
        )

        # Determine if outlier
        # Outlier if any z-score exceeds threshold OR Mahalanobis > threshold * sqrt(3)
        # (sqrt(3) scales for 3 dimensions)
        is_outlier = (
            any(abs(z) > request.threshold for z in z_scores.values()) or
            mahal_dist > request.threshold * np.sqrt(3)
        )

        # Generate interpretation
        interpretation = generate_outlier_interpretation(z_scores, request.threshold)

        return OutlierDetectionResponse(
            is_outlier=is_outlier,
            z_scores={k: round(v, 4) for k, v in z_scores.items()},
            mahalanobis_distance=round(mahal_dist, 4),
            nearest_corpus_text=nearest_text,
            interpretation=interpretation
        )

    except Exception as e:
        logger.error(f"Outlier detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics/cohorts", response_model=CohortAnalysisResponse)
async def analyze_cohorts(request: CohortAnalysisRequest):
    """
    Analyze multiple groups (cohorts) with ANOVA and pairwise comparisons.

    Performs one-way ANOVA on each axis to determine if there are significant
    differences between cohorts. Also computes pairwise Euclidean distances
    between cohort centroids to quantify how different each pair of groups is.

    Significance is determined at p < 0.05 level.

    Requirements:
    - At least 2 cohorts
    - Each cohort should have at least 2 texts for meaningful statistics
    """
    from main import (
        model_manager, embedding_extractor, current_projection,
        ModelType
    )

    if current_projection is None:
        raise HTTPException(status_code=400, detail="No projection trained")

    if len(request.cohorts) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 cohorts for comparison"
        )

    if not model_manager.is_loaded(request.model_id):
        model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

    try:
        # Import mode classifier
        try:
            from analysis.mode_classifier import get_mode_classifier
            classifier = get_mode_classifier()
        except ImportError:
            classifier = None

        # Process each cohort
        cohort_data = {}  # cohort_name -> (vectors, modes)

        for cohort_name, texts in request.cohorts.items():
            if len(texts) == 0:
                continue

            # Embed texts
            results = embedding_extractor.extract(texts, request.model_id)
            embeddings = np.array([r.embedding for r in results])

            # Project to 3D
            vectors = np.array([
                current_projection.project(emb).to_list()
                for emb in embeddings
            ])

            # Classify modes
            modes = [classify_mode(v, classifier) for v in vectors]

            cohort_data[cohort_name] = {
                "vectors": vectors,
                "modes": modes
            }

        if len(cohort_data) < 2:
            raise HTTPException(
                status_code=400,
                detail="Need at least 2 non-empty cohorts for comparison"
            )

        # Build cohort profiles
        cohort_profiles = {}

        for cohort_name, data in cohort_data.items():
            vectors = data["vectors"]
            modes = data["modes"]

            # Find dominant mode
            mode_counts = Counter(modes)
            dominant_mode = mode_counts.most_common(1)[0][0]

            # Compute statistics
            mean_vec = vectors.mean(axis=0)
            std_vec = vectors.std(axis=0) if len(vectors) > 1 else np.zeros(3)

            cohort_profiles[cohort_name] = CohortProfile(
                mean_agency=round(float(mean_vec[0]), 4),
                mean_perceived_justice=round(float(mean_vec[1]), 4),
                mean_belonging=round(float(mean_vec[2]), 4),
                dominant_mode=dominant_mode,
                n_texts=len(vectors),
                std_agency=round(float(std_vec[0]), 4),
                std_perceived_justice=round(float(std_vec[1]), 4),
                std_belonging=round(float(std_vec[2]), 4)
            )

        # Perform ANOVA for each axis
        axis_names = ["agency", "perceived_justice", "belonging"]
        anova_results = {}

        for i, axis in enumerate(axis_names):
            # Collect axis values per cohort
            groups = [
                data["vectors"][:, i] for data in cohort_data.values()
            ]

            # Perform one-way ANOVA
            try:
                f_stat, p_value = stats.f_oneway(*groups)

                # Handle NaN values
                if np.isnan(f_stat):
                    f_stat = 0.0
                if np.isnan(p_value):
                    p_value = 1.0

            except Exception:
                f_stat = 0.0
                p_value = 1.0

            anova_results[axis] = ANOVAResult(
                f_stat=round(float(f_stat), 4),
                p_value=round(float(p_value), 6),
                significant=p_value < 0.05
            )

        # Compute pairwise distances between cohort centroids
        pairwise_distances = {}
        cohort_names = list(cohort_data.keys())

        for i, name1 in enumerate(cohort_names):
            centroid1 = cohort_data[name1]["vectors"].mean(axis=0)
            for name2 in cohort_names[i+1:]:
                centroid2 = cohort_data[name2]["vectors"].mean(axis=0)
                dist = compute_euclidean_distance(centroid1, centroid2)
                key = f"{name1}_vs_{name2}"
                pairwise_distances[key] = round(dist, 4)

        return CohortAnalysisResponse(
            cohort_profiles=cohort_profiles,
            anova_results=anova_results,
            pairwise_distances=pairwise_distances
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cohort analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics/mode-flow", response_model=ModeFlowResponse)
async def analyze_mode_flow(request: ModeFlowRequest):
    """
    Analyze mode transitions in a sequence of texts.

    This endpoint is useful for tracking narrative evolution over time,
    such as analyzing how a person's or group's narrative mode changes
    across multiple statements or documents.

    Detects patterns like:
    - radicalization: positive -> shadow/exit progression
    - recovery: shadow -> positive movement
    - disengagement: gradual withdrawal from mainstream narratives
    - oscillation: unstable narrative identity
    - stabilization: convergence to consistent mode

    Transition probabilities are computed based on empirical mode transition
    research and adjusted for the specific sequence observed.
    """
    from main import (
        model_manager, embedding_extractor, current_projection,
        ModelType
    )

    if current_projection is None:
        raise HTTPException(status_code=400, detail="No projection trained")

    if len(request.texts) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 texts to analyze mode flow"
        )

    if not model_manager.is_loaded(request.model_id):
        model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

    try:
        # Import mode classifier
        try:
            from analysis.mode_classifier import get_mode_classifier
            classifier = get_mode_classifier()
        except ImportError:
            classifier = None

        # Embed and project all texts
        results = embedding_extractor.extract(request.texts, request.model_id)
        embeddings = np.array([r.embedding for r in results])
        vectors = np.array([
            current_projection.project(emb).to_list()
            for emb in embeddings
        ])

        # Classify modes
        sequence = [classify_mode(v, classifier) for v in vectors]

        # Build transition matrix
        transition_counts: Dict[str, Dict[str, int]] = {}
        for i in range(len(sequence) - 1):
            from_mode = sequence[i]
            to_mode = sequence[i + 1]

            if from_mode not in transition_counts:
                transition_counts[from_mode] = {}
            if to_mode not in transition_counts[from_mode]:
                transition_counts[from_mode][to_mode] = 0

            transition_counts[from_mode][to_mode] += 1

        # Compute transitions with probabilities
        transitions = []
        for i in range(len(sequence) - 1):
            from_mode = sequence[i]
            to_mode = sequence[i + 1]
            prob = compute_transition_probability(from_mode, to_mode, transition_counts)

            transitions.append({
                "from": from_mode,
                "to": to_mode,
                "probability": round(prob, 4)
            })

        # Classify flow pattern
        flow_pattern, interpretation = classify_flow_pattern(sequence, transitions)

        return ModeFlowResponse(
            sequence=sequence,
            transitions=transitions,
            flow_pattern=flow_pattern,
            interpretation=interpretation
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mode flow analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
