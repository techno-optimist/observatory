"""
Calibration Baseline System for the Cultural Soliton Observatory

This module compares human language against minimal AI-style coordination codes
to find the irreducible coordination kernel - the "meaning with the adjectives
burned off."

CONCEPTUAL FRAMEWORK:
--------------------
Human language is dense with cultural accretion: metaphors, emotional coloring,
hedging, politeness markers, grammatical complexity. But beneath this ornamentation
lies a coordination substrate - the minimal structure needed for two minds to
achieve shared understanding and aligned action.

AI systems trained on minimal reward signals develop stripped-down protocols:
"execute_task(param1, param2)" rather than "Could you perhaps consider, if it's
not too much trouble, carrying out this task with these parameters?"

By projecting both through the Observatory's manifold and finding where they
overlap, we identify:

1. COORDINATION CORE: The invariant structure present in both verbose human
   language AND minimal codes. This is the bedrock - remove it and coordination
   fails entirely.

2. EFFICIENCY LAYER: Features present in minimal codes that improve performance
   but aren't strictly necessary. The optimization surface.

3. DECORATIVE LAYER: Features only in human language with no coordination
   function - cultural ornament, aesthetic elaboration, social signaling.

4. NOISE: Random variation in either corpus that serves no purpose.

METHODOLOGY:
-----------
1. Load human corpus (natural language with full complexity)
2. Load minimal corpus (stripped versions OR synthetic minimal codes)
3. Project both through Observatory to get manifold positions
4. Find overlap regions (coordination core)
5. Identify divergence regions (decorative vs efficient)
6. Measure drift when features are added/removed

The key insight: if removing a feature from human text moves it TOWARD the
minimal corpus centroid, that feature was decorative. If it moves it AWAY,
the feature was coordination-necessary.

Usage:
    python -m research.calibration_baseline --human human_texts.txt --minimal minimal_texts.txt
    python -m research.calibration_baseline --interactive

API Integration:
    Uses httpx for async HTTP calls to http://127.0.0.1:8000
"""

import asyncio
import logging
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import httpx

logger = logging.getLogger(__name__)

# Observatory API base URL
OBSERVATORY_API = "http://127.0.0.1:8000"


class FeatureClassification(str, Enum):
    """Classification of linguistic features by coordination function."""
    NECESSARY = "necessary"    # Present in both, high drift when removed from human
    EFFICIENT = "efficient"    # Present in minimal, improves coordination performance
    DECORATIVE = "decorative"  # Only in human, no coordination function
    NOISE = "noise"            # Random variation with no consistent pattern


@dataclass
class ProjectionResult:
    """Result of projecting a text through the Observatory."""
    text: str
    coordinates: Dict[str, float]  # agency, perceived_justice, belonging
    mode: str
    uncertainty: Optional[Dict[str, float]] = None
    raw_response: Optional[Dict] = None

    @property
    def vector(self) -> np.ndarray:
        """Get coordinates as numpy array."""
        return np.array([
            self.coordinates.get("agency", 0.0),
            self.coordinates.get("perceived_justice", 0.0),
            self.coordinates.get("belonging", 0.0)
        ])


@dataclass
class CorpusStatistics:
    """Statistical summary of a projected corpus."""
    centroid: np.ndarray
    std_dev: np.ndarray
    mode_distribution: Dict[str, float]
    per_axis_range: Dict[str, Tuple[float, float]]
    sample_count: int
    projections: List[ProjectionResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "centroid": {
                "agency": float(self.centroid[0]),
                "perceived_justice": float(self.centroid[1]),
                "belonging": float(self.centroid[2])
            },
            "std_dev": {
                "agency": float(self.std_dev[0]),
                "perceived_justice": float(self.std_dev[1]),
                "belonging": float(self.std_dev[2])
            },
            "mode_distribution": self.mode_distribution,
            "per_axis_range": self.per_axis_range,
            "sample_count": self.sample_count
        }


@dataclass
class FeatureAnalysis:
    """Analysis of a single linguistic feature."""
    feature_name: str
    classification: FeatureClassification
    mean_drift_toward_minimal: float  # Positive = removing moves toward minimal
    mean_drift_away_from_minimal: float  # Positive = removing moves away
    presence_in_human: float  # Fraction of human texts containing feature
    presence_in_minimal: float  # Fraction of minimal texts containing feature
    coordination_importance: float  # Composite score 0-1
    axis_effects: Dict[str, float]  # Per-axis drift when removed
    confidence: float  # Confidence in classification

    def to_dict(self) -> Dict:
        return {
            "feature_name": self.feature_name,
            "classification": self.classification.value,
            "mean_drift_toward_minimal": self.mean_drift_toward_minimal,
            "mean_drift_away_from_minimal": self.mean_drift_away_from_minimal,
            "presence_in_human": self.presence_in_human,
            "presence_in_minimal": self.presence_in_minimal,
            "coordination_importance": self.coordination_importance,
            "axis_effects": self.axis_effects,
            "confidence": self.confidence
        }


@dataclass
class CalibrationResult:
    """Complete calibration analysis result."""
    human_stats: CorpusStatistics
    minimal_stats: CorpusStatistics
    overlap_region: Dict[str, Tuple[float, float]]  # Per-axis overlap ranges
    centroid_distance: float
    mode_overlap: float  # Jaccard similarity of mode distributions
    per_axis_correlation: Dict[str, float]
    feature_rankings: List[FeatureAnalysis]
    coordination_core_description: str

    def to_dict(self) -> Dict:
        return {
            "human_stats": self.human_stats.to_dict(),
            "minimal_stats": self.minimal_stats.to_dict(),
            "overlap_region": self.overlap_region,
            "centroid_distance": self.centroid_distance,
            "mode_overlap": self.mode_overlap,
            "per_axis_correlation": self.per_axis_correlation,
            "feature_rankings": [f.to_dict() for f in self.feature_rankings],
            "coordination_core_description": self.coordination_core_description
        }


class CalibrationBaseline:
    """
    Compares human language against minimal AI coordination codes.

    This class implements the core calibration methodology: project both
    corpora through the Observatory, find the overlap (coordination core),
    and classify features by their coordination necessity.

    The goal is to find "meaning with the adjectives burned off" - the
    irreducible substrate that enables coordination between minds.
    """

    def __init__(
        self,
        api_base: str = OBSERVATORY_API,
        drift_threshold: float = 0.2,
        timeout: float = 30.0
    ):
        """
        Initialize the calibration baseline system.

        Args:
            api_base: Base URL for Observatory API
            drift_threshold: Threshold for considering drift "significant"
            timeout: HTTP request timeout in seconds
        """
        self.api_base = api_base
        self.drift_threshold = drift_threshold
        self.timeout = timeout

        self.human_corpus: List[str] = []
        self.minimal_corpus: List[str] = []
        self.human_projections: List[ProjectionResult] = []
        self.minimal_projections: List[ProjectionResult] = []

        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                timeout=self.timeout
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def load_human_corpus(self, texts: List[str]) -> int:
        """
        Load full human language samples for calibration.

        These should be natural language texts with full grammatical
        complexity, emotional coloring, and cultural markers.

        Args:
            texts: List of human-written text samples

        Returns:
            Number of texts loaded
        """
        self.human_corpus = [t.strip() for t in texts if t.strip()]
        self.human_projections = []  # Clear cached projections
        logger.info(f"Loaded {len(self.human_corpus)} human corpus texts")
        return len(self.human_corpus)

    def load_minimal_corpus(self, texts: List[str]) -> int:
        """
        Load stripped/minimal coordination codes for calibration.

        These should be minimal-structure texts that still achieve
        coordination: commands, protocols, telegraphic messages,
        or AI-generated minimal codes.

        Args:
            texts: List of minimal text samples

        Returns:
            Number of texts loaded
        """
        self.minimal_corpus = [t.strip() for t in texts if t.strip()]
        self.minimal_projections = []  # Clear cached projections
        logger.info(f"Loaded {len(self.minimal_corpus)} minimal corpus texts")
        return len(self.minimal_corpus)

    async def _project_text(self, text: str) -> ProjectionResult:
        """
        Project a single text through the Observatory.

        Args:
            text: Text to project

        Returns:
            ProjectionResult with coordinates and mode
        """
        client = await self._get_client()

        try:
            response = await client.post(
                "/v2/analyze",
                json={
                    "text": text,
                    "include_uncertainty": True,
                    "include_legacy_mode": True,
                    "include_force_field": False
                }
            )
            response.raise_for_status()
            data = response.json()

            coordinates = data.get("vector", {})
            mode = data.get("mode", {}).get("primary_mode", "NEUTRAL")
            uncertainty = data.get("uncertainty", {})

            return ProjectionResult(
                text=text,
                coordinates=coordinates,
                mode=mode,
                uncertainty=uncertainty,
                raw_response=data
            )

        except httpx.HTTPError as e:
            logger.error(f"HTTP error projecting text: {e}")
            raise
        except Exception as e:
            logger.error(f"Error projecting text: {e}")
            raise

    async def _project_corpus(
        self,
        texts: List[str],
        batch_size: int = 10
    ) -> List[ProjectionResult]:
        """
        Project a corpus through the Observatory.

        Args:
            texts: List of texts to project
            batch_size: Number of concurrent requests

        Returns:
            List of ProjectionResults
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self._project_text(t) for t in batch],
                return_exceptions=True
            )

            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to project text {i+j}: {result}")
                else:
                    results.append(result)

        return results

    def _compute_statistics(
        self,
        projections: List[ProjectionResult]
    ) -> CorpusStatistics:
        """
        Compute statistical summary of projected corpus.

        Args:
            projections: List of projection results

        Returns:
            CorpusStatistics with centroid, variance, mode distribution
        """
        if not projections:
            return CorpusStatistics(
                centroid=np.zeros(3),
                std_dev=np.zeros(3),
                mode_distribution={},
                per_axis_range={},
                sample_count=0,
                projections=[]
            )

        # Extract vectors
        vectors = np.array([p.vector for p in projections])

        # Compute centroid and std
        centroid = np.mean(vectors, axis=0)
        std_dev = np.std(vectors, axis=0)

        # Mode distribution
        mode_counts: Dict[str, int] = {}
        for p in projections:
            mode_counts[p.mode] = mode_counts.get(p.mode, 0) + 1
        total = len(projections)
        mode_distribution = {m: c / total for m, c in mode_counts.items()}

        # Per-axis ranges
        axes = ["agency", "perceived_justice", "belonging"]
        per_axis_range = {}
        for i, axis in enumerate(axes):
            per_axis_range[axis] = (
                float(np.min(vectors[:, i])),
                float(np.max(vectors[:, i]))
            )

        return CorpusStatistics(
            centroid=centroid,
            std_dev=std_dev,
            mode_distribution=mode_distribution,
            per_axis_range=per_axis_range,
            sample_count=len(projections),
            projections=projections
        )

    async def compute_coordination_core(self) -> Dict:
        """
        Find invariant structure present in both corpora.

        This is the heart of the calibration: project both corpora,
        find where they overlap in the manifold, and identify what
        features are present in that overlap region.

        The coordination core is the "meaning with adjectives burned off" -
        the irreducible structure that both verbose human language and
        minimal codes share.

        Returns:
            Dict with coordination core analysis
        """
        if not self.human_corpus:
            raise ValueError("Human corpus not loaded. Call load_human_corpus() first.")
        if not self.minimal_corpus:
            raise ValueError("Minimal corpus not loaded. Call load_minimal_corpus() first.")

        # Project both corpora
        logger.info("Projecting human corpus...")
        self.human_projections = await self._project_corpus(self.human_corpus)
        logger.info("Projecting minimal corpus...")
        self.minimal_projections = await self._project_corpus(self.minimal_corpus)

        # Compute statistics
        human_stats = self._compute_statistics(self.human_projections)
        minimal_stats = self._compute_statistics(self.minimal_projections)

        # Find overlap region
        axes = ["agency", "perceived_justice", "belonging"]
        overlap_region = {}
        for axis in axes:
            h_range = human_stats.per_axis_range.get(axis, (0, 0))
            m_range = minimal_stats.per_axis_range.get(axis, (0, 0))

            overlap_min = max(h_range[0], m_range[0])
            overlap_max = min(h_range[1], m_range[1])

            if overlap_max > overlap_min:
                overlap_region[axis] = (overlap_min, overlap_max)
            else:
                overlap_region[axis] = (0, 0)  # No overlap

        # Compute centroid distance
        centroid_distance = float(np.linalg.norm(
            human_stats.centroid - minimal_stats.centroid
        ))

        # Mode overlap (Jaccard similarity)
        human_modes = set(human_stats.mode_distribution.keys())
        minimal_modes = set(minimal_stats.mode_distribution.keys())
        if human_modes or minimal_modes:
            mode_overlap = len(human_modes & minimal_modes) / len(human_modes | minimal_modes)
        else:
            mode_overlap = 0.0

        # Per-axis correlation
        per_axis_correlation = {}
        if len(self.human_projections) > 1 and len(self.minimal_projections) > 1:
            human_vectors = np.array([p.vector for p in self.human_projections])
            minimal_vectors = np.array([p.vector for p in self.minimal_projections])

            for i, axis in enumerate(axes):
                # Compare distributions using overlap coefficient
                h_mean, h_std = human_vectors[:, i].mean(), human_vectors[:, i].std()
                m_mean, m_std = minimal_vectors[:, i].mean(), minimal_vectors[:, i].std()

                # Distribution similarity (1 - normalized difference)
                if h_std + m_std > 0:
                    similarity = 1 - abs(h_mean - m_mean) / (h_std + m_std + 0.01)
                    per_axis_correlation[axis] = max(0, min(1, similarity))
                else:
                    per_axis_correlation[axis] = 1.0 if abs(h_mean - m_mean) < 0.1 else 0.0

        # Identify texts in overlap region
        overlap_texts = {
            "human": [],
            "minimal": []
        }

        for p in self.human_projections:
            in_overlap = all(
                overlap_region[axis][0] <= p.coordinates.get(axis, 0) <= overlap_region[axis][1]
                for axis in axes
            )
            if in_overlap:
                overlap_texts["human"].append(p.text[:100])

        for p in self.minimal_projections:
            in_overlap = all(
                overlap_region[axis][0] <= p.coordinates.get(axis, 0) <= overlap_region[axis][1]
                for axis in axes
            )
            if in_overlap:
                overlap_texts["minimal"].append(p.text[:100])

        return {
            "human_stats": human_stats.to_dict(),
            "minimal_stats": minimal_stats.to_dict(),
            "overlap_region": overlap_region,
            "centroid_distance": centroid_distance,
            "mode_overlap": mode_overlap,
            "per_axis_correlation": per_axis_correlation,
            "overlap_sample_count": {
                "human": len(overlap_texts["human"]),
                "minimal": len(overlap_texts["minimal"])
            },
            "overlap_examples": overlap_texts,
            "interpretation": self._interpret_core(
                centroid_distance, mode_overlap, per_axis_correlation, overlap_region
            )
        }

    def _interpret_core(
        self,
        centroid_distance: float,
        mode_overlap: float,
        per_axis_correlation: Dict[str, float],
        overlap_region: Dict[str, Tuple[float, float]]
    ) -> str:
        """Generate human-readable interpretation of coordination core."""
        interpretations = []

        # Centroid distance interpretation
        if centroid_distance < 0.3:
            interpretations.append(
                "Human and minimal corpora occupy very similar manifold positions, "
                "suggesting most human language IS coordination-relevant."
            )
        elif centroid_distance < 0.7:
            interpretations.append(
                "Moderate separation between corpora centroids indicates "
                "significant decorative content in human language."
            )
        else:
            interpretations.append(
                "Large centroid separation suggests human language carries "
                "substantial non-coordination content (cultural/aesthetic)."
            )

        # Mode overlap interpretation
        if mode_overlap > 0.7:
            interpretations.append(
                "High mode overlap: both corpora trigger similar narrative modes, "
                "indicating shared deep structure."
            )
        elif mode_overlap > 0.3:
            interpretations.append(
                "Partial mode overlap: some narrative modes are shared, others unique to human expression."
            )
        else:
            interpretations.append(
                "Low mode overlap: human language accesses narrative modes "
                "that minimal codes don't trigger - possibly emotional/cultural."
            )

        # Axis interpretation
        strongest_axis = max(per_axis_correlation.items(), key=lambda x: x[1])
        weakest_axis = min(per_axis_correlation.items(), key=lambda x: x[1])

        interpretations.append(
            f"Strongest alignment on '{strongest_axis[0]}' axis ({strongest_axis[1]:.2f}), "
            f"weakest on '{weakest_axis[0]}' ({weakest_axis[1]:.2f}). "
            f"This suggests {weakest_axis[0]} carries more decorative elaboration."
        )

        return " ".join(interpretations)

    def classify_feature(
        self,
        feature_name: str,
        human_with_feature: List[ProjectionResult],
        human_without_feature: List[ProjectionResult],
        minimal_with_feature: List[ProjectionResult],
        minimal_without_feature: List[ProjectionResult]
    ) -> FeatureClassification:
        """
        Classify a linguistic feature by its coordination function.

        Classification criteria:
        - NECESSARY: Present in minimal, causes high drift when removed from human
        - EFFICIENT: Present in minimal, improves coordination performance
        - DECORATIVE: Only in human, no coordination function
        - NOISE: Random variation with no consistent pattern

        Args:
            feature_name: Name of the feature being classified
            human_with_feature: Human texts containing the feature
            human_without_feature: Human texts after feature removal
            minimal_with_feature: Minimal texts containing the feature
            minimal_without_feature: Minimal texts after feature removal

        Returns:
            FeatureClassification enum value
        """
        # Get minimal corpus centroid as reference
        if self.minimal_projections:
            minimal_centroid = self._compute_statistics(self.minimal_projections).centroid
        else:
            minimal_centroid = np.zeros(3)

        # Compute drift when feature is removed from human texts
        human_drifts = []
        for with_f, without_f in zip(human_with_feature, human_without_feature):
            drift = np.linalg.norm(with_f.vector - without_f.vector)
            # Also check if removal moves TOWARD minimal centroid
            dist_before = np.linalg.norm(with_f.vector - minimal_centroid)
            dist_after = np.linalg.norm(without_f.vector - minimal_centroid)
            toward_minimal = dist_before - dist_after  # Positive = moved toward
            human_drifts.append((drift, toward_minimal))

        # Feature presence
        presence_in_minimal = len(minimal_with_feature) / max(1, len(self.minimal_projections))
        presence_in_human = len(human_with_feature) / max(1, len(self.human_projections))

        if not human_drifts:
            return FeatureClassification.NOISE

        avg_drift = np.mean([d[0] for d in human_drifts])
        avg_toward_minimal = np.mean([d[1] for d in human_drifts])

        # Classification logic
        if presence_in_minimal > 0.3:
            # Feature is present in minimal codes
            if avg_drift > self.drift_threshold:
                # Removing it causes significant drift
                return FeatureClassification.NECESSARY
            else:
                # Present but low impact - efficiency optimization
                return FeatureClassification.EFFICIENT
        else:
            # Feature is NOT in minimal codes (human-only)
            if avg_toward_minimal > self.drift_threshold * 0.5:
                # Removing it moves TOWARD minimal - it was decorative
                return FeatureClassification.DECORATIVE
            elif avg_drift < self.drift_threshold * 0.3:
                # Low drift, no direction - noise
                return FeatureClassification.NOISE
            else:
                # High drift but not toward minimal - possibly necessary for human context
                return FeatureClassification.NECESSARY

    async def compare_projections(
        self,
        human_texts: List[str],
        minimal_texts: List[str]
    ) -> Dict:
        """
        Side-by-side comparison of human vs minimal text projections.

        Provides detailed comparison of how the same meaning is expressed
        differently and where those expressions land in the manifold.

        Args:
            human_texts: List of verbose human texts
            minimal_texts: List of corresponding minimal versions

        Returns:
            Dict with comparison analysis
        """
        if len(human_texts) != len(minimal_texts):
            logger.warning(
                f"Unequal text counts: {len(human_texts)} human, {len(minimal_texts)} minimal. "
                "Using minimum length."
            )
            min_len = min(len(human_texts), len(minimal_texts))
            human_texts = human_texts[:min_len]
            minimal_texts = minimal_texts[:min_len]

        # Project both sets
        human_projections = await self._project_corpus(human_texts)
        minimal_projections = await self._project_corpus(minimal_texts)

        comparisons = []
        axes = ["agency", "perceived_justice", "belonging"]

        for hp, mp in zip(human_projections, minimal_projections):
            drift = float(np.linalg.norm(hp.vector - mp.vector))
            axis_diffs = {
                axis: hp.coordinates.get(axis, 0) - mp.coordinates.get(axis, 0)
                for axis in axes
            }

            comparisons.append({
                "human_text": hp.text[:200],
                "minimal_text": mp.text[:200],
                "human_coordinates": hp.coordinates,
                "minimal_coordinates": mp.coordinates,
                "human_mode": hp.mode,
                "minimal_mode": mp.mode,
                "drift": drift,
                "axis_differences": axis_diffs,
                "mode_preserved": hp.mode == mp.mode
            })

        # Aggregate statistics
        all_drifts = [c["drift"] for c in comparisons]
        mode_preservation_rate = sum(1 for c in comparisons if c["mode_preserved"]) / max(1, len(comparisons))

        per_axis_drift = {axis: [] for axis in axes}
        for c in comparisons:
            for axis in axes:
                per_axis_drift[axis].append(abs(c["axis_differences"].get(axis, 0)))

        return {
            "pair_comparisons": comparisons,
            "summary": {
                "total_pairs": len(comparisons),
                "mean_drift": float(np.mean(all_drifts)) if all_drifts else 0.0,
                "max_drift": float(np.max(all_drifts)) if all_drifts else 0.0,
                "min_drift": float(np.min(all_drifts)) if all_drifts else 0.0,
                "mode_preservation_rate": mode_preservation_rate,
                "per_axis_mean_drift": {
                    axis: float(np.mean(drifts)) if drifts else 0.0
                    for axis, drifts in per_axis_drift.items()
                }
            },
            "interpretation": self._interpret_comparison(
                np.mean(all_drifts) if all_drifts else 0.0,
                mode_preservation_rate,
                per_axis_drift
            )
        }

    def _interpret_comparison(
        self,
        mean_drift: float,
        mode_preservation_rate: float,
        per_axis_drift: Dict[str, List[float]]
    ) -> str:
        """Generate human-readable comparison interpretation."""
        interpretations = []

        if mean_drift < 0.2:
            interpretations.append(
                "Minimal versions capture nearly all coordination content of human texts. "
                "The decorative layer is thin."
            )
        elif mean_drift < 0.5:
            interpretations.append(
                "Moderate drift between versions. Human elaboration adds significant "
                "but not dominant content to the coordination signal."
            )
        else:
            interpretations.append(
                "Large drift indicates human texts carry substantial content "
                "beyond what minimal versions capture - either decorative OR "
                "coordination-relevant nuance lost in stripping."
            )

        if mode_preservation_rate > 0.8:
            interpretations.append(
                "High mode preservation suggests the narrative 'type' is robust to surface elaboration."
            )
        elif mode_preservation_rate < 0.5:
            interpretations.append(
                "Low mode preservation - stripping changes the perceived narrative type, "
                "suggesting some 'decorative' features actually carry mode-relevant information."
            )

        # Find most affected axis
        axis_means = {
            axis: np.mean(drifts) if drifts else 0.0
            for axis, drifts in per_axis_drift.items()
        }
        most_affected = max(axis_means.items(), key=lambda x: x[1])
        interpretations.append(
            f"The '{most_affected[0]}' axis shows highest drift ({most_affected[1]:.3f}), "
            f"indicating human elaboration most affects {most_affected[0]} perception."
        )

        return " ".join(interpretations)

    async def generate_feature_ranking(
        self,
        feature_tests: Dict[str, Tuple[List[str], List[str]]]
    ) -> List[FeatureAnalysis]:
        """
        Generate ranked list of features by coordination importance.

        Args:
            feature_tests: Dict mapping feature_name to (texts_with_feature, texts_without_feature)

        Returns:
            List of FeatureAnalysis objects, ranked by importance
        """
        analyses = []

        # Get minimal centroid
        if self.minimal_projections:
            minimal_centroid = self._compute_statistics(self.minimal_projections).centroid
        else:
            minimal_centroid = np.zeros(3)

        for feature_name, (with_feature, without_feature) in feature_tests.items():
            # Project both versions
            with_projections = await self._project_corpus(with_feature)
            without_projections = await self._project_corpus(without_feature)

            if not with_projections or not without_projections:
                continue

            # Compute metrics
            drifts_toward = []
            drifts_away = []
            axis_effects = {"agency": [], "perceived_justice": [], "belonging": []}

            for wp, wop in zip(with_projections, without_projections):
                drift = np.linalg.norm(wp.vector - wop.vector)
                dist_before = np.linalg.norm(wp.vector - minimal_centroid)
                dist_after = np.linalg.norm(wop.vector - minimal_centroid)

                if dist_after < dist_before:
                    drifts_toward.append(drift)
                else:
                    drifts_away.append(drift)

                for i, axis in enumerate(["agency", "perceived_justice", "belonging"]):
                    axis_effects[axis].append(wop.vector[i] - wp.vector[i])

            # Calculate presence rates
            presence_human = len(with_projections) / max(1, len(self.human_projections))
            presence_minimal = 0.0  # Would need separate tracking

            # Classify
            mean_toward = np.mean(drifts_toward) if drifts_toward else 0.0
            mean_away = np.mean(drifts_away) if drifts_away else 0.0

            if mean_away > self.drift_threshold:
                classification = FeatureClassification.NECESSARY
                importance = min(1.0, mean_away / 0.5)
            elif mean_toward > self.drift_threshold:
                classification = FeatureClassification.DECORATIVE
                importance = 1 - min(1.0, mean_toward / 0.5)
            elif mean_toward + mean_away < self.drift_threshold * 0.3:
                classification = FeatureClassification.NOISE
                importance = 0.1
            else:
                classification = FeatureClassification.EFFICIENT
                importance = 0.5

            # Confidence based on consistency
            total_drifts = len(drifts_toward) + len(drifts_away)
            if total_drifts > 0:
                consistency = max(len(drifts_toward), len(drifts_away)) / total_drifts
            else:
                consistency = 0.5

            analyses.append(FeatureAnalysis(
                feature_name=feature_name,
                classification=classification,
                mean_drift_toward_minimal=mean_toward,
                mean_drift_away_from_minimal=mean_away,
                presence_in_human=presence_human,
                presence_in_minimal=presence_minimal,
                coordination_importance=importance,
                axis_effects={
                    axis: float(np.mean(effects)) if effects else 0.0
                    for axis, effects in axis_effects.items()
                },
                confidence=consistency
            ))

        # Sort by importance (necessary first, then efficient, then decorative, then noise)
        priority = {
            FeatureClassification.NECESSARY: 0,
            FeatureClassification.EFFICIENT: 1,
            FeatureClassification.DECORATIVE: 2,
            FeatureClassification.NOISE: 3
        }
        analyses.sort(key=lambda a: (priority[a.classification], -a.coordination_importance))

        return analyses

    async def full_calibration(self) -> CalibrationResult:
        """
        Run complete calibration analysis.

        Returns:
            CalibrationResult with full analysis
        """
        # Compute coordination core
        core_analysis = await self.compute_coordination_core()

        # Generate description
        description = self._generate_core_description(core_analysis)

        return CalibrationResult(
            human_stats=self._compute_statistics(self.human_projections),
            minimal_stats=self._compute_statistics(self.minimal_projections),
            overlap_region=core_analysis["overlap_region"],
            centroid_distance=core_analysis["centroid_distance"],
            mode_overlap=core_analysis["mode_overlap"],
            per_axis_correlation=core_analysis["per_axis_correlation"],
            feature_rankings=[],  # Populated by generate_feature_ranking
            coordination_core_description=description
        )

    def _generate_core_description(self, core_analysis: Dict) -> str:
        """Generate prose description of coordination core."""
        lines = [
            "COORDINATION CORE ANALYSIS",
            "=" * 40,
            "",
            f"Centroid Distance: {core_analysis['centroid_distance']:.3f}",
            f"Mode Overlap: {core_analysis['mode_overlap']:.1%}",
            "",
            "Per-Axis Correlation:",
        ]

        for axis, corr in core_analysis["per_axis_correlation"].items():
            lines.append(f"  {axis}: {corr:.3f}")

        lines.extend([
            "",
            "Overlap Region:",
        ])

        for axis, (low, high) in core_analysis["overlap_region"].items():
            if high > low:
                lines.append(f"  {axis}: [{low:.3f}, {high:.3f}]")
            else:
                lines.append(f"  {axis}: NO OVERLAP")

        lines.extend([
            "",
            "INTERPRETATION:",
            core_analysis["interpretation"]
        ])

        return "\n".join(lines)


# --- CLI Interface ---

async def main():
    """CLI entry point for calibration baseline analysis."""
    parser = argparse.ArgumentParser(
        description="Calibration Baseline - Compare human vs minimal language"
    )
    parser.add_argument(
        "--human",
        type=str,
        help="Path to file containing human texts (one per line)"
    )
    parser.add_argument(
        "--minimal",
        type=str,
        help="Path to file containing minimal texts (one per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save JSON results"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Drift threshold for classification (default: 0.2)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    args = parser.parse_args()

    calibrator = CalibrationBaseline(drift_threshold=args.threshold)

    try:
        if args.interactive:
            await interactive_mode(calibrator)
        elif args.human and args.minimal:
            # Load corpora
            with open(args.human, 'r') as f:
                human_texts = [line.strip() for line in f if line.strip()]
            with open(args.minimal, 'r') as f:
                minimal_texts = [line.strip() for line in f if line.strip()]

            calibrator.load_human_corpus(human_texts)
            calibrator.load_minimal_corpus(minimal_texts)

            # Run calibration
            print("\nRunning calibration analysis...")
            result = await calibrator.full_calibration()

            # Display results
            print("\n" + result.coordination_core_description)

            # Save if output specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
                print(f"\nResults saved to {args.output}")

        else:
            parser.print_help()

    finally:
        await calibrator.close()


async def interactive_mode(calibrator: CalibrationBaseline):
    """Interactive mode for exploring calibration."""
    print("\n=== Calibration Baseline Interactive Mode ===")
    print("Enter human texts, then minimal texts, then analyze.\n")

    human_texts = []
    minimal_texts = []

    while True:
        print("\nCommands:")
        print("  h <text>  - Add human text")
        print("  m <text>  - Add minimal text")
        print("  hf <file> - Load human texts from file")
        print("  mf <file> - Load minimal texts from file")
        print("  analyze   - Run calibration analysis")
        print("  compare   - Compare paired texts")
        print("  status    - Show loaded counts")
        print("  clear     - Clear all texts")
        print("  quit      - Exit")

        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        if cmd.startswith("h "):
            human_texts.append(cmd[2:].strip())
            print(f"Added human text ({len(human_texts)} total)")

        elif cmd.startswith("m "):
            minimal_texts.append(cmd[2:].strip())
            print(f"Added minimal text ({len(minimal_texts)} total)")

        elif cmd.startswith("hf "):
            path = cmd[3:].strip()
            try:
                with open(path, 'r') as f:
                    texts = [line.strip() for line in f if line.strip()]
                human_texts.extend(texts)
                print(f"Loaded {len(texts)} human texts ({len(human_texts)} total)")
            except Exception as e:
                print(f"Error loading file: {e}")

        elif cmd.startswith("mf "):
            path = cmd[3:].strip()
            try:
                with open(path, 'r') as f:
                    texts = [line.strip() for line in f if line.strip()]
                minimal_texts.extend(texts)
                print(f"Loaded {len(texts)} minimal texts ({len(minimal_texts)} total)")
            except Exception as e:
                print(f"Error loading file: {e}")

        elif cmd == "analyze":
            if not human_texts or not minimal_texts:
                print("Need both human and minimal texts to analyze")
                continue

            calibrator.load_human_corpus(human_texts)
            calibrator.load_minimal_corpus(minimal_texts)

            print("\nRunning calibration...")
            try:
                result = await calibrator.full_calibration()
                print("\n" + result.coordination_core_description)
            except Exception as e:
                print(f"Analysis failed: {e}")

        elif cmd == "compare":
            if not human_texts or not minimal_texts:
                print("Need both human and minimal texts to compare")
                continue

            print("\nComparing projections...")
            try:
                result = await calibrator.compare_projections(human_texts, minimal_texts)
                print(f"\nSummary:")
                print(f"  Pairs compared: {result['summary']['total_pairs']}")
                print(f"  Mean drift: {result['summary']['mean_drift']:.3f}")
                print(f"  Mode preservation: {result['summary']['mode_preservation_rate']:.1%}")
                print(f"\nInterpretation:")
                print(f"  {result['interpretation']}")
            except Exception as e:
                print(f"Comparison failed: {e}")

        elif cmd == "status":
            print(f"Human texts: {len(human_texts)}")
            print(f"Minimal texts: {len(minimal_texts)}")

        elif cmd == "clear":
            human_texts = []
            minimal_texts = []
            print("Cleared all texts")

        elif cmd == "quit":
            break

        else:
            print("Unknown command. Type a command from the list above.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
