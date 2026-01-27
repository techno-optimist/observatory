"""
Soliton Detector

Finds persistent narrative structures in the projected space.
Uses clustering to identify coherent narrative modes and tracks
their stability across perturbations.

AXIS NAMING (January 2026):
The "fairness" axis has been renamed to "perceived_justice" in API responses.
Internal coordinate order: [agency, fairness(=perceived_justice), belonging]
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class NarrativeCluster:
    """A detected cluster of narratives in the manifold."""
    cluster_id: int
    centroid: np.ndarray  # 3D position [agency, fairness(perceived_justice), belonging]
    size: int
    texts: List[str]
    stability_score: float  # How coherent is this cluster
    exemplar: str  # Most representative text

    def to_dict(self, use_canonical_names: bool = True) -> dict:
        """
        Convert to dictionary.

        Args:
            use_canonical_names: If True (default), uses 'perceived_justice' instead of 'fairness'.
        """
        fairness_key = "perceived_justice" if use_canonical_names else "fairness"
        return {
            "cluster_id": self.cluster_id,
            "centroid": {
                "agency": float(self.centroid[0]),
                fairness_key: float(self.centroid[1]),
                "belonging": float(self.centroid[2])
            },
            "size": self.size,
            "exemplar": self.exemplar,
            "stability_score": self.stability_score,
            "sample_texts": self.texts[:5]  # First 5
        }


@dataclass
class SolitonCandidate:
    """A potential soliton - a stable narrative structure."""
    id: int
    core_narrative: str
    vector: np.ndarray  # [agency, fairness(perceived_justice), belonging]
    cohesion: float  # How tightly clustered
    persistence: float  # Stability under perturbation
    mode: str  # Inferred narrative mode

    def to_dict(self, use_canonical_names: bool = True) -> dict:
        """
        Convert to dictionary.

        Args:
            use_canonical_names: If True (default), uses 'perceived_justice' instead of 'fairness'.
        """
        fairness_key = "perceived_justice" if use_canonical_names else "fairness"
        return {
            "id": self.id,
            "core_narrative": self.core_narrative,
            "vector": {
                "agency": float(self.vector[0]),
                fairness_key: float(self.vector[1]),
                "belonging": float(self.vector[2])
            },
            "cohesion": self.cohesion,
            "persistence": self.persistence,
            "mode": self.mode
        }


class SolitonDetector:
    """
    Detects stable narrative structures (solitons) in the cultural manifold.

    A soliton is a narrative that:
    1. Forms a coherent cluster in embedding space
    2. Remains stable under small perturbations
    3. Has clear membership boundaries
    """

    # Mode definitions for classification
    MODE_CENTROIDS = {
        "DREAM_POSITIVE": np.array([1.0, 1.0, 0.5]),
        "DREAM_SHADOW": np.array([1.0, -1.0, -0.5]),
        "DREAM_EXIT": np.array([0.0, 0.0, 1.0]),
        "NOISE_OTHER": np.array([0.0, 0.0, 0.0])
    }

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        cluster_selection_epsilon: float = 0.3
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon

    def detect_clusters(
        self,
        projected_points: np.ndarray,
        texts: List[str],
        method: str = "hdbscan"
    ) -> List[NarrativeCluster]:
        """
        Detect narrative clusters in the 3D projected space.

        Args:
            projected_points: (N, 3) array of projected coordinates
            texts: Corresponding texts
            method: Clustering method ("hdbscan" or "kmeans")
        """
        if len(projected_points) < self.min_cluster_size:
            logger.warning("Not enough points for clustering")
            return []

        if method == "hdbscan":
            clusters = self._cluster_hdbscan(projected_points)
        elif method == "kmeans":
            clusters = self._cluster_kmeans(projected_points)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Build cluster objects
        result = []
        unique_labels = set(clusters) - {-1}  # Exclude noise

        for label in unique_labels:
            mask = clusters == label
            cluster_points = projected_points[mask]
            cluster_texts = [texts[i] for i in np.where(mask)[0]]

            centroid = cluster_points.mean(axis=0)
            size = len(cluster_texts)

            # Find exemplar (closest to centroid)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            exemplar_idx = distances.argmin()
            exemplar = cluster_texts[exemplar_idx]

            # Compute stability (inverse of spread)
            spread = distances.mean()
            stability = 1.0 / (1.0 + spread)

            result.append(NarrativeCluster(
                cluster_id=int(label),
                centroid=centroid,
                size=size,
                texts=cluster_texts,
                stability_score=stability,
                exemplar=exemplar
            ))

        # Sort by size
        result.sort(key=lambda c: c.size, reverse=True)

        logger.info(f"Detected {len(result)} clusters from {len(texts)} texts")
        return result

    def _cluster_hdbscan(self, points: np.ndarray) -> np.ndarray:
        """Cluster using HDBSCAN (density-based)."""
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon
        )
        return clusterer.fit_predict(points)

    def _cluster_kmeans(self, points: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """Cluster using KMeans."""
        if k is None:
            # Use silhouette score to find optimal k
            k = self._find_optimal_k(points)

        clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
        return clusterer.fit_predict(points)

    def _find_optimal_k(self, points: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using silhouette score."""
        max_k = min(max_k, len(points) - 1)
        if max_k < 2:
            return 2

        best_k = 2
        best_score = -1

        for k in range(2, max_k + 1):
            clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = clusterer.fit_predict(points)
            score = silhouette_score(points, labels)
            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def identify_solitons(
        self,
        clusters: List[NarrativeCluster],
        min_stability: float = 0.5,
        min_size: int = 2
    ) -> List[SolitonCandidate]:
        """
        Identify which clusters qualify as solitons.

        Args:
            clusters: Detected clusters
            min_stability: Minimum stability score
            min_size: Minimum cluster size
        """
        solitons = []

        for i, cluster in enumerate(clusters):
            if cluster.stability_score < min_stability:
                continue
            if cluster.size < min_size:
                continue

            # Classify mode
            mode = self._classify_mode(cluster.centroid)

            soliton = SolitonCandidate(
                id=i + 1,
                core_narrative=cluster.exemplar,
                vector=cluster.centroid,
                cohesion=cluster.stability_score,
                persistence=cluster.stability_score,  # For now, same as cohesion
                mode=mode
            )
            solitons.append(soliton)

        logger.info(f"Identified {len(solitons)} soliton candidates")
        return solitons

    def _classify_mode(self, centroid: np.ndarray) -> str:
        """Classify a point into a narrative mode."""
        min_dist = float('inf')
        best_mode = "NOISE_OTHER"

        for mode, mode_centroid in self.MODE_CENTROIDS.items():
            dist = np.linalg.norm(centroid - mode_centroid)
            if dist < min_dist:
                min_dist = dist
                best_mode = mode

        return best_mode

    def compute_field_gradient(
        self,
        projected_points: np.ndarray,
        grid_resolution: int = 20
    ) -> Dict:
        """
        Compute the gradient field in the projected space.

        Returns a grid showing the "flow" of narratives.
        """
        # Create grid
        x_range = np.linspace(-2.5, 2.5, grid_resolution)
        y_range = np.linspace(-2.5, 2.5, grid_resolution)

        density = np.zeros((grid_resolution, grid_resolution))
        gradient_x = np.zeros((grid_resolution, grid_resolution))
        gradient_y = np.zeros((grid_resolution, grid_resolution))

        # Compute density at each grid point
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                point = np.array([x, y, 0])  # Project to 2D for visualization
                distances = np.linalg.norm(
                    projected_points[:, :2] - point[:2],
                    axis=1
                )
                # Gaussian kernel
                weights = np.exp(-distances**2 / 0.5)
                density[j, i] = weights.sum()

        # Compute gradient
        gradient_y, gradient_x = np.gradient(density)

        return {
            "x_range": x_range.tolist(),
            "y_range": y_range.tolist(),
            "density": density.tolist(),
            "gradient_x": gradient_x.tolist(),
            "gradient_y": gradient_y.tolist()
        }


def detect_narrative_clusters(
    embeddings: np.ndarray,
    projection,
    texts: List[str]
) -> Tuple[List[NarrativeCluster], List[SolitonCandidate]]:
    """
    Convenience function to detect clusters and solitons.

    Args:
        embeddings: High-dimensional embeddings
        projection: Trained ProjectionHead
        texts: Corresponding texts

    Returns:
        (clusters, solitons)
    """
    # Project to 3D
    projected = np.array([
        projection.project(emb).to_list()
        for emb in embeddings
    ])

    # Detect clusters
    detector = SolitonDetector()
    clusters = detector.detect_clusters(projected, texts)
    solitons = detector.identify_solitons(clusters)

    return clusters, solitons
