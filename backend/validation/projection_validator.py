"""
Projection Validator

Compares different projection methods (Ridge, PCA, UMAP, t-SNE)
and validates them against human annotations.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Metrics for validating a projection against ground truth."""
    # Regression metrics
    mse: float
    rmse: float
    r2: float

    # Per-axis R2
    r2_agency: float
    r2_fairness: float
    r2_belonging: float

    # Correlation metrics
    spearman_rho: float
    pearson_r: float

    # Cross-validation
    cv_mean: float
    cv_std: float

    # Semantic structure preservation
    cosine_sim_preservation: float  # How well cosine sims are preserved

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProjectionComparison:
    """Comparison of multiple projection methods."""
    method_metrics: Dict[str, ValidationMetrics]
    best_method: str
    ranking: List[str]  # Ordered by performance
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "method_metrics": {k: v.to_dict() for k, v in self.method_metrics.items()},
            "best_method": self.best_method,
            "ranking": self.ranking,
            "recommendation": self.recommendation
        }


class ProjectionValidator:
    """
    Validates and compares projection methods.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.methods = {
            "ridge": self._fit_ridge,
            "pca": self._fit_pca,
            "tsne": self._fit_tsne,
        }
        if UMAP_AVAILABLE:
            self.methods["umap"] = self._fit_umap

    def validate_projection(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        method: str = "ridge",
        n_cv_folds: int = 5
    ) -> ValidationMetrics:
        """
        Validate a single projection method.

        Args:
            embeddings: (N, D) array of embeddings
            targets: (N, 3) array of target coordinates
            method: Projection method name
            n_cv_folds: Number of cross-validation folds
        """
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.methods.keys())}")

        n_samples = len(embeddings)
        if n_samples < n_cv_folds:
            n_cv_folds = max(2, n_samples)

        # Fit and predict
        fit_fn = self.methods[method]
        predictions, model = fit_fn(embeddings, targets)

        # Compute metrics
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)

        # Per-axis R2
        r2_agency = r2_score(targets[:, 0], predictions[:, 0])
        r2_fairness = r2_score(targets[:, 1], predictions[:, 1])
        r2_belonging = r2_score(targets[:, 2], predictions[:, 2])

        # Correlation
        # Flatten for overall correlation
        flat_targets = targets.flatten()
        flat_preds = predictions.flatten()
        spearman_rho, _ = spearmanr(flat_targets, flat_preds)
        pearson_r, _ = pearsonr(flat_targets, flat_preds)

        # Cross-validation (only for supervised methods)
        if method == "ridge" and n_samples >= n_cv_folds:
            cv = KFold(n_splits=n_cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(
                Ridge(alpha=1.0), embeddings, targets,
                cv=cv, scoring='r2'
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        else:
            # For unsupervised methods, use leave-one-out estimate
            cv_mean = r2
            cv_std = 0.0

        # Semantic structure preservation
        cosine_sim_preservation = self._compute_structure_preservation(
            embeddings, predictions
        )

        return ValidationMetrics(
            mse=float(mse),
            rmse=float(rmse),
            r2=float(r2),
            r2_agency=float(r2_agency),
            r2_fairness=float(r2_fairness),
            r2_belonging=float(r2_belonging),
            spearman_rho=float(spearman_rho),
            pearson_r=float(pearson_r),
            cv_mean=float(cv_mean),
            cv_std=float(cv_std),
            cosine_sim_preservation=float(cosine_sim_preservation)
        )

    def compare_methods(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        methods: Optional[List[str]] = None
    ) -> ProjectionComparison:
        """
        Compare multiple projection methods.

        Args:
            embeddings: (N, D) array of embeddings
            targets: (N, 3) array of target coordinates
            methods: List of methods to compare (default: all available)
        """
        if methods is None:
            methods = list(self.methods.keys())

        # Validate each method
        method_metrics = {}
        for method in methods:
            try:
                metrics = self.validate_projection(embeddings, targets, method)
                method_metrics[method] = metrics
            except Exception as e:
                logger.warning(f"Failed to validate {method}: {e}")
                continue

        if not method_metrics:
            raise ValueError("All projection methods failed")

        # Rank by R2 score
        ranking = sorted(
            method_metrics.keys(),
            key=lambda m: method_metrics[m].r2,
            reverse=True
        )

        best_method = ranking[0]
        best_metrics = method_metrics[best_method]

        # Generate recommendation
        if best_metrics.r2 < 0.3:
            recommendation = (
                "Warning: All projection methods show weak performance (R² < 0.3). "
                "This suggests the chosen axes may not be well-captured by the embedding space, "
                "or the training data may be insufficient or inconsistent."
            )
        elif best_metrics.r2 < 0.6:
            recommendation = (
                f"Moderate performance achieved with {best_method} (R² = {best_metrics.r2:.2f}). "
                "Consider adding more training examples or refining axis definitions."
            )
        else:
            recommendation = (
                f"Good performance with {best_method} (R² = {best_metrics.r2:.2f}). "
                f"Semantic structure preservation: {best_metrics.cosine_sim_preservation:.2f}"
            )

        return ProjectionComparison(
            method_metrics=method_metrics,
            best_method=best_method,
            ranking=ranking,
            recommendation=recommendation
        )

    def _fit_ridge(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[np.ndarray, Any]:
        """Fit Ridge regression."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(embeddings)

        model = Ridge(alpha=1.0)
        model.fit(scaled, targets)
        predictions = model.predict(scaled)

        return predictions, (model, scaler)

    def _fit_pca(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[np.ndarray, Any]:
        """
        Fit PCA to 3D, then Procrustes-align to targets.

        This tests whether the first 3 principal components
        capture the semantic structure.
        """
        # Reduce to 3D
        pca = PCA(n_components=3, random_state=self.random_state)
        projected = pca.fit_transform(embeddings)

        # Procrustes alignment to targets
        aligned = self._procrustes_align(projected, targets)

        return aligned, pca

    def _fit_tsne(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[np.ndarray, Any]:
        """
        Fit t-SNE to 3D, then Procrustes-align to targets.

        This tests whether local neighborhood structure
        matches the semantic labels.
        """
        # t-SNE to 3D
        tsne = TSNE(
            n_components=3,
            perplexity=min(30, len(embeddings) - 1),
            random_state=self.random_state,
            init='pca'
        )
        projected = tsne.fit_transform(embeddings)

        # Procrustes alignment
        aligned = self._procrustes_align(projected, targets)

        return aligned, tsne

    def _fit_umap(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[np.ndarray, Any]:
        """
        Fit UMAP to 3D, then Procrustes-align to targets.
        """
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.1,
            random_state=self.random_state
        )
        projected = reducer.fit_transform(embeddings)

        # Procrustes alignment
        aligned = self._procrustes_align(projected, targets)

        return aligned, reducer

    def _procrustes_align(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """
        Align source to target using Procrustes analysis.

        Finds optimal rotation, scaling, and translation.
        """
        # Center both
        source_centered = source - source.mean(axis=0)
        target_centered = target - target.mean(axis=0)

        # Compute optimal rotation via SVD
        M = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(M)
        R = Vt.T @ U.T

        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute scaling
        scale = np.sum(S) / np.sum(source_centered ** 2)

        # Apply transformation
        aligned = scale * (source_centered @ R) + target.mean(axis=0)

        return aligned

    def _compute_structure_preservation(
        self,
        embeddings: np.ndarray,
        projections: np.ndarray,
        sample_size: int = 500
    ) -> float:
        """
        Compute how well the projection preserves pairwise similarities.

        Uses Spearman correlation between original cosine similarities
        and projected Euclidean distances.
        """
        n = len(embeddings)
        if n < 2:
            return 0.0

        # Sample pairs if too many
        if n * (n - 1) // 2 > sample_size:
            np.random.seed(self.random_state)
            indices = np.random.choice(n, size=min(n, int(np.sqrt(2 * sample_size))), replace=False)
        else:
            indices = np.arange(n)

        # Compute original cosine similarities
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        orig_sims = []
        proj_dists = []

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i, idx_j = indices[i], indices[j]
                # Cosine similarity in embedding space
                cos_sim = np.dot(embeddings_norm[idx_i], embeddings_norm[idx_j])
                orig_sims.append(cos_sim)

                # Euclidean distance in projection space (inverted for correlation)
                dist = np.linalg.norm(projections[idx_i] - projections[idx_j])
                proj_dists.append(-dist)  # Negative so higher = more similar

        if len(orig_sims) < 2:
            return 0.0

        # Spearman correlation
        rho, _ = spearmanr(orig_sims, proj_dists)
        return rho if not np.isnan(rho) else 0.0

    def validate_against_annotations(
        self,
        embeddings: np.ndarray,
        annotation_targets: np.ndarray,
        annotation_uncertainties: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Validate projection against human annotations,
        accounting for annotation uncertainty.
        """
        # Compare all methods
        comparison = self.compare_methods(embeddings, annotation_targets)

        # Weight by inverse uncertainty if available
        if annotation_uncertainties is not None:
            weights = 1.0 / (annotation_uncertainties + 0.1)
            weights = weights / weights.sum()

            # Compute weighted metrics for best method
            best = comparison.best_method
            fit_fn = self.methods[best]
            predictions, _ = fit_fn(embeddings, annotation_targets)

            weighted_mse = np.average(
                (predictions - annotation_targets) ** 2,
                weights=weights.mean(axis=1)
            )

            comparison_dict = comparison.to_dict()
            comparison_dict["weighted_mse"] = float(weighted_mse)
            comparison_dict["uncertainty_weighted"] = True

            return comparison_dict

        return comparison.to_dict()
