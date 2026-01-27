"""
Projection Head

Maps high-dimensional embeddings to the 3D cultural manifold (agency, perceived_justice, belonging).

Note: Internal storage still uses "fairness" for backward compatibility with saved projections.
API responses use "perceived_justice" as the canonical name.

Two main approaches:
1. Linear Probe: Train a simple linear layer on labeled examples
2. Anchor Projection: Define anchor texts and project relative to them

The projection is the "lens" that interprets the latent space geometry
through our semantic framework.

Axis Naming (as of Jan 2026):
- "fairness" is now "perceived_justice" in API responses
- Internal storage retains "fairness" for backward compatibility
- Use get_axis_display_name() and translate_axis_name() for proper naming
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

# Import axis configuration for name translation
from .axis_config import (
    translate_axis_name,
    get_axis_display_name,
    convert_vector_keys_to_canonical,
    convert_vector_keys_to_internal,
    CANONICAL_AXES,
    INTERNAL_AXES
)

from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, KFold
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import joblib
import warnings

# Optional PyTorch import for neural projection
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Constants for scientific rigor
MINIMUM_TRAINING_SAMPLES = 100
MINIMUM_SAMPLES_PER_MODE = 10
CV_WARNING_THRESHOLD = 0.0  # Warn if CV score is negative
OVERFITTING_GAP_THRESHOLD = 0.3  # Warn if R² - CV > this value
DEFAULT_ALPHA_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


@dataclass
class Vector3:
    """
    3D position in the cultural manifold.

    Attributes:
        agency: Sense of personal control and self-determination
        fairness: Internal name for perceived_justice (backward compatibility)
        belonging: Sense of social connection and group membership

    Note: The 'fairness' attribute is kept for backward compatibility with saved
    projections. Use to_dict() for API responses which returns 'perceived_justice'.
    """
    agency: float
    fairness: float  # Internal name, maps to 'perceived_justice' in API
    belonging: float

    def to_list(self) -> List[float]:
        """Return values as list in order: [agency, fairness/perceived_justice, belonging]."""
        return [self.agency, self.fairness, self.belonging]

    def to_dict(self, use_canonical_names: bool = True) -> dict:
        """
        Convert to dictionary.

        Args:
            use_canonical_names: If True (default), uses 'perceived_justice' instead of 'fairness'.
                               Set to False for backward compatibility with internal storage.

        Returns:
            Dictionary with axis names as keys.
        """
        if use_canonical_names:
            return {
                "agency": self.agency,
                "perceived_justice": self.fairness,  # Use canonical name
                "belonging": self.belonging
            }
        else:
            return {
                "agency": self.agency,
                "fairness": self.fairness,  # Internal name for storage
                "belonging": self.belonging
            }

    def to_internal_dict(self) -> dict:
        """Return dictionary with internal axis names (for storage/backward compatibility)."""
        return self.to_dict(use_canonical_names=False)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Vector3":
        """Create Vector3 from numpy array [agency, fairness, belonging]."""
        return cls(agency=float(arr[0]), fairness=float(arr[1]), belonging=float(arr[2]))

    @classmethod
    def from_dict(cls, d: dict) -> "Vector3":
        """
        Create Vector3 from dictionary (accepts both canonical and internal names).

        Args:
            d: Dictionary with axis names. Accepts 'fairness', 'perceived_justice', or 'justice'.
        """
        agency = d.get("agency", 0.0)
        # Accept multiple names for the fairness/perceived_justice axis
        fairness = d.get("fairness") or d.get("perceived_justice") or d.get("justice", 0.0)
        belonging = d.get("belonging", 0.0)
        return cls(agency=agency, fairness=fairness, belonging=belonging)


@dataclass
class ProjectionWithUncertainty:
    """3D projection with per-axis uncertainty quantification."""
    coords: Vector3
    std_per_axis: Vector3  # Standard deviation per axis
    confidence_intervals: Dict[str, Tuple[float, float]]  # 95% CI per axis
    method: str = "unknown"  # Which projection method produced this

    def to_dict(self, use_canonical_names: bool = True) -> dict:
        """
        Convert to dictionary.

        Args:
            use_canonical_names: If True (default), uses 'perceived_justice' instead of 'fairness'.
        """
        # Convert confidence intervals to use canonical names if needed
        ci_dict = {}
        for k, v in self.confidence_intervals.items():
            if use_canonical_names and k == "fairness":
                ci_dict["perceived_justice"] = list(v)
            else:
                ci_dict[k] = list(v)

        return {
            "coords": self.coords.to_dict(use_canonical_names=use_canonical_names),
            "std_per_axis": self.std_per_axis.to_dict(use_canonical_names=use_canonical_names),
            "confidence_intervals": ci_dict,
            "method": self.method,
            "overall_confidence": self.overall_confidence
        }

    @property
    def overall_confidence(self) -> float:
        """Compute overall confidence as inverse of mean uncertainty (0-1 scale)."""
        mean_std = np.mean([
            self.std_per_axis.agency,
            self.std_per_axis.fairness,  # Internal name
            self.std_per_axis.belonging
        ])
        return float(1.0 / (1.0 + mean_std))


@dataclass
class TrainingExample:
    """
    A labeled example for training the projection.

    Note: The 'fairness' field is kept for backward compatibility with saved training data.
    It represents 'perceived_justice' in the current terminology.
    """
    text: str
    agency: float
    fairness: float  # Internal name, represents 'perceived_justice'
    belonging: float
    embedding: Optional[np.ndarray] = None
    source: str = "manual"  # manual, synthetic, imported

    def to_dict(self, use_canonical_names: bool = False) -> dict:
        """
        Convert to dictionary.

        Args:
            use_canonical_names: If True, uses 'perceived_justice'. Default False for storage.
        """
        if use_canonical_names:
            return {
                "text": self.text,
                "agency": self.agency,
                "perceived_justice": self.fairness,
                "belonging": self.belonging,
                "source": self.source
            }
        else:
            return {
                "text": self.text,
                "agency": self.agency,
                "fairness": self.fairness,  # Internal name for backward compat
                "belonging": self.belonging,
                "source": self.source
            }


@dataclass
class ProjectionMetrics:
    """Quality metrics for the trained projection."""
    r2_agency: float
    r2_fairness: float  # Internal name, represents r2_perceived_justice
    r2_belonging: float
    r2_overall: float
    cv_score_mean: float
    cv_score_std: float
    n_examples: int
    # New fields for scientific rigor
    test_r2: Optional[float] = None  # R² on held-out test set
    cv_scores_per_fold: Optional[List[float]] = None  # Individual fold scores
    best_alpha: Optional[float] = None  # Best regularization strength found
    train_test_gap: Optional[float] = None  # R² - test_r2 (overfitting indicator)
    mode_distribution: Optional[Dict[str, int]] = None  # Samples per mode
    warnings: List[str] = field(default_factory=list)  # Quality warnings

    def to_dict(self, use_canonical_names: bool = True) -> dict:
        """
        Convert to dictionary.

        Args:
            use_canonical_names: If True (default), uses 'perceived_justice' instead of 'fairness'.
        """
        fairness_key = "r2_perceived_justice" if use_canonical_names else "r2_fairness"
        return {
            "r2_agency": self.r2_agency,
            fairness_key: self.r2_fairness,
            "r2_belonging": self.r2_belonging,
            "r2_overall": self.r2_overall,
            "cv_score_mean": self.cv_score_mean,
            "cv_score_std": self.cv_score_std,
            "n_examples": self.n_examples,
            "test_r2": self.test_r2,
            "cv_scores_per_fold": self.cv_scores_per_fold,
            "best_alpha": self.best_alpha,
            "train_test_gap": self.train_test_gap,
            "mode_distribution": self.mode_distribution,
            "warnings": self.warnings
        }

    def is_overfit(self) -> bool:
        """Check if model shows signs of overfitting."""
        if self.cv_score_mean < CV_WARNING_THRESHOLD:
            return True
        if self.train_test_gap is not None and self.train_test_gap > OVERFITTING_GAP_THRESHOLD:
            return True
        return False

    @classmethod
    def from_dict(cls, data: dict) -> 'ProjectionMetrics':
        """
        Create ProjectionMetrics from a dictionary, handling axis name translation.

        This handles loading metrics saved with canonical names (r2_perceived_justice)
        and converting them to internal names (r2_fairness).
        """
        # Translate canonical names back to internal names for metrics
        translated = dict(data)
        if "r2_perceived_justice" in translated:
            translated["r2_fairness"] = translated.pop("r2_perceived_justice")

        return cls(**translated)


class ProjectionHead:
    """
    Trainable projection from embedding space to 3D manifold.

    Uses Ridge regression for stability and interpretability.
    The learned weights show which embedding dimensions
    contribute to each semantic axis.
    """

    def __init__(self, embedding_dim: Optional[int] = None):
        self.embedding_dim = embedding_dim
        self.model: Optional[Ridge] = None
        self.scaler: Optional[StandardScaler] = None
        self.metrics: Optional[ProjectionMetrics] = None
        self.is_trained = False

    def train(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        alpha: Optional[float] = None,
        normalize_inputs: bool = True,
        mode_labels: Optional[np.ndarray] = None,
        auto_tune_alpha: bool = True,
        n_cv_folds: int = 5,
        test_size: float = 0.2,
        enforce_minimum_samples: bool = True
    ) -> ProjectionMetrics:
        """
        Train the projection on labeled examples with scientific rigor.

        Args:
            embeddings: (N, D) array of embeddings
            targets: (N, 3) array of [agency, fairness, belonging]
            alpha: Ridge regularization strength (if None, auto-tune)
            normalize_inputs: Whether to standardize embeddings
            mode_labels: Optional mode labels for stratified splitting
            auto_tune_alpha: Whether to try multiple alpha values
            n_cv_folds: Number of cross-validation folds
            test_size: Fraction of data to hold out for testing
            enforce_minimum_samples: Whether to enforce minimum sample requirements

        Returns:
            ProjectionMetrics with comprehensive quality indicators
        """
        warnings_list = []

        if embeddings.shape[0] != targets.shape[0]:
            raise ValueError("Embeddings and targets must have same number of examples")

        n_samples = embeddings.shape[0]

        # Minimum sample size validation
        if enforce_minimum_samples and n_samples < MINIMUM_TRAINING_SAMPLES:
            warnings_list.append(
                f"INSUFFICIENT_DATA: Only {n_samples} examples provided. "
                f"Minimum recommended is {MINIMUM_TRAINING_SAMPLES} for reliable results. "
                f"Model may severely overfit."
            )
            logger.warning(f"Training with insufficient data: {n_samples} < {MINIMUM_TRAINING_SAMPLES}")

        if n_samples < 10:
            raise ValueError(f"Need at least 10 examples to train, have {n_samples}")

        self.embedding_dim = embeddings.shape[1]

        # Generate pseudo-mode labels for stratification if not provided
        if mode_labels is None:
            mode_labels = self._assign_mode_labels(targets)

        # Compute mode distribution
        unique_modes, mode_counts = np.unique(mode_labels, return_counts=True)
        mode_distribution = {str(m): int(c) for m, c in zip(unique_modes, mode_counts)}

        # Warn about underrepresented modes
        for mode, count in zip(unique_modes, mode_counts):
            if count < MINIMUM_SAMPLES_PER_MODE:
                warnings_list.append(
                    f"UNDERREPRESENTED_MODE: Mode '{mode}' has only {count} samples. "
                    f"Minimum recommended is {MINIMUM_SAMPLES_PER_MODE}."
                )

        # Stratified train/test split (80/20)
        try:
            X_train, X_test, y_train, y_test, mode_train, mode_test = train_test_split(
                embeddings, targets, mode_labels,
                test_size=test_size,
                stratify=mode_labels,
                random_state=42
            )
        except ValueError:
            # Fall back to non-stratified if stratification fails
            warnings_list.append(
                "STRATIFICATION_FAILED: Could not stratify by mode. Using random split."
            )
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, targets,
                test_size=test_size,
                random_state=42
            )
            mode_train = None

        # Normalize inputs
        if normalize_inputs:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            X_full_scaled = self.scaler.transform(embeddings)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            X_full_scaled = embeddings

        # Auto-tune alpha using cross-validation on training set
        if auto_tune_alpha or alpha is None:
            best_alpha, alpha_cv_scores = self._tune_alpha(
                X_train_scaled, y_train, n_cv_folds, mode_train
            )
        else:
            best_alpha = alpha
            alpha_cv_scores = None

        # Train final model on full training set
        self.model = Ridge(alpha=best_alpha)
        self.model.fit(X_train_scaled, y_train)

        # Compute training metrics
        train_predictions = self.model.predict(X_train_scaled)
        r2_train = self.model.score(X_train_scaled, y_train)

        # Compute test metrics (held-out data)
        test_predictions = self.model.predict(X_test_scaled)
        r2_test = self.model.score(X_test_scaled, y_test)

        # Per-axis R² on test set
        r2_agency = self._compute_r2(y_test[:, 0], test_predictions[:, 0])
        r2_fairness = self._compute_r2(y_test[:, 1], test_predictions[:, 1])
        r2_belonging = self._compute_r2(y_test[:, 2], test_predictions[:, 2])

        # K-fold cross-validation on full dataset for reporting
        n_cv = min(n_cv_folds, n_samples)
        try:
            if mode_labels is not None and len(np.unique(mode_labels)) >= n_cv:
                cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
                cv_scores = cross_val_score(
                    Ridge(alpha=best_alpha),
                    X_full_scaled, targets,
                    cv=cv.split(X_full_scaled, mode_labels),
                    scoring='r2'
                )
            else:
                cv_scores = cross_val_score(
                    Ridge(alpha=best_alpha),
                    X_full_scaled, targets,
                    cv=KFold(n_splits=n_cv, shuffle=True, random_state=42),
                    scoring='r2'
                )
        except Exception as e:
            warnings_list.append(f"CV_FAILED: Cross-validation failed: {str(e)}")
            cv_scores = np.array([r2_test])

        # Check for overfitting
        train_test_gap = r2_train - r2_test
        if train_test_gap > OVERFITTING_GAP_THRESHOLD:
            warnings_list.append(
                f"OVERFITTING_DETECTED: Training R²={r2_train:.3f}, Test R²={r2_test:.3f}. "
                f"Gap of {train_test_gap:.3f} exceeds threshold of {OVERFITTING_GAP_THRESHOLD}."
            )

        # Check for negative CV score
        if cv_scores.mean() < CV_WARNING_THRESHOLD:
            warnings_list.append(
                f"NEGATIVE_CV_SCORE: Mean CV score is {cv_scores.mean():.3f}. "
                f"This indicates severe overfitting or insufficient data diversity. "
                f"Model predictions may be worse than a simple mean baseline."
            )

        # Retrain on full dataset for deployment
        self.model = Ridge(alpha=best_alpha)
        self.model.fit(X_full_scaled, targets)

        # Final full-data metrics (for reference only)
        full_predictions = self.model.predict(X_full_scaled)
        r2_overall = self.model.score(X_full_scaled, targets)

        self.metrics = ProjectionMetrics(
            r2_agency=float(r2_agency),
            r2_fairness=float(r2_fairness),
            r2_belonging=float(r2_belonging),
            r2_overall=float(r2_overall),
            cv_score_mean=float(cv_scores.mean()),
            cv_score_std=float(cv_scores.std()),
            n_examples=n_samples,
            test_r2=float(r2_test),
            cv_scores_per_fold=[float(s) for s in cv_scores],
            best_alpha=float(best_alpha),
            train_test_gap=float(train_test_gap),
            mode_distribution=mode_distribution,
            warnings=warnings_list
        )

        self.is_trained = True

        # Log with appropriate severity
        if warnings_list:
            logger.warning(
                f"Projection trained with warnings: R²={r2_overall:.3f}, "
                f"Test R²={r2_test:.3f}, CV={cv_scores.mean():.3f}. "
                f"Warnings: {len(warnings_list)}"
            )
            for w in warnings_list:
                logger.warning(f"  - {w}")
        else:
            logger.info(
                f"Projection trained successfully: R²={r2_overall:.3f}, "
                f"Test R²={r2_test:.3f}, CV={cv_scores.mean():.3f}, alpha={best_alpha}"
            )

        return self.metrics

    def _assign_mode_labels(self, targets: np.ndarray) -> np.ndarray:
        """Assign mode labels based on coordinate space position."""
        labels = []
        for t in targets:
            agency, fairness, belonging = t
            # Determine dominant axis and sign
            abs_vals = np.abs(t)
            max_idx = np.argmax(abs_vals)

            if abs_vals[max_idx] < 0.5:
                labels.append("neutral")
            elif max_idx == 0:
                labels.append("high_agency" if agency > 0 else "low_agency")
            elif max_idx == 1:
                labels.append("high_fairness" if fairness > 0 else "low_fairness")
            else:
                labels.append("high_belonging" if belonging > 0 else "low_belonging")

        return np.array(labels)

    def _tune_alpha(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int,
        mode_labels: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[float, float]]:
        """Tune alpha via cross-validation, return best alpha and scores."""
        alphas = DEFAULT_ALPHA_VALUES
        scores = {}

        n_cv = min(n_folds, len(X))

        for alpha in alphas:
            try:
                if mode_labels is not None and len(np.unique(mode_labels)) >= n_cv:
                    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(
                        Ridge(alpha=alpha), X, y,
                        cv=cv.split(X, mode_labels),
                        scoring='r2'
                    )
                else:
                    cv_scores = cross_val_score(
                        Ridge(alpha=alpha), X, y,
                        cv=KFold(n_splits=n_cv, shuffle=True, random_state=42),
                        scoring='r2'
                    )
                scores[alpha] = cv_scores.mean()
            except Exception:
                scores[alpha] = -np.inf

        best_alpha = max(scores, key=scores.get)
        logger.info(f"Alpha tuning: best={best_alpha} with CV={scores[best_alpha]:.3f}")
        return best_alpha, scores

    def _compute_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² score, handling edge cases."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        if ss_tot == 0:
            return 0.0 if ss_res > 0 else 1.0
        return 1 - (ss_res / ss_tot)

    def project(self, embedding: np.ndarray) -> Vector3:
        """Project a single embedding to 3D."""
        if not self.is_trained:
            raise ValueError("Projection not trained. Call train() first.")

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Validate dimension matches what projection was trained on
        actual_dim = embedding.shape[1]
        if self.embedding_dim is not None and actual_dim != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: projection trained on {self.embedding_dim} dims, "
                f"but received {actual_dim} dims. Please retrain the projection with the current model."
            )

        if self.scaler is not None:
            embedding = self.scaler.transform(embedding)

        coords = self.model.predict(embedding)[0]
        return Vector3.from_array(coords)

    def project_batch(self, embeddings: np.ndarray) -> List[Vector3]:
        """Project multiple embeddings."""
        if not self.is_trained:
            raise ValueError("Projection not trained. Call train() first.")

        # Validate dimension matches what projection was trained on
        actual_dim = embeddings.shape[1]
        if self.embedding_dim is not None and actual_dim != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: projection trained on {self.embedding_dim} dims, "
                f"but received {actual_dim} dims. Please retrain the projection with the current model."
            )

        if self.scaler is not None:
            embeddings = self.scaler.transform(embeddings)

        coords = self.model.predict(embeddings)
        return [Vector3.from_array(c) for c in coords]

    def get_axis_weights(self, use_canonical_names: bool = True) -> Dict[str, np.ndarray]:
        """
        Get the learned weights for each axis.

        These show which embedding dimensions contribute to each semantic axis.
        Useful for interpretability.

        Args:
            use_canonical_names: If True (default), uses 'perceived_justice' instead of 'fairness'.
        """
        if not self.is_trained:
            raise ValueError("Projection not trained.")

        weights = self.model.coef_  # (3, D)
        fairness_key = "perceived_justice" if use_canonical_names else "fairness"
        return {
            "agency": weights[0],
            fairness_key: weights[1],
            "belonging": weights[2]
        }

    def get_important_dimensions(self, top_k: int = 20) -> Dict[str, List[int]]:
        """Get the most important embedding dimensions for each axis."""
        weights = self.get_axis_weights()
        result = {}
        for axis, w in weights.items():
            # Get indices sorted by absolute weight
            indices = np.argsort(np.abs(w))[::-1][:top_k]
            result[axis] = indices.tolist()
        return result

    def save(self, path: Path):
        """Save the trained projection."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained projection.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self.model, path / "model.joblib")

        # Save scaler if present
        if self.scaler is not None:
            joblib.dump(self.scaler, path / "scaler.joblib")

        # Save metadata
        metadata = {
            "embedding_dim": self.embedding_dim,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "has_scaler": self.scaler is not None
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Projection saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "ProjectionHead":
        """Load a saved projection."""
        path = Path(path)

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        proj = cls(embedding_dim=metadata["embedding_dim"])
        proj.model = joblib.load(path / "model.joblib")

        if metadata.get("has_scaler"):
            proj.scaler = joblib.load(path / "scaler.joblib")

        if metadata.get("metrics"):
            proj.metrics = ProjectionMetrics.from_dict(metadata["metrics"])

        proj.is_trained = True
        logger.info(f"Projection loaded from {path}")
        return proj


class AnchorProjection:
    """
    Alternative projection using anchor texts.

    Instead of training on labeled examples, we define anchor texts
    that represent the extremes of each axis and project relative to them.

    Advantages:
    - No training required
    - Very interpretable
    - Easy to adjust

    Disadvantages:
    - Less flexible
    - May not capture nuance
    """

    # Default anchors for each axis
    DEFAULT_ANCHORS = {
        "agency_positive": [
            "I achieved this through my own hard work and determination.",
            "I take full control of my destiny.",
            "Success comes from individual effort and willpower.",
            "I made this happen entirely on my own."
        ],
        "agency_negative": [
            "Circumstances beyond my control determine my fate.",
            "The system decides what happens to people like me.",
            "There's nothing I can do to change my situation.",
            "External forces shape everything in my life."
        ],
        "fairness_positive": [
            "Everyone gets what they truly deserve.",
            "The rules apply equally to everyone.",
            "Merit is rewarded and corruption punished.",
            "Justice prevails in the end."
        ],
        "fairness_negative": [
            "The system is rigged against ordinary people.",
            "Those in power play by different rules.",
            "Hard work doesn't matter, only connections.",
            "The game is fixed from the start."
        ],
        "belonging_positive": [
            "We are all part of something greater than ourselves.",
            "Our community gives life meaning.",
            "Together we are stronger than apart.",
            "I feel deeply connected to my people."
        ],
        "belonging_negative": [
            "Everyone is alone in this world.",
            "Communities are just illusions.",
            "I don't belong anywhere.",
            "Connection to others is meaningless."
        ]
    }

    def __init__(self, anchors: Optional[Dict[str, List[str]]] = None):
        self.anchors = anchors or self.DEFAULT_ANCHORS
        self.anchor_embeddings: Dict[str, np.ndarray] = {}
        self.axis_directions: Dict[str, np.ndarray] = {}

    def calibrate(self, embedding_extractor, model_id: str, layer: int = -1):
        """
        Compute anchor embeddings and axis directions.

        Args:
            embedding_extractor: EmbeddingExtractor instance
            model_id: Model to use for embeddings
            layer: Which layer to extract from
        """
        from .embedding import EmbeddingExtractor

        # Embed all anchors
        for anchor_name, texts in self.anchors.items():
            results = embedding_extractor.extract(texts, model_id, layer=layer)
            embeddings = np.array([r.embedding for r in results])
            self.anchor_embeddings[anchor_name] = embeddings.mean(axis=0)

        # Compute axis directions (positive - negative)
        for axis in ["agency", "fairness", "belonging"]:
            pos = self.anchor_embeddings[f"{axis}_positive"]
            neg = self.anchor_embeddings[f"{axis}_negative"]
            direction = pos - neg
            # Normalize
            direction = direction / np.linalg.norm(direction)
            self.axis_directions[axis] = direction

        logger.info("Anchor projection calibrated")

    def project(self, embedding: np.ndarray) -> Vector3:
        """Project embedding using anchor directions."""
        if not self.axis_directions:
            raise ValueError("Projection not calibrated. Call calibrate() first.")

        # Project onto each axis direction
        agency = float(np.dot(embedding, self.axis_directions["agency"]))
        fairness = float(np.dot(embedding, self.axis_directions["fairness"]))
        belonging = float(np.dot(embedding, self.axis_directions["belonging"]))

        # Scale to roughly -2 to 2 range (anchors are at ~1 and ~-1)
        return Vector3(agency=agency, fairness=fairness, belonging=belonging)


# =============================================================================
# Non-Linear Projection Methods
# =============================================================================

class GaussianProcessProjection:
    """
    Gaussian Process projection with proper uncertainty quantification.

    Uses RBF kernel with automatic length scale optimization.
    Returns mean predictions AND variance estimates for each axis.

    Advantages:
    - True uncertainty quantification via posterior variance
    - Non-linear decision boundaries
    - Automatic hyperparameter optimization

    Disadvantages:
    - O(n^3) complexity - slow for large datasets
    - Memory intensive for large training sets
    """

    def __init__(self, embedding_dim: Optional[int] = None):
        self.embedding_dim = embedding_dim
        self.models: Dict[str, GaussianProcessRegressor] = {}  # One GP per axis
        self.scaler: Optional[StandardScaler] = None
        self.metrics: Optional[ProjectionMetrics] = None
        self.is_trained = False
        self.axis_names = ["agency", "fairness", "belonging"]

    def train(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        normalize_inputs: bool = True,
        length_scale_bounds: Tuple[float, float] = (1e-2, 1e3),
        noise_level: float = 0.1,
        n_restarts: int = 5
    ) -> ProjectionMetrics:
        """
        Train separate GP regressors for each axis.

        Args:
            embeddings: (N, D) array of embeddings
            targets: (N, 3) array of [agency, fairness, belonging]
            normalize_inputs: Whether to standardize embeddings
            length_scale_bounds: Bounds for RBF length scale optimization
            noise_level: Initial noise level for WhiteKernel
            n_restarts: Number of optimizer restarts for hyperparameter tuning
        """
        if embeddings.shape[0] != targets.shape[0]:
            raise ValueError("Embeddings and targets must have same number of examples")

        n_samples = embeddings.shape[0]
        if n_samples < 5:
            raise ValueError(f"Need at least 5 examples to train, have {n_samples}")

        self.embedding_dim = embeddings.shape[1]

        # Normalize inputs
        if normalize_inputs:
            self.scaler = StandardScaler()
            embeddings_scaled = self.scaler.fit_transform(embeddings)
        else:
            embeddings_scaled = embeddings

        # Define kernel: RBF + WhiteKernel for noise
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            RBF(length_scale=1.0, length_scale_bounds=length_scale_bounds) +
            WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-5, 1e1))
        )

        # Train one GP per axis
        r2_scores = {}
        cv_scores_all = []

        for i, axis in enumerate(self.axis_names):
            logger.info(f"Training GP for {axis}...")
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restarts,
                random_state=42,
                normalize_y=True
            )
            gp.fit(embeddings_scaled, targets[:, i])
            self.models[axis] = gp

            # Compute R^2
            predictions, _ = gp.predict(embeddings_scaled, return_std=True)
            r2 = 1 - np.sum((targets[:, i] - predictions)**2) / np.sum((targets[:, i] - targets[:, i].mean())**2)
            r2_scores[axis] = float(r2)

            # Cross-validation score
            cv_scores = cross_val_score(
                GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=42),
                embeddings_scaled, targets[:, i],
                cv=min(5, n_samples),
                scoring='r2'
            )
            cv_scores_all.extend(cv_scores)

            logger.info(f"  {axis}: R^2={r2:.3f}, kernel={gp.kernel_}")

        cv_scores_array = np.array(cv_scores_all)
        r2_overall = np.mean(list(r2_scores.values()))

        self.metrics = ProjectionMetrics(
            r2_agency=r2_scores["agency"],
            r2_fairness=r2_scores["fairness"],
            r2_belonging=r2_scores["belonging"],
            r2_overall=float(r2_overall),
            cv_score_mean=float(cv_scores_array.mean()),
            cv_score_std=float(cv_scores_array.std()),
            n_examples=n_samples
        )

        self.is_trained = True
        logger.info(f"GP Projection trained: R^2={r2_overall:.3f}, CV={cv_scores_array.mean():.3f}")

        return self.metrics

    def project(self, embedding: np.ndarray) -> ProjectionWithUncertainty:
        """
        Project embedding with uncertainty quantification.

        Returns mean prediction AND standard deviation for each axis.
        """
        if not self.is_trained:
            raise ValueError("Projection not trained. Call train() first.")

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Validate dimension
        if self.embedding_dim is not None and embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[1]}"
            )

        # Scale
        if self.scaler is not None:
            embedding = self.scaler.transform(embedding)

        # Predict with uncertainty for each axis
        means = {}
        stds = {}
        for axis in self.axis_names:
            mean, std = self.models[axis].predict(embedding, return_std=True)
            means[axis] = float(mean[0])
            stds[axis] = float(std[0])

        # Compute 95% confidence intervals (mean +/- 1.96 * std)
        z = 1.96
        confidence_intervals = {
            axis: (means[axis] - z * stds[axis], means[axis] + z * stds[axis])
            for axis in self.axis_names
        }

        return ProjectionWithUncertainty(
            coords=Vector3(
                agency=means["agency"],
                fairness=means["fairness"],
                belonging=means["belonging"]
            ),
            std_per_axis=Vector3(
                agency=stds["agency"],
                fairness=stds["fairness"],
                belonging=stds["belonging"]
            ),
            confidence_intervals=confidence_intervals,
            method="gaussian_process"
        )

    def project_batch(self, embeddings: np.ndarray) -> List[ProjectionWithUncertainty]:
        """Project multiple embeddings with uncertainty."""
        return [self.project(emb) for emb in embeddings]

    def save(self, path: Path):
        """Save the trained GP projection."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained projection.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save models
        for axis, model in self.models.items():
            joblib.dump(model, path / f"gp_{axis}.joblib")

        # Save scaler
        if self.scaler is not None:
            joblib.dump(self.scaler, path / "scaler.joblib")

        # Save metadata
        metadata = {
            "embedding_dim": self.embedding_dim,
            "method": "gaussian_process",
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "has_scaler": self.scaler is not None
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"GP Projection saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "GaussianProcessProjection":
        """Load a saved GP projection."""
        path = Path(path)

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        proj = cls(embedding_dim=metadata["embedding_dim"])

        for axis in proj.axis_names:
            proj.models[axis] = joblib.load(path / f"gp_{axis}.joblib")

        if metadata.get("has_scaler"):
            proj.scaler = joblib.load(path / "scaler.joblib")

        if metadata.get("metrics"):
            proj.metrics = ProjectionMetrics.from_dict(metadata["metrics"])

        proj.is_trained = True
        logger.info(f"GP Projection loaded from {path}")
        return proj


class NeuralProbeProjection:
    """
    Neural network projection using a 2-layer MLP with MC Dropout.

    Architecture: embedding_dim -> 128 -> 3
    Uses dropout during inference for Monte Carlo uncertainty estimation.

    Advantages:
    - Non-linear projection
    - Fast inference once trained
    - MC Dropout provides uncertainty estimates

    Disadvantages:
    - Requires PyTorch
    - More hyperparameters to tune
    - May overfit with small datasets
    """

    def __init__(self, embedding_dim: Optional[int] = None, hidden_dim: int = 128, dropout: float = 0.2):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NeuralProbeProjection. Install with: pip install torch")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.model: Optional[nn.Module] = None
        self.scaler: Optional[StandardScaler] = None
        self.metrics: Optional[ProjectionMetrics] = None
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, input_dim: int) -> nn.Module:
        """Build the MLP model."""
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 3)
        )

    def train(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        normalize_inputs: bool = True,
        epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 20,
        validation_split: float = 0.2
    ) -> ProjectionMetrics:
        """
        Train the neural projection with early stopping.

        Args:
            embeddings: (N, D) array of embeddings
            targets: (N, 3) array of [agency, fairness, belonging]
            normalize_inputs: Whether to standardize embeddings
            epochs: Maximum training epochs
            batch_size: Training batch size
            learning_rate: Adam learning rate
            patience: Early stopping patience
            validation_split: Fraction of data for validation
        """
        if embeddings.shape[0] != targets.shape[0]:
            raise ValueError("Embeddings and targets must have same number of examples")

        n_samples = embeddings.shape[0]
        if n_samples < 10:
            raise ValueError(f"Need at least 10 examples to train, have {n_samples}")

        self.embedding_dim = embeddings.shape[1]

        # Normalize inputs
        if normalize_inputs:
            self.scaler = StandardScaler()
            embeddings_scaled = self.scaler.fit_transform(embeddings)
        else:
            embeddings_scaled = embeddings

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            embeddings_scaled, targets,
            test_size=validation_split,
            random_state=42
        )

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        # Create dataloader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Build and train model
        self.model = self._build_model(self.embedding_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 50 == 0:
                logger.info(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Compute metrics
        self.model.eval()
        with torch.no_grad():
            X_full_t = torch.FloatTensor(embeddings_scaled).to(self.device)
            predictions = self.model(X_full_t).cpu().numpy()

        # Per-axis R^2
        r2_agency = 1 - np.sum((targets[:, 0] - predictions[:, 0])**2) / np.sum((targets[:, 0] - targets[:, 0].mean())**2)
        r2_fairness = 1 - np.sum((targets[:, 1] - predictions[:, 1])**2) / np.sum((targets[:, 1] - targets[:, 1].mean())**2)
        r2_belonging = 1 - np.sum((targets[:, 2] - predictions[:, 2])**2) / np.sum((targets[:, 2] - targets[:, 2].mean())**2)
        r2_overall = np.mean([r2_agency, r2_fairness, r2_belonging])

        # Validation R^2 as CV proxy
        with torch.no_grad():
            val_predictions = self.model(X_val_t).cpu().numpy()
        val_r2 = 1 - np.sum((y_val - val_predictions)**2) / np.sum((y_val - y_val.mean(axis=0))**2)

        self.metrics = ProjectionMetrics(
            r2_agency=float(r2_agency),
            r2_fairness=float(r2_fairness),
            r2_belonging=float(r2_belonging),
            r2_overall=float(r2_overall),
            cv_score_mean=float(val_r2),
            cv_score_std=0.0,  # Single validation fold
            n_examples=n_samples,
            test_r2=float(val_r2)
        )

        self.is_trained = True
        logger.info(f"Neural Projection trained: R^2={r2_overall:.3f}, Val R^2={val_r2:.3f}")

        return self.metrics

    def project(self, embedding: np.ndarray, n_samples: int = 30) -> ProjectionWithUncertainty:
        """
        Project embedding with MC Dropout uncertainty estimation.

        Args:
            embedding: Input embedding
            n_samples: Number of forward passes for MC Dropout
        """
        if not self.is_trained:
            raise ValueError("Projection not trained. Call train() first.")

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Validate dimension
        if self.embedding_dim is not None and embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[1]}"
            )

        # Scale
        if self.scaler is not None:
            embedding = self.scaler.transform(embedding)

        # Convert to tensor
        X_t = torch.FloatTensor(embedding).to(self.device)

        # MC Dropout: multiple forward passes with dropout enabled
        self.model.train()  # Enable dropout
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(X_t).cpu().numpy()
                predictions.append(pred[0])

        predictions = np.array(predictions)
        means = predictions.mean(axis=0)
        stds = predictions.std(axis=0)

        # 95% confidence intervals
        z = 1.96
        confidence_intervals = {
            "agency": (float(means[0] - z * stds[0]), float(means[0] + z * stds[0])),
            "fairness": (float(means[1] - z * stds[1]), float(means[1] + z * stds[1])),
            "belonging": (float(means[2] - z * stds[2]), float(means[2] + z * stds[2]))
        }

        return ProjectionWithUncertainty(
            coords=Vector3(
                agency=float(means[0]),
                fairness=float(means[1]),
                belonging=float(means[2])
            ),
            std_per_axis=Vector3(
                agency=float(stds[0]),
                fairness=float(stds[1]),
                belonging=float(stds[2])
            ),
            confidence_intervals=confidence_intervals,
            method="neural_mc_dropout"
        )

    def project_batch(self, embeddings: np.ndarray, n_samples: int = 30) -> List[ProjectionWithUncertainty]:
        """Project multiple embeddings with uncertainty."""
        return [self.project(emb, n_samples) for emb in embeddings]

    def save(self, path: Path):
        """Save the trained neural projection."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained projection.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(self.model.state_dict(), path / "model.pt")

        # Save scaler
        if self.scaler is not None:
            joblib.dump(self.scaler, path / "scaler.joblib")

        # Save metadata
        metadata = {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "method": "neural_probe",
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "has_scaler": self.scaler is not None
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Neural Projection saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "NeuralProbeProjection":
        """Load a saved neural projection."""
        path = Path(path)

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        proj = cls(
            embedding_dim=metadata["embedding_dim"],
            hidden_dim=metadata.get("hidden_dim", 128),
            dropout=metadata.get("dropout_rate", 0.2)
        )

        proj.model = proj._build_model(proj.embedding_dim).to(proj.device)
        proj.model.load_state_dict(torch.load(path / "model.pt", map_location=proj.device))

        if metadata.get("has_scaler"):
            proj.scaler = joblib.load(path / "scaler.joblib")

        if metadata.get("metrics"):
            proj.metrics = ProjectionMetrics.from_dict(metadata["metrics"])

        proj.is_trained = True
        logger.info(f"Neural Projection loaded from {path}")
        return proj


class ConceptActivationVectors:
    """
    Concept Activation Vectors (CAV) projection.

    Trains separate binary classifiers for high/low on each axis.
    Projects by computing distance to CAV hyperplanes.

    Advantages:
    - Interpretable directions in embedding space
    - Based on established interpretability methods
    - Works well with small datasets

    Disadvantages:
    - Requires binarization of continuous labels
    - May lose nuance in middle-range values
    """

    def __init__(self, embedding_dim: Optional[int] = None, classifier_type: str = "svm"):
        self.embedding_dim = embedding_dim
        self.classifier_type = classifier_type  # "svm" or "logistic"
        self.classifiers: Dict[str, object] = {}  # Binary classifiers per axis
        self.cav_directions: Dict[str, np.ndarray] = {}  # CAV vectors
        self.scaler: Optional[StandardScaler] = None
        self.metrics: Optional[ProjectionMetrics] = None
        self.is_trained = False
        self.axis_names = ["agency", "fairness", "belonging"]
        self.threshold = 0.0  # Threshold for high/low classification

    def train(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        normalize_inputs: bool = True,
        threshold: float = 0.0,
        svm_c: float = 1.0,
        logistic_c: float = 1.0
    ) -> ProjectionMetrics:
        """
        Train CAV classifiers for each axis.

        Args:
            embeddings: (N, D) array of embeddings
            targets: (N, 3) array of [agency, fairness, belonging]
            normalize_inputs: Whether to standardize embeddings
            threshold: Value for binarizing into high/low (default: 0.0)
            svm_c: Regularization for SVM
            logistic_c: Regularization for logistic regression
        """
        if embeddings.shape[0] != targets.shape[0]:
            raise ValueError("Embeddings and targets must have same number of examples")

        n_samples = embeddings.shape[0]
        if n_samples < 10:
            raise ValueError(f"Need at least 10 examples to train, have {n_samples}")

        self.embedding_dim = embeddings.shape[1]
        self.threshold = threshold

        # Normalize inputs
        if normalize_inputs:
            self.scaler = StandardScaler()
            embeddings_scaled = self.scaler.fit_transform(embeddings)
        else:
            embeddings_scaled = embeddings

        # Train binary classifier for each axis
        r2_scores = {}
        cv_scores_all = []

        for i, axis in enumerate(self.axis_names):
            # Binarize labels: high (>= threshold) vs low (< threshold)
            binary_labels = (targets[:, i] >= threshold).astype(int)

            # Check if we have both classes
            unique_labels = np.unique(binary_labels)
            if len(unique_labels) < 2:
                logger.warning(f"Axis {axis}: Only one class present. Skipping CAV training.")
                # Create dummy classifier
                self.classifiers[axis] = None
                self.cav_directions[axis] = np.zeros(self.embedding_dim)
                r2_scores[axis] = 0.0
                continue

            # Train classifier
            if self.classifier_type == "svm":
                clf = SVC(kernel='linear', C=svm_c, random_state=42)
            else:
                clf = LogisticRegression(C=logistic_c, random_state=42, max_iter=1000)

            clf.fit(embeddings_scaled, binary_labels)
            self.classifiers[axis] = clf

            # Extract CAV direction (normal to hyperplane)
            if hasattr(clf, 'coef_'):
                cav = clf.coef_[0]
                cav = cav / (np.linalg.norm(cav) + 1e-10)
                self.cav_directions[axis] = cav
            else:
                self.cav_directions[axis] = np.zeros(self.embedding_dim)

            # Compute projection-based R^2
            # Project embeddings onto CAV direction
            projections = embeddings_scaled @ self.cav_directions[axis]
            # Scale to approximate target range
            proj_mean, proj_std = projections.mean(), projections.std() + 1e-10
            scaled_proj = (projections - proj_mean) / proj_std * targets[:, i].std() + targets[:, i].mean()

            r2 = 1 - np.sum((targets[:, i] - scaled_proj)**2) / np.sum((targets[:, i] - targets[:, i].mean())**2)
            r2_scores[axis] = max(0.0, float(r2))

            # Cross-validation for classification accuracy
            cv_scores = cross_val_score(
                SVC(kernel='linear', C=svm_c) if self.classifier_type == "svm"
                else LogisticRegression(C=logistic_c, max_iter=1000),
                embeddings_scaled, binary_labels,
                cv=min(5, n_samples),
                scoring='accuracy'
            )
            cv_scores_all.extend(cv_scores)

            logger.info(f"  {axis}: CAV trained, R^2={r2_scores[axis]:.3f}, CV acc={cv_scores.mean():.3f}")

        cv_scores_array = np.array(cv_scores_all) if cv_scores_all else np.array([0.5])
        r2_overall = np.mean(list(r2_scores.values()))

        self.metrics = ProjectionMetrics(
            r2_agency=r2_scores.get("agency", 0.0),
            r2_fairness=r2_scores.get("fairness", 0.0),
            r2_belonging=r2_scores.get("belonging", 0.0),
            r2_overall=float(r2_overall),
            cv_score_mean=float(cv_scores_array.mean()),
            cv_score_std=float(cv_scores_array.std()),
            n_examples=n_samples
        )

        self.is_trained = True
        logger.info(f"CAV Projection trained: R^2={r2_overall:.3f}")

        return self.metrics

    def project(self, embedding: np.ndarray) -> Vector3:
        """
        Project embedding using CAV distances.

        Projects onto each CAV direction to get axis values.
        """
        if not self.is_trained:
            raise ValueError("Projection not trained. Call train() first.")

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Validate dimension
        if self.embedding_dim is not None and embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[1]}"
            )

        # Scale
        if self.scaler is not None:
            embedding = self.scaler.transform(embedding)

        # Project onto each CAV direction
        values = {}
        for axis in self.axis_names:
            cav = self.cav_directions[axis]
            # Distance along CAV direction
            projection = float(np.dot(embedding[0], cav))
            values[axis] = projection

        return Vector3(
            agency=values["agency"],
            fairness=values["fairness"],
            belonging=values["belonging"]
        )

    def project_batch(self, embeddings: np.ndarray) -> List[Vector3]:
        """Project multiple embeddings."""
        return [self.project(emb) for emb in embeddings]

    def get_cav_directions(self) -> Dict[str, np.ndarray]:
        """Get the learned CAV directions."""
        if not self.is_trained:
            raise ValueError("Projection not trained.")
        return self.cav_directions.copy()

    def save(self, path: Path):
        """Save the trained CAV projection."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained projection.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save classifiers
        for axis, clf in self.classifiers.items():
            if clf is not None:
                joblib.dump(clf, path / f"clf_{axis}.joblib")

        # Save CAV directions
        np.savez(path / "cav_directions.npz", **self.cav_directions)

        # Save scaler
        if self.scaler is not None:
            joblib.dump(self.scaler, path / "scaler.joblib")

        # Save metadata
        metadata = {
            "embedding_dim": self.embedding_dim,
            "classifier_type": self.classifier_type,
            "threshold": self.threshold,
            "method": "cav",
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "has_scaler": self.scaler is not None
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"CAV Projection saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "ConceptActivationVectors":
        """Load a saved CAV projection."""
        path = Path(path)

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        proj = cls(
            embedding_dim=metadata["embedding_dim"],
            classifier_type=metadata.get("classifier_type", "svm")
        )
        proj.threshold = metadata.get("threshold", 0.0)

        # Load classifiers
        for axis in proj.axis_names:
            clf_path = path / f"clf_{axis}.joblib"
            if clf_path.exists():
                proj.classifiers[axis] = joblib.load(clf_path)
            else:
                proj.classifiers[axis] = None

        # Load CAV directions
        cav_data = np.load(path / "cav_directions.npz")
        for axis in proj.axis_names:
            if axis in cav_data:
                proj.cav_directions[axis] = cav_data[axis]

        if metadata.get("has_scaler"):
            proj.scaler = joblib.load(path / "scaler.joblib")

        if metadata.get("metrics"):
            proj.metrics = ProjectionMetrics.from_dict(metadata["metrics"])

        proj.is_trained = True
        logger.info(f"CAV Projection loaded from {path}")
        return proj


# Type alias for any projection method
ProjectionMethod = Union[ProjectionHead, GaussianProcessProjection, NeuralProbeProjection, ConceptActivationVectors]


class ProjectionTrainer:
    """
    Manages training data and trains projection heads.

    This is the main interface for building and refining projections.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("./projection_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.examples: List[TrainingExample] = []
        self._load_examples()

    def _load_examples(self):
        """Load saved examples from disk."""
        examples_file = self.data_dir / "examples.json"
        if examples_file.exists():
            with open(examples_file) as f:
                data = json.load(f)
                self.examples = [
                    TrainingExample(**ex) for ex in data
                ]
            logger.info(f"Loaded {len(self.examples)} training examples")

    def _save_examples(self):
        """Save examples to disk."""
        examples_file = self.data_dir / "examples.json"
        with open(examples_file, "w") as f:
            json.dump([ex.to_dict() for ex in self.examples], f, indent=2)

    def add_example(
        self,
        text: str,
        agency: float,
        fairness: float,
        belonging: float,
        source: str = "manual"
    ):
        """Add a labeled example."""
        example = TrainingExample(
            text=text,
            agency=agency,
            fairness=fairness,
            belonging=belonging,
            source=source
        )
        self.examples.append(example)
        self._save_examples()
        logger.info(f"Added example: '{text[:50]}...' -> ({agency}, {fairness}, {belonging})")

    def add_examples_batch(self, examples: List[Dict]):
        """Add multiple examples at once."""
        for ex in examples:
            self.examples.append(TrainingExample(**ex))
        self._save_examples()
        logger.info(f"Added {len(examples)} examples")

    def remove_example(self, index: int):
        """Remove an example by index."""
        if 0 <= index < len(self.examples):
            removed = self.examples.pop(index)
            self._save_examples()
            logger.info(f"Removed example: '{removed.text[:50]}...'")

    def get_examples(self) -> List[TrainingExample]:
        """Get all training examples."""
        return self.examples

    def train(
        self,
        embedding_extractor,
        model_id: str,
        layer: int = -1,
        alpha: Optional[float] = None,
        auto_tune_alpha: bool = True,
        enforce_minimum_samples: bool = True
    ) -> Tuple[ProjectionHead, ProjectionMetrics]:
        """
        Train a projection head on current examples with scientific rigor.

        Args:
            embedding_extractor: EmbeddingExtractor instance
            model_id: Model to use for embeddings
            layer: Which layer to extract from
            alpha: Ridge regularization (None for auto-tuning)
            auto_tune_alpha: Whether to automatically tune regularization
            enforce_minimum_samples: Whether to warn about insufficient data

        Returns:
            Trained ProjectionHead and metrics with quality warnings
        """
        if len(self.examples) < 10:
            raise ValueError(f"Need at least 10 examples, have {len(self.examples)}")

        # Embed all examples
        texts = [ex.text for ex in self.examples]
        results = embedding_extractor.extract(texts, model_id, layer=layer)
        embeddings = np.array([r.embedding for r in results])

        # Build target array
        targets = np.array([
            [ex.agency, ex.fairness, ex.belonging]
            for ex in self.examples
        ])

        # Train with new scientific rigor
        projection = ProjectionHead()
        metrics = projection.train(
            embeddings,
            targets,
            alpha=alpha,
            auto_tune_alpha=auto_tune_alpha,
            enforce_minimum_samples=enforce_minimum_samples
        )

        # Save
        projection.save(self.data_dir / "current_projection")

        return projection, metrics

    def load_projection(self) -> Optional[ProjectionHead]:
        """Load the most recently trained projection."""
        proj_path = self.data_dir / "current_projection"
        if proj_path.exists():
            return ProjectionHead.load(proj_path)
        return None

    def train_projection(
        self,
        embedding_extractor,
        model_id: str,
        method: str = 'ridge',
        layer: int = -1,
        **kwargs
    ) -> Tuple[ProjectionMethod, ProjectionMetrics]:
        """
        Train a projection using the specified method.

        Args:
            embedding_extractor: EmbeddingExtractor instance
            model_id: Model to use for embeddings
            method: Projection method: 'ridge', 'gp', 'neural', or 'cav'
            layer: Which layer to extract from
            **kwargs: Method-specific parameters:
                - ridge: alpha, auto_tune_alpha, enforce_minimum_samples
                - gp: length_scale_bounds, noise_level, n_restarts
                - neural: hidden_dim, dropout, epochs, batch_size, learning_rate, patience
                - cav: classifier_type, threshold, svm_c, logistic_c

        Returns:
            Trained projection and metrics
        """
        valid_methods = ['ridge', 'gp', 'neural', 'cav']
        if method not in valid_methods:
            raise ValueError(f"Unknown method: {method}. Valid: {valid_methods}")

        min_examples = 10
        if len(self.examples) < min_examples:
            raise ValueError(f"Need at least {min_examples} examples, have {len(self.examples)}")

        # Embed all examples
        texts = [ex.text for ex in self.examples]
        results = embedding_extractor.extract(texts, model_id, layer=layer)
        embeddings = np.array([r.embedding for r in results])

        # Build target array
        targets = np.array([
            [ex.agency, ex.fairness, ex.belonging]
            for ex in self.examples
        ])

        # Train based on method
        if method == 'ridge':
            projection = ProjectionHead()
            metrics = projection.train(
                embeddings,
                targets,
                alpha=kwargs.get('alpha'),
                auto_tune_alpha=kwargs.get('auto_tune_alpha', True),
                enforce_minimum_samples=kwargs.get('enforce_minimum_samples', True)
            )
            save_path = self.data_dir / "current_projection"

        elif method == 'gp':
            projection = GaussianProcessProjection()
            metrics = projection.train(
                embeddings,
                targets,
                length_scale_bounds=kwargs.get('length_scale_bounds', (1e-2, 1e3)),
                noise_level=kwargs.get('noise_level', 0.1),
                n_restarts=kwargs.get('n_restarts', 5)
            )
            save_path = self.data_dir / "gp_projection"

        elif method == 'neural':
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for neural projection")
            projection = NeuralProbeProjection(
                hidden_dim=kwargs.get('hidden_dim', 128),
                dropout=kwargs.get('dropout', 0.2)
            )
            metrics = projection.train(
                embeddings,
                targets,
                epochs=kwargs.get('epochs', 200),
                batch_size=kwargs.get('batch_size', 32),
                learning_rate=kwargs.get('learning_rate', 0.001),
                patience=kwargs.get('patience', 20)
            )
            save_path = self.data_dir / "neural_projection"

        elif method == 'cav':
            projection = ConceptActivationVectors(
                classifier_type=kwargs.get('classifier_type', 'svm')
            )
            metrics = projection.train(
                embeddings,
                targets,
                threshold=kwargs.get('threshold', 0.0),
                svm_c=kwargs.get('svm_c', 1.0),
                logistic_c=kwargs.get('logistic_c', 1.0)
            )
            save_path = self.data_dir / "cav_projection"

        # Save projection
        projection.save(save_path)

        # Store method type in metadata for the saved projection
        metadata_path = save_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            metadata['method'] = method
            metadata['model_id'] = model_id
            metadata['layer'] = layer
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Trained {method} projection with {len(self.examples)} examples")
        return projection, metrics

    def compare_methods(
        self,
        embedding_extractor,
        model_id: str,
        layer: int = -1,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Train all methods and return comparative metrics.

        Args:
            embedding_extractor: EmbeddingExtractor instance
            model_id: Model to use for embeddings
            layer: Which layer to extract from
            methods: List of methods to compare (default: all available)

        Returns:
            Dictionary of method -> {metrics, trained_projection}
        """
        if methods is None:
            methods = ['ridge', 'gp', 'cav']
            if TORCH_AVAILABLE:
                methods.append('neural')

        results = {}

        for method in methods:
            try:
                logger.info(f"Training {method} projection...")
                projection, metrics = self.train_projection(
                    embedding_extractor,
                    model_id,
                    method=method,
                    layer=layer
                )
                results[method] = {
                    'metrics': metrics.to_dict(),
                    'success': True,
                    'projection': projection
                }
            except Exception as e:
                logger.error(f"Failed to train {method}: {e}")
                results[method] = {
                    'success': False,
                    'error': str(e)
                }

        # Compute comparison summary
        successful = {k: v for k, v in results.items() if v.get('success')}
        if successful:
            # Rank by CV score
            ranked = sorted(
                successful.items(),
                key=lambda x: x[1]['metrics'].get('cv_score_mean', 0),
                reverse=True
            )
            results['_summary'] = {
                'best_method': ranked[0][0] if ranked else None,
                'ranking': [m for m, _ in ranked],
                'cv_scores': {m: v['metrics'].get('cv_score_mean', 0) for m, v in successful.items()},
                'r2_scores': {m: v['metrics'].get('r2_overall', 0) for m, v in successful.items()}
            }

        return results

    def load_projection_by_method(self, method: str) -> Optional[ProjectionMethod]:
        """Load a projection by its method type."""
        method_paths = {
            'ridge': self.data_dir / "current_projection",
            'gp': self.data_dir / "gp_projection",
            'neural': self.data_dir / "neural_projection",
            'cav': self.data_dir / "cav_projection"
        }

        path = method_paths.get(method)
        if path is None or not path.exists():
            return None

        loaders = {
            'ridge': ProjectionHead.load,
            'gp': GaussianProcessProjection.load,
            'neural': NeuralProbeProjection.load if TORCH_AVAILABLE else None,
            'cav': ConceptActivationVectors.load
        }

        loader = loaders.get(method)
        if loader is None:
            return None

        try:
            return loader(path)
        except Exception as e:
            logger.error(f"Failed to load {method} projection: {e}")
            return None

    def train_ensemble(
        self,
        embedding_extractor,
        model_id: str,
        layer: int = -1,
        alphas: Optional[List[float]] = None,
        n_bootstrap: int = 5
    ) -> Tuple["EnsembleProjection", Dict]:
        """
        Train an ensemble projection for uncertainty quantification.

        Creates 25 models (5 alphas x 5 bootstrap samples) for robust
        uncertainty estimates.

        Args:
            embedding_extractor: EmbeddingExtractor instance
            model_id: Model to use for embeddings
            layer: Which layer to extract from
            alphas: List of regularization strengths (default: [0.01, 0.1, 1.0, 10.0, 100.0])
            n_bootstrap: Number of bootstrap samples per alpha

        Returns:
            Tuple of (trained EnsembleProjection, training metrics dict)
        """
        from .ensemble_projection import EnsembleProjection

        min_examples = 10
        if len(self.examples) < min_examples:
            raise ValueError(f"Need at least {min_examples} examples, have {len(self.examples)}")

        # Embed all examples
        texts = [ex.text for ex in self.examples]
        results = embedding_extractor.extract(texts, model_id, layer=layer)
        embeddings = np.array([r.embedding for r in results])

        # Build target array
        targets = np.array([
            [ex.agency, ex.fairness, ex.belonging]
            for ex in self.examples
        ])

        # Create and train ensemble
        ensemble = EnsembleProjection(
            alphas=alphas,
            n_bootstrap=n_bootstrap
        )
        metrics = ensemble.train(embeddings, targets)

        # Save ensemble
        save_path = self.data_dir / "ensemble_projection"
        ensemble.save(save_path)

        # Store metadata about what model was used
        metadata_path = save_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            metadata['model_id'] = model_id
            metadata['layer'] = layer
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Trained ensemble projection with {len(self.examples)} examples")
        return ensemble, metrics

    def load_ensemble(self) -> Optional["EnsembleProjection"]:
        """
        Load a saved ensemble projection.

        Returns:
            EnsembleProjection if available, None otherwise
        """
        from .ensemble_projection import EnsembleProjection

        ensemble_path = self.data_dir / "ensemble_projection"
        if not ensemble_path.exists():
            return None

        try:
            return EnsembleProjection.load(ensemble_path)
        except Exception as e:
            logger.error(f"Failed to load ensemble projection: {e}")
            return None
