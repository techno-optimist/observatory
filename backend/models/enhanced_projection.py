"""
Enhanced Projection

Advanced projection features:
- Uncertainty quantification (confidence intervals)
- Layer-wise analysis
- Contrastive explanations
- User-defined axis framework
- Multi-model ensemble
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import logging

from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class UncertainProjection:
    """A projection with uncertainty estimates."""
    agency: float
    fairness: float
    belonging: float

    # Uncertainty (standard deviation)
    agency_std: float
    fairness_std: float
    belonging_std: float

    # Confidence intervals (95%)
    agency_ci: Tuple[float, float]
    fairness_ci: Tuple[float, float]
    belonging_ci: Tuple[float, float]

    # Overall confidence score (0-1)
    confidence: float

    def to_dict(self) -> dict:
        return {
            "vector": {
                "agency": self.agency,
                "fairness": self.fairness,
                "belonging": self.belonging
            },
            "uncertainty": {
                "agency_std": self.agency_std,
                "fairness_std": self.fairness_std,
                "belonging_std": self.belonging_std
            },
            "confidence_intervals": {
                "agency": list(self.agency_ci),
                "fairness": list(self.fairness_ci),
                "belonging": list(self.belonging_ci)
            },
            "confidence": self.confidence
        }


@dataclass
class ContrastiveExplanation:
    """Explanation of why a text received certain coordinates."""
    original_text: str
    original_projection: Dict[str, float]

    # Most influential changes
    counterfactuals: List[Dict]  # {change: str, new_projection: dict, delta: dict}

    # Feature importance
    axis_explanations: Dict[str, str]

    # Similar texts in each direction
    nearest_high: Dict[str, List[str]]  # axis -> texts with higher values
    nearest_low: Dict[str, List[str]]   # axis -> texts with lower values

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LayerProjection:
    """Projection from a specific layer."""
    layer_index: int
    projection: Dict[str, float]
    confidence: float
    layer_name: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AxisDefinition:
    """A user-defined semantic axis."""
    name: str
    positive_anchor: str  # Text representing positive end
    negative_anchor: str  # Text representing negative end
    description: str
    examples: List[Dict[str, Any]] = field(default_factory=list)  # [{text, value}]

    def to_dict(self) -> dict:
        return asdict(self)


class BayesianProjectionHead:
    """
    Projection head with Bayesian uncertainty quantification.

    Uses Bayesian Ridge Regression to provide confidence intervals
    on projections.
    """

    def __init__(self, embedding_dim: Optional[int] = None):
        self.embedding_dim = embedding_dim
        self.models: Dict[str, BayesianRidge] = {}  # One per axis
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False

    def train(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        axis_names: List[str] = ["agency", "fairness", "belonging"]
    ) -> Dict:
        """
        Train Bayesian Ridge models for each axis.

        Returns training metrics including uncertainty estimates.
        """
        if embeddings.shape[0] < 5:
            raise ValueError("Need at least 5 examples")

        self.embedding_dim = embeddings.shape[1]

        # Normalize inputs
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(embeddings)

        # Train one model per axis
        metrics = {}
        for i, axis in enumerate(axis_names):
            model = BayesianRidge(compute_score=True)
            model.fit(scaled, targets[:, i])
            self.models[axis] = model

            # Compute metrics
            predictions = model.predict(scaled)
            r2 = 1 - np.sum((targets[:, i] - predictions)**2) / np.sum((targets[:, i] - targets[:, i].mean())**2)

            metrics[axis] = {
                "r2": float(r2),
                "alpha": float(model.alpha_),
                "lambda": float(model.lambda_)
            }

        self.is_trained = True
        self.axis_names = axis_names

        return {
            "n_examples": len(embeddings),
            "embedding_dim": self.embedding_dim,
            "per_axis_metrics": metrics
        }

    def project(self, embedding: np.ndarray) -> UncertainProjection:
        """
        Project embedding with uncertainty quantification.
        """
        if not self.is_trained:
            raise ValueError("Not trained")

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Validate dimension
        if embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[1]}"
            )

        # Scale
        scaled = self.scaler.transform(embedding)

        # Predict with uncertainty
        results = {}
        for axis in self.axis_names:
            model = self.models[axis]
            mean, std = model.predict(scaled, return_std=True)
            results[axis] = (float(mean[0]), float(std[0]))

        # Extract values
        agency, agency_std = results.get("agency", (0, 0.5))
        fairness, fairness_std = results.get("fairness", (0, 0.5))
        belonging, belonging_std = results.get("belonging", (0, 0.5))

        # 95% confidence intervals
        z = 1.96
        agency_ci = (agency - z * agency_std, agency + z * agency_std)
        fairness_ci = (fairness - z * fairness_std, fairness + z * fairness_std)
        belonging_ci = (belonging - z * belonging_std, belonging + z * belonging_std)

        # Overall confidence (inverse of mean uncertainty, scaled to 0-1)
        mean_std = np.mean([agency_std, fairness_std, belonging_std])
        confidence = 1.0 / (1.0 + mean_std)

        return UncertainProjection(
            agency=agency,
            fairness=fairness,
            belonging=belonging,
            agency_std=agency_std,
            fairness_std=fairness_std,
            belonging_std=belonging_std,
            agency_ci=agency_ci,
            fairness_ci=fairness_ci,
            belonging_ci=belonging_ci,
            confidence=confidence
        )


class LayerWiseAnalyzer:
    """
    Analyzes how projections change across model layers.
    """

    def __init__(self, projection_head):
        self.projection_head = projection_head

    def analyze_layers(
        self,
        embedding_extractor,
        text: str,
        model_id: str,
        layers: Optional[List[int]] = None
    ) -> List[LayerProjection]:
        """
        Extract embeddings from multiple layers and project each.
        """
        # Get multi-layer embeddings
        layer_embeddings = embedding_extractor.extract_multi_layer(
            text, model_id, layers=layers
        )

        results = []
        for layer_idx, embedding in layer_embeddings.items():
            try:
                projection = self.projection_head.project(embedding)
                results.append(LayerProjection(
                    layer_index=layer_idx,
                    projection={
                        "agency": projection.agency if hasattr(projection, 'agency') else projection.to_list()[0],
                        "fairness": projection.fairness if hasattr(projection, 'fairness') else projection.to_list()[1],
                        "belonging": projection.belonging if hasattr(projection, 'belonging') else projection.to_list()[2]
                    },
                    confidence=projection.confidence if hasattr(projection, 'confidence') else 0.5,
                    layer_name=f"Layer {layer_idx}"
                ))
            except Exception as e:
                logger.warning(f"Failed to project layer {layer_idx}: {e}")

        return results

    def find_semantic_emergence_layer(
        self,
        layer_projections: List[LayerProjection]
    ) -> Dict:
        """
        Find at which layer semantic axes emerge most strongly.

        Returns layer where variance in projections is highest,
        suggesting the layer most "relevant" to the semantic task.
        """
        if len(layer_projections) < 2:
            return {"emergence_layer": 0, "confidence": 0}

        # Compute variance across projections
        projections = np.array([
            [lp.projection["agency"], lp.projection["fairness"], lp.projection["belonging"]]
            for lp in layer_projections
        ])

        # Variance per layer (how "decisive" is this layer)
        layer_variances = np.var(projections, axis=1)

        # Find layer with highest variance
        emergence_idx = np.argmax(layer_variances)
        emergence_layer = layer_projections[emergence_idx].layer_index

        return {
            "emergence_layer": emergence_layer,
            "layer_variances": layer_variances.tolist(),
            "confidence": float(layer_variances[emergence_idx] / (layer_variances.mean() + 1e-6))
        }


class ContrastiveExplainer:
    """
    Generates contrastive explanations for projections.
    """

    def __init__(
        self,
        embedding_extractor,
        projection_head,
        training_texts: List[str],
        training_targets: np.ndarray
    ):
        self.embedding_extractor = embedding_extractor
        self.projection_head = projection_head
        self.training_texts = training_texts
        self.training_targets = training_targets

    def explain(
        self,
        text: str,
        model_id: str,
        n_counterfactuals: int = 3,
        n_nearest: int = 3
    ) -> ContrastiveExplanation:
        """
        Generate contrastive explanation for a text's projection.
        """
        # Get original projection
        embedding = self.embedding_extractor.extract(text, model_id)
        projection = self.projection_head.project(embedding.embedding)

        orig_proj = {
            "agency": projection.agency if hasattr(projection, 'agency') else projection.to_list()[0],
            "fairness": projection.fairness if hasattr(projection, 'fairness') else projection.to_list()[1],
            "belonging": projection.belonging if hasattr(projection, 'belonging') else projection.to_list()[2]
        }

        # Find nearest texts in each direction
        nearest_high = {"agency": [], "fairness": [], "belonging": []}
        nearest_low = {"agency": [], "fairness": [], "belonging": []}

        for axis_idx, axis in enumerate(["agency", "fairness", "belonging"]):
            # Sort training texts by axis value
            sorted_indices = np.argsort(self.training_targets[:, axis_idx])

            # Find texts with higher values
            current_val = orig_proj[axis]
            higher = [(self.training_texts[i], self.training_targets[i, axis_idx])
                     for i in sorted_indices if self.training_targets[i, axis_idx] > current_val]
            nearest_high[axis] = [t for t, v in higher[:n_nearest]]

            # Find texts with lower values
            lower = [(self.training_texts[i], self.training_targets[i, axis_idx])
                    for i in reversed(sorted_indices) if self.training_targets[i, axis_idx] < current_val]
            nearest_low[axis] = [t for t, v in lower[:n_nearest]]

        # Generate counterfactual explanations
        counterfactuals = []
        for axis in ["agency", "fairness", "belonging"]:
            if nearest_high[axis]:
                high_text = nearest_high[axis][0]
                counterfactuals.append({
                    "direction": f"higher_{axis}",
                    "example": high_text,
                    "explanation": f"Changing text to be more like '{high_text[:50]}...' would increase {axis}"
                })

        # Generate axis explanations
        axis_explanations = {}
        for axis in ["agency", "fairness", "belonging"]:
            val = orig_proj[axis]
            if val > 1.0:
                axis_explanations[axis] = f"Very high {axis} ({val:.2f}): text conveys strong sense of {axis}"
            elif val > 0.3:
                axis_explanations[axis] = f"Moderately positive {axis} ({val:.2f})"
            elif val < -1.0:
                axis_explanations[axis] = f"Very low {axis} ({val:.2f}): text conveys opposite of {axis}"
            elif val < -0.3:
                axis_explanations[axis] = f"Moderately negative {axis} ({val:.2f})"
            else:
                axis_explanations[axis] = f"Neutral {axis} ({val:.2f})"

        return ContrastiveExplanation(
            original_text=text,
            original_projection=orig_proj,
            counterfactuals=counterfactuals,
            axis_explanations=axis_explanations,
            nearest_high=nearest_high,
            nearest_low=nearest_low
        )


class UserDefinedAxesProjection:
    """
    Projection using user-defined semantic axes.

    Instead of training on labeled examples, users can define axes
    by specifying anchor texts for positive and negative ends.
    """

    def __init__(self):
        self.axes: Dict[str, AxisDefinition] = {}
        self.axis_directions: Dict[str, np.ndarray] = {}
        self.calibrated = False

    def add_axis(
        self,
        name: str,
        positive_anchor: str,
        negative_anchor: str,
        description: str = ""
    ) -> AxisDefinition:
        """Add a new semantic axis."""
        axis = AxisDefinition(
            name=name,
            positive_anchor=positive_anchor,
            negative_anchor=negative_anchor,
            description=description
        )
        self.axes[name] = axis
        self.calibrated = False
        return axis

    def add_example(self, axis_name: str, text: str, value: float):
        """Add a calibration example for an axis."""
        if axis_name not in self.axes:
            raise ValueError(f"Unknown axis: {axis_name}")
        self.axes[axis_name].examples.append({"text": text, "value": value})
        self.calibrated = False

    def calibrate(self, embedding_extractor, model_id: str, layer: int = -1):
        """
        Calibrate axis directions by embedding anchor texts.
        """
        for axis_name, axis in self.axes.items():
            # Embed anchors
            pos_result = embedding_extractor.extract(axis.positive_anchor, model_id, layer=layer)
            neg_result = embedding_extractor.extract(axis.negative_anchor, model_id, layer=layer)

            # Compute direction
            direction = pos_result.embedding - neg_result.embedding
            direction = direction / (np.linalg.norm(direction) + 1e-10)

            self.axis_directions[axis_name] = direction

        self.calibrated = True
        logger.info(f"Calibrated {len(self.axes)} user-defined axes")

    def project(self, embedding: np.ndarray) -> Dict[str, float]:
        """Project embedding onto user-defined axes."""
        if not self.calibrated:
            raise ValueError("Not calibrated. Call calibrate() first.")

        result = {}
        for axis_name, direction in self.axis_directions.items():
            # Project onto axis direction
            value = float(np.dot(embedding, direction))
            result[axis_name] = value

        return result

    def get_axes(self) -> List[Dict]:
        """Get all axis definitions."""
        return [axis.to_dict() for axis in self.axes.values()]

    def save(self, path: Path):
        """Save axes to disk."""
        data = {
            "axes": {name: axis.to_dict() for name, axis in self.axes.items()}
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "UserDefinedAxesProjection":
        """Load axes from disk."""
        with open(path) as f:
            data = json.load(f)

        proj = cls()
        for name, axis_data in data.get("axes", {}).items():
            proj.axes[name] = AxisDefinition(**axis_data)

        return proj


class EnsembleProjection:
    """
    Ensemble projection across multiple models.

    Averages projections from multiple embedding models to get
    more robust, model-agnostic coordinates.
    """

    def __init__(self, projection_head):
        self.projection_head = projection_head
        self.model_weights: Dict[str, float] = {}

    def project_ensemble(
        self,
        embedding_extractor,
        text: str,
        model_ids: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Project text using multiple models and average results.
        """
        if weights is None:
            weights = {m: 1.0 / len(model_ids) for m in model_ids}

        projections = []
        model_results = {}

        for model_id in model_ids:
            try:
                result = embedding_extractor.extract(text, model_id)
                proj = self.projection_head.project(result.embedding)

                proj_dict = {
                    "agency": proj.agency if hasattr(proj, 'agency') else proj.to_list()[0],
                    "fairness": proj.fairness if hasattr(proj, 'fairness') else proj.to_list()[1],
                    "belonging": proj.belonging if hasattr(proj, 'belonging') else proj.to_list()[2]
                }

                projections.append((model_id, proj_dict, weights.get(model_id, 1.0)))
                model_results[model_id] = proj_dict

            except Exception as e:
                logger.warning(f"Failed to project with {model_id}: {e}")

        if not projections:
            raise ValueError("All models failed")

        # Weighted average
        total_weight = sum(w for _, _, w in projections)
        ensemble = {
            "agency": sum(p["agency"] * w for _, p, w in projections) / total_weight,
            "fairness": sum(p["fairness"] * w for _, p, w in projections) / total_weight,
            "belonging": sum(p["belonging"] * w for _, p, w in projections) / total_weight
        }

        # Compute disagreement (standard deviation across models)
        if len(projections) > 1:
            disagreement = {
                "agency": np.std([p["agency"] for _, p, _ in projections]),
                "fairness": np.std([p["fairness"] for _, p, _ in projections]),
                "belonging": np.std([p["belonging"] for _, p, _ in projections])
            }
        else:
            disagreement = {"agency": 0, "fairness": 0, "belonging": 0}

        return {
            "ensemble": ensemble,
            "per_model": model_results,
            "disagreement": disagreement,
            "models_used": [m for m, _, _ in projections]
        }
