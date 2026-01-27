"""
Ensemble-based Uncertainty Quantification for Projections

This module implements an ensemble of ridge regression models for robust
uncertainty quantification in cultural manifold projections.

The ensemble consists of 25 models:
- 5 different regularization strengths (alpha = 0.01, 0.1, 1.0, 10.0, 100.0)
- 5 bootstrap samples of training data per alpha

For inference, all 25 models are run and aggregated to produce:
- Mean projection coordinates
- Standard deviation per axis
- 95% confidence intervals per axis
- Overall confidence score

This approach provides well-calibrated uncertainty estimates that capture
both model uncertainty (different alphas) and data uncertainty (bootstrap).
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import joblib

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Import the existing dataclasses for compatibility
from .projection import Vector3, ProjectionWithUncertainty, ProjectionMetrics

logger = logging.getLogger(__name__)


# Configuration constants
DEFAULT_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
N_BOOTSTRAP_SAMPLES = 5
TOTAL_ENSEMBLE_SIZE = len(DEFAULT_ALPHAS) * N_BOOTSTRAP_SAMPLES  # 25 models


@dataclass
class EnsembleMember:
    """A single member of the ensemble."""
    model: Ridge
    alpha: float
    bootstrap_index: int
    train_r2: float


class EnsembleProjection:
    """
    Ensemble-based projection with uncertainty quantification.

    Trains multiple ridge regression models with different regularization
    strengths and bootstrap samples to provide robust uncertainty estimates.

    Attributes:
        embedding_dim: Dimension of input embeddings
        alphas: List of regularization strengths
        n_bootstrap: Number of bootstrap samples per alpha
        ensemble: List of trained ensemble members
        scaler: StandardScaler for input normalization
        is_trained: Whether the ensemble has been trained
        metrics: Training metrics

    Example:
        >>> ensemble = EnsembleProjection()
        >>> ensemble.train(embeddings, labels)
        >>> result = ensemble.project_with_uncertainty(new_embedding)
        >>> print(f"Agency: {result.coords.agency:.2f} +/- {result.std_per_axis.agency:.2f}")
    """

    def __init__(
        self,
        alphas: Optional[List[float]] = None,
        n_bootstrap: int = N_BOOTSTRAP_SAMPLES,
        embedding_dim: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Initialize the ensemble projection.

        Args:
            alphas: List of regularization strengths. Default: [0.01, 0.1, 1.0, 10.0, 100.0]
            n_bootstrap: Number of bootstrap samples per alpha. Default: 5
            embedding_dim: Expected embedding dimension (optional, inferred from training)
            random_state: Random seed for reproducibility
        """
        self.alphas = alphas if alphas is not None else DEFAULT_ALPHAS
        self.n_bootstrap = n_bootstrap
        self.embedding_dim = embedding_dim
        self.random_state = random_state

        self.ensemble: List[EnsembleMember] = []
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        self.metrics: Optional[Dict] = None

        # Expected ensemble size
        self.ensemble_size = len(self.alphas) * self.n_bootstrap

        logger.info(
            f"EnsembleProjection initialized: {len(self.alphas)} alphas x "
            f"{self.n_bootstrap} bootstraps = {self.ensemble_size} models"
        )

    def train(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        normalize_inputs: bool = True,
        bootstrap_fraction: float = 1.0
    ) -> Dict:
        """
        Train the ensemble on labeled examples.

        Args:
            embeddings: (N, D) array of embeddings
            labels: (N, 3) array of [agency, fairness, belonging] labels
            normalize_inputs: Whether to standardize embeddings before training
            bootstrap_fraction: Fraction of data to sample in each bootstrap
                               (1.0 means sample N examples with replacement)

        Returns:
            Dictionary containing training metrics:
            - n_examples: Number of training examples
            - n_models: Number of models in ensemble
            - per_model_r2: R^2 scores for each model
            - mean_r2: Mean R^2 across ensemble
            - std_r2: Standard deviation of R^2

        Raises:
            ValueError: If input dimensions don't match or insufficient examples
        """
        # Validate inputs
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Embeddings and labels must have same number of examples. "
                f"Got {embeddings.shape[0]} embeddings and {labels.shape[0]} labels."
            )

        if labels.shape[1] != 3:
            raise ValueError(
                f"Labels must have 3 columns (agency, fairness, belonging). "
                f"Got {labels.shape[1]} columns."
            )

        n_samples = embeddings.shape[0]
        if n_samples < 10:
            raise ValueError(f"Need at least 10 examples to train, got {n_samples}")

        self.embedding_dim = embeddings.shape[1]
        logger.info(f"Training ensemble on {n_samples} examples with {self.embedding_dim}-dim embeddings")

        # Normalize inputs
        if normalize_inputs:
            self.scaler = StandardScaler()
            embeddings_scaled = self.scaler.fit_transform(embeddings)
        else:
            self.scaler = None
            embeddings_scaled = embeddings

        # Clear any existing ensemble
        self.ensemble = []

        # Set random state for reproducibility
        rng = np.random.RandomState(self.random_state)

        # Train ensemble members
        n_bootstrap_samples = int(n_samples * bootstrap_fraction)
        per_model_r2 = []

        for alpha_idx, alpha in enumerate(self.alphas):
            for bootstrap_idx in range(self.n_bootstrap):
                # Generate bootstrap sample
                bootstrap_indices = rng.choice(
                    n_samples,
                    size=n_bootstrap_samples,
                    replace=True
                )

                X_boot = embeddings_scaled[bootstrap_indices]
                y_boot = labels[bootstrap_indices]

                # Train Ridge model
                model = Ridge(alpha=alpha, random_state=self.random_state)
                model.fit(X_boot, y_boot)

                # Compute training R^2 on bootstrap sample
                train_predictions = model.predict(X_boot)
                ss_res = np.sum((y_boot - train_predictions) ** 2)
                ss_tot = np.sum((y_boot - y_boot.mean(axis=0)) ** 2)
                train_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                # Store ensemble member
                member = EnsembleMember(
                    model=model,
                    alpha=alpha,
                    bootstrap_index=bootstrap_idx,
                    train_r2=float(train_r2)
                )
                self.ensemble.append(member)
                per_model_r2.append(train_r2)

                logger.debug(
                    f"Trained model {len(self.ensemble)}/{self.ensemble_size}: "
                    f"alpha={alpha}, bootstrap={bootstrap_idx}, R^2={train_r2:.3f}"
                )

        self.is_trained = True

        # Compute ensemble metrics
        per_model_r2 = np.array(per_model_r2)

        # Compute validation R^2 on full dataset (not bootstrap)
        all_predictions = self._predict_all(embeddings_scaled)
        mean_predictions = np.mean(all_predictions, axis=0)

        ss_res = np.sum((labels - mean_predictions) ** 2)
        ss_tot = np.sum((labels - labels.mean(axis=0)) ** 2)
        ensemble_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Per-axis R^2
        r2_per_axis = {}
        for i, axis in enumerate(["agency", "fairness", "belonging"]):
            ss_res_axis = np.sum((labels[:, i] - mean_predictions[:, i]) ** 2)
            ss_tot_axis = np.sum((labels[:, i] - labels[:, i].mean()) ** 2)
            r2_per_axis[axis] = float(1 - (ss_res_axis / ss_tot_axis)) if ss_tot_axis > 0 else 0.0

        self.metrics = {
            "n_examples": n_samples,
            "embedding_dim": self.embedding_dim,
            "n_models": len(self.ensemble),
            "alphas": self.alphas,
            "n_bootstrap": self.n_bootstrap,
            "per_model_r2": per_model_r2.tolist(),
            "mean_model_r2": float(per_model_r2.mean()),
            "std_model_r2": float(per_model_r2.std()),
            "ensemble_r2": float(ensemble_r2),
            "r2_per_axis": r2_per_axis,
            "normalized": normalize_inputs
        }

        logger.info(
            f"Ensemble training complete: {len(self.ensemble)} models, "
            f"mean R^2={per_model_r2.mean():.3f} (+/- {per_model_r2.std():.3f}), "
            f"ensemble R^2={ensemble_r2:.3f}"
        )

        return self.metrics

    def _predict_all(self, embeddings_scaled: np.ndarray) -> np.ndarray:
        """
        Get predictions from all ensemble members.

        Args:
            embeddings_scaled: Scaled embeddings (N, D)

        Returns:
            Array of predictions (M, N, 3) where M is ensemble size
        """
        predictions = []
        for member in self.ensemble:
            pred = member.model.predict(embeddings_scaled)
            predictions.append(pred)
        return np.array(predictions)

    def project_with_uncertainty(
        self,
        embedding: np.ndarray
    ) -> ProjectionWithUncertainty:
        """
        Project a single embedding with uncertainty quantification.

        Runs the embedding through all 25 ensemble models and aggregates
        results to compute mean, standard deviation, and confidence intervals.

        Args:
            embedding: (D,) or (1, D) array representing the embedding

        Returns:
            ProjectionWithUncertainty containing:
            - coords: Mean projection as Vector3
            - std_per_axis: Standard deviation per axis as Vector3
            - confidence_intervals: 95% CI per axis as dict
            - overall_confidence: Confidence score (0-1)
            - method: "ensemble_ridge"

        Raises:
            ValueError: If ensemble not trained or dimension mismatch
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call train() first.")

        # Handle 1D input
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Validate dimension
        if embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embedding.shape[1]}. Please retrain the ensemble with "
                f"the current embedding model."
            )

        # Scale input
        if self.scaler is not None:
            embedding_scaled = self.scaler.transform(embedding)
        else:
            embedding_scaled = embedding

        # Get predictions from all ensemble members
        predictions = self._predict_all(embedding_scaled)  # (M, 1, 3)
        predictions = predictions[:, 0, :]  # (M, 3)

        # Compute mean and standard deviation
        mean_pred = np.mean(predictions, axis=0)  # (3,)
        std_pred = np.std(predictions, axis=0)    # (3,)

        # Compute 95% confidence intervals (mean +/- 1.96 * std)
        z = 1.96
        confidence_intervals = {
            "agency": (float(mean_pred[0] - z * std_pred[0]),
                      float(mean_pred[0] + z * std_pred[0])),
            "fairness": (float(mean_pred[1] - z * std_pred[1]),
                        float(mean_pred[1] + z * std_pred[1])),
            "belonging": (float(mean_pred[2] - z * std_pred[2]),
                         float(mean_pred[2] + z * std_pred[2]))
        }

        # Create result using existing dataclasses
        result = ProjectionWithUncertainty(
            coords=Vector3(
                agency=float(mean_pred[0]),
                fairness=float(mean_pred[1]),
                belonging=float(mean_pred[2])
            ),
            std_per_axis=Vector3(
                agency=float(std_pred[0]),
                fairness=float(std_pred[1]),
                belonging=float(std_pred[2])
            ),
            confidence_intervals=confidence_intervals,
            method="ensemble_ridge"
        )

        return result

    def project_batch_with_uncertainty(
        self,
        embeddings: np.ndarray
    ) -> List[ProjectionWithUncertainty]:
        """
        Project multiple embeddings with uncertainty quantification.

        Args:
            embeddings: (N, D) array of embeddings

        Returns:
            List of ProjectionWithUncertainty, one per input embedding
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call train() first.")

        # Validate dimension
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        # Scale inputs
        if self.scaler is not None:
            embeddings_scaled = self.scaler.transform(embeddings)
        else:
            embeddings_scaled = embeddings

        # Get all predictions at once for efficiency
        all_predictions = self._predict_all(embeddings_scaled)  # (M, N, 3)

        # Compute statistics
        mean_preds = np.mean(all_predictions, axis=0)  # (N, 3)
        std_preds = np.std(all_predictions, axis=0)    # (N, 3)

        # Build results
        z = 1.96
        results = []
        for i in range(embeddings.shape[0]):
            confidence_intervals = {
                "agency": (float(mean_preds[i, 0] - z * std_preds[i, 0]),
                          float(mean_preds[i, 0] + z * std_preds[i, 0])),
                "fairness": (float(mean_preds[i, 1] - z * std_preds[i, 1]),
                            float(mean_preds[i, 1] + z * std_preds[i, 1])),
                "belonging": (float(mean_preds[i, 2] - z * std_preds[i, 2]),
                             float(mean_preds[i, 2] + z * std_preds[i, 2]))
            }

            result = ProjectionWithUncertainty(
                coords=Vector3(
                    agency=float(mean_preds[i, 0]),
                    fairness=float(mean_preds[i, 1]),
                    belonging=float(mean_preds[i, 2])
                ),
                std_per_axis=Vector3(
                    agency=float(std_preds[i, 0]),
                    fairness=float(std_preds[i, 1]),
                    belonging=float(std_preds[i, 2])
                ),
                confidence_intervals=confidence_intervals,
                method="ensemble_ridge"
            )
            results.append(result)

        return results

    def get_ensemble_statistics(self) -> Dict:
        """
        Get detailed statistics about the ensemble.

        Returns:
            Dictionary containing:
            - per_alpha_r2: Mean R^2 for each alpha value
            - per_bootstrap_r2: Mean R^2 for each bootstrap sample
            - weight_statistics: Statistics about model weights
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained.")

        # Group by alpha
        per_alpha_r2 = {}
        for alpha in self.alphas:
            members = [m for m in self.ensemble if m.alpha == alpha]
            r2_values = [m.train_r2 for m in members]
            per_alpha_r2[str(alpha)] = {
                "mean": float(np.mean(r2_values)),
                "std": float(np.std(r2_values)),
                "min": float(np.min(r2_values)),
                "max": float(np.max(r2_values))
            }

        # Group by bootstrap index
        per_bootstrap_r2 = {}
        for b_idx in range(self.n_bootstrap):
            members = [m for m in self.ensemble if m.bootstrap_index == b_idx]
            r2_values = [m.train_r2 for m in members]
            per_bootstrap_r2[str(b_idx)] = {
                "mean": float(np.mean(r2_values)),
                "std": float(np.std(r2_values))
            }

        # Weight statistics (model coefficients)
        all_weights = []
        for member in self.ensemble:
            all_weights.append(member.model.coef_.flatten())
        all_weights = np.array(all_weights)

        weight_stats = {
            "mean_magnitude": float(np.mean(np.abs(all_weights))),
            "std_magnitude": float(np.std(np.abs(all_weights))),
            "weight_diversity": float(np.std(all_weights, axis=0).mean())
        }

        return {
            "per_alpha_r2": per_alpha_r2,
            "per_bootstrap_r2": per_bootstrap_r2,
            "weight_statistics": weight_stats,
            "total_models": len(self.ensemble)
        }

    def save(self, path: Path):
        """
        Save the trained ensemble to disk.

        Args:
            path: Directory path where ensemble will be saved

        Creates:
            - path/metadata.json: Configuration and metrics
            - path/scaler.joblib: Input scaler (if used)
            - path/models/model_*.joblib: Individual ensemble members
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained ensemble.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Create models subdirectory
        models_dir = path / "models"
        models_dir.mkdir(exist_ok=True)

        # Save scaler
        if self.scaler is not None:
            joblib.dump(self.scaler, path / "scaler.joblib")

        # Save each ensemble member
        for i, member in enumerate(self.ensemble):
            member_path = models_dir / f"model_{i:03d}.joblib"
            member_data = {
                "model": member.model,
                "alpha": member.alpha,
                "bootstrap_index": member.bootstrap_index,
                "train_r2": member.train_r2
            }
            joblib.dump(member_data, member_path)

        # Save metadata
        metadata = {
            "embedding_dim": self.embedding_dim,
            "alphas": self.alphas,
            "n_bootstrap": self.n_bootstrap,
            "ensemble_size": len(self.ensemble),
            "random_state": self.random_state,
            "has_scaler": self.scaler is not None,
            "metrics": self.metrics
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Ensemble saved to {path} ({len(self.ensemble)} models)")

    @classmethod
    def load(cls, path: Path) -> "EnsembleProjection":
        """
        Load a saved ensemble from disk.

        Args:
            path: Directory path containing saved ensemble

        Returns:
            Loaded EnsembleProjection instance

        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If saved data is corrupted
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Ensemble path not found: {path}")

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        # Create instance
        ensemble = cls(
            alphas=metadata["alphas"],
            n_bootstrap=metadata["n_bootstrap"],
            embedding_dim=metadata["embedding_dim"],
            random_state=metadata.get("random_state", 42)
        )

        # Load scaler
        if metadata.get("has_scaler"):
            ensemble.scaler = joblib.load(path / "scaler.joblib")

        # Load ensemble members
        models_dir = path / "models"
        ensemble.ensemble = []

        for i in range(metadata["ensemble_size"]):
            member_path = models_dir / f"model_{i:03d}.joblib"
            member_data = joblib.load(member_path)

            member = EnsembleMember(
                model=member_data["model"],
                alpha=member_data["alpha"],
                bootstrap_index=member_data["bootstrap_index"],
                train_r2=member_data["train_r2"]
            )
            ensemble.ensemble.append(member)

        ensemble.metrics = metadata.get("metrics")
        ensemble.is_trained = True

        logger.info(f"Ensemble loaded from {path} ({len(ensemble.ensemble)} models)")

        return ensemble


# =============================================================================
# Testing
# =============================================================================

def _run_tests():
    """
    Run tests to verify the EnsembleProjection implementation.

    Creates synthetic data and tests:
    1. Training with various configurations
    2. Projection with uncertainty
    3. Save/load functionality
    4. Edge cases
    """
    import tempfile
    import shutil

    print("=" * 60)
    print("Testing EnsembleProjection")
    print("=" * 60)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create synthetic training data
    np.random.seed(42)
    n_samples = 100
    embedding_dim = 64

    # Generate embeddings (random)
    embeddings = np.random.randn(n_samples, embedding_dim)

    # Generate labels with some structure
    # Labels are a linear function of first few embedding dimensions plus noise
    true_weights = np.random.randn(10, 3) * 0.5
    labels = embeddings[:, :10] @ true_weights + np.random.randn(n_samples, 3) * 0.2

    # Clip labels to reasonable range
    labels = np.clip(labels, -2, 2)

    print(f"\nSynthetic data: {n_samples} samples, {embedding_dim}-dim embeddings")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels range: [{labels.min():.2f}, {labels.max():.2f}]")

    # Test 1: Basic training
    print("\n" + "-" * 40)
    print("Test 1: Basic training")
    print("-" * 40)

    ensemble = EnsembleProjection()
    metrics = ensemble.train(embeddings, labels)

    print(f"Trained {metrics['n_models']} models")
    print(f"Mean R^2: {metrics['mean_model_r2']:.3f} (+/- {metrics['std_model_r2']:.3f})")
    print(f"Ensemble R^2: {metrics['ensemble_r2']:.3f}")
    print(f"Per-axis R^2: agency={metrics['r2_per_axis']['agency']:.3f}, "
          f"fairness={metrics['r2_per_axis']['fairness']:.3f}, "
          f"belonging={metrics['r2_per_axis']['belonging']:.3f}")

    assert ensemble.is_trained, "Ensemble should be trained"
    assert len(ensemble.ensemble) == 25, f"Should have 25 models, got {len(ensemble.ensemble)}"

    # Test 2: Projection with uncertainty
    print("\n" + "-" * 40)
    print("Test 2: Projection with uncertainty")
    print("-" * 40)

    test_embedding = np.random.randn(embedding_dim)
    result = ensemble.project_with_uncertainty(test_embedding)

    print(f"Projection result:")
    print(f"  Agency:    {result.coords.agency:.3f} +/- {result.std_per_axis.agency:.3f} "
          f"(95% CI: [{result.confidence_intervals['agency'][0]:.3f}, {result.confidence_intervals['agency'][1]:.3f}])")
    print(f"  Fairness:  {result.coords.fairness:.3f} +/- {result.std_per_axis.fairness:.3f} "
          f"(95% CI: [{result.confidence_intervals['fairness'][0]:.3f}, {result.confidence_intervals['fairness'][1]:.3f}])")
    print(f"  Belonging: {result.coords.belonging:.3f} +/- {result.std_per_axis.belonging:.3f} "
          f"(95% CI: [{result.confidence_intervals['belonging'][0]:.3f}, {result.confidence_intervals['belonging'][1]:.3f}])")
    print(f"  Overall confidence: {result.overall_confidence:.3f}")
    print(f"  Method: {result.method}")

    assert result.method == "ensemble_ridge", "Method should be ensemble_ridge"
    assert result.overall_confidence > 0, "Confidence should be positive"

    # Test 3: Batch projection
    print("\n" + "-" * 40)
    print("Test 3: Batch projection")
    print("-" * 40)

    batch_embeddings = np.random.randn(5, embedding_dim)
    batch_results = ensemble.project_batch_with_uncertainty(batch_embeddings)

    print(f"Batch of {len(batch_results)} projections:")
    for i, res in enumerate(batch_results):
        print(f"  [{i}] Agency={res.coords.agency:.2f}, Confidence={res.overall_confidence:.3f}")

    assert len(batch_results) == 5, "Should have 5 results"

    # Test 4: Ensemble statistics
    print("\n" + "-" * 40)
    print("Test 4: Ensemble statistics")
    print("-" * 40)

    stats = ensemble.get_ensemble_statistics()
    print(f"Per-alpha R^2:")
    for alpha, data in stats["per_alpha_r2"].items():
        print(f"  alpha={alpha}: mean={data['mean']:.3f} +/- {data['std']:.3f}")
    print(f"Weight diversity: {stats['weight_statistics']['weight_diversity']:.4f}")

    # Test 5: Save and load
    print("\n" + "-" * 40)
    print("Test 5: Save and load")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_ensemble"

        # Save
        ensemble.save(save_path)
        print(f"Saved ensemble to {save_path}")

        # Load
        loaded_ensemble = EnsembleProjection.load(save_path)
        print(f"Loaded ensemble: {len(loaded_ensemble.ensemble)} models")

        # Verify loaded ensemble produces same results
        loaded_result = loaded_ensemble.project_with_uncertainty(test_embedding)

        assert np.isclose(result.coords.agency, loaded_result.coords.agency, atol=1e-6), \
            "Loaded ensemble should produce same results"
        assert np.isclose(result.coords.fairness, loaded_result.coords.fairness, atol=1e-6), \
            "Loaded ensemble should produce same results"
        assert np.isclose(result.coords.belonging, loaded_result.coords.belonging, atol=1e-6), \
            "Loaded ensemble should produce same results"

        print("Loaded ensemble produces identical results!")

    # Test 6: ProjectionWithUncertainty compatibility
    print("\n" + "-" * 40)
    print("Test 6: ProjectionWithUncertainty compatibility")
    print("-" * 40)

    result_dict = result.to_dict()
    print(f"Result as dict:")
    print(f"  coords: {result_dict['coords']}")
    print(f"  std_per_axis: {result_dict['std_per_axis']}")
    print(f"  overall_confidence: {result_dict['overall_confidence']}")
    print(f"  method: {result_dict['method']}")

    assert "coords" in result_dict, "Should have coords"
    assert "std_per_axis" in result_dict, "Should have std_per_axis"
    assert "confidence_intervals" in result_dict, "Should have confidence_intervals"
    assert "overall_confidence" in result_dict, "Should have overall_confidence"

    # Test 7: Error handling
    print("\n" + "-" * 40)
    print("Test 7: Error handling")
    print("-" * 40)

    # Test dimension mismatch
    wrong_dim_embedding = np.random.randn(32)  # Wrong dimension
    try:
        ensemble.project_with_uncertainty(wrong_dim_embedding)
        print("ERROR: Should have raised ValueError for dimension mismatch")
    except ValueError as e:
        print(f"Correctly raised ValueError: {str(e)[:60]}...")

    # Test untrained ensemble
    new_ensemble = EnsembleProjection()
    try:
        new_ensemble.project_with_uncertainty(test_embedding)
        print("ERROR: Should have raised ValueError for untrained ensemble")
    except ValueError as e:
        print(f"Correctly raised ValueError: {str(e)[:60]}...")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
