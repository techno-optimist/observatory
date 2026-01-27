#!/usr/bin/env python3
"""
Push Observatory Capabilities

Tests and compares multiple projection methods and ensemble strategies
to find the best configuration for robust cultural narrative analysis.
"""

import sys
import json
import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model_manager, ModelType
from models.embedding import EmbeddingExtractor
from models.projection import (
    ProjectionHead,
    ProjectionTrainer,
    GaussianProcessProjection,
    NeuralProbeProjection,
    TrainingExample
)
from models.ensemble_projection import EnsembleProjection
from analysis.mode_classifier import get_mode_classifier

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ProjectionResult:
    """Result from testing a projection method."""
    method: str
    r2_overall: float
    cv_score: float
    paraphrase_spread: float
    mode_consistency: float
    details: Dict


class CapabilityPusher:
    """Push the observatory's analytical capabilities."""

    # Test paraphrases for robustness
    PARAPHRASE_TESTS = [
        {
            "concept": "collective_action",
            "variants": [
                "We can accomplish more together than alone.",
                "Together we achieve more than individually.",
                "Collective effort surpasses individual action.",
                "Working as a group yields greater results.",
                "Unity produces outcomes that solitude cannot."
            ]
        },
        {
            "concept": "personal_agency",
            "variants": [
                "I control my own destiny.",
                "My fate is in my own hands.",
                "I am the master of my future.",
                "I determine the course of my life.",
                "My choices shape my outcomes."
            ]
        }
    ]

    def __init__(self):
        self.model_manager = get_model_manager()
        self.embedding_extractor = EmbeddingExtractor(self.model_manager)
        self.data_dir = Path(__file__).parent.parent / "data" / "projections"
        self.classifier = get_mode_classifier()
        self.results: List[ProjectionResult] = []

        # Pre-load embedding model
        if not self.model_manager.is_loaded("all-MiniLM-L6-v2"):
            self.model_manager.load_model("all-MiniLM-L6-v2", ModelType.SENTENCE_TRANSFORMER)

    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray, List]:
        """Load expanded training examples and compute embeddings."""
        expanded_file = self.data_dir / "examples_expanded.json"

        with open(expanded_file) as f:
            data = json.load(f)

        examples = []
        for item in data:
            fairness_val = item.get("perceived_justice", item.get("fairness", 0.0))
            examples.append(TrainingExample(
                text=item["text"],
                agency=item["agency"],
                fairness=fairness_val,
                belonging=item["belonging"],
                source=item.get("source", "unknown")
            ))

        logger.info(f"Computing embeddings for {len(examples)} examples...")
        embeddings = []
        targets = []

        for ex in examples:
            result = self.embedding_extractor.extract(ex.text, "all-MiniLM-L6-v2")
            embeddings.append(result.embedding)
            targets.append([ex.agency, ex.fairness, ex.belonging])

        return np.array(embeddings), np.array(targets), examples

    def _extract_coords(self, result) -> List[float]:
        """Extract coordinates from projection result (handles Vector3 or ProjectionWithUncertainty)."""
        if hasattr(result, 'coords'):
            # ProjectionWithUncertainty
            return [result.coords.agency, result.coords.fairness, result.coords.belonging]
        else:
            # Vector3
            return [result.agency, result.fairness, result.belonging]

    def test_paraphrase_robustness(self, projection, model_id: str = "all-MiniLM-L6-v2") -> Tuple[float, float]:
        """Test paraphrase robustness for a projection."""
        # Ensure model is loaded
        if not self.model_manager.is_loaded(model_id):
            self.model_manager.load_model(model_id, ModelType.SENTENCE_TRANSFORMER)

        spreads = []
        consistencies = []

        for group in self.PARAPHRASE_TESTS:
            projections = []
            modes = []

            for text in group["variants"]:
                emb_result = self.embedding_extractor.extract(text, model_id)
                result = projection.project(emb_result.embedding)
                coords_arr = self._extract_coords(result)
                projections.append(coords_arr)

                mode_result = self.classifier.classify(np.array(coords_arr))
                modes.append(mode_result["primary_mode"])

            coords_array = np.array(projections)
            centroid = np.mean(coords_array, axis=0)
            distances = np.linalg.norm(coords_array - centroid, axis=1)
            spreads.append(np.max(distances))
            consistencies.append(1.0 if len(set(modes)) == 1 else 0.0)

        return np.mean(spreads), np.mean(consistencies)

    def test_ridge_projection(self, embeddings, targets) -> ProjectionResult:
        """Test standard ridge regression projection."""
        logger.info("Testing Ridge Regression projection...")

        projection = ProjectionHead()
        metrics = projection.train(embeddings, targets, auto_tune_alpha=True)

        spread, consistency = self.test_paraphrase_robustness(projection)

        result = ProjectionResult(
            method="Ridge Regression",
            r2_overall=metrics.r2_overall,
            cv_score=metrics.cv_score_mean,
            paraphrase_spread=spread,
            mode_consistency=consistency,
            details={
                "alpha": metrics.best_alpha,
                "r2_agency": metrics.r2_agency,
                "r2_fairness": metrics.r2_fairness,
                "r2_belonging": metrics.r2_belonging
            }
        )
        self.results.append(result)
        return result

    def test_gp_projection(self, embeddings, targets) -> ProjectionResult:
        """Test Gaussian Process projection with uncertainty."""
        logger.info("Testing Gaussian Process projection...")

        try:
            projection = GaussianProcessProjection()
            metrics = projection.train(embeddings, targets)

            spread, consistency = self.test_paraphrase_robustness(projection)

            result = ProjectionResult(
                method="Gaussian Process",
                r2_overall=metrics.r2_overall,
                cv_score=metrics.cv_score_mean,
                paraphrase_spread=spread,
                mode_consistency=consistency,
                details={
                    "r2_agency": metrics.r2_agency,
                    "r2_fairness": metrics.r2_fairness,
                    "r2_belonging": metrics.r2_belonging,
                    "has_uncertainty": True
                }
            )
            self.results.append(result)
            return result
        except Exception as e:
            logger.warning(f"GP projection failed: {e}")
            return None

    def test_neural_projection(self, embeddings, targets) -> ProjectionResult:
        """Test neural network projection."""
        logger.info("Testing Neural Network projection...")

        try:
            projection = NeuralProbeProjection(
                embedding_dim=embeddings.shape[1],
                hidden_dim=128,
                dropout=0.2
            )
            metrics = projection.train(
                embeddings, targets,
                epochs=100,
                batch_size=32,
                patience=10
            )

            spread, consistency = self.test_paraphrase_robustness(projection)

            result = ProjectionResult(
                method="Neural Network",
                r2_overall=metrics.r2_overall,
                cv_score=metrics.cv_score_mean,
                paraphrase_spread=spread,
                mode_consistency=consistency,
                details={
                    "r2_agency": metrics.r2_agency,
                    "r2_fairness": metrics.r2_fairness,
                    "r2_belonging": metrics.r2_belonging,
                    "hidden_dim": 128
                }
            )
            self.results.append(result)
            return result
        except Exception as e:
            logger.warning(f"Neural projection failed: {e}")
            return None

    def test_ensemble_projection(self, embeddings, targets) -> ProjectionResult:
        """Test ensemble projection with multiple alphas and bootstraps."""
        logger.info("Testing Ensemble projection (25 models)...")

        try:
            ensemble = EnsembleProjection(
                n_bootstrap=5,
                alphas=[0.01, 0.1, 1.0, 10.0, 100.0]
            )
            ensemble.train(embeddings, targets)

            # Test robustness using ensemble
            spreads = []
            consistencies = []

            for group in self.PARAPHRASE_TESTS:
                projections = []
                modes = []

                for text in group["variants"]:
                    emb_result = self.embedding_extractor.extract(text, "all-MiniLM-L6-v2")
                    proj_result = ensemble.project_with_uncertainty(emb_result.embedding)
                    coords_arr = [proj_result.coords.agency, proj_result.coords.fairness, proj_result.coords.belonging]
                    projections.append(coords_arr)

                    mode_result = self.classifier.classify(np.array(coords_arr))
                    modes.append(mode_result["primary_mode"])

                coords_array = np.array(projections)
                centroid = np.mean(coords_array, axis=0)
                distances = np.linalg.norm(coords_array - centroid, axis=1)
                spreads.append(np.max(distances))
                consistencies.append(1.0 if len(set(modes)) == 1 else 0.0)

            result = ProjectionResult(
                method="Ensemble (25 models)",
                r2_overall=ensemble.metrics["ensemble_r2"],
                cv_score=ensemble.metrics["mean_model_r2"],
                paraphrase_spread=np.mean(spreads),
                mode_consistency=np.mean(consistencies),
                details={
                    "n_models": 25,
                    "has_uncertainty": True,
                    "std_model_r2": ensemble.metrics["std_model_r2"]
                }
            )
            self.results.append(result)
            return result
        except Exception as e:
            logger.warning(f"Ensemble projection failed: {e}")
            return None

    def test_multi_model_ensemble(self, targets, examples) -> ProjectionResult:
        """Test ensemble across multiple embedding models."""
        logger.info("Testing Multi-Model Ensemble projection...")

        # Increase max cached models temporarily
        original_max = self.model_manager.max_cached_models
        self.model_manager.max_cached_models = 5

        model_ids = []
        all_models = [
            ("all-MiniLM-L6-v2", ModelType.SENTENCE_TRANSFORMER),
            ("all-mpnet-base-v2", ModelType.SENTENCE_TRANSFORMER),
            ("paraphrase-MiniLM-L6-v2", ModelType.SENTENCE_TRANSFORMER)
        ]

        # Load all models
        for model_id, model_type in all_models:
            try:
                if not self.model_manager.is_loaded(model_id):
                    self.model_manager.load_model(model_id, model_type)
                model_ids.append(model_id)
                logger.info(f"  Added model: {model_id}")
            except Exception as e:
                logger.warning(f"  Could not load {model_id}: {e}")

        if len(model_ids) < 2:
            logger.warning("Not enough models for multi-model ensemble")
            self.model_manager.max_cached_models = original_max
            return None

        # Train projections for each model
        model_projections = {}
        for model_id in model_ids:
            logger.info(f"  Training projection for {model_id}...")
            embeddings = []
            for ex in examples:
                result = self.embedding_extractor.extract(ex.text, model_id)
                embeddings.append(result.embedding)
            embeddings = np.array(embeddings)

            projection = ProjectionHead()
            projection.train(embeddings, targets, auto_tune_alpha=True)
            model_projections[model_id] = (projection, model_id)

        # Test with averaged projections
        spreads = []
        consistencies = []

        for group in self.PARAPHRASE_TESTS:
            projections = []
            modes = []

            for text in group["variants"]:
                # Average projections from all models
                all_coords = []
                for model_id, (projection, mid) in model_projections.items():
                    if not self.model_manager.is_loaded(mid):
                        self.model_manager.load_model(mid, ModelType.SENTENCE_TRANSFORMER)
                    emb_result = self.embedding_extractor.extract(text, mid)
                    coords = projection.project(emb_result.embedding)
                    all_coords.append(self._extract_coords(coords))

                avg_coords = np.mean(all_coords, axis=0)
                projections.append(avg_coords)

                mode_result = self.classifier.classify(avg_coords)
                modes.append(mode_result["primary_mode"])

            coords_array = np.array(projections)
            centroid = np.mean(coords_array, axis=0)
            distances = np.linalg.norm(coords_array - centroid, axis=1)
            spreads.append(np.max(distances))
            consistencies.append(1.0 if len(set(modes)) == 1 else 0.0)

        self.model_manager.max_cached_models = original_max

        result = ProjectionResult(
            method=f"Multi-Model Ensemble ({len(model_ids)} models)",
            r2_overall=0.0,  # N/A for ensemble
            cv_score=0.0,
            paraphrase_spread=np.mean(spreads),
            mode_consistency=np.mean(consistencies),
            details={
                "models": model_ids,
                "n_models": len(model_ids)
            }
        )
        self.results.append(result)
        return result

    def generate_report(self) -> Dict:
        """Generate capability comparison report."""
        print("\n" + "=" * 70)
        print("CAPABILITY COMPARISON REPORT")
        print("=" * 70)

        # Sort by paraphrase spread (lower is better)
        sorted_results = sorted(self.results, key=lambda r: r.paraphrase_spread)

        print("\n{:<30} {:>10} {:>10} {:>12} {:>10}".format(
            "Method", "R²", "CV", "Para.Spread", "ModeConsist"
        ))
        print("-" * 70)

        for r in sorted_results:
            print("{:<30} {:>10.3f} {:>10.3f} {:>12.3f} {:>10.1%}".format(
                r.method,
                r.r2_overall,
                r.cv_score,
                r.paraphrase_spread,
                r.mode_consistency
            ))

        print("\n" + "=" * 70)

        best = sorted_results[0]
        print(f"\nBEST METHOD: {best.method}")
        print(f"  Paraphrase Spread: {best.paraphrase_spread:.3f}")
        print(f"  Mode Consistency: {best.mode_consistency:.1%}")

        # Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "method": r.method,
                    "r2_overall": float(r.r2_overall),
                    "cv_score": float(r.cv_score),
                    "paraphrase_spread": float(r.paraphrase_spread),
                    "mode_consistency": float(r.mode_consistency),
                    "details": r.details
                }
                for r in sorted_results
            ],
            "best_method": best.method
        }

        output_path = Path(__file__).parent.parent / "data" / f"capability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")

        return report


def main():
    """Run capability tests."""
    print("=" * 70)
    print("PUSHING OBSERVATORY CAPABILITIES")
    print("=" * 70)

    pusher = CapabilityPusher()

    # Load data
    embeddings, targets, examples = pusher.load_training_data()
    logger.info(f"Loaded {len(examples)} training examples")

    # Test each projection method
    pusher.test_ridge_projection(embeddings, targets)
    pusher.test_ensemble_projection(embeddings, targets)
    pusher.test_gp_projection(embeddings, targets)
    pusher.test_neural_projection(embeddings, targets)
    pusher.test_multi_model_ensemble(targets, examples)

    # Generate report
    report = pusher.generate_report()

    return report


if __name__ == "__main__":
    main()
