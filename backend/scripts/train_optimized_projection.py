#!/usr/bin/env python3
"""
Train Optimized Projection

Creates the best projection configuration based on capability testing:
1. Primary: all-mpnet-base-v2 (best CV score: 0.612)
2. Ensemble: Average across MiniLM, MPNet, and paraphrase models
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model_manager, ModelType
from models.embedding import EmbeddingExtractor
from models.projection import ProjectionHead, ProjectionTrainer, TrainingExample
from models.ensemble_projection import EnsembleProjection

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_training_data(data_dir: Path):
    """Load expanded training examples."""
    expanded_file = data_dir / "examples_expanded.json"

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

    return examples


def train_mpnet_projection(examples, data_dir, model_manager, embedding_extractor):
    """Train projection using all-mpnet-base-v2 (best individual model)."""
    logger.info("=" * 60)
    logger.info("Training MPNet-based projection (best CV: 0.612)")
    logger.info("=" * 60)

    model_id = "all-mpnet-base-v2"

    if not model_manager.is_loaded(model_id):
        model_manager.load_model(model_id, ModelType.SENTENCE_TRANSFORMER)

    # Compute embeddings
    embeddings = []
    targets = []
    for ex in examples:
        result = embedding_extractor.extract(ex.text, model_id)
        embeddings.append(result.embedding)
        targets.append([ex.agency, ex.fairness, ex.belonging])

    embeddings = np.array(embeddings)
    targets = np.array(targets)

    # Train projection
    projection = ProjectionHead()
    metrics = projection.train(embeddings, targets, auto_tune_alpha=True)

    logger.info(f"MPNet projection metrics:")
    logger.info(f"  R² overall: {metrics.r2_overall:.4f}")
    logger.info(f"  CV score: {metrics.cv_score_mean:.4f} ± {metrics.cv_score_std:.4f}")
    logger.info(f"  Test R²: {metrics.test_r2:.4f}")

    # Save projection
    proj_path = data_dir / "mpnet_projection"
    projection.save(proj_path)
    logger.info(f"MPNet projection saved to {proj_path}")

    return projection, embeddings, targets


def train_multi_model_ensemble(examples, data_dir, model_manager, embedding_extractor):
    """Train projections for multi-model ensemble (best robustness)."""
    logger.info("=" * 60)
    logger.info("Training Multi-Model Ensemble (best robustness)")
    logger.info("=" * 60)

    model_manager.max_cached_models = 5

    model_ids = [
        ("all-MiniLM-L6-v2", ModelType.SENTENCE_TRANSFORMER),
        ("all-mpnet-base-v2", ModelType.SENTENCE_TRANSFORMER),
        ("paraphrase-MiniLM-L6-v2", ModelType.SENTENCE_TRANSFORMER),
    ]

    ensemble_config = {
        "models": [],
        "projections_dir": str(data_dir / "multi_model_ensemble"),
        "aggregation": "mean"
    }

    ensemble_dir = data_dir / "multi_model_ensemble"
    ensemble_dir.mkdir(exist_ok=True)

    targets = None

    for model_id, model_type in model_ids:
        logger.info(f"Training projection for {model_id}...")

        if not model_manager.is_loaded(model_id):
            model_manager.load_model(model_id, model_type)

        embeddings = []
        target_list = []
        for ex in examples:
            result = embedding_extractor.extract(ex.text, model_id)
            embeddings.append(result.embedding)
            target_list.append([ex.agency, ex.fairness, ex.belonging])

        embeddings = np.array(embeddings)
        targets = np.array(target_list)

        projection = ProjectionHead()
        metrics = projection.train(embeddings, targets, auto_tune_alpha=True)

        logger.info(f"  R²: {metrics.r2_overall:.3f}, CV: {metrics.cv_score_mean:.3f}")

        # Save projection for this model
        proj_path = ensemble_dir / model_id.replace("/", "_")
        projection.save(proj_path)

        ensemble_config["models"].append({
            "model_id": model_id,
            "projection_path": str(proj_path),
            "r2": metrics.r2_overall,
            "cv_score": metrics.cv_score_mean,
            "weight": 1.0 / len(model_ids)  # Equal weights
        })

    # Save ensemble configuration
    config_path = ensemble_dir / "ensemble_config.json"
    with open(config_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2)

    logger.info(f"Multi-model ensemble saved to {ensemble_dir}")
    return ensemble_config


def create_summary(data_dir):
    """Create a summary of available projections."""
    summary = {
        "available_projections": [
            {
                "name": "current_projection",
                "model": "all-MiniLM-L6-v2",
                "description": "Default ridge regression projection",
                "cv_score": 0.383,
                "path": str(data_dir / "current_projection")
            },
            {
                "name": "mpnet_projection",
                "model": "all-mpnet-base-v2",
                "description": "Best single-model projection (highest CV)",
                "cv_score": 0.612,
                "path": str(data_dir / "mpnet_projection")
            },
            {
                "name": "multi_model_ensemble",
                "models": ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
                "description": "Best paraphrase robustness (averages 3 models)",
                "path": str(data_dir / "multi_model_ensemble")
            },
            {
                "name": "ensemble_projection",
                "model": "all-MiniLM-L6-v2",
                "description": "25-model bootstrap ensemble with uncertainty",
                "path": str(data_dir / "ensemble_projection.json")
            }
        ],
        "recommended": "mpnet_projection",
        "most_robust": "multi_model_ensemble"
    }

    summary_path = data_dir / "projection_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Projection summary saved to {summary_path}")
    return summary


def main():
    logger.info("=" * 60)
    logger.info("TRAINING OPTIMIZED PROJECTIONS")
    logger.info("=" * 60)

    data_dir = Path(__file__).parent.parent / "data" / "projections"

    model_manager = get_model_manager()
    embedding_extractor = EmbeddingExtractor(model_manager)

    # Load examples
    examples = load_training_data(data_dir)
    logger.info(f"Loaded {len(examples)} training examples")

    # Train MPNet projection (best CV)
    mpnet_proj, embeddings, targets = train_mpnet_projection(
        examples, data_dir, model_manager, embedding_extractor
    )

    # Train multi-model ensemble (best robustness)
    ensemble_config = train_multi_model_ensemble(
        examples, data_dir, model_manager, embedding_extractor
    )

    # Create summary
    summary = create_summary(data_dir)

    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Recommended projection: {summary['recommended']}")
    logger.info(f"Most robust: {summary['most_robust']}")


if __name__ == "__main__":
    main()
