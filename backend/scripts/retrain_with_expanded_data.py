#!/usr/bin/env python3
"""
Retrain Projection with Expanded Training Data

This script retrains the projection using the expanded dataset
with paraphrase augmentation for improved robustness.
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model_manager, ModelType
from models.embedding import EmbeddingExtractor
from models.projection import ProjectionTrainer, TrainingExample

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_expanded_examples(data_dir: Path) -> list:
    """Load expanded training examples."""
    expanded_file = data_dir / "examples_expanded.json"
    original_file = data_dir / "examples.json"

    # Prefer expanded if available
    if expanded_file.exists():
        logger.info(f"Loading expanded examples from {expanded_file}")
        with open(expanded_file) as f:
            data = json.load(f)
    elif original_file.exists():
        logger.info(f"Expanded file not found, using original: {original_file}")
        with open(original_file) as f:
            data = json.load(f)
    else:
        raise FileNotFoundError("No training examples found")

    examples = []
    for item in data:
        # Handle both old "fairness" and new "perceived_justice" naming
        fairness_val = item.get("perceived_justice", item.get("fairness", 0.0))

        examples.append(TrainingExample(
            text=item["text"],
            agency=item["agency"],
            fairness=fairness_val,  # Internal name stays "fairness" for compatibility
            belonging=item["belonging"],
            source=item.get("source", "unknown")
        ))

    return examples


def compute_embeddings(examples: list, model_manager, embedding_extractor) -> tuple:
    """Compute embeddings for all examples."""
    import numpy as np

    model_id = "all-MiniLM-L6-v2"

    if not model_manager.is_loaded(model_id):
        logger.info(f"Loading model: {model_id}")
        model_manager.load_model(model_id, ModelType.SENTENCE_TRANSFORMER)

    logger.info(f"Computing embeddings for {len(examples)} examples...")

    embeddings = []
    targets = []

    for i, ex in enumerate(examples):
        if i % 50 == 0:
            logger.info(f"  Processing {i}/{len(examples)}...")

        result = embedding_extractor.extract(ex.text, model_id)
        embeddings.append(result.embedding)
        targets.append([ex.agency, ex.fairness, ex.belonging])

    return np.array(embeddings), np.array(targets)


def train_and_save(trainer: ProjectionTrainer, embeddings, targets, examples):
    """Train projection and save."""
    import numpy as np
    from models.projection import ProjectionHead

    logger.info("Training projection with ridge regression...")

    # Create and train a new ProjectionHead
    projection = ProjectionHead()
    metrics = projection.train(embeddings, targets, auto_tune_alpha=True)

    logger.info(f"Training complete. Metrics:")
    logger.info(f"  R² overall: {metrics.r2_overall:.4f}")
    logger.info(f"  R² agency: {metrics.r2_agency:.4f}")
    logger.info(f"  R² fairness: {metrics.r2_fairness:.4f}")
    logger.info(f"  R² belonging: {metrics.r2_belonging:.4f}")
    logger.info(f"  CV score: {metrics.cv_score_mean:.4f} ± {metrics.cv_score_std:.4f}")
    logger.info(f"  Training examples: {metrics.n_examples}")
    if metrics.warnings:
        for warning in metrics.warnings:
            logger.warning(f"  Warning: {warning}")

    # Save the projection to the standard location
    proj_path = trainer.data_dir / "current_projection"
    projection.save(proj_path)
    logger.info(f"Projection saved to {proj_path}")

    # Update examples file
    examples_file = trainer.data_dir / "examples.json"
    examples_data = [
        {
            "text": ex.text,
            "agency": ex.agency,
            "fairness": ex.fairness,
            "belonging": ex.belonging,
            "source": ex.source
        }
        for ex in examples
    ]

    with open(examples_file, 'w') as f:
        json.dump(examples_data, f, indent=2)

    logger.info(f"Saved {len(examples)} examples to {examples_file}")

    return projection


def train_ensemble(trainer: ProjectionTrainer, embeddings, targets):
    """Train ensemble projection for uncertainty quantification."""
    try:
        from models.ensemble_projection import EnsembleProjection

        logger.info("Training ensemble projection (25 models)...")

        ensemble = EnsembleProjection(n_bootstrap=5, alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        ensemble.train(embeddings, targets)

        # Save ensemble
        ensemble_path = trainer.data_dir / "ensemble_projection.json"
        ensemble.save(ensemble_path)

        logger.info(f"Ensemble saved to {ensemble_path}")
        return ensemble

    except ImportError as e:
        logger.warning(f"Could not train ensemble: {e}")
        return None


def validate_projection(projection, embeddings, targets):
    """Quick validation of the trained projection."""
    import numpy as np

    logger.info("Validating projection...")

    # Project all embeddings
    predictions = []
    for emb in embeddings:
        coords = projection.project(emb)
        predictions.append([coords.agency, coords.fairness, coords.belonging])

    predictions = np.array(predictions)

    # Calculate errors
    errors = np.abs(predictions - targets)

    logger.info("Validation results:")
    logger.info(f"  Mean absolute error per axis:")
    logger.info(f"    Agency: {errors[:, 0].mean():.4f}")
    logger.info(f"    Perceived Justice: {errors[:, 1].mean():.4f}")
    logger.info(f"    Belonging: {errors[:, 2].mean():.4f}")
    logger.info(f"  Max error: {errors.max():.4f}")

    # Check for outliers
    outliers = np.any(errors > 1.0, axis=1)
    if outliers.any():
        logger.warning(f"  Found {outliers.sum()} examples with error > 1.0")


def main():
    """Main retraining workflow."""
    logger.info("=" * 60)
    logger.info("RETRAINING PROJECTION WITH EXPANDED DATA")
    logger.info("=" * 60)

    # Setup paths
    backend_dir = Path(__file__).parent.parent
    data_dir = backend_dir / "data" / "projections"

    # Initialize components
    model_manager = get_model_manager()
    embedding_extractor = EmbeddingExtractor(model_manager)
    trainer = ProjectionTrainer(data_dir)

    # Load examples
    examples = load_expanded_examples(data_dir)
    logger.info(f"Loaded {len(examples)} training examples")

    # Check distribution
    sources = {}
    for ex in examples:
        sources[ex.source] = sources.get(ex.source, 0) + 1
    logger.info(f"Examples by source: {sources}")

    # Compute embeddings
    embeddings, targets = compute_embeddings(examples, model_manager, embedding_extractor)

    # Train main projection
    projection = train_and_save(trainer, embeddings, targets, examples)

    # Train ensemble for uncertainty
    ensemble = train_ensemble(trainer, embeddings, targets)

    # Validate
    validate_projection(projection, embeddings, targets)

    logger.info("=" * 60)
    logger.info("RETRAINING COMPLETE")
    logger.info("=" * 60)

    return projection, ensemble


if __name__ == "__main__":
    main()
