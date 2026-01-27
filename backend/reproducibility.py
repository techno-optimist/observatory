"""
Reproducibility Infrastructure

Provides utilities for ensuring ML experiments are reproducible:
- Global seed management
- Training data hashing
- Environment capture
- Full provenance chain

This module should be used before any training to ensure
experiments can be reproduced precisely.
"""

import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReproducibilityContext:
    """
    Captures all information needed to reproduce an experiment.

    This includes random seeds, environment info, and data versions.
    """
    seed: int
    numpy_seed: int
    python_seed: int
    torch_seed: Optional[int] = None
    cuda_seed: Optional[int] = None

    # Environment info
    python_version: str = field(default_factory=lambda: sys.version)
    platform_info: str = field(default_factory=lambda: platform.platform())
    numpy_version: str = field(default_factory=lambda: np.__version__)
    torch_version: Optional[str] = None

    # Data versioning
    training_data_hash: Optional[str] = None
    num_training_examples: int = 0

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "seeds": {
                "global": self.seed,
                "numpy": self.numpy_seed,
                "python": self.python_seed,
                "torch": self.torch_seed,
                "cuda": self.cuda_seed
            },
            "environment": {
                "python_version": self.python_version,
                "platform": self.platform_info,
                "numpy_version": self.numpy_version,
                "torch_version": self.torch_version
            },
            "data": {
                "training_data_hash": self.training_data_hash,
                "num_training_examples": self.num_training_examples
            },
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReproducibilityContext":
        seeds = data.get("seeds", {})
        env = data.get("environment", {})
        data_info = data.get("data", {})

        return cls(
            seed=seeds.get("global", 0),
            numpy_seed=seeds.get("numpy", 0),
            python_seed=seeds.get("python", 0),
            torch_seed=seeds.get("torch"),
            cuda_seed=seeds.get("cuda"),
            python_version=env.get("python_version", ""),
            platform_info=env.get("platform", ""),
            numpy_version=env.get("numpy_version", ""),
            torch_version=env.get("torch_version"),
            training_data_hash=data_info.get("training_data_hash"),
            num_training_examples=data_info.get("num_training_examples", 0),
            created_at=data.get("created_at", datetime.utcnow().isoformat())
        )


class SeedManager:
    """
    Manages random seeds across all relevant libraries.

    Usage:
        seed_manager = SeedManager()
        seed_manager.set_global_seed(42)

        # Later, to reproduce:
        context = seed_manager.get_context()
    """

    def __init__(self):
        self._global_seed: Optional[int] = None
        self._context: Optional[ReproducibilityContext] = None

    def set_global_seed(self, seed: int) -> ReproducibilityContext:
        """
        Set seeds for all random number generators.

        Args:
            seed: The seed value to use

        Returns:
            ReproducibilityContext with all seed values
        """
        self._global_seed = seed

        # Python random
        random.seed(seed)
        python_seed = seed

        # NumPy
        np.random.seed(seed)
        numpy_seed = seed

        # PyTorch (if available)
        torch_seed = None
        cuda_seed = None
        torch_version = None

        try:
            import torch
            torch.manual_seed(seed)
            torch_seed = seed
            torch_version = torch.__version__

            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                cuda_seed = seed

                # For full determinism (may impact performance)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

        self._context = ReproducibilityContext(
            seed=seed,
            numpy_seed=numpy_seed,
            python_seed=python_seed,
            torch_seed=torch_seed,
            cuda_seed=cuda_seed,
            torch_version=torch_version
        )

        logger.info(f"Global seed set to {seed}")
        return self._context

    def get_context(self) -> Optional[ReproducibilityContext]:
        """Get the current reproducibility context."""
        return self._context

    def get_seed(self) -> Optional[int]:
        """Get the current global seed."""
        return self._global_seed


# Global seed manager instance
_seed_manager = SeedManager()


def set_seed(seed: int) -> ReproducibilityContext:
    """
    Convenience function to set global seed.

    Args:
        seed: The seed value

    Returns:
        ReproducibilityContext
    """
    return _seed_manager.set_global_seed(seed)


def get_seed() -> Optional[int]:
    """Get the current global seed."""
    return _seed_manager.get_seed()


def get_reproducibility_context() -> Optional[ReproducibilityContext]:
    """Get the current reproducibility context."""
    return _seed_manager.get_context()


# --- Data Hashing ---

def hash_training_examples(
    examples: List[Dict[str, Any]],
    include_fields: Optional[List[str]] = None
) -> str:
    """
    Compute a deterministic hash of training examples.

    This ensures that identical training data produces the same hash,
    enabling tracking of which experiments used the same data.

    Args:
        examples: List of training example dictionaries
        include_fields: Optional list of fields to include in hash
                       (default: all fields)

    Returns:
        SHA256 hash of the serialized data
    """
    if include_fields:
        # Filter to only included fields
        filtered_examples = [
            {k: v for k, v in ex.items() if k in include_fields}
            for ex in examples
        ]
    else:
        filtered_examples = examples

    # Sort examples by a stable key to ensure deterministic ordering
    # Use text as primary sort key if available
    try:
        sorted_examples = sorted(
            filtered_examples,
            key=lambda x: (
                str(x.get("text", "")),
                float(x.get("agency", 0)),
                float(x.get("fairness", 0)),
                float(x.get("belonging", 0))
            )
        )
    except (TypeError, ValueError):
        # Fall back to string representation if sorting fails
        sorted_examples = sorted(filtered_examples, key=lambda x: json.dumps(x, sort_keys=True))

    # Serialize to JSON with sorted keys for determinism
    serialized = json.dumps(sorted_examples, sort_keys=True, ensure_ascii=True)

    return hashlib.sha256(serialized.encode()).hexdigest()


def hash_text_corpus(texts: List[str]) -> str:
    """
    Compute a deterministic hash of a text corpus.

    Args:
        texts: List of text strings

    Returns:
        SHA256 hash
    """
    # Sort for determinism
    sorted_texts = sorted(texts)
    serialized = json.dumps(sorted_texts, ensure_ascii=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def hash_embeddings(embeddings: np.ndarray) -> str:
    """
    Compute a hash of an embedding matrix.

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)

    Returns:
        SHA256 hash
    """
    # Ensure consistent dtype
    embeddings = embeddings.astype(np.float32)
    return hashlib.sha256(embeddings.tobytes()).hexdigest()


def hash_hyperparams(hyperparams: Dict[str, Any]) -> str:
    """
    Compute a hash of hyperparameters.

    Args:
        hyperparams: Dictionary of hyperparameter names and values

    Returns:
        SHA256 hash
    """
    serialized = json.dumps(hyperparams, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


# --- Environment Capture ---

def capture_environment() -> Dict[str, Any]:
    """
    Capture current environment information for reproducibility.

    Returns:
        Dictionary with environment details
    """
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Try to capture package versions
    try:
        import pkg_resources
        installed_packages = {
            pkg.key: pkg.version
            for pkg in pkg_resources.working_set
        }
        # Only include relevant packages
        relevant_packages = [
            "torch", "transformers", "sentence-transformers",
            "scikit-learn", "fastapi", "uvicorn", "numpy",
            "huggingface-hub"
        ]
        env_info["packages"] = {
            k: v for k, v in installed_packages.items()
            if k in relevant_packages
        }
    except ImportError:
        env_info["packages"] = {}

    # PyTorch specific
    try:
        import torch
        env_info["torch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda
            env_info["gpu_count"] = torch.cuda.device_count()
            env_info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        env_info["torch_version"] = None

    return env_info


# --- Full Provenance Chain ---

@dataclass
class ExperimentProvenance:
    """
    Complete provenance chain for an experiment.

    Combines model provenance, data provenance, and reproducibility context
    into a single record suitable for publication.
    """
    experiment_id: str

    # Model info
    model_id: str
    model_revision: Optional[str]
    model_sha256: Optional[str]

    # Data info
    training_data_hash: str
    num_training_examples: int

    # Reproducibility
    random_seed: int
    reproducibility_context: Dict[str, Any]

    # Hyperparameters
    hyperparams: Dict[str, Any]
    hyperparams_hash: str

    # Environment
    environment: Dict[str, Any]

    # Timestamps
    created_at: str

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "model": {
                "model_id": self.model_id,
                "revision": self.model_revision,
                "sha256": self.model_sha256
            },
            "data": {
                "hash": self.training_data_hash,
                "num_examples": self.num_training_examples
            },
            "reproducibility": {
                "seed": self.random_seed,
                "context": self.reproducibility_context
            },
            "hyperparams": {
                "values": self.hyperparams,
                "hash": self.hyperparams_hash
            },
            "environment": self.environment,
            "created_at": self.created_at
        }

    def to_citation_string(self) -> str:
        """
        Generate a citation-ready string for the experiment.

        Suitable for inclusion in papers.
        """
        return (
            f"Experiment {self.experiment_id}: "
            f"model={self.model_id} (rev={self.model_revision[:8] if self.model_revision else 'N/A'}), "
            f"data_hash={self.training_data_hash[:12]}..., "
            f"n={self.num_training_examples}, "
            f"seed={self.random_seed}"
        )


def create_experiment_provenance(
    experiment_id: str,
    model_provenance: Any,  # ModelProvenance object
    training_examples: List[Dict[str, Any]],
    hyperparams: Dict[str, Any],
    seed: int
) -> ExperimentProvenance:
    """
    Create a complete provenance record for an experiment.

    Args:
        experiment_id: Unique experiment identifier
        model_provenance: ModelProvenance object from model_manager
        training_examples: List of training examples
        hyperparams: Hyperparameter dictionary
        seed: Random seed used

    Returns:
        ExperimentProvenance record
    """
    # Extract model info
    if hasattr(model_provenance, 'to_dict'):
        model_dict = model_provenance.to_dict()
    elif isinstance(model_provenance, dict):
        model_dict = model_provenance
    else:
        model_dict = {"model_id": str(model_provenance)}

    # Set seed and capture context
    context = set_seed(seed)

    # Hash training data
    data_hash = hash_training_examples(training_examples)

    # Hash hyperparams
    hp_hash = hash_hyperparams(hyperparams)

    # Capture environment
    env = capture_environment()

    return ExperimentProvenance(
        experiment_id=experiment_id,
        model_id=model_dict.get("model_id", "unknown"),
        model_revision=model_dict.get("revision"),
        model_sha256=model_dict.get("sha256"),
        training_data_hash=data_hash,
        num_training_examples=len(training_examples),
        random_seed=seed,
        reproducibility_context=context.to_dict() if context else {},
        hyperparams=hyperparams,
        hyperparams_hash=hp_hash,
        environment=env,
        created_at=datetime.utcnow().isoformat()
    )
