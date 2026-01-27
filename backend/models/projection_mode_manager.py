"""
Projection Mode Manager

Manages switching between different projection configurations:
1. current_projection - Default MiniLM-based (CV: 0.383)
2. mpnet_projection - Best accuracy with all-mpnet-base-v2 (CV: 0.612)
3. multi_model_ensemble - Best robustness, averages 3 models
4. ensemble_projection - 25-model bootstrap ensemble for uncertainty

The manager handles loading appropriate models and projections based on
the selected mode.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .projection import ProjectionHead, Vector3, ProjectionWithUncertainty

logger = logging.getLogger(__name__)


class ProjectionMode(str, Enum):
    """Available projection modes."""
    CURRENT = "current_projection"
    MPNET = "mpnet_projection"
    MULTI_MODEL_ENSEMBLE = "multi_model_ensemble"
    BOOTSTRAP_ENSEMBLE = "ensemble_projection"


@dataclass
class ProjectionModeInfo:
    """Information about a projection mode."""
    name: str
    display_name: str
    description: str
    cv_score: Optional[float] = None
    r2_score: Optional[float] = None
    models: List[str] = field(default_factory=list)
    path: str = ""
    is_available: bool = False
    is_active: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "cv_score": self.cv_score,
            "r2_score": self.r2_score,
            "models": self.models,
            "is_available": self.is_available,
            "is_active": self.is_active
        }


@dataclass
class MultiModelEnsembleConfig:
    """Configuration for multi-model ensemble."""
    models: List[Dict[str, Any]]
    projections_dir: str
    aggregation: str = "mean"


class ProjectionModeManager:
    """
    Manages multiple projection configurations and enables runtime switching.

    Supports four modes:
    1. current_projection: Default MiniLM-based single model
    2. mpnet_projection: Higher accuracy MPNet-based single model
    3. multi_model_ensemble: Averages projections from 3 different embedding models
    4. ensemble_projection: 25-model bootstrap ensemble for uncertainty quantification

    Usage:
        manager = ProjectionModeManager(projections_dir)
        await manager.initialize()

        # List available modes
        modes = manager.list_modes()

        # Select a mode
        manager.select_mode("mpnet_projection")

        # Project with current mode
        coords = await manager.project(embedding)
    """

    def __init__(self, projections_dir: Path):
        """
        Initialize the projection mode manager.

        Args:
            projections_dir: Path to directory containing projection configurations
        """
        self.projections_dir = Path(projections_dir)
        self.current_mode: Optional[ProjectionMode] = None

        # Loaded projections
        self._current_projection: Optional[ProjectionHead] = None
        self._mpnet_projection: Optional[ProjectionHead] = None
        self._multi_model_config: Optional[MultiModelEnsembleConfig] = None
        self._multi_model_projections: Dict[str, ProjectionHead] = {}
        self._bootstrap_ensemble: Optional[Any] = None  # EnsembleProjection

        # Model info cache
        self._mode_info: Dict[str, ProjectionModeInfo] = {}

        logger.info(f"ProjectionModeManager initialized with dir: {projections_dir}")

    def initialize(self) -> None:
        """
        Initialize by loading projection summary and checking availability.

        Loads the projection_summary.json and verifies which projections
        are actually available on disk.
        """
        # Load projection summary
        summary_path = self.projections_dir / "projection_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            logger.info(f"Loaded projection summary from {summary_path}")
        else:
            summary = {"available_projections": []}
            logger.warning(f"No projection summary found at {summary_path}")

        # Build mode info from summary
        self._build_mode_info(summary)

        # Set default mode to current_projection if available
        if self._mode_info.get(ProjectionMode.CURRENT.value, ProjectionModeInfo("", "", "")).is_available:
            self.select_mode(ProjectionMode.CURRENT.value)
        elif self._mode_info.get(ProjectionMode.MPNET.value, ProjectionModeInfo("", "", "")).is_available:
            self.select_mode(ProjectionMode.MPNET.value)

        logger.info(f"ProjectionModeManager initialized. Active mode: {self.current_mode}")

    def _build_mode_info(self, summary: dict) -> None:
        """Build mode info from projection summary."""
        available = summary.get("available_projections", [])

        # Create mode info for each known projection type
        mode_configs = {
            ProjectionMode.CURRENT.value: {
                "display_name": "MiniLM Default",
                "description": "Default ridge regression projection using all-MiniLM-L6-v2"
            },
            ProjectionMode.MPNET.value: {
                "display_name": "MPNet (Best Accuracy)",
                "description": "Best single-model projection using all-mpnet-base-v2"
            },
            ProjectionMode.MULTI_MODEL_ENSEMBLE.value: {
                "display_name": "Multi-Model Ensemble",
                "description": "Averages projections from 3 embedding models for robustness"
            },
            ProjectionMode.BOOTSTRAP_ENSEMBLE.value: {
                "display_name": "Bootstrap Ensemble",
                "description": "25-model bootstrap ensemble for uncertainty quantification"
            }
        }

        for proj in available:
            name = proj.get("name", "")
            if name in mode_configs:
                config = mode_configs[name]

                # Check if projection exists on disk
                path = proj.get("path", "")
                is_available = Path(path).exists() if path else False

                # Handle ensemble_projection which is a .json directory
                if name == ProjectionMode.BOOTSTRAP_ENSEMBLE.value:
                    # It's stored as ensemble_projection.json directory
                    is_available = (self.projections_dir / "ensemble_projection.json").exists()

                models = proj.get("models", [])
                if not models and proj.get("model"):
                    models = [proj["model"]]

                self._mode_info[name] = ProjectionModeInfo(
                    name=name,
                    display_name=config["display_name"],
                    description=proj.get("description", config["description"]),
                    cv_score=proj.get("cv_score"),
                    r2_score=proj.get("r2"),
                    models=models,
                    path=path,
                    is_available=is_available,
                    is_active=False
                )

        # Add any missing modes with default info
        for mode_name, config in mode_configs.items():
            if mode_name not in self._mode_info:
                path = self.projections_dir / mode_name
                is_available = path.exists()

                self._mode_info[mode_name] = ProjectionModeInfo(
                    name=mode_name,
                    display_name=config["display_name"],
                    description=config["description"],
                    is_available=is_available
                )

    def list_modes(self) -> List[ProjectionModeInfo]:
        """
        List all available projection modes.

        Returns:
            List of ProjectionModeInfo objects describing each mode.
        """
        return list(self._mode_info.values())

    def get_mode_info(self, mode_name: str) -> Optional[ProjectionModeInfo]:
        """
        Get information about a specific mode.

        Args:
            mode_name: Name of the projection mode

        Returns:
            ProjectionModeInfo or None if mode doesn't exist
        """
        return self._mode_info.get(mode_name)

    def get_current_mode(self) -> Optional[str]:
        """Get the name of the currently active mode."""
        return self.current_mode.value if self.current_mode else None

    def select_mode(self, mode_name: str) -> ProjectionModeInfo:
        """
        Select and load a projection mode.

        Args:
            mode_name: Name of the mode to select

        Returns:
            ProjectionModeInfo for the selected mode

        Raises:
            ValueError: If mode is not available
        """
        if mode_name not in self._mode_info:
            raise ValueError(f"Unknown projection mode: {mode_name}")

        mode_info = self._mode_info[mode_name]
        if not mode_info.is_available:
            raise ValueError(f"Projection mode '{mode_name}' is not available on disk")

        # Update active status
        for m in self._mode_info.values():
            m.is_active = False
        mode_info.is_active = True

        # Load the projection
        self._load_mode(mode_name)

        # Convert string to enum
        try:
            self.current_mode = ProjectionMode(mode_name)
        except ValueError:
            self.current_mode = None

        logger.info(f"Selected projection mode: {mode_name}")
        return mode_info

    def _load_mode(self, mode_name: str) -> None:
        """Load the projection for the specified mode."""
        if mode_name == ProjectionMode.CURRENT.value:
            self._load_current_projection()
        elif mode_name == ProjectionMode.MPNET.value:
            self._load_mpnet_projection()
        elif mode_name == ProjectionMode.MULTI_MODEL_ENSEMBLE.value:
            self._load_multi_model_ensemble()
        elif mode_name == ProjectionMode.BOOTSTRAP_ENSEMBLE.value:
            self._load_bootstrap_ensemble()

    def _load_current_projection(self) -> None:
        """Load the default MiniLM projection."""
        if self._current_projection is not None:
            return  # Already loaded

        path = self.projections_dir / "current_projection"
        if path.exists():
            self._current_projection = ProjectionHead.load(path)
            logger.info(f"Loaded current_projection from {path}")

    def _load_mpnet_projection(self) -> None:
        """Load the MPNet projection."""
        if self._mpnet_projection is not None:
            return  # Already loaded

        path = self.projections_dir / "mpnet_projection"
        if path.exists():
            self._mpnet_projection = ProjectionHead.load(path)
            logger.info(f"Loaded mpnet_projection from {path}")

    def _load_multi_model_ensemble(self) -> None:
        """Load the multi-model ensemble configuration and projections."""
        if self._multi_model_config is not None:
            return  # Already loaded

        config_path = self.projections_dir / "multi_model_ensemble" / "ensemble_config.json"
        if not config_path.exists():
            raise ValueError(f"Multi-model ensemble config not found at {config_path}")

        with open(config_path) as f:
            config_data = json.load(f)

        self._multi_model_config = MultiModelEnsembleConfig(
            models=config_data["models"],
            projections_dir=config_data["projections_dir"],
            aggregation=config_data.get("aggregation", "mean")
        )

        # Load each model's projection
        for model_info in self._multi_model_config.models:
            model_id = model_info["model_id"]
            proj_path = Path(model_info["projection_path"])

            if proj_path.exists():
                self._multi_model_projections[model_id] = ProjectionHead.load(proj_path)
                logger.info(f"Loaded projection for {model_id}")
            else:
                logger.warning(f"Projection not found for {model_id} at {proj_path}")

        logger.info(f"Loaded multi-model ensemble with {len(self._multi_model_projections)} projections")

    def _load_bootstrap_ensemble(self) -> None:
        """Load the bootstrap ensemble for uncertainty quantification."""
        if self._bootstrap_ensemble is not None:
            return  # Already loaded

        from .ensemble_projection import EnsembleProjection

        # The ensemble is stored in ensemble_projection.json directory
        path = self.projections_dir / "ensemble_projection.json"
        if path.exists():
            self._bootstrap_ensemble = EnsembleProjection.load(path)
            logger.info(f"Loaded bootstrap ensemble from {path}")

    def get_required_models(self) -> List[str]:
        """
        Get the list of embedding models required for the current mode.

        Returns:
            List of model IDs that need to be loaded
        """
        if self.current_mode is None:
            return []

        mode_info = self._mode_info.get(self.current_mode.value)
        if mode_info is None:
            return []

        return mode_info.models

    def project(
        self,
        embeddings: Dict[str, np.ndarray],
        model_manager: Any,
        embedding_extractor: Any
    ) -> Union[Vector3, ProjectionWithUncertainty]:
        """
        Project embeddings using the current projection mode.

        For single-model modes (current, mpnet), uses the embedding for that model.
        For multi-model ensemble, averages projections from all models.
        For bootstrap ensemble, returns projection with uncertainty.

        Args:
            embeddings: Dict mapping model_id -> embedding array
            model_manager: ModelManager instance (for loading models if needed)
            embedding_extractor: EmbeddingExtractor instance

        Returns:
            Vector3 for simple projections, ProjectionWithUncertainty for ensemble

        Raises:
            ValueError: If no mode is selected or required embeddings missing
        """
        if self.current_mode is None:
            raise ValueError("No projection mode selected")

        if self.current_mode == ProjectionMode.CURRENT:
            return self._project_current(embeddings)
        elif self.current_mode == ProjectionMode.MPNET:
            return self._project_mpnet(embeddings)
        elif self.current_mode == ProjectionMode.MULTI_MODEL_ENSEMBLE:
            return self._project_multi_model(embeddings)
        elif self.current_mode == ProjectionMode.BOOTSTRAP_ENSEMBLE:
            return self._project_bootstrap_ensemble(embeddings)
        else:
            raise ValueError(f"Unknown projection mode: {self.current_mode}")

    def _project_current(self, embeddings: Dict[str, np.ndarray]) -> Vector3:
        """Project using current (MiniLM) projection."""
        if self._current_projection is None:
            self._load_current_projection()

        model_id = "all-MiniLM-L6-v2"
        if model_id not in embeddings:
            raise ValueError(f"Missing embedding for {model_id}")

        return self._current_projection.project(embeddings[model_id])

    def _project_mpnet(self, embeddings: Dict[str, np.ndarray]) -> Vector3:
        """Project using MPNet projection."""
        if self._mpnet_projection is None:
            self._load_mpnet_projection()

        model_id = "all-mpnet-base-v2"
        if model_id not in embeddings:
            raise ValueError(f"Missing embedding for {model_id}")

        return self._mpnet_projection.project(embeddings[model_id])

    def _project_multi_model(self, embeddings: Dict[str, np.ndarray]) -> Vector3:
        """Project using multi-model ensemble (average of 3 models)."""
        if self._multi_model_config is None:
            self._load_multi_model_ensemble()

        projections = []
        weights = []

        for model_info in self._multi_model_config.models:
            model_id = model_info["model_id"]
            weight = model_info.get("weight", 1.0)

            if model_id not in embeddings:
                logger.warning(f"Missing embedding for {model_id}, skipping")
                continue

            if model_id not in self._multi_model_projections:
                logger.warning(f"Missing projection for {model_id}, skipping")
                continue

            proj = self._multi_model_projections[model_id].project(embeddings[model_id])
            projections.append(np.array(proj.to_list()))
            weights.append(weight)

        if not projections:
            raise ValueError("No projections computed - missing embeddings or projections")

        # Average projections (weighted if weights differ)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        projections_array = np.array(projections)
        avg_projection = np.average(projections_array, axis=0, weights=weights)

        return Vector3.from_array(avg_projection)

    def _project_bootstrap_ensemble(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> ProjectionWithUncertainty:
        """Project using bootstrap ensemble with uncertainty."""
        if self._bootstrap_ensemble is None:
            self._load_bootstrap_ensemble()

        model_id = "all-MiniLM-L6-v2"  # Bootstrap ensemble uses MiniLM
        if model_id not in embeddings:
            raise ValueError(f"Missing embedding for {model_id}")

        return self._bootstrap_ensemble.project_with_uncertainty(embeddings[model_id])

    def get_projection_for_mode(self, mode_name: str) -> Optional[ProjectionHead]:
        """
        Get the projection head for a specific mode (if it's a simple projection).

        This is useful for getting the raw projection object.

        Args:
            mode_name: Name of the projection mode

        Returns:
            ProjectionHead or None
        """
        if mode_name == ProjectionMode.CURRENT.value:
            if self._current_projection is None:
                self._load_current_projection()
            return self._current_projection
        elif mode_name == ProjectionMode.MPNET.value:
            if self._mpnet_projection is None:
                self._load_mpnet_projection()
            return self._mpnet_projection
        return None


# Singleton instance
_projection_mode_manager: Optional[ProjectionModeManager] = None


def get_projection_mode_manager(projections_dir: Optional[Path] = None) -> ProjectionModeManager:
    """
    Get the singleton ProjectionModeManager instance.

    Args:
        projections_dir: Path to projections directory (only used on first call)

    Returns:
        ProjectionModeManager instance
    """
    global _projection_mode_manager

    if _projection_mode_manager is None:
        if projections_dir is None:
            projections_dir = Path("./data/projections")
        _projection_mode_manager = ProjectionModeManager(projections_dir)
        _projection_mode_manager.initialize()

    return _projection_mode_manager


def reset_projection_mode_manager() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _projection_mode_manager
    _projection_mode_manager = None
