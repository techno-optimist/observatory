from .model_manager import ModelManager, get_model_manager, ModelType, ModelInfo, ModelProvenance
from .embedding import EmbeddingExtractor, PoolingStrategy, EmbeddingResult
from .projection import (
    ProjectionHead, ProjectionTrainer, AnchorProjection,
    Vector3, TrainingExample, ProjectionMetrics,
    # Non-linear projection methods
    ProjectionWithUncertainty,
    GaussianProcessProjection,
    NeuralProbeProjection,
    ConceptActivationVectors,
    ProjectionMethod,
    TORCH_AVAILABLE
)
from .embedding_cache import EmbeddingCache, get_embedding_cache
from .enhanced_projection import (
    BayesianProjectionHead, UncertainProjection,
    LayerWiseAnalyzer, ContrastiveExplainer,
    UserDefinedAxesProjection, EnsembleProjection,
    AxisDefinition, ContrastiveExplanation, LayerProjection
)
# Axis configuration and aliasing (for fairness -> perceived_justice rename)
from .axis_config import (
    AXIS_CONFIG,
    CANONICAL_AXES,
    INTERNAL_AXES,
    translate_axis_name,
    get_axis_display_name,
    get_canonical_axis_name,
    get_internal_axis_name,
    get_all_axis_aliases,
    get_axis_description,
    get_axis_deprecation_notice,
    convert_vector_keys_to_canonical,
    convert_vector_keys_to_internal,
    is_valid_axis_name,
    get_axis_config
)
# Projection mode management
from .projection_mode_manager import (
    ProjectionModeManager,
    ProjectionMode,
    ProjectionModeInfo,
    get_projection_mode_manager,
    reset_projection_mode_manager
)

__all__ = [
    # Model management
    "ModelManager",
    "get_model_manager",
    "ModelType",
    "ModelInfo",
    "ModelProvenance",
    # Embedding
    "EmbeddingExtractor",
    "PoolingStrategy",
    "EmbeddingResult",
    # Caching
    "EmbeddingCache",
    "get_embedding_cache",
    # Projection - Linear
    "ProjectionHead",
    "ProjectionTrainer",
    "AnchorProjection",
    "Vector3",
    "TrainingExample",
    "ProjectionMetrics",
    # Projection - Non-linear
    "ProjectionWithUncertainty",
    "GaussianProcessProjection",
    "NeuralProbeProjection",
    "ConceptActivationVectors",
    "ProjectionMethod",
    "TORCH_AVAILABLE",
    # Enhanced projection
    "BayesianProjectionHead",
    "UncertainProjection",
    "LayerWiseAnalyzer",
    "ContrastiveExplainer",
    "UserDefinedAxesProjection",
    "EnsembleProjection",
    "AxisDefinition",
    "ContrastiveExplanation",
    "LayerProjection",
    # Axis configuration and aliasing
    "AXIS_CONFIG",
    "CANONICAL_AXES",
    "INTERNAL_AXES",
    "translate_axis_name",
    "get_axis_display_name",
    "get_canonical_axis_name",
    "get_internal_axis_name",
    "get_all_axis_aliases",
    "get_axis_description",
    "get_axis_deprecation_notice",
    "convert_vector_keys_to_canonical",
    "convert_vector_keys_to_internal",
    "is_valid_axis_name",
    "get_axis_config",
    # Projection mode management
    "ProjectionModeManager",
    "ProjectionMode",
    "ProjectionModeInfo",
    "get_projection_mode_manager",
    "reset_projection_mode_manager"
]
