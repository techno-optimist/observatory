"""
Validation Module

Provides human annotation collection, inter-annotator agreement metrics,
projection validation frameworks, and statistical validation tests.
"""

from .annotation import (
    AnnotationDataset,
    Annotator,
    TextAnnotation,
    AnnotationStats,
    compute_agreement_metrics
)
from .projection_validator import (
    ProjectionValidator,
    ValidationMetrics,
    ProjectionComparison
)
from .statistical_validation_tests import (
    ValidationResult,
    ValidationSuite,
    test_bootstrap_coverage,
    test_effect_size_calibration,
    test_multiple_comparison_fwer,
    test_fisher_rao_metric_properties,
    test_phase_detection_roc,
    run_full_validation_suite
)

__all__ = [
    "AnnotationDataset",
    "Annotator",
    "TextAnnotation",
    "AnnotationStats",
    "compute_agreement_metrics",
    "ProjectionValidator",
    "ValidationMetrics",
    "ProjectionComparison",
    # Statistical validation
    "ValidationResult",
    "ValidationSuite",
    "test_bootstrap_coverage",
    "test_effect_size_calibration",
    "test_multiple_comparison_fwer",
    "test_fisher_rao_metric_properties",
    "test_phase_detection_roc",
    "run_full_validation_suite"
]
