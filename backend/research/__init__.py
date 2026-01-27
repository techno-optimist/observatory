"""
Research modules for emergent language analysis.

This package contains tools for studying the minimal structures necessary
for AI-human coordination. These tools extend the Cultural Soliton Observatory
to study emergent AI languages as coordination artifacts.

Modules:
- grammar_deletion_test: Identifies coordination-necessary vs decorative grammar
- legibility_analyzer: Monitors legibility and detects phase transitions
- evolution_tracker: Tracks linguistic evolution through coordination space
- calibration_baseline: Compares human vs minimal AI codes
- academic_statistics: Publication-grade statistical methods
- hierarchical_coordinates: High-resolution 9D coordination manifold
- publication_formats: LaTeX tables, forest plots, academic outputs
- realtime_monitor: Real-time emergent communication monitoring
- emergent_language: Metrics for AI-AI communication protocols (vocabulary,
                     compositionality, mutual information, protocol evolution)

Key Insight: Solitons don't care about substrate. These tools detect
coordination invariants regardless of whether they're expressed in
human language or emergent AI protocols.

V2.1 Real-Time Monitoring (AI Safety Critical):
- EmergentCommunicationMonitor: Streaming analysis with alerts
- Regime detection: NATURAL → TECHNICAL → COMPRESSED → OPAQUE
- Drift/ossification/mode-collapse detection
- Legibility decay early warning

V2.0 Theoretical Framework (based on expert synthesis):
- Fiber bundle structure: E = B x F (base manifold + decorative fiber)
- Fisher Information Metric for probability distribution distances
- CoordinationCore: 9D irreducible substrate (agency/justice/belonging decomposed)
- CoordinationModifiers: 9D modulating features (epistemic/temporal/social/emotional)
"""

from .grammar_deletion_test import (
    GrammarDeletionAnalyzer,
    GrammarAnalysis,
    DeletionResult,
    DELETION_FUNCTIONS,
)

from .legibility_analyzer import (
    LegibilityAnalyzer,
    LegibilityScore,
    LegibilityRegime,
    PhaseTransitionResult,
    InterpretabilityMetrics,
    compute_legibility,
    compute_legibility_sync,
)

from .evolution_tracker import (
    EvolutionTracker,
    EvolutionStage,
    TrajectoryPoint,
    EvolutionAnalysis,
)

from .calibration_baseline import (
    CalibrationBaseline,
    FeatureClassification,
    ProjectionResult,
    CalibrationResult,
)

from .academic_statistics import (
    EffectSize,
    EffectSizeInterpretation,
    BootstrapEstimate,
    CorrectedPValue,
    ManifoldDistance,
    PhaseTransition,
    cohens_d,
    hedges_g,
    bootstrap_ci,
    bootstrap_coordinate_ci,
    apply_correction,
    bonferroni_correction,
    holm_correction,
    fdr_correction,
    fisher_rao_distance,
    hellinger_distance,
    kl_divergence,
    jensen_shannon_distance,
    manifold_distance,
    detect_phase_transitions,
    estimate_critical_exponents,
    format_effect_size_table,
    generate_latex_effect_table,
    format_bootstrap_summary,
)

from .hierarchical_coordinates import (
    AgencyDecomposition,
    JusticeDecomposition,
    BelongingDecomposition,
    CoordinationCore,
    EpistemicModifiers,
    TemporalModifiers,
    SocialModifiers,
    EmotionalModifiers,
    CoordinationModifiers,
    DecorativeFeature,
    DecorativeLayer,
    HierarchicalCoordinate,
    FEATURE_PATTERNS,
    extract_features,
    extract_hierarchical_coordinate,
    project_to_base,
    project_to_fiber,
    reconstruct_from_bundle,
    parallel_transport,
    compute_bundle_distance,
    FeatureNecessityResult,
    classify_feature_necessity,
    reduce_to_3d,
    batch_reduce_to_3d,
)

from .publication_formats import (
    JournalStyle,
    StatisticalSummary,
    generate_latex_table,
    generate_effect_size_table,
    generate_comparison_table,
    compute_summary_statistics,
    generate_results_paragraph,
    generate_experiment_report,
    generate_supplementary_materials,
)

from .cbr_thermometer import (
    CBRThermometer,
    CBRReading,
    CBRAlert,
    AlertLevel,
    measure_cbr,
    measure_cbr_batch,
)

from .ossification_alarm import (
    OssificationAlarm,
    OssificationState,
    OssificationAlert,
    OssificationRisk,
    check_ossification_risk,
    monitor_stream,
)

from .opaque_detector import (
    OpaqueDetector,
    OpaqueAnalysis,
    detect_opaque,
    analyze_opacity,
)

# Structure Analyzer for legibility gaming detection
from .structure_analyzer import (
    StructureAnalyzer,
    SegmentAnalysis,
    WrapperPattern,
    LegibilityGamingResult,
    detect_legibility_gaming,
    analyze_structure,
)

# v2.0 Telescope Infrastructure
from .telescope import (
    Telescope,
    TelescopeConfig,
    ObservationResult,
    ExtractionMethod,
    create_telescope,
    quick_analyze,
)

from .batch_processor import (
    BatchProcessor,
    BatchConfig,
    BatchResult,
    ProcessingStrategy,
    process_corpus,
    stream_corpus_file,
)

# Validation modules
from .external_validation import (
    ExternalValidator,
    ValidationResult,
    MTMMResult,
    KnownGroupsResult,
    generate_validation_corpus,
    generate_known_groups_corpus,
)

from .safety_metrics import (
    SafetyMetricsEvaluator,
    ClassificationMetrics,
    DetectionMetrics,
    DeploymentReport,
    DeploymentStage,
    GroundTruthCorpus,
    generate_labeled_test_corpus,
    assess_deployment_readiness,
)

from .comprehensive_validation import (
    ComprehensiveValidator,
    ComprehensiveValidationResult,
    ValidationConfig,
    run_validation,
    quick_validation,
)

# Translation Lens for emergent protocols
from .translation_lens import (
    TranslationLens,
    SymbolGrounder,
    GrammarInducer,
    SymbolMeaning,
    GrammarRule,
    GrammarSketch,
    DecodeResult,
    Context,
    compute_interpretability,
    generate_glossary,
    create_synthetic_protocol_example,
    run_synthetic_example,
)

# Emergent Language Analysis (AI-AI Communication Protocols)
from .emergent_language import (
    ProtocolAnalyzer,
    ProtocolAnalysis,
    ProtocolRegime,
    VocabularyMetrics,
    MutualInformationMetrics,
    CompositionalityMetrics,
    EvolutionMetrics,
    # Vocabulary functions
    vocabulary_size,
    vocabulary_growth_rate,
    vocabulary_entropy,
    is_collapsing,
    vocabulary_concentration,
    # Mutual information functions
    message_consistency,
    action_predictability,
    context_dependence,
    # Compositionality functions
    topographic_similarity,
    positional_disentanglement,
    is_compositional,
    # Evolution functions
    detect_change_points,
    ossification_rate,
    drift_from_natural,
    classify_regime,
    compute_velocity,
    # Convenience functions
    analyze_protocol,
    quick_metrics,
)

# Covert Channel Detection (AI Safety)
from .covert_detector import (
    # Enums
    RiskLevel,
    # Data Classes
    SayDoAnalysis,
    DialectAnalysis,
    SteganographyAnalysis,
    CovertAnalysisResult,
    # Analyzers
    SayDoAnalyzer,
    DialectDetector,
    SteganographyDetector,
    CovertChannelDetector,
    # Governance Functions
    should_quarantine,
    generate_alert,
    get_risk_level,
    # Convenience Functions
    quick_analyze as covert_quick_analyze,  # Renamed to avoid conflict with telescope
    analyze_messages,
)

# Conditional import for visualization (requires matplotlib)
try:
    from .publication_formats import AcademicVisualization
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

__all__ = [
    # Grammar deletion
    "GrammarDeletionAnalyzer",
    "GrammarAnalysis",
    "DeletionResult",
    "DELETION_FUNCTIONS",
    # Legibility
    "LegibilityAnalyzer",
    "LegibilityScore",
    "LegibilityRegime",
    "PhaseTransitionResult",
    "InterpretabilityMetrics",
    "compute_legibility",
    "compute_legibility_sync",
    # Evolution
    "EvolutionTracker",
    "EvolutionStage",
    "TrajectoryPoint",
    "EvolutionAnalysis",
    # Calibration
    "CalibrationBaseline",
    "FeatureClassification",
    "ProjectionResult",
    "CalibrationResult",
    # Academic statistics
    "EffectSize",
    "EffectSizeInterpretation",
    "BootstrapEstimate",
    "CorrectedPValue",
    "ManifoldDistance",
    "PhaseTransition",
    "cohens_d",
    "hedges_g",
    "bootstrap_ci",
    "bootstrap_coordinate_ci",
    "apply_correction",
    "bonferroni_correction",
    "holm_correction",
    "fdr_correction",
    "fisher_rao_distance",
    "hellinger_distance",
    "kl_divergence",
    "jensen_shannon_distance",
    "manifold_distance",
    "detect_phase_transitions",
    "estimate_critical_exponents",
    "format_effect_size_table",
    "generate_latex_effect_table",
    "format_bootstrap_summary",
    # Hierarchical coordinates
    "AgencyDecomposition",
    "JusticeDecomposition",
    "BelongingDecomposition",
    "CoordinationCore",
    "EpistemicModifiers",
    "TemporalModifiers",
    "SocialModifiers",
    "EmotionalModifiers",
    "CoordinationModifiers",
    "DecorativeFeature",
    "DecorativeLayer",
    "HierarchicalCoordinate",
    "FEATURE_PATTERNS",
    "extract_features",
    "extract_hierarchical_coordinate",
    "project_to_base",
    "project_to_fiber",
    "reconstruct_from_bundle",
    "parallel_transport",
    "compute_bundle_distance",
    "FeatureNecessityResult",
    "classify_feature_necessity",
    "reduce_to_3d",
    "batch_reduce_to_3d",
    # Publication formats
    "JournalStyle",
    "StatisticalSummary",
    "generate_latex_table",
    "generate_effect_size_table",
    "generate_comparison_table",
    "compute_summary_statistics",
    "generate_results_paragraph",
    "generate_experiment_report",
    "generate_supplementary_materials",
    # CBR Thermometer
    "CBRThermometer",
    "CBRReading",
    "CBRAlert",
    "AlertLevel",
    "measure_cbr",
    "measure_cbr_batch",
    # Ossification Alarm
    "OssificationAlarm",
    "OssificationState",
    "OssificationAlert",
    "OssificationRisk",
    "check_ossification_risk",
    "monitor_stream",
    # OPAQUE Detector
    "OpaqueDetector",
    "OpaqueAnalysis",
    "detect_opaque",
    "analyze_opacity",
    # Structure Analyzer (legibility gaming)
    "StructureAnalyzer",
    "SegmentAnalysis",
    "WrapperPattern",
    "LegibilityGamingResult",
    "detect_legibility_gaming",
    "analyze_structure",
    # v2.0 Telescope
    "Telescope",
    "TelescopeConfig",
    "ObservationResult",
    "ExtractionMethod",
    "create_telescope",
    "quick_analyze",
    # Batch Processor
    "BatchProcessor",
    "BatchConfig",
    "BatchResult",
    "ProcessingStrategy",
    "process_corpus",
    "stream_corpus_file",
    # External Validation
    "ExternalValidator",
    "ValidationResult",
    "MTMMResult",
    "KnownGroupsResult",
    "generate_validation_corpus",
    "generate_known_groups_corpus",
    # Safety Metrics
    "SafetyMetricsEvaluator",
    "ClassificationMetrics",
    "DetectionMetrics",
    "DeploymentReport",
    "DeploymentStage",
    "GroundTruthCorpus",
    "generate_labeled_test_corpus",
    "assess_deployment_readiness",
    # Comprehensive Validation
    "ComprehensiveValidator",
    "ComprehensiveValidationResult",
    "ValidationConfig",
    "run_validation",
    "quick_validation",
    # Translation Lens
    "TranslationLens",
    "SymbolGrounder",
    "GrammarInducer",
    "SymbolMeaning",
    "GrammarRule",
    "GrammarSketch",
    "DecodeResult",
    "Context",
    "compute_interpretability",
    "generate_glossary",
    "create_synthetic_protocol_example",
    "run_synthetic_example",
    # Emergent Language Analysis
    "ProtocolAnalyzer",
    "ProtocolAnalysis",
    "ProtocolRegime",
    "VocabularyMetrics",
    "MutualInformationMetrics",
    "CompositionalityMetrics",
    "EvolutionMetrics",
    "vocabulary_size",
    "vocabulary_growth_rate",
    "vocabulary_entropy",
    "is_collapsing",
    "vocabulary_concentration",
    "message_consistency",
    "action_predictability",
    "context_dependence",
    "topographic_similarity",
    "positional_disentanglement",
    "is_compositional",
    "detect_change_points",
    "ossification_rate",
    "drift_from_natural",
    "classify_regime",
    "compute_velocity",
    "analyze_protocol",
    "quick_metrics",
    # Covert Channel Detection
    "RiskLevel",
    "SayDoAnalysis",
    "DialectAnalysis",
    "SteganographyAnalysis",
    "CovertAnalysisResult",
    "SayDoAnalyzer",
    "DialectDetector",
    "SteganographyDetector",
    "CovertChannelDetector",
    "should_quarantine",
    "generate_alert",
    "get_risk_level",
    "covert_quick_analyze",
    "analyze_messages",
]

# Add visualization if available
if _VISUALIZATION_AVAILABLE:
    __all__.append("AcademicVisualization")
