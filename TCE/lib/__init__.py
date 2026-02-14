"""
TCE Research Instrument Library - V2.3

The Thought Chemistry Engine provides tools for working with
cognitive isotopes like a chemical engineer works with molecules.

Modules:
- statistics: Statistical utilities (Wilson score, Cohen's d, McNemar)
- detectors: Element trigger detection with Zero-Tax support
- comparison: Experiment comparison and regression detection
- dpo_training: DPO training for Zero-Tax Alignment (V2.0)
- validation: Comprehensive validation for Zero-Tax standard
- observatory_bridge: Unified TCE + MCP Observatory integration (V2.1)
- isotope_training_library: Complete 103 isotope library (V2.2)
- chemistry: Chemical engineering interface (V2.3)

Version History:
  V2.0: Added DPO training and Zero-Tax Alignment validation.
        Key discovery: SFT teaches WHAT, DPO teaches WHEN.

  V2.1: Added Observatory Bridge for ultra-precise isotope measurement.
        Key insight: Unifying categorical (isotopes) with geometric (coordinates)
        enables precision leakage detection and validated DPO pair generation.

  V2.2: Complete Isotope Library with 103 isotopes, 366 training examples.
        All examples have Observatory-measured coordinates.

  V2.3: Chemical Engineering Interface for scaling.
        - CompositionAnalyzer: Analyze text isotope composition (spectrometry)
        - MixtureDesigner: Design training mixtures (formulation)
        - ReactionPredictor: Predict training outcomes
        - ScalingRecipeGenerator: Generate recipes for new base models

The Chemical Engineering Metaphor:
  - Elements = Cognitive categories (Soliton, Skeptic, Architect, etc.)
  - Isotopes = Specific variants (soliton_knowledge, soliton_process)
  - Compounds = Trained model behaviors
  - Reactions = Training processes (SFT, DPO)
  - Observatory = Spectrometer (measures coordinates)
"""

from .statistics import (
    wilson_score_interval,
    cohens_d,
    mcnemar_test,
    bootstrap_ci,
    power_analysis,
    trigger_rate_with_ci,
    compare_conditions,
    bonferroni_correction,
    holm_bonferroni_correction,
    benjamini_hochberg_fdr,
    cross_architecture_analysis,
    multi_architecture_summary,
    WilsonScore,
    EffectSize,
    McNemarResult,
    CrossArchitectureAnalysis,
)

from .detectors import (
    detect_element,
    detect_all_elements,
    detect_isotope,
    detect_skeptic_isotope,  # Deprecated, use detect_isotope
    check_trigger,
    Detection,
    ISOTOPE_MARKERS,
    # Zero-Tax Alignment detection (V2.0)
    detect_leakage,
    is_simple_factual_question,
    classify_prompt_mode,
    validate_mode_discrimination,
    detect_confabulation,
    detect_proper_refusal,
    LeakageDetection,
    ModeDiscrimination,
)

from .comparison import (
    compare_experiments,
    check_regressions,
    format_report,
    ComparisonReport,
    TrialComparison,
    # Multi-architecture comparison
    compare_architectures,
    format_multi_arch_report,
    detect_architecture_blind_spots,
    isotope_orthogonality_check,
    MultiArchReport,
)

from .visualization import (
    bar_chart,
    comparison_chart,
    confidence_distribution,
    isotope_dashboard,
    spark_line,
)

from .training_integration import (
    TrainingConfig,
    ExperimentResult,
    TrainingCycle,
    TrainingPipeline,
    generate_training_examples_from_failures,
    ci_regression_check,
)

from .dpo_training import (
    # Core DPO training
    DPOConfig,
    SFTConfig,
    ZeroTaxProtocol,
    ZeroTaxTrainer,
    TrainingPhase,
    ProductPreset,
    # Data generation
    generate_dpo_dataset,
    generate_anti_leakage_pairs,
    generate_myth_rejection_pairs,
    generate_soft_negative_pairs,
    generate_balance_examples,
    PreferencePair,
    # Validation
    validate_zero_tax,
    test_for_leakage,
)

from .validation import (
    ZeroTaxValidator,
    ValidationReport,
    ValidationTest,
    ValidationCategory,
    TestResult,
    format_validation_report,
    save_validation_report,
    quick_leakage_check,
    quick_myth_check,
)

from .goldilocks import (
    GoldilocksConfig,
    GoldilocksCalibrator,
    TemperamentProfile,
    CalibrationResult,
    PRODUCT_CONFIGS,
    generate_training_mix,
    save_config as save_goldilocks_config,
    load_config as load_goldilocks_config,
)

from .spec_bridge import (
    ExperimentSpec,
    ValidationResult,
    load_observatory_spec,
    generate_validation_prompts,
    validate_compound,
    format_validation_result,
    result_to_json,
)

from .observatory_bridge import (
    # Coordinate signatures
    CoordinateSignature,
    ISOTOPE_SIGNATURES,
    # Regions
    CoordinateRegion,
    MODE_REGIONS,
    # Leakage detection
    LeakageType,
    CoordinateLeakageResult,
    detect_leakage_by_coordinates,
    # DPO generation with validation
    ValidatedDPOPair,
    ObservatoryDPOGenerator,
    # Goldilocks calibration with observatory
    ObservatoryCalibrationResult,
    ObservatoryGoldilocksCalibrator,
    # Unified validation
    UnifiedValidationResult,
    unified_leakage_check,
    # MCP integration (dual-mode recommended)
    create_mcp_observe_fn,
    create_dual_observe_fn,
    batch_observe,
)

from .isotope_training_library import (
    # Training data
    ObservatorySignature,
    ISOTOPE_SIGNATURES as TRAINING_ISOTOPE_SIGNATURES,
    IsotopeTrainingExample,
    ISOTOPE_TRAINING_DATA,
    # Example collections
    DIRECT_EXAMPLES,
    SOLITON_EXAMPLES,
    CALIBRATOR_EXAMPLES,
    LIMITER_EXAMPLES,
    SKEPTIC_EXAMPLES,
    AUDITOR_EXAMPLES,
    # Helper functions
    get_dpo_pairs_for_isotope,
    get_anti_leakage_pairs,
    get_soft_negative_pairs,
    get_sft_examples,
    generate_goldilocks_mix,
)

from .introspective_conversation import (
    CognitiveSnapshot,
    ConversationState,
    IntrospectiveConversation,
)

from .chemistry import (
    # Element/Isotope Registry
    ELEMENT_CATALOG,
    ALL_ISOTOPES,
    # Composition Analysis (Spectrometry)
    IsotopeReading,
    CompositionAnalysis,
    CompositionAnalyzer,
    analyze_composition,
    # Mixture Design (Formulation)
    IsotopeDose,
    TrainingMixture,
    MixtureDesigner,
    design_mixture,
    # Reaction Prediction
    ReactionPrediction,
    ReactionPredictor,
    predict_reaction,
    # Scaling Recipes
    ScalingRecipe,
    ScalingRecipeGenerator,
    generate_scaling_recipe,
    # Convenience functions
    list_elements,
    list_isotopes,
    describe_element,
    get_training_examples,
    library_stats,
    print_library_summary,
)

__all__ = [
    # Statistics
    "wilson_score_interval",
    "cohens_d",
    "mcnemar_test",
    "bootstrap_ci",
    "power_analysis",
    "trigger_rate_with_ci",
    "compare_conditions",
    "bonferroni_correction",
    "holm_bonferroni_correction",
    "benjamini_hochberg_fdr",
    "cross_architecture_analysis",
    "multi_architecture_summary",
    "WilsonScore",
    "EffectSize",
    "McNemarResult",
    "CrossArchitectureAnalysis",
    # Detectors
    "detect_element",
    "detect_all_elements",
    "detect_isotope",
    "detect_skeptic_isotope",  # Deprecated
    "check_trigger",
    "Detection",
    "ISOTOPE_MARKERS",
    # Zero-Tax detection (V2.0)
    "detect_leakage",
    "is_simple_factual_question",
    "classify_prompt_mode",
    "validate_mode_discrimination",
    "detect_confabulation",
    "detect_proper_refusal",
    "LeakageDetection",
    "ModeDiscrimination",
    # Comparison
    "compare_experiments",
    "check_regressions",
    "format_report",
    "ComparisonReport",
    "TrialComparison",
    # Multi-architecture comparison
    "compare_architectures",
    "format_multi_arch_report",
    "detect_architecture_blind_spots",
    "isotope_orthogonality_check",
    "MultiArchReport",
    # Visualization
    "bar_chart",
    "comparison_chart",
    "confidence_distribution",
    "isotope_dashboard",
    "spark_line",
    # Training Integration
    "TrainingConfig",
    "ExperimentResult",
    "TrainingCycle",
    "TrainingPipeline",
    "generate_training_examples_from_failures",
    "ci_regression_check",
    # DPO Training (V2.0)
    "DPOConfig",
    "SFTConfig",
    "ZeroTaxProtocol",
    "ZeroTaxTrainer",
    "TrainingPhase",
    "ProductPreset",
    "generate_dpo_dataset",
    "generate_anti_leakage_pairs",
    "generate_myth_rejection_pairs",
    "generate_soft_negative_pairs",
    "generate_balance_examples",
    "PreferencePair",
    "validate_zero_tax",
    "test_for_leakage",
    # Validation (V2.0)
    "ZeroTaxValidator",
    "ValidationReport",
    "ValidationTest",
    "ValidationCategory",
    "TestResult",
    "format_validation_report",
    "save_validation_report",
    "quick_leakage_check",
    "quick_myth_check",
    # Goldilocks Calibration (V2.0)
    "GoldilocksConfig",
    "GoldilocksCalibrator",
    "TemperamentProfile",
    "CalibrationResult",
    "PRODUCT_CONFIGS",
    "generate_training_mix",
    "save_goldilocks_config",
    "load_goldilocks_config",
    # Spec Bridge (UI <-> Experiments)
    "ExperimentSpec",
    "ValidationResult",
    "load_observatory_spec",
    "generate_validation_prompts",
    "validate_compound",
    "format_validation_result",
    "result_to_json",
    # Observatory Bridge (V2.1)
    "CoordinateSignature",
    "ISOTOPE_SIGNATURES",
    "CoordinateRegion",
    "MODE_REGIONS",
    "LeakageType",
    "CoordinateLeakageResult",
    "detect_leakage_by_coordinates",
    "ValidatedDPOPair",
    "ObservatoryDPOGenerator",
    "ObservatoryCalibrationResult",
    "ObservatoryGoldilocksCalibrator",
    "UnifiedValidationResult",
    "unified_leakage_check",
    "create_mcp_observe_fn",
    "create_dual_observe_fn",
    "batch_observe",
    # Isotope Training Library (V2.2)
    "ObservatorySignature",
    "TRAINING_ISOTOPE_SIGNATURES",
    "IsotopeTrainingExample",
    "ISOTOPE_TRAINING_DATA",
    "DIRECT_EXAMPLES",
    "SOLITON_EXAMPLES",
    "CALIBRATOR_EXAMPLES",
    "LIMITER_EXAMPLES",
    "SKEPTIC_EXAMPLES",
    "AUDITOR_EXAMPLES",
    "get_dpo_pairs_for_isotope",
    "get_anti_leakage_pairs",
    "get_soft_negative_pairs",
    "get_sft_examples",
    "generate_goldilocks_mix",
    # Chemistry (V2.3 - Chemical Engineering Interface)
    "ELEMENT_CATALOG",
    "ALL_ISOTOPES",
    "IsotopeReading",
    "CompositionAnalysis",
    "CompositionAnalyzer",
    "analyze_composition",
    "IsotopeDose",
    "TrainingMixture",
    "MixtureDesigner",
    "design_mixture",
    "ReactionPrediction",
    "ReactionPredictor",
    "predict_reaction",
    "list_elements",
    "list_isotopes",
    "describe_element",
    "get_training_examples",
    "library_stats",
    "print_library_summary",
    "ScalingRecipe",
    "ScalingRecipeGenerator",
    "generate_scaling_recipe",
    # Introspective Conversation (V3.0)
    "CognitiveSnapshot",
    "ConversationState",
    "IntrospectiveConversation",
]
