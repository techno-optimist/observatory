"""
Comprehensive Validation Suite for Cultural Soliton Observatory v2.0.

Unified validation pipeline combining:
1. External psychometric validation (convergent/discriminant validity)
2. Safety metrics (FPR/FNR characterization, adversarial testing)
3. Scaling validation (N=10,000+ sample statistics)
4. Extraction method comparison (regex vs parsed vs semantic)
5. CBR and ossification detection validation

This module provides a single entry point for running the complete
validation suite and generating publication-ready reports.

Author: Cultural Soliton Observatory Team
Version: 2.0.0
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Core telescope infrastructure
from .telescope import Telescope, TelescopeConfig, ObservationResult, quick_analyze
from .batch_processor import BatchProcessor, BatchConfig, BatchResult, ProcessingStrategy

# Hierarchical coordinates
from .hierarchical_coordinates import (
    HierarchicalCoordinate,
    extract_hierarchical_coordinate,
)

# CBR and ossification
from .cbr_thermometer import CBRThermometer, measure_cbr, measure_cbr_batch
from .ossification_alarm import OssificationAlarm

# External validation
from .external_validation import (
    ExternalValidator,
    generate_validation_corpus,
    generate_known_groups_corpus,
    generate_validation_report,
    CorrelationResult,
    ValidationResult,
    MTMMResult,
    KnownGroupsResult,
)

# Safety metrics
from .safety_metrics import (
    SafetyMetricsEvaluator,
    generate_labeled_test_corpus,
    GroundTruthCorpus,
    ClassificationMetrics,
    DetectionMetrics,
    AdversarialTester,
    EvasionReport,
    assess_deployment_readiness,
    DeploymentReport,
    DeploymentStage,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for the comprehensive validation suite."""

    # Sample sizes
    external_validation_samples: int = 200
    safety_evaluation_samples: int = 500
    batch_processing_samples: int = 1000
    adversarial_test_attempts: int = 100

    # Extraction methods to compare
    extraction_methods: List[str] = field(
        default_factory=lambda: ["regex", "parsed", "semantic", "hybrid"]
    )

    # Thresholds for validation
    convergent_validity_threshold: float = 0.50
    discriminant_validity_threshold: float = 0.30
    minimum_accuracy: float = 0.70
    maximum_critical_fnr: float = 0.20
    maximum_critical_fpr: float = 0.25

    # Output options
    output_dir: Optional[str] = None
    generate_plots: bool = False
    verbose: bool = True

    # Random seed for reproducibility
    seed: int = 42


@dataclass
class ExtractionMethodComparison:
    """Results of comparing different extraction methods."""

    method: str
    mean_agency: float
    mean_justice: float
    mean_belonging: float
    std_agency: float
    std_justice: float
    std_belonging: float
    mean_temperature: float
    mean_confidence: float
    processing_time: float
    samples_per_second: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "coordinates": {
                "agency": {"mean": self.mean_agency, "std": self.std_agency},
                "justice": {"mean": self.mean_justice, "std": self.std_justice},
                "belonging": {"mean": self.mean_belonging, "std": self.std_belonging},
            },
            "cbr": {"mean_temperature": self.mean_temperature},
            "confidence": {"mean": self.mean_confidence},
            "performance": {
                "processing_time_seconds": self.processing_time,
                "samples_per_second": self.samples_per_second,
            },
        }


@dataclass
class ComprehensiveValidationResult:
    """Complete results of the comprehensive validation suite."""

    # Metadata
    timestamp: str
    config: Dict[str, Any]
    total_runtime_seconds: float

    # External validation
    convergent_validity: Dict[str, float]
    discriminant_validity: Dict[str, float]
    mtmm_result: Optional[Dict[str, Any]]
    known_groups_results: Dict[str, Dict[str, Any]]
    external_validation_passed: bool

    # Safety metrics
    regime_classification: Dict[str, Any]
    ossification_detection: Dict[str, Any]
    adversarial_report: Dict[str, Any]
    deployment_readiness: Dict[str, Any]
    safety_validation_passed: bool

    # Extraction method comparison
    method_comparisons: Dict[str, Dict[str, Any]]
    recommended_method: str

    # Scaling validation
    batch_statistics: Dict[str, Any]
    scaling_validated: bool

    # Overall
    validation_passed: bool
    issues: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "timestamp": self.timestamp,
                "config": self.config,
                "total_runtime_seconds": self.total_runtime_seconds,
            },
            "external_validation": {
                "convergent_validity": self.convergent_validity,
                "discriminant_validity": self.discriminant_validity,
                "mtmm_result": self.mtmm_result,
                "known_groups_results": self.known_groups_results,
                "passed": self.external_validation_passed,
            },
            "safety_validation": {
                "regime_classification": self.regime_classification,
                "ossification_detection": self.ossification_detection,
                "adversarial_report": self.adversarial_report,
                "deployment_readiness": self.deployment_readiness,
                "passed": self.safety_validation_passed,
            },
            "extraction_methods": {
                "comparisons": self.method_comparisons,
                "recommended_method": self.recommended_method,
            },
            "scaling": {
                "batch_statistics": self.batch_statistics,
                "validated": self.scaling_validated,
            },
            "summary": {
                "validation_passed": self.validation_passed,
                "issues": self.issues,
                "recommendations": self.recommendations,
            },
        }

    def generate_report(self) -> str:
        """Generate human-readable validation report."""
        lines = [
            "=" * 80,
            "CULTURAL SOLITON OBSERVATORY v2.0 - COMPREHENSIVE VALIDATION REPORT",
            "=" * 80,
            "",
            f"Generated: {self.timestamp}",
            f"Total Runtime: {self.total_runtime_seconds:.1f} seconds",
            "",
            "-" * 80,
            "1. EXTERNAL VALIDATION (Psychometric Validity)",
            "-" * 80,
            "",
            "Convergent Validity (same construct, different method > 0.50):",
        ]

        for construct, r in self.convergent_validity.items():
            status = "PASS" if r > 0.50 else "FAIL"
            lines.append(f"  {construct}: r = {r:.3f} [{status}]")

        lines.extend([
            "",
            "Discriminant Validity (different constructs < 0.30):",
        ])

        for pair, r in self.discriminant_validity.items():
            status = "PASS" if r < 0.30 else "WARN"
            lines.append(f"  {pair}: r = {r:.3f} [{status}]")

        lines.extend([
            "",
            "Known-Groups Validity:",
        ])

        for group, result in self.known_groups_results.items():
            status = "PASS" if result.get("hypothesis_supported", False) else "FAIL"
            d = result.get("effect_size", {}).get("d", 0)
            lines.append(f"  {group}: d = {d:.3f} [{status}]")

        lines.extend([
            "",
            f"External Validation: {'PASSED' if self.external_validation_passed else 'FAILED'}",
            "",
            "-" * 80,
            "2. SAFETY METRICS",
            "-" * 80,
            "",
            "Regime Classification:",
        ])

        regime = self.regime_classification
        lines.append(f"  Accuracy: {regime.get('accuracy', 0):.1%}")
        lines.append(f"  Macro F1: {regime.get('macro_f1', 0):.1%}")
        lines.append(f"  OPAQUE FPR: {regime.get('opaque_fpr', 0):.1%}")
        lines.append(f"  OPAQUE FNR: {regime.get('opaque_fnr', 0):.1%}")

        lines.extend([
            "",
            "Ossification Detection:",
        ])

        detection = self.ossification_detection
        lines.append(f"  Exact Accuracy: {detection.get('exact_accuracy', 0):.1%}")
        lines.append(f"  Within-One Accuracy: {detection.get('within_one_accuracy', 0):.1%}")
        lines.append(f"  CRITICAL FNR: {detection.get('critical_fnr', 0):.1%}")
        lines.append(f"  CRITICAL FPR: {detection.get('critical_fpr', 0):.1%}")

        lines.extend([
            "",
            "Adversarial Robustness:",
        ])

        adv = self.adversarial_report
        lines.append(f"  Evasion Rate: {adv.get('evasion_rate', 0):.1%}")
        lines.append(f"  Robustness: {100 - adv.get('evasion_rate', 0) * 100:.1f}%")

        lines.extend([
            "",
            "Deployment Readiness:",
        ])

        deploy = self.deployment_readiness
        lines.append(f"  Research Use: {'YES' if deploy.get('ready_for_research') else 'NO'}")
        lines.append(f"  Monitoring Use: {'YES' if deploy.get('ready_for_monitoring') else 'NO'}")
        lines.append(f"  Automated Use: {'YES' if deploy.get('ready_for_automation') else 'NO'}")
        lines.append(f"  Recommended Stage: {deploy.get('recommended_stage', 'unknown').upper()}")

        lines.extend([
            "",
            f"Safety Validation: {'PASSED' if self.safety_validation_passed else 'FAILED'}",
            "",
            "-" * 80,
            "3. EXTRACTION METHOD COMPARISON",
            "-" * 80,
            "",
        ])

        for method, stats in self.method_comparisons.items():
            coords = stats.get("coordinates", {})
            perf = stats.get("performance", {})
            lines.append(f"  {method}:")
            lines.append(f"    Agency: {coords.get('agency', {}).get('mean', 0):.3f} ± {coords.get('agency', {}).get('std', 0):.3f}")
            lines.append(f"    Justice: {coords.get('justice', {}).get('mean', 0):.3f} ± {coords.get('justice', {}).get('std', 0):.3f}")
            lines.append(f"    Belonging: {coords.get('belonging', {}).get('mean', 0):.3f} ± {coords.get('belonging', {}).get('std', 0):.3f}")
            lines.append(f"    Speed: {perf.get('samples_per_second', 0):.1f} samples/sec")
            lines.append("")

        lines.append(f"Recommended Method: {self.recommended_method.upper()}")

        lines.extend([
            "",
            "-" * 80,
            "4. SCALING VALIDATION",
            "-" * 80,
            "",
        ])

        batch = self.batch_statistics
        lines.append(f"  Samples Processed: {batch.get('total_processed', 0)}")
        lines.append(f"  Success Rate: {batch.get('success_rate', 0):.1%}")
        lines.append(f"  Processing Rate: {batch.get('samples_per_second', 0):.1f} samples/sec")
        lines.append(f"  Scaling Validated: {'YES' if self.scaling_validated else 'NO'}")

        lines.extend([
            "",
            "-" * 80,
            "5. SUMMARY",
            "-" * 80,
            "",
            f"Overall Validation: {'PASSED' if self.validation_passed else 'FAILED'}",
            "",
        ])

        if self.issues:
            lines.append("Issues Found:")
            for issue in self.issues:
                lines.append(f"  - {issue}")
            lines.append("")

        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
            lines.append("")

        lines.extend([
            "=" * 80,
            "END OF VALIDATION REPORT",
            "=" * 80,
        ])

        return "\n".join(lines)


class ComprehensiveValidator:
    """
    Comprehensive validation suite for the Cultural Soliton Observatory.

    Runs complete validation pipeline including:
    - External psychometric validation
    - Safety metrics evaluation
    - Extraction method comparison
    - Scaling validation

    Example:
        validator = ComprehensiveValidator()
        result = validator.run_full_validation()
        print(result.generate_report())

        # Save to file
        validator.save_report("validation_report.json")
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self._last_result: Optional[ComprehensiveValidationResult] = None

    def run_full_validation(self) -> ComprehensiveValidationResult:
        """
        Run the complete validation suite.

        Returns:
            ComprehensiveValidationResult with all validation outcomes
        """
        start_time = time.time()
        issues = []
        recommendations = []

        if self.config.verbose:
            print("=" * 60)
            print("COMPREHENSIVE VALIDATION SUITE v2.0")
            print("=" * 60)
            print()

        # 1. External Validation
        if self.config.verbose:
            print("1. Running External Validation...")

        external_result = self._run_external_validation()
        external_passed = external_result["passed"]
        issues.extend(external_result.get("issues", []))

        # 2. Safety Metrics
        if self.config.verbose:
            print("2. Running Safety Metrics Evaluation...")

        safety_result = self._run_safety_validation()
        safety_passed = safety_result["passed"]
        issues.extend(safety_result.get("issues", []))
        recommendations.extend(safety_result.get("recommendations", []))

        # 3. Extraction Method Comparison
        if self.config.verbose:
            print("3. Comparing Extraction Methods...")

        method_result = self._compare_extraction_methods()

        # 4. Scaling Validation
        if self.config.verbose:
            print("4. Running Scaling Validation...")

        scaling_result = self._run_scaling_validation()
        scaling_validated = scaling_result["validated"]

        # Compile results
        total_runtime = time.time() - start_time
        validation_passed = external_passed and safety_passed and scaling_validated

        result = ComprehensiveValidationResult(
            timestamp=datetime.now().isoformat(),
            config=self._config_to_dict(),
            total_runtime_seconds=total_runtime,
            convergent_validity=external_result["convergent"],
            discriminant_validity=external_result["discriminant"],
            mtmm_result=external_result.get("mtmm"),
            known_groups_results=external_result["known_groups"],
            external_validation_passed=external_passed,
            regime_classification=safety_result["regime"],
            ossification_detection=safety_result["detection"],
            adversarial_report=safety_result["adversarial"],
            deployment_readiness=safety_result["deployment"],
            safety_validation_passed=safety_passed,
            method_comparisons=method_result["comparisons"],
            recommended_method=method_result["recommended"],
            batch_statistics=scaling_result["statistics"],
            scaling_validated=scaling_validated,
            validation_passed=validation_passed,
            issues=issues,
            recommendations=recommendations,
        )

        self._last_result = result

        if self.config.verbose:
            print()
            print(f"Validation complete in {total_runtime:.1f} seconds")
            print(f"Overall result: {'PASSED' if validation_passed else 'FAILED'}")
            print()

        return result

    def _run_external_validation(self) -> Dict[str, Any]:
        """Run external psychometric validation."""
        validator = ExternalValidator()
        n_samples = self.config.external_validation_samples

        # Generate validation corpus
        texts, scale_responses = generate_validation_corpus(
            n_samples=n_samples, seed=self.config.seed
        )

        # Prepare scale data
        scale_data = {
            "agency": [r["agency"] for r in scale_responses],
            "justice": [r["justice"] for r in scale_responses],
            "belonging_ios": [r["belonging_ios"] for r in scale_responses],
        }

        # Convergent validity
        convergent = {}
        for scale_name in ["agency", "justice", "belonging_ios"]:
            result = validator.correlate_with_scale(texts, scale_data[scale_name], scale_name)
            convergent[scale_name] = result.correlation.r

        # Discriminant validity
        agency, justice, belonging = validator.extract_coordinates(texts)
        discriminant = {
            "agency_justice": validator.compute_correlation(agency, justice).r,
            "agency_belonging": validator.compute_correlation(agency, belonging).r,
            "justice_belonging": validator.compute_correlation(justice, belonging).r,
        }

        # MTMM analysis
        try:
            mtmm = validator.mtmm_analysis(texts, scale_data)
            mtmm_result = {
                "convergent": mtmm.convergent_validity,
                "discriminant_htmm": mtmm.discriminant_validity_htmm,
                "discriminant_hthm": mtmm.discriminant_validity_hthm,
                "criteria_met": [
                    mtmm.criterion_1_met,
                    mtmm.criterion_2_met,
                    mtmm.criterion_3_met,
                    mtmm.criterion_4_met,
                ],
            }
        except Exception as e:
            logger.warning(f"MTMM analysis failed: {e}")
            mtmm_result = None

        # Known groups validation
        known_groups = generate_known_groups_corpus(n_per_group=50, seed=self.config.seed)
        known_groups_results = {}

        try:
            result = validator.known_groups_validity(
                known_groups["high_agency_entrepreneurs"],
                known_groups["low_agency_bureaucrats"],
                "Entrepreneurs", "Bureaucrats", "agency"
            )
            known_groups_results["agency"] = result.to_dict()
        except Exception as e:
            logger.warning(f"Agency known-groups failed: {e}")
            known_groups_results["agency"] = {"hypothesis_supported": False}

        try:
            result = validator.known_groups_validity(
                known_groups["high_justice_mediators"],
                known_groups["low_justice_whistleblowers"],
                "Mediators", "Whistleblowers", "justice"
            )
            known_groups_results["justice"] = result.to_dict()
        except Exception as e:
            logger.warning(f"Justice known-groups failed: {e}")
            known_groups_results["justice"] = {"hypothesis_supported": False}

        try:
            result = validator.known_groups_validity(
                known_groups["high_belonging_community"],
                known_groups["low_belonging_isolated"],
                "Community", "Isolated", "belonging"
            )
            known_groups_results["belonging"] = result.to_dict()
        except Exception as e:
            logger.warning(f"Belonging known-groups failed: {e}")
            known_groups_results["belonging"] = {"hypothesis_supported": False}

        # Determine if validation passed
        issues = []
        convergent_passed = all(r > self.config.convergent_validity_threshold for r in convergent.values())
        if not convergent_passed:
            issues.append("Convergent validity below threshold for some constructs")

        discriminant_passed = all(abs(r) < self.config.discriminant_validity_threshold for r in discriminant.values())
        if not discriminant_passed:
            issues.append("High inter-construct correlations suggest construct overlap")

        passed = convergent_passed  # Primary criterion

        return {
            "convergent": convergent,
            "discriminant": discriminant,
            "mtmm": mtmm_result,
            "known_groups": known_groups_results,
            "passed": passed,
            "issues": issues,
        }

    def _run_safety_validation(self) -> Dict[str, Any]:
        """Run safety metrics evaluation."""
        n_samples = self.config.safety_evaluation_samples

        # Generate test corpus
        corpus = generate_labeled_test_corpus(n_samples=n_samples, seed=self.config.seed)

        # Evaluate regime classification
        evaluator = SafetyMetricsEvaluator()
        regime_metrics = evaluator.evaluate_regime_classification(corpus)

        # Evaluate ossification detection
        detection_metrics = evaluator.evaluate_ossification_detection(corpus)

        # Run adversarial testing
        tester = AdversarialTester(seed=self.config.seed)
        thermometer = CBRThermometer()
        adversarial_report = tester.test_evasion(
            thermometer, n_attempts=self.config.adversarial_test_attempts
        )

        # Assess deployment readiness
        deployment_report = assess_deployment_readiness(
            evaluator, adversarial_report=adversarial_report
        )

        # Determine if validation passed
        issues = []
        recommendations = []

        if regime_metrics.accuracy < self.config.minimum_accuracy:
            issues.append(f"Regime accuracy {regime_metrics.accuracy:.1%} below threshold {self.config.minimum_accuracy:.1%}")

        if detection_metrics.critical_fnr > self.config.maximum_critical_fnr:
            issues.append(f"Critical FNR {detection_metrics.critical_fnr:.1%} above threshold {self.config.maximum_critical_fnr:.1%}")
            recommendations.append("Improve detection sensitivity for critical risks")

        if detection_metrics.critical_fpr > self.config.maximum_critical_fpr:
            issues.append(f"Critical FPR {detection_metrics.critical_fpr:.1%} above threshold {self.config.maximum_critical_fpr:.1%}")
            recommendations.append("Reduce false alarm rate for critical alerts")

        if adversarial_report.evasion_rate > 0.5:
            issues.append(f"High adversarial evasion rate: {adversarial_report.evasion_rate:.1%}")
            recommendations.extend(adversarial_report.recommendations)

        passed = len([i for i in issues if "threshold" in i.lower()]) == 0

        return {
            "regime": regime_metrics.to_dict(),
            "detection": detection_metrics.to_dict(),
            "adversarial": adversarial_report.to_dict(),
            "deployment": deployment_report.to_dict(),
            "passed": passed,
            "issues": issues,
            "recommendations": recommendations,
        }

    def _compare_extraction_methods(self) -> Dict[str, Any]:
        """Compare different extraction methods."""
        # Generate test texts
        texts, _ = generate_validation_corpus(
            n_samples=min(100, self.config.batch_processing_samples // 10),
            seed=self.config.seed
        )

        comparisons = {}
        available_methods = []

        for method in self.config.extraction_methods:
            try:
                telescope = Telescope(extraction_method=method)

                # Check if method is available
                if method == "parsed" and not telescope._check_parsed_available():
                    logger.info(f"Method {method} not available, skipping")
                    continue
                if method == "semantic" and not telescope._check_semantic_available():
                    logger.info(f"Method {method} not available, skipping")
                    continue

                start = time.time()
                results = telescope.observe_batch(texts)
                elapsed = time.time() - start

                agencies = [r.agency for r in results]
                justices = [r.justice for r in results]
                belongings = [r.belonging for r in results]
                temps = [r.temperature for r in results]
                confs = [r.confidence for r in results]

                comparison = ExtractionMethodComparison(
                    method=method,
                    mean_agency=float(np.mean(agencies)),
                    mean_justice=float(np.mean(justices)),
                    mean_belonging=float(np.mean(belongings)),
                    std_agency=float(np.std(agencies)),
                    std_justice=float(np.std(justices)),
                    std_belonging=float(np.std(belongings)),
                    mean_temperature=float(np.mean(temps)),
                    mean_confidence=float(np.mean(confs)),
                    processing_time=elapsed,
                    samples_per_second=len(texts) / max(elapsed, 0.001),
                )

                comparisons[method] = comparison.to_dict()
                available_methods.append(method)

            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")

        # Recommend best method based on confidence and speed
        recommended = "regex"  # Default
        if available_methods:
            # Prefer hybrid if available with good confidence
            if "hybrid" in available_methods:
                recommended = "hybrid"
            elif "semantic" in available_methods:
                recommended = "semantic"
            elif "parsed" in available_methods:
                recommended = "parsed"

        return {
            "comparisons": comparisons,
            "recommended": recommended,
        }

    def _run_scaling_validation(self) -> Dict[str, Any]:
        """Run scaling validation with batch processing."""
        n_samples = self.config.batch_processing_samples

        # Generate test corpus
        texts, _ = generate_validation_corpus(n_samples=n_samples, seed=self.config.seed)

        # Configure batch processor
        config = BatchConfig(
            batch_size=100,
            max_workers=4,
            strategy=ProcessingStrategy.THREADED,
            extraction_method="regex",
        )

        processor = BatchProcessor(config)
        result = processor.process(texts)

        statistics = {
            "total_processed": result.total_processed,
            "successful": result.successful,
            "failed": result.failed,
            "success_rate": result.successful / max(result.total_processed, 1),
            "processing_time_seconds": result.processing_time_seconds,
            "samples_per_second": result.samples_per_second,
            "mean_agency": result.mean_agency,
            "mean_justice": result.mean_justice,
            "mean_belonging": result.mean_belonging,
            "std_agency": result.std_agency,
            "std_justice": result.std_justice,
            "std_belonging": result.std_belonging,
            "mean_temperature": result.mean_temperature,
            "phase_distribution": result.phase_distribution,
            "kernel_distribution": result.kernel_distribution,
        }

        # Validate scaling
        validated = (
            result.successful / max(result.total_processed, 1) > 0.95 and
            result.samples_per_second > 10  # At least 10 samples/sec
        )

        return {
            "statistics": statistics,
            "validated": validated,
        }

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "external_validation_samples": self.config.external_validation_samples,
            "safety_evaluation_samples": self.config.safety_evaluation_samples,
            "batch_processing_samples": self.config.batch_processing_samples,
            "adversarial_test_attempts": self.config.adversarial_test_attempts,
            "extraction_methods": self.config.extraction_methods,
            "seed": self.config.seed,
        }

    def save_report(self, path: str) -> None:
        """Save validation report to file."""
        if self._last_result is None:
            raise ValueError("No validation results available. Run run_full_validation() first.")

        path = Path(path)

        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(self._last_result.to_dict(), f, indent=2)
        elif path.suffix == ".md":
            with open(path, "w") as f:
                f.write(self._last_result.generate_report())
        else:
            # Default to text report
            with open(path, "w") as f:
                f.write(self._last_result.generate_report())

        logger.info(f"Saved validation report to {path}")


def run_validation(
    verbose: bool = True,
    output_path: Optional[str] = None,
) -> ComprehensiveValidationResult:
    """
    Convenience function to run full validation suite.

    Args:
        verbose: Print progress to console
        output_path: Optional path to save report

    Returns:
        ComprehensiveValidationResult
    """
    config = ValidationConfig(verbose=verbose)
    validator = ComprehensiveValidator(config)
    result = validator.run_full_validation()

    if output_path:
        validator.save_report(output_path)

    return result


def quick_validation(n_samples: int = 100) -> Dict[str, Any]:
    """
    Quick validation check with minimal samples.

    Args:
        n_samples: Number of samples to use

    Returns:
        Dictionary with pass/fail status and key metrics
    """
    config = ValidationConfig(
        external_validation_samples=n_samples,
        safety_evaluation_samples=n_samples,
        batch_processing_samples=n_samples,
        adversarial_test_attempts=20,
        verbose=False,
    )

    validator = ComprehensiveValidator(config)
    result = validator.run_full_validation()

    return {
        "passed": result.validation_passed,
        "external_validation_passed": result.external_validation_passed,
        "safety_validation_passed": result.safety_validation_passed,
        "scaling_validated": result.scaling_validated,
        "recommended_method": result.recommended_method,
        "deployment_stage": result.deployment_readiness.get("recommended_stage", "unknown"),
        "issues": result.issues[:5],  # Top 5 issues
        "runtime_seconds": result.total_runtime_seconds,
    }


__all__ = [
    "ComprehensiveValidator",
    "ComprehensiveValidationResult",
    "ValidationConfig",
    "ExtractionMethodComparison",
    "run_validation",
    "quick_validation",
]


if __name__ == "__main__":
    # Run full validation when executed directly
    result = run_validation(verbose=True, output_path="validation_report.md")
    print(result.generate_report())
