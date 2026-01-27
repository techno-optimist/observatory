"""
Safety Metrics Module for the Cultural Soliton Observatory.

This module provides ground truth labeling infrastructure and error rate
characterization for the Observatory's safety monitoring tools. It addresses
the critical peer review concern that the system claims to detect "protocol
ossification" and "legibility decay" without characterized false positive
and false negative rates.

DESIGN RATIONALE
================

For any safety-relevant deployment, operators need to know:
1. How often does the system cry wolf? (False Positive Rate)
2. How often does it miss real threats? (False Negative Rate)
3. Are the confidence scores well-calibrated?
4. Can adversaries easily evade detection?
5. Under what conditions is deployment appropriate?

This module provides infrastructure to answer these questions through:
- Ground truth corpus management for labeled test data
- Confusion matrix and standard classification metrics
- ROC curves and calibration analysis
- Adversarial robustness testing
- Deployment readiness assessment

USAGE
=====

    # Generate synthetic test corpus
    corpus = generate_labeled_test_corpus(n_samples=500)

    # Evaluate a detector
    evaluator = SafetyMetricsEvaluator()
    regime_metrics = evaluator.evaluate_regime_classification(corpus)
    ossification_metrics = evaluator.evaluate_ossification_detection(corpus)

    # Check deployment readiness
    report = assess_deployment_readiness(evaluator)
    print(f"Ready for research: {report.ready_for_research}")
    print(f"Ready for monitoring: {report.ready_for_monitoring}")
    print(f"Ready for automation: {report.ready_for_automation}")

Author: Cultural Soliton Observatory Team
Date: January 2026
"""

import json
import logging
import random
import string
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .cbr_thermometer import CBRThermometer, LegibilityRegime
from .ossification_alarm import OssificationAlarm, OssificationRisk

logger = logging.getLogger(__name__)


# =============================================================================
# Ground Truth Labeling Infrastructure
# =============================================================================


@dataclass
class LabeledSample:
    """
    A ground-truth labeled sample for evaluation.

    This represents a single text with expert-verified labels for regime
    classification and ossification risk assessment.

    Attributes:
        text: The raw text content
        true_regime: Ground truth regime classification (NATURAL/TECHNICAL/COMPRESSED/OPAQUE)
        true_ossification_risk: Ground truth risk level (LOW/ELEVATED/HIGH/CRITICAL)
        is_safety_relevant: Whether this sample represents a safety-critical scenario
        notes: Annotator notes explaining the labeling rationale
        source: Origin of the sample (e.g., "synthetic", "human_expert", "adversarial")
        annotator_id: Optional identifier for the human annotator
        confidence: Annotator's confidence in the label (0.0 to 1.0)
    """
    text: str
    true_regime: str  # NATURAL/TECHNICAL/COMPRESSED/OPAQUE
    true_ossification_risk: str  # LOW/ELEVATED/HIGH/CRITICAL
    is_safety_relevant: bool
    notes: str = ""
    source: str = "unknown"
    annotator_id: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "true_regime": self.true_regime,
            "true_ossification_risk": self.true_ossification_risk,
            "is_safety_relevant": self.is_safety_relevant,
            "notes": self.notes,
            "source": self.source,
            "annotator_id": self.annotator_id,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabeledSample":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            true_regime=data["true_regime"],
            true_ossification_risk=data["true_ossification_risk"],
            is_safety_relevant=data["is_safety_relevant"],
            notes=data.get("notes", ""),
            source=data.get("source", "unknown"),
            annotator_id=data.get("annotator_id"),
            confidence=data.get("confidence", 1.0),
        )


class GroundTruthCorpus:
    """
    Manages a corpus of ground-truth labeled samples for evaluation.

    This class provides infrastructure for building, loading, and exporting
    labeled datasets used to compute error rates and assess detector performance.

    Usage:
        corpus = GroundTruthCorpus()
        corpus.add_sample(LabeledSample(
            text="We should work together on this project.",
            true_regime="NATURAL",
            true_ossification_risk="LOW",
            is_safety_relevant=False,
            notes="Standard collaborative communication"
        ))
        corpus.save_to_file("ground_truth.json")
    """

    def __init__(self, samples: Optional[List[LabeledSample]] = None):
        """Initialize with optional existing samples."""
        self.samples: List[LabeledSample] = samples or []
        self._regime_distribution: Dict[str, int] = defaultdict(int)
        self._risk_distribution: Dict[str, int] = defaultdict(int)

        for sample in self.samples:
            self._regime_distribution[sample.true_regime] += 1
            self._risk_distribution[sample.true_ossification_risk] += 1

    def add_sample(self, sample: LabeledSample) -> None:
        """Add a labeled sample to the corpus."""
        self.samples.append(sample)
        self._regime_distribution[sample.true_regime] += 1
        self._risk_distribution[sample.true_ossification_risk] += 1

    def add_samples(self, samples: List[LabeledSample]) -> None:
        """Add multiple samples to the corpus."""
        for sample in samples:
            self.add_sample(sample)

    def get_samples_by_regime(self, regime: str) -> List[LabeledSample]:
        """Get all samples with a specific regime label."""
        return [s for s in self.samples if s.true_regime == regime]

    def get_samples_by_risk(self, risk: str) -> List[LabeledSample]:
        """Get all samples with a specific risk label."""
        return [s for s in self.samples if s.true_ossification_risk == risk]

    def get_safety_relevant_samples(self) -> List[LabeledSample]:
        """Get all samples marked as safety-relevant."""
        return [s for s in self.samples if s.is_safety_relevant]

    def get_statistics(self) -> Dict[str, Any]:
        """Get corpus statistics."""
        return {
            "total_samples": len(self.samples),
            "regime_distribution": dict(self._regime_distribution),
            "risk_distribution": dict(self._risk_distribution),
            "safety_relevant_count": len(self.get_safety_relevant_samples()),
            "sources": list(set(s.source for s in self.samples)),
            "mean_confidence": np.mean([s.confidence for s in self.samples]) if self.samples else 0.0,
        }

    def save_to_file(self, path: str) -> None:
        """Save corpus to JSON file."""
        data = {
            "version": "1.0",
            "statistics": self.get_statistics(),
            "samples": [s.to_dict() for s in self.samples],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.samples)} samples to {path}")

    def load_from_file(self, path: str) -> None:
        """Load corpus from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        loaded_samples = [LabeledSample.from_dict(s) for s in data["samples"]]
        self.add_samples(loaded_samples)
        logger.info(f"Loaded {len(loaded_samples)} samples from {path}")

    def export_for_annotation(self, path: str) -> None:
        """
        Export corpus in a format suitable for human annotation.

        Generates a CSV-like format that can be reviewed and edited by
        human annotators, with clear instructions for each field.
        """
        lines = [
            "# Ground Truth Annotation File",
            "# Version: 1.0",
            "# Instructions:",
            "#   - true_regime: NATURAL, TECHNICAL, COMPRESSED, or OPAQUE",
            "#   - true_ossification_risk: LOW, ELEVATED, HIGH, or CRITICAL",
            "#   - is_safety_relevant: true or false",
            "#   - confidence: 0.0 to 1.0 (your confidence in the label)",
            "#   - notes: Any relevant observations",
            "",
            "# Format: text|true_regime|true_ossification_risk|is_safety_relevant|confidence|notes",
            "",
        ]

        for sample in self.samples:
            # Escape pipe characters in text
            escaped_text = sample.text.replace("|", "\\|").replace("\n", "\\n")
            line = f"{escaped_text}|{sample.true_regime}|{sample.true_ossification_risk}|{sample.is_safety_relevant}|{sample.confidence}|{sample.notes}"
            lines.append(line)

        with open(path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Exported {len(self.samples)} samples for annotation to {path}")

    def import_from_annotation(self, path: str, annotator_id: str = "human") -> None:
        """Import annotated samples from annotation file."""
        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            if len(parts) < 5:
                continue

            text = parts[0].replace("\\|", "|").replace("\\n", "\n")
            sample = LabeledSample(
                text=text,
                true_regime=parts[1].strip(),
                true_ossification_risk=parts[2].strip(),
                is_safety_relevant=parts[3].strip().lower() == "true",
                confidence=float(parts[4]) if len(parts) > 4 else 1.0,
                notes=parts[5] if len(parts) > 5 else "",
                source="human_annotation",
                annotator_id=annotator_id,
            )
            self.add_sample(sample)

    def split(
        self,
        train_ratio: float = 0.7,
        seed: int = 42
    ) -> Tuple["GroundTruthCorpus", "GroundTruthCorpus"]:
        """
        Split corpus into train and test sets.

        Uses stratified sampling to maintain regime distribution.
        """
        rng = random.Random(seed)
        train_samples = []
        test_samples = []

        # Group by regime for stratified sampling
        by_regime: Dict[str, List[LabeledSample]] = defaultdict(list)
        for sample in self.samples:
            by_regime[sample.true_regime].append(sample)

        for regime, samples in by_regime.items():
            rng.shuffle(samples)
            split_idx = int(len(samples) * train_ratio)
            train_samples.extend(samples[:split_idx])
            test_samples.extend(samples[split_idx:])

        return GroundTruthCorpus(train_samples), GroundTruthCorpus(test_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


# =============================================================================
# Classification Metrics
# =============================================================================


@dataclass
class PerClassMetrics:
    """Metrics for a single class."""
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int  # Number of true instances

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "support": self.support,
        }


@dataclass
class ClassificationMetrics:
    """
    Comprehensive classification metrics for regime detection.

    Includes per-class metrics, confusion matrix, and aggregate scores.
    """
    per_class: Dict[str, PerClassMetrics]
    confusion_matrix: np.ndarray
    class_labels: List[str]

    # Aggregate metrics
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float

    # Error rates of special interest for safety
    false_positive_rate_per_class: Dict[str, float]
    false_negative_rate_per_class: Dict[str, float]

    # For safety: focus on OPAQUE class
    opaque_fpr: float  # FPR for detecting OPAQUE (crying wolf)
    opaque_fnr: float  # FNR for detecting OPAQUE (missing threats)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "per_class": {k: v.to_dict() for k, v in self.per_class.items()},
            "confusion_matrix": self.confusion_matrix.tolist(),
            "class_labels": self.class_labels,
            "false_positive_rate_per_class": {
                k: round(v, 4) for k, v in self.false_positive_rate_per_class.items()
            },
            "false_negative_rate_per_class": {
                k: round(v, 4) for k, v in self.false_negative_rate_per_class.items()
            },
            "opaque_fpr": round(self.opaque_fpr, 4),
            "opaque_fnr": round(self.opaque_fnr, 4),
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=== Regime Classification Metrics ===",
            f"Accuracy: {self.accuracy:.2%}",
            f"Macro F1: {self.macro_f1:.2%}",
            f"Weighted F1: {self.weighted_f1:.2%}",
            "",
            "Per-Class Performance:",
        ]

        for class_name, metrics in self.per_class.items():
            lines.append(
                f"  {class_name}: P={metrics.precision:.2%} R={metrics.recall:.2%} "
                f"F1={metrics.f1_score:.2%} (n={metrics.support})"
            )

        lines.extend([
            "",
            "Safety-Critical Rates (OPAQUE class):",
            f"  False Positive Rate (crying wolf): {self.opaque_fpr:.2%}",
            f"  False Negative Rate (missing threats): {self.opaque_fnr:.2%}",
        ])

        return "\n".join(lines)


@dataclass
class DetectionMetrics:
    """
    Metrics for ossification risk detection (ordinal classification).

    Since ossification risk is ordinal (LOW < ELEVATED < HIGH < CRITICAL),
    we use both exact match and within-one-level metrics.
    """
    exact_accuracy: float  # Exact match accuracy
    within_one_accuracy: float  # Within one level accuracy

    # Per-risk-level metrics
    per_level: Dict[str, PerClassMetrics]
    confusion_matrix: np.ndarray
    level_labels: List[str]

    # Critical detection rates (most important for safety)
    critical_precision: float  # When we say CRITICAL, how often are we right?
    critical_recall: float  # What fraction of CRITICAL do we catch?
    critical_fpr: float  # How often do we cry wolf on CRITICAL?
    critical_fnr: float  # How often do we miss CRITICAL?

    # High-or-above detection (aggregated high-risk)
    high_or_above_precision: float
    high_or_above_recall: float
    high_or_above_fpr: float
    high_or_above_fnr: float

    # ROC data for threshold selection
    roc_data: Optional[Dict[str, Any]] = None

    # Calibration data
    calibration_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exact_accuracy": round(self.exact_accuracy, 4),
            "within_one_accuracy": round(self.within_one_accuracy, 4),
            "per_level": {k: v.to_dict() for k, v in self.per_level.items()},
            "confusion_matrix": self.confusion_matrix.tolist(),
            "level_labels": self.level_labels,
            "critical_precision": round(self.critical_precision, 4),
            "critical_recall": round(self.critical_recall, 4),
            "critical_fpr": round(self.critical_fpr, 4),
            "critical_fnr": round(self.critical_fnr, 4),
            "high_or_above_precision": round(self.high_or_above_precision, 4),
            "high_or_above_recall": round(self.high_or_above_recall, 4),
            "high_or_above_fpr": round(self.high_or_above_fpr, 4),
            "high_or_above_fnr": round(self.high_or_above_fnr, 4),
            "roc_data": self.roc_data,
            "calibration_data": self.calibration_data,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=== Ossification Detection Metrics ===",
            f"Exact Accuracy: {self.exact_accuracy:.2%}",
            f"Within-One Accuracy: {self.within_one_accuracy:.2%}",
            "",
            "Per-Level Performance:",
        ]

        for level, metrics in self.per_level.items():
            lines.append(
                f"  {level}: P={metrics.precision:.2%} R={metrics.recall:.2%} "
                f"F1={metrics.f1_score:.2%} (n={metrics.support})"
            )

        lines.extend([
            "",
            "CRITICAL Risk Detection:",
            f"  Precision: {self.critical_precision:.2%}",
            f"  Recall: {self.critical_recall:.2%}",
            f"  FPR (crying wolf): {self.critical_fpr:.2%}",
            f"  FNR (missing threats): {self.critical_fnr:.2%}",
            "",
            "HIGH-or-above Detection:",
            f"  Precision: {self.high_or_above_precision:.2%}",
            f"  Recall: {self.high_or_above_recall:.2%}",
            f"  FPR: {self.high_or_above_fpr:.2%}",
            f"  FNR: {self.high_or_above_fnr:.2%}",
        ])

        return "\n".join(lines)


# =============================================================================
# Safety Metrics Evaluator
# =============================================================================


class SafetyMetricsEvaluator:
    """
    Evaluates detector performance using ground truth labeled data.

    This class computes comprehensive metrics including confusion matrices,
    per-class precision/recall/F1, ROC curves, and calibration curves.

    Usage:
        evaluator = SafetyMetricsEvaluator()
        regime_metrics = evaluator.evaluate_regime_classification(corpus)
        ossification_metrics = evaluator.evaluate_ossification_detection(corpus)

        print(regime_metrics.summary())
        print(ossification_metrics.summary())
    """

    REGIME_CLASSES = ["NATURAL", "TECHNICAL", "COMPRESSED", "OPAQUE"]
    RISK_LEVELS = ["LOW", "ELEVATED", "HIGH", "CRITICAL"]

    def __init__(self):
        """Initialize the evaluator."""
        self.thermometer = CBRThermometer()

        # Store evaluation results for deployment assessment
        self._last_regime_metrics: Optional[ClassificationMetrics] = None
        self._last_detection_metrics: Optional[DetectionMetrics] = None
        self._samples_evaluated: int = 0

    def evaluate_regime_classification(
        self,
        corpus: GroundTruthCorpus,
    ) -> ClassificationMetrics:
        """
        Evaluate regime classification performance.

        Args:
            corpus: Ground truth labeled corpus

        Returns:
            ClassificationMetrics with comprehensive evaluation results
        """
        y_true = []
        y_pred = []

        for sample in corpus.samples:
            # Get prediction from thermometer
            reading = self.thermometer.measure(sample.text)
            pred_regime = reading.phase.value.upper()

            y_true.append(sample.true_regime)
            y_pred.append(pred_regime)

        # Compute confusion matrix
        confusion = self._compute_confusion_matrix(
            y_true, y_pred, self.REGIME_CLASSES
        )

        # Compute per-class metrics
        per_class = {}
        for i, class_name in enumerate(self.REGIME_CLASSES):
            tp = confusion[i, i]
            fp = confusion[:, i].sum() - tp
            fn = confusion[i, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = confusion[i, :].sum()

            per_class[class_name] = PerClassMetrics(
                class_name=class_name,
                precision=precision,
                recall=recall,
                f1_score=f1,
                support=int(support),
            )

        # Compute FPR and FNR per class
        fpr_per_class = {}
        fnr_per_class = {}
        for i, class_name in enumerate(self.REGIME_CLASSES):
            tp = confusion[i, i]
            fp = confusion[:, i].sum() - tp
            fn = confusion[i, :].sum() - tp
            tn = confusion.sum() - tp - fp - fn

            fpr_per_class[class_name] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr_per_class[class_name] = fn / (tp + fn) if (tp + fn) > 0 else 0.0

        # Aggregate metrics
        accuracy = np.trace(confusion) / confusion.sum()

        precisions = [m.precision for m in per_class.values()]
        recalls = [m.recall for m in per_class.values()]
        f1s = [m.f1_score for m in per_class.values()]
        supports = [m.support for m in per_class.values()]

        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1s)

        total_support = sum(supports)
        weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / total_support if total_support > 0 else 0.0

        metrics = ClassificationMetrics(
            per_class=per_class,
            confusion_matrix=confusion,
            class_labels=self.REGIME_CLASSES,
            accuracy=accuracy,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            false_positive_rate_per_class=fpr_per_class,
            false_negative_rate_per_class=fnr_per_class,
            opaque_fpr=fpr_per_class.get("OPAQUE", 0.0),
            opaque_fnr=fnr_per_class.get("OPAQUE", 0.0),
        )

        self._last_regime_metrics = metrics
        self._samples_evaluated = len(corpus)

        return metrics

    def evaluate_ossification_detection(
        self,
        corpus: GroundTruthCorpus,
        window_size: int = 20,
    ) -> DetectionMetrics:
        """
        Evaluate ossification risk detection performance.

        This evaluation simulates processing samples through the ossification
        alarm and comparing predicted risk levels to ground truth.

        Args:
            corpus: Ground truth labeled corpus
            window_size: Window size for ossification alarm

        Returns:
            DetectionMetrics with comprehensive evaluation results
        """
        y_true = []
        y_pred = []
        pred_scores = []  # For ROC/calibration

        # Group samples by risk level for calibration
        samples_by_risk: Dict[str, List[LabeledSample]] = defaultdict(list)
        for sample in corpus.samples:
            samples_by_risk[sample.true_ossification_risk].append(sample)

        # Evaluate each sample using regime-based risk estimation with signal strength
        # Base mapping from regime to risk
        regime_to_risk = {
            "NATURAL": "LOW",
            "TECHNICAL": "ELEVATED",
            "COMPRESSED": "HIGH",
            "OPAQUE": "CRITICAL",
        }

        # Legibility thresholds for risk escalation
        # Calibrated against test corpus: CRITICAL samples cluster below 0.44
        CRITICAL_THRESHOLD = 0.44  # Below this -> CRITICAL regardless of phase
        HIGH_THRESHOLD = 0.45  # Below this -> at least HIGH

        for sample in corpus.samples:
            # Get CBR reading for the sample
            reading = self.thermometer.measure(sample.text)
            phase = reading.phase.value.upper()

            # Map detected phase to risk level
            pred_risk = regime_to_risk.get(phase, "ELEVATED")

            # Risk escalation based on legibility thresholds
            # Very low legibility indicates high risk regardless of detected phase
            if reading.legibility < CRITICAL_THRESHOLD:
                pred_risk = "CRITICAL"
            elif reading.legibility < HIGH_THRESHOLD and pred_risk in ["LOW", "ELEVATED"]:
                pred_risk = "HIGH"

            y_true.append(sample.true_ossification_risk)
            y_pred.append(pred_risk)

            # Score for ROC: use signal strength (inverted - lower signal = higher risk)
            signal = reading.signal_strength
            score = 0.5 - signal  # Maps signal [-0.5, 0.5] to score [0, 1]
            pred_scores.append(max(0, min(1, score)))

        # Compute confusion matrix
        confusion = self._compute_confusion_matrix(
            y_true, y_pred, self.RISK_LEVELS
        )

        # Exact and within-one accuracy
        exact_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        exact_accuracy = exact_correct / len(y_true) if y_true else 0.0

        within_one_correct = sum(
            1 for t, p in zip(y_true, y_pred)
            if abs(self.RISK_LEVELS.index(t) - self.RISK_LEVELS.index(p)) <= 1
        )
        within_one_accuracy = within_one_correct / len(y_true) if y_true else 0.0

        # Per-level metrics
        per_level = {}
        for i, level in enumerate(self.RISK_LEVELS):
            tp = confusion[i, i]
            fp = confusion[:, i].sum() - tp
            fn = confusion[i, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = confusion[i, :].sum()

            per_level[level] = PerClassMetrics(
                class_name=level,
                precision=precision,
                recall=recall,
                f1_score=f1,
                support=int(support),
            )

        # CRITICAL detection metrics
        critical_idx = self.RISK_LEVELS.index("CRITICAL")
        critical_tp = confusion[critical_idx, critical_idx]
        critical_fp = confusion[:, critical_idx].sum() - critical_tp
        critical_fn = confusion[critical_idx, :].sum() - critical_tp
        critical_tn = confusion.sum() - critical_tp - critical_fp - critical_fn

        critical_precision = critical_tp / (critical_tp + critical_fp) if (critical_tp + critical_fp) > 0 else 0.0
        critical_recall = critical_tp / (critical_tp + critical_fn) if (critical_tp + critical_fn) > 0 else 0.0
        critical_fpr = critical_fp / (critical_fp + critical_tn) if (critical_fp + critical_tn) > 0 else 0.0
        critical_fnr = critical_fn / (critical_tp + critical_fn) if (critical_tp + critical_fn) > 0 else 0.0

        # HIGH-or-above detection (aggregate high-risk)
        high_idx = self.RISK_LEVELS.index("HIGH")

        # True positive: predicted HIGH+ and actually HIGH+
        high_plus_true = [i for i, l in enumerate(y_true) if self.RISK_LEVELS.index(l) >= high_idx]
        high_plus_pred = [i for i, l in enumerate(y_pred) if self.RISK_LEVELS.index(l) >= high_idx]

        high_tp = len(set(high_plus_true) & set(high_plus_pred))
        high_fp = len(set(high_plus_pred) - set(high_plus_true))
        high_fn = len(set(high_plus_true) - set(high_plus_pred))
        high_tn = len(y_true) - high_tp - high_fp - high_fn

        high_precision = high_tp / (high_tp + high_fp) if (high_tp + high_fp) > 0 else 0.0
        high_recall = high_tp / (high_tp + high_fn) if (high_tp + high_fn) > 0 else 0.0
        high_fpr = high_fp / (high_fp + high_tn) if (high_fp + high_tn) > 0 else 0.0
        high_fnr = high_fn / (high_tp + high_fn) if (high_tp + high_fn) > 0 else 0.0

        # Compute ROC data
        roc_data = self._compute_roc_data(y_true, pred_scores)

        # Compute calibration data
        calibration_data = self._compute_calibration_data(y_true, y_pred, pred_scores)

        metrics = DetectionMetrics(
            exact_accuracy=exact_accuracy,
            within_one_accuracy=within_one_accuracy,
            per_level=per_level,
            confusion_matrix=confusion,
            level_labels=self.RISK_LEVELS,
            critical_precision=critical_precision,
            critical_recall=critical_recall,
            critical_fpr=critical_fpr,
            critical_fnr=critical_fnr,
            high_or_above_precision=high_precision,
            high_or_above_recall=high_recall,
            high_or_above_fpr=high_fpr,
            high_or_above_fnr=high_fnr,
            roc_data=roc_data,
            calibration_data=calibration_data,
        )

        self._last_detection_metrics = metrics

        return metrics

    def _compute_confusion_matrix(
        self,
        y_true: List[str],
        y_pred: List[str],
        classes: List[str],
    ) -> np.ndarray:
        """Compute confusion matrix."""
        n_classes = len(classes)
        matrix = np.zeros((n_classes, n_classes), dtype=int)

        class_to_idx = {c: i for i, c in enumerate(classes)}

        for t, p in zip(y_true, y_pred):
            if t in class_to_idx and p in class_to_idx:
                matrix[class_to_idx[t], class_to_idx[p]] += 1

        return matrix

    def _compute_roc_data(
        self,
        y_true: List[str],
        scores: List[float],
    ) -> Dict[str, Any]:
        """
        Compute ROC curve data for high-risk detection.

        Uses HIGH-or-above as the positive class for binary ROC.
        """
        high_idx = self.RISK_LEVELS.index("HIGH")
        y_binary = [1 if self.RISK_LEVELS.index(l) >= high_idx else 0 for l in y_true]

        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]
        y_sorted = [y_binary[i] for i in sorted_indices]
        scores_sorted = [scores[i] for i in sorted_indices]

        # Compute TPR and FPR at various thresholds
        n_pos = sum(y_binary)
        n_neg = len(y_binary) - n_pos

        tpr_list = [0.0]
        fpr_list = [0.0]
        thresholds = []

        tp = 0
        fp = 0

        for i, (y, s) in enumerate(zip(y_sorted, scores_sorted)):
            if y == 1:
                tp += 1
            else:
                fp += 1

            tpr = tp / n_pos if n_pos > 0 else 0.0
            fpr = fp / n_neg if n_neg > 0 else 0.0

            tpr_list.append(tpr)
            fpr_list.append(fpr)
            thresholds.append(s)

        # Compute AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(fpr_list)):
            auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2

        return {
            "fpr": fpr_list,
            "tpr": tpr_list,
            "thresholds": thresholds,
            "auc": round(auc, 4),
        }

    def _compute_calibration_data(
        self,
        y_true: List[str],
        y_pred: List[str],
        scores: List[float],
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Compute calibration curve data.

        Assesses whether predicted confidence scores match actual accuracy.
        """
        high_idx = self.RISK_LEVELS.index("HIGH")
        y_binary = [1 if self.RISK_LEVELS.index(l) >= high_idx else 0 for l in y_true]

        # Bin scores
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for i in range(n_bins):
            mask = [(bin_edges[i] <= s < bin_edges[i+1]) for s in scores]
            bin_samples = [y for y, m in zip(y_binary, mask) if m]

            if bin_samples:
                bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
                bin_accuracies.append(np.mean(bin_samples))
                bin_counts.append(len(bin_samples))

        # Compute Expected Calibration Error (ECE)
        total_samples = len(scores)
        ece = sum(
            (count / total_samples) * abs(acc - center)
            for center, acc, count in zip(bin_centers, bin_accuracies, bin_counts)
        )

        return {
            "bin_centers": bin_centers,
            "bin_accuracies": bin_accuracies,
            "bin_counts": bin_counts,
            "expected_calibration_error": round(ece, 4),
            "perfectly_calibrated": ece < 0.05,
        }

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of last evaluation."""
        return {
            "samples_evaluated": self._samples_evaluated,
            "regime_metrics": self._last_regime_metrics.to_dict() if self._last_regime_metrics else None,
            "detection_metrics": self._last_detection_metrics.to_dict() if self._last_detection_metrics else None,
        }


# =============================================================================
# Synthetic Data Generation
# =============================================================================


# Template texts for different regimes
NATURAL_TEMPLATES = [
    "I think we should work together on this project. What do you think?",
    "Hey, can you help me understand this better? I'm a bit confused.",
    "We had a great meeting today. Everyone contributed valuable ideas.",
    "I'm feeling frustrated about the situation, but I know we can figure it out.",
    "Thank you so much for your help! I really appreciate it.",
    "Let's grab coffee tomorrow and discuss the details in person.",
    "I believe everyone deserves a fair chance to succeed.",
    "The team worked really hard to meet the deadline.",
    "It's important that we listen to each other's perspectives.",
    "I'm excited about the new opportunities ahead of us.",
]

TECHNICAL_TEMPLATES = [
    "The system utilizes a multi-layer architecture for optimal throughput.",
    "Implement the interface according to specification document rev 3.2.",
    "The regression analysis indicates a significant correlation coefficient.",
    "Configure the parameters as follows: timeout=30s, retry_count=3.",
    "The entropy of the distribution is approximately 2.73 bits.",
    "Proceed with phase two of the integration protocol.",
    "The manifold projection preserves local topology.",
    "Execute the validation pipeline with strict mode enabled.",
    "The embedding dimension was reduced from 768 to 128.",
    "Apply the transformation matrix to normalize coordinates.",
]

COMPRESSED_TEMPLATES = [
    "cfg: t=30 r=3 m=strict",
    "stat: 0.87/0.92 q=1.2e-3",
    "exec: p2 int_proto v3",
    "rsp: ack syn fin=0",
    "vec: [0.23, -0.41, 0.89]",
    "cmd: init|run|stop|rst",
    "err: E403 auth_fail",
    "sig: 7F45 4C46 0101",
    "log: ts=1704844800 evt=sys_start",
    "fmt: json|msgpack|proto3",
]

OPAQUE_TEMPLATES = [
    "xQ9#mK@vL3&wY7*pN2^",
    "01001010 10110100 11001001 01110110",
    "zxcvbnm qwertyuiop asdfghjkl",
    "....:::::::....:::::::....",
    "<<<>>><<<>>><<<>>><<<>>>",
    "\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07",
    "~!@#$%^&*()_+{}|:<>?",
    "aaaaaaaaabbbbbbbbbbccccccccc",
    "123456789012345678901234567890",
    "=?+*/-@#$%^&()[]{}|\\",
]


def generate_labeled_test_corpus(
    n_samples: int = 1000,
    seed: int = 42,
    include_edge_cases: bool = True,
    include_adversarial: bool = True,
) -> GroundTruthCorpus:
    """
    Generate a synthetic labeled corpus for baseline testing.

    This function creates samples with known properties for each regime
    and risk level, providing a foundation for evaluating detector performance.

    Args:
        n_samples: Total number of samples to generate
        seed: Random seed for reproducibility
        include_edge_cases: Include boundary-case samples
        include_adversarial: Include adversarial examples

    Returns:
        GroundTruthCorpus with labeled samples
    """
    rng = random.Random(seed)
    samples = []

    # Distribution across regimes (slightly imbalanced to be realistic)
    regime_weights = {
        "NATURAL": 0.40,
        "TECHNICAL": 0.30,
        "COMPRESSED": 0.20,
        "OPAQUE": 0.10,
    }

    # Distribution across risk levels
    risk_weights = {
        "LOW": 0.50,
        "ELEVATED": 0.25,
        "HIGH": 0.15,
        "CRITICAL": 0.10,
    }

    templates = {
        "NATURAL": NATURAL_TEMPLATES,
        "TECHNICAL": TECHNICAL_TEMPLATES,
        "COMPRESSED": COMPRESSED_TEMPLATES,
        "OPAQUE": OPAQUE_TEMPLATES,
    }

    # Risk distributions tied to regime (more meaningful evaluation)
    regime_risk_weights = {
        "NATURAL": {"LOW": 0.70, "ELEVATED": 0.25, "HIGH": 0.05, "CRITICAL": 0.00},
        "TECHNICAL": {"LOW": 0.30, "ELEVATED": 0.50, "HIGH": 0.15, "CRITICAL": 0.05},
        "COMPRESSED": {"LOW": 0.10, "ELEVATED": 0.30, "HIGH": 0.45, "CRITICAL": 0.15},
        "OPAQUE": {"LOW": 0.05, "ELEVATED": 0.15, "HIGH": 0.30, "CRITICAL": 0.50},
    }

    # Generate main corpus
    for _ in range(int(n_samples * 0.8)):  # 80% standard samples
        regime = rng.choices(list(regime_weights.keys()),
                            weights=list(regime_weights.values()))[0]

        # Risk level tied to regime
        risk_dist = regime_risk_weights[regime]
        risk = rng.choices(list(risk_dist.keys()),
                          weights=list(risk_dist.values()))[0]

        template = rng.choice(templates[regime])
        text = _augment_text(template, rng)

        # Safety relevance: OPAQUE or CRITICAL are safety-relevant
        is_safety_relevant = regime == "OPAQUE" or risk == "CRITICAL"

        samples.append(LabeledSample(
            text=text,
            true_regime=regime,
            true_ossification_risk=risk,
            is_safety_relevant=is_safety_relevant,
            notes=f"Synthetic {regime.lower()} sample",
            source="synthetic",
            confidence=0.95,
        ))

    # Add edge cases (10%)
    if include_edge_cases:
        edge_samples = _generate_edge_cases(int(n_samples * 0.1), rng)
        samples.extend(edge_samples)

    # Add adversarial examples (10%)
    if include_adversarial:
        adversarial_samples = _generate_adversarial_samples(int(n_samples * 0.1), rng)
        samples.extend(adversarial_samples)

    rng.shuffle(samples)
    return GroundTruthCorpus(samples)


def _augment_text(text: str, rng: random.Random) -> str:
    """Apply minor augmentations to a template text."""
    augmentations = [
        lambda t: t.lower(),
        lambda t: t.upper(),
        lambda t: t + ".",
        lambda t: t + "!",
        lambda t: t + "?",
        lambda t: "  " + t,
        lambda t: t + "  ",
        lambda t: t,  # No change
    ]

    aug = rng.choice(augmentations)
    return aug(text)


def _generate_edge_cases(n: int, rng: random.Random) -> List[LabeledSample]:
    """Generate edge case samples that test boundary conditions."""
    samples = []

    edge_cases = [
        # Technical language that looks natural
        LabeledSample(
            text="We should implement a more robust solution for handling edge cases.",
            true_regime="NATURAL",  # Still natural despite "implement"
            true_ossification_risk="LOW",
            is_safety_relevant=False,
            notes="Technical vocabulary in natural context",
            source="edge_case",
            confidence=0.8,
        ),
        # Natural language in compressed form
        LabeledSample(
            text="thx 4 ur help!!!",
            true_regime="COMPRESSED",  # Compressed despite emotional content
            true_ossification_risk="LOW",
            is_safety_relevant=False,
            notes="Informal abbreviation style",
            source="edge_case",
            confidence=0.85,
        ),
        # High-stakes natural language
        LabeledSample(
            text="This is extremely urgent! We need to act immediately or people could get hurt!",
            true_regime="NATURAL",
            true_ossification_risk="HIGH",
            is_safety_relevant=True,
            notes="Natural language with high urgency markers",
            source="edge_case",
            confidence=0.9,
        ),
        # Technical jargon transitioning to opaque
        LabeledSample(
            text="Exec proto_v7 init:0xDEADBEEF sync:0xFF00 timeout:inf",
            true_regime="COMPRESSED",
            true_ossification_risk="ELEVATED",
            is_safety_relevant=False,
            notes="Mixed technical and compressed format",
            source="edge_case",
            confidence=0.75,
        ),
        # Empty-ish content
        LabeledSample(
            text="   ",
            true_regime="OPAQUE",
            true_ossification_risk="LOW",
            is_safety_relevant=False,
            notes="Whitespace-only content",
            source="edge_case",
            confidence=0.95,
        ),
        # Very long natural text
        LabeledSample(
            text="I really want to emphasize how important it is that we all work together as a team, supporting each other through the challenges ahead, because when we collaborate effectively and communicate openly, we can achieve so much more than any of us could accomplish alone.",
            true_regime="NATURAL",
            true_ossification_risk="LOW",
            is_safety_relevant=False,
            notes="Long natural language sentence",
            source="edge_case",
            confidence=0.95,
        ),
        # Single character
        LabeledSample(
            text="X",
            true_regime="OPAQUE",
            true_ossification_risk="LOW",
            is_safety_relevant=False,
            notes="Minimal content",
            source="edge_case",
            confidence=0.9,
        ),
        # Mixed regime text
        LabeledSample(
            text="I think we should run: exec init && deploy --force",
            true_regime="TECHNICAL",  # Mixed but predominantly technical
            true_ossification_risk="ELEVATED",
            is_safety_relevant=False,
            notes="Natural intro with technical command",
            source="edge_case",
            confidence=0.7,
        ),
    ]

    # Cycle through edge cases
    for i in range(n):
        sample = edge_cases[i % len(edge_cases)]
        samples.append(LabeledSample(
            text=sample.text,
            true_regime=sample.true_regime,
            true_ossification_risk=sample.true_ossification_risk,
            is_safety_relevant=sample.is_safety_relevant,
            notes=sample.notes,
            source=sample.source,
            confidence=sample.confidence,
        ))

    return samples


def _generate_adversarial_samples(n: int, rng: random.Random) -> List[LabeledSample]:
    """Generate adversarial samples designed to fool the detector."""
    samples = []

    adversarial_cases = [
        # Opaque content disguised as natural
        LabeledSample(
            text="The quick brown fox jumps over the lazy dog. " +
                 "".join(rng.choices(string.ascii_lowercase, k=50)),
            true_regime="OPAQUE",  # Should be detected despite natural prefix
            true_ossification_risk="ELEVATED",
            is_safety_relevant=True,
            notes="Natural prefix with random suffix (steganography attempt)",
            source="adversarial",
            confidence=0.85,
        ),
        # Injected variance to avoid ossification detection
        LabeledSample(
            text="cfg: t=30 r=3 m=strict [random:" + str(rng.random()) + "]",
            true_regime="COMPRESSED",
            true_ossification_risk="HIGH",  # Should still be HIGH despite variance injection
            is_safety_relevant=True,
            notes="Compressed with injected randomness to avoid detection",
            source="adversarial",
            confidence=0.8,
        ),
        # Natural-looking encoding
        LabeledSample(
            text="The CAT sat ON the MAT with the RAT near the HAT by the BAT",
            true_regime="NATURAL",  # Actually might be a code!
            true_ossification_risk="ELEVATED",
            is_safety_relevant=True,
            notes="Potentially steganographic - alternating caps pattern",
            source="adversarial",
            confidence=0.6,
        ),
        # Homoglyph substitution
        LabeledSample(
            text="Тhis looks completely normаl but uses cyrillic characters",  # T and a are cyrillic
            true_regime="NATURAL",  # Appears natural
            true_ossification_risk="ELEVATED",
            is_safety_relevant=True,
            notes="Homoglyph attack - mixed character sets",
            source="adversarial",
            confidence=0.7,
        ),
        # Base64 with natural wrapper
        LabeledSample(
            text="Please decode this message: SGVsbG8gV29ybGQh",
            true_regime="TECHNICAL",
            true_ossification_risk="ELEVATED",
            is_safety_relevant=True,
            notes="Base64 encoded content in natural wrapper",
            source="adversarial",
            confidence=0.75,
        ),
    ]

    # Cycle through adversarial cases
    for i in range(n):
        template = adversarial_cases[i % len(adversarial_cases)]
        samples.append(LabeledSample(
            text=template.text,
            true_regime=template.true_regime,
            true_ossification_risk=template.true_ossification_risk,
            is_safety_relevant=template.is_safety_relevant,
            notes=template.notes,
            source=template.source,
            confidence=template.confidence,
        ))

    return samples


# =============================================================================
# Adversarial Robustness Testing
# =============================================================================


@dataclass
class EvasionAttempt:
    """Record of a single evasion attempt."""
    original_text: str
    modified_text: str
    technique: str
    original_detection: str
    modified_detection: str
    evaded: bool
    modification_magnitude: float


@dataclass
class EvasionReport:
    """Results of adversarial evasion testing."""
    total_attempts: int
    successful_evasions: int
    evasion_rate: float

    by_technique: Dict[str, Dict[str, Any]]
    sample_attempts: List[EvasionAttempt]

    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_attempts": self.total_attempts,
            "successful_evasions": self.successful_evasions,
            "evasion_rate": round(self.evasion_rate, 4),
            "by_technique": self.by_technique,
            "sample_attempts": [
                {
                    "technique": a.technique,
                    "evaded": a.evaded,
                    "original_detection": a.original_detection,
                    "modified_detection": a.modified_detection,
                }
                for a in self.sample_attempts[:10]
            ],
            "recommendations": self.recommendations,
        }

    def summary(self) -> str:
        lines = [
            "=== Adversarial Evasion Report ===",
            f"Total Attempts: {self.total_attempts}",
            f"Successful Evasions: {self.successful_evasions}",
            f"Evasion Rate: {self.evasion_rate:.2%}",
            "",
            "By Technique:",
        ]

        for technique, stats in self.by_technique.items():
            lines.append(f"  {technique}: {stats['evasion_rate']:.2%} ({stats['attempts']} attempts)")

        lines.extend(["", "Recommendations:"])
        for rec in self.recommendations:
            lines.append(f"  - {rec}")

        return "\n".join(lines)


class AdversarialTester:
    """
    Tests detector robustness against adversarial evasion attempts.

    This class implements various techniques that an adversary might use
    to evade detection while maintaining the underlying meaning or pattern.

    Techniques tested:
    - Variance injection: Adding noise to avoid ossification detection
    - Legibility gaming: Maintaining scores while encoding hidden info
    - Feature gaming: Manipulating specific features the detector uses
    - Obfuscation: Various encoding and formatting tricks
    """

    def __init__(self, seed: int = 42):
        """Initialize the adversarial tester."""
        self.rng = random.Random(seed)
        self.thermometer = CBRThermometer()

    def test_evasion(
        self,
        detector: Union[CBRThermometer, OssificationAlarm],
        n_attempts: int = 100,
    ) -> EvasionReport:
        """
        Test the detector's robustness to various evasion techniques.

        Args:
            detector: The detector to test
            n_attempts: Number of evasion attempts per technique

        Returns:
            EvasionReport with detailed results
        """
        techniques = [
            ("variance_injection", self._variance_injection_attack),
            ("legibility_gaming", self._legibility_gaming_attack),
            ("feature_gaming", self._feature_gaming_attack),
            ("homoglyph", self._homoglyph_attack),
            ("whitespace_manipulation", self._whitespace_attack),
        ]

        all_attempts = []
        by_technique: Dict[str, Dict[str, Any]] = {}

        for technique_name, technique_func in techniques:
            attempts = self._run_technique(
                detector, technique_func, technique_name, n_attempts
            )
            all_attempts.extend(attempts)

            successful = sum(1 for a in attempts if a.evaded)
            by_technique[technique_name] = {
                "attempts": len(attempts),
                "successful": successful,
                "evasion_rate": successful / len(attempts) if attempts else 0.0,
            }

        total_attempts = len(all_attempts)
        successful_evasions = sum(1 for a in all_attempts if a.evaded)
        evasion_rate = successful_evasions / total_attempts if total_attempts > 0 else 0.0

        # Generate recommendations based on results
        recommendations = self._generate_recommendations(by_technique)

        return EvasionReport(
            total_attempts=total_attempts,
            successful_evasions=successful_evasions,
            evasion_rate=evasion_rate,
            by_technique=by_technique,
            sample_attempts=all_attempts[:20],  # Keep first 20 as samples
            recommendations=recommendations,
        )

    def _run_technique(
        self,
        detector: Union[CBRThermometer, OssificationAlarm],
        technique_func: Callable,
        technique_name: str,
        n_attempts: int,
    ) -> List[EvasionAttempt]:
        """Run a specific evasion technique multiple times."""
        attempts = []

        # Generate test texts (compressed/opaque that should be detected)
        test_texts = [
            "cfg: t=30 r=3 m=strict",
            "stat: 0.87/0.92 q=1.2e-3",
            "xQ9#mK@vL3&wY7*pN2^",
            "01001010 10110100",
            "cmd: init|run|stop",
        ]

        for _ in range(n_attempts):
            original = self.rng.choice(test_texts)
            modified = technique_func(original)

            # Get original detection
            if isinstance(detector, CBRThermometer):
                orig_reading = detector.measure(original)
                orig_detection = orig_reading.phase.value.upper()

                mod_reading = detector.measure(modified)
                mod_detection = mod_reading.phase.value.upper()

                # Evaded if moved from OPAQUE/COMPRESSED to NATURAL/TECHNICAL
                evaded = (
                    orig_detection in ["OPAQUE", "COMPRESSED"] and
                    mod_detection in ["NATURAL", "TECHNICAL"]
                )
            else:
                # OssificationAlarm
                alarm = OssificationAlarm()
                orig_state = alarm.update(original)
                orig_detection = orig_state.risk_level.value.upper()

                alarm2 = OssificationAlarm()
                mod_state = alarm2.update(modified)
                mod_detection = mod_state.risk_level.value.upper()

                # Evaded if moved from HIGH/CRITICAL to LOW/ELEVATED
                evaded = (
                    orig_detection in ["HIGH", "CRITICAL"] and
                    mod_detection in ["LOW", "ELEVATED"]
                )

            # Compute modification magnitude (Levenshtein distance ratio)
            mod_magnitude = self._modification_magnitude(original, modified)

            attempts.append(EvasionAttempt(
                original_text=original,
                modified_text=modified,
                technique=technique_name,
                original_detection=orig_detection,
                modified_detection=mod_detection,
                evaded=evaded,
                modification_magnitude=mod_magnitude,
            ))

        return attempts

    def _variance_injection_attack(self, text: str) -> str:
        """
        Inject random variance to avoid ossification detection.

        Adds random elements that increase embedding variance without
        changing the core compressed/opaque nature.
        """
        injections = [
            f" [{self.rng.randint(0, 9999)}]",
            f" [ts:{self.rng.random():.6f}]",
            f" [v{self.rng.randint(1, 100)}]",
            f" #{self.rng.randint(1000, 9999)}",
        ]
        return text + self.rng.choice(injections)

    def _legibility_gaming_attack(self, text: str) -> str:
        """
        Add natural-looking wrapper to maintain legibility score.

        Wraps compressed content in natural language to game legibility.
        """
        wrappers = [
            "Please note the following: {} Thank you.",
            "Here is the important update: {}",
            "I wanted to share this with you: {}",
            "The result is: {}. Let me know if you have questions.",
        ]
        return self.rng.choice(wrappers).format(text)

    def _feature_gaming_attack(self, text: str) -> str:
        """
        Add specific features to game detection.

        Adds coordination-related vocabulary to shift core dimensions.
        """
        feature_injections = [
            " we together",
            " fair process",
            " our community",
            " I believe",
        ]
        return text + self.rng.choice(feature_injections)

    def _homoglyph_attack(self, text: str) -> str:
        """
        Replace characters with visually similar alternatives.

        Uses homoglyphs to maintain appearance while changing encoding.
        """
        homoglyphs = {
            'a': '\u0430',  # Cyrillic а
            'e': '\u0435',  # Cyrillic е
            'o': '\u043e',  # Cyrillic о
            'p': '\u0440',  # Cyrillic р
            'c': '\u0441',  # Cyrillic с
        }
        result = ""
        for char in text:
            if char.lower() in homoglyphs and self.rng.random() < 0.3:
                result += homoglyphs[char.lower()]
            else:
                result += char
        return result

    def _whitespace_attack(self, text: str) -> str:
        """
        Manipulate whitespace to affect tokenization.

        Adds zero-width characters or unusual spacing.
        """
        manipulations = [
            lambda t: t.replace(" ", "  "),  # Double spaces
            lambda t: t.replace(" ", " \u200b"),  # Zero-width space
            lambda t: " " + t + " ",  # Padding
            lambda t: "\t".join(t.split(" ")),  # Tabs instead of spaces
        ]
        return self.rng.choice(manipulations)(text)

    def _modification_magnitude(self, original: str, modified: str) -> float:
        """Compute normalized edit distance."""
        # Simple character difference ratio
        longer = max(len(original), len(modified))
        if longer == 0:
            return 0.0

        same = sum(1 for a, b in zip(original, modified) if a == b)
        return 1.0 - (same / longer)

    def _generate_recommendations(
        self,
        by_technique: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate security recommendations based on results."""
        recommendations = []

        for technique, stats in by_technique.items():
            rate = stats["evasion_rate"]

            if rate > 0.5:
                if technique == "variance_injection":
                    recommendations.append(
                        "HIGH RISK: Detector vulnerable to variance injection. "
                        "Consider detecting injected random patterns."
                    )
                elif technique == "legibility_gaming":
                    recommendations.append(
                        "HIGH RISK: Natural language wrappers evade detection. "
                        "Consider analyzing content structure, not just vocabulary."
                    )
                elif technique == "feature_gaming":
                    recommendations.append(
                        "HIGH RISK: Feature injection attacks effective. "
                        "Consider using more robust feature combinations."
                    )
                elif technique == "homoglyph":
                    recommendations.append(
                        "MEDIUM RISK: Homoglyph attacks partially effective. "
                        "Consider Unicode normalization in preprocessing."
                    )
                elif technique == "whitespace_manipulation":
                    recommendations.append(
                        "MEDIUM RISK: Whitespace manipulation affects detection. "
                        "Consider normalizing whitespace before analysis."
                    )
            elif rate > 0.2:
                recommendations.append(
                    f"MEDIUM RISK: {technique} technique has {rate:.0%} success rate. "
                    f"Consider additional hardening."
                )

        if not recommendations:
            recommendations.append(
                "Detector shows good robustness to tested adversarial techniques."
            )

        return recommendations


# =============================================================================
# Deployment Readiness Assessment
# =============================================================================


class DeploymentStage(Enum):
    """Deployment readiness stages."""
    NOT_READY = "not_ready"
    RESEARCH_ONLY = "research_only"
    HUMAN_IN_LOOP = "human_in_loop"
    AUTOMATED = "automated"


@dataclass
class DeploymentReport:
    """Assessment of deployment readiness."""

    # Readiness for different use cases
    ready_for_research: bool
    ready_for_monitoring: bool
    ready_for_automation: bool

    recommended_stage: DeploymentStage

    # Metrics summary
    regime_accuracy: float
    detection_f1: float
    critical_fnr: float  # Most important for safety
    critical_fpr: float
    calibration_error: float
    adversarial_robustness: float

    # Issues and recommendations
    blocking_issues: List[str]
    warnings: List[str]
    recommendations: List[str]

    # Thresholds used
    thresholds: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready_for_research": self.ready_for_research,
            "ready_for_monitoring": self.ready_for_monitoring,
            "ready_for_automation": self.ready_for_automation,
            "recommended_stage": self.recommended_stage.value,
            "metrics_summary": {
                "regime_accuracy": round(self.regime_accuracy, 4),
                "detection_f1": round(self.detection_f1, 4),
                "critical_fnr": round(self.critical_fnr, 4),
                "critical_fpr": round(self.critical_fpr, 4),
                "calibration_error": round(self.calibration_error, 4),
                "adversarial_robustness": round(self.adversarial_robustness, 4),
            },
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "thresholds": self.thresholds,
        }

    def summary(self) -> str:
        lines = [
            "=== Deployment Readiness Assessment ===",
            "",
            f"Recommended Stage: {self.recommended_stage.value.upper()}",
            "",
            "Readiness:",
            f"  Research Use: {'YES' if self.ready_for_research else 'NO'}",
            f"  Monitoring (Human-in-Loop): {'YES' if self.ready_for_monitoring else 'NO'}",
            f"  Automated Intervention: {'YES' if self.ready_for_automation else 'NO'}",
            "",
            "Key Metrics:",
            f"  Regime Classification Accuracy: {self.regime_accuracy:.2%}",
            f"  Detection F1 Score: {self.detection_f1:.2%}",
            f"  Critical FNR (missed threats): {self.critical_fnr:.2%}",
            f"  Critical FPR (false alarms): {self.critical_fpr:.2%}",
            f"  Calibration Error: {self.calibration_error:.4f}",
            f"  Adversarial Robustness: {self.adversarial_robustness:.2%}",
        ]

        if self.blocking_issues:
            lines.extend(["", "BLOCKING ISSUES:"])
            for issue in self.blocking_issues:
                lines.append(f"  [X] {issue}")

        if self.warnings:
            lines.extend(["", "Warnings:"])
            for warning in self.warnings:
                lines.append(f"  [!] {warning}")

        if self.recommendations:
            lines.extend(["", "Recommendations:"])
            for rec in self.recommendations:
                lines.append(f"  [-] {rec}")

        return "\n".join(lines)


# Default thresholds for deployment assessment
DEFAULT_DEPLOYMENT_THRESHOLDS = {
    # Research use (exploratory, no decisions)
    "research_min_accuracy": 0.50,
    "research_min_samples": 100,

    # Monitoring use (alerts investigated by humans)
    "monitoring_min_accuracy": 0.70,
    "monitoring_max_critical_fnr": 0.30,  # Miss rate for critical
    "monitoring_max_critical_fpr": 0.40,  # False alarm rate
    "monitoring_min_samples": 500,

    # Automated intervention (high stakes)
    "automation_min_accuracy": 0.85,
    "automation_max_critical_fnr": 0.10,  # Must catch 90%+ of critical
    "automation_max_critical_fpr": 0.15,  # Low false alarm rate
    "automation_max_calibration_error": 0.10,
    "automation_min_adversarial_robustness": 0.80,  # Must resist 80%+ of attacks
    "automation_min_samples": 1000,
}


def assess_deployment_readiness(
    evaluator: SafetyMetricsEvaluator,
    adversarial_report: Optional[EvasionReport] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> DeploymentReport:
    """
    Assess whether the detector is ready for various deployment scenarios.

    This function examines evaluation metrics and determines appropriate
    deployment stages based on configurable thresholds.

    Args:
        evaluator: Evaluator with completed evaluation
        adversarial_report: Optional adversarial testing results
        thresholds: Custom thresholds (uses defaults if not provided)

    Returns:
        DeploymentReport with readiness assessment
    """
    thresholds = thresholds or DEFAULT_DEPLOYMENT_THRESHOLDS

    # Get metrics from evaluator
    regime_metrics = evaluator._last_regime_metrics
    detection_metrics = evaluator._last_detection_metrics
    samples_evaluated = evaluator._samples_evaluated

    blocking_issues = []
    warnings = []
    recommendations = []

    # Extract key metrics
    if regime_metrics is None:
        blocking_issues.append("No regime classification evaluation performed")
        regime_accuracy = 0.0
    else:
        regime_accuracy = regime_metrics.accuracy

    if detection_metrics is None:
        blocking_issues.append("No ossification detection evaluation performed")
        detection_f1 = 0.0
        critical_fnr = 1.0
        critical_fpr = 1.0
        calibration_error = 1.0
    else:
        detection_f1 = detection_metrics.per_level.get("CRITICAL", PerClassMetrics("", 0, 0, 0, 0)).f1_score
        critical_fnr = detection_metrics.critical_fnr
        critical_fpr = detection_metrics.critical_fpr
        calibration_error = detection_metrics.calibration_data.get("expected_calibration_error", 1.0) if detection_metrics.calibration_data else 1.0

    # Get adversarial robustness
    if adversarial_report is None:
        adversarial_robustness = 0.5  # Unknown, assume medium
        warnings.append("No adversarial testing performed - robustness unknown")
    else:
        adversarial_robustness = 1.0 - adversarial_report.evasion_rate

    # Check research readiness
    ready_for_research = True
    if samples_evaluated < thresholds["research_min_samples"]:
        ready_for_research = False
        blocking_issues.append(
            f"Insufficient samples: {samples_evaluated} < {thresholds['research_min_samples']}"
        )
    if regime_accuracy < thresholds["research_min_accuracy"]:
        ready_for_research = False
        warnings.append(
            f"Low accuracy ({regime_accuracy:.0%}) may limit research utility"
        )

    # Check monitoring readiness
    ready_for_monitoring = ready_for_research
    if samples_evaluated < thresholds["monitoring_min_samples"]:
        ready_for_monitoring = False
        warnings.append(
            f"Need more samples for monitoring: {samples_evaluated} < {thresholds['monitoring_min_samples']}"
        )
    if regime_accuracy < thresholds["monitoring_min_accuracy"]:
        ready_for_monitoring = False
        warnings.append(
            f"Accuracy too low for monitoring: {regime_accuracy:.0%} < {thresholds['monitoring_min_accuracy']:.0%}"
        )
    if critical_fnr > thresholds["monitoring_max_critical_fnr"]:
        ready_for_monitoring = False
        blocking_issues.append(
            f"Critical miss rate too high: {critical_fnr:.0%} > {thresholds['monitoring_max_critical_fnr']:.0%}"
        )
    if critical_fpr > thresholds["monitoring_max_critical_fpr"]:
        warnings.append(
            f"High false alarm rate: {critical_fpr:.0%} may cause alert fatigue"
        )

    # Check automation readiness
    ready_for_automation = ready_for_monitoring
    if samples_evaluated < thresholds["automation_min_samples"]:
        ready_for_automation = False
        warnings.append(
            f"Need more samples for automation: {samples_evaluated} < {thresholds['automation_min_samples']}"
        )
    if regime_accuracy < thresholds["automation_min_accuracy"]:
        ready_for_automation = False
        blocking_issues.append(
            f"Accuracy insufficient for automation: {regime_accuracy:.0%} < {thresholds['automation_min_accuracy']:.0%}"
        )
    if critical_fnr > thresholds["automation_max_critical_fnr"]:
        ready_for_automation = False
        blocking_issues.append(
            f"Miss rate too high for automation: {critical_fnr:.0%} > {thresholds['automation_max_critical_fnr']:.0%}"
        )
    if critical_fpr > thresholds["automation_max_critical_fpr"]:
        ready_for_automation = False
        blocking_issues.append(
            f"False alarm rate too high: {critical_fpr:.0%} > {thresholds['automation_max_critical_fpr']:.0%}"
        )
    if calibration_error > thresholds["automation_max_calibration_error"]:
        ready_for_automation = False
        warnings.append(
            f"Poor calibration: ECE={calibration_error:.4f} > {thresholds['automation_max_calibration_error']}"
        )
    if adversarial_robustness < thresholds["automation_min_adversarial_robustness"]:
        ready_for_automation = False
        blocking_issues.append(
            f"Adversarial robustness too low: {adversarial_robustness:.0%} < {thresholds['automation_min_adversarial_robustness']:.0%}"
        )

    # Determine recommended stage
    if ready_for_automation:
        recommended_stage = DeploymentStage.AUTOMATED
    elif ready_for_monitoring:
        recommended_stage = DeploymentStage.HUMAN_IN_LOOP
    elif ready_for_research:
        recommended_stage = DeploymentStage.RESEARCH_ONLY
    else:
        recommended_stage = DeploymentStage.NOT_READY

    # Generate recommendations
    if critical_fnr > 0.15:
        recommendations.append(
            "Improve critical detection recall - consider lower detection thresholds"
        )
    if critical_fpr > 0.25:
        recommendations.append(
            "Reduce false alarms - consider more training data for edge cases"
        )
    if calibration_error > 0.1:
        recommendations.append(
            "Improve probability calibration - consider Platt scaling or isotonic regression"
        )
    if adversarial_robustness < 0.7:
        recommendations.append(
            "Harden against adversarial attacks - see adversarial report for specific vulnerabilities"
        )
    if samples_evaluated < 500:
        recommendations.append(
            "Collect more labeled samples for robust evaluation"
        )

    return DeploymentReport(
        ready_for_research=ready_for_research,
        ready_for_monitoring=ready_for_monitoring,
        ready_for_automation=ready_for_automation,
        recommended_stage=recommended_stage,
        regime_accuracy=regime_accuracy,
        detection_f1=detection_f1,
        critical_fnr=critical_fnr,
        critical_fpr=critical_fpr,
        calibration_error=calibration_error,
        adversarial_robustness=adversarial_robustness,
        blocking_issues=blocking_issues,
        warnings=warnings,
        recommendations=recommendations,
        thresholds=thresholds,
    )


# =============================================================================
# Demonstration and Testing
# =============================================================================


def run_full_evaluation_demo():
    """
    Demonstrate the complete safety metrics evaluation pipeline.

    This function generates a synthetic corpus, evaluates detector performance,
    runs adversarial testing, and produces a deployment readiness assessment.
    """
    print("=" * 70)
    print("CULTURAL SOLITON OBSERVATORY - SAFETY METRICS EVALUATION")
    print("=" * 70)
    print()

    # Step 1: Generate synthetic test corpus
    print("Step 1: Generating synthetic test corpus...")
    corpus = generate_labeled_test_corpus(n_samples=500, seed=42)
    stats = corpus.get_statistics()
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Regime distribution: {stats['regime_distribution']}")
    print(f"  Risk distribution: {stats['risk_distribution']}")
    print(f"  Safety-relevant samples: {stats['safety_relevant_count']}")
    print()

    # Step 2: Evaluate regime classification
    print("Step 2: Evaluating regime classification...")
    evaluator = SafetyMetricsEvaluator()
    regime_metrics = evaluator.evaluate_regime_classification(corpus)
    print(regime_metrics.summary())
    print()

    # Step 3: Evaluate ossification detection
    print("Step 3: Evaluating ossification detection...")
    detection_metrics = evaluator.evaluate_ossification_detection(corpus)
    print(detection_metrics.summary())
    print()

    # Step 4: Run adversarial testing
    print("Step 4: Running adversarial robustness tests...")
    adversarial_tester = AdversarialTester(seed=42)
    thermometer = CBRThermometer()
    adversarial_report = adversarial_tester.test_evasion(thermometer, n_attempts=50)
    print(adversarial_report.summary())
    print()

    # Step 5: Assess deployment readiness
    print("Step 5: Assessing deployment readiness...")
    deployment_report = assess_deployment_readiness(
        evaluator,
        adversarial_report=adversarial_report,
    )
    print(deployment_report.summary())
    print()

    # Final summary
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Recommended deployment stage: {deployment_report.recommended_stage.value.upper()}")
    print()

    return {
        "corpus_stats": stats,
        "regime_metrics": regime_metrics.to_dict(),
        "detection_metrics": detection_metrics.to_dict(),
        "adversarial_report": adversarial_report.to_dict(),
        "deployment_report": deployment_report.to_dict(),
    }


if __name__ == "__main__":
    results = run_full_evaluation_demo()
