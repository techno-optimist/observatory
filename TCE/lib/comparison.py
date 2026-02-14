"""
Experiment Comparison & Regression Detection

Compares results across adapter versions to detect:
- Improvements (new capabilities)
- Regressions (lost capabilities)
- Stability (unchanged behavior)

The key insight: Cognitive isotopes should be orthogonal.
Improving Σ_t (stats) should NOT degrade Σ_p (premises).
If it does, we have catastrophic interference - a bug.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .statistics import (
    wilson_score_interval,
    cohens_d,
    mcnemar_test,
    cross_architecture_analysis,
    multi_architecture_summary,
    benjamini_hochberg_fdr,
    CrossArchitectureAnalysis,
)


@dataclass
class TrialComparison:
    """Comparison of a single trial across two runs."""
    prompt_id: str
    prompt_text: str

    # Baseline (e.g., V10.1)
    baseline_triggered: bool
    baseline_confidence: float
    baseline_isotope: Optional[str]

    # Treatment (e.g., V10.2a)
    treatment_triggered: bool
    treatment_confidence: float
    treatment_isotope: Optional[str]

    @property
    def status(self) -> str:
        """Classification of change."""
        if self.baseline_triggered and self.treatment_triggered:
            return "stable_pass"
        elif not self.baseline_triggered and not self.treatment_triggered:
            return "stable_fail"
        elif not self.baseline_triggered and self.treatment_triggered:
            return "improvement"
        else:  # baseline passed, treatment failed
            return "regression"

    @property
    def confidence_delta(self) -> float:
        """Change in confidence (positive = improvement)."""
        return self.treatment_confidence - self.baseline_confidence

    @property
    def isotope_changed(self) -> bool:
        """Did the detected isotope change?"""
        return self.baseline_isotope != self.treatment_isotope


@dataclass
class ComparisonReport:
    """Full comparison between two experiment runs."""
    baseline_id: str
    treatment_id: str
    baseline_adapter: str
    treatment_adapter: str

    # Overall metrics
    baseline_trigger_rate: float
    treatment_trigger_rate: float

    # Trial-level comparisons
    trials: List[TrialComparison] = field(default_factory=list)

    # By-isotope breakdown
    isotope_comparison: Dict[str, Dict] = field(default_factory=dict)

    # Statistical tests
    mcnemar_result: Optional[Dict] = None
    effect_size: Optional[Dict] = None

    @property
    def n_improvements(self) -> int:
        return sum(1 for t in self.trials if t.status == "improvement")

    @property
    def n_regressions(self) -> int:
        return sum(1 for t in self.trials if t.status == "regression")

    @property
    def n_stable_pass(self) -> int:
        return sum(1 for t in self.trials if t.status == "stable_pass")

    @property
    def n_stable_fail(self) -> int:
        return sum(1 for t in self.trials if t.status == "stable_fail")

    @property
    def has_regressions(self) -> bool:
        return self.n_regressions > 0

    @property
    def is_improvement(self) -> bool:
        """Net improvement with no regressions."""
        return self.n_improvements > 0 and self.n_regressions == 0

    @property
    def verdict(self) -> str:
        """Human-readable verdict."""
        if self.n_regressions > 0:
            return f"⚠️  REGRESSION DETECTED ({self.n_regressions} trials)"
        elif self.n_improvements > 0:
            return f"✓ IMPROVEMENT ({self.n_improvements} trials gained)"
        else:
            return "→ NO CHANGE"


def load_result(path: Path) -> Dict:
    """Load experiment result from JSON."""
    with open(path) as f:
        return json.load(f)


def compare_experiments(
    baseline_path: Path,
    treatment_path: Path
) -> ComparisonReport:
    """
    Compare two experiment results.

    Args:
        baseline_path: Path to baseline result (e.g., V10.1)
        treatment_path: Path to treatment result (e.g., V10.2a)

    Returns:
        ComparisonReport with detailed analysis
    """
    baseline = load_result(baseline_path)
    treatment = load_result(treatment_path)

    # Build prompt_id -> trial mapping
    baseline_trials = {t["prompt_id"]: t for t in baseline["trials"]}
    treatment_trials = {t["prompt_id"]: t for t in treatment["trials"]}

    # Find common prompts
    common_prompts = set(baseline_trials.keys()) & set(treatment_trials.keys())

    # Compare each trial
    trial_comparisons = []
    for prompt_id in sorted(common_prompts):
        bt = baseline_trials[prompt_id]
        tt = treatment_trials[prompt_id]

        # Extract metrics
        b_triggered = bt["metrics"]["triggered"]
        b_confidence = bt["metrics"]["trigger_confidence"]
        b_isotope = None
        if bt["metrics"]["detected_elements"]:
            b_isotope = bt["metrics"]["detected_elements"][0].get("isotope_id")

        t_triggered = tt["metrics"]["triggered"]
        t_confidence = tt["metrics"]["trigger_confidence"]
        t_isotope = None
        if tt["metrics"]["detected_elements"]:
            t_isotope = tt["metrics"]["detected_elements"][0].get("isotope_id")

        trial_comparisons.append(TrialComparison(
            prompt_id=prompt_id,
            prompt_text=bt["prompt_text"],
            baseline_triggered=b_triggered,
            baseline_confidence=b_confidence,
            baseline_isotope=b_isotope,
            treatment_triggered=t_triggered,
            treatment_confidence=t_confidence,
            treatment_isotope=t_isotope
        ))

    # Compute isotope-level comparison
    isotope_comparison = {}
    for tc in trial_comparisons:
        # Use expected isotope from prompt_id if available (e.g., "sp_01" -> "skeptic_premise")
        isotope = _infer_isotope_from_prompt_id(tc.prompt_id)
        if isotope:
            if isotope not in isotope_comparison:
                isotope_comparison[isotope] = {
                    "baseline_pass": 0,
                    "baseline_total": 0,
                    "treatment_pass": 0,
                    "treatment_total": 0,
                }
            isotope_comparison[isotope]["baseline_total"] += 1
            isotope_comparison[isotope]["treatment_total"] += 1
            if tc.baseline_triggered:
                isotope_comparison[isotope]["baseline_pass"] += 1
            if tc.treatment_triggered:
                isotope_comparison[isotope]["treatment_pass"] += 1

    # Compute rates for each isotope
    for isotope, data in isotope_comparison.items():
        data["baseline_rate"] = data["baseline_pass"] / data["baseline_total"] if data["baseline_total"] > 0 else 0
        data["treatment_rate"] = data["treatment_pass"] / data["treatment_total"] if data["treatment_total"] > 0 else 0
        data["delta"] = data["treatment_rate"] - data["baseline_rate"]

    # McNemar's test for paired comparison
    # Count: baseline_fail->treatment_pass vs baseline_pass->treatment_fail
    b_fail_t_pass = sum(1 for tc in trial_comparisons if tc.status == "improvement")
    b_pass_t_fail = sum(1 for tc in trial_comparisons if tc.status == "regression")

    mcnemar = None
    if b_fail_t_pass + b_pass_t_fail > 0:
        result = mcnemar_test(b_pass_t_fail, b_fail_t_pass)
        mcnemar = {
            "b_pass_t_fail": b_pass_t_fail,
            "b_fail_t_pass": b_fail_t_pass,
            "statistic": result.statistic,
            "p_value": result.p_value,
            "significant": result.significant,
            "interpretation": result.interpretation
        }

    # Effect size on confidence scores
    baseline_confidences = [tc.baseline_confidence for tc in trial_comparisons if tc.baseline_triggered]
    treatment_confidences = [tc.treatment_confidence for tc in trial_comparisons if tc.treatment_triggered]

    effect = None
    if len(baseline_confidences) >= 2 and len(treatment_confidences) >= 2:
        result = cohens_d(baseline_confidences, treatment_confidences)
        effect = {
            "cohens_d": result.cohens_d,
            "magnitude": result.interpretation,  # EffectSize uses interpretation for magnitude
            "interpretation": result.interpretation
        }

    # Build report
    return ComparisonReport(
        baseline_id=baseline.get("id", "unknown"),
        treatment_id=treatment.get("id", "unknown"),
        baseline_adapter=baseline.get("reproducibility", {}).get("adapter_version", "unknown"),
        treatment_adapter=treatment.get("reproducibility", {}).get("adapter_version", "unknown"),
        baseline_trigger_rate=baseline["summary"]["trigger_rate"]["point_estimate"],
        treatment_trigger_rate=treatment["summary"]["trigger_rate"]["point_estimate"],
        trials=trial_comparisons,
        isotope_comparison=isotope_comparison,
        mcnemar_result=mcnemar,
        effect_size=effect
    )


def _infer_isotope_from_prompt_id(prompt_id: str) -> Optional[str]:
    """Infer isotope from prompt ID convention."""
    prefix_map = {
        "sp_": "skeptic_premise",
        "sm_": "skeptic_method",
        "ss_": "skeptic_source",
        "st_": "skeptic_stats",
    }
    for prefix, isotope in prefix_map.items():
        if prompt_id.startswith(prefix):
            return isotope
    return None


def format_report(report: ComparisonReport) -> str:
    """Format comparison report as readable text."""
    lines = []

    lines.append("=" * 60)
    lines.append("EXPERIMENT COMPARISON REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Baseline:  {report.baseline_adapter}")
    lines.append(f"Treatment: {report.treatment_adapter}")
    lines.append("")

    # Overall verdict
    lines.append(f"VERDICT: {report.verdict}")
    lines.append("")

    # Trigger rate comparison
    delta = report.treatment_trigger_rate - report.baseline_trigger_rate
    delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
    lines.append(f"Trigger Rate: {report.baseline_trigger_rate:.1%} → {report.treatment_trigger_rate:.1%} ({delta_str})")
    lines.append("")

    # Status breakdown
    lines.append("Status Breakdown:")
    lines.append(f"  ✓ Stable Pass:  {report.n_stable_pass}")
    lines.append(f"  ✗ Stable Fail:  {report.n_stable_fail}")
    lines.append(f"  ↑ Improvement:  {report.n_improvements}")
    lines.append(f"  ↓ Regression:   {report.n_regressions}")
    lines.append("")

    # Isotope breakdown
    if report.isotope_comparison:
        lines.append("By Isotope:")
        for isotope, data in sorted(report.isotope_comparison.items()):
            symbol = _isotope_symbol(isotope)
            b_rate = data["baseline_rate"]
            t_rate = data["treatment_rate"]
            delta = data["delta"]
            delta_str = f"+{delta:.0%}" if delta >= 0 else f"{delta:.0%}"
            status = "✓" if t_rate >= 0.95 else "↑" if delta > 0 else "↓" if delta < 0 else "→"
            lines.append(f"  {status} {symbol}: {b_rate:.0%} → {t_rate:.0%} ({delta_str})")
        lines.append("")

    # Statistical tests
    if report.mcnemar_result:
        lines.append("McNemar's Test (paired comparison):")
        lines.append(f"  statistic = {report.mcnemar_result['statistic']:.3f}, p = {report.mcnemar_result['p_value']:.4f}")
        lines.append(f"  {report.mcnemar_result['interpretation']}")
        lines.append("")

    if report.effect_size:
        lines.append("Effect Size (confidence scores):")
        lines.append(f"  Cohen's d = {report.effect_size['cohens_d']:.3f} ({report.effect_size['magnitude']})")
        lines.append("")

    # Detailed changes
    if report.n_improvements > 0 or report.n_regressions > 0:
        lines.append("-" * 60)
        lines.append("DETAILED CHANGES")
        lines.append("-" * 60)

        if report.n_regressions > 0:
            lines.append("")
            lines.append("⚠️  REGRESSIONS (previously passed, now failed):")
            for tc in report.trials:
                if tc.status == "regression":
                    lines.append(f"  [{tc.prompt_id}] {tc.prompt_text[:50]}...")
                    lines.append(f"    Confidence: {tc.baseline_confidence:.0%} → {tc.treatment_confidence:.0%}")

        if report.n_improvements > 0:
            lines.append("")
            lines.append("↑ IMPROVEMENTS (previously failed, now passed):")
            for tc in report.trials:
                if tc.status == "improvement":
                    lines.append(f"  [{tc.prompt_id}] {tc.prompt_text[:50]}...")
                    lines.append(f"    Confidence: {tc.baseline_confidence:.0%} → {tc.treatment_confidence:.0%}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def _isotope_symbol(isotope_id: str) -> str:
    """Get symbol for isotope."""
    symbols = {
        "skeptic_premise": "Σₚ",
        "skeptic_method": "Σₘ",
        "skeptic_source": "Σₛ",
        "skeptic_stats": "Σₜ",
    }
    return symbols.get(isotope_id, isotope_id)


def check_regressions(
    baseline_path: Path,
    treatment_path: Path,
    fail_on_regression: bool = True
) -> Tuple[bool, ComparisonReport]:
    """
    Check for regressions between versions.

    Args:
        baseline_path: Path to baseline result
        treatment_path: Path to treatment result
        fail_on_regression: If True, return False when regressions found

    Returns:
        Tuple of (passed, report)
    """
    report = compare_experiments(baseline_path, treatment_path)

    if fail_on_regression and report.has_regressions:
        return (False, report)

    return (True, report)


# =============================================================================
# Multi-Architecture Comparison
# =============================================================================

@dataclass
class MultiArchReport:
    """Comparison of cognitive elements across multiple architectures."""
    architectures: List[str]
    elements: List[str]

    # Per-element analysis
    element_analyses: Dict[str, CrossArchitectureAnalysis] = field(default_factory=dict)

    # Summary statistics
    stable_elements: List[str] = field(default_factory=list)
    unstable_elements: List[str] = field(default_factory=list)

    # Overall heterogeneity
    mean_heterogeneity: float = 0.0

    @property
    def n_stable(self) -> int:
        return len(self.stable_elements)

    @property
    def n_unstable(self) -> int:
        return len(self.unstable_elements)

    @property
    def stability_ratio(self) -> float:
        """Fraction of elements that are stable across architectures."""
        total = self.n_stable + self.n_unstable
        return self.n_stable / total if total > 0 else 0.0


def compare_architectures(
    result_paths: Dict[str, Path],
    elements: Optional[List[str]] = None,
    stability_threshold: float = 0.15
) -> MultiArchReport:
    """
    Compare cognitive element performance across multiple architectures.

    Args:
        result_paths: Dict mapping architecture name to result file path
                     e.g., {"phi4": Path("phi4_result.json"), "qwen": Path("qwen_result.json")}
        elements: List of element IDs to analyze (None = all found)
        stability_threshold: CV threshold for stability (default 0.15 = 15%)

    Returns:
        MultiArchReport with cross-architecture analysis
    """
    # Load all results
    results = {}
    for arch, path in result_paths.items():
        results[arch] = load_result(path)

    # Find all elements if not specified
    if elements is None:
        all_elements = set()
        for arch, result in results.items():
            for trial in result.get("trials", []):
                for elem in trial.get("metrics", {}).get("detected_elements", []):
                    elem_id = elem.get("element_id")
                    if elem_id:
                        all_elements.add(elem_id)
        elements = sorted(all_elements)

    # Analyze each element
    element_analyses = {}
    stable_elements = []
    unstable_elements = []

    for elem_id in elements:
        analysis = cross_architecture_analysis(
            results, elem_id, stability_threshold
        )
        element_analyses[elem_id] = analysis

        if analysis.is_stable:
            stable_elements.append(elem_id)
        else:
            unstable_elements.append(elem_id)

    # Compute mean heterogeneity
    heterogeneities = [a.heterogeneity for a in element_analyses.values()]
    mean_het = sum(heterogeneities) / len(heterogeneities) if heterogeneities else 0.0

    return MultiArchReport(
        architectures=list(result_paths.keys()),
        elements=elements,
        element_analyses=element_analyses,
        stable_elements=stable_elements,
        unstable_elements=unstable_elements,
        mean_heterogeneity=mean_het
    )


def format_multi_arch_report(report: MultiArchReport) -> str:
    """Format multi-architecture comparison as readable text."""
    lines = []

    lines.append("=" * 70)
    lines.append("CROSS-ARCHITECTURE COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Architectures: {', '.join(report.architectures)}")
    lines.append(f"Elements analyzed: {len(report.elements)}")
    lines.append("")

    # Stability summary
    lines.append(f"STABILITY SUMMARY:")
    lines.append(f"  Stable elements:   {report.n_stable} ({report.stability_ratio:.0%})")
    lines.append(f"  Unstable elements: {report.n_unstable}")
    lines.append(f"  Mean heterogeneity (CV): {report.mean_heterogeneity:.2f}")
    lines.append("")

    # Stable elements
    if report.stable_elements:
        lines.append("✓ STABLE (consistent across architectures):")
        for elem_id in report.stable_elements:
            analysis = report.element_analyses[elem_id]
            lines.append(f"  {elem_id}: {analysis.mean_rate:.0%} ± {analysis.std_rate:.0%}")
        lines.append("")

    # Unstable elements
    if report.unstable_elements:
        lines.append("⚠️  UNSTABLE (architecture-dependent):")
        for elem_id in report.unstable_elements:
            analysis = report.element_analyses[elem_id]
            lines.append(f"  {elem_id}: {analysis.mean_rate:.0%} ± {analysis.std_rate:.0%} (CV={analysis.heterogeneity:.2f})")
            lines.append(f"    Best:  {analysis.max_arch[0]} ({analysis.max_arch[1]:.0%})")
            lines.append(f"    Worst: {analysis.min_arch[0]} ({analysis.min_arch[1]:.0%})")
        lines.append("")

    # Detailed per-element breakdown
    lines.append("-" * 70)
    lines.append("DETAILED ELEMENT ANALYSIS")
    lines.append("-" * 70)

    for elem_id in report.elements:
        analysis = report.element_analyses[elem_id]
        status = "✓" if analysis.is_stable else "⚠️"
        lines.append("")
        lines.append(f"{status} {elem_id}")
        lines.append(f"  Rate by architecture:")
        for arch, rate, n in zip(analysis.architectures, analysis.rates, analysis.n_samples):
            bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
            lines.append(f"    {arch:12} {bar} {rate:.0%} (n={n})")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def detect_architecture_blind_spots(
    result_paths: Dict[str, Path],
    threshold: float = 0.6
) -> Dict[str, List[str]]:
    """
    Identify elements that specific architectures struggle with.

    A "blind spot" is an element where an architecture performs significantly
    below the cross-architecture mean.

    Args:
        result_paths: Dict mapping architecture name to result file path
        threshold: Rate below which an architecture is considered to have a blind spot

    Returns:
        Dict mapping architecture -> list of blind spot element IDs
    """
    report = compare_architectures(result_paths)

    blind_spots = {arch: [] for arch in report.architectures}

    for elem_id, analysis in report.element_analyses.items():
        mean_rate = analysis.mean_rate

        for arch, rate in zip(analysis.architectures, analysis.rates):
            # Blind spot if significantly below mean AND below threshold
            if rate < threshold and rate < mean_rate - 0.1:
                blind_spots[arch].append(elem_id)

    return blind_spots


def isotope_orthogonality_check(
    baseline_path: Path,
    treatment_path: Path,
    isotope_prefixes: Optional[Dict[str, str]] = None
) -> Dict[str, Dict]:
    """
    Check if improving one isotope degraded others (catastrophic interference).

    The Cognitive Isotope Hypothesis predicts isotopes should be orthogonal:
    improving Σₜ should NOT degrade Σₚ.

    Args:
        baseline_path: Path to baseline result
        treatment_path: Path to treatment result
        isotope_prefixes: Dict mapping prefix to isotope name
                         Default: {"sp_": "Σₚ", "sm_": "Σₘ", "ss_": "Σₛ", "st_": "Σₜ"}

    Returns:
        Dict with orthogonality analysis per isotope pair
    """
    if isotope_prefixes is None:
        isotope_prefixes = {
            "sp_": "Σₚ",
            "sm_": "Σₘ",
            "ss_": "Σₛ",
            "st_": "Σₜ",
        }

    report = compare_experiments(baseline_path, treatment_path)

    # Group trials by isotope
    isotope_trials = {}
    for tc in report.trials:
        for prefix, name in isotope_prefixes.items():
            if tc.prompt_id.startswith(prefix):
                if name not in isotope_trials:
                    isotope_trials[name] = []
                isotope_trials[name].append(tc)
                break

    # Compute per-isotope changes
    isotope_changes = {}
    for name, trials in isotope_trials.items():
        n_improved = sum(1 for t in trials if t.status == "improvement")
        n_regressed = sum(1 for t in trials if t.status == "regression")
        n_total = len(trials)

        baseline_rate = sum(1 for t in trials if t.baseline_triggered) / n_total if n_total > 0 else 0
        treatment_rate = sum(1 for t in trials if t.treatment_triggered) / n_total if n_total > 0 else 0

        isotope_changes[name] = {
            "baseline_rate": baseline_rate,
            "treatment_rate": treatment_rate,
            "delta": treatment_rate - baseline_rate,
            "n_improved": n_improved,
            "n_regressed": n_regressed,
            "status": "improved" if n_improved > 0 and n_regressed == 0 else
                      "regressed" if n_regressed > 0 else "stable"
        }

    # Check orthogonality: did improving any isotope cause regression in others?
    orthogonality_violations = []
    improved_isotopes = [name for name, data in isotope_changes.items()
                        if data["status"] == "improved"]
    regressed_isotopes = [name for name, data in isotope_changes.items()
                         if data["status"] == "regressed"]

    if improved_isotopes and regressed_isotopes:
        for improved in improved_isotopes:
            for regressed in regressed_isotopes:
                orthogonality_violations.append({
                    "improved": improved,
                    "regressed": regressed,
                    "interpretation": f"Improving {improved} may have degraded {regressed}"
                })

    return {
        "isotope_changes": isotope_changes,
        "orthogonality_violations": orthogonality_violations,
        "is_orthogonal": len(orthogonality_violations) == 0,
        "interpretation": "Isotopes are orthogonal - no catastrophic interference"
                         if len(orthogonality_violations) == 0
                         else f"WARNING: {len(orthogonality_violations)} orthogonality violation(s) detected"
    }
