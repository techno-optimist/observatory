"""
Statistical Audit Module for the Cultural Soliton Observatory.

Provides proper uncertainty quantification for all telescope measurements:
- Bootstrap confidence intervals
- Power analysis
- Effect size benchmarks
- Bayesian alternatives for small samples
- Simulation-based validation

Author: Observatory Research Team
Version: 1.0.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import warnings


class EvidenceStrength(Enum):
    """Strength of evidence based on Bayesian interpretation."""
    ANECDOTAL = "anecdotal"  # BF < 3
    MODERATE = "moderate"  # 3 <= BF < 10
    STRONG = "strong"  # 10 <= BF < 30
    VERY_STRONG = "very_strong"  # 30 <= BF < 100
    EXTREME = "extreme"  # BF >= 100


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval result."""
    estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int
    se: float  # Standard error


@dataclass
class PowerAnalysis:
    """Power analysis result."""
    current_n: int
    current_power: float
    n_for_80_power: int
    n_for_95_power: int
    detectable_effect: float  # Minimum detectable effect at current N


@dataclass
class EffectSizeResult:
    """Effect size with interpretation."""
    cohens_d: float
    ci_lower: float
    ci_upper: float
    interpretation: str  # "small", "medium", "large", "very large"
    n1: int
    n2: int


@dataclass
class StatisticalAudit:
    """Complete statistical audit of a measurement."""
    measurement_name: str
    point_estimate: float
    bootstrap_ci: BootstrapCI
    power: PowerAnalysis
    effect_size: Optional[EffectSizeResult]
    evidence_strength: EvidenceStrength
    replication_probability: float
    warnings: List[str]
    recommendations: List[str]


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42
) -> BootstrapCI:
    """
    Compute bootstrap confidence interval.

    Args:
        data: Array of observations
        statistic: Function to compute (default: mean)
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (default: 0.95)
        seed: Random seed

    Returns:
        BootstrapCI with estimate and interval
    """
    np.random.seed(seed)
    n = len(data)

    # Point estimate
    estimate = statistic(data)

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(resample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile method
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    # Standard error
    se = np.std(bootstrap_stats)

    return BootstrapCI(
        estimate=estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        se=se,
    )


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> EffectSizeResult:
    """
    Compute Cohen's d with confidence interval.

    Args:
        group1: First group observations
        group2: Second group observations

    Returns:
        EffectSizeResult with effect size and interpretation
    """
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    # Cohen's d
    if pooled_std == 0:
        d = 0.0 if m1 == m2 else np.inf
    else:
        d = (m1 - m2) / pooled_std

    # Standard error of d (Hedges & Olkin, 1985)
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))

    # 95% CI
    ci_lower = d - 1.96 * se_d
    ci_upper = d + 1.96 * se_d

    # Interpretation (Cohen's conventions, adjusted for coordination space)
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    elif abs_d < 1.2:
        interpretation = "large"
    else:
        interpretation = "very large"

    return EffectSizeResult(
        cohens_d=d,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        interpretation=interpretation,
        n1=n1,
        n2=n2,
    )


def power_analysis(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    two_tailed: bool = True
) -> PowerAnalysis:
    """
    Compute statistical power and required sample sizes.

    Uses normal approximation for simplicity.

    Args:
        effect_size: Expected Cohen's d
        n: Current sample size (per group)
        alpha: Significance level
        two_tailed: Two-tailed test?

    Returns:
        PowerAnalysis with current power and required Ns
    """
    from scipy import stats

    # Standard error
    se = np.sqrt(2 / n)

    # Non-centrality parameter
    ncp = effect_size / se

    # Critical value
    if two_tailed:
        z_crit = stats.norm.ppf(1 - alpha / 2)
    else:
        z_crit = stats.norm.ppf(1 - alpha)

    # Power
    power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)

    # Required N for 80% power
    def required_n(target_power):
        for test_n in range(5, 10000):
            test_se = np.sqrt(2 / test_n)
            test_ncp = effect_size / test_se
            test_power = 1 - stats.norm.cdf(z_crit - test_ncp) + stats.norm.cdf(-z_crit - test_ncp)
            if test_power >= target_power:
                return test_n
        return 10000

    n_80 = required_n(0.80) if effect_size > 0 else np.inf
    n_95 = required_n(0.95) if effect_size > 0 else np.inf

    # Minimum detectable effect at current N with 80% power
    def detectable_effect():
        for d in np.arange(0.01, 3.0, 0.01):
            test_se = np.sqrt(2 / n)
            test_ncp = d / test_se
            test_power = 1 - stats.norm.cdf(z_crit - test_ncp) + stats.norm.cdf(-z_crit - test_ncp)
            if test_power >= 0.80:
                return d
        return 3.0

    mde = detectable_effect()

    return PowerAnalysis(
        current_n=n,
        current_power=power,
        n_for_80_power=n_80,
        n_for_95_power=n_95,
        detectable_effect=mde,
    )


def bayes_factor_t(group1: np.ndarray, group2: np.ndarray, prior_scale: float = 0.707) -> float:
    """
    Compute Bayes Factor for two-sample comparison.

    Uses JZS prior (Rouder et al., 2009).
    Simplified approximation based on t-statistic.

    Args:
        group1: First group
        group2: Second group
        prior_scale: Scale of Cauchy prior on effect size

    Returns:
        Bayes Factor (BF10: evidence for H1 vs H0)
    """
    n1, n2 = len(group1), len(group2)
    n = n1 + n2

    # t-statistic
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n - 2)
    se = np.sqrt(pooled_var * (1/n1 + 1/n2))

    if se == 0:
        return 1.0 if m1 == m2 else np.inf

    t = (m1 - m2) / se
    df = n - 2

    # BIC approximation to Bayes Factor
    # BF ≈ sqrt(n) * exp(-0.5 * BIC_diff)
    # Using Wagenmakers (2007) approximation
    bic_diff = np.log(n) - t**2

    # Approximate BF (simplified)
    bf = np.sqrt(n / (2 * np.pi)) * np.exp(t**2 / 2) * (1 + t**2 / df) ** (-(df + 1) / 2)

    # Bound to reasonable range
    bf = max(0.001, min(bf, 1000))

    return bf


def classify_evidence(bf: float) -> EvidenceStrength:
    """Classify Bayes Factor into evidence strength category."""
    if bf < 3:
        return EvidenceStrength.ANECDOTAL
    elif bf < 10:
        return EvidenceStrength.MODERATE
    elif bf < 30:
        return EvidenceStrength.STRONG
    elif bf < 100:
        return EvidenceStrength.VERY_STRONG
    else:
        return EvidenceStrength.EXTREME


def audit_measurement(
    name: str,
    values: np.ndarray,
    comparison_values: Optional[np.ndarray] = None,
    expected_direction: Optional[str] = None,  # "positive", "negative", "zero"
) -> StatisticalAudit:
    """
    Perform complete statistical audit of a telescope measurement.

    Args:
        name: Name of the measurement
        values: Observed values
        comparison_values: Optional comparison group
        expected_direction: Expected direction of effect

    Returns:
        StatisticalAudit with all relevant statistics
    """
    warnings_list = []
    recommendations = []

    # Sample size check
    n = len(values)
    if n < 10:
        warnings_list.append(f"Very small sample (N={n}). Results are exploratory only.")
        recommendations.append("Increase sample size to at least N=30 for reliable inference.")
    elif n < 30:
        warnings_list.append(f"Small sample (N={n}). Use Bayesian interpretation.")

    # Bootstrap CI
    ci = bootstrap_ci(values)

    # Effect size and power (if comparison group provided)
    effect = None
    power = None
    bf = 1.0

    if comparison_values is not None:
        effect = cohens_d(values, comparison_values)

        # Power analysis
        power = power_analysis(abs(effect.cohens_d), min(n, len(comparison_values)))

        if power.current_power < 0.80:
            warnings_list.append(f"Underpowered (power={power.current_power:.2f}). Need N≥{power.n_for_80_power} per group.")
            recommendations.append(f"Collect at least {power.n_for_80_power} samples per group for 80% power.")

        # Bayes Factor
        bf = bayes_factor_t(values, comparison_values)

        # CI crosses zero check
        if effect.ci_lower < 0 < effect.ci_upper:
            warnings_list.append("95% CI crosses zero. Effect may be null.")

    else:
        # Single sample: test against zero
        if ci.ci_lower < 0 < ci.ci_upper:
            warnings_list.append("95% CI includes zero. May be no effect.")

        # Simplified BF for single sample
        t_stat = ci.estimate / ci.se if ci.se > 0 else 0
        bf = np.sqrt(n) * np.exp(t_stat**2 / 2) / (1 + t_stat**2) if n > 2 else 1.0
        bf = max(0.001, min(bf, 1000))

    # Evidence strength
    evidence = classify_evidence(bf)

    # Replication probability (simplified)
    # Based on power and effect size
    if effect is not None:
        rep_prob = min(0.95, power.current_power * (1 - 1 / (1 + abs(effect.cohens_d))))
    else:
        rep_prob = 0.5 * (1 - abs(ci.ci_lower - ci.ci_upper) / (abs(ci.estimate) + 0.01))
        rep_prob = max(0.1, min(0.9, rep_prob))

    # Direction check
    if expected_direction == "positive" and ci.estimate < 0:
        warnings_list.append("Effect in opposite direction from expected.")
    elif expected_direction == "negative" and ci.estimate > 0:
        warnings_list.append("Effect in opposite direction from expected.")
    elif expected_direction == "zero" and (ci.ci_lower > 0.1 or ci.ci_upper < -0.1):
        warnings_list.append("Effect differs from expected null.")

    return StatisticalAudit(
        measurement_name=name,
        point_estimate=ci.estimate,
        bootstrap_ci=ci,
        power=power,
        effect_size=effect,
        evidence_strength=evidence,
        replication_probability=rep_prob,
        warnings=warnings_list,
        recommendations=recommendations,
    )


def audit_telescope_claim(
    claim: str,
    data: Dict[str, np.ndarray],
) -> str:
    """
    Audit a specific claim from the telescope paper.

    Args:
        claim: The claim being made
        data: Dictionary of relevant data arrays

    Returns:
        Formatted audit report
    """
    report = []
    report.append(f"STATISTICAL AUDIT: {claim}")
    report.append("=" * 60)

    for name, values in data.items():
        audit = audit_measurement(name, values)

        report.append(f"\n{name}:")
        report.append(f"  Point estimate: {audit.point_estimate:.3f}")
        report.append(f"  95% CI: [{audit.bootstrap_ci.ci_lower:.3f}, {audit.bootstrap_ci.ci_upper:.3f}]")
        report.append(f"  Evidence: {audit.evidence_strength.value}")
        report.append(f"  Replication probability: {audit.replication_probability:.2f}")

        if audit.effect_size:
            report.append(f"  Effect size: d={audit.effect_size.cohens_d:.2f} ({audit.effect_size.interpretation})")

        if audit.warnings:
            report.append(f"  WARNINGS:")
            for w in audit.warnings:
                report.append(f"    - {w}")

        if audit.recommendations:
            report.append(f"  RECOMMENDATIONS:")
            for r in audit.recommendations:
                report.append(f"    - {r}")

    return "\n".join(report)


def demo_audit():
    """Demonstrate the statistical audit on telescope claims."""
    print("STATISTICAL AUDIT DEMONSTRATION")
    print("=" * 60)

    # Simulate data from the paper's claims
    np.random.seed(42)

    # Claim 1: Uncertainty → Self-Agency
    uncertainty_scores = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.94, 1.0, 1.0, 0.8, 1.0])
    technical_scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    audit = audit_measurement(
        "self_agency (uncertainty vs technical)",
        uncertainty_scores,
        technical_scores,
    )

    print(f"\nCLAIM: Uncertainty activates self-agency")
    print(f"  Effect size: d={audit.effect_size.cohens_d:.2f} ({audit.effect_size.interpretation})")
    print(f"  95% CI: [{audit.effect_size.ci_lower:.2f}, {audit.effect_size.ci_upper:.2f}]")
    print(f"  Evidence: {audit.evidence_strength.value}")
    if audit.warnings:
        print(f"  WARNINGS: {audit.warnings}")

    # Claim 2: "We" activates belonging
    we_scores = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    i_scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    audit2 = audit_measurement(
        "belonging (we vs I)",
        we_scores,
        i_scores,
    )

    print(f"\nCLAIM: 'We' activates belonging")
    print(f"  Effect size: d={audit2.effect_size.cohens_d:.2f} ({audit2.effect_size.interpretation})")
    print(f"  NOTE: This is by construction (regex detects pronouns)")
    if audit2.warnings:
        print(f"  WARNINGS: {audit2.warnings}")

    # Claim 3: Small sample naturalistic
    # Simulate what real data might look like
    real_uncertainty = np.random.normal(0.7, 0.3, 15)
    real_technical = np.random.normal(0.2, 0.25, 15)

    audit3 = audit_measurement(
        "naturalistic self_agency",
        real_uncertainty,
        real_technical,
    )

    print(f"\nSIMULATED NATURALISTIC DATA:")
    print(f"  Effect size: d={audit3.effect_size.cohens_d:.2f} ({audit3.effect_size.interpretation})")
    print(f"  95% CI: [{audit3.effect_size.ci_lower:.2f}, {audit3.effect_size.ci_upper:.2f}]")
    print(f"  Power: {audit3.power.current_power:.2f}")
    print(f"  N needed for 80% power: {audit3.power.n_for_80_power}")
    if audit3.warnings:
        print(f"  WARNINGS: {audit3.warnings}")


if __name__ == "__main__":
    demo_audit()
