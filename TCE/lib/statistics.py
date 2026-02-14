"""
Statistical Utilities for Cognitive Elements Research

Provides proper uncertainty quantification for experiment results:
- Wilson score confidence intervals (for proportions)
- Cohen's d effect size
- McNemar's test (for paired comparisons)
- Bootstrap confidence intervals
"""

import math
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class WilsonScore:
    """Wilson score confidence interval for a proportion."""
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    n: int
    successes: int


@dataclass
class EffectSize:
    """Cohen's d effect size with interpretation."""
    cohens_d: float
    interpretation: str  # negligible, small, medium, large

    @staticmethod
    def interpret(d: float) -> str:
        """Interpret Cohen's d magnitude."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"


@dataclass
class McNemarResult:
    """McNemar's test result for paired comparisons."""
    statistic: float
    p_value: float
    significant: bool
    n_discordant: int
    interpretation: str


def wilson_score_interval(
    successes: int,
    n: int,
    confidence: float = 0.95
) -> WilsonScore:
    """
    Calculate Wilson score confidence interval for a proportion.

    More accurate than normal approximation, especially for small samples
    or proportions near 0 or 1.

    Args:
        successes: Number of successes (e.g., triggered trials)
        n: Total number of trials
        confidence: Confidence level (default 0.95)

    Returns:
        WilsonScore with point estimate and CI bounds
    """
    if n == 0:
        return WilsonScore(
            point_estimate=0.0,
            lower=0.0,
            upper=0.0,
            confidence_level=confidence,
            n=0,
            successes=0
        )

    # Z-score for confidence level
    # For 95%: z ≈ 1.96
    z = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }.get(confidence, 1.96)

    p_hat = successes / n

    # Wilson score formula
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    margin = (z / denominator) * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))

    lower = max(0, center - margin)
    upper = min(1, center + margin)

    return WilsonScore(
        point_estimate=p_hat,
        lower=lower,
        upper=upper,
        confidence_level=confidence,
        n=n,
        successes=successes
    )


def cohens_d(
    group1: List[float],
    group2: List[float]
) -> EffectSize:
    """
    Calculate Cohen's d effect size between two groups.

    Args:
        group1: Values from first group
        group2: Values from second group

    Returns:
        EffectSize with d value and interpretation
    """
    if not group1 or not group2:
        return EffectSize(cohens_d=0.0, interpretation="negligible")

    n1, n2 = len(group1), len(group2)
    mean1 = sum(group1) / n1
    mean2 = sum(group2) / n2

    # Pooled standard deviation
    var1 = sum((x - mean1)**2 for x in group1) / max(1, n1 - 1)
    var2 = sum((x - mean2)**2 for x in group2) / max(1, n2 - 1)

    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        d = 0.0
    else:
        d = (mean1 - mean2) / pooled_std

    return EffectSize(
        cohens_d=d,
        interpretation=EffectSize.interpret(d)
    )


def mcnemar_test(
    before_success_after_fail: int,
    before_fail_after_success: int,
    alpha: float = 0.05
) -> McNemarResult:
    """
    McNemar's test for paired nominal data.

    Useful for comparing two conditions (e.g., baseline vs trained)
    on the same set of prompts.

    Args:
        before_success_after_fail: Count where condition 1 succeeded, condition 2 failed
        before_fail_after_success: Count where condition 1 failed, condition 2 succeeded
        alpha: Significance level

    Returns:
        McNemarResult with test statistic and p-value
    """
    b = before_success_after_fail
    c = before_fail_after_success
    n_discordant = b + c

    if n_discordant == 0:
        return McNemarResult(
            statistic=0.0,
            p_value=1.0,
            significant=False,
            n_discordant=0,
            interpretation="No discordant pairs - conditions are identical"
        )

    # McNemar statistic with continuity correction
    statistic = (abs(b - c) - 1)**2 / (b + c)

    # Chi-square distribution with 1 df
    # Approximate p-value using chi-square CDF
    # For simplicity, use lookup table for common values
    p_value = _chi2_p_value(statistic, df=1)

    significant = p_value < alpha

    if significant:
        if c > b:
            interpretation = f"Condition 2 significantly better (p={p_value:.4f})"
        else:
            interpretation = f"Condition 1 significantly better (p={p_value:.4f})"
    else:
        interpretation = f"No significant difference (p={p_value:.4f})"

    return McNemarResult(
        statistic=statistic,
        p_value=p_value,
        significant=significant,
        n_discordant=n_discordant,
        interpretation=interpretation
    )


def _chi2_p_value(x: float, df: int = 1) -> float:
    """
    Approximate chi-square p-value.

    For df=1, uses the relationship with normal distribution.
    """
    if x <= 0:
        return 1.0

    # For df=1: chi2 = z^2, so we can use normal approximation
    z = math.sqrt(x)

    # Approximate normal CDF using error function approximation
    # P(Z > z) = 0.5 * erfc(z / sqrt(2))
    p = 0.5 * math.erfc(z / math.sqrt(2))

    # Two-tailed
    return 2 * p


def bootstrap_ci(
    values: List[float],
    statistic: str = "mean",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a statistic.

    Args:
        values: Sample values
        statistic: "mean" or "median"
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        seed: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    import random

    if seed is not None:
        random.seed(seed)

    if not values:
        return (0.0, 0.0, 0.0)

    n = len(values)

    def compute_stat(sample: List[float]) -> float:
        if statistic == "mean":
            return sum(sample) / len(sample)
        elif statistic == "median":
            sorted_sample = sorted(sample)
            mid = len(sorted_sample) // 2
            if len(sorted_sample) % 2 == 0:
                return (sorted_sample[mid-1] + sorted_sample[mid]) / 2
            return sorted_sample[mid]
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

    # Point estimate
    point_estimate = compute_stat(values)

    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = [random.choice(values) for _ in range(n)]
        bootstrap_stats.append(compute_stat(sample))

    # Percentile method for CI
    bootstrap_stats.sort()
    alpha = 1 - confidence
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)

    return (
        point_estimate,
        bootstrap_stats[lower_idx],
        bootstrap_stats[min(upper_idx, n_bootstrap - 1)]
    )


def power_analysis(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    Compute required sample size for detecting an effect.

    Simplified formula for two-group comparison.

    Args:
        effect_size: Expected Cohen's d
        alpha: Significance level
        power: Desired power (1 - beta)

    Returns:
        Required sample size per group
    """
    # Z-scores
    z_alpha = {0.01: 2.576, 0.05: 1.96, 0.10: 1.645}.get(alpha, 1.96)
    z_beta = {0.80: 0.84, 0.90: 1.28, 0.95: 1.645}.get(power, 0.84)

    if effect_size == 0:
        return float('inf')

    # Sample size formula
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

    return int(math.ceil(n))


# Convenience functions for common use cases

def trigger_rate_with_ci(
    triggered: int,
    total: int,
    confidence: float = 0.95
) -> dict:
    """
    Calculate trigger rate with Wilson score CI.

    Returns dict suitable for JSON serialization.
    """
    ws = wilson_score_interval(triggered, total, confidence)
    return {
        "point_estimate": ws.point_estimate,
        "wilson_lower": ws.lower,
        "wilson_upper": ws.upper,
        "confidence_level": ws.confidence_level,
        "n": ws.n,
        "successes": ws.successes
    }


def compare_conditions(
    baseline_results: List[bool],
    treatment_results: List[bool]
) -> dict:
    """
    Compare two conditions on the same prompts.

    Args:
        baseline_results: List of success/fail for baseline
        treatment_results: List of success/fail for treatment

    Returns:
        Dict with McNemar test and effect size
    """
    if len(baseline_results) != len(treatment_results):
        raise ValueError("Results must have same length for paired comparison")

    # Count discordant pairs
    b = sum(1 for bl, tr in zip(baseline_results, treatment_results) if bl and not tr)
    c = sum(1 for bl, tr in zip(baseline_results, treatment_results) if not bl and tr)

    mcnemar = mcnemar_test(b, c)

    # Effect size as difference in proportions
    baseline_rate = sum(baseline_results) / len(baseline_results)
    treatment_rate = sum(treatment_results) / len(treatment_results)

    return {
        "baseline_rate": baseline_rate,
        "treatment_rate": treatment_rate,
        "difference": treatment_rate - baseline_rate,
        "mcnemar": {
            "statistic": mcnemar.statistic,
            "p_value": mcnemar.p_value,
            "significant": mcnemar.significant,
            "interpretation": mcnemar.interpretation
        }
    }


# Multiple comparison correction

def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> List[Tuple[float, bool]]:
    """
    Bonferroni correction for multiple comparisons.

    Most conservative - controls family-wise error rate.

    Args:
        p_values: List of p-values to correct
        alpha: Significance threshold

    Returns:
        List of (adjusted_p, significant) tuples
    """
    n = len(p_values)
    adjusted_alpha = alpha / n

    return [
        (min(p * n, 1.0), p * n <= alpha)
        for p in p_values
    ]


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> List[Tuple[float, bool, int]]:
    """
    Holm-Bonferroni step-down correction.

    Less conservative than Bonferroni, still controls FWER.

    Args:
        p_values: List of p-values to correct
        alpha: Significance threshold

    Returns:
        List of (adjusted_p, significant, original_index) tuples
    """
    n = len(p_values)

    # Sort p-values with indices
    indexed = [(p, i) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda x: x[0])

    results = [None] * n
    cumulative_reject = True

    for rank, (p, orig_idx) in enumerate(indexed):
        # Adjusted threshold for this rank
        threshold = alpha / (n - rank)

        # Can only reject if all smaller p-values were rejected
        significant = cumulative_reject and p <= threshold
        if not significant:
            cumulative_reject = False

        # Adjusted p-value (step-down)
        adjusted_p = min(p * (n - rank), 1.0)

        results[orig_idx] = (adjusted_p, significant, orig_idx)

    return results


def benjamini_hochberg_fdr(
    p_values: List[float],
    alpha: float = 0.05
) -> List[Tuple[float, bool, int]]:
    """
    Benjamini-Hochberg FDR correction.

    Controls false discovery rate instead of family-wise error rate.
    More powerful when making many comparisons.

    Args:
        p_values: List of p-values to correct
        alpha: Target FDR (q-value threshold)

    Returns:
        List of (adjusted_p, significant, original_index) tuples
    """
    n = len(p_values)

    # Sort p-values with indices
    indexed = [(p, i) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda x: x[0])

    results = [None] * n

    # Find largest k where p_(k) <= k/n * alpha
    max_significant_k = 0
    for k, (p, _) in enumerate(indexed, 1):
        threshold = (k / n) * alpha
        if p <= threshold:
            max_significant_k = k

    # Compute adjusted p-values (q-values)
    # Work backwards to ensure monotonicity
    prev_adjusted = 1.0
    for k in range(n, 0, -1):
        p, orig_idx = indexed[k - 1]
        adjusted = min(p * n / k, prev_adjusted)
        prev_adjusted = adjusted
        significant = k <= max_significant_k
        results[orig_idx] = (adjusted, significant, orig_idx)

    return results


# Cross-architecture analysis

@dataclass
class CrossArchitectureAnalysis:
    """Analysis of results across multiple architectures."""
    element_id: str
    architectures: List[str]
    rates: List[float]
    n_samples: List[int]
    mean_rate: float
    std_rate: float
    min_arch: Tuple[str, float]
    max_arch: Tuple[str, float]
    heterogeneity: float  # Coefficient of variation
    is_stable: bool  # CV < 0.15 considered stable


def cross_architecture_analysis(
    results: dict,  # {arch: {element: rate}}
    element_id: str,
    stability_threshold: float = 0.15
) -> CrossArchitectureAnalysis:
    """
    Analyze element performance across architectures.

    Identifies:
    - Variance across architectures
    - Best/worst performing architectures
    - Whether element is stable (low variance)

    Args:
        results: Dict mapping architecture to element rates
        element_id: Element to analyze
        stability_threshold: CV threshold for stability

    Returns:
        CrossArchitectureAnalysis with metrics
    """
    architectures = []
    rates = []
    n_samples = []

    for arch, arch_results in results.items():
        if element_id in arch_results:
            result = arch_results[element_id]
            architectures.append(arch)
            rates.append(result.get("rate", result) if isinstance(result, dict) else result)
            n_samples.append(result.get("n", 25) if isinstance(result, dict) else 25)

    if not rates:
        raise ValueError(f"No results for element {element_id}")

    mean_rate = sum(rates) / len(rates)

    # Standard deviation
    if len(rates) > 1:
        variance = sum((r - mean_rate) ** 2 for r in rates) / (len(rates) - 1)
        std_rate = math.sqrt(variance)
    else:
        std_rate = 0.0

    # Coefficient of variation (heterogeneity)
    cv = std_rate / mean_rate if mean_rate > 0 else 0.0

    # Find min/max
    min_idx = rates.index(min(rates))
    max_idx = rates.index(max(rates))

    return CrossArchitectureAnalysis(
        element_id=element_id,
        architectures=architectures,
        rates=rates,
        n_samples=n_samples,
        mean_rate=mean_rate,
        std_rate=std_rate,
        min_arch=(architectures[min_idx], rates[min_idx]),
        max_arch=(architectures[max_idx], rates[max_idx]),
        heterogeneity=cv,
        is_stable=cv < stability_threshold
    )


def multi_architecture_summary(
    results: dict,  # {arch: {element: rate}}
    elements: Optional[List[str]] = None
) -> dict:
    """
    Summary statistics across all architectures and elements.

    Returns:
        Dict with:
        - per_element: CrossArchitectureAnalysis for each element
        - overall: Aggregate metrics
        - unstable_elements: Elements with high cross-arch variance
    """
    if elements is None:
        # Get all elements from first architecture
        first_arch = next(iter(results.values()))
        elements = list(first_arch.keys())

    per_element = {}
    unstable = []

    for elem in elements:
        try:
            analysis = cross_architecture_analysis(results, elem)
            per_element[elem] = analysis
            if not analysis.is_stable:
                unstable.append(elem)
        except ValueError:
            continue

    # Overall statistics
    all_rates = [a.mean_rate for a in per_element.values()]
    mean_of_means = sum(all_rates) / len(all_rates) if all_rates else 0

    return {
        "per_element": per_element,
        "n_elements": len(per_element),
        "n_architectures": len(results),
        "mean_rate": mean_of_means,
        "unstable_elements": unstable,
        "stability_rate": (len(per_element) - len(unstable)) / len(per_element) if per_element else 1.0
    }
