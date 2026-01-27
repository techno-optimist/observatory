"""
Academic Statistics Module for Publication-Grade Research.

Provides statistical methods for rigorous academic analysis including:
- Bootstrap confidence intervals
- Effect size calculations (Cohen's d, Hedge's g)
- Multiple comparison correction (Bonferroni, FDR, Holm)
- Fisher Information Metric for probability distributions
- Phase transition detection with critical exponents

Based on expert analysis synthesis:
- Mathematical physicist: Fisher metric, fiber bundle structure
- Cognitive scientist: Proper effect sizes for cognitive claims
- Linguistics expert: Rigorous feature testing methodology
- AI researcher: Adversarial robustness metrics
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.special import rel_entr

logger = logging.getLogger(__name__)


# =============================================================================
# Effect Size Calculations
# =============================================================================

class EffectSizeInterpretation(Enum):
    """Cohen's d interpretation thresholds."""
    NEGLIGIBLE = "negligible"      # d < 0.2
    SMALL = "small"                # 0.2 <= d < 0.5
    MEDIUM = "medium"              # 0.5 <= d < 0.8
    LARGE = "large"                # d >= 0.8

    # For grammar deletion classification
    DECORATIVE = "decorative"      # d < 0.2 - can be removed without coordination loss
    MODIFYING = "modifying"        # 0.2 <= d < 0.5 - affects but doesn't determine
    NECESSARY = "necessary"        # 0.5 <= d < 0.8 - required for coordination
    CRITICAL = "critical"          # d >= 0.8 - fundamental to coordination


@dataclass
class EffectSize:
    """Effect size with interpretation and confidence interval."""

    d: float                                    # Cohen's d or Hedge's g
    standard_error: float
    confidence_interval: Tuple[float, float]    # 95% CI
    interpretation: EffectSizeInterpretation
    feature_classification: str                 # For grammar deletion
    n1: int                                     # Sample size group 1
    n2: int                                     # Sample size group 2
    method: str = "cohens_d"                    # or "hedges_g"

    def to_dict(self) -> dict:
        return {
            "d": self.d,
            "standard_error": self.standard_error,
            "confidence_interval": list(self.confidence_interval),
            "interpretation": self.interpretation.value,
            "feature_classification": self.feature_classification,
            "n1": self.n1,
            "n2": self.n2,
            "method": self.method
        }

    @property
    def is_significant(self) -> bool:
        """Check if CI excludes zero."""
        return self.confidence_interval[0] > 0 or self.confidence_interval[1] < 0


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    pooled: bool = True
) -> EffectSize:
    """
    Compute Cohen's d effect size between two groups.

    Args:
        group1: First group observations
        group2: Second group observations
        pooled: Use pooled standard deviation (default True)

    Returns:
        EffectSize with interpretation
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)

    if pooled:
        # Pooled standard deviation
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
    else:
        # Control group standard deviation
        std2 = np.std(group2, ddof=1)
        d = (mean1 - mean2) / std2 if std2 > 0 else 0.0

    # Standard error of d
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))

    # 95% CI
    z = 1.96
    ci = (d - z * se, d + z * se)

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interp = EffectSizeInterpretation.NEGLIGIBLE
        feature_class = "decorative"
    elif abs_d < 0.5:
        interp = EffectSizeInterpretation.SMALL
        feature_class = "modifying"
    elif abs_d < 0.8:
        interp = EffectSizeInterpretation.MEDIUM
        feature_class = "necessary"
    else:
        interp = EffectSizeInterpretation.LARGE
        feature_class = "critical"

    return EffectSize(
        d=d,
        standard_error=se,
        confidence_interval=ci,
        interpretation=interp,
        feature_classification=feature_class,
        n1=n1,
        n2=n2,
        method="cohens_d"
    )


def hedges_g(
    group1: np.ndarray,
    group2: np.ndarray
) -> EffectSize:
    """
    Compute Hedge's g (bias-corrected Cohen's d) for small samples.

    Recommended when n1 + n2 < 50.
    """
    effect = cohens_d(group1, group2)
    n = effect.n1 + effect.n2

    # Bias correction factor
    correction = 1 - (3 / (4 * n - 9))
    g = effect.d * correction
    se = effect.standard_error * correction
    ci = (effect.confidence_interval[0] * correction,
          effect.confidence_interval[1] * correction)

    return EffectSize(
        d=g,
        standard_error=se,
        confidence_interval=ci,
        interpretation=effect.interpretation,
        feature_classification=effect.feature_classification,
        n1=effect.n1,
        n2=effect.n2,
        method="hedges_g"
    )


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

@dataclass
class BootstrapEstimate:
    """Point estimate with bootstrap uncertainty quantification."""

    value: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    n_bootstrap: int
    bootstrap_distribution: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "standard_error": self.standard_error,
            "confidence_interval": list(self.confidence_interval),
            "confidence_level": self.confidence_level,
            "n_bootstrap": self.n_bootstrap
        }


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    method: str = "percentile"
) -> BootstrapEstimate:
    """
    Compute bootstrap confidence interval for any statistic.

    Args:
        data: Input data array
        statistic: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 for 95% CI)
        method: CI method - "percentile", "bca", or "basic"

    Returns:
        BootstrapEstimate with CI and standard error
    """
    n = len(data)
    point_estimate = statistic(data)

    # Generate bootstrap samples
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = data[np.random.randint(0, n, size=n)]
        bootstrap_stats[i] = statistic(sample)

    # Standard error
    se = np.std(bootstrap_stats)

    # Confidence interval
    alpha = 1 - confidence
    if method == "percentile":
        ci_low = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_high = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    elif method == "basic":
        ci_low = 2 * point_estimate - np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        ci_high = 2 * point_estimate - np.percentile(bootstrap_stats, 100 * alpha / 2)
    elif method == "bca":
        # Bias-corrected and accelerated bootstrap
        # Compute bias correction
        prop_below = np.mean(bootstrap_stats < point_estimate)
        z0 = stats.norm.ppf(prop_below) if 0 < prop_below < 1 else 0

        # Compute acceleration (jackknife)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jackknife_sample = np.delete(data, i)
            jackknife_stats[i] = statistic(jackknife_sample)

        jack_mean = np.mean(jackknife_stats)
        a = np.sum((jack_mean - jackknife_stats)**3) / (
            6 * (np.sum((jack_mean - jackknife_stats)**2))**(3/2) + 1e-10
        )

        # Adjusted percentiles
        z_alpha_low = stats.norm.ppf(alpha / 2)
        z_alpha_high = stats.norm.ppf(1 - alpha / 2)

        alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low)))
        alpha2 = stats.norm.cdf(z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high)))

        ci_low = np.percentile(bootstrap_stats, 100 * alpha1)
        ci_high = np.percentile(bootstrap_stats, 100 * alpha2)
    else:
        raise ValueError(f"Unknown method: {method}")

    return BootstrapEstimate(
        value=point_estimate,
        standard_error=se,
        confidence_interval=(ci_low, ci_high),
        confidence_level=confidence,
        n_bootstrap=n_bootstrap,
        bootstrap_distribution=bootstrap_stats
    )


def bootstrap_coordinate_ci(
    coordinates: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Dict[str, BootstrapEstimate]:
    """
    Compute bootstrap CIs for each axis of 3D coordinates.

    Args:
        coordinates: (N, 3) array of [agency, justice, belonging]
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Dictionary with bootstrap estimates per axis
    """
    axes = ["agency", "perceived_justice", "belonging"]
    results = {}

    for i, axis in enumerate(axes):
        results[axis] = bootstrap_ci(
            coordinates[:, i],
            statistic=np.mean,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            method="bca"
        )

    return results


# =============================================================================
# Multiple Comparison Correction
# =============================================================================

@dataclass
class CorrectedPValue:
    """P-value with multiple comparison correction."""

    original_p: float
    corrected_p: float
    is_significant: bool
    alpha: float
    correction_method: str
    test_index: int
    feature_name: str

    def to_dict(self) -> dict:
        return {
            "original_p": self.original_p,
            "corrected_p": self.corrected_p,
            "is_significant": self.is_significant,
            "alpha": self.alpha,
            "correction_method": self.correction_method,
            "feature_name": self.feature_name
        }


def bonferroni_correction(
    p_values: List[float],
    feature_names: List[str],
    alpha: float = 0.05
) -> List[CorrectedPValue]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Most conservative method - controls family-wise error rate (FWER).
    """
    m = len(p_values)
    corrected_alpha = alpha / m

    results = []
    for i, (p, name) in enumerate(zip(p_values, feature_names)):
        corrected_p = min(p * m, 1.0)
        results.append(CorrectedPValue(
            original_p=p,
            corrected_p=corrected_p,
            is_significant=p < corrected_alpha,
            alpha=alpha,
            correction_method="bonferroni",
            test_index=i,
            feature_name=name
        ))

    return results


def holm_correction(
    p_values: List[float],
    feature_names: List[str],
    alpha: float = 0.05
) -> List[CorrectedPValue]:
    """
    Apply Holm-Bonferroni step-down correction.

    Less conservative than Bonferroni while still controlling FWER.
    """
    m = len(p_values)

    # Sort by p-value
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    sorted_names = np.array(feature_names)[sorted_indices]

    # Holm correction
    rejected = np.zeros(m, dtype=bool)
    corrected_p = np.zeros(m)

    for i in range(m):
        threshold = alpha / (m - i)
        corrected_p[i] = min(sorted_p[i] * (m - i), 1.0)
        if sorted_p[i] < threshold:
            rejected[i] = True
        else:
            break

    # Ensure monotonicity
    for i in range(1, m):
        corrected_p[i] = max(corrected_p[i], corrected_p[i-1])

    # Build results in original order
    results = [None] * m
    for i, orig_idx in enumerate(sorted_indices):
        results[orig_idx] = CorrectedPValue(
            original_p=sorted_p[i],
            corrected_p=corrected_p[i],
            is_significant=rejected[i],
            alpha=alpha,
            correction_method="holm",
            test_index=orig_idx,
            feature_name=sorted_names[i]
        )

    return results


def fdr_correction(
    p_values: List[float],
    feature_names: List[str],
    alpha: float = 0.05
) -> List[CorrectedPValue]:
    """
    Apply Benjamini-Hochberg FDR correction.

    Controls false discovery rate - less conservative, more power.
    Recommended for exploratory analysis.
    """
    m = len(p_values)

    # Sort by p-value
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    sorted_names = np.array(feature_names)[sorted_indices]

    # BH correction
    corrected_p = np.zeros(m)
    for i in range(m):
        corrected_p[i] = sorted_p[i] * m / (i + 1)

    # Ensure monotonicity (from end)
    for i in range(m - 2, -1, -1):
        corrected_p[i] = min(corrected_p[i], corrected_p[i + 1])

    corrected_p = np.minimum(corrected_p, 1.0)

    # Build results in original order
    results = [None] * m
    for i, orig_idx in enumerate(sorted_indices):
        results[orig_idx] = CorrectedPValue(
            original_p=sorted_p[i],
            corrected_p=corrected_p[i],
            is_significant=corrected_p[i] < alpha,
            alpha=alpha,
            correction_method="fdr_bh",
            test_index=orig_idx,
            feature_name=sorted_names[i]
        )

    return results


def apply_correction(
    p_values: List[float],
    feature_names: List[str],
    method: str = "holm",
    alpha: float = 0.05
) -> List[CorrectedPValue]:
    """
    Apply multiple comparison correction.

    Args:
        p_values: List of p-values from individual tests
        feature_names: Names of features being tested
        method: "bonferroni", "holm", or "fdr"
        alpha: Significance level

    Returns:
        List of CorrectedPValue results
    """
    if method == "bonferroni":
        return bonferroni_correction(p_values, feature_names, alpha)
    elif method == "holm":
        return holm_correction(p_values, feature_names, alpha)
    elif method == "fdr":
        return fdr_correction(p_values, feature_names, alpha)
    else:
        raise ValueError(f"Unknown correction method: {method}")


# =============================================================================
# Fisher Information Metric
# =============================================================================

def fisher_rao_distance(
    p1: np.ndarray,
    p2: np.ndarray
) -> float:
    """
    Compute Fisher-Rao distance between probability distributions.

    This is the natural metric on the statistical manifold of probability
    distributions. It measures how distinguishable two distributions are.

    Formula: d_FR(p, q) = 2 * arccos(sum(sqrt(p_i * q_i)))

    Args:
        p1: First probability distribution (must sum to 1)
        p2: Second probability distribution (must sum to 1)

    Returns:
        Fisher-Rao distance (0 to pi)
    """
    # Normalize to ensure valid distributions
    p1 = np.array(p1) / np.sum(p1)
    p2 = np.array(p2) / np.sum(p2)

    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p1 * p2))

    # Clamp for numerical stability
    bc = np.clip(bc, -1.0, 1.0)

    # Fisher-Rao distance
    return 2.0 * np.arccos(bc)


def hellinger_distance(
    p1: np.ndarray,
    p2: np.ndarray
) -> float:
    """
    Compute Hellinger distance between distributions.

    Related to Fisher-Rao: H(p, q) = sin(d_FR(p, q) / 2)
    Range: [0, 1]
    """
    p1 = np.array(p1) / np.sum(p1)
    p2 = np.array(p2) / np.sum(p2)

    return np.sqrt(0.5 * np.sum((np.sqrt(p1) - np.sqrt(p2))**2))


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray
) -> float:
    """
    Compute KL divergence D_KL(p || q).

    Not symmetric. Measures information lost when q approximates p.
    """
    p = np.array(p) / np.sum(p)
    q = np.array(q) / np.sum(q)

    # Add small epsilon for numerical stability
    q = np.clip(q, 1e-10, None)

    return np.sum(rel_entr(p, q))


def jensen_shannon_distance(
    p1: np.ndarray,
    p2: np.ndarray
) -> float:
    """
    Compute Jensen-Shannon distance (sqrt of JS divergence).

    Symmetric and bounded [0, 1]. Often preferred over KL.
    """
    p1 = np.array(p1) / np.sum(p1)
    p2 = np.array(p2) / np.sum(p2)

    m = 0.5 * (p1 + p2)
    js_div = 0.5 * kl_divergence(p1, m) + 0.5 * kl_divergence(p2, m)

    return np.sqrt(js_div)


@dataclass
class ManifoldDistance:
    """Distance measurement between two manifold positions."""

    euclidean: float                    # L2 distance in coordinate space
    fisher_rao: float                   # Fisher-Rao on mode distribution
    hellinger: float                    # Hellinger distance
    jensen_shannon: float               # JS distance
    cosine: float                       # Cosine distance (1 - similarity)

    def to_dict(self) -> dict:
        return {
            "euclidean": self.euclidean,
            "fisher_rao": self.fisher_rao,
            "hellinger": self.hellinger,
            "jensen_shannon": self.jensen_shannon,
            "cosine": self.cosine
        }


def manifold_distance(
    coords1: np.ndarray,
    coords2: np.ndarray,
    mode_dist1: Optional[np.ndarray] = None,
    mode_dist2: Optional[np.ndarray] = None
) -> ManifoldDistance:
    """
    Compute multiple distance metrics between manifold positions.

    Args:
        coords1: First 3D coordinate [agency, justice, belonging]
        coords2: Second 3D coordinate
        mode_dist1: Optional mode probability distribution for first point
        mode_dist2: Optional mode probability distribution for second point

    Returns:
        ManifoldDistance with multiple metrics
    """
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)

    # Euclidean distance
    euclidean = np.linalg.norm(coords1 - coords2)

    # Cosine distance
    norm1, norm2 = np.linalg.norm(coords1), np.linalg.norm(coords2)
    if norm1 > 0 and norm2 > 0:
        cosine = 1.0 - np.dot(coords1, coords2) / (norm1 * norm2)
    else:
        cosine = 1.0

    # Distribution-based distances (if mode distributions provided)
    if mode_dist1 is not None and mode_dist2 is not None:
        fr = fisher_rao_distance(mode_dist1, mode_dist2)
        hell = hellinger_distance(mode_dist1, mode_dist2)
        js = jensen_shannon_distance(mode_dist1, mode_dist2)
    else:
        # Use coordinate-based proxy distributions
        # Convert coordinates to pseudo-probability distribution
        pseudo1 = np.exp(coords1 - np.max(coords1))
        pseudo1 = pseudo1 / np.sum(pseudo1)
        pseudo2 = np.exp(coords2 - np.max(coords2))
        pseudo2 = pseudo2 / np.sum(pseudo2)

        fr = fisher_rao_distance(pseudo1, pseudo2)
        hell = hellinger_distance(pseudo1, pseudo2)
        js = jensen_shannon_distance(pseudo1, pseudo2)

    return ManifoldDistance(
        euclidean=euclidean,
        fisher_rao=fr,
        hellinger=hell,
        jensen_shannon=js,
        cosine=cosine
    )


# =============================================================================
# Phase Transition Detection
# =============================================================================

@dataclass
class PhaseTransition:
    """Detected phase transition with critical exponents."""

    transition_point: float             # Control parameter value at transition
    order_parameter_jump: float         # Discontinuity in order parameter
    beta: Optional[float] = None        # Critical exponent (order parameter)
    gamma: Optional[float] = None       # Critical exponent (susceptibility)
    correlation_length: Optional[float] = None
    transition_type: str = "unknown"    # "first_order", "second_order", "crossover"
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "transition_point": self.transition_point,
            "order_parameter_jump": self.order_parameter_jump,
            "beta": self.beta,
            "gamma": self.gamma,
            "correlation_length": self.correlation_length,
            "transition_type": self.transition_type,
            "confidence": self.confidence
        }


def detect_phase_transitions(
    control_parameter: np.ndarray,
    order_parameter: np.ndarray,
    window_size: int = 5,
    threshold: float = 0.3
) -> List[PhaseTransition]:
    """
    Detect phase transitions in order parameter time series.

    Uses derivative analysis and variance peaks to find transitions.

    Args:
        control_parameter: Independent variable (e.g., compression level)
        order_parameter: Dependent variable (e.g., legibility score)
        window_size: Smoothing window size
        threshold: Minimum derivative magnitude for transition

    Returns:
        List of detected phase transitions
    """
    n = len(control_parameter)
    if n < 2 * window_size:
        return []

    # Sort by control parameter
    sort_idx = np.argsort(control_parameter)
    x = control_parameter[sort_idx]
    y = order_parameter[sort_idx]

    # Compute numerical derivative
    dy = np.gradient(y, x)

    # Smooth derivative
    kernel = np.ones(window_size) / window_size
    dy_smooth = np.convolve(dy, kernel, mode='valid')
    x_smooth = x[window_size//2:-(window_size//2) or None]

    if len(x_smooth) == 0:
        return []

    # Find peaks in absolute derivative (rapid changes)
    abs_dy = np.abs(dy_smooth)

    transitions = []

    # Find local maxima in derivative
    for i in range(1, len(abs_dy) - 1):
        if abs_dy[i] > abs_dy[i-1] and abs_dy[i] > abs_dy[i+1]:
            if abs_dy[i] > threshold:
                # Found a potential transition
                trans_point = x_smooth[i]

                # Estimate order parameter jump
                left_mean = np.mean(y[x < trans_point][-window_size:]) if np.sum(x < trans_point) >= window_size else y[0]
                right_mean = np.mean(y[x > trans_point][:window_size]) if np.sum(x > trans_point) >= window_size else y[-1]
                jump = abs(right_mean - left_mean)

                # Classify transition type based on derivative continuity
                if jump > 0.5:
                    trans_type = "first_order"
                elif abs_dy[i] > 2 * threshold:
                    trans_type = "second_order"
                else:
                    trans_type = "crossover"

                # Confidence based on derivative magnitude
                confidence = min(abs_dy[i] / (2 * threshold), 1.0)

                transitions.append(PhaseTransition(
                    transition_point=trans_point,
                    order_parameter_jump=jump,
                    transition_type=trans_type,
                    confidence=confidence
                ))

    return transitions


def estimate_critical_exponents(
    control_parameter: np.ndarray,
    order_parameter: np.ndarray,
    transition_point: float,
    window: float = 0.2
) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate critical exponents near a phase transition.

    Near a continuous transition, order parameter ~ |t|^beta
    where t = (control - critical) / critical

    Args:
        control_parameter: Control parameter values
        order_parameter: Order parameter values
        transition_point: Critical point estimate
        window: Fraction of data to use around transition

    Returns:
        Tuple of (beta, gamma) estimates, or None if estimation fails
    """
    # Select data near transition
    x_range = np.max(control_parameter) - np.min(control_parameter)
    mask = np.abs(control_parameter - transition_point) < window * x_range

    if np.sum(mask) < 10:
        return None, None

    x = control_parameter[mask]
    y = order_parameter[mask]

    # Reduced control parameter
    t = (x - transition_point) / (transition_point + 1e-10)
    t = t[t != 0]  # Exclude exact transition point
    y_near = y[np.abs(control_parameter[mask] - transition_point) > 1e-10]

    if len(t) < 5:
        return None, None

    # Log-log fit for power law: log(y) ~ beta * log(|t|)
    try:
        log_t = np.log(np.abs(t))
        log_y = np.log(np.abs(y_near) + 1e-10)

        # Linear regression
        A = np.vstack([log_t, np.ones(len(log_t))]).T
        beta, _ = np.linalg.lstsq(A, log_y, rcond=None)[0]

        return float(beta), None  # gamma requires susceptibility data
    except Exception:
        return None, None


# =============================================================================
# Academic Output Formatting
# =============================================================================

def format_effect_size_table(
    results: List[EffectSize],
    feature_names: List[str]
) -> str:
    """Format effect sizes as publication-ready table."""
    lines = [
        "Feature | Cohen's d | 95% CI | Classification",
        "--------|-----------|--------|---------------"
    ]

    for name, es in zip(feature_names, results):
        ci_str = f"[{es.confidence_interval[0]:.2f}, {es.confidence_interval[1]:.2f}]"
        lines.append(f"{name} | {es.d:.3f} | {ci_str} | {es.feature_classification}")

    return "\n".join(lines)


def generate_latex_effect_table(
    results: List[EffectSize],
    feature_names: List[str],
    caption: str = "Grammar deletion effect sizes",
    label: str = "tab:effect_sizes"
) -> str:
    """Generate LaTeX table for publication."""
    rows = []
    for name, es in zip(feature_names, results):
        sig = "*" if es.is_significant else ""
        rows.append(
            f"    {name} & {es.d:.3f}{sig} & "
            f"[{es.confidence_interval[0]:.2f}, {es.confidence_interval[1]:.2f}] & "
            f"{es.feature_classification} \\\\"
        )

    return f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{lccc}}
\\toprule
Feature & Cohen's $d$ & 95\\% CI & Classification \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def format_bootstrap_summary(
    estimates: Dict[str, BootstrapEstimate]
) -> str:
    """Format bootstrap estimates as summary text."""
    lines = []
    for axis, est in estimates.items():
        lines.append(
            f"{axis}: {est.value:.3f} "
            f"(95% CI: [{est.confidence_interval[0]:.3f}, {est.confidence_interval[1]:.3f}], "
            f"SE = {est.standard_error:.3f})"
        )
    return "\n".join(lines)
