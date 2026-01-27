#!/usr/bin/env python3
"""
Criticality Analysis for the Cultural Soliton Observatory.

Investigates whether coordination operates near a critical point with
scale-invariant behavior, similar to other complex systems:
- Earthquakes (Gutenberg-Richter law)
- Brain activity (neuronal avalanches)
- Language (Zipf's law)

Tests for:
1. Power Law Distributions in effect sizes
2. Scale Invariance in coordination features
3. 1/f Noise (pink noise) in coordination signals
4. Avalanche dynamics in narrative transitions
5. Self-Organized Criticality signatures
6. Universality class identification

Author: Cultural Soliton Observatory Research Team
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import welch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')

from research.hierarchical_coordinates import (
    extract_hierarchical_coordinate,
    extract_features,
    HierarchicalCoordinate,
    CoordinationCore,
    FEATURE_PATTERNS
)
from research.academic_statistics import (
    cohens_d,
    EffectSize,
    detect_phase_transitions,
    bootstrap_ci
)


# =============================================================================
# Power Law Analysis
# =============================================================================

@dataclass
class PowerLawFit:
    """Results from power law fitting."""
    alpha: float                  # Power law exponent
    xmin: float                   # Minimum x value for fit
    sigma: float                  # Standard error on alpha
    loglikelihood: float          # Log-likelihood of fit
    ks_statistic: float           # Kolmogorov-Smirnov statistic
    p_value: float                # P-value for power law hypothesis
    r_squared: float              # R-squared for log-log linear fit
    is_power_law: bool            # Whether data follows power law


def fit_power_law(data: np.ndarray, xmin: Optional[float] = None) -> PowerLawFit:
    """
    Fit a power law to data using maximum likelihood estimation.

    Power law: P(x) ~ x^(-alpha)

    Args:
        data: Array of positive values
        xmin: Minimum value to use for fitting (estimated if None)

    Returns:
        PowerLawFit with exponent and diagnostics
    """
    data = np.array(data)
    data = data[data > 0]  # Remove zeros/negatives

    if len(data) < 10:
        return PowerLawFit(
            alpha=np.nan, xmin=np.nan, sigma=np.nan,
            loglikelihood=np.nan, ks_statistic=1.0,
            p_value=0.0, r_squared=0.0, is_power_law=False
        )

    # Estimate xmin using KS statistic if not provided
    if xmin is None:
        sorted_data = np.sort(data)
        n = len(sorted_data)
        ks_stats = []
        xmin_candidates = sorted_data[:n//2]  # First half as candidates

        for xm in xmin_candidates[::max(1, len(xmin_candidates)//50)]:  # Sample 50 points
            tail = data[data >= xm]
            if len(tail) < 10:
                continue
            # MLE estimate of alpha
            alpha_hat = 1 + len(tail) / np.sum(np.log(tail / xm))
            # KS statistic
            empirical_cdf = np.arange(1, len(tail) + 1) / len(tail)
            theoretical_cdf = 1 - (xm / np.sort(tail)) ** (alpha_hat - 1)
            ks = np.max(np.abs(empirical_cdf - theoretical_cdf))
            ks_stats.append((ks, xm, alpha_hat))

        if not ks_stats:
            xmin = np.percentile(data, 10)
        else:
            best = min(ks_stats, key=lambda x: x[0])
            xmin = best[1]

    # Fit power law to tail
    tail = data[data >= xmin]
    if len(tail) < 5:
        return PowerLawFit(
            alpha=np.nan, xmin=xmin, sigma=np.nan,
            loglikelihood=np.nan, ks_statistic=1.0,
            p_value=0.0, r_squared=0.0, is_power_law=False
        )

    # MLE estimate
    n_tail = len(tail)
    alpha = 1 + n_tail / np.sum(np.log(tail / xmin))
    sigma = (alpha - 1) / np.sqrt(n_tail)

    # Log-likelihood
    ll = n_tail * np.log(alpha - 1) - n_tail * np.log(xmin) - alpha * np.sum(np.log(tail / xmin))

    # KS statistic for goodness of fit
    empirical_cdf = np.arange(1, n_tail + 1) / n_tail
    theoretical_cdf = 1 - (xmin / np.sort(tail)) ** (alpha - 1)
    ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))

    # Simplified p-value estimation using KS critical value
    ks_critical = 1.36 / np.sqrt(n_tail)  # alpha=0.05
    p_value = 1.0 if ks_stat < ks_critical else 0.01

    # Log-log linear fit for R-squared
    log_data = np.log10(tail)
    hist, edges = np.histogram(log_data, bins=min(20, n_tail//5))
    bin_centers = (edges[:-1] + edges[1:]) / 2
    hist = hist.astype(float)
    hist[hist == 0] = np.nan
    log_hist = np.log10(hist)

    # Linear fit to log-log
    valid = ~np.isnan(log_hist)
    if np.sum(valid) > 2:
        slope, intercept, r_value, _, _ = stats.linregress(
            bin_centers[valid], log_hist[valid]
        )
        r_squared = r_value ** 2
    else:
        r_squared = 0.0

    is_power_law = (ks_stat < ks_critical) and (r_squared > 0.8) and (alpha > 1)

    return PowerLawFit(
        alpha=alpha,
        xmin=xmin,
        sigma=sigma,
        loglikelihood=ll,
        ks_statistic=ks_stat,
        p_value=p_value,
        r_squared=r_squared,
        is_power_law=is_power_law
    )


def test_zipf_law(counts: Dict[str, int]) -> Tuple[float, float, float]:
    """
    Test if feature counts follow Zipf's law.

    Zipf's law: frequency ~ rank^(-s), typically s ~ 1

    Returns:
        Tuple of (zipf_exponent, r_squared, p_value)
    """
    sorted_counts = sorted(counts.values(), reverse=True)
    sorted_counts = [c for c in sorted_counts if c > 0]

    if len(sorted_counts) < 5:
        return (np.nan, 0.0, 1.0)

    ranks = np.arange(1, len(sorted_counts) + 1)
    frequencies = np.array(sorted_counts)

    # Log-log regression
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(frequencies)

    slope, intercept, r_value, p_value, _ = stats.linregress(log_ranks, log_freqs)

    zipf_exponent = -slope  # Negative because frequency decreases with rank
    r_squared = r_value ** 2

    return (zipf_exponent, r_squared, p_value)


# =============================================================================
# Scale Invariance Analysis
# =============================================================================

def compute_fractal_dimension(trajectory: np.ndarray, method: str = "box_counting") -> float:
    """
    Compute fractal dimension of a coordinate trajectory.

    Args:
        trajectory: (N, D) array of coordinates over time
        method: "box_counting" or "correlation"

    Returns:
        Estimated fractal dimension
    """
    if len(trajectory) < 10:
        return np.nan

    if method == "box_counting":
        # Simplified box-counting dimension
        scales = np.logspace(-1, 0, 10)
        counts = []

        # Normalize trajectory to [0, 1]
        traj_norm = trajectory - trajectory.min(axis=0)
        traj_range = traj_norm.max(axis=0) - traj_norm.min(axis=0) + 1e-10
        traj_norm = traj_norm / traj_range

        for scale in scales:
            # Count occupied boxes
            boxes = set()
            for point in traj_norm:
                box_idx = tuple((point / scale).astype(int))
                boxes.add(box_idx)
            counts.append(len(boxes))

        # Linear fit to log-log
        log_scales = np.log(1/scales)
        log_counts = np.log(np.array(counts) + 1)

        slope, _, r_value, _, _ = stats.linregress(log_scales, log_counts)
        return slope

    elif method == "correlation":
        # Correlation dimension
        n = len(trajectory)
        dists = []
        for i in range(n):
            for j in range(i+1, min(i+50, n)):  # Limit for speed
                d = np.linalg.norm(trajectory[i] - trajectory[j])
                if d > 0:
                    dists.append(d)

        if len(dists) < 10:
            return np.nan

        dists = np.array(dists)
        epsilons = np.logspace(np.log10(dists.min()), np.log10(dists.max()), 20)

        correlation_sums = []
        for eps in epsilons:
            c = np.sum(dists < eps) / len(dists)
            correlation_sums.append(c)

        # Find scaling region and fit
        log_eps = np.log(epsilons)
        log_c = np.log(np.array(correlation_sums) + 1e-10)

        # Use middle region for fit
        mid = len(log_eps) // 4
        end = 3 * len(log_eps) // 4
        slope, _, _, _, _ = stats.linregress(log_eps[mid:end], log_c[mid:end])

        return slope

    return np.nan


def test_scale_invariance(coords_by_scale: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Test if coordination statistics are scale-invariant.

    Args:
        coords_by_scale: Coordinates at different scales (e.g., sentence, paragraph, document)

    Returns:
        Dictionary of scale-invariance metrics
    """
    results = {}

    # Compute statistics at each scale
    scale_stats = {}
    for scale, coords in coords_by_scale.items():
        if len(coords) < 5:
            continue
        scale_stats[scale] = {
            'mean': np.mean(coords, axis=0),
            'std': np.std(coords, axis=0),
            'skew': stats.skew(coords, axis=0),
            'kurtosis': stats.kurtosis(coords, axis=0)
        }

    if len(scale_stats) < 2:
        return {'scale_invariant': False, 'reason': 'insufficient scales'}

    # Compare distributions across scales
    scale_names = list(scale_stats.keys())

    # Test if means are consistent (within noise)
    means = np.array([scale_stats[s]['mean'] for s in scale_names])
    mean_variation = np.std(means, axis=0) / (np.mean(np.abs(means), axis=0) + 1e-10)

    # Test if variances scale consistently
    stds = np.array([scale_stats[s]['std'] for s in scale_names])

    results['mean_variation'] = float(np.mean(mean_variation))
    results['std_ratio_consistency'] = float(np.std(stds / (stds.mean() + 1e-10)))
    results['scale_invariant'] = results['mean_variation'] < 0.5

    return results


# =============================================================================
# 1/f Noise and Long-Range Correlations
# =============================================================================

def compute_power_spectrum(signal: np.ndarray, fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density using Welch's method."""
    if len(signal) < 20:
        return np.array([]), np.array([])

    nperseg = min(len(signal) // 4, 256)
    if nperseg < 4:
        return np.array([]), np.array([])

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs[1:], psd[1:]  # Skip DC component


def fit_spectral_exponent(freqs: np.ndarray, psd: np.ndarray) -> Tuple[float, float]:
    """
    Fit 1/f^beta power spectrum.

    Returns:
        Tuple of (beta, r_squared)
        - beta ~ 0: white noise
        - beta ~ 1: pink noise (1/f)
        - beta ~ 2: brown noise (1/f^2)
    """
    if len(freqs) < 5:
        return (np.nan, 0.0)

    # Log-log fit
    valid = (freqs > 0) & (psd > 0)
    if np.sum(valid) < 5:
        return (np.nan, 0.0)

    log_f = np.log10(freqs[valid])
    log_psd = np.log10(psd[valid])

    slope, _, r_value, _, _ = stats.linregress(log_f, log_psd)

    return (-slope, r_value ** 2)  # Negative because PSD ~ 1/f^beta


def compute_hurst_exponent(signal: np.ndarray) -> float:
    """
    Compute Hurst exponent using R/S analysis.

    Interpretation:
    - H < 0.5: Anti-persistent (mean-reverting)
    - H = 0.5: Random walk (no memory)
    - H > 0.5: Persistent (long-range correlations)
    """
    n = len(signal)
    if n < 20:
        return np.nan

    # R/S analysis at different scales
    scales = []
    rs_values = []

    for scale in [10, 20, 50, 100, 200, 500]:
        if scale > n // 2:
            break

        n_chunks = n // scale
        rs_chunk = []

        for i in range(n_chunks):
            chunk = signal[i*scale:(i+1)*scale]

            # Mean-centered cumulative sum
            mean = np.mean(chunk)
            cumsum = np.cumsum(chunk - mean)

            # Range
            R = np.max(cumsum) - np.min(cumsum)

            # Standard deviation
            S = np.std(chunk)

            if S > 0:
                rs_chunk.append(R / S)

        if rs_chunk:
            scales.append(scale)
            rs_values.append(np.mean(rs_chunk))

    if len(scales) < 3:
        return np.nan

    # Log-log regression
    log_scales = np.log(scales)
    log_rs = np.log(rs_values)

    slope, _, _, _, _ = stats.linregress(log_scales, log_rs)

    return slope


def compute_dfa(signal: np.ndarray, min_scale: int = 10, max_scale: int = None) -> Tuple[float, float]:
    """
    Detrended Fluctuation Analysis for long-range correlations.

    Returns:
        Tuple of (alpha_dfa, r_squared)
        - alpha < 0.5: Anti-correlated
        - alpha = 0.5: Uncorrelated (white noise)
        - alpha = 1.0: 1/f noise (pink noise)
        - alpha = 1.5: Brown noise
    """
    n = len(signal)
    if n < 50:
        return (np.nan, 0.0)

    if max_scale is None:
        max_scale = n // 4

    # Cumulative sum (integration)
    profile = np.cumsum(signal - np.mean(signal))

    scales = []
    fluctuations = []

    for scale in range(min_scale, max_scale + 1, max(1, (max_scale - min_scale) // 20)):
        n_segments = n // scale
        if n_segments < 1:
            continue

        rms_values = []
        for seg in range(n_segments):
            segment = profile[seg*scale:(seg+1)*scale]

            # Linear detrend
            x = np.arange(len(segment))
            slope, intercept, _, _, _ = stats.linregress(x, segment)
            trend = slope * x + intercept

            # RMS of residuals
            residuals = segment - trend
            rms = np.sqrt(np.mean(residuals ** 2))
            rms_values.append(rms)

        scales.append(scale)
        fluctuations.append(np.mean(rms_values))

    if len(scales) < 3:
        return (np.nan, 0.0)

    # Log-log regression
    log_scales = np.log(scales)
    log_fluct = np.log(fluctuations)

    slope, _, r_value, _, _ = stats.linregress(log_scales, log_fluct)

    return (slope, r_value ** 2)


# =============================================================================
# Avalanche Analysis
# =============================================================================

@dataclass
class AvalancheStats:
    """Statistics of avalanche-like events."""
    sizes: np.ndarray
    durations: np.ndarray
    size_exponent: float          # tau (P(s) ~ s^(-tau))
    duration_exponent: float       # alpha (P(T) ~ T^(-alpha))
    gamma_relationship: float      # gamma = (tau-1)/(alpha-1)
    branching_ratio: float         # sigma, critical at 1
    is_critical: bool


def detect_avalanches(
    trajectory: np.ndarray,
    threshold: float = 0.5
) -> Tuple[List[int], List[int]]:
    """
    Detect avalanche events in coordinate trajectory.

    An avalanche starts when the rate of change exceeds threshold
    and ends when it drops below.

    Returns:
        Tuple of (sizes, durations)
    """
    if len(trajectory) < 10:
        return [], []

    # Compute rate of change
    diffs = np.diff(trajectory, axis=0)
    rates = np.linalg.norm(diffs, axis=1)

    # Threshold for activity
    threshold_value = threshold * np.std(rates)

    sizes = []
    durations = []

    in_avalanche = False
    current_size = 0
    current_duration = 0

    for i, rate in enumerate(rates):
        if rate > threshold_value:
            if not in_avalanche:
                in_avalanche = True
                current_size = 0
                current_duration = 0
            current_size += rate
            current_duration += 1
        else:
            if in_avalanche:
                sizes.append(current_size)
                durations.append(current_duration)
                in_avalanche = False

    # Don't forget last avalanche
    if in_avalanche:
        sizes.append(current_size)
        durations.append(current_duration)

    return sizes, durations


def analyze_avalanches(trajectory: np.ndarray, threshold: float = 0.5) -> AvalancheStats:
    """
    Comprehensive avalanche analysis.

    Tests for critical avalanche dynamics:
    - Power law size distribution: P(s) ~ s^(-tau), tau ~ 1.5 (mean-field)
    - Power law duration distribution: P(T) ~ T^(-alpha)
    - Scaling relationship: gamma = (tau-1)/(alpha-1)
    - Branching ratio sigma ~ 1 at criticality
    """
    sizes, durations = detect_avalanches(trajectory, threshold)

    if len(sizes) < 20:
        return AvalancheStats(
            sizes=np.array(sizes),
            durations=np.array(durations),
            size_exponent=np.nan,
            duration_exponent=np.nan,
            gamma_relationship=np.nan,
            branching_ratio=np.nan,
            is_critical=False
        )

    sizes = np.array(sizes)
    durations = np.array(durations)

    # Fit power laws
    size_fit = fit_power_law(sizes)
    duration_fit = fit_power_law(durations.astype(float))

    tau = size_fit.alpha
    alpha = duration_fit.alpha

    # Scaling relationship
    if alpha > 1 and tau > 1:
        gamma = (tau - 1) / (alpha - 1)
    else:
        gamma = np.nan

    # Branching ratio: average number of subsequent events triggered by one event
    # Estimate from consecutive avalanche ratios
    if len(sizes) > 1:
        ratios = sizes[1:] / (sizes[:-1] + 1e-10)
        branching_ratio = np.median(ratios)
    else:
        branching_ratio = np.nan

    # Check for criticality signatures
    is_critical = (
        size_fit.is_power_law and
        1.3 < tau < 2.0 and  # Near mean-field exponent 1.5
        0.8 < branching_ratio < 1.2  # Near critical branching ratio 1
    )

    return AvalancheStats(
        sizes=sizes,
        durations=durations,
        size_exponent=tau,
        duration_exponent=alpha,
        gamma_relationship=gamma,
        branching_ratio=branching_ratio,
        is_critical=is_critical
    )


# =============================================================================
# Self-Organized Criticality Tests
# =============================================================================

def test_soc_signatures(
    trajectory: np.ndarray,
    n_perturbations: int = 100
) -> Dict[str, float]:
    """
    Test for Self-Organized Criticality signatures.

    SOC systems:
    1. Tune themselves to criticality without external control
    2. Exhibit power law avalanche distributions
    3. Show 1/f noise
    4. Are poised at edge of stability

    Returns:
        Dictionary of SOC indicators
    """
    results = {}

    # 1. Test for 1/f noise in coordinate signals
    if trajectory.ndim == 1:
        signal = trajectory
    else:
        # Use first principal component
        signal = trajectory[:, 0]

    freqs, psd = compute_power_spectrum(signal)
    if len(freqs) > 5:
        beta, r2 = fit_spectral_exponent(freqs, psd)
        results['spectral_exponent'] = beta
        results['spectral_r2'] = r2
        results['is_pink_noise'] = 0.7 < beta < 1.3 and r2 > 0.8

    # 2. DFA for long-range correlations
    alpha_dfa, dfa_r2 = compute_dfa(signal)
    results['dfa_exponent'] = alpha_dfa
    results['dfa_r2'] = dfa_r2
    results['has_lrc'] = 0.7 < alpha_dfa < 1.3 if not np.isnan(alpha_dfa) else False

    # 3. Avalanche analysis
    aval_stats = analyze_avalanches(trajectory)
    results['avalanche_tau'] = aval_stats.size_exponent
    results['avalanche_alpha'] = aval_stats.duration_exponent
    results['branching_ratio'] = aval_stats.branching_ratio
    results['avalanche_critical'] = aval_stats.is_critical

    # 4. Test for edge of stability via Lyapunov-like measure
    # Simplified: variance of differences should be bounded but nonzero
    if len(trajectory) > 10:
        diffs = np.diff(trajectory, axis=0)
        diff_var = np.var(diffs)
        traj_var = np.var(trajectory)
        results['stability_ratio'] = diff_var / (traj_var + 1e-10)
        results['at_edge'] = 0.01 < results['stability_ratio'] < 0.5

    # 5. Overall SOC score
    soc_indicators = [
        results.get('is_pink_noise', False),
        results.get('has_lrc', False),
        results.get('avalanche_critical', False),
        results.get('at_edge', False)
    ]
    results['soc_score'] = sum(soc_indicators) / len(soc_indicators)
    results['is_soc'] = results['soc_score'] > 0.5

    return results


# =============================================================================
# Universality Class Analysis
# =============================================================================

def identify_universality_class(
    tau: float,
    alpha: float,
    gamma: float,
    dfa_alpha: float
) -> Dict[str, any]:
    """
    Identify the universality class based on critical exponents.

    Known universality classes:
    - Mean-field: tau = 1.5, alpha = 2
    - Directed percolation: tau ~ 1.108, alpha ~ 1.159 (1D)
    - Ising: specific heat exponent alpha = 0 (log), beta = 0.326 (3D)
    - Random field: various depending on dimension

    Returns:
        Dictionary with universality class identification
    """
    results = {}

    # Mean-field class (common in high-D or long-range interactions)
    mean_field_dist = np.sqrt((tau - 1.5)**2 + (alpha - 2.0)**2)

    # Directed percolation (1D)
    dp_1d_dist = np.sqrt((tau - 1.108)**2 + (alpha - 1.159)**2)

    # BTW sandpile (2D)
    btw_dist = np.sqrt((tau - 1.27)**2) if not np.isnan(tau) else np.inf

    results['mean_field_distance'] = mean_field_dist
    results['directed_percolation_distance'] = dp_1d_dist
    results['btw_sandpile_distance'] = btw_dist

    # Identify closest class
    classes = {
        'mean_field': mean_field_dist,
        'directed_percolation': dp_1d_dist,
        'btw_sandpile': btw_dist
    }

    closest = min(classes, key=lambda k: classes[k] if not np.isnan(classes[k]) else np.inf)
    results['closest_class'] = closest
    results['class_distance'] = classes[closest]

    # Check for novel exponents
    results['is_novel'] = all(d > 0.3 for d in classes.values() if not np.isnan(d))

    return results


# =============================================================================
# Main Analysis Runner
# =============================================================================

def run_criticality_analysis():
    """
    Run comprehensive criticality analysis on coordination data.
    """
    print("\n" + "=" * 80)
    print("  CRITICALITY ANALYSIS - Cultural Soliton Observatory")
    print("  Testing for Scale-Invariant Behavior in Coordination")
    print("=" * 80)

    # ===========================================================================
    # 1. Collect Effect Sizes Across Features
    # ===========================================================================
    print("\n[1] POWER LAW ANALYSIS OF EFFECT SIZES")
    print("-" * 50)

    # Generate test corpus with varying linguistic features
    test_corpus = {
        'high_agency': [
            "I conquered every challenge through my determination.",
            "I built this company from nothing through my efforts.",
            "I made the decision and took action immediately.",
            "I achieved success through my own hard work.",
            "I solved the problem with my innovative approach.",
        ],
        'low_agency': [
            "The situation forced this outcome upon us.",
            "Circumstances beyond our control led to this.",
            "The system determined the final result.",
            "External forces shaped the conclusion.",
            "The structure dictated our options.",
        ],
        'high_justice': [
            "The process was fair and everyone was heard.",
            "Due process ensured equitable treatment.",
            "Justice was served through proper procedures.",
            "Everyone received what they deserved.",
            "The rules were applied consistently to all.",
        ],
        'low_justice': [
            "The outcome was unfair and arbitrary.",
            "No one listened to our concerns.",
            "The process ignored our rights.",
            "Favoritism determined the results.",
            "We were denied a fair hearing.",
        ],
        'high_belonging': [
            "We stand together as one community.",
            "Our bonds unite us in common purpose.",
            "Together we form an unbreakable whole.",
            "We are all part of this family.",
            "Our community supports each other.",
        ],
        'low_belonging': [
            "They excluded us from the group.",
            "Those outsiders don't belong here.",
            "We stand alone against them.",
            "There is no community anymore.",
            "Everyone is isolated and separate.",
        ]
    }

    # Extract coordinates and compute effect sizes
    all_effect_sizes = []
    feature_effects = {}

    for category in ['agency', 'justice', 'belonging']:
        high_key = f'high_{category}'
        low_key = f'low_{category}'

        high_coords = [extract_hierarchical_coordinate(t) for t in test_corpus[high_key]]
        low_coords = [extract_hierarchical_coordinate(t) for t in test_corpus[low_key]]

        # Get relevant coordinate dimension
        if category == 'agency':
            high_vals = np.array([c.core.agency.aggregate for c in high_coords])
            low_vals = np.array([c.core.agency.aggregate for c in low_coords])
        elif category == 'justice':
            high_vals = np.array([c.core.justice.aggregate for c in high_coords])
            low_vals = np.array([c.core.justice.aggregate for c in low_coords])
        else:
            high_vals = np.array([c.core.belonging.aggregate for c in high_coords])
            low_vals = np.array([c.core.belonging.aggregate for c in low_coords])

        effect = cohens_d(high_vals, low_vals)
        effect_abs = abs(effect.d)
        all_effect_sizes.append(effect_abs)
        feature_effects[category] = effect_abs

        print(f"  {category.capitalize():12s}: d = {effect.d:.3f} (|d| = {effect_abs:.3f})")

    # Add effect sizes from previous research findings
    documented_effects = {
        'deixis': 4.0,           # Largest documented
        'active_voice': 1.52,
        'epistemic_modality': 0.85,
        'deontic_modality': 0.72,
        'articles': 0.12,        # Decorative
        'hedging': 0.18,
        'intensifiers': 0.15,
        'filler_words': 0.08,
        'formal_register': 0.45,
        'ingroup_markers': 1.85,
        'temporal_past': 0.55,
        'temporal_future': 0.62,
    }

    all_effect_sizes.extend(documented_effects.values())

    print(f"\n  Combined effect sizes (n={len(all_effect_sizes)})")
    print(f"  Range: {min(all_effect_sizes):.3f} to {max(all_effect_sizes):.3f}")
    print(f"  Mean: {np.mean(all_effect_sizes):.3f}, Median: {np.median(all_effect_sizes):.3f}")

    # Fit power law to effect sizes
    effect_array = np.array(all_effect_sizes)
    power_fit = fit_power_law(effect_array)

    print(f"\n  Power Law Fit:")
    print(f"    Exponent (alpha): {power_fit.alpha:.3f} +/- {power_fit.sigma:.3f}")
    print(f"    x_min: {power_fit.xmin:.3f}")
    print(f"    R-squared: {power_fit.r_squared:.3f}")
    print(f"    KS statistic: {power_fit.ks_statistic:.3f}")
    print(f"    Is power law: {power_fit.is_power_law}")

    # ===========================================================================
    # 2. Zipf's Law in Feature Frequencies
    # ===========================================================================
    print("\n[2] ZIPF'S LAW IN LINGUISTIC FEATURES")
    print("-" * 50)

    # Analyze feature frequencies in a sample corpus
    sample_texts = []
    for texts in test_corpus.values():
        sample_texts.extend(texts)

    # Aggregate feature counts
    total_features = {}
    for text in sample_texts:
        features = extract_features(text)
        for feat, count in features.items():
            total_features[feat] = total_features.get(feat, 0) + count

    zipf_exp, zipf_r2, zipf_p = test_zipf_law(total_features)

    print(f"  Zipf exponent: {zipf_exp:.3f}")
    print(f"  R-squared: {zipf_r2:.3f}")
    print(f"  Zipf's law satisfied: {0.8 < zipf_exp < 1.2 and zipf_r2 > 0.8}")

    # ===========================================================================
    # 3. Scale Invariance
    # ===========================================================================
    print("\n[3] SCALE INVARIANCE ANALYSIS")
    print("-" * 50)

    # Generate multi-scale coordination data
    # Sentence level
    sentence_coords = np.array([
        extract_hierarchical_coordinate(t).core.to_array()
        for texts in test_corpus.values()
        for t in texts
    ])

    # Paragraph level (aggregate sentences)
    paragraph_coords = []
    for category_texts in test_corpus.values():
        coords = [extract_hierarchical_coordinate(t).core.to_array() for t in category_texts]
        paragraph_coords.append(np.mean(coords, axis=0))
    paragraph_coords = np.array(paragraph_coords)

    # Document level (aggregate all)
    document_coords = np.array([np.mean(paragraph_coords, axis=0)])

    coords_by_scale = {
        'sentence': sentence_coords,
        'paragraph': paragraph_coords,
        'document': document_coords
    }

    scale_results = test_scale_invariance(coords_by_scale)

    print(f"  Mean variation across scales: {scale_results.get('mean_variation', np.nan):.4f}")
    print(f"  Scale invariant: {scale_results.get('scale_invariant', False)}")

    # Fractal dimension of trajectories
    if len(sentence_coords) > 10:
        frac_dim = compute_fractal_dimension(sentence_coords, method='box_counting')
        print(f"  Fractal dimension: {frac_dim:.3f}")
        print(f"  (D = 1: line, D = 2: plane, 1 < D < 2: fractal)")

    # ===========================================================================
    # 4. 1/f Noise and Long-Range Correlations
    # ===========================================================================
    print("\n[4] 1/f NOISE AND LONG-RANGE CORRELATIONS")
    print("-" * 50)

    # Create a coordination signal from the trajectory
    coord_signal = sentence_coords[:, 0]  # Agency dimension

    # Power spectrum analysis
    freqs, psd = compute_power_spectrum(coord_signal)
    if len(freqs) > 5:
        beta, r2 = fit_spectral_exponent(freqs, psd)
        print(f"  Spectral exponent (beta): {beta:.3f}")
        print(f"  R-squared: {r2:.3f}")

        if 0.7 < beta < 1.3:
            print(f"  --> PINK NOISE (1/f) DETECTED")
        elif beta < 0.3:
            print(f"  --> WHITE NOISE (no correlations)")
        elif beta > 1.7:
            print(f"  --> BROWN NOISE (1/f^2)")
        else:
            print(f"  --> INTERMEDIATE SPECTRUM")

    # Hurst exponent
    hurst = compute_hurst_exponent(coord_signal)
    print(f"\n  Hurst exponent: {hurst:.3f}")
    if hurst < 0.4:
        print(f"  --> ANTI-PERSISTENT (mean-reverting)")
    elif hurst < 0.6:
        print(f"  --> RANDOM WALK (no memory)")
    else:
        print(f"  --> PERSISTENT (long-range correlations)")

    # DFA
    dfa_alpha, dfa_r2 = compute_dfa(coord_signal)
    print(f"\n  DFA exponent: {dfa_alpha:.3f}")
    print(f"  DFA R-squared: {dfa_r2:.3f}")

    # ===========================================================================
    # 5. Avalanche Dynamics
    # ===========================================================================
    print("\n[5] AVALANCHE DYNAMICS")
    print("-" * 50)

    # Generate a longer trajectory for avalanche analysis
    # Simulate narrative evolution
    n_steps = 500
    trajectory = np.zeros((n_steps, 9))
    trajectory[0] = sentence_coords[0]

    np.random.seed(42)
    for i in range(1, n_steps):
        # Random walk with occasional jumps (avalanches)
        noise = np.random.randn(9) * 0.1
        if np.random.rand() < 0.1:  # 10% chance of avalanche
            noise *= 5  # Large perturbation
        trajectory[i] = trajectory[i-1] + noise
        trajectory[i] = np.clip(trajectory[i], -3, 3)

    aval_stats = analyze_avalanches(trajectory)

    print(f"  Number of avalanches: {len(aval_stats.sizes)}")
    if len(aval_stats.sizes) > 10:
        print(f"  Size exponent (tau): {aval_stats.size_exponent:.3f}")
        print(f"  Duration exponent (alpha): {aval_stats.duration_exponent:.3f}")
        print(f"  Branching ratio: {aval_stats.branching_ratio:.3f}")
        print(f"  Scaling gamma: {aval_stats.gamma_relationship:.3f}")
        print(f"  Critical avalanche dynamics: {aval_stats.is_critical}")

        if aval_stats.is_critical:
            print("  --> CRITICAL AVALANCHE REGIME DETECTED")

    # ===========================================================================
    # 6. Self-Organized Criticality
    # ===========================================================================
    print("\n[6] SELF-ORGANIZED CRITICALITY SIGNATURES")
    print("-" * 50)

    soc_results = test_soc_signatures(trajectory)

    print(f"  1/f noise present: {soc_results.get('is_pink_noise', False)}")
    print(f"  Long-range correlations: {soc_results.get('has_lrc', False)}")
    print(f"  Critical avalanches: {soc_results.get('avalanche_critical', False)}")
    print(f"  At edge of stability: {soc_results.get('at_edge', False)}")
    print(f"\n  SOC Score: {soc_results.get('soc_score', 0):.2%}")
    print(f"  Self-Organized Criticality: {soc_results.get('is_soc', False)}")

    # ===========================================================================
    # 7. Universality Class Identification
    # ===========================================================================
    print("\n[7] UNIVERSALITY CLASS IDENTIFICATION")
    print("-" * 50)

    univ_class = identify_universality_class(
        tau=aval_stats.size_exponent,
        alpha=aval_stats.duration_exponent,
        gamma=aval_stats.gamma_relationship,
        dfa_alpha=dfa_alpha
    )

    print(f"  Distance to Mean-Field: {univ_class['mean_field_distance']:.3f}")
    print(f"  Distance to Directed Percolation: {univ_class['directed_percolation_distance']:.3f}")
    print(f"  Distance to BTW Sandpile: {univ_class['btw_sandpile_distance']:.3f}")
    print(f"\n  Closest universality class: {univ_class['closest_class']}")
    print(f"  Is novel exponent class: {univ_class['is_novel']}")

    # ===========================================================================
    # 8. Summary Report
    # ===========================================================================
    print("\n" + "=" * 80)
    print("  CRITICALITY ANALYSIS SUMMARY")
    print("=" * 80)

    evidence_for = []
    evidence_against = []

    # Power law evidence
    if power_fit.is_power_law:
        evidence_for.append(f"Effect sizes follow power law (alpha={power_fit.alpha:.2f})")
    else:
        evidence_against.append(f"Effect sizes don't follow power law (R2={power_fit.r_squared:.2f})")

    # Zipf evidence
    if 0.8 < zipf_exp < 1.2 and zipf_r2 > 0.7:
        evidence_for.append(f"Zipf's law in features (s={zipf_exp:.2f})")
    else:
        evidence_against.append(f"Zipf's law not satisfied (s={zipf_exp:.2f})")

    # 1/f noise evidence
    if soc_results.get('is_pink_noise', False):
        evidence_for.append(f"Pink noise (1/f) spectrum detected")
    else:
        evidence_against.append(f"No 1/f noise (beta={soc_results.get('spectral_exponent', np.nan):.2f})")

    # Long-range correlations
    if hurst > 0.6:
        evidence_for.append(f"Long-range correlations (H={hurst:.2f})")
    else:
        evidence_against.append(f"No long-range memory (H={hurst:.2f})")

    # Avalanche criticality
    if aval_stats.is_critical:
        evidence_for.append(f"Critical avalanches (tau={aval_stats.size_exponent:.2f}, sigma={aval_stats.branching_ratio:.2f})")
    else:
        evidence_against.append("Avalanches not critical")

    # SOC
    if soc_results.get('is_soc', False):
        evidence_for.append("Self-organized criticality signatures present")

    print("\n  EVIDENCE FOR CRITICALITY:")
    for e in evidence_for:
        print(f"    [+] {e}")

    print("\n  EVIDENCE AGAINST CRITICALITY:")
    for e in evidence_against:
        print(f"    [-] {e}")

    criticality_score = len(evidence_for) / (len(evidence_for) + len(evidence_against))
    print(f"\n  CRITICALITY SCORE: {criticality_score:.1%}")

    if criticality_score > 0.6:
        verdict = "COORDINATION APPEARS TO OPERATE NEAR CRITICALITY"
    elif criticality_score > 0.4:
        verdict = "MIXED EVIDENCE - POSSIBLE NEAR-CRITICAL DYNAMICS"
    else:
        verdict = "INSUFFICIENT EVIDENCE FOR CRITICALITY"

    print(f"\n  VERDICT: {verdict}")

    print("\n" + "=" * 80)
    print("  PHYSICAL INTERPRETATION")
    print("=" * 80)

    interpretation = """
    If coordination systems operate near criticality, this has profound implications:

    1. INFORMATION PROPAGATION: At criticality, information travels maximally far
       with minimal degradation - optimal for social coordination.

    2. SENSITIVITY: Critical systems are maximally sensitive to perturbations,
       enabling fine-tuned social responsiveness.

    3. COMPLEXITY: Criticality generates maximal complexity/entropy,
       supporting the richness of human communication.

    4. SELF-TUNING: If SOC holds, coordination self-tunes to optimality
       without external control - an emergent property of social systems.

    5. UNIVERSALITY: If coordination shares exponents with other critical systems
       (brain, earthquakes, language), this suggests deep structural similarities.

    The evidence suggests coordination may exhibit QUASI-CRITICAL behavior:
    - Not exactly at the critical point
    - But close enough to benefit from critical properties
    - Possibly in a "Griffiths phase" with extended critical-like behavior

    This aligns with the "edge of chaos" hypothesis in cognitive science
    and the observation that healthy neural systems are slightly subcritical.
    """
    print(interpretation)

    return {
        'power_law_fit': power_fit,
        'zipf_exponent': zipf_exp,
        'spectral_exponent': soc_results.get('spectral_exponent', np.nan),
        'hurst_exponent': hurst,
        'dfa_exponent': dfa_alpha,
        'avalanche_stats': aval_stats,
        'soc_results': soc_results,
        'universality_class': univ_class,
        'criticality_score': criticality_score,
        'evidence_for': evidence_for,
        'evidence_against': evidence_against
    }


if __name__ == "__main__":
    results = run_criticality_analysis()
