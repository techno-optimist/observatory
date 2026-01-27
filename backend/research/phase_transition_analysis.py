"""
Phase Transition Analysis of Coordination Space
================================================

A statistical physics deep-dive into the phase structure of communication
regimes: NATURAL -> TECHNICAL -> COMPRESSED -> OPAQUE

This module investigates:
1. Critical exponents and scaling behavior near phase boundaries
2. Order parameters for coordination (legibility, variance, entropy)
3. Phase diagram mapping with precise boundary locations
4. Hysteresis effects indicating first-order vs continuous transitions
5. Critical point phenomena (diverging fluctuations, critical slowing down)

THEORETICAL FRAMEWORK
=====================

From statistical mechanics, phase transitions occur when:
- An order parameter (legibility) changes discontinuously (first-order)
- Or continuously with diverging susceptibility (second-order/continuous)

For coordination space, we treat:
- Control parameter: compression level (0=natural, 1=maximally compressed)
- Order parameter: legibility score (1=fully interpretable, 0=opaque)
- Temperature analog: variance in coordination patterns

Key predictions:
- First-order transitions: discontinuous jumps, hysteresis, latent "heat"
- Second-order transitions: power-law scaling, critical exponents, universality
- Crossovers: smooth changes without true phase boundaries

Author: Statistical Physics Analysis Module
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Phase Analysis
# =============================================================================

@dataclass
class CriticalExponents:
    """Critical exponents characterizing a phase transition.

    Standard exponents from scaling theory:
    - beta: order parameter ~ |t|^beta (where t = (T-Tc)/Tc)
    - gamma: susceptibility ~ |t|^(-gamma)
    - nu: correlation length ~ |t|^(-nu)
    - alpha: specific heat ~ |t|^(-alpha)
    - delta: critical isotherm M ~ H^(1/delta) at T=Tc

    Universality: Systems in same universality class share exponents
    regardless of microscopic details.
    """
    beta: Optional[float] = None           # Order parameter exponent
    gamma: Optional[float] = None          # Susceptibility exponent
    nu: Optional[float] = None             # Correlation length exponent
    alpha: Optional[float] = None          # Specific heat exponent
    delta: Optional[float] = None          # Critical isotherm exponent

    # Fitting quality
    beta_error: Optional[float] = None
    gamma_error: Optional[float] = None
    r_squared: Optional[float] = None

    # Universality class (if identified)
    universality_class: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beta": self.beta,
            "gamma": self.gamma,
            "nu": self.nu,
            "alpha": self.alpha,
            "delta": self.delta,
            "beta_error": self.beta_error,
            "gamma_error": self.gamma_error,
            "r_squared": self.r_squared,
            "universality_class": self.universality_class,
        }

    def check_scaling_relations(self) -> Dict[str, float]:
        """Check hyperscaling and Rushbrooke relations."""
        relations = {}

        # Rushbrooke relation: alpha + 2*beta + gamma = 2
        if self.alpha and self.beta and self.gamma:
            rushbrooke = self.alpha + 2*self.beta + self.gamma
            relations["rushbrooke_sum"] = rushbrooke
            relations["rushbrooke_deviation"] = abs(rushbrooke - 2.0)

        # Widom relation: gamma = beta * (delta - 1)
        if self.beta and self.gamma and self.delta:
            widom_lhs = self.gamma
            widom_rhs = self.beta * (self.delta - 1)
            relations["widom_deviation"] = abs(widom_lhs - widom_rhs)

        return relations


class TransitionType(Enum):
    """Classification of phase transition types."""
    FIRST_ORDER = "first_order"           # Discontinuous, hysteresis
    SECOND_ORDER = "second_order"         # Continuous, power-law scaling
    CROSSOVER = "crossover"               # Smooth, no true transition
    TRICRITICAL = "tricritical"           # Where first and second order meet
    UNKNOWN = "unknown"


@dataclass
class PhaseBoundary:
    """A boundary between two phases in coordination space."""
    phase_from: str                        # e.g., "NATURAL"
    phase_to: str                          # e.g., "TECHNICAL"
    control_parameter_value: float         # Compression level at boundary
    order_parameter_jump: float            # Discontinuity in legibility
    transition_type: TransitionType
    critical_exponents: Optional[CriticalExponents] = None

    # Boundary width (for crossovers)
    boundary_width: float = 0.0

    # Hysteresis (for first-order)
    hysteresis_width: float = 0.0

    # Confidence in boundary detection
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_from": self.phase_from,
            "phase_to": self.phase_to,
            "control_parameter_value": self.control_parameter_value,
            "order_parameter_jump": self.order_parameter_jump,
            "transition_type": self.transition_type.value,
            "critical_exponents": self.critical_exponents.to_dict() if self.critical_exponents else None,
            "boundary_width": self.boundary_width,
            "hysteresis_width": self.hysteresis_width,
            "confidence": self.confidence,
        }


@dataclass
class PhaseDiagram:
    """Complete phase diagram of coordination space."""
    boundaries: List[PhaseBoundary]
    phases: List[str]                      # ["NATURAL", "TECHNICAL", "COMPRESSED", "OPAQUE"]

    # Critical points (where phase boundaries meet)
    critical_points: List[Dict[str, float]] = field(default_factory=list)

    # Tricritical points (where first-order becomes second-order)
    tricritical_points: List[Dict[str, float]] = field(default_factory=list)

    # Overall characteristics
    mean_field_regime: bool = False        # Does mean-field theory apply?
    universality_evidence: bool = False    # Same exponents across texts?

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boundaries": [b.to_dict() for b in self.boundaries],
            "phases": self.phases,
            "critical_points": self.critical_points,
            "tricritical_points": self.tricritical_points,
            "mean_field_regime": self.mean_field_regime,
            "universality_evidence": self.universality_evidence,
        }


@dataclass
class HysteresisResult:
    """Result of hysteresis analysis for a phase transition."""
    forward_path: np.ndarray               # NATURAL -> OPAQUE trajectory
    backward_path: np.ndarray              # OPAQUE -> NATURAL trajectory
    hysteresis_area: float                 # Enclosed area (first-order indicator)
    max_difference: float                  # Maximum path separation
    transition_forward: float              # Compression level going forward
    transition_backward: float             # Compression level going backward
    hysteresis_width: float                # Difference in transition points
    is_first_order: bool                   # True if significant hysteresis

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hysteresis_area": self.hysteresis_area,
            "max_difference": self.max_difference,
            "transition_forward": self.transition_forward,
            "transition_backward": self.transition_backward,
            "hysteresis_width": self.hysteresis_width,
            "is_first_order": self.is_first_order,
        }


@dataclass
class CriticalSlowingDown:
    """Analysis of critical slowing down near phase transition."""
    relaxation_times: np.ndarray
    control_parameters: np.ndarray
    critical_point: float
    divergence_exponent: float             # Relaxation time ~ |t|^(-z*nu)
    fit_quality: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "critical_point": self.critical_point,
            "divergence_exponent": self.divergence_exponent,
            "fit_quality": self.fit_quality,
        }


# =============================================================================
# Core Analysis Functions
# =============================================================================

def power_law(x: np.ndarray, a: float, exponent: float) -> np.ndarray:
    """Power law function: y = a * x^exponent"""
    return a * np.power(np.abs(x) + 1e-10, exponent)


def sigmoid(x: np.ndarray, x0: float, k: float, L: float, b: float) -> np.ndarray:
    """Sigmoid function for crossover fitting."""
    return L / (1 + np.exp(-k * (x - x0))) + b


def compute_susceptibility(order_parameter: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Compute susceptibility (fluctuation response) of order parameter.

    In statistical mechanics: chi = <M^2> - <M>^2 = variance
    Susceptibility diverges at continuous phase transitions.
    """
    n = len(order_parameter)
    susceptibility = np.zeros(n)

    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2)
        window_data = order_parameter[start:end]
        susceptibility[i] = np.var(window_data) if len(window_data) > 1 else 0.0

    return susceptibility


def compute_correlation_length(coordinates: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Estimate correlation length from coordinate time series.

    Uses autocorrelation decay as proxy for spatial correlation length.
    Correlation length diverges at continuous phase transitions.
    """
    n = len(coordinates)
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(-1, 1)

    correlation_lengths = np.zeros(n)

    for i in range(window, n):
        segment = coordinates[i-window:i]
        if len(segment) > 5:
            # Compute autocorrelation
            centered = segment - np.mean(segment, axis=0)
            autocorr = np.zeros(min(10, len(segment) - 1))

            for lag in range(len(autocorr)):
                if lag == 0:
                    autocorr[lag] = 1.0
                else:
                    c0 = centered[:-lag]
                    c1 = centered[lag:]
                    if len(c0) > 0:
                        numerator = np.sum(c0 * c1)
                        denominator = np.sqrt(np.sum(c0**2) * np.sum(c1**2))
                        autocorr[lag] = numerator / (denominator + 1e-10)

            # Fit exponential decay to get correlation length
            try:
                lags = np.arange(len(autocorr))
                # Correlation length = where autocorr drops to 1/e
                decay_idx = np.argmax(autocorr < 1/np.e)
                if decay_idx == 0:
                    decay_idx = len(autocorr) - 1
                correlation_lengths[i] = decay_idx
            except:
                correlation_lengths[i] = 1.0

    return correlation_lengths


def fit_critical_exponent(
    control_param: np.ndarray,
    order_param: np.ndarray,
    critical_point: float,
    approach: str = "below"  # "below", "above", or "both"
) -> Tuple[float, float, float]:
    """
    Fit critical exponent beta near phase transition.

    Order parameter ~ |t|^beta where t = (control - critical) / critical

    Returns: (beta, error, r_squared)
    """
    # Reduced control parameter
    t = (control_param - critical_point) / (critical_point + 1e-10)

    # Select data based on approach direction
    if approach == "below":
        mask = t < 0
    elif approach == "above":
        mask = t > 0
    else:
        mask = np.ones(len(t), dtype=bool)

    # Filter out points too close to critical point
    mask &= np.abs(t) > 0.01
    mask &= np.abs(t) < 0.5  # Stay in scaling regime

    if np.sum(mask) < 5:
        return None, None, 0.0

    t_fit = np.abs(t[mask])
    y_fit = np.abs(order_param[mask])

    # Remove zeros
    nonzero = y_fit > 1e-10
    if np.sum(nonzero) < 5:
        return None, None, 0.0

    t_fit = t_fit[nonzero]
    y_fit = y_fit[nonzero]

    # Log-log fit for power law
    log_t = np.log(t_fit)
    log_y = np.log(y_fit)

    try:
        # Linear regression in log-log space
        slope, intercept, r_value, _, std_err = stats.linregress(log_t, log_y)
        return slope, std_err, r_value**2
    except:
        return None, None, 0.0


def detect_phase_boundary(
    control_param: np.ndarray,
    order_param: np.ndarray,
    threshold: float = 0.2
) -> List[float]:
    """
    Detect phase boundaries from order parameter vs control parameter curve.

    Looks for:
    1. Discontinuities (first-order)
    2. Inflection points (crossovers)
    3. Derivative maxima (second-order)
    """
    # Sort by control parameter
    sort_idx = np.argsort(control_param)
    x = control_param[sort_idx]
    y = order_param[sort_idx]

    # Compute numerical derivative
    dy = np.gradient(y, x)

    # Find peaks in absolute derivative (rapid changes)
    abs_dy = np.abs(dy)
    peaks, properties = find_peaks(abs_dy, height=threshold, distance=10)

    boundaries = x[peaks].tolist()

    return boundaries


def classify_transition_type(
    control_param: np.ndarray,
    order_param: np.ndarray,
    boundary: float,
    window: float = 0.1
) -> TransitionType:
    """
    Classify a transition as first-order, second-order, or crossover.

    Criteria:
    - First-order: discontinuous jump, derivative has delta function
    - Second-order: continuous but derivative diverges (power law)
    - Crossover: smooth, finite derivative everywhere
    """
    x_range = np.max(control_param) - np.min(control_param)

    # Select data near boundary
    near_boundary = np.abs(control_param - boundary) < window * x_range

    if np.sum(near_boundary) < 10:
        return TransitionType.UNKNOWN

    x_near = control_param[near_boundary]
    y_near = order_param[near_boundary]

    # Sort
    sort_idx = np.argsort(x_near)
    x_near = x_near[sort_idx]
    y_near = y_near[sort_idx]

    # Check for discontinuity
    left_mask = x_near < boundary
    right_mask = x_near > boundary

    if np.sum(left_mask) > 3 and np.sum(right_mask) > 3:
        left_mean = np.mean(y_near[left_mask][-5:])
        right_mean = np.mean(y_near[right_mask][:5])
        jump = abs(right_mean - left_mean)

        # First-order: large discontinuity
        if jump > 0.3:
            return TransitionType.FIRST_ORDER

        # Second-order: continuous but steep
        dy = np.gradient(y_near, x_near)
        max_derivative = np.max(np.abs(dy))

        if max_derivative > 5.0:  # Steep but continuous
            return TransitionType.SECOND_ORDER

    # Default to crossover
    return TransitionType.CROSSOVER


def measure_hysteresis(
    compress_func,
    decompress_func,
    n_points: int = 100
) -> HysteresisResult:
    """
    Measure hysteresis by comparing forward and backward paths.

    compress_func: Maps text to compressed form at given compression level
    decompress_func: Maps compressed text back toward natural

    Hysteresis indicates first-order phase transition.
    """
    compression_levels = np.linspace(0, 1, n_points)

    # Forward path: NATURAL -> OPAQUE
    forward_legibility = np.zeros(n_points)
    for i, level in enumerate(compression_levels):
        # Simulate compression increasing legibility loss
        forward_legibility[i] = compress_func(level)

    # Backward path: OPAQUE -> NATURAL
    backward_legibility = np.zeros(n_points)
    for i, level in enumerate(reversed(compression_levels)):
        # Simulate decompression restoring legibility
        backward_legibility[n_points - 1 - i] = decompress_func(level)

    # Compute hysteresis metrics
    difference = np.abs(forward_legibility - backward_legibility)

    # Area enclosed by hysteresis loop
    hysteresis_area = np.trapezoid(difference, compression_levels)

    # Find transition points
    forward_derivative = np.gradient(forward_legibility, compression_levels)
    backward_derivative = np.gradient(backward_legibility, compression_levels)

    forward_transition = compression_levels[np.argmax(np.abs(forward_derivative))]
    backward_transition = compression_levels[np.argmax(np.abs(backward_derivative))]

    hysteresis_width = abs(forward_transition - backward_transition)

    return HysteresisResult(
        forward_path=forward_legibility,
        backward_path=backward_legibility,
        hysteresis_area=hysteresis_area,
        max_difference=np.max(difference),
        transition_forward=forward_transition,
        transition_backward=backward_transition,
        hysteresis_width=hysteresis_width,
        is_first_order=hysteresis_width > 0.05  # Threshold for significant hysteresis
    )


def compute_specific_heat(
    control_param: np.ndarray,
    order_param: np.ndarray,
    window: int = 10
) -> np.ndarray:
    """
    Compute specific heat analog from order parameter fluctuations.

    In statistical mechanics: C = d<E>/dT ~ d<M>/d(control)
    Specific heat can diverge at second-order transitions.
    """
    # Sort by control parameter
    sort_idx = np.argsort(control_param)
    x = control_param[sort_idx]
    y = order_param[sort_idx]

    # Compute derivative
    dy_dx = np.gradient(y, x)

    # Smooth with rolling window
    n = len(dy_dx)
    specific_heat = np.zeros(n)

    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2)
        specific_heat[i] = np.mean(np.abs(dy_dx[start:end]))

    return specific_heat


def analyze_critical_slowing_down(
    control_param: np.ndarray,
    time_series_generator,  # Function that generates time series at each control value
    critical_point: float,
    n_samples: int = 100
) -> CriticalSlowingDown:
    """
    Measure critical slowing down near phase transition.

    Relaxation time tau ~ |t|^(-z*nu) diverges at critical point.
    This is a signature of continuous phase transitions.
    """
    # Select control parameters near critical point
    near_critical = np.abs(control_param - critical_point) < 0.3
    x_near = control_param[near_critical]

    relaxation_times = np.zeros(len(x_near))

    for i, x in enumerate(x_near):
        # Generate time series at this control parameter
        ts = time_series_generator(x, n_samples)

        # Compute autocorrelation time as proxy for relaxation time
        if len(ts) > 10:
            centered = ts - np.mean(ts)
            autocorr = np.correlate(centered, centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]

            # Find first zero crossing
            zero_cross = np.argmax(autocorr < 0)
            if zero_cross == 0:
                zero_cross = len(autocorr) // 2

            relaxation_times[i] = zero_cross

    # Fit power law divergence
    t = np.abs(x_near - critical_point) / (critical_point + 1e-10)

    mask = (t > 0.01) & (relaxation_times > 0)
    if np.sum(mask) < 5:
        return CriticalSlowingDown(
            relaxation_times=relaxation_times,
            control_parameters=x_near,
            critical_point=critical_point,
            divergence_exponent=0.0,
            fit_quality=0.0
        )

    try:
        log_t = np.log(t[mask])
        log_tau = np.log(relaxation_times[mask])
        slope, _, r_value, _, _ = stats.linregress(log_t, log_tau)

        return CriticalSlowingDown(
            relaxation_times=relaxation_times,
            control_parameters=x_near,
            critical_point=critical_point,
            divergence_exponent=-slope,  # Negative because tau diverges as t->0
            fit_quality=r_value**2
        )
    except:
        return CriticalSlowingDown(
            relaxation_times=relaxation_times,
            control_parameters=x_near,
            critical_point=critical_point,
            divergence_exponent=0.0,
            fit_quality=0.0
        )


# =============================================================================
# Phase Diagram Construction
# =============================================================================

def construct_phase_diagram(
    compression_levels: np.ndarray,
    legibility_scores: np.ndarray,
    variance_data: Optional[np.ndarray] = None,
    coordinate_trajectories: Optional[np.ndarray] = None
) -> PhaseDiagram:
    """
    Construct full phase diagram from experimental data.

    This is the main entry point for phase transition analysis.

    Args:
        compression_levels: Control parameter values (0-1)
        legibility_scores: Order parameter values (0-1)
        variance_data: Optional variance measurements
        coordinate_trajectories: Optional manifold coordinates

    Returns:
        Complete PhaseDiagram with boundaries, exponents, and critical points
    """
    # Detect phase boundaries
    boundary_locations = detect_phase_boundary(
        compression_levels, legibility_scores, threshold=0.2
    )

    # Standard phase boundaries from legibility thresholds
    # NATURAL: legibility > 0.7
    # TECHNICAL: 0.5 < legibility <= 0.7
    # COMPRESSED: 0.3 < legibility <= 0.5
    # OPAQUE: legibility <= 0.3

    expected_transitions = {
        ("NATURAL", "TECHNICAL"): 0.7,
        ("TECHNICAL", "COMPRESSED"): 0.5,
        ("COMPRESSED", "OPAQUE"): 0.3,
    }

    boundaries = []
    phases = ["NATURAL", "TECHNICAL", "COMPRESSED", "OPAQUE"]

    # Match detected boundaries to expected transitions
    for (phase_from, phase_to), expected_legibility in expected_transitions.items():
        # Find compression level where legibility crosses threshold
        sort_idx = np.argsort(compression_levels)
        x_sorted = compression_levels[sort_idx]
        y_sorted = legibility_scores[sort_idx]

        # Find crossing point
        crossing_idx = np.argmax(y_sorted < expected_legibility)
        if crossing_idx > 0:
            boundary_compression = x_sorted[crossing_idx]
        else:
            # Estimate from interpolation
            above_threshold = y_sorted >= expected_legibility
            if np.any(above_threshold) and np.any(~above_threshold):
                last_above = np.max(np.where(above_threshold)[0])
                if last_above < len(x_sorted) - 1:
                    boundary_compression = (x_sorted[last_above] + x_sorted[last_above + 1]) / 2
                else:
                    boundary_compression = x_sorted[-1]
            else:
                continue

        # Classify transition type
        trans_type = classify_transition_type(
            compression_levels, legibility_scores, boundary_compression
        )

        # Compute jump magnitude
        near_boundary = np.abs(compression_levels - boundary_compression) < 0.1
        if np.sum(near_boundary) > 5:
            x_near = compression_levels[near_boundary]
            y_near = legibility_scores[near_boundary]
            left_mean = np.mean(y_near[x_near < boundary_compression])
            right_mean = np.mean(y_near[x_near > boundary_compression])
            jump = abs(left_mean - right_mean) if not np.isnan(left_mean) and not np.isnan(right_mean) else 0.0
        else:
            jump = 0.0

        # Fit critical exponents if second-order
        exponents = None
        if trans_type == TransitionType.SECOND_ORDER:
            beta, beta_err, r2 = fit_critical_exponent(
                compression_levels, legibility_scores, boundary_compression
            )

            # Also fit susceptibility exponent
            if variance_data is not None:
                gamma, gamma_err, _ = fit_critical_exponent(
                    compression_levels, variance_data, boundary_compression
                )
            else:
                gamma, gamma_err = None, None

            exponents = CriticalExponents(
                beta=beta,
                gamma=gamma,
                beta_error=beta_err,
                gamma_error=gamma_err,
                r_squared=r2
            )

            # Check for known universality classes
            if beta is not None:
                if 0.3 < beta < 0.4:
                    exponents.universality_class = "3D_Ising"
                elif 0.1 < beta < 0.2:
                    exponents.universality_class = "2D_Ising"
                elif 0.45 < beta < 0.55:
                    exponents.universality_class = "mean_field"

        boundary = PhaseBoundary(
            phase_from=phase_from,
            phase_to=phase_to,
            control_parameter_value=boundary_compression,
            order_parameter_jump=jump,
            transition_type=trans_type,
            critical_exponents=exponents,
            confidence=0.8 if trans_type != TransitionType.UNKNOWN else 0.3
        )

        boundaries.append(boundary)

    # Look for critical points (where multiple phases meet)
    critical_points = []

    # Check if there's a tricritical point where first-order meets second-order
    tricritical_points = []
    first_order_boundaries = [b for b in boundaries if b.transition_type == TransitionType.FIRST_ORDER]
    second_order_boundaries = [b for b in boundaries if b.transition_type == TransitionType.SECOND_ORDER]

    if first_order_boundaries and second_order_boundaries:
        # Potential tricritical point
        tcp = {
            "compression": (first_order_boundaries[0].control_parameter_value +
                          second_order_boundaries[0].control_parameter_value) / 2,
            "legibility": 0.5,  # Approximate
            "type": "tricritical"
        }
        tricritical_points.append(tcp)

    # Check for universality
    exponent_values = [b.critical_exponents.beta for b in boundaries
                      if b.critical_exponents and b.critical_exponents.beta]
    universality_evidence = len(exponent_values) >= 2 and np.std(exponent_values) < 0.1

    return PhaseDiagram(
        boundaries=boundaries,
        phases=phases,
        critical_points=critical_points,
        tricritical_points=tricritical_points,
        universality_evidence=universality_evidence
    )


# =============================================================================
# Experimental Analysis Functions
# =============================================================================

def run_legibility_compression_experiment(
    text_samples: List[str],
    compression_simulator,  # Function: (text, level) -> compressed_text
    legibility_scorer,      # Function: text -> float
    n_levels: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run systematic experiment measuring legibility vs compression.

    Returns: (compression_levels, mean_legibility, std_legibility)
    """
    compression_levels = np.linspace(0, 1, n_levels)

    mean_legibility = np.zeros(n_levels)
    std_legibility = np.zeros(n_levels)

    for i, level in enumerate(compression_levels):
        legibilities = []
        for text in text_samples:
            compressed = compression_simulator(text, level)
            leg = legibility_scorer(compressed)
            legibilities.append(leg)

        mean_legibility[i] = np.mean(legibilities)
        std_legibility[i] = np.std(legibilities)

    return compression_levels, mean_legibility, std_legibility


def analyze_variance_near_transition(
    compression_levels: np.ndarray,
    legibility_scores: np.ndarray,
    transition_point: float,
    window_fraction: float = 0.2
) -> Dict[str, Any]:
    """
    Analyze variance behavior near a phase transition.

    For second-order transitions, variance should peak at the transition.
    This is analogous to susceptibility divergence in thermal physics.
    """
    x_range = np.max(compression_levels) - np.min(compression_levels)
    window = window_fraction * x_range

    # Divide data into bins approaching transition
    n_bins = 20
    bins = np.linspace(transition_point - window, transition_point + window, n_bins)

    bin_variances = []
    bin_centers = []

    for j in range(len(bins) - 1):
        mask = (compression_levels >= bins[j]) & (compression_levels < bins[j+1])
        if np.sum(mask) >= 3:
            bin_variances.append(np.var(legibility_scores[mask]))
            bin_centers.append((bins[j] + bins[j+1]) / 2)

    bin_centers = np.array(bin_centers)
    bin_variances = np.array(bin_variances)

    if len(bin_variances) > 0:
        # Check if variance peaks at transition
        peak_idx = np.argmax(bin_variances)
        peak_location = bin_centers[peak_idx] if len(bin_centers) > 0 else transition_point

        # Compute variance ratio (peak / baseline)
        if len(bin_variances) > 2:
            baseline = (bin_variances[0] + bin_variances[-1]) / 2
            peak_ratio = bin_variances[peak_idx] / (baseline + 1e-10)
        else:
            peak_ratio = 1.0
    else:
        peak_location = transition_point
        peak_ratio = 1.0

    return {
        "transition_point": transition_point,
        "peak_variance_location": float(peak_location),
        "peak_variance_ratio": float(peak_ratio),
        "variance_diverges": peak_ratio > 3.0,  # Threshold for divergence
        "bin_centers": bin_centers.tolist() if len(bin_centers) > 0 else [],
        "bin_variances": bin_variances.tolist() if len(bin_variances) > 0 else [],
    }


def test_universality(
    text_collection_a: List[str],
    text_collection_b: List[str],
    compression_simulator,
    legibility_scorer,
    n_levels: int = 30
) -> Dict[str, Any]:
    """
    Test universality hypothesis: do different text types have same critical exponents?

    Universality is a key prediction of renormalization group theory.
    Systems in the same universality class share critical exponents
    regardless of microscopic details.
    """
    # Run experiments on both text collections
    levels_a, leg_a, _ = run_legibility_compression_experiment(
        text_collection_a, compression_simulator, legibility_scorer, n_levels
    )
    levels_b, leg_b, _ = run_legibility_compression_experiment(
        text_collection_b, compression_simulator, legibility_scorer, n_levels
    )

    # Construct phase diagrams
    diagram_a = construct_phase_diagram(levels_a, leg_a)
    diagram_b = construct_phase_diagram(levels_b, leg_b)

    # Compare critical exponents
    exponents_a = [b.critical_exponents for b in diagram_a.boundaries
                   if b.critical_exponents and b.critical_exponents.beta]
    exponents_b = [b.critical_exponents for b in diagram_b.boundaries
                   if b.critical_exponents and b.critical_exponents.beta]

    if exponents_a and exponents_b:
        beta_a = np.mean([e.beta for e in exponents_a])
        beta_b = np.mean([e.beta for e in exponents_b])

        # Statistical test for equality
        beta_diff = abs(beta_a - beta_b)
        pooled_error = np.sqrt(
            (exponents_a[0].beta_error or 0.1)**2 +
            (exponents_b[0].beta_error or 0.1)**2
        )

        universality_test = beta_diff / (pooled_error + 1e-10)

        return {
            "beta_collection_a": beta_a,
            "beta_collection_b": beta_b,
            "beta_difference": beta_diff,
            "pooled_error": pooled_error,
            "z_score": universality_test,
            "universality_supported": universality_test < 2.0,  # 95% confidence
            "shared_universality_class": (
                exponents_a[0].universality_class
                if exponents_a[0].universality_class == exponents_b[0].universality_class
                else "different"
            ),
        }

    return {
        "error": "Insufficient critical exponent data",
        "universality_supported": False,
    }


# =============================================================================
# Synthetic Data Generation for Testing
# =============================================================================

def generate_synthetic_legibility_data(
    n_points: int = 1000,
    noise_level: float = 0.05,
    transition_type: str = "mixed"  # "first_order", "second_order", "crossover", "mixed"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic legibility vs compression data for testing.

    Models realistic phase transition behavior including:
    - Sharp drops at phase boundaries
    - Variance peaks near transitions
    - Different transition types
    """
    compression = np.linspace(0, 1, n_points)

    if transition_type == "first_order":
        # Step function with some smoothing
        legibility = np.piecewise(
            compression,
            [compression < 0.3,
             (compression >= 0.3) & (compression < 0.5),
             (compression >= 0.5) & (compression < 0.7),
             compression >= 0.7],
            [0.9, 0.65, 0.4, 0.15]
        )
        # Add sharp transitions
        legibility = legibility + noise_level * np.random.randn(n_points)

    elif transition_type == "second_order":
        # Power law approach to transitions
        # L(c) = 1 - c^beta
        beta = 0.35  # 3D Ising-like
        legibility = 1 - compression**beta
        legibility = np.clip(legibility + noise_level * np.random.randn(n_points), 0, 1)

    elif transition_type == "crossover":
        # Smooth sigmoid transitions
        legibility = sigmoid(compression, 0.3, 15, -0.3, 0.9)
        legibility += sigmoid(compression, 0.5, 12, -0.25, 0)
        legibility += sigmoid(compression, 0.7, 10, -0.2, 0)
        legibility = np.clip(legibility + noise_level * np.random.randn(n_points), 0, 1)

    else:  # "mixed" - realistic case
        # NATURAL -> TECHNICAL: crossover around 0.3
        # TECHNICAL -> COMPRESSED: second-order around 0.5
        # COMPRESSED -> OPAQUE: first-order around 0.7

        base = np.ones(n_points) * 0.9

        # Crossover at 0.3
        base -= 0.2 * (1 / (1 + np.exp(-30 * (compression - 0.3))))

        # Second-order at 0.5
        mask = compression > 0.4
        base[mask] -= 0.2 * ((compression[mask] - 0.4) / 0.1)**0.35
        base = np.clip(base, 0.3, 0.9)

        # First-order at 0.7
        mask = compression > 0.65
        base[mask] = np.where(
            compression[mask] < 0.75,
            base[mask] - 0.15 * (compression[mask] - 0.65) / 0.1,
            0.15
        )

        legibility = np.clip(base + noise_level * np.random.randn(n_points), 0, 1)

    return compression, legibility


def simulate_compression_process(text: str, level: float) -> str:
    """
    Simulate text compression at various levels.

    level = 0: No compression (natural)
    level = 1: Maximum compression (opaque)

    Compression operations:
    - Remove stopwords
    - Abbreviate words
    - Remove vowels
    - Replace with symbols
    """
    if level < 0.2:
        return text

    words = text.split()

    if level < 0.4:
        # Technical: remove some articles and connectors
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been'}
        words = [w for w in words if w.lower() not in stopwords or np.random.random() > level]

    elif level < 0.6:
        # Compressed: abbreviate and remove more
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'and', 'or', 'but', 'so', 'yet', 'for'}
        words = [w for w in words if w.lower() not in stopwords]
        # Abbreviate longer words
        words = [w[:3] if len(w) > 5 and np.random.random() < level else w for w in words]

    elif level < 0.8:
        # More compressed: remove vowels
        def remove_vowels(word):
            if len(word) > 3:
                return ''.join(c for c in word if c.lower() not in 'aeiou')
            return word
        words = [remove_vowels(w) if np.random.random() < level - 0.4 else w for w in words]
        words = [w for w in words if len(w) > 0]

    else:
        # Opaque: replace with symbols
        def to_symbol(word):
            return ''.join(chr(ord('A') + (ord(c) - ord('a')) % 26) if c.isalpha() else c
                         for c in word[:2])
        words = [to_symbol(w) if np.random.random() < level - 0.6 else w for w in words]

    return ' '.join(words)


def simple_legibility_score(text: str) -> float:
    """
    Simple legibility scoring for testing.

    Based on:
    - Word recognition (are words in dictionary?)
    - Average word length
    - Symbol ratio
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    words = text.split()
    if not words:
        return 0.0

    # Common English words
    common_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'is', 'was', 'are', 'were', 'been', 'being', 'have', 'has', 'had',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'very', 'really', 'just', 'also', 'so', 'now', 'then', 'there', 'here',
    }

    # Word recognition score
    recognized = sum(1 for w in words if w.lower() in common_words)
    recognition_score = recognized / len(words)

    # Average word length (optimal around 4-5)
    avg_length = sum(len(w) for w in words) / len(words)
    length_score = max(0, 1 - abs(avg_length - 4.5) / 5)

    # Letter ratio (vs symbols and numbers)
    total_chars = len(text.replace(' ', ''))
    if total_chars > 0:
        letter_chars = sum(1 for c in text if c.isalpha())
        letter_ratio = letter_chars / total_chars
    else:
        letter_ratio = 0.0

    # Combine scores
    legibility = 0.4 * recognition_score + 0.3 * length_score + 0.3 * letter_ratio

    return float(np.clip(legibility, 0, 1))


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_full_phase_analysis(
    text_samples: Optional[List[str]] = None,
    use_synthetic: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete phase transition analysis pipeline.

    This is the main entry point for comprehensive analysis.

    Returns detailed results including:
    - Phase diagram with boundaries and exponents
    - Hysteresis measurements
    - Variance analysis
    - Universality tests
    """
    results = {}

    if use_synthetic or text_samples is None:
        # Generate synthetic data
        if verbose:
            print("Generating synthetic phase transition data...")

        compression, legibility = generate_synthetic_legibility_data(
            n_points=500,
            noise_level=0.03,
            transition_type="mixed"
        )
        results["data_source"] = "synthetic"
    else:
        # Use real text samples
        if verbose:
            print(f"Analyzing {len(text_samples)} text samples...")

        compression, legibility, _ = run_legibility_compression_experiment(
            text_samples,
            simulate_compression_process,
            simple_legibility_score,
            n_levels=50
        )
        results["data_source"] = "experimental"

    # 1. Construct Phase Diagram
    if verbose:
        print("\n1. Constructing phase diagram...")

    variance = compute_susceptibility(legibility, window=20)
    diagram = construct_phase_diagram(compression, legibility, variance)
    results["phase_diagram"] = diagram.to_dict()

    if verbose:
        print(f"   Found {len(diagram.boundaries)} phase boundaries:")
        for b in diagram.boundaries:
            print(f"     {b.phase_from} -> {b.phase_to} at compression={b.control_parameter_value:.3f}")
            print(f"       Type: {b.transition_type.value}, Jump: {b.order_parameter_jump:.3f}")
            if b.critical_exponents and b.critical_exponents.beta:
                print(f"       Beta exponent: {b.critical_exponents.beta:.3f}")

    # 2. Analyze variance near each transition
    if verbose:
        print("\n2. Analyzing variance near phase boundaries...")

    variance_results = {}
    for boundary in diagram.boundaries:
        var_analysis = analyze_variance_near_transition(
            compression, legibility, boundary.control_parameter_value
        )
        variance_results[f"{boundary.phase_from}_{boundary.phase_to}"] = var_analysis

        if verbose:
            print(f"   {boundary.phase_from} -> {boundary.phase_to}:")
            print(f"     Variance peak ratio: {var_analysis['peak_variance_ratio']:.2f}x")
            print(f"     Divergence detected: {var_analysis['variance_diverges']}")

    results["variance_analysis"] = variance_results

    # 3. Estimate critical exponents
    if verbose:
        print("\n3. Fitting critical exponents...")

    exponent_results = {}
    for boundary in diagram.boundaries:
        if boundary.transition_type == TransitionType.SECOND_ORDER:
            beta, err, r2 = fit_critical_exponent(
                compression, legibility, boundary.control_parameter_value
            )

            # Also compute gamma from susceptibility
            gamma, gamma_err, gamma_r2 = fit_critical_exponent(
                compression, variance, boundary.control_parameter_value
            )

            exponent_results[f"{boundary.phase_from}_{boundary.phase_to}"] = {
                "beta": beta,
                "beta_error": err,
                "beta_r_squared": r2,
                "gamma": gamma,
                "gamma_error": gamma_err,
                "gamma_r_squared": gamma_r2,
            }

            if verbose and beta:
                print(f"   {boundary.phase_from} -> {boundary.phase_to}:")
                print(f"     Beta = {beta:.3f} +/- {err or 0:.3f} (R^2 = {r2:.3f})")
                if gamma:
                    print(f"     Gamma = {gamma:.3f} +/- {gamma_err or 0:.3f}")

    results["critical_exponents"] = exponent_results

    # 4. Hysteresis analysis (simulated)
    if verbose:
        print("\n4. Testing for hysteresis (first-order transitions)...")

    def forward_compress(level):
        # Simulate forward compression path
        base = 1 - 0.3 * level - 0.4 * max(0, level - 0.5) - 0.2 * max(0, level - 0.7)
        return max(0.1, base + 0.02 * np.random.randn())

    def backward_decompress(level):
        # Backward path with hysteresis
        base = 0.2 + 0.3 * (1 - level) + 0.3 * max(0, 0.6 - level) + 0.2 * max(0, 0.4 - level)
        return min(0.9, base + 0.02 * np.random.randn())

    hysteresis = measure_hysteresis(forward_compress, backward_decompress, n_points=100)
    results["hysteresis"] = hysteresis.to_dict()

    if verbose:
        print(f"   Hysteresis area: {hysteresis.hysteresis_area:.4f}")
        print(f"   Forward transition at: {hysteresis.transition_forward:.3f}")
        print(f"   Backward transition at: {hysteresis.transition_backward:.3f}")
        print(f"   Hysteresis width: {hysteresis.hysteresis_width:.3f}")
        print(f"   First-order transition: {hysteresis.is_first_order}")

    # 5. Compute specific heat analog
    if verbose:
        print("\n5. Computing specific heat (derivative of order parameter)...")

    specific_heat = compute_specific_heat(compression, legibility, window=15)

    # Find peaks in specific heat
    peaks, _ = find_peaks(specific_heat, height=np.mean(specific_heat))

    results["specific_heat"] = {
        "peak_locations": compression[peaks].tolist() if len(peaks) > 0 else [],
        "peak_values": specific_heat[peaks].tolist() if len(peaks) > 0 else [],
        "mean_specific_heat": float(np.mean(specific_heat)),
        "max_specific_heat": float(np.max(specific_heat)),
    }

    if verbose:
        print(f"   Specific heat peaks at compression: {compression[peaks]}")
        print(f"   Maximum specific heat: {np.max(specific_heat):.3f}")

    # 6. Summary statistics
    if verbose:
        print("\n" + "="*60)
        print("PHASE TRANSITION ANALYSIS SUMMARY")
        print("="*60)

    summary = {
        "total_boundaries": len(diagram.boundaries),
        "first_order_transitions": sum(1 for b in diagram.boundaries
                                       if b.transition_type == TransitionType.FIRST_ORDER),
        "second_order_transitions": sum(1 for b in diagram.boundaries
                                        if b.transition_type == TransitionType.SECOND_ORDER),
        "crossovers": sum(1 for b in diagram.boundaries
                         if b.transition_type == TransitionType.CROSSOVER),
        "universality_evidence": diagram.universality_evidence,
        "hysteresis_detected": hysteresis.is_first_order,
        "variance_divergence_count": sum(1 for v in variance_results.values()
                                         if v.get("variance_diverges", False)),
    }

    results["summary"] = summary

    if verbose:
        print(f"\nPhase Boundaries: {summary['total_boundaries']}")
        print(f"  - First Order: {summary['first_order_transitions']}")
        print(f"  - Second Order: {summary['second_order_transitions']}")
        print(f"  - Crossovers: {summary['crossovers']}")
        print(f"\nUniversality Evidence: {summary['universality_evidence']}")
        print(f"Hysteresis Detected: {summary['hysteresis_detected']}")
        print(f"Variance Divergences: {summary['variance_divergence_count']}")

    return results


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Phase Transition Analysis of Coordination Space"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=True,
        help="Use synthetic data for analysis"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File with text samples (one per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    text_samples = None
    if args.file:
        with open(args.file, 'r') as f:
            text_samples = [line.strip() for line in f if line.strip()]

    results = run_full_phase_analysis(
        text_samples=text_samples,
        use_synthetic=args.synthetic or text_samples is None,
        verbose=not args.quiet
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
