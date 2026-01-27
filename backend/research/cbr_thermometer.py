"""
CBR Thermometer - Coordination Background Radiation monitoring.

Operational tool for measuring coordination signal strength in real-time.
The "heartbeat" for coordination content analysis.

Key metrics:
- T_CBR: Centroid norm (baseline ~2.73, undetected ~3.0)
- Phase position: NATURAL / TECHNICAL / COMPRESSED / OPAQUE
- Signal strength: Δ = baseline - observed (higher = more coordination)
- Variance: Detection of ossification precursors
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
from collections import deque
import numpy as np

from .hierarchical_coordinates import (
    extract_hierarchical_coordinate,
    HierarchicalCoordinate,
)
from .legibility_analyzer import LegibilityRegime, compute_legibility_sync
from .opaque_detector import OpaqueDetector


class AlertLevel(Enum):
    """Alert severity levels."""
    NONE = "none"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CBRReading:
    """Single CBR measurement."""
    temperature: float  # ||x|| for this sample
    signal_strength: float  # 3.0 - temperature (higher = more signal)
    core_norm: float  # 9D core norm
    modifier_norm: float  # 9D modifier norm
    phase: LegibilityRegime
    legibility: float
    coordinate: HierarchicalCoordinate

    # 3-bit kernel state
    agency_bit: bool  # Any agency detected?
    justice_bit: bool  # Any justice detected?
    belonging_bit: bool  # Any belonging detected?
    kernel_state: int  # 0-7 encoding

    @property
    def kernel_label(self) -> str:
        """Human-readable kernel state label."""
        labels = [
            "ANOMIE",       # 000
            "OPPRESSION",   # 001
            "NEGLECT",      # 010
            "DEPENDENCE",   # 011
            "ALIENATION",   # 100
            "EXPLOITATION", # 101
            "AUTONOMY",     # 110
            "COORDINATION", # 111
        ]
        return labels[self.kernel_state]


@dataclass
class CBRAlert:
    """Alert from CBR monitoring."""
    level: AlertLevel
    message: str
    metric: str
    value: float
    threshold: float


@dataclass
class CBRThermometerState:
    """Running state of the CBR thermometer."""
    # Rolling window of readings
    window_size: int = 50
    readings: deque = field(default_factory=lambda: deque(maxlen=50))

    # Baseline statistics (computed from initial readings)
    baseline_temperature: float = 3.0  # √9 undetected
    baseline_variance: float = 0.1

    # Alert thresholds
    variance_collapse_threshold: float = 0.7  # Alert if variance drops 70%
    phase_transition_threshold: float = 0.3   # Alert if legibility jumps

    # Tracking
    total_readings: int = 0
    alerts_issued: int = 0


class CBRThermometer:
    """
    Real-time CBR monitoring instrument.

    Usage:
        thermometer = CBRThermometer()
        for message in stream:
            reading = thermometer.measure(message)
            print(f"T={reading.temperature:.2f} Phase={reading.phase.value}")

            alerts = thermometer.check_alerts()
            for alert in alerts:
                print(f"ALERT: {alert.message}")
    """

    # Phase boundaries (calibrated via grid search optimization - 75% accuracy)
    # Note: These are legibility score thresholds, not compression levels
    PHASE_BOUNDARIES = {
        'c1': 0.465,  # NATURAL -> TECHNICAL
        'c2': 0.440,  # TECHNICAL -> COMPRESSED
        'c3': 0.410,  # COMPRESSED -> OPAQUE
    }

    def __init__(self, window_size: int = 50):
        self.state = CBRThermometerState(window_size=window_size)
        self.state.readings = deque(maxlen=window_size)
        self.opaque_detector = OpaqueDetector(threshold=0.55)

    def measure(self, text: str) -> CBRReading:
        """Take a single CBR measurement."""
        # Extract hierarchical coordinate
        coord = extract_hierarchical_coordinate(text)
        core = coord.core
        mods = coord.modifiers

        # Build vectors
        core_vec = np.array([
            core.agency.self_agency, core.agency.other_agency, core.agency.system_agency,
            core.justice.procedural, core.justice.distributive, core.justice.interactional,
            core.belonging.ingroup, core.belonging.outgroup, core.belonging.universal
        ])

        mod_vec = np.array([
            mods.epistemic.certainty, mods.epistemic.evidentiality, mods.epistemic.commitment,
            mods.temporal.focus, mods.temporal.scope,
            mods.social.power_differential, mods.social.social_distance,
            mods.emotional.arousal, mods.emotional.valence
        ])

        # Compute norms
        core_norm = float(np.linalg.norm(core_vec))
        mod_norm = float(np.linalg.norm(mod_vec))
        full_vec = np.concatenate([core_vec, mod_vec])
        temperature = float(np.linalg.norm(full_vec))

        # Signal strength (higher = more coordination detected)
        signal_strength = 3.0 - temperature

        # Phase classification (two-stage: OPAQUE detection first, then legibility)
        legibility_result = compute_legibility_sync(text)
        legibility = legibility_result['score'] if isinstance(legibility_result, dict) else legibility_result
        phase = self._classify_phase(legibility, text)

        # 3-bit kernel
        agency_detected = any(v > -0.5 for v in [
            core.agency.self_agency, core.agency.other_agency, core.agency.system_agency
        ])
        justice_detected = any(v > -0.5 for v in [
            core.justice.procedural, core.justice.distributive, core.justice.interactional
        ])
        belonging_detected = any(v > -0.5 for v in [
            core.belonging.ingroup, core.belonging.outgroup, core.belonging.universal
        ])

        kernel_state = (
            (1 if agency_detected else 0) << 2 |
            (1 if justice_detected else 0) << 1 |
            (1 if belonging_detected else 0)
        )

        reading = CBRReading(
            temperature=temperature,
            signal_strength=signal_strength,
            core_norm=core_norm,
            modifier_norm=mod_norm,
            phase=phase,
            legibility=legibility,
            coordinate=coord,
            agency_bit=agency_detected,
            justice_bit=justice_detected,
            belonging_bit=belonging_detected,
            kernel_state=kernel_state,
        )

        # Update state
        self.state.readings.append(reading)
        self.state.total_readings += 1

        return reading

    def _classify_phase(self, legibility: float, text: str = "") -> LegibilityRegime:
        """Classify into phase regime using two-stage detection.

        Stage 1: OPAQUE detection using character-level analysis
        - Detects random characters, binary strings, keyboard patterns
        - Uses entropy, character profile, bigram naturalness

        Stage 2: Legibility-based classification for non-OPAQUE content
        - NATURAL: legibility >= 0.465
        - TECHNICAL: 0.440 <= legibility < 0.465
        - COMPRESSED: legibility < 0.440

        Note: Thresholds calibrated via grid search optimization.
        """
        # Stage 1: Character-level OPAQUE detection (high priority)
        if text:
            opacity_analysis = self.opaque_detector.analyze(text)
            if opacity_analysis.is_opaque:
                return LegibilityRegime.OPAQUE

        # Stage 2: Legibility-based classification
        if legibility >= 0.465:
            return LegibilityRegime.NATURAL
        elif legibility >= 0.440:
            return LegibilityRegime.TECHNICAL
        else:
            return LegibilityRegime.COMPRESSED

    def check_alerts(self) -> List[CBRAlert]:
        """Check for alert conditions."""
        alerts = []

        if len(self.state.readings) < 5:
            return alerts  # Need minimum readings

        readings = list(self.state.readings)

        # Variance collapse check (ossification precursor)
        temps = [r.temperature for r in readings]
        current_variance = float(np.var(temps[-10:]) if len(temps) >= 10 else np.var(temps))

        if len(readings) >= 20:
            historical_variance = float(np.var(temps[:-10]))
            if historical_variance > 0:
                variance_ratio = current_variance / historical_variance
                if variance_ratio < (1 - self.state.variance_collapse_threshold):
                    alerts.append(CBRAlert(
                        level=AlertLevel.WARNING,
                        message="Variance collapse detected - possible ossification approaching",
                        metric="variance_ratio",
                        value=variance_ratio,
                        threshold=1 - self.state.variance_collapse_threshold,
                    ))
                    self.state.alerts_issued += 1

        # Phase transition check
        if len(readings) >= 3:
            recent_phases = [r.phase for r in readings[-3:]]
            if len(set(recent_phases)) > 1:
                # Phase instability
                legibilities = [r.legibility for r in readings[-5:]]
                legibility_jump = max(legibilities) - min(legibilities)
                if legibility_jump > self.state.phase_transition_threshold:
                    alerts.append(CBRAlert(
                        level=AlertLevel.INFO,
                        message=f"Phase transition detected: legibility jump {legibility_jump:.2f}",
                        metric="legibility_delta",
                        value=legibility_jump,
                        threshold=self.state.phase_transition_threshold,
                    ))

        # OPAQUE trap warning
        recent = readings[-5:] if len(readings) >= 5 else readings
        opaque_count = sum(1 for r in recent if r.phase == LegibilityRegime.OPAQUE)
        if opaque_count >= 3:
            alerts.append(CBRAlert(
                level=AlertLevel.CRITICAL,
                message="Entering OPAQUE basin - high risk of trapping",
                metric="opaque_fraction",
                value=opaque_count / len(recent),
                threshold=0.6,
            ))
            self.state.alerts_issued += 1

        return alerts

    def get_summary(self) -> dict:
        """Get summary statistics from thermometer."""
        if not self.state.readings:
            return {"error": "No readings yet"}

        readings = list(self.state.readings)
        temps = [r.temperature for r in readings]
        signals = [r.signal_strength for r in readings]
        legibilities = [r.legibility for r in readings]

        # Kernel state distribution
        kernel_counts = [0] * 8
        for r in readings:
            kernel_counts[r.kernel_state] += 1
        kernel_entropy = self._entropy(kernel_counts)

        # Phase distribution
        phase_counts = {}
        for r in readings:
            phase_counts[r.phase.value] = phase_counts.get(r.phase.value, 0) + 1

        return {
            "total_readings": self.state.total_readings,
            "window_size": len(readings),
            "temperature": {
                "mean": float(np.mean(temps)),
                "std": float(np.std(temps)),
                "min": float(np.min(temps)),
                "max": float(np.max(temps)),
                "current": temps[-1],
            },
            "signal_strength": {
                "mean": float(np.mean(signals)),
                "current": signals[-1],
            },
            "legibility": {
                "mean": float(np.mean(legibilities)),
                "current": legibilities[-1],
            },
            "phase": {
                "current": readings[-1].phase.value,
                "distribution": phase_counts,
            },
            "kernel": {
                "current_state": readings[-1].kernel_state,
                "current_label": readings[-1].kernel_label,
                "distribution": kernel_counts,
                "entropy": kernel_entropy,
            },
            "alerts_issued": self.state.alerts_issued,
        }

    def _entropy(self, counts: List[int]) -> float:
        """Compute Shannon entropy of distribution."""
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = [c / total for c in counts if c > 0]
        return -sum(p * np.log2(p) for p in probs)

    def get_phase_position(self) -> Tuple[str, float, str]:
        """
        Get current phase position with distance to boundaries.
        Returns: (phase, legibility, nearest_boundary_description)
        """
        if not self.state.readings:
            return ("unknown", 0.0, "no readings")

        current = self.state.readings[-1]
        leg = current.legibility
        phase = current.phase.value

        # Compute distances to boundaries
        boundaries = [
            (0.85, "NATURAL→TECHNICAL"),
            (0.60, "TECHNICAL→COMPRESSED"),
            (0.30, "COMPRESSED→OPAQUE"),
        ]

        nearest_dist = float('inf')
        nearest_desc = ""
        for boundary, desc in boundaries:
            dist = abs(leg - boundary)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_desc = f"{desc} (Δ={nearest_dist:.2f})"

        return (phase, leg, nearest_desc)


# Convenience function for quick measurement
def measure_cbr(text: str) -> dict:
    """Quick CBR measurement of a single text."""
    thermometer = CBRThermometer()
    reading = thermometer.measure(text)
    return {
        "temperature": reading.temperature,
        "signal_strength": reading.signal_strength,
        "phase": reading.phase.value,
        "legibility": reading.legibility,
        "kernel_state": reading.kernel_state,
        "kernel_label": reading.kernel_label,
    }


def measure_cbr_batch(texts: List[str]) -> dict:
    """Batch CBR measurement with summary statistics."""
    thermometer = CBRThermometer(window_size=len(texts))
    for text in texts:
        thermometer.measure(text)
    return thermometer.get_summary()
