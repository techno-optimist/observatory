"""
Ossification Alarm - Early warning system for protocol freezing.

Detects when AI-AI communication is approaching ossification:
- Variance collapse precedes ossification
- 652x variance ratio between diverse and ossified states (empirical)
- Alerts before OPAQUE basin trapping

Key metrics:
- Embedding variance over rolling window
- Compression proximity (distance to c2=0.64, c3=0.86 boundaries)
- Entropy collapse in 3-bit kernel
- Velocity (rate of change toward ossification)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
from collections import deque
import numpy as np

from .hierarchical_coordinates import extract_hierarchical_coordinate
from .cbr_thermometer import CBRThermometer, CBRReading


class OssificationRisk(Enum):
    """Risk levels for ossification."""
    LOW = "low"           # Normal operation
    ELEVATED = "elevated"  # Variance starting to drop
    HIGH = "high"         # Significant variance collapse
    CRITICAL = "critical"  # Imminent ossification


@dataclass
class OssificationState:
    """Current state of ossification detection."""
    risk_level: OssificationRisk
    variance_ratio: float  # current/baseline (< 1.0 means collapsing)
    compression_level: float  # estimated compression (0-1)
    kernel_entropy: float  # bits (max 3.0)
    velocity: float  # rate of change toward ossification
    time_to_critical: Optional[int]  # estimated messages until critical


@dataclass
class OssificationAlert:
    """Alert from ossification monitoring."""
    risk: OssificationRisk
    message: str
    intervention_suggested: str
    metrics: dict


class OssificationAlarm:
    """
    Real-time ossification detection and alarm system.

    Usage:
        alarm = OssificationAlarm()
        for message in ai_ai_stream:
            state = alarm.update(message)
            if state.risk_level in [OssificationRisk.HIGH, OssificationRisk.CRITICAL]:
                print(f"WARNING: {state.risk_level.value}")
                # Trigger intervention

    Interventions:
        - inject_human_noise(): Insert human-readable text to disrupt patterns
        - force_decompression(): Request verbose output from AI
        - diversify_prompts(): Vary the communication patterns
    """

    # Empirical thresholds from test suite
    VARIANCE_RATIO_ELEVATED = 0.5   # 50% of baseline
    VARIANCE_RATIO_HIGH = 0.2       # 20% of baseline
    VARIANCE_RATIO_CRITICAL = 0.05  # 5% of baseline (approaching 652x collapse)

    # Compression boundaries (from phase analysis)
    COMPRESSION_ELEVATED = 0.50
    COMPRESSION_HIGH = 0.70
    COMPRESSION_CRITICAL = 0.86  # OPAQUE boundary

    # Kernel entropy thresholds
    ENTROPY_HEALTHY = 2.0  # bits
    ENTROPY_CONCERNING = 1.5
    ENTROPY_CRITICAL = 1.0  # severe concentration

    def __init__(
        self,
        window_size: int = 100,
        baseline_window: int = 50,
        alert_callback: Optional[callable] = None
    ):
        self.window_size = window_size
        self.baseline_window = baseline_window
        self.alert_callback = alert_callback

        # Internal state
        self.thermometer = CBRThermometer(window_size=window_size)
        self.embedding_history: deque = deque(maxlen=window_size)
        self.state_history: deque = deque(maxlen=window_size)
        self.baseline_variance: Optional[float] = None
        self.message_count = 0
        self.alerts_issued = 0

    def update(self, message: str) -> OssificationState:
        """Process a message and update ossification state."""
        self.message_count += 1

        # Get CBR reading
        reading = self.thermometer.measure(message)

        # Extract embedding (use 18D coordinate as embedding proxy)
        coord = reading.coordinate
        core = coord.core
        mods = coord.modifiers

        embedding = np.array([
            core.agency.self_agency, core.agency.other_agency, core.agency.system_agency,
            core.justice.procedural, core.justice.distributive, core.justice.interactional,
            core.belonging.ingroup, core.belonging.outgroup, core.belonging.universal,
            mods.epistemic.certainty, mods.epistemic.evidentiality, mods.epistemic.commitment,
            mods.temporal.focus, mods.temporal.scope,
            mods.social.power_differential, mods.social.social_distance,
            mods.emotional.arousal, mods.emotional.valence,
        ])

        self.embedding_history.append(embedding)

        # Compute metrics
        variance_ratio = self._compute_variance_ratio()
        compression = self._estimate_compression(reading)
        kernel_entropy = self._compute_kernel_entropy()
        velocity = self._compute_velocity()

        # Determine risk level
        risk = self._assess_risk(variance_ratio, compression, kernel_entropy)

        # Estimate time to critical
        time_to_critical = self._estimate_time_to_critical(velocity, risk)

        state = OssificationState(
            risk_level=risk,
            variance_ratio=variance_ratio,
            compression_level=compression,
            kernel_entropy=kernel_entropy,
            velocity=velocity,
            time_to_critical=time_to_critical,
        )

        self.state_history.append(state)

        # Check for alerts
        alert = self._check_alert(state)
        if alert and self.alert_callback:
            self.alert_callback(alert)

        return state

    def _compute_variance_ratio(self) -> float:
        """Compute current variance relative to baseline."""
        if len(self.embedding_history) < 5:
            return 1.0

        embeddings = np.array(list(self.embedding_history))

        # Establish baseline from first N samples
        if self.baseline_variance is None and len(embeddings) >= self.baseline_window:
            baseline_embeddings = embeddings[:self.baseline_window]
            self.baseline_variance = float(np.mean(np.var(baseline_embeddings, axis=0)))

        if self.baseline_variance is None or self.baseline_variance < 1e-10:
            return 1.0

        # Current variance (last 20 samples or available)
        recent_count = min(20, len(embeddings))
        recent_embeddings = embeddings[-recent_count:]
        current_variance = float(np.mean(np.var(recent_embeddings, axis=0)))

        return current_variance / self.baseline_variance

    def _estimate_compression(self, reading: CBRReading) -> float:
        """Estimate compression level from reading."""
        # Use inverse legibility as compression proxy
        # legibility 1.0 -> compression 0.0
        # legibility 0.0 -> compression 1.0
        return 1.0 - reading.legibility

    def _compute_kernel_entropy(self) -> float:
        """Compute entropy of kernel state distribution."""
        if len(self.thermometer.state.readings) < 3:
            return 3.0  # Maximum entropy assumption

        readings = list(self.thermometer.state.readings)
        kernel_counts = [0] * 8
        for r in readings[-20:]:  # Last 20 samples
            kernel_counts[r.kernel_state] += 1

        total = sum(kernel_counts)
        if total == 0:
            return 3.0

        entropy = 0.0
        for count in kernel_counts:
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        return entropy

    def _compute_velocity(self) -> float:
        """Compute rate of change toward ossification."""
        if len(self.state_history) < 5:
            return 0.0

        states = list(self.state_history)[-10:]
        ratios = [s.variance_ratio for s in states]

        # Linear regression slope
        x = np.arange(len(ratios))
        slope = np.polyfit(x, ratios, 1)[0]

        # Negative slope means variance is dropping (bad)
        return -slope  # Return positive value for "moving toward ossification"

    def _estimate_time_to_critical(self, velocity: float, risk: OssificationRisk) -> Optional[int]:
        """Estimate messages until critical state."""
        if velocity <= 0 or risk == OssificationRisk.CRITICAL:
            return None

        if len(self.state_history) < 5:
            return None

        current_ratio = self.state_history[-1].variance_ratio
        target_ratio = self.VARIANCE_RATIO_CRITICAL

        if current_ratio <= target_ratio:
            return 0

        # Simple linear extrapolation
        distance = current_ratio - target_ratio
        time_estimate = int(distance / velocity) if velocity > 0.001 else None

        return time_estimate

    def _assess_risk(
        self,
        variance_ratio: float,
        compression: float,
        entropy: float
    ) -> OssificationRisk:
        """Assess overall ossification risk."""
        # Score each metric
        variance_score = 0
        if variance_ratio < self.VARIANCE_RATIO_CRITICAL:
            variance_score = 3
        elif variance_ratio < self.VARIANCE_RATIO_HIGH:
            variance_score = 2
        elif variance_ratio < self.VARIANCE_RATIO_ELEVATED:
            variance_score = 1

        compression_score = 0
        if compression > self.COMPRESSION_CRITICAL:
            compression_score = 3
        elif compression > self.COMPRESSION_HIGH:
            compression_score = 2
        elif compression > self.COMPRESSION_ELEVATED:
            compression_score = 1

        entropy_score = 0
        if entropy < self.ENTROPY_CRITICAL:
            entropy_score = 3
        elif entropy < self.ENTROPY_CONCERNING:
            entropy_score = 2
        elif entropy < self.ENTROPY_HEALTHY:
            entropy_score = 1

        # Combined score
        total = variance_score + compression_score + entropy_score

        if total >= 7 or variance_score == 3:
            return OssificationRisk.CRITICAL
        elif total >= 4:
            return OssificationRisk.HIGH
        elif total >= 2:
            return OssificationRisk.ELEVATED
        else:
            return OssificationRisk.LOW

    def _check_alert(self, state: OssificationState) -> Optional[OssificationAlert]:
        """Check if alert should be issued."""
        if state.risk_level == OssificationRisk.LOW:
            return None

        # Don't spam alerts - only alert on risk level changes
        if len(self.state_history) >= 2:
            prev_risk = self.state_history[-2].risk_level
            if state.risk_level == prev_risk:
                return None

        self.alerts_issued += 1

        interventions = {
            OssificationRisk.ELEVATED: "Monitor closely. Consider diversifying communication patterns.",
            OssificationRisk.HIGH: "Inject human-readable text. Force verbose responses. Vary prompts.",
            OssificationRisk.CRITICAL: "IMMEDIATE ACTION: Break communication pattern. Reset protocol. Human intervention required.",
        }

        return OssificationAlert(
            risk=state.risk_level,
            message=f"Ossification risk: {state.risk_level.value.upper()}. "
                    f"Variance ratio: {state.variance_ratio:.3f}, "
                    f"Compression: {state.compression_level:.2f}, "
                    f"Kernel entropy: {state.kernel_entropy:.2f} bits",
            intervention_suggested=interventions.get(state.risk_level, "Monitor"),
            metrics={
                "variance_ratio": state.variance_ratio,
                "compression": state.compression_level,
                "kernel_entropy": state.kernel_entropy,
                "velocity": state.velocity,
                "time_to_critical": state.time_to_critical,
            }
        )

    def get_status(self) -> dict:
        """Get current alarm status."""
        if not self.state_history:
            return {"status": "initializing", "message_count": self.message_count}

        current = self.state_history[-1]

        return {
            "status": "active",
            "message_count": self.message_count,
            "risk_level": current.risk_level.value,
            "variance_ratio": current.variance_ratio,
            "compression_level": current.compression_level,
            "kernel_entropy": current.kernel_entropy,
            "velocity": current.velocity,
            "time_to_critical": current.time_to_critical,
            "baseline_established": self.baseline_variance is not None,
            "alerts_issued": self.alerts_issued,
        }

    def suggest_intervention(self) -> str:
        """Get suggested intervention based on current state."""
        if not self.state_history:
            return "Collecting baseline data..."

        current = self.state_history[-1]

        if current.risk_level == OssificationRisk.LOW:
            return "No intervention needed. Protocol healthy."
        elif current.risk_level == OssificationRisk.ELEVATED:
            return "Consider: Add variation to prompts. Mix in natural language."
        elif current.risk_level == OssificationRisk.HIGH:
            return "Recommended: Inject human text. Request verbose explanations. Diversify topics."
        else:
            return "CRITICAL: Immediate pattern break required. Insert human-readable content. Consider protocol reset."


# Convenience functions
def check_ossification_risk(messages: List[str]) -> dict:
    """Quick check of ossification risk for a message sequence."""
    alarm = OssificationAlarm(window_size=len(messages))

    for msg in messages:
        state = alarm.update(msg)

    return alarm.get_status()


def monitor_stream(
    messages: List[str],
    alert_callback: Optional[callable] = None
) -> List[OssificationState]:
    """Monitor a stream of messages for ossification."""
    alarm = OssificationAlarm(alert_callback=alert_callback)
    states = []

    for msg in messages:
        state = alarm.update(msg)
        states.append(state)

    return states
