"""
Evolution Tracker - Linguistic Deep Time on Fast-Forward

Tracks how communication protocols evolve through coordination space, enabling
observation of compressed evolution that would take centuries in natural language.

The Core Insight:
-----------------
During AI training, we observe something remarkable: emergent communication systems
that develop in hours what took human languages millennia. This module provides
the instrumentation to study this phenomenon.

When AI agents learn to communicate, they traverse a trajectory through
coordination space. At first, signals are random noise. Gradually, variance
decreases as agents stabilize on shared conventions. Eventually, compositional
structure emerges - the ability to combine primitive symbols into meaningful
combinations. Finally, the protocol "ossifies" - becoming frozen and resistant
to change, just as natural languages do over centuries.

This is "linguistic deep time on fast-forward" - watching language evolution
compressed from millennia to training runs. The Cultural Soliton Observatory
provides the coordinate system; this module provides the tracking.

Evolution Stages:
-----------------
RANDOM_SIGNAL: No structure, high variance. Agents emit noise. Entropy is maximal.
STABILIZING: Variance decreasing, clusters forming. Agents beginning to coordinate.
COMPOSITIONAL: Consistent patterns, reusable elements. True language emerging.
OSSIFIED: Frozen protocol, no further evolution. A "soliton" has formed.

Key Metrics:
------------
- Evolution velocity: Rate of change through manifold (should decrease over training)
- Compositionality score: Degree of reusable structure (should increase)
- Ossification: Protocol rigidity (should increase toward 1.0 as training ends)
- Stabilization point: When soliton forms (velocity → 0, structure → stable)

Usage:
    python -m research.evolution_tracker --demo
    python -m research.evolution_tracker --texts-file trajectory.jsonl
"""

import asyncio
import logging
import json
import math
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


class EvolutionStage(Enum):
    """
    Stages of communication protocol evolution.

    These map onto the lifecycle of emergent language:
    - Random signals → Stabilizing clusters → Compositional structure → Frozen protocol

    This mirrors natural language evolution:
    - Proto-language → Early grammar → Full syntax → Ossified structure

    But compressed from millennia to training runs.
    """
    RANDOM_SIGNAL = "RANDOM_SIGNAL"      # High entropy, no patterns
    STABILIZING = "STABILIZING"          # Variance decreasing, clusters forming
    COMPOSITIONAL = "COMPOSITIONAL"      # Reusable patterns, combining elements
    OSSIFIED = "OSSIFIED"                # Frozen protocol, maximum rigidity


@dataclass
class TrajectoryPoint:
    """
    A single point in the evolution trajectory through coordination space.

    Captures both the raw text and its projection into the 3D manifold:
    (agency, perceived_justice, belonging)
    """
    timestamp: datetime
    text: str
    agency: float
    perceived_justice: float
    belonging: float
    mode: str
    confidence: float = 0.0
    raw_response: Dict[str, Any] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        """Return coordinates as numpy array."""
        return np.array([self.agency, self.perceived_justice, self.belonging])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "text": self.text,
            "agency": self.agency,
            "perceived_justice": self.perceived_justice,
            "belonging": self.belonging,
            "mode": self.mode,
            "confidence": self.confidence
        }


@dataclass
class VelocityVector:
    """
    Velocity through the coordination manifold between two points.

    This measures how fast the communication protocol is evolving.
    High velocity = rapid change. Low velocity = stabilization.
    As training progresses, velocity should decrease toward zero
    as the protocol ossifies into a stable soliton.
    """
    from_time: datetime
    to_time: datetime
    delta_agency: float
    delta_perceived_justice: float
    delta_belonging: float
    magnitude: float
    time_delta_seconds: float

    @classmethod
    def compute(cls, p1: TrajectoryPoint, p2: TrajectoryPoint) -> "VelocityVector":
        """Compute velocity vector between two trajectory points."""
        delta = p2.to_vector() - p1.to_vector()
        time_delta = (p2.timestamp - p1.timestamp).total_seconds()

        # Avoid division by zero
        if time_delta == 0:
            time_delta = 1.0

        magnitude = float(np.linalg.norm(delta)) / time_delta

        return cls(
            from_time=p1.timestamp,
            to_time=p2.timestamp,
            delta_agency=float(delta[0]),
            delta_perceived_justice=float(delta[1]),
            delta_belonging=float(delta[2]),
            magnitude=magnitude,
            time_delta_seconds=time_delta
        )


@dataclass
class CompositionalityMetrics:
    """
    Metrics for detecting compositional structure in communication.

    Compositionality is the hallmark of true language: the ability to combine
    primitive symbols into meaningful combinations. When agents develop
    compositional communication, they've crossed a crucial threshold.

    We detect this by looking for:
    - Pattern reuse: Same sub-sequences appearing in different contexts
    - Structural consistency: Similar transformations having consistent effects
    - Mode stability: Consistent classification despite surface variation
    """
    reuse_score: float              # How often patterns repeat (0-1)
    structural_consistency: float    # Consistency of transformations (0-1)
    mode_stability: float           # How stable modes are across variations (0-1)
    composite_score: float          # Overall compositionality (0-1)
    detected_patterns: List[str]    # Identified reusable patterns
    emergence_timestamp: Optional[datetime] = None  # When compositionality emerged


@dataclass
class EvolutionAnalysis:
    """
    Complete analysis of a communication protocol's evolution.

    This is the main output of the EvolutionTracker, capturing the entire
    trajectory from random signals to (potentially) ossified soliton.
    """
    trajectory_points: List[TrajectoryPoint]
    velocity_vectors: List[VelocityVector]
    current_stage: EvolutionStage
    ossification_score: float           # 0.0 (fluid) to 1.0 (frozen)
    evolution_velocity: float           # Current rate of change
    stabilization_point: Optional[TrajectoryPoint]  # When soliton formed
    compositionality: CompositionalityMetrics
    stage_history: List[Tuple[datetime, EvolutionStage]]
    total_distance_traveled: float
    net_displacement: float
    interpretation: str


class EvolutionTracker:
    """
    Tracks communication protocol evolution through coordination space.

    This is the core instrument for observing "linguistic deep time on fast-forward."
    Feed it texts generated over time (whether from AI training or human communication)
    and it will track the trajectory through the coordination manifold, detecting:

    1. Evolution stages: RANDOM_SIGNAL → STABILIZING → COMPOSITIONAL → OSSIFIED
    2. Velocity: How fast the protocol is changing
    3. Compositionality: When true language structure emerges
    4. Ossification: When the protocol freezes into a stable soliton

    The magic is that what takes natural languages millennia, AI training
    accomplishes in hours - and this tracker lets us observe it.
    """

    # Thresholds for stage detection
    VARIANCE_THRESHOLD_STABILIZING = 0.5     # Below this = stabilizing
    VARIANCE_THRESHOLD_COMPOSITIONAL = 0.2   # Below this = potentially compositional
    MODE_STABILITY_THRESHOLD = 0.7           # Above this = compositional
    VELOCITY_THRESHOLD_OSSIFIED = 0.05       # Below this = ossified
    OSSIFICATION_WINDOW = 5                  # Points to check for ossification

    def __init__(self,
                 api_base_url: str = "http://127.0.0.1:8000",
                 timeout: float = 30.0):
        """
        Initialize the Evolution Tracker.

        Args:
            api_base_url: Base URL for the Cultural Soliton Observatory API
            timeout: HTTP request timeout in seconds
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for EvolutionTracker. "
                "Install with: pip install httpx"
            )

        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        self._trajectory: List[TrajectoryPoint] = []
        self._stage_history: List[Tuple[datetime, EvolutionStage]] = []

    async def project_text(self, text: str) -> Dict[str, Any]:
        """
        Project a single text through the Observatory API.

        Args:
            text: Text to analyze

        Returns:
            Full analysis result from /v2/analyze endpoint
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.api_base_url}/v2/analyze",
                json={
                    "text": text,
                    "include_uncertainty": True,
                    "include_legacy_mode": True
                }
            )
            response.raise_for_status()
            return response.json()

    async def track_evolution(
        self,
        texts_over_time: List[Tuple[datetime, str]]
    ) -> EvolutionAnalysis:
        """
        Track trajectory through coordination space over time.

        This is the main entry point for evolution tracking. Feed it a sequence
        of (timestamp, text) pairs and it will analyze the entire trajectory,
        detecting stage transitions, computing velocities, and identifying
        when (if) the protocol ossifies into a stable soliton.

        Args:
            texts_over_time: List of (timestamp, text) tuples, ordered by time

        Returns:
            Complete EvolutionAnalysis with trajectory, stages, and metrics
        """
        if len(texts_over_time) < 2:
            raise ValueError("Need at least 2 texts to track evolution")

        # Sort by timestamp
        sorted_texts = sorted(texts_over_time, key=lambda x: x[0])

        # Project all texts through the API
        trajectory_points = []
        for timestamp, text in sorted_texts:
            try:
                result = await self.project_text(text)

                vector = result.get("vector", {})
                mode_info = result.get("mode", {})

                point = TrajectoryPoint(
                    timestamp=timestamp,
                    text=text,
                    agency=vector.get("agency", 0.0),
                    perceived_justice=vector.get("perceived_justice", 0.0),
                    belonging=vector.get("belonging", 0.0),
                    mode=mode_info.get("primary_mode", "UNKNOWN"),
                    confidence=mode_info.get("confidence", 0.0),
                    raw_response=result
                )
                trajectory_points.append(point)

            except Exception as e:
                logger.warning(f"Failed to project text at {timestamp}: {e}")
                continue

        if len(trajectory_points) < 2:
            raise ValueError("Need at least 2 successful projections")

        self._trajectory = trajectory_points

        # Compute velocity vectors
        velocity_vectors = self._compute_velocity_vectors(trajectory_points)

        # Detect stages
        current_stage = self._detect_current_stage(trajectory_points, velocity_vectors)

        # Update stage history
        self._update_stage_history(current_stage, trajectory_points[-1].timestamp)

        # Compute metrics
        ossification = self.measure_protocol_ossification(
            [p.text for p in trajectory_points]
        )

        evolution_velocity = self.compute_evolution_velocity(trajectory_points)

        stabilization = self.find_stabilization_point(trajectory_points)

        compositionality = self.detect_compositionality_emergence(
            [p.text for p in trajectory_points]
        )

        # Compute distances
        total_distance = self._compute_total_distance(trajectory_points)
        net_displacement = self._compute_net_displacement(trajectory_points)

        # Generate interpretation
        interpretation = self._generate_interpretation(
            current_stage, ossification, evolution_velocity,
            compositionality, stabilization
        )

        return EvolutionAnalysis(
            trajectory_points=trajectory_points,
            velocity_vectors=velocity_vectors,
            current_stage=current_stage,
            ossification_score=ossification,
            evolution_velocity=evolution_velocity,
            stabilization_point=stabilization,
            compositionality=compositionality,
            stage_history=list(self._stage_history),
            total_distance_traveled=total_distance,
            net_displacement=net_displacement,
            interpretation=interpretation
        )

    def _compute_velocity_vectors(
        self,
        points: List[TrajectoryPoint]
    ) -> List[VelocityVector]:
        """Compute velocity vectors between consecutive points."""
        vectors = []
        for i in range(1, len(points)):
            vectors.append(VelocityVector.compute(points[i-1], points[i]))
        return vectors

    def _detect_current_stage(
        self,
        points: List[TrajectoryPoint],
        velocities: List[VelocityVector]
    ) -> EvolutionStage:
        """
        Detect the current evolution stage based on trajectory characteristics.

        Stage detection logic:
        1. RANDOM_SIGNAL: High variance, no consistent patterns
        2. STABILIZING: Decreasing variance, clusters forming
        3. COMPOSITIONAL: Low variance, stable modes, pattern reuse
        4. OSSIFIED: Near-zero velocity, completely frozen
        """
        if len(points) < 2:
            return EvolutionStage.RANDOM_SIGNAL

        # Compute recent variance (last 5 points or all if fewer)
        window_size = min(5, len(points))
        recent_points = points[-window_size:]

        vectors = np.array([p.to_vector() for p in recent_points])
        variance = np.mean(np.var(vectors, axis=0))

        # Compute recent velocity (average of last few)
        recent_velocities = velocities[-min(3, len(velocities)):]
        avg_velocity = np.mean([v.magnitude for v in recent_velocities]) if recent_velocities else float('inf')

        # Check mode stability
        modes = [p.mode for p in recent_points]
        unique_modes = set(modes)
        mode_stability = modes.count(modes[-1]) / len(modes) if modes else 0

        # Stage detection
        if avg_velocity < self.VELOCITY_THRESHOLD_OSSIFIED and variance < self.VARIANCE_THRESHOLD_COMPOSITIONAL:
            return EvolutionStage.OSSIFIED
        elif variance < self.VARIANCE_THRESHOLD_COMPOSITIONAL and mode_stability >= self.MODE_STABILITY_THRESHOLD:
            return EvolutionStage.COMPOSITIONAL
        elif variance < self.VARIANCE_THRESHOLD_STABILIZING:
            return EvolutionStage.STABILIZING
        else:
            return EvolutionStage.RANDOM_SIGNAL

    def _update_stage_history(self, stage: EvolutionStage, timestamp: datetime):
        """Update stage history, adding new entry if stage changed."""
        if not self._stage_history or self._stage_history[-1][1] != stage:
            self._stage_history.append((timestamp, stage))

    def detect_compositionality_emergence(
        self,
        texts: List[str]
    ) -> CompositionalityMetrics:
        """
        Detect when compositional structure emerges in the communication.

        Compositionality is detected through:
        1. Pattern reuse: Looking for repeated sub-sequences
        2. Structural consistency: Similar structures having consistent effects
        3. Mode stability: Same underlying meaning despite surface variation

        This is crucial because compositionality marks the transition from
        proto-communication to true language.
        """
        if len(texts) < 3:
            return CompositionalityMetrics(
                reuse_score=0.0,
                structural_consistency=0.0,
                mode_stability=0.0,
                composite_score=0.0,
                detected_patterns=[],
                emergence_timestamp=None
            )

        # Extract n-grams for pattern detection
        all_ngrams: Dict[str, int] = {}
        for text in texts:
            words = text.lower().split()
            for n in range(2, min(4, len(words))):
                for i in range(len(words) - n + 1):
                    ngram = " ".join(words[i:i+n])
                    all_ngrams[ngram] = all_ngrams.get(ngram, 0) + 1

        # Find repeated patterns
        repeated_patterns = [ng for ng, count in all_ngrams.items() if count >= 2]
        reuse_score = min(len(repeated_patterns) / (len(texts) / 2), 1.0) if texts else 0.0

        # Structural consistency: Check if similar patterns appear
        structural_score = 0.0
        if repeated_patterns:
            # Simple heuristic: more repeated patterns = more structure
            structural_score = min(len(repeated_patterns) / 10, 1.0)

        # Mode stability from trajectory
        mode_stability = 0.0
        if self._trajectory:
            recent = self._trajectory[-min(5, len(self._trajectory)):]
            modes = [p.mode for p in recent]
            if modes:
                most_common = max(set(modes), key=modes.count)
                mode_stability = modes.count(most_common) / len(modes)

        # Composite score
        composite = (reuse_score + structural_score + mode_stability) / 3

        # Find emergence point (when score first exceeds threshold)
        emergence_time = None
        if composite >= 0.5 and self._trajectory:
            # Scan backward to find when compositionality emerged
            for i in range(len(self._trajectory) - 1, -1, -1):
                window = self._trajectory[max(0, i-4):i+1]
                if len(window) >= 3:
                    modes = [p.mode for p in window]
                    if modes:
                        stability = modes.count(max(set(modes), key=modes.count)) / len(modes)
                        if stability < 0.5:
                            emergence_time = self._trajectory[min(i+1, len(self._trajectory)-1)].timestamp
                            break
            if emergence_time is None and self._trajectory:
                emergence_time = self._trajectory[0].timestamp

        return CompositionalityMetrics(
            reuse_score=float(reuse_score),
            structural_consistency=float(structural_score),
            mode_stability=float(mode_stability),
            composite_score=float(composite),
            detected_patterns=repeated_patterns[:10],  # Top 10 patterns
            emergence_timestamp=emergence_time
        )

    def measure_protocol_ossification(self, texts: List[str]) -> float:
        """
        Measure how "frozen" the communication protocol is.

        Returns a score from 0.0 (completely fluid, rapidly evolving)
        to 1.0 (completely frozen, no further evolution).

        Ossification is detected by:
        1. Low velocity: The trajectory is barely moving
        2. High mode stability: Same mode classifications
        3. Low variance: Points cluster tightly

        High ossification indicates a "soliton" has formed - a stable
        communication pattern that resists further change.
        """
        if not self._trajectory or len(self._trajectory) < 2:
            return 0.0

        # Use recent window for ossification detection
        window_size = min(self.OSSIFICATION_WINDOW, len(self._trajectory))
        recent = self._trajectory[-window_size:]

        # Component 1: Velocity decay
        if len(recent) >= 2:
            velocities = []
            for i in range(1, len(recent)):
                v = VelocityVector.compute(recent[i-1], recent[i])
                velocities.append(v.magnitude)
            avg_velocity = np.mean(velocities) if velocities else 1.0
            velocity_score = max(0, 1 - avg_velocity * 10)  # Low velocity = high score
        else:
            velocity_score = 0.0

        # Component 2: Mode stability
        modes = [p.mode for p in recent]
        if modes:
            most_common = max(set(modes), key=modes.count)
            mode_score = modes.count(most_common) / len(modes)
        else:
            mode_score = 0.0

        # Component 3: Coordinate variance (low = ossified)
        vectors = np.array([p.to_vector() for p in recent])
        variance = np.mean(np.var(vectors, axis=0))
        variance_score = max(0, 1 - variance * 2)  # Low variance = high score

        # Weighted combination
        ossification = (velocity_score * 0.4 + mode_score * 0.3 + variance_score * 0.3)

        return float(np.clip(ossification, 0.0, 1.0))

    def compute_evolution_velocity(
        self,
        trajectory: List[TrajectoryPoint]
    ) -> float:
        """
        Compute the current rate of change through the manifold.

        Returns the average velocity magnitude over recent points.
        This tells us how fast the protocol is evolving right now.

        Interpretation:
        - High velocity (>0.5): Rapid evolution, protocol still forming
        - Medium velocity (0.1-0.5): Active development, patterns emerging
        - Low velocity (<0.1): Stabilizing, approaching ossification
        - Near-zero (<0.05): Ossified, soliton formed
        """
        if len(trajectory) < 2:
            return float('inf')

        # Compute velocities for recent window
        window_size = min(5, len(trajectory))
        recent = trajectory[-window_size:]

        velocities = []
        for i in range(1, len(recent)):
            v = VelocityVector.compute(recent[i-1], recent[i])
            velocities.append(v.magnitude)

        return float(np.mean(velocities)) if velocities else 0.0

    def find_stabilization_point(
        self,
        trajectory: List[TrajectoryPoint]
    ) -> Optional[TrajectoryPoint]:
        """
        Find when the soliton forms (velocity approaches zero).

        Returns the trajectory point where the protocol first stabilized,
        or None if no stabilization has been detected yet.

        This is the key moment in language evolution: when the protocol
        transitions from fluid development to stable structure.
        """
        if len(trajectory) < 3:
            return None

        # Look for sustained low velocity
        sustained_low = 0
        required_sustained = 3  # Need 3 consecutive low-velocity points

        for i in range(2, len(trajectory)):
            # Compute velocity for this transition
            v = VelocityVector.compute(trajectory[i-1], trajectory[i])

            if v.magnitude < self.VELOCITY_THRESHOLD_OSSIFIED:
                sustained_low += 1
                if sustained_low >= required_sustained:
                    # Found stabilization point
                    return trajectory[i - required_sustained + 1]
            else:
                sustained_low = 0

        return None

    def _compute_total_distance(self, points: List[TrajectoryPoint]) -> float:
        """Compute total distance traveled through manifold."""
        if len(points) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(points)):
            delta = points[i].to_vector() - points[i-1].to_vector()
            total += float(np.linalg.norm(delta))
        return total

    def _compute_net_displacement(self, points: List[TrajectoryPoint]) -> float:
        """Compute net displacement from start to end."""
        if len(points) < 2:
            return 0.0

        delta = points[-1].to_vector() - points[0].to_vector()
        return float(np.linalg.norm(delta))

    def _generate_interpretation(
        self,
        stage: EvolutionStage,
        ossification: float,
        velocity: float,
        compositionality: CompositionalityMetrics,
        stabilization: Optional[TrajectoryPoint]
    ) -> str:
        """Generate human-readable interpretation of evolution analysis."""

        lines = []

        # Stage interpretation
        stage_descriptions = {
            EvolutionStage.RANDOM_SIGNAL:
                "The communication protocol is in its earliest stage - signals appear random "
                "with high variance and no discernible patterns. This is the 'primordial soup' "
                "of language evolution.",
            EvolutionStage.STABILIZING:
                "The protocol is beginning to stabilize. Variance is decreasing as agents "
                "converge on shared conventions. Clusters are forming in coordination space.",
            EvolutionStage.COMPOSITIONAL:
                "Compositional structure has emerged! The protocol now shows reusable patterns "
                "and consistent mode classifications. This is true language structure.",
            EvolutionStage.OSSIFIED:
                "The protocol has ossified into a stable soliton. Evolution has effectively "
                "stopped. The communication system has frozen into its final form."
        }
        lines.append(f"Stage: {stage.value}")
        lines.append(stage_descriptions[stage])
        lines.append("")

        # Ossification
        if ossification < 0.3:
            rigidity = "highly fluid"
        elif ossification < 0.6:
            rigidity = "moderately rigid"
        elif ossification < 0.8:
            rigidity = "largely ossified"
        else:
            rigidity = "completely frozen"
        lines.append(f"Ossification: {ossification:.2f} ({rigidity})")
        lines.append("")

        # Velocity
        if velocity < 0.05:
            velocity_desc = "near-zero - evolution has stopped"
        elif velocity < 0.2:
            velocity_desc = "low - protocol is stabilizing"
        elif velocity < 0.5:
            velocity_desc = "moderate - active development"
        else:
            velocity_desc = "high - rapid evolution"
        lines.append(f"Evolution velocity: {velocity:.3f} ({velocity_desc})")
        lines.append("")

        # Compositionality
        lines.append(f"Compositionality score: {compositionality.composite_score:.2f}")
        if compositionality.composite_score >= 0.5:
            lines.append("  True compositional structure has emerged.")
            if compositionality.detected_patterns:
                lines.append(f"  Detected patterns: {', '.join(compositionality.detected_patterns[:3])}")
        else:
            lines.append("  Compositionality has not yet emerged.")
        lines.append("")

        # Stabilization
        if stabilization:
            lines.append(f"Soliton formed at: {stabilization.timestamp.isoformat()}")
            lines.append(f"  Mode at stabilization: {stabilization.mode}")
        else:
            lines.append("No stabilization point detected yet - protocol still evolving.")

        return "\n".join(lines)


# =============================================================================
# CLI Interface
# =============================================================================

def generate_sample_trajectory() -> List[Tuple[datetime, str]]:
    """
    Generate a sample trajectory demonstrating protocol evolution.

    This simulates what we might observe during AI training:
    1. Early random signals
    2. Stabilizing coordination
    3. Compositional emergence
    4. Final ossification
    """
    from datetime import timedelta

    base_time = datetime(2024, 1, 1, 0, 0, 0)

    # Stage 1: Random signals (high variance, chaotic)
    random_phase = [
        (base_time + timedelta(hours=0), "fragmented chaos nobody understands anything"),
        (base_time + timedelta(hours=1), "we tried but failed utterly complete mess"),
        (base_time + timedelta(hours=2), "random attempts no coordination happening here"),
        (base_time + timedelta(hours=3), "everyone for themselves no shared meaning yet"),
    ]

    # Stage 2: Stabilizing (clusters forming)
    stabilizing_phase = [
        (base_time + timedelta(hours=4), "we are starting to understand each other better now"),
        (base_time + timedelta(hours=5), "together we can figure this out as a team"),
        (base_time + timedelta(hours=6), "our group is finding common ground finally"),
        (base_time + timedelta(hours=7), "the team is working well and sharing ideas"),
    ]

    # Stage 3: Compositional (patterns emerge)
    compositional_phase = [
        (base_time + timedelta(hours=8), "our team works together and we share ideas effectively"),
        (base_time + timedelta(hours=9), "we share ideas and our team works together on solutions"),
        (base_time + timedelta(hours=10), "together our team shares ideas and works effectively"),
        (base_time + timedelta(hours=11), "our team works together sharing ideas with each other"),
    ]

    # Stage 4: Ossified (frozen protocol)
    ossified_phase = [
        (base_time + timedelta(hours=12), "our team works together and we share ideas effectively"),
        (base_time + timedelta(hours=13), "our team works together and we share ideas effectively"),
        (base_time + timedelta(hours=14), "our team works together and we share ideas effectively"),
        (base_time + timedelta(hours=15), "our team works together and we share ideas effectively"),
    ]

    return random_phase + stabilizing_phase + compositional_phase + ossified_phase


async def run_demo():
    """Run a demonstration of the Evolution Tracker."""
    print("\n" + "=" * 70)
    print("EVOLUTION TRACKER DEMO - Linguistic Deep Time on Fast-Forward")
    print("=" * 70)
    print()
    print("This demo simulates observing language evolution during AI training,")
    print("compressed from millennia to hours. Watch as random signals evolve")
    print("into compositional language and finally ossify into a stable soliton.")
    print()

    # Generate sample trajectory
    trajectory = generate_sample_trajectory()

    print(f"Sample trajectory: {len(trajectory)} texts over {len(trajectory)} hours")
    print("-" * 70)

    # Create tracker
    tracker = EvolutionTracker()

    try:
        # Track evolution
        print("\nProjecting texts through coordination manifold...")
        analysis = await tracker.track_evolution(trajectory)

        print("\n" + "=" * 70)
        print("EVOLUTION ANALYSIS")
        print("=" * 70)

        # Print trajectory points
        print("\nTrajectory Through Coordination Space:")
        print("-" * 70)
        for i, point in enumerate(analysis.trajectory_points):
            print(f"  [{i+1:2d}] {point.timestamp.strftime('%H:%M')} | "
                  f"A:{point.agency:+.2f} J:{point.perceived_justice:+.2f} B:{point.belonging:+.2f} | "
                  f"{point.mode}")

        # Print velocity vectors
        print("\nVelocity Through Manifold:")
        print("-" * 70)
        for v in analysis.velocity_vectors:
            print(f"  {v.from_time.strftime('%H:%M')} -> {v.to_time.strftime('%H:%M')}: "
                  f"magnitude={v.magnitude:.4f}")

        # Print stage history
        print("\nStage Transitions:")
        print("-" * 70)
        for timestamp, stage in analysis.stage_history:
            print(f"  {timestamp.strftime('%H:%M')}: {stage.value}")

        # Print metrics
        print("\nKey Metrics:")
        print("-" * 70)
        print(f"  Current Stage: {analysis.current_stage.value}")
        print(f"  Ossification Score: {analysis.ossification_score:.2f}")
        print(f"  Evolution Velocity: {analysis.evolution_velocity:.4f}")
        print(f"  Total Distance Traveled: {analysis.total_distance_traveled:.2f}")
        print(f"  Net Displacement: {analysis.net_displacement:.2f}")

        # Compositionality
        print("\nCompositionality Metrics:")
        print("-" * 70)
        c = analysis.compositionality
        print(f"  Reuse Score: {c.reuse_score:.2f}")
        print(f"  Structural Consistency: {c.structural_consistency:.2f}")
        print(f"  Mode Stability: {c.mode_stability:.2f}")
        print(f"  Composite Score: {c.composite_score:.2f}")
        if c.detected_patterns:
            print(f"  Detected Patterns: {', '.join(c.detected_patterns[:5])}")

        # Stabilization point
        print("\nStabilization:")
        print("-" * 70)
        if analysis.stabilization_point:
            print(f"  Soliton formed at: {analysis.stabilization_point.timestamp.strftime('%H:%M')}")
            print(f"  Mode: {analysis.stabilization_point.mode}")
        else:
            print("  No stabilization point detected yet")

        # Full interpretation
        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)
        print(analysis.interpretation)

    except httpx.ConnectError:
        print("\nERROR: Could not connect to Observatory API at http://127.0.0.1:8000")
        print("Make sure the server is running: uvicorn main:app --reload")
        print()
        print("Running in offline mode with mock data...")
        print()

        # Show what the trajectory would look like
        print("Sample trajectory (would be analyzed if API were available):")
        for i, (ts, text) in enumerate(trajectory[:5]):
            print(f"  [{i+1}] {ts.strftime('%H:%M')}: {text[:50]}...")
        print(f"  ... and {len(trajectory) - 5} more")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


async def run_from_file(file_path: str):
    """
    Run evolution tracking from a JSONL file.

    Expected format (one per line):
    {"timestamp": "2024-01-01T00:00:00", "text": "..."}
    """
    import json

    print(f"\nLoading trajectory from: {file_path}")

    texts_over_time = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                ts = datetime.fromisoformat(data["timestamp"])
                texts_over_time.append((ts, data["text"]))

    print(f"Loaded {len(texts_over_time)} texts")

    tracker = EvolutionTracker()
    analysis = await tracker.track_evolution(texts_over_time)

    # Print results (same as demo)
    print("\n" + "=" * 70)
    print("EVOLUTION ANALYSIS")
    print("=" * 70)
    print(analysis.interpretation)


async def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evolution Tracker - Observe linguistic deep time on fast-forward",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m research.evolution_tracker --demo
  python -m research.evolution_tracker --texts-file trajectory.jsonl

The --demo flag runs with sample data showing protocol evolution from
random signals through ossification.

For --texts-file, provide a JSONL file with format:
  {"timestamp": "2024-01-01T00:00:00", "text": "..."}
  {"timestamp": "2024-01-01T01:00:00", "text": "..."}
        """
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with sample trajectory")
    parser.add_argument("--texts-file", type=str,
                        help="JSONL file with timestamped texts")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000",
                        help="Observatory API base URL")

    args = parser.parse_args()

    if args.demo:
        await run_demo()
    elif args.texts_file:
        await run_from_file(args.texts_file)
    else:
        # Default to demo
        print("No input specified, running demo...")
        await run_demo()


if __name__ == "__main__":
    asyncio.run(main())
