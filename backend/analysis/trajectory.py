"""
Temporal Trajectory Tracking for Narratives

Tracks how entities (companies, speakers, etc.) move through the cultural
manifold over time. Computes dynamics like velocity, phase transitions,
and narrative arc classification.
"""

import json
import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class NarrativeArc(str, Enum):
    """Classification of overall narrative trajectory shape."""
    RISE = "RISE"                           # Consistent upward movement
    FALL = "FALL"                           # Consistent downward movement
    FALL_AND_REDEMPTION = "FALL_AND_REDEMPTION"  # Down then up
    STEADY = "STEADY"                       # Little movement
    CHAOTIC = "CHAOTIC"                     # Unpredictable oscillation


@dataclass
class TrajectoryPoint:
    """A single point in a narrative trajectory."""
    timestamp: str  # ISO format datetime
    text: str
    coordinates: Tuple[float, float, float]  # (agency, fairness, belonging)
    mode: str  # DREAM_POSITIVE, DREAM_SHADOW, DREAM_EXIT, NOISE_OTHER

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "text": self.text,
            "coordinates": {
                "agency": self.coordinates[0],
                "fairness": self.coordinates[1],
                "belonging": self.coordinates[2]
            },
            "mode": self.mode
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrajectoryPoint":
        coords = data["coordinates"]
        if isinstance(coords, dict):
            coordinates = (coords["agency"], coords["fairness"], coords["belonging"])
        else:
            coordinates = tuple(coords)
        return cls(
            timestamp=data["timestamp"],
            text=data["text"],
            coordinates=coordinates,
            mode=data["mode"]
        )


@dataclass
class TrajectoryDynamics:
    """Computed dynamics for a trajectory."""
    total_distance: float
    average_velocity: float  # Distance per time period
    dominant_drift_direction: str  # "agency", "fairness", or "belonging"
    dominant_drift_magnitude: float
    phase_transitions: List[Dict]  # List of {from_mode, to_mode, timestamp}
    narrative_arc: NarrativeArc
    attractor_pull: Optional[str]  # Mode trajectory is moving toward

    def to_dict(self) -> dict:
        return {
            "total_distance": self.total_distance,
            "average_velocity": self.average_velocity,
            "dominant_drift_direction": self.dominant_drift_direction,
            "dominant_drift_magnitude": self.dominant_drift_magnitude,
            "phase_transitions": self.phase_transitions,
            "narrative_arc": self.narrative_arc.value,
            "attractor_pull": self.attractor_pull
        }


@dataclass
class Trajectory:
    """Full trajectory for an entity."""
    entity_name: str
    points: List[TrajectoryPoint] = field(default_factory=list)
    dynamics: Optional[TrajectoryDynamics] = None

    def to_dict(self) -> dict:
        return {
            "entity_name": self.entity_name,
            "points": [p.to_dict() for p in self.points],
            "dynamics": self.dynamics.to_dict() if self.dynamics else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        points = [TrajectoryPoint.from_dict(p) for p in data["points"]]
        dynamics = None
        if data.get("dynamics"):
            d = data["dynamics"]
            dynamics = TrajectoryDynamics(
                total_distance=d["total_distance"],
                average_velocity=d["average_velocity"],
                dominant_drift_direction=d["dominant_drift_direction"],
                dominant_drift_magnitude=d["dominant_drift_magnitude"],
                phase_transitions=d["phase_transitions"],
                narrative_arc=NarrativeArc(d["narrative_arc"]),
                attractor_pull=d.get("attractor_pull")
            )
        return cls(
            entity_name=data["entity_name"],
            points=points,
            dynamics=dynamics
        )


class TrajectoryAnalyzer:
    """
    Analyzes temporal trajectories of entities through the cultural manifold.

    Tracks how narratives evolve over time, detecting phase transitions,
    computing velocities, and classifying overall narrative arcs.
    """

    # Mode centroids for attractor analysis
    MODE_CENTROIDS = {
        "DREAM_POSITIVE": np.array([1.0, 1.0, 0.5]),
        "DREAM_SHADOW": np.array([1.0, -1.0, -0.5]),
        "DREAM_EXIT": np.array([0.0, 0.0, 1.0]),
        "NOISE_OTHER": np.array([0.0, 0.0, 0.0])
    }

    AXIS_NAMES = ["agency", "fairness", "belonging"]

    def __init__(self):
        self.trajectories: Dict[str, Trajectory] = {}

    def add_point(
        self,
        entity: str,
        timestamp: str,
        text: str,
        coords: Tuple[float, float, float],
        mode: str
    ) -> None:
        """
        Add a trajectory point for an entity.

        Args:
            entity: Entity name (e.g., company name, speaker)
            timestamp: ISO format timestamp
            text: The text/statement at this point
            coords: 3D coordinates (agency, fairness, belonging)
            mode: Narrative mode classification
        """
        if entity not in self.trajectories:
            self.trajectories[entity] = Trajectory(entity_name=entity)

        point = TrajectoryPoint(
            timestamp=timestamp,
            text=text,
            coordinates=coords,
            mode=mode
        )
        self.trajectories[entity].points.append(point)

        # Sort points by timestamp
        self.trajectories[entity].points.sort(key=lambda p: p.timestamp)

        logger.debug(f"Added point for {entity} at {timestamp}")

    def get_trajectory(self, entity: str) -> Optional[Trajectory]:
        """
        Get the full trajectory for an entity.

        Args:
            entity: Entity name

        Returns:
            Trajectory object or None if not found
        """
        return self.trajectories.get(entity)

    def list_entities(self) -> List[str]:
        """List all tracked entities."""
        return list(self.trajectories.keys())

    def analyze_trajectory(self, entity: str) -> Optional[TrajectoryDynamics]:
        """
        Compute dynamics for an entity's trajectory.

        Args:
            entity: Entity name

        Returns:
            TrajectoryDynamics object or None if entity not found
        """
        trajectory = self.trajectories.get(entity)
        if not trajectory or len(trajectory.points) < 2:
            logger.warning(f"Not enough points to analyze trajectory for {entity}")
            return None

        points = trajectory.points

        # Compute total distance
        total_distance = self._compute_total_distance(points)

        # Compute average velocity
        average_velocity = self._compute_average_velocity(points, total_distance)

        # Compute dominant drift direction
        drift_direction, drift_magnitude = self._compute_dominant_drift(points)

        # Detect phase transitions
        phase_transitions = self._detect_phase_transitions(points)

        # Classify narrative arc
        narrative_arc = self._classify_narrative_arc(points)

        # Find attractor pull
        attractor_pull = self._find_attractor_pull(points)

        dynamics = TrajectoryDynamics(
            total_distance=total_distance,
            average_velocity=average_velocity,
            dominant_drift_direction=drift_direction,
            dominant_drift_magnitude=drift_magnitude,
            phase_transitions=phase_transitions,
            narrative_arc=narrative_arc,
            attractor_pull=attractor_pull
        )

        trajectory.dynamics = dynamics
        logger.info(f"Analyzed trajectory for {entity}: arc={narrative_arc.value}")

        return dynamics

    def detect_phase_transition(self, entity: str) -> List[Dict]:
        """
        Find mode changes in an entity's trajectory.

        Args:
            entity: Entity name

        Returns:
            List of phase transitions with from_mode, to_mode, timestamp
        """
        trajectory = self.trajectories.get(entity)
        if not trajectory or len(trajectory.points) < 2:
            return []

        return self._detect_phase_transitions(trajectory.points)

    def compute_velocity(self, entity: str) -> Optional[List[Dict]]:
        """
        Compute rate of change over time for an entity.

        Args:
            entity: Entity name

        Returns:
            List of velocity measurements between consecutive points
        """
        trajectory = self.trajectories.get(entity)
        if not trajectory or len(trajectory.points) < 2:
            return None

        velocities = []
        points = trajectory.points

        for i in range(1, len(points)):
            prev_point = points[i - 1]
            curr_point = points[i]

            # Compute distance
            prev_coords = np.array(prev_point.coordinates)
            curr_coords = np.array(curr_point.coordinates)
            distance = float(np.linalg.norm(curr_coords - prev_coords))

            # Compute time difference in hours
            try:
                prev_time = datetime.fromisoformat(prev_point.timestamp.replace('Z', '+00:00'))
                curr_time = datetime.fromisoformat(curr_point.timestamp.replace('Z', '+00:00'))
                time_diff = (curr_time - prev_time).total_seconds() / 3600.0
            except (ValueError, TypeError):
                time_diff = 1.0  # Default to 1 hour if parsing fails

            velocity = distance / max(time_diff, 0.001)  # Avoid division by zero

            # Direction of movement
            direction = curr_coords - prev_coords
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)

            velocities.append({
                "from_timestamp": prev_point.timestamp,
                "to_timestamp": curr_point.timestamp,
                "distance": distance,
                "time_hours": time_diff,
                "velocity": velocity,
                "direction": {
                    "agency": float(direction[0]),
                    "fairness": float(direction[1]),
                    "belonging": float(direction[2])
                }
            })

        return velocities

    def find_attractor_pull(self, entity: str) -> Optional[str]:
        """
        Determine which mode the trajectory is moving toward.

        Args:
            entity: Entity name

        Returns:
            Mode name that trajectory is gravitating toward, or None
        """
        trajectory = self.trajectories.get(entity)
        if not trajectory or len(trajectory.points) < 2:
            return None

        return self._find_attractor_pull(trajectory.points)

    def _compute_total_distance(self, points: List[TrajectoryPoint]) -> float:
        """Compute total distance traveled through manifold."""
        total = 0.0
        for i in range(1, len(points)):
            prev_coords = np.array(points[i - 1].coordinates)
            curr_coords = np.array(points[i].coordinates)
            total += float(np.linalg.norm(curr_coords - prev_coords))
        return total

    def _compute_average_velocity(
        self,
        points: List[TrajectoryPoint],
        total_distance: float
    ) -> float:
        """Compute average velocity (distance per hour)."""
        if len(points) < 2:
            return 0.0

        try:
            start_time = datetime.fromisoformat(points[0].timestamp.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(points[-1].timestamp.replace('Z', '+00:00'))
            total_hours = (end_time - start_time).total_seconds() / 3600.0
        except (ValueError, TypeError):
            total_hours = len(points) - 1  # Default to 1 hour per step

        return total_distance / max(total_hours, 0.001)

    def _compute_dominant_drift(
        self,
        points: List[TrajectoryPoint]
    ) -> Tuple[str, float]:
        """Compute which axis changed most overall."""
        if len(points) < 2:
            return "agency", 0.0

        start_coords = np.array(points[0].coordinates)
        end_coords = np.array(points[-1].coordinates)
        delta = end_coords - start_coords

        # Find axis with largest absolute change
        abs_delta = np.abs(delta)
        max_idx = int(np.argmax(abs_delta))

        return self.AXIS_NAMES[max_idx], float(delta[max_idx])

    def _detect_phase_transitions(self, points: List[TrajectoryPoint]) -> List[Dict]:
        """Detect mode changes (phase transitions)."""
        transitions = []

        for i in range(1, len(points)):
            if points[i].mode != points[i - 1].mode:
                transitions.append({
                    "from_mode": points[i - 1].mode,
                    "to_mode": points[i].mode,
                    "timestamp": points[i].timestamp,
                    "from_text": points[i - 1].text[:100],
                    "to_text": points[i].text[:100]
                })

        return transitions

    def _classify_narrative_arc(self, points: List[TrajectoryPoint]) -> NarrativeArc:
        """
        Classify the overall shape of the trajectory.

        Uses the "agency" axis as the primary indicator of narrative fortune,
        where high agency = positive narrative position.
        """
        if len(points) < 2:
            return NarrativeArc.STEADY

        # Extract agency values over time
        agency_values = [p.coordinates[0] for p in points]

        # Compute overall trend
        start_agency = agency_values[0]
        end_agency = agency_values[-1]
        mid_idx = len(agency_values) // 2
        mid_agency = agency_values[mid_idx] if mid_idx > 0 else start_agency

        # Compute variance to detect chaos
        variance = float(np.var(agency_values))
        mean_change = abs(end_agency - start_agency) / max(len(points), 1)

        # Thresholds
        MOVEMENT_THRESHOLD = 0.3
        VARIANCE_THRESHOLD = 0.5

        # High variance relative to movement indicates chaos
        if variance > VARIANCE_THRESHOLD and mean_change < 0.1:
            return NarrativeArc.CHAOTIC

        # Little overall movement
        if abs(end_agency - start_agency) < MOVEMENT_THRESHOLD:
            return NarrativeArc.STEADY

        # Check for fall and redemption pattern
        if mid_agency < start_agency and end_agency > mid_agency:
            if (start_agency - mid_agency) > MOVEMENT_THRESHOLD:
                return NarrativeArc.FALL_AND_REDEMPTION

        # Simple rise or fall
        if end_agency > start_agency + MOVEMENT_THRESHOLD:
            return NarrativeArc.RISE
        elif end_agency < start_agency - MOVEMENT_THRESHOLD:
            return NarrativeArc.FALL

        return NarrativeArc.STEADY

    def _find_attractor_pull(self, points: List[TrajectoryPoint]) -> Optional[str]:
        """
        Determine which mode centroid the trajectory is moving toward.

        Computes the direction of recent movement and finds which mode
        centroid is most aligned with that direction.
        """
        if len(points) < 2:
            return None

        # Use last few points to determine direction
        lookback = min(3, len(points) - 1)
        recent_points = points[-lookback - 1:]

        # Compute average direction of recent movement
        directions = []
        for i in range(1, len(recent_points)):
            prev = np.array(recent_points[i - 1].coordinates)
            curr = np.array(recent_points[i].coordinates)
            diff = curr - prev
            if np.linalg.norm(diff) > 0.01:
                directions.append(diff / np.linalg.norm(diff))

        if not directions:
            return None

        avg_direction = np.mean(directions, axis=0)
        if np.linalg.norm(avg_direction) < 0.01:
            return None

        avg_direction = avg_direction / np.linalg.norm(avg_direction)

        # Current position
        current_pos = np.array(points[-1].coordinates)

        # Find which mode centroid we're moving toward
        best_mode = None
        best_alignment = -2.0

        for mode, centroid in self.MODE_CENTROIDS.items():
            # Vector from current position to mode centroid
            to_centroid = centroid - current_pos
            if np.linalg.norm(to_centroid) < 0.01:
                continue
            to_centroid = to_centroid / np.linalg.norm(to_centroid)

            # Alignment with movement direction
            alignment = float(np.dot(avg_direction, to_centroid))

            if alignment > best_alignment:
                best_alignment = alignment
                best_mode = mode

        # Only return if alignment is positive (actually moving toward it)
        return best_mode if best_alignment > 0.1 else None

    def save_to_json(self, filepath: str) -> None:
        """
        Save all trajectories to a JSON file.

        Args:
            filepath: Path to output JSON file
        """
        data = {
            "trajectories": {
                entity: traj.to_dict()
                for entity, traj in self.trajectories.items()
            },
            "metadata": {
                "saved_at": datetime.utcnow().isoformat(),
                "entity_count": len(self.trajectories)
            }
        }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.trajectories)} trajectories to {filepath}")

    def load_from_json(self, filepath: str) -> None:
        """
        Load trajectories from a JSON file.

        Args:
            filepath: Path to input JSON file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.trajectories = {}
        for entity, traj_data in data.get("trajectories", {}).items():
            self.trajectories[entity] = Trajectory.from_dict(traj_data)

        logger.info(f"Loaded {len(self.trajectories)} trajectories from {filepath}")

    def clear(self, entity: Optional[str] = None) -> None:
        """
        Clear trajectory data.

        Args:
            entity: If provided, clear only that entity. Otherwise clear all.
        """
        if entity:
            if entity in self.trajectories:
                del self.trajectories[entity]
                logger.info(f"Cleared trajectory for {entity}")
        else:
            self.trajectories = {}
            logger.info("Cleared all trajectories")


# Singleton instance
_trajectory_analyzer = None


def get_trajectory_analyzer() -> TrajectoryAnalyzer:
    """Get singleton trajectory analyzer instance."""
    global _trajectory_analyzer
    if _trajectory_analyzer is None:
        _trajectory_analyzer = TrajectoryAnalyzer()
    return _trajectory_analyzer


# -----------------------------------------------------------------------------
# Test/Demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Temporal Trajectory Tracking - Demo")
    print("=" * 60)

    # Create analyzer
    analyzer = TrajectoryAnalyzer()

    # Simulate a startup's communications over time
    # This startup goes through: optimistic launch -> crisis -> recovery

    startup_data = [
        {
            "timestamp": "2024-01-15T10:00:00Z",
            "text": "We're thrilled to announce our Series A funding! The future is bright.",
            "coords": (0.8, 0.6, 0.7),
            "mode": "DREAM_POSITIVE"
        },
        {
            "timestamp": "2024-03-20T14:30:00Z",
            "text": "Our team is growing fast. We're hiring across all departments!",
            "coords": (0.9, 0.7, 0.8),
            "mode": "DREAM_POSITIVE"
        },
        {
            "timestamp": "2024-06-10T09:00:00Z",
            "text": "We're facing some unexpected challenges in scaling our infrastructure.",
            "coords": (0.4, 0.3, 0.4),
            "mode": "DREAM_SHADOW"
        },
        {
            "timestamp": "2024-08-05T16:45:00Z",
            "text": "Difficult decisions ahead. We must restructure to survive.",
            "coords": (0.1, -0.2, 0.2),
            "mode": "DREAM_SHADOW"
        },
        {
            "timestamp": "2024-09-15T11:00:00Z",
            "text": "We've learned hard lessons. New leadership is bringing fresh perspective.",
            "coords": (0.3, 0.1, 0.5),
            "mode": "DREAM_EXIT"
        },
        {
            "timestamp": "2024-11-20T10:00:00Z",
            "text": "Proud to announce our pivot has paid off. Revenue is up 200%!",
            "coords": (0.7, 0.5, 0.6),
            "mode": "DREAM_POSITIVE"
        },
        {
            "timestamp": "2025-01-10T09:30:00Z",
            "text": "Starting the new year strong. Our community has never been more engaged.",
            "coords": (0.85, 0.65, 0.75),
            "mode": "DREAM_POSITIVE"
        }
    ]

    # Add points to trajectory
    print("\n[1] Adding trajectory points for 'TechStartup Inc.'")
    for point in startup_data:
        analyzer.add_point(
            entity="TechStartup Inc.",
            timestamp=point["timestamp"],
            text=point["text"],
            coords=point["coords"],
            mode=point["mode"]
        )

    # Also add a second entity for demonstration
    competitor_data = [
        {
            "timestamp": "2024-01-01T10:00:00Z",
            "text": "Business as usual. Steady growth continues.",
            "coords": (0.5, 0.5, 0.5),
            "mode": "NOISE_OTHER"
        },
        {
            "timestamp": "2024-06-01T10:00:00Z",
            "text": "Mid-year report shows consistent performance.",
            "coords": (0.55, 0.52, 0.48),
            "mode": "NOISE_OTHER"
        },
        {
            "timestamp": "2025-01-01T10:00:00Z",
            "text": "Another stable year in the books.",
            "coords": (0.52, 0.51, 0.49),
            "mode": "NOISE_OTHER"
        }
    ]

    print("[1] Adding trajectory points for 'SteadyCorp LLC'")
    for point in competitor_data:
        analyzer.add_point(
            entity="SteadyCorp LLC",
            timestamp=point["timestamp"],
            text=point["text"],
            coords=point["coords"],
            mode=point["mode"]
        )

    # List entities
    print(f"\n[2] Tracked entities: {analyzer.list_entities()}")

    # Analyze trajectories
    print("\n[3] Analyzing TechStartup Inc. trajectory...")
    dynamics = analyzer.analyze_trajectory("TechStartup Inc.")
    if dynamics:
        print(f"    Total distance traveled: {dynamics.total_distance:.3f}")
        print(f"    Average velocity: {dynamics.average_velocity:.4f} units/hour")
        print(f"    Dominant drift: {dynamics.dominant_drift_direction} ({dynamics.dominant_drift_magnitude:+.3f})")
        print(f"    Narrative arc: {dynamics.narrative_arc.value}")
        print(f"    Attractor pull: {dynamics.attractor_pull}")
        print(f"    Phase transitions: {len(dynamics.phase_transitions)}")
        for t in dynamics.phase_transitions:
            print(f"      - {t['from_mode']} -> {t['to_mode']} at {t['timestamp']}")

    print("\n[4] Analyzing SteadyCorp LLC trajectory...")
    dynamics2 = analyzer.analyze_trajectory("SteadyCorp LLC")
    if dynamics2:
        print(f"    Total distance traveled: {dynamics2.total_distance:.3f}")
        print(f"    Narrative arc: {dynamics2.narrative_arc.value}")
        print(f"    Phase transitions: {len(dynamics2.phase_transitions)}")

    # Compute velocity
    print("\n[5] Velocity analysis for TechStartup Inc.:")
    velocities = analyzer.compute_velocity("TechStartup Inc.")
    if velocities:
        for v in velocities[:3]:  # Show first 3
            print(f"    {v['from_timestamp'][:10]} -> {v['to_timestamp'][:10]}: "
                  f"velocity={v['velocity']:.4f}")

    # Detect phase transitions
    print("\n[6] Phase transitions for TechStartup Inc.:")
    transitions = analyzer.detect_phase_transition("TechStartup Inc.")
    for t in transitions:
        print(f"    {t['from_mode']} -> {t['to_mode']}")
        print(f"      From: \"{t['from_text'][:50]}...\"")
        print(f"      To:   \"{t['to_text'][:50]}...\"")

    # Find attractor pull
    print("\n[7] Attractor pull analysis:")
    pull = analyzer.find_attractor_pull("TechStartup Inc.")
    print(f"    TechStartup Inc. is moving toward: {pull}")
    pull2 = analyzer.find_attractor_pull("SteadyCorp LLC")
    print(f"    SteadyCorp LLC is moving toward: {pull2}")

    # Save to JSON
    print("\n[8] Saving trajectories to JSON...")
    analyzer.save_to_json("/tmp/trajectory_demo.json")

    # Load from JSON
    print("[9] Loading trajectories from JSON...")
    analyzer2 = TrajectoryAnalyzer()
    analyzer2.load_from_json("/tmp/trajectory_demo.json")
    print(f"    Loaded entities: {analyzer2.list_entities()}")

    # Get full trajectory
    print("\n[10] Full trajectory for TechStartup Inc.:")
    traj = analyzer.get_trajectory("TechStartup Inc.")
    if traj:
        print(f"    Entity: {traj.entity_name}")
        print(f"    Points: {len(traj.points)}")
        print(f"    Has dynamics: {traj.dynamics is not None}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
