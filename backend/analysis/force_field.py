"""
Force Field Analysis: Attractors and Detractors
================================================

This module extends the Cultural Soliton Observatory with force field analysis,
adding two new dimensions to the narrative topology:

- Attractor Strength (A+): Pull toward positive states
- Detractor Strength (D-): Push from negative states

Plus identification of specific attractor targets and detractor sources.

Attractor Targets:
    AUTONOMY    - Self-determination, freedom, control
    COMMUNITY   - Belonging, connection, togetherness
    JUSTICE     - Fairness, equity, rightness
    MEANING     - Purpose, transcendence, significance
    SECURITY    - Stability, safety, predictability
    RECOGNITION - Visibility, validation, appreciation

Detractor Sources:
    OPPRESSION      - Control, domination, powerlessness
    ISOLATION       - Alienation, abandonment, loneliness
    INJUSTICE       - Unfairness, corruption, betrayal
    MEANINGLESSNESS - Futility, nihilism, absurdity
    INSTABILITY     - Chaos, threat, unpredictability
    INVISIBILITY    - Being ignored, dismissed, devalued
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# Canonical targets
ATTRACTOR_TARGETS = ["AUTONOMY", "COMMUNITY", "JUSTICE", "MEANING", "SECURITY", "RECOGNITION"]
DETRACTOR_SOURCES = ["OPPRESSION", "ISOLATION", "INJUSTICE", "MEANINGLESSNESS", "INSTABILITY", "INVISIBILITY"]

# Force field quadrant definitions
FORCE_QUADRANTS = {
    "ACTIVE_TRANSFORMATION": {
        "attractor_min": 0.5,
        "detractor_min": 0.5,
        "description": "Moving from something bad toward something good",
        "energy": "high",
    },
    "PURE_ASPIRATION": {
        "attractor_min": 0.5,
        "detractor_max": 0.5,
        "description": "Drawn to positive without fleeing negative",
        "energy": "moderate",
    },
    "PURE_ESCAPE": {
        "attractor_max": 0.5,
        "detractor_min": 0.5,
        "description": "Fleeing negative without clear positive direction",
        "energy": "moderate",
    },
    "STASIS": {
        "attractor_max": 0.5,
        "detractor_max": 0.5,
        "description": "No strong forces acting on narrative",
        "energy": "low",
    },
}


class ForceFieldAnalyzer:
    """
    Analyzes the attractor/detractor force field of narratives.

    This extends the 3D manifold (Agency, Perceived Justice, Belonging) with
    two additional dimensions that capture the DIRECTIONALITY of narratives:
    - What they're moving TOWARD (attractor)
    - What they're moving AWAY FROM (detractor)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = None
        self.scaler = StandardScaler()

        # Models for aggregate scores
        self.attractor_model = Ridge(alpha=1.0)
        self.detractor_model = Ridge(alpha=1.0)

        # Models for specific targets (multi-output)
        self.attractor_target_models = {target: Ridge(alpha=1.0) for target in ATTRACTOR_TARGETS}
        self.detractor_source_models = {source: Ridge(alpha=1.0) for source in DETRACTOR_SOURCES}

        self.is_trained = False
        self.training_stats = {}

    def _get_encoder(self):
        """Lazy load the sentence transformer."""
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.model_name)
        return self.encoder

    def load_training_data(self, path: str = None) -> List[Dict]:
        """Load training data from JSON file."""
        if path is None:
            path = Path(__file__).parent.parent / "data" / "attractor_detractor_training.json"

        with open(path, "r") as f:
            data = json.load(f)

        return data.get("examples", [])

    def train(self, training_data: List[Dict] = None):
        """
        Train the force field models on labeled examples.

        Each example should have:
        - text: The narrative text
        - attractor_strength: Overall pull toward positive (-2 to +2)
        - detractor_strength: Overall push from negative (-2 to +2)
        - attractor_scores: Dict of {target: score} for specific targets
        - detractor_scores: Dict of {source: score} for specific sources
        """
        if training_data is None:
            training_data = self.load_training_data()

        if not training_data:
            logger.warning("No training data provided for force field analysis")
            return

        encoder = self._get_encoder()

        # Encode all texts
        texts = [ex["text"] for ex in training_data]
        embeddings = encoder.encode(texts)
        X = self.scaler.fit_transform(embeddings)

        # Train aggregate models
        y_attractor = np.array([ex.get("attractor_strength", 0) for ex in training_data])
        y_detractor = np.array([ex.get("detractor_strength", 0) for ex in training_data])

        self.attractor_model.fit(X, y_attractor)
        self.detractor_model.fit(X, y_detractor)

        # Train specific target models
        for target in ATTRACTOR_TARGETS:
            y_target = np.array([
                ex.get("attractor_scores", {}).get(target, 0)
                for ex in training_data
            ])
            if np.any(y_target > 0):  # Only train if we have positive examples
                self.attractor_target_models[target].fit(X, y_target)

        for source in DETRACTOR_SOURCES:
            y_source = np.array([
                ex.get("detractor_scores", {}).get(source, 0)
                for ex in training_data
            ])
            if np.any(y_source > 0):
                self.detractor_source_models[source].fit(X, y_source)

        self.is_trained = True
        self.training_stats = {
            "n_examples": len(training_data),
            "attractor_mean": float(np.mean(y_attractor)),
            "attractor_std": float(np.std(y_attractor)),
            "detractor_mean": float(np.mean(y_detractor)),
            "detractor_std": float(np.std(y_detractor)),
        }

        logger.info(f"Force field analyzer trained on {len(training_data)} examples")

    def analyze(self, text: str, embedding: np.ndarray = None) -> Dict[str, Any]:
        """
        Analyze the force field of a narrative.

        Returns:
            {
                "attractor_strength": float,  # Overall pull toward positive
                "detractor_strength": float,  # Overall push from negative
                "primary_attractor": str,     # Main target being drawn toward
                "primary_detractor": str,     # Main source being fled from
                "attractor_scores": Dict,     # Scores for each attractor target
                "detractor_scores": Dict,     # Scores for each detractor source
                "force_quadrant": str,        # Which quadrant of force space
                "net_force": float,           # Combined force magnitude
                "force_direction": str,       # Toward/Away/Balanced
            }
        """
        if not self.is_trained:
            self.train()

        # Get embedding
        if embedding is None:
            encoder = self._get_encoder()
            embedding = encoder.encode([text])[0]

        X = self.scaler.transform([embedding])

        # Predict aggregate scores
        attractor_strength = float(self.attractor_model.predict(X)[0])
        detractor_strength = float(self.detractor_model.predict(X)[0])

        # Clamp to valid range
        attractor_strength = max(-2, min(2, attractor_strength))
        detractor_strength = max(-2, min(2, detractor_strength))

        # Predict specific targets
        attractor_scores = {}
        for target in ATTRACTOR_TARGETS:
            score = float(self.attractor_target_models[target].predict(X)[0])
            score = max(0, min(1, score))  # Clamp to 0-1
            if score > 0.1:  # Only include significant scores
                attractor_scores[target] = round(score, 3)

        detractor_scores = {}
        for source in DETRACTOR_SOURCES:
            score = float(self.detractor_source_models[source].predict(X)[0])
            score = max(0, min(1, score))
            if score > 0.1:
                detractor_scores[source] = round(score, 3)

        # Identify primary attractor/detractor
        primary_attractor = max(attractor_scores, key=attractor_scores.get) if attractor_scores else None
        primary_detractor = max(detractor_scores, key=detractor_scores.get) if detractor_scores else None

        # Determine force quadrant
        force_quadrant = self._determine_quadrant(attractor_strength, detractor_strength)

        # Calculate net force and direction
        net_force = np.sqrt(attractor_strength**2 + detractor_strength**2)
        if attractor_strength > detractor_strength + 0.3:
            force_direction = "TOWARD"
        elif detractor_strength > attractor_strength + 0.3:
            force_direction = "AWAY"
        else:
            force_direction = "BALANCED"

        return {
            "attractor_strength": round(attractor_strength, 4),
            "detractor_strength": round(detractor_strength, 4),
            "primary_attractor": primary_attractor,
            "primary_detractor": primary_detractor,
            "secondary_attractor": self._get_secondary(attractor_scores, primary_attractor),
            "secondary_detractor": self._get_secondary(detractor_scores, primary_detractor),
            "attractor_scores": attractor_scores,
            "detractor_scores": detractor_scores,
            "force_quadrant": force_quadrant,
            "quadrant_description": FORCE_QUADRANTS.get(force_quadrant, {}).get("description", ""),
            "energy_level": FORCE_QUADRANTS.get(force_quadrant, {}).get("energy", "unknown"),
            "net_force": round(float(net_force), 4),
            "force_direction": force_direction,
        }

    def _determine_quadrant(self, attractor: float, detractor: float) -> str:
        """Determine which force quadrant the narrative falls into."""
        high_a = attractor > 0.5
        high_d = detractor > 0.5

        if high_a and high_d:
            return "ACTIVE_TRANSFORMATION"
        elif high_a and not high_d:
            return "PURE_ASPIRATION"
        elif not high_a and high_d:
            return "PURE_ESCAPE"
        else:
            return "STASIS"

    def _get_secondary(self, scores: Dict, primary: str) -> Optional[str]:
        """Get the second-highest scoring target/source."""
        if not scores or len(scores) < 2:
            return None

        sorted_items = sorted(scores.items(), key=lambda x: -x[1])
        for target, score in sorted_items:
            if target != primary:
                return target
        return None

    def analyze_trajectory_forces(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze force field dynamics across a trajectory of texts.

        Returns force field changes over time, identifying:
        - Attractor shifts
        - Detractor emergence/resolution
        - Energy changes
        - Direction changes
        """
        if not texts:
            return {}

        analyses = [self.analyze(text) for text in texts]

        # Track changes
        attractor_values = [a["attractor_strength"] for a in analyses]
        detractor_values = [a["detractor_strength"] for a in analyses]

        # Calculate trends
        attractor_trend = self._calculate_trend(attractor_values)
        detractor_trend = self._calculate_trend(detractor_values)

        # Identify shifts
        attractor_shifts = []
        detractor_shifts = []

        for i in range(1, len(analyses)):
            prev_a = analyses[i-1]["primary_attractor"]
            curr_a = analyses[i]["primary_attractor"]
            if prev_a != curr_a and curr_a is not None:
                attractor_shifts.append({
                    "index": i,
                    "from": prev_a,
                    "to": curr_a,
                })

            prev_d = analyses[i-1]["primary_detractor"]
            curr_d = analyses[i]["primary_detractor"]
            if prev_d != curr_d:
                detractor_shifts.append({
                    "index": i,
                    "from": prev_d,
                    "to": curr_d,
                })

        # Calculate energy trajectory
        energy_values = [a["net_force"] for a in analyses]
        energy_trend = self._calculate_trend(energy_values)

        return {
            "points": analyses,
            "n_points": len(analyses),
            "attractor_trajectory": {
                "values": attractor_values,
                "trend": attractor_trend,
                "start": attractor_values[0],
                "end": attractor_values[-1],
                "change": attractor_values[-1] - attractor_values[0],
            },
            "detractor_trajectory": {
                "values": detractor_values,
                "trend": detractor_trend,
                "start": detractor_values[0],
                "end": detractor_values[-1],
                "change": detractor_values[-1] - detractor_values[0],
            },
            "energy_trajectory": {
                "values": energy_values,
                "trend": energy_trend,
                "start": energy_values[0],
                "end": energy_values[-1],
            },
            "attractor_shifts": attractor_shifts,
            "detractor_shifts": detractor_shifts,
            "interpretation": self._interpret_trajectory(analyses, attractor_trend, detractor_trend),
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values."""
        if len(values) < 2:
            return "stable"

        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

    def _interpret_trajectory(
        self,
        analyses: List[Dict],
        attractor_trend: str,
        detractor_trend: str
    ) -> str:
        """Generate interpretation of force field trajectory."""
        start_quadrant = analyses[0]["force_quadrant"]
        end_quadrant = analyses[-1]["force_quadrant"]

        if start_quadrant == end_quadrant:
            if attractor_trend == "increasing" and detractor_trend == "decreasing":
                return "Narrative is resolving tensions - moving toward pure aspiration"
            elif attractor_trend == "decreasing" and detractor_trend == "increasing":
                return "Narrative is entering crisis - losing direction while gaining escape urgency"
            elif attractor_trend == "increasing" and detractor_trend == "increasing":
                return "Narrative is energizing - both pull and push forces intensifying"
            elif attractor_trend == "decreasing" and detractor_trend == "decreasing":
                return "Narrative is de-energizing - moving toward stasis"
            else:
                return f"Narrative is stable in {start_quadrant} quadrant"
        else:
            return f"Narrative shifted from {start_quadrant} to {end_quadrant}"


# Singleton instance
_force_field_analyzer = None

def get_force_field_analyzer() -> ForceFieldAnalyzer:
    """Get or create the singleton force field analyzer."""
    global _force_field_analyzer
    if _force_field_analyzer is None:
        _force_field_analyzer = ForceFieldAnalyzer()
        _force_field_analyzer.train()
    return _force_field_analyzer


def analyze_force_field(text: str, embedding: np.ndarray = None) -> Dict[str, Any]:
    """Convenience function to analyze a single text."""
    analyzer = get_force_field_analyzer()
    return analyzer.analyze(text, embedding)


def analyze_trajectory_forces(texts: List[str]) -> Dict[str, Any]:
    """Convenience function to analyze a trajectory."""
    analyzer = get_force_field_analyzer()
    return analyzer.analyze_trajectory_forces(texts)
