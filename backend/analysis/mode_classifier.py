"""
Enhanced Mode Classification

Replaces the simple 4-mode classification with a hierarchical 12-mode taxonomy
plus probabilistic mode membership.

AXIS NAMING (January 2026):
The "fairness" axis has been renamed to "perceived_justice" in API responses.
Mode descriptions below use "perceived justice" for accuracy.
Internal coordinate order: [agency, fairness(=perceived_justice), belonging]

Modes:
├── POSITIVE
│   ├── HEROIC: High agency + high perceived justice + moderate belonging
│   ├── COMMUNAL: Moderate agency + high perceived justice + high belonging
│   └── TRANSCENDENT: Low agency + high perceived justice + high belonging
├── SHADOW
│   ├── CYNICAL_ACHIEVER: High agency + low perceived justice
│   ├── VICTIM: Low agency + low perceived justice
│   └── PARANOID: Moderate agency + very low perceived justice + low belonging
├── EXIT
│   ├── SPIRITUAL: Low agency + neutral perceived justice + high belonging
│   ├── SOCIAL: Moderate agency + neutral perceived justice + low belonging
│   └── PROTEST: Moderate agency + low perceived justice + moderate belonging
├── AMBIVALENT
│   ├── CONFLICTED: High variance across axes
│   ├── TRANSITIONAL: Near mode boundaries
│   └── NEUTRAL: All axes near zero
└── NOISE
    └── UNCLASSIFIABLE: Very low confidence

"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy.stats import multivariate_normal


@dataclass
class ModeProfile:
    """Profile for a narrative mode."""
    name: str
    category: str  # POSITIVE, SHADOW, EXIT, AMBIVALENT, NOISE
    centroid: np.ndarray  # [agency, perceived_justice(internal:fairness), belonging]
    covariance: np.ndarray  # 3x3 covariance matrix
    description: str


# Define mode profiles with centroids and covariances
MODE_PROFILES = {
    # POSITIVE modes
    "HEROIC": ModeProfile(
        name="HEROIC",
        category="POSITIVE",
        centroid=np.array([1.5, 1.0, 0.5]),
        covariance=np.array([[0.3, 0.1, 0.0], [0.1, 0.3, 0.1], [0.0, 0.1, 0.3]]),
        description="High agency, high perceived justice, moderate belonging - classic hero narrative"
    ),
    "COMMUNAL": ModeProfile(
        name="COMMUNAL",
        category="POSITIVE",
        centroid=np.array([0.5, 1.0, 1.5]),
        covariance=np.array([[0.3, 0.1, 0.0], [0.1, 0.3, 0.1], [0.0, 0.1, 0.3]]),
        description="Moderate agency, high perceived justice, high belonging - community-centered"
    ),
    "TRANSCENDENT": ModeProfile(
        name="TRANSCENDENT",
        category="POSITIVE",
        centroid=np.array([-0.5, 0.8, 1.2]),
        covariance=np.array([[0.4, 0.0, 0.0], [0.0, 0.4, 0.1], [0.0, 0.1, 0.4]]),
        description="Low agency, high perceived justice, high belonging - spiritual acceptance"
    ),

    # SHADOW modes
    "CYNICAL_ACHIEVER": ModeProfile(
        name="CYNICAL_ACHIEVER",
        category="SHADOW",
        centroid=np.array([1.2, -0.8, -0.2]),
        covariance=np.array([[0.3, -0.1, 0.0], [-0.1, 0.3, 0.0], [0.0, 0.0, 0.3]]),
        description="High agency, low perceived justice - succeeds despite unjust system"
    ),
    "VICTIM": ModeProfile(
        name="VICTIM",
        category="SHADOW",
        centroid=np.array([-0.5, -1.0, -0.3]),
        covariance=np.array([[0.4, 0.1, 0.0], [0.1, 0.3, 0.0], [0.0, 0.0, 0.4]]),
        description="Low agency, low perceived justice - powerless in unjust system"
    ),
    "PARANOID": ModeProfile(
        name="PARANOID",
        category="SHADOW",
        centroid=np.array([0.3, -1.5, -0.8]),
        covariance=np.array([[0.4, 0.0, 0.0], [0.0, 0.3, 0.1], [0.0, 0.1, 0.3]]),
        description="Moderate agency, very low perceived justice, low belonging - conspiracy mindset"
    ),

    # EXIT modes
    "SPIRITUAL_EXIT": ModeProfile(
        name="SPIRITUAL_EXIT",
        category="EXIT",
        centroid=np.array([-0.8, 0.0, 1.0]),
        covariance=np.array([[0.3, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.3]]),
        description="Low agency, neutral perceived justice, high belonging - spiritual withdrawal"
    ),
    "SOCIAL_EXIT": ModeProfile(
        name="SOCIAL_EXIT",
        category="EXIT",
        centroid=np.array([0.5, 0.0, -1.0]),
        covariance=np.array([[0.4, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.3]]),
        description="Moderate agency, neutral perceived justice, low belonging - social withdrawal"
    ),
    "PROTEST_EXIT": ModeProfile(
        name="PROTEST_EXIT",
        category="EXIT",
        centroid=np.array([0.6, -0.5, 0.4]),
        covariance=np.array([[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]]),
        description="Moderate agency, low perceived justice, moderate belonging - protest/rebellion"
    ),

    # AMBIVALENT modes
    "CONFLICTED": ModeProfile(
        name="CONFLICTED",
        category="AMBIVALENT",
        centroid=np.array([0.0, 0.0, 0.0]),  # Center, but high variance
        covariance=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        description="High variance - torn between narratives"
    ),
    "TRANSITIONAL": ModeProfile(
        name="TRANSITIONAL",
        category="AMBIVALENT",
        centroid=np.array([0.3, 0.3, 0.3]),  # Slight positive, but near boundaries
        covariance=np.array([[0.5, 0.2, 0.2], [0.2, 0.5, 0.2], [0.2, 0.2, 0.5]]),
        description="Near mode boundaries - in transition"
    ),
    "NEUTRAL": ModeProfile(
        name="NEUTRAL",
        category="AMBIVALENT",
        centroid=np.array([0.0, 0.0, 0.0]),
        covariance=np.array([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]]),
        description="All axes near zero - minimal narrative signal"
    ),
}


class EnhancedModeClassifier:
    """
    Probabilistic mode classifier using Gaussian mixture components.
    """

    def __init__(self):
        self.mode_profiles = MODE_PROFILES
        self._build_distributions()

    def _build_distributions(self):
        """Build multivariate normal distributions for each mode."""
        self.distributions = {}
        for name, profile in self.mode_profiles.items():
            self.distributions[name] = multivariate_normal(
                mean=profile.centroid,
                cov=profile.covariance,
                allow_singular=True
            )

    def classify(self, coords: np.ndarray) -> Dict:
        """
        Classify coordinates into modes with probabilities.

        Args:
            coords: Array of shape (3,) with [agency, fairness, belonging]

        Returns:
            Dict with mode probabilities, primary/secondary modes, etc.
        """
        coords = np.array(coords).flatten()

        # Calculate likelihood for each mode
        likelihoods = {}
        for name, dist in self.distributions.items():
            likelihoods[name] = dist.pdf(coords)

        # Normalize to probabilities
        total = sum(likelihoods.values())
        if total > 0:
            probabilities = {k: v / total for k, v in likelihoods.items()}
        else:
            probabilities = {k: 1.0 / len(likelihoods) for k in likelihoods}

        # Sort by probability
        sorted_probs = sorted(probabilities.items(), key=lambda x: -x[1])

        primary_mode = sorted_probs[0][0]
        primary_prob = sorted_probs[0][1]
        secondary_mode = sorted_probs[1][0] if len(sorted_probs) > 1 else None
        secondary_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0

        # Get category
        category = self.mode_profiles[primary_mode].category

        # Calculate distances to mode boundaries
        boundary_distances = self._calculate_boundary_distances(coords)

        # Calculate confidence
        confidence = self._calculate_confidence(primary_prob, secondary_prob, coords)

        return {
            "primary_mode": primary_mode,
            "primary_probability": round(primary_prob, 4),
            "secondary_mode": secondary_mode,
            "secondary_probability": round(secondary_prob, 4),
            "category": category,
            "mode_probabilities": {k: round(v, 4) for k, v in sorted_probs[:5]},
            "boundary_distances": boundary_distances,
            "confidence": round(confidence, 4),
            "description": self.mode_profiles[primary_mode].description
        }

    def _calculate_boundary_distances(self, coords: np.ndarray) -> Dict[str, float]:
        """Calculate distance to nearest mode boundaries."""
        distances = {}

        # Distance to each mode centroid
        for name, profile in self.mode_profiles.items():
            dist = np.linalg.norm(coords - profile.centroid)
            distances[f"to_{name}"] = round(dist, 3)

        return distances

    def _calculate_confidence(self, primary_prob: float, secondary_prob: float,
                             coords: np.ndarray) -> float:
        """
        Calculate overall classification confidence.

        Higher when:
        - Primary probability is much higher than secondary
        - Not near the center (neutral zone)
        - Not near category boundaries
        """
        # Probability gap
        prob_gap = primary_prob - secondary_prob

        # Distance from center
        center_distance = np.linalg.norm(coords)
        center_factor = min(center_distance / 1.5, 1.0)  # Normalize to ~1 at distance 1.5

        # Combined confidence
        confidence = 0.5 * prob_gap + 0.5 * center_factor

        return max(0.0, min(1.0, confidence))

    def get_mode_description(self, mode_name: str) -> str:
        """Get description for a mode."""
        if mode_name in self.mode_profiles:
            return self.mode_profiles[mode_name].description
        return "Unknown mode"

    def get_all_modes(self) -> List[Dict]:
        """Get information about all modes."""
        return [
            {
                "name": name,
                "category": profile.category,
                "centroid": profile.centroid.tolist(),
                "description": profile.description
            }
            for name, profile in self.mode_profiles.items()
        ]


# Legacy compatibility: map to old 4-mode system
def legacy_mode(enhanced_mode: str) -> str:
    """Map enhanced mode to legacy 4-mode system."""
    category_map = {
        "POSITIVE": "DREAM_POSITIVE",
        "SHADOW": "DREAM_SHADOW",
        "EXIT": "DREAM_EXIT",
        "AMBIVALENT": "NOISE_OTHER",
        "NOISE": "NOISE_OTHER"
    }

    profile = MODE_PROFILES.get(enhanced_mode)
    if profile:
        return category_map.get(profile.category, "NOISE_OTHER")
    return "NOISE_OTHER"


# Singleton instance
_classifier = None


def get_mode_classifier() -> EnhancedModeClassifier:
    """Get singleton mode classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = EnhancedModeClassifier()
    return _classifier


def classify_coordinates(agency: float, fairness: float, belonging: float) -> Dict:
    """
    Convenience function to classify coordinates.

    Args:
        agency: Agency score (-2 to 2)
        fairness: Perceived Justice score (-2 to 2). Parameter named 'fairness' for
                  backward compatibility, but represents 'perceived_justice'.
        belonging: Belonging score (-2 to 2)

    Returns:
        Classification result with probabilities and metadata
    """
    classifier = get_mode_classifier()
    return classifier.classify(np.array([agency, fairness, belonging]))


# Test
if __name__ == "__main__":
    # Test classifications
    test_cases = [
        ("High agency hero", [1.5, 1.0, 0.5]),
        ("Victim narrative", [-0.5, -1.0, -0.3]),
        ("Conspiracy paranoid", [0.3, -1.5, -0.8]),
        ("Spiritual acceptance", [-0.8, 0.0, 1.0]),
        ("Neutral center", [0.0, 0.0, 0.0]),
        ("Platform capitalism", [1.14, -1.09, -0.40]),
        ("Conservative values", [-0.30, 0.63, 1.30]),
    ]

    classifier = get_mode_classifier()

    for name, coords in test_cases:
        result = classifier.classify(np.array(coords))
        print(f"\n{name}: {coords}")
        print(f"  Primary: {result['primary_mode']} ({result['primary_probability']:.1%})")
        print(f"  Secondary: {result['secondary_mode']} ({result['secondary_probability']:.1%})")
        print(f"  Category: {result['category']}")
        print(f"  Confidence: {result['confidence']:.1%}")
