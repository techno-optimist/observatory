"""
Semantic Extraction Layer for the Cultural Soliton Observatory.

Moves beyond regex to semantic similarity matching, enabling:
- Paraphrase robustness ("I don't know" ≈ "I'm uncertain")
- Context-aware disambiguation
- Confidence scoring based on method agreement

Author: Observatory Research Team
Version: 1.0.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ExtractionMethod(Enum):
    REGEX = "regex"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class SemanticMatch:
    """Result of semantic similarity matching."""
    dimension: str
    subdimension: str
    score: float
    confidence: float
    matched_prototype: str
    method: ExtractionMethod


@dataclass
class HybridExtractionResult:
    """Combined result from regex and semantic extraction."""
    regex_score: float
    semantic_score: float
    final_score: float
    confidence: float
    agreement: float  # How much regex and semantic agree
    method_used: ExtractionMethod


# Prototype phrases for each dimension
# These represent the "canonical" expression of each coordination dimension
# v2.3: Expanded prototypes and reduced self_agency bias
DIMENSION_PROTOTYPES = {
    "agency": {
        "self_agency": [
            # Core self-agency (specific, not generic)
            "I took deliberate action to change my situation",
            "I exercised my own judgment in making this decision",
            "I am the author of my own choices here",
            "I acted autonomously without external pressure",
            "My personal initiative drove this outcome",
            "I asserted my own will in this matter",
            # Note: Removed generic phrases like "I can" that overlap with many contexts
        ],
        "other_agency": [
            "You can do this",
            "They made the decision",
            "She chose her path",
            "He has control",
            "You have the power to change",
            "Their choice matters",
            "Someone else is responsible for this",
            "Another person caused this to happen",
        ],
        "system_agency": [
            "The system decided",
            "It was determined by policy",
            "The rules dictate",
            "Circumstances forced this",
            "The structure requires",
            "External forces caused",
            "Institutional requirements mandate",
            "Bureaucratic processes determined",
            "The algorithm decided",
            # v2.4: Technical/infrastructure patterns (for technical docs)
            "The configuration requires these settings",
            "The service automatically handles this",
            "Deployment is managed by the pipeline",
            "The framework enforces this pattern",
            "System constraints dictate the architecture",
            "Infrastructure determines the approach",
            "The platform manages resources automatically",
            "Technical requirements specify the format",
        ],
    },
    "justice": {
        "procedural": [
            # v2.4: Expanded with emotional reactions to process fairness
            "The process was fair and transparent",
            "Everyone followed the rules properly",
            "Due process was observed correctly",
            "Fair procedures were used in the decision",
            "Proper channels were followed throughout",
            "Everyone had a chance to be heard",
            "The decision was made through legitimate means",
            # Negative - unfair process
            "The process was rigged against me",
            "They broke the rules to get what they wanted",
            "Unfair procedures were used",
            "They made decisions without consulting us",
            "No one asked for our input",
            "They changed the rules midway through",
        ],
        "distributive": [
            # v2.4: Expanded with emotional reactions to unfair distribution
            "Resources were shared equally among everyone",
            "Everyone got their fair share of the rewards",
            "The distribution was just and equitable",
            "Rewards matched the contributions made",
            "Benefits were allocated fairly to all",
            # Negative - unfair distribution
            "They got more than they deserved",
            "Some people got special treatment",
            "The rewards were not proportional to effort",
            "Others got what should have been mine",
            "Those people have advantages we don't have",
            "Unfair that they get so much while we struggle",
        ],
        "interactional": [
            # v2.4: Focus on TREATMENT BY OTHERS, not just feelings
            # Positive treatment
            "They treated me with dignity and respect",
            "They spoke to me fairly and gave me time",
            "They explained their decision to me properly",
            "They gave me a chance to tell my side",
            "They acknowledged my concerns respectfully",
            # Negative treatment - explicit THEY/SOMEONE did something
            "They disrespected me to my face",
            "Someone treated me rudely and unfairly",
            "They talked down to me condescendingly",
            "They dismissed what I said completely",
            "They humiliated me in front of everyone",
            "They ignored my input entirely",
            "Someone made me feel unwelcome on purpose",
            "They acted like I wasn't even there",
        ],
    },
    "belonging": {
        "ingroup": [
            # v2.4: Strong ingroup markers - collective positive identity
            "We are in this together as a team",
            "Our group is working on this",
            "We succeeded as a team",
            "Our community came together",
            "We share a common purpose",
            "Together we achieved this",
            "Our people understand each other",
            "We support one another",
            # Inclusion and acceptance
            "I am part of this group",
            "They welcomed me warmly",
            "I belong with these people",
            "We have each other's backs",
            "This is my community",
            "I'm one of them now",
            # Collaborative language
            "We built this together",
            "Our shared accomplishment",
            "Working alongside my teammates",
        ],
        "outgroup": [
            # v2.4: Clear outgroup markers - distance and difference
            "They are not like us at all",
            "Those outsiders don't belong here",
            "We don't associate with them",
            "They are the enemy",
            "The others are against us",
            "They don't understand our ways",
            # Exclusion from speaker's perspective
            "I don't belong with those people",
            "I am not one of them",
            "I was excluded from their group",
            "They rejected me",
            "I'm an outsider to them",
            "I don't fit in with that crowd",
            # Social distance
            "There is a wall between us and them",
            "We have nothing in common with those people",
        ],
        "universal": [
            "All humans share this",
            "Everyone deserves",
            "Universal rights",
            "Humanity as a whole",
            "We are all connected",
            "Common human experience",
            # v2.3: Added universal belonging patterns
            "Beyond borders and boundaries",
            "Shared across cultures",
            "What unites us all",
            "The human family",
            "Citizens of the world",
            "Transcends group differences",
        ],
    },
}

# Uncertainty prototypes (for the key finding about presence)
# v2.4: Rewritten to avoid overlap with belonging "I feel" patterns
UNCERTAINTY_PROTOTYPES = {
    "experiential": [
        # Focus on confusion/disorientation rather than feelings
        "I can't make sense of what's happening to me",
        "My own experience confuses me",
        "I don't understand my own reaction",
        "What I'm going through is bewildering",
        "I can't grasp my own state of mind",
        "My inner experience is murky and unclear",
        "I'm confused about what I'm experiencing",
        "This sensation is hard to interpret",
    ],
    "epistemic": [
        # Knowledge/fact uncertainty
        "I don't know the answer to that question",
        "I'm uncertain about whether this is factually correct",
        "The truth of the matter is unclear to me",
        "I lack the knowledge to say for certain",
        "The evidence is ambiguous",
        "I cannot determine if this is accurate",
        "Generally speaking it might be true but I'm not sure",
        "There are notable exceptions that make this uncertain",
    ],
    "moral": [
        "I'm uncertain what the right thing to do is",
        "The ethical path forward is unclear",
        "I'm torn between competing values",
        "I don't know if this is morally justified",
        "The right choice is not obvious here",
        "I'm wrestling with a moral dilemma",
    ],
}


class SemanticExtractor:
    """
    Semantic similarity-based coordination extraction.

    Uses sentence embeddings to match text against prototype phrases,
    enabling paraphrase-robust extraction.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic extractor.

        Args:
            model_name: Sentence transformer model to use
        """
        self.model_name = model_name
        self._model = None
        self._prototype_embeddings: Dict[str, np.ndarray] = {}
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of embeddings."""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._compute_prototype_embeddings()
            self._initialized = True
        except ImportError:
            # Fallback: semantic extraction not available
            self._initialized = False

    def _compute_prototype_embeddings(self):
        """Pre-compute embeddings for all prototype phrases."""
        if self._model is None:
            return

        for dimension, subdims in DIMENSION_PROTOTYPES.items():
            for subdim, phrases in subdims.items():
                key = f"{dimension}.{subdim}"
                embeddings = self._model.encode(phrases, convert_to_numpy=True)
                self._prototype_embeddings[key] = embeddings

        # Also compute uncertainty embeddings
        for utype, phrases in UNCERTAINTY_PROTOTYPES.items():
            key = f"uncertainty.{utype}"
            embeddings = self._model.encode(phrases, convert_to_numpy=True)
            self._prototype_embeddings[key] = embeddings

    def extract(self, text: str) -> Dict[str, SemanticMatch]:
        """
        Extract coordination dimensions using semantic similarity.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary mapping dimension.subdimension to SemanticMatch
        """
        self._ensure_initialized()

        if not self._initialized or self._model is None:
            return {}

        # Encode the input text
        text_embedding = self._model.encode([text], convert_to_numpy=True)[0]

        results = {}

        for key, proto_embeddings in self._prototype_embeddings.items():
            # Compute cosine similarities
            similarities = np.dot(proto_embeddings, text_embedding) / (
                np.linalg.norm(proto_embeddings, axis=1) * np.linalg.norm(text_embedding)
            )

            # Get best match
            best_idx = np.argmax(similarities)
            best_sim = similarities[best_idx]

            # Get the prototype phrase
            parts = key.split(".")
            if parts[0] == "uncertainty":
                prototype = UNCERTAINTY_PROTOTYPES[parts[1]][best_idx]
            else:
                prototype = DIMENSION_PROTOTYPES[parts[0]][parts[1]][best_idx]

            # Convert similarity to score
            # Threshold: similarity > 0.3 indicates meaningful match
            if best_sim > 0.3:
                # Scale to [-1, 1] range
                score = (best_sim - 0.3) / 0.7  # Maps 0.3-1.0 to 0-1
                score = min(score, 1.0)

                # Check for negative prototypes
                if any(neg in prototype.lower() for neg in ["unfair", "rigged", "disrespect", "rude"]):
                    score = -score
            else:
                score = 0.0

            # Confidence based on how much higher than threshold
            confidence = min((best_sim - 0.2) / 0.5, 1.0) if best_sim > 0.2 else 0.0

            results[key] = SemanticMatch(
                dimension=parts[0],
                subdimension=parts[1] if len(parts) > 1 else "",
                score=score,
                confidence=confidence,
                matched_prototype=prototype,
                method=ExtractionMethod.SEMANTIC,
            )

        return results

    def extract_uncertainty_type(self, text: str) -> Tuple[str, float, float]:
        """
        Classify the type of uncertainty expressed.

        Returns:
            (uncertainty_type, score, confidence)
        """
        self._ensure_initialized()

        if not self._initialized or self._model is None:
            return ("unknown", 0.0, 0.0)

        text_embedding = self._model.encode([text], convert_to_numpy=True)[0]

        best_type = "unknown"
        best_score = 0.0
        best_conf = 0.0

        for utype in ["experiential", "epistemic", "moral"]:
            key = f"uncertainty.{utype}"
            if key not in self._prototype_embeddings:
                continue

            proto_embeddings = self._prototype_embeddings[key]
            similarities = np.dot(proto_embeddings, text_embedding) / (
                np.linalg.norm(proto_embeddings, axis=1) * np.linalg.norm(text_embedding)
            )

            max_sim = np.max(similarities)
            if max_sim > best_score:
                best_score = max_sim
                best_type = utype
                best_conf = min((max_sim - 0.3) / 0.4, 1.0) if max_sim > 0.3 else 0.0

        return (best_type, best_score, best_conf)


class HybridExtractor:
    """
    Combines regex and semantic extraction for robust coordination detection.

    Strategy:
    - Use regex for speed on clear cases
    - Fall back to semantic for ambiguous cases
    - Report confidence based on method agreement
    """

    def __init__(self):
        self.semantic = SemanticExtractor()
        self._regex_extractor = None

    def _get_regex_extractor(self):
        """Lazy load the regex extractor."""
        if self._regex_extractor is None:
            from .hierarchical_coordinates import HierarchicalCoordinateExtractor
            self._regex_extractor = HierarchicalCoordinateExtractor()
        return self._regex_extractor

    def extract(self, text: str) -> Dict[str, HybridExtractionResult]:
        """
        Extract coordination using both methods and combine.

        Args:
            text: Input text

        Returns:
            Dictionary of hybrid results per dimension
        """
        # Get regex results
        regex_ext = self._get_regex_extractor()
        regex_coord = regex_ext.extract(text)

        # Get semantic results
        semantic_results = self.semantic.extract(text)

        # Combine results
        hybrid_results = {}

        # Process core dimensions
        for dim in ["agency", "justice", "belonging"]:
            for subdim in self._get_subdims(dim):
                key = f"{dim}.{subdim}"

                # Get regex score
                regex_score = self._get_regex_score(regex_coord, dim, subdim)

                # Get semantic score
                sem_result = semantic_results.get(key)
                semantic_score = sem_result.score if sem_result else 0.0
                semantic_conf = sem_result.confidence if sem_result else 0.0

                # Compute agreement
                if regex_score != 0 and semantic_score != 0:
                    # Both detected something
                    agreement = 1.0 - abs(regex_score - semantic_score)
                elif regex_score == 0 and semantic_score == 0:
                    # Both agree nothing
                    agreement = 1.0
                else:
                    # Disagreement
                    agreement = 0.5

                # Choose method and compute final score
                if agreement > 0.7:
                    # High agreement: average
                    final_score = (regex_score + semantic_score) / 2
                    method = ExtractionMethod.HYBRID
                    confidence = min(1.0, agreement * max(0.5, semantic_conf))
                elif abs(regex_score) > abs(semantic_score):
                    # Regex stronger: use regex
                    final_score = regex_score
                    method = ExtractionMethod.REGEX
                    confidence = 0.6 * agreement
                else:
                    # Semantic stronger: use semantic
                    final_score = semantic_score
                    method = ExtractionMethod.SEMANTIC
                    confidence = semantic_conf * agreement

                hybrid_results[key] = HybridExtractionResult(
                    regex_score=regex_score,
                    semantic_score=semantic_score,
                    final_score=final_score,
                    confidence=confidence,
                    agreement=agreement,
                    method_used=method,
                )

        return hybrid_results

    def _get_subdims(self, dim: str) -> List[str]:
        """Get subdimensions for a dimension."""
        if dim == "agency":
            return ["self_agency", "other_agency", "system_agency"]
        elif dim == "justice":
            return ["procedural", "distributive", "interactional"]
        elif dim == "belonging":
            return ["ingroup", "outgroup", "universal"]
        return []

    def _get_regex_score(self, coord, dim: str, subdim: str) -> float:
        """Extract regex score for a specific subdimension."""
        try:
            if dim == "agency":
                if subdim == "self_agency":
                    return coord.core.agency.self_agency
                elif subdim == "other_agency":
                    return coord.core.agency.other_agency
                elif subdim == "system_agency":
                    return coord.core.agency.system_agency
            elif dim == "justice":
                if subdim == "procedural":
                    return coord.core.justice.procedural
                elif subdim == "distributive":
                    return coord.core.justice.distributive
                elif subdim == "interactional":
                    return coord.core.justice.interactional
            elif dim == "belonging":
                if subdim == "ingroup":
                    return coord.core.belonging.ingroup
                elif subdim == "outgroup":
                    return coord.core.belonging.outgroup
                elif subdim == "universal":
                    return coord.core.belonging.universal
        except Exception:
            pass
        return 0.0


class ConfidenceCalibrator:
    """
    v2.3: Confidence calibration using Platt scaling and isotonic regression.

    Problem: ECE = 0.220, under-confidence in high bins (actual 98% vs reported 70%)

    Solution: Learn a mapping from raw confidence to calibrated confidence using
    validation data. Supports:
    - Platt scaling (logistic regression)
    - Isotonic regression (monotonic piecewise linear)
    - Temperature scaling (simple division)

    Usage:
        calibrator = ConfidenceCalibrator()
        calibrator.fit(raw_confidences, actual_outcomes)
        calibrated = calibrator.calibrate(new_confidence)
    """

    def __init__(self, method: str = "isotonic"):
        """
        Initialize confidence calibrator.

        Args:
            method: Calibration method - "platt", "isotonic", or "temperature"
        """
        self.method = method
        self._fitted = False
        self._temperature = 1.0  # For temperature scaling
        self._platt_a = 0.0  # For Platt scaling: P = 1 / (1 + exp(a*x + b))
        self._platt_b = 0.0
        self._isotonic_x = None  # For isotonic regression
        self._isotonic_y = None

    def fit(
        self,
        raw_confidences: List[float],
        actual_outcomes: List[bool],
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Fit calibration mapping from validation data.

        Args:
            raw_confidences: Raw confidence scores from the model
            actual_outcomes: Whether the model was actually correct (True/False)
            n_bins: Number of bins for reliability diagram

        Returns:
            Dict with ECE before/after calibration
        """
        confidences = np.array(raw_confidences)
        outcomes = np.array(actual_outcomes, dtype=float)

        if len(confidences) < 10:
            return {"error": "Need at least 10 samples for calibration"}

        # Compute ECE before calibration
        ece_before = self._compute_ece(confidences, outcomes, n_bins)

        if self.method == "temperature":
            self._fit_temperature(confidences, outcomes)
        elif self.method == "platt":
            self._fit_platt(confidences, outcomes)
        elif self.method == "isotonic":
            self._fit_isotonic(confidences, outcomes)

        self._fitted = True

        # Compute ECE after calibration
        calibrated = np.array([self.calibrate(c) for c in confidences])
        ece_after = self._compute_ece(calibrated, outcomes, n_bins)

        return {
            "ece_before": ece_before,
            "ece_after": ece_after,
            "improvement": ece_before - ece_after,
            "method": self.method,
            "samples_used": len(confidences),
        }

    def calibrate(self, raw_confidence: float) -> float:
        """
        Apply calibration to a raw confidence score.

        Args:
            raw_confidence: Raw confidence from model (0-1)

        Returns:
            Calibrated confidence (0-1)
        """
        if not self._fitted:
            return raw_confidence  # Return uncalibrated if not fitted

        if self.method == "temperature":
            # Temperature scaling: scale logits
            if raw_confidence <= 0 or raw_confidence >= 1:
                return raw_confidence
            logit = np.log(raw_confidence / (1 - raw_confidence))
            scaled_logit = logit / self._temperature
            return 1 / (1 + np.exp(-scaled_logit))

        elif self.method == "platt":
            # Platt scaling: P = 1 / (1 + exp(a*x + b))
            return 1 / (1 + np.exp(self._platt_a * raw_confidence + self._platt_b))

        elif self.method == "isotonic":
            # Isotonic regression: interpolate
            if self._isotonic_x is None or len(self._isotonic_x) == 0:
                return raw_confidence
            return float(np.interp(raw_confidence, self._isotonic_x, self._isotonic_y))

        return raw_confidence

    def _compute_ece(
        self,
        confidences: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue

            bin_confidence = confidences[mask].mean()
            bin_accuracy = outcomes[mask].mean()
            bin_size = mask.sum()

            ece += (bin_size / len(confidences)) * abs(bin_accuracy - bin_confidence)

        return ece

    def _fit_temperature(self, confidences: np.ndarray, outcomes: np.ndarray):
        """Fit temperature scaling parameter."""
        # Grid search for optimal temperature
        best_ece = float('inf')
        best_temp = 1.0

        for temp in np.linspace(0.5, 3.0, 50):
            # Apply temperature scaling
            logits = np.log(np.clip(confidences, 1e-10, 1 - 1e-10) /
                          (1 - np.clip(confidences, 1e-10, 1 - 1e-10)))
            scaled_logits = logits / temp
            calibrated = 1 / (1 + np.exp(-scaled_logits))

            ece = self._compute_ece(calibrated, outcomes)
            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        self._temperature = best_temp

    def _fit_platt(self, confidences: np.ndarray, outcomes: np.ndarray):
        """Fit Platt scaling parameters using gradient descent."""
        # Simple grid search (scipy.optimize would be better but adds dependency)
        best_ece = float('inf')
        best_a, best_b = 1.0, 0.0

        for a in np.linspace(-5, 5, 50):
            for b in np.linspace(-3, 3, 30):
                calibrated = 1 / (1 + np.exp(a * confidences + b))
                ece = self._compute_ece(calibrated, outcomes)
                if ece < best_ece:
                    best_ece = ece
                    best_a, best_b = a, b

        self._platt_a = best_a
        self._platt_b = best_b

    def _fit_isotonic(self, confidences: np.ndarray, outcomes: np.ndarray):
        """Fit isotonic regression using binned approach for stability."""
        # Bin the data and compute mean accuracy per bin
        n_bins = min(20, len(confidences) // 5)  # At least 5 samples per bin
        if n_bins < 3:
            n_bins = 3

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
                bin_accuracies.append(outcomes[mask].mean())

        if len(bin_centers) < 2:
            # Not enough data, use identity
            self._isotonic_x = np.array([0, 1])
            self._isotonic_y = np.array([0, 1])
            return

        bin_centers = np.array(bin_centers)
        bin_accuracies = np.array(bin_accuracies)

        # Apply PAV (Pool Adjacent Violators) to enforce monotonicity
        y = bin_accuracies.copy()
        n = len(y)

        # Iterate until no violations
        while True:
            changed = False
            i = 0
            while i < len(y) - 1:
                if y[i] > y[i + 1]:
                    # Pool: replace both with their average
                    avg = (y[i] + y[i + 1]) / 2
                    y[i] = avg
                    y[i + 1] = avg
                    changed = True
                i += 1
            if not changed:
                break

        # Add endpoints for interpolation
        x_final = np.concatenate([[0], bin_centers, [1]])
        y_final = np.concatenate([[max(0, y[0] - 0.1)], y, [min(1, y[-1] + 0.1)]])

        self._isotonic_x = x_final
        self._isotonic_y = y_final

    def get_reliability_diagram_data(
        self,
        confidences: List[float],
        outcomes: List[bool],
        n_bins: int = 10
    ) -> Dict[str, List[float]]:
        """
        Get data for plotting a reliability diagram.

        Returns bin midpoints, accuracies, and sizes for visualization.
        """
        confidences = np.array(confidences)
        outcomes = np.array(outcomes, dtype=float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_midpoints = []
        bin_accuracies = []
        bin_confidences = []
        bin_sizes = []

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue

            bin_midpoints.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accuracies.append(outcomes[mask].mean())
            bin_confidences.append(confidences[mask].mean())
            bin_sizes.append(mask.sum())

        return {
            "bin_midpoints": bin_midpoints,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_sizes": bin_sizes,
            "ideal_diagonal": [0, 1],  # For plotting y=x reference line
        }


def test_paraphrase_robustness():
    """Test that semantic extraction handles paraphrases."""
    extractor = SemanticExtractor()

    # Paraphrase pairs that should score similarly
    paraphrase_pairs = [
        ("I don't know", "I'm uncertain"),
        ("I chose this", "I made this decision"),
        ("We did it together", "Our team succeeded"),
        ("That's unfair", "This is unjust"),
        ("I feel lost", "I'm confused about this"),
    ]

    print("PARAPHRASE ROBUSTNESS TEST")
    print("=" * 60)

    for p1, p2 in paraphrase_pairs:
        r1 = extractor.extract(p1)
        r2 = extractor.extract(p2)

        print(f"\n'{p1}' vs '{p2}'")

        # Compare scores across dimensions
        for key in set(r1.keys()) | set(r2.keys()):
            s1 = r1.get(key, SemanticMatch("", "", 0, 0, "", ExtractionMethod.SEMANTIC)).score
            s2 = r2.get(key, SemanticMatch("", "", 0, 0, "", ExtractionMethod.SEMANTIC)).score
            if abs(s1) > 0.1 or abs(s2) > 0.1:
                diff = abs(s1 - s2)
                status = "SIMILAR" if diff < 0.3 else "DIFFERENT"
                print(f"  {key}: {s1:.2f} vs {s2:.2f} ({status})")


if __name__ == "__main__":
    test_paraphrase_robustness()
