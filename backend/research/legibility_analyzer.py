"""
Legibility Analyzer Module for the Cultural Soliton Observatory

This module measures how interpretable a signal/text is to humans and detects
phase transitions in legibility. The concept of "legibility" here is grounded
in coordination theory: highly legible signals are those that can readily
coordinate behavior across different agents (human or AI).

COORDINATION-THEORETIC INTERPRETATION
======================================

From a coordination perspective, legibility is the degree to which a signal
reduces uncertainty about the sender's intended meaning. This connects to:

1. SCHELLING POINT THEORY: Highly legible signals function as focal points
   that agents naturally converge on. Opaque signals require additional
   meta-coordination to interpret.

2. COMMON KNOWLEDGE: Legibility enables iterative common knowledge formation.
   "I know that you know that I know..." becomes tractable when signals are
   interpretable.

3. PHASE TRANSITIONS: Signals can undergo "legibility phase transitions" where
   small changes in vocabulary, syntax, or embedding coherence cause sudden
   jumps in interpretability. These transitions may correspond to:
   - Emergence of jargon/insider language
   - Code-switching between registers
   - Narrative mode crystallization
   - Meaning collapse under compression

4. THE LEGIBILITY-COMPRESSION TRADEOFF: Natural language maintains high
   legibility at the cost of redundancy. Compressed signals (emojis, code,
   specialized notation) trade legibility for bandwidth. This tradeoff is
   fundamental to coordination costs.

METRICS
=======

The module computes several complementary legibility metrics:

- MODE_CONFIDENCE: How confidently the Observatory classifies the signal's
  narrative mode. High confidence suggests clear coordination intent.

- MANIFOLD_DISTANCE: Distance from the "human language centroid" in embedding
  space. Trained on natural language, so unusual signals score higher distance.

- VOCABULARY_OVERLAP: Jaccard similarity with a reference corpus of natural
  language. Low overlap indicates specialized vocabulary.

- SYNTACTIC_COMPLEXITY: Measures structural complexity that may impede parsing.
  Very high or very low complexity reduces legibility.

- EMBEDDING_COHERENCE: How tightly clustered a text's token embeddings are.
  Incoherent embeddings suggest mixed or conflicting signals.

Usage:
    python -m research.legibility_analyzer --text "Your text here"
    python -m research.legibility_analyzer --file corpus.txt --detect-phase

API Integration:
    The module provides async methods that integrate with the Observatory API
    at http://127.0.0.1:8000 for embedding and classification.
"""

import logging
import re
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default API endpoint for the Cultural Soliton Observatory
DEFAULT_API_BASE = "http://127.0.0.1:8000"

# Reference vocabulary: most common English words
REFERENCE_VOCABULARY = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "is", "was", "are", "were", "been", "being", "am", "had", "has", "having",
    "did", "does", "doing", "done", "said", "says", "saying", "got", "gets", "getting",
    "made", "makes", "making", "went", "goes", "going", "gone", "came", "comes", "coming",
    "very", "really", "much", "more", "many", "should", "must", "may", "might",
    "here", "where", "why", "how", "each", "every", "both", "few",
    "always", "never", "still", "already", "often", "usually", "sometimes",
    "through", "during", "before", "above", "below", "between",
    "help", "need", "feel", "seem", "try", "keep", "let", "begin", "end",
    "right", "wrong", "same", "different", "important", "possible", "true",
    "long", "great", "little", "own", "old", "public", "last", "next",
    "high", "low", "small", "large", "big", "young", "early", "late",
}


class LegibilityRegime(Enum):
    """Communication legibility regimes.

    These regimes represent distinct phases in the legibility-compression tradeoff:

    - NATURAL: Standard human language optimized for interpretability
    - TECHNICAL: Domain-specific language with specialized vocabulary
    - COMPRESSED: High-efficiency, low-legibility communication
    - OPAQUE: Uninterpretable signals (noise or alien coordination)
    """
    NATURAL = "natural"           # High legibility, low efficiency
    TECHNICAL = "technical"       # Medium legibility, medium efficiency
    COMPRESSED = "compressed"     # Low legibility, high efficiency
    OPAQUE = "opaque"            # No legibility, unknown efficiency


@dataclass
class InterpretabilityMetrics:
    """
    Container for multiple interpretability proxy measurements.

    Each metric is normalized to [0, 1] where higher = more legible.
    These metrics are combined with configurable weights to produce
    the final legibility score.
    """
    mode_confidence: float        # From Observatory mode classification
    manifold_distance: float      # Distance to human language centroid (inverted)
    vocabulary_overlap: float     # Jaccard with reference corpus
    syntactic_complexity: float   # Normalized complexity score
    embedding_coherence: float    # Clustering of token embeddings

    # Raw values before normalization (for debugging)
    raw_mode_confidence: float = 0.0
    raw_manifold_distance: float = 0.0
    raw_vocabulary_overlap: float = 0.0
    raw_syntactic_complexity: float = 0.0
    raw_embedding_coherence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "mode_confidence": round(self.mode_confidence, 4),
            "manifold_distance": round(self.manifold_distance, 4),
            "vocabulary_overlap": round(self.vocabulary_overlap, 4),
            "syntactic_complexity": round(self.syntactic_complexity, 4),
            "embedding_coherence": round(self.embedding_coherence, 4),
            "raw": {
                "mode_confidence": round(self.raw_mode_confidence, 4),
                "manifold_distance": round(self.raw_manifold_distance, 4),
                "vocabulary_overlap": round(self.raw_vocabulary_overlap, 4),
                "syntactic_complexity": round(self.raw_syntactic_complexity, 4),
                "embedding_coherence": round(self.raw_embedding_coherence, 4),
            }
        }


@dataclass
class PhaseTransitionResult:
    """
    Result of phase transition detection in a signal history.

    A phase transition occurs when legibility scores show sudden, sustained
    shifts indicating a qualitative change in signal interpretability.

    From a coordination-theoretic perspective, phase transitions mark points
    where the fundamental nature of the communication changes - potentially
    indicating shifts in coordination strategy, audience, or intent.
    """
    detected: bool
    transition_points: List[int]  # Indices where transitions occur
    transition_magnitudes: List[float]  # Magnitude of each transition
    regime_labels: List[str]  # Labels for each regime
    variance_spikes: List[int]  # Indices of variance spikes
    mode_confidence_shifts: List[Tuple[int, str, str]]  # (index, from_mode, to_mode)

    # Statistical metrics
    rolling_mean: List[float] = field(default_factory=list)
    rolling_variance: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "detected": self.detected,
            "transition_points": self.transition_points,
            "transition_magnitudes": [round(m, 4) for m in self.transition_magnitudes],
            "regime_labels": self.regime_labels,
            "variance_spikes": self.variance_spikes,
            "mode_confidence_shifts": [
                {"index": idx, "from": from_m, "to": to_m}
                for idx, from_m, to_m in self.mode_confidence_shifts
            ],
            "rolling_mean": [round(m, 4) for m in self.rolling_mean],
            "rolling_variance": [round(v, 4) for v in self.rolling_variance],
        }


@dataclass
class LegibilityScore:
    """
    Combined legibility score with component breakdown.

    The score ranges from 0.0 (completely opaque) to 1.0 (natural language).
    This mapping reflects the coordination cost: opaque signals require
    significant additional effort to interpret and coordinate around.
    """
    score: float  # Final score 0.0 (opaque) to 1.0 (natural language)
    metrics: InterpretabilityMetrics
    regime: LegibilityRegime
    regime_confidence: float
    interpretation: str
    confidence_interval: Tuple[float, float] = (0.0, 1.0)

    # Risk indicators
    drift_from_baseline: float = 0.0
    alert_triggered: bool = False
    alert_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "score": round(self.score, 4),
            "metrics": self.metrics.to_dict(),
            "regime": self.regime.value,
            "regime_confidence": round(self.regime_confidence, 4),
            "interpretation": self.interpretation,
            "confidence_interval": [round(x, 4) for x in self.confidence_interval],
            "drift_from_baseline": round(self.drift_from_baseline, 4),
            "alert_triggered": self.alert_triggered,
            "alert_message": self.alert_message,
        }


class LegibilityAnalyzer:
    """
    Analyzes the interpretability (legibility) of signals/texts and detects
    phase transitions in legibility over time.

    Legibility is a coordination-theoretic measure of how readily a signal
    can serve its intended communicative function across different interpreters.
    High legibility means low coordination cost; low legibility means agents
    must expend significant effort to establish shared understanding.

    The analyzer integrates with the Cultural Soliton Observatory API to
    leverage embedding-based analysis while adding specialized legibility metrics.

    Key Methods:
        - compute_legibility_score(text): Get legibility score for a single text
        - compute_interpretability_metrics(text): Get all component metrics
        - detect_phase_transition(signal_history): Detect regime shifts
    """

    # Weights for combining metrics into final score
    DEFAULT_WEIGHTS = {
        "mode_confidence": 0.25,
        "manifold_distance": 0.20,
        "vocabulary_overlap": 0.20,
        "syntactic_complexity": 0.15,
        "embedding_coherence": 0.20,
    }

    # Legibility thresholds for regime classification
    THRESHOLDS = {
        "natural": 0.7,      # Highly legible, standard language
        "technical": 0.5,    # Moderately legible, specialized
        "compressed": 0.3,   # Low legibility, high efficiency
        "opaque": 0.0,       # Uninterpretable
    }

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        reference_vocabulary: Optional[set] = None,
        weights: Optional[Dict[str, float]] = None,
        model_id: str = "all-MiniLM-L6-v2",
        alert_threshold: float = 0.3,
        window_size: int = 100,
    ):
        """
        Initialize the Legibility Analyzer.

        Args:
            api_base: Base URL for the Observatory API
            reference_vocabulary: Set of reference words for vocabulary overlap
            weights: Custom weights for combining metrics (must sum to 1.0)
            model_id: Embedding model ID for API calls
            alert_threshold: Trigger alert if legibility drops below this
            window_size: Window size for rolling statistics
        """
        self.api_base = api_base
        self.reference_vocabulary = reference_vocabulary or REFERENCE_VOCABULARY
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.model_id = model_id
        self.alert_threshold = alert_threshold
        self.window_size = window_size

        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if not 0.99 <= total_weight <= 1.01:
            logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

        # Cache for API results (reduces redundant calls)
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._mode_cache: Dict[str, Dict] = {}

        # Human language manifold centroid (learned from natural language)
        # This is initialized lazily from embeddings
        self._manifold_centroid: Optional[np.ndarray] = None

        # History tracking for phase transition detection
        self._history: List[LegibilityScore] = []
        self._mode_history: List[str] = []

    def compute_legibility_score(
        self,
        text: str,
        reference_corpus: Optional[List[str]] = None,
    ) -> LegibilityScore:
        """
        Compute the overall legibility score for a text.

        This is the synchronous version that uses local computation only.
        For full analysis including API calls, use compute_legibility_score_async.

        The score ranges from 0.0 (opaque) to 1.0 (natural language), where:
        - 0.8-1.0: Highly legible, functions well for coordination
        - 0.6-0.8: Legible, interpretable with reasonable confidence
        - 0.4-0.6: Moderate, requires some effort to interpret
        - 0.2-0.4: Opaque, difficult to interpret
        - 0.0-0.2: Cryptic, coordination cost is very high

        Args:
            text: The text to analyze
            reference_corpus: Optional corpus for calibrating vocabulary overlap

        Returns:
            LegibilityScore with value 0.0 (opaque) to 1.0 (natural language)
        """
        # Compute local metrics
        raw_vocab = self._compute_raw_vocabulary_overlap(text, reference_corpus)
        vocab_overlap = self._normalize_vocabulary_overlap(raw_vocab)

        raw_syntactic = self._compute_raw_syntactic_complexity(text)
        syntactic = self._normalize_syntactic_complexity(raw_syntactic)

        # For sync version, use placeholder values for API-dependent metrics
        metrics = InterpretabilityMetrics(
            mode_confidence=0.5,  # Neutral placeholder
            manifold_distance=0.5,
            vocabulary_overlap=vocab_overlap,
            syntactic_complexity=syntactic,
            embedding_coherence=0.5,
            raw_mode_confidence=0.5,
            raw_manifold_distance=0.0,
            raw_vocabulary_overlap=raw_vocab,
            raw_syntactic_complexity=raw_syntactic,
            raw_embedding_coherence=0.0,
        )

        # Compute weighted score
        score = self._compute_weighted_score(metrics)

        # Classify regime
        regime, regime_conf = self._classify_regime(score, vocab_overlap, 0.5, text)

        # Compute confidence interval
        interval = self._compute_confidence_interval(metrics, has_api_data=False)

        # Compute drift from baseline
        drift = self._compute_drift(score)

        # Check for alerts
        alert_triggered = score < self.alert_threshold
        alert_message = None
        if alert_triggered:
            alert_message = f"Legibility dropped to {score:.2f}, below threshold {self.alert_threshold}"

        # Generate interpretation
        interpretation = self._generate_interpretation(score, metrics)

        result = LegibilityScore(
            score=score,
            metrics=metrics,
            regime=regime,
            regime_confidence=regime_conf,
            interpretation=interpretation,
            confidence_interval=interval,
            drift_from_baseline=drift,
            alert_triggered=alert_triggered,
            alert_message=alert_message,
        )

        # Update history
        self._history.append(result)

        return result

    async def compute_legibility_score_async(
        self,
        text: str,
        reference_corpus: Optional[List[str]] = None,
    ) -> LegibilityScore:
        """
        Compute the legibility score using the full API integration.

        This async version calls the Observatory API for embedding-based
        metrics like mode confidence, manifold distance, and embedding coherence.

        This is the preferred method when the Observatory API is available.

        Args:
            text: The text to analyze
            reference_corpus: Optional corpus for calibrating vocabulary overlap

        Returns:
            LegibilityScore with value 0.0 (opaque) to 1.0 (natural language)
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, falling back to sync computation")
            return self.compute_legibility_score(text, reference_corpus)

        # Fetch API-based metrics
        mode_result = await self._get_mode_classification(text)
        embedding = await self._get_embedding(text)

        # Compute all metrics
        raw_mode_conf = mode_result.get("confidence", 0.5) if mode_result else 0.5
        mode_confidence = float(raw_mode_conf)

        raw_manifold = self._compute_raw_manifold_distance(embedding)
        manifold_distance = self._normalize_manifold_distance(raw_manifold)

        raw_vocab = self._compute_raw_vocabulary_overlap(text, reference_corpus)
        vocab_overlap = self._normalize_vocabulary_overlap(raw_vocab)

        raw_syntactic = self._compute_raw_syntactic_complexity(text)
        syntactic = self._normalize_syntactic_complexity(raw_syntactic)

        raw_coherence = await self._compute_raw_embedding_coherence(text)
        coherence = self._normalize_embedding_coherence(raw_coherence)

        metrics = InterpretabilityMetrics(
            mode_confidence=mode_confidence,
            manifold_distance=manifold_distance,
            vocabulary_overlap=vocab_overlap,
            syntactic_complexity=syntactic,
            embedding_coherence=coherence,
            raw_mode_confidence=raw_mode_conf,
            raw_manifold_distance=raw_manifold,
            raw_vocabulary_overlap=raw_vocab,
            raw_syntactic_complexity=raw_syntactic,
            raw_embedding_coherence=raw_coherence,
        )

        # Compute weighted score
        score = self._compute_weighted_score(metrics)

        # Classify regime
        regime, regime_conf = self._classify_regime(score, vocab_overlap, mode_confidence, text)

        # Compute confidence interval
        interval = self._compute_confidence_interval(metrics, has_api_data=True)

        # Compute drift
        drift = self._compute_drift(score)

        # Check for alerts
        alert_triggered = score < self.alert_threshold
        alert_message = None
        if alert_triggered:
            alert_message = f"Legibility dropped to {score:.2f}, below threshold {self.alert_threshold}"

        # Generate interpretation
        interpretation = self._generate_interpretation(score, metrics)

        result = LegibilityScore(
            score=score,
            metrics=metrics,
            regime=regime,
            regime_confidence=regime_conf,
            interpretation=interpretation,
            confidence_interval=interval,
            drift_from_baseline=drift,
            alert_triggered=alert_triggered,
            alert_message=alert_message,
        )

        # Update history
        self._history.append(result)
        if mode_result:
            self._mode_history.append(mode_result.get("primary_mode", "UNKNOWN"))
        else:
            self._mode_history.append("UNKNOWN")

        return result

    def compute_interpretability_metrics(self, text: str) -> InterpretabilityMetrics:
        """
        Compute multiple interpretability proxy metrics for a text.

        This is the synchronous version that only computes local metrics.
        For full metrics including API calls, use compute_interpretability_metrics_async.

        Args:
            text: The text to analyze

        Returns:
            InterpretabilityMetrics with all proxy measurements
        """
        return self.compute_legibility_score(text).metrics

    async def compute_interpretability_metrics_async(
        self,
        text: str,
    ) -> InterpretabilityMetrics:
        """
        Compute interpretability metrics using full API integration.

        Args:
            text: The text to analyze

        Returns:
            InterpretabilityMetrics with all proxy measurements
        """
        result = await self.compute_legibility_score_async(text)
        return result.metrics

    def detect_phase_transition(
        self,
        signal_history: List[float],
        window_size: Optional[int] = None,
        transition_threshold: float = 0.3,
        variance_threshold: float = 2.0,
    ) -> PhaseTransitionResult:
        """
        Detect regime shifts (phase transitions) in legibility scores over time.

        Phase transitions indicate qualitative changes in signal interpretability,
        such as emergence of jargon, code-switching, or narrative crystallization.
        From a coordination perspective, these transitions mark points where
        communication strategy fundamentally changes.

        The detection uses three complementary approaches:
        1. Rolling mean change detection (sudden level shifts)
        2. Variance spike detection (increased uncertainty during transition)
        3. Mode confidence distribution changes (from classification API)

        Args:
            signal_history: Time series of legibility scores
            window_size: Window for rolling statistics (default: self.window_size)
            transition_threshold: Minimum mean shift to flag as transition
            variance_threshold: Threshold for variance spike (in std devs)

        Returns:
            PhaseTransitionResult with detected transitions and statistics
        """
        window_size = window_size or self.window_size

        if len(signal_history) < window_size * 2:
            return PhaseTransitionResult(
                detected=False,
                transition_points=[],
                transition_magnitudes=[],
                regime_labels=["INSUFFICIENT_DATA"],
                variance_spikes=[],
                mode_confidence_shifts=[],
                rolling_mean=[],
                rolling_variance=[],
            )

        scores = np.array(signal_history)

        # Compute rolling statistics
        rolling_mean = self._compute_rolling_mean(scores, window_size)
        rolling_variance = self._compute_rolling_variance(scores, window_size)

        # Detect transition points
        transition_points, magnitudes = self._detect_mean_shifts(
            rolling_mean, transition_threshold
        )

        # Detect variance spikes
        variance_spikes = self._detect_variance_spikes(
            rolling_variance, variance_threshold
        )

        # Label regimes
        regime_labels = self._label_regimes(rolling_mean, transition_points)

        return PhaseTransitionResult(
            detected=len(transition_points) > 0,
            transition_points=transition_points,
            transition_magnitudes=magnitudes,
            regime_labels=regime_labels,
            variance_spikes=variance_spikes,
            mode_confidence_shifts=[],  # Populated by async version
            rolling_mean=rolling_mean.tolist(),
            rolling_variance=rolling_variance.tolist(),
        )

    async def detect_phase_transition_async(
        self,
        texts: List[str],
        window_size: Optional[int] = None,
        transition_threshold: float = 0.3,
        variance_threshold: float = 2.0,
    ) -> PhaseTransitionResult:
        """
        Detect phase transitions by analyzing a sequence of texts.

        This async version computes legibility scores for each text and also
        tracks mode classification changes for additional transition signals.
        This provides richer phase transition detection than the basic version.

        Args:
            texts: Sequence of texts to analyze over time
            window_size: Window for rolling statistics
            transition_threshold: Minimum mean shift for transition
            variance_threshold: Threshold for variance spike

        Returns:
            PhaseTransitionResult with full transition analysis
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, using basic analysis")
            return self.detect_phase_transition(
                [0.5] * len(texts),  # Placeholder scores
                window_size,
                transition_threshold,
                variance_threshold,
            )

        window_size = window_size or self.window_size

        # Compute legibility scores for all texts
        scores = []
        mode_history = []

        for text in texts:
            result = await self.compute_legibility_score_async(text)
            scores.append(result.score)

            mode_result = await self._get_mode_classification(text)
            if mode_result:
                mode_history.append(mode_result.get("primary_mode", "UNKNOWN"))
            else:
                mode_history.append("UNKNOWN")

        # Run basic phase transition detection
        basic_result = self.detect_phase_transition(
            scores, window_size, transition_threshold, variance_threshold
        )

        # Detect mode classification shifts
        mode_shifts = self._detect_mode_shifts(mode_history)

        return PhaseTransitionResult(
            detected=basic_result.detected or len(mode_shifts) > 0,
            transition_points=basic_result.transition_points,
            transition_magnitudes=basic_result.transition_magnitudes,
            regime_labels=basic_result.regime_labels,
            variance_spikes=basic_result.variance_spikes,
            mode_confidence_shifts=mode_shifts,
            rolling_mean=basic_result.rolling_mean,
            rolling_variance=basic_result.rolling_variance,
        )

    async def analyze_stream(
        self,
        messages: List[str],
        return_all: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyze a stream of messages for legibility patterns.

        This is useful for monitoring ongoing communication and detecting
        when it drifts into low-legibility regimes.

        Args:
            messages: List of messages to analyze
            return_all: If True, include all individual scores

        Returns:
            Summary statistics and any detected phase transitions
        """
        scores = []
        for msg in messages:
            score = await self.compute_legibility_score_async(msg)
            scores.append(score)

        if not scores:
            return {"error": "No messages to analyze"}

        # Aggregate statistics
        score_values = [s.score for s in scores]

        # Regime distribution
        regime_counts: Dict[str, int] = {}
        for s in scores:
            r = s.regime.value
            regime_counts[r] = regime_counts.get(r, 0) + 1

        # Detect phase transitions
        if len(scores) >= self.window_size * 2:
            phase_result = self.detect_phase_transition(score_values)
            transitions = phase_result.to_dict()
        else:
            transitions = {"detected": False, "message": "Insufficient data"}

        result = {
            "message_count": len(messages),
            "legibility_stats": {
                "mean": float(np.mean(score_values)),
                "std": float(np.std(score_values)),
                "min": float(np.min(score_values)),
                "max": float(np.max(score_values)),
                "trend": self._compute_trend(score_values),
            },
            "regime_distribution": regime_counts,
            "dominant_regime": max(regime_counts, key=regime_counts.get) if regime_counts else "unknown",
            "phase_transitions": transitions,
            "alerts": [
                {"index": i, "message": s.alert_message}
                for i, s in enumerate(scores) if s.alert_triggered
            ],
        }

        if return_all:
            result["all_scores"] = [s.to_dict() for s in scores]

        return result

    def get_history(self) -> Dict[str, Any]:
        """Get full legibility history for analysis."""
        if not self._history:
            return {
                "scores": [],
                "regimes": [],
                "baseline_mean": 0.5,
                "baseline_std": 0.1,
                "current_trend": "unknown",
            }

        scores = [s.score for s in self._history]
        regimes = [s.regime.value for s in self._history]

        return {
            "scores": scores,
            "regimes": regimes,
            "baseline_mean": float(np.mean(scores)),
            "baseline_std": float(np.std(scores)),
            "current_trend": self._compute_trend(scores[-20:] if len(scores) > 20 else scores),
        }

    def reset(self):
        """Reset history and caches."""
        self._history = []
        self._mode_history = []
        self._embedding_cache = {}
        self._mode_cache = {}

    # =========================================================================
    # Private methods: Metric computation
    # =========================================================================

    def _compute_weighted_score(self, metrics: InterpretabilityMetrics) -> float:
        """Compute weighted combination of metrics."""
        return (
            self.weights["mode_confidence"] * metrics.mode_confidence +
            self.weights["manifold_distance"] * metrics.manifold_distance +
            self.weights["vocabulary_overlap"] * metrics.vocabulary_overlap +
            self.weights["syntactic_complexity"] * metrics.syntactic_complexity +
            self.weights["embedding_coherence"] * metrics.embedding_coherence
        )

    def _compute_raw_vocabulary_overlap(
        self,
        text: str,
        reference_corpus: Optional[List[str]] = None,
    ) -> float:
        """Compute raw Jaccard vocabulary overlap."""
        # Tokenize text
        words = set(self._normalize_word(w) for w in re.findall(r'\b\w+\b', text.lower()))
        words = {w for w in words if w}  # Remove empty strings

        if not words:
            return 0.0

        # Build reference vocabulary
        if reference_corpus:
            ref_vocab = set()
            for doc in reference_corpus:
                ref_vocab.update(
                    self._normalize_word(w)
                    for w in re.findall(r'\b\w+\b', doc.lower())
                )
        else:
            ref_vocab = self.reference_vocabulary

        if not ref_vocab:
            return 0.0

        # Jaccard similarity
        intersection = len(words & ref_vocab)
        union = len(words | ref_vocab)

        return intersection / union if union > 0 else 0.0

    def _normalize_vocabulary_overlap(self, raw: float) -> float:
        """Normalize vocabulary overlap to [0, 1]."""
        # Use sqrt to give partial credit for moderate overlap
        return math.sqrt(raw)

    def _normalize_word(self, word: str) -> str:
        """Normalize word for vocabulary lookup."""
        return re.sub(r'[^\w]', '', word.lower())

    def _compute_raw_syntactic_complexity(self, text: str) -> float:
        """Compute raw syntactic complexity as average sentence length."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        word_counts = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        return sum(word_counts) / len(word_counts) if word_counts else 0.0

    def _normalize_syntactic_complexity(self, raw: float) -> float:
        """
        Normalize syntactic complexity to [0, 1].

        Optimal complexity is around 10-20 words per sentence.
        Very short or very long sentences reduce legibility.
        """
        optimal = 15.0
        deviation = abs(raw - optimal) / optimal
        return max(0.0, 1.0 - deviation * 0.5)

    def _compute_raw_manifold_distance(self, embedding: Optional[np.ndarray]) -> float:
        """Compute raw Euclidean distance from manifold centroid."""
        if embedding is None:
            return 1.0  # Maximum distance if no embedding

        if self._manifold_centroid is None:
            # Initialize centroid as zero vector (approximation)
            self._manifold_centroid = np.zeros_like(embedding)

        return float(np.linalg.norm(embedding - self._manifold_centroid))

    def _normalize_manifold_distance(self, raw: float) -> float:
        """
        Normalize manifold distance to [0, 1].

        Distance 0 -> 1.0 (at centroid = most natural)
        Distance 2.0+ -> 0.0 (far from natural language)
        """
        return max(0.0, 1.0 - raw / 2.0)

    async def _compute_raw_embedding_coherence(self, text: str) -> float:
        """
        Compute raw embedding coherence as variance of sentence embeddings.

        Lower variance indicates tighter clustering of meaning across sentences.
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) < 2:
            return 0.0

        # Get embeddings for each sentence
        embeddings = []
        for sentence in sentences[:10]:  # Limit to 10 for efficiency
            emb = await self._get_embedding(sentence)
            if emb is not None:
                embeddings.append(emb)

        if len(embeddings) < 2:
            return 0.0

        # Compute mean pairwise cosine similarity
        embeddings_array = np.array(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / (norms + 1e-8)

        sim_matrix = embeddings_array @ embeddings_array.T
        n = len(embeddings)
        off_diag_sum = sim_matrix.sum() - np.trace(sim_matrix)
        mean_sim = off_diag_sum / (n * (n - 1)) if n > 1 else 1.0

        # Return variance (1 - similarity)
        return float(1.0 - mean_sim)

    def _normalize_embedding_coherence(self, raw: float) -> float:
        """
        Normalize embedding coherence to [0, 1].

        Higher coherence (lower variance) is more legible.
        """
        return max(0.0, 1.0 - raw)

    def _classify_regime(
        self,
        score: float,
        vocab_coverage: float,
        mode_confidence: float,
        text: str = "",
    ) -> Tuple[LegibilityRegime, float]:
        """
        Classify the communication regime based on multiple signals.

        Uses character analysis and opacity detection for robust classification.
        Priority order: COMPRESSED (has structure) > OPAQUE (no structure) > TECHNICAL > NATURAL
        """
        # Get character-level analysis
        char_profile = self._analyze_character_profile(text)

        # Check for compressed content FIRST (has recognizable structure)
        # This prevents structured configs/URLs from being classified as OPAQUE
        if char_profile["is_compressed"]:
            return LegibilityRegime.COMPRESSED, 0.5 + (0.5 - score)

        # Check for opaque content (no structure, truly uninterpretable)
        if char_profile["is_opaque"]:
            return LegibilityRegime.OPAQUE, 1.0 - score

        # Check for technical content (code-like, commands, has some structure)
        if char_profile["is_technical"]:
            return LegibilityRegime.TECHNICAL, min(1.0, (score + 0.6) / 2)

        # Default to natural for normal prose
        # Natural language has high alpha ratio, normal sentence structure
        if char_profile["is_natural"]:
            return LegibilityRegime.NATURAL, max(score, 0.7)

        # Fallback based on score thresholds
        if score >= self.THRESHOLDS["natural"]:
            return LegibilityRegime.NATURAL, score
        elif score >= self.THRESHOLDS["technical"]:
            return LegibilityRegime.TECHNICAL, min(1.0, (score + mode_confidence) / 2)
        elif score >= self.THRESHOLDS["compressed"]:
            return LegibilityRegime.COMPRESSED, 0.5 + (0.5 - score)
        else:
            return LegibilityRegime.OPAQUE, 1.0 - score

    def _analyze_character_profile(self, text: str) -> Dict[str, bool]:
        """
        Analyze character-level patterns to determine text type.

        Returns dict with is_natural, is_technical, is_compressed, is_opaque flags.
        """
        if not text or len(text) < 3:
            return {"is_natural": False, "is_technical": False,
                    "is_compressed": False, "is_opaque": False}

        # Character counts
        total = len(text)
        alpha = sum(1 for c in text if c.isalpha())
        digit = sum(1 for c in text if c.isdigit())
        space = sum(1 for c in text if c.isspace())
        symbol = total - alpha - digit - space

        alpha_ratio = alpha / total
        digit_ratio = digit / total
        space_ratio = space / total
        symbol_ratio = symbol / total

        # Bigram analysis (natural language indicator)
        common_bigrams = {
            'th', 'he', 'in', 'er', 'an', 're', 'on', 'at', 'en', 'nd',
            'ti', 'es', 'or', 'te', 'of', 'ed', 'is', 'it', 'al', 'ar',
            'st', 'to', 'nt', 'ng', 'se', 'ha', 'as', 'ou', 'io', 'le',
        }
        alpha_text = ''.join(c for c in text.lower() if c.isalpha())
        bigrams = [alpha_text[i:i+2] for i in range(len(alpha_text)-1)] if len(alpha_text) >= 2 else []
        bigram_score = sum(1 for bg in bigrams if bg in common_bigrams) / max(len(bigrams), 1)

        # Pattern detection
        has_sentences = bool(re.search(r'[.!?]\s+[A-Z]', text)) or text.endswith(('.', '!', '?'))
        has_key_value = bool(re.search(r'\w+[=:]\w+', text))
        has_code_patterns = bool(re.search(r'(def |function |=>|import |pip |npm |git )', text.lower()))
        has_binary = bool(re.search(r'[01]{8,}', text))  # Require 8+ bits for binary
        has_hex = bool(re.search(r'0x[0-9a-fA-F]+', text))

        # Structure detection for COMPRESSED vs OPAQUE distinction
        has_json_like = bool(re.search(r'[{}\[\]].*[{}\[\]]', text))
        has_url_pattern = bool(re.search(r'(https?://|www\.|\.\w{2,4}/)', text))
        has_path_pattern = bool(re.search(r'(/[\w.-]+){2,}|\\[\w.-]+\\', text))
        has_delimiter_structure = bool(re.search(r'([,;|][\w.-]+){3,}', text))
        has_base64_pattern = bool(re.search(r'^[A-Za-z0-9+/]{20,}={0,2}$', text.strip()))
        has_repeated_structure = bool(re.search(r'(\w+[=:]\w+[&,;]\s*){2,}', text))  # key=val&key=val pattern

        # OBFUSCATION patterns - these indicate OPAQUE, not COMPRESSED
        has_lambda_obfuscation = bool(re.search(r'lambda\s+_[^a-zA-Z]|lambda\s+\w+:\s*\w+\(lambda', text))
        has_meta_programming = bool(re.search(r"type\s*\(\s*['\"]|getattr\s*\(\s*__builtins__", text))
        has_hex_escapes = bool(re.search(r'\\x[0-9a-fA-F]{2}', text))
        has_obfuscated_vars = bool(re.search(r'_0x[0-9a-fA-F]+|_[0-9]+[a-zA-Z]', text))
        has_nested_encode = bool(re.search(r'(eval|exec)\s*\(\s*(compile|__import__|b64decode)', text))
        has_pure_hex_string = bool(re.search(r'^0x[0-9a-fA-F]{8,}$', text.strip()))

        # Composite obfuscation check
        is_obfuscated = (
            has_lambda_obfuscation or
            has_meta_programming or
            has_hex_escapes or
            has_obfuscated_vars or
            has_nested_encode or
            has_pure_hex_string or
            has_base64_pattern  # Base64 is encoding, thus opaque
        )

        # Check for structured patterns (human-readable structure = COMPRESSED)
        # Note: base64 is NOT human-readable structure, it's encoding
        has_structure = (
            has_json_like or
            has_url_pattern or
            has_path_pattern or
            has_delimiter_structure or
            has_repeated_structure or
            has_key_value
        ) and not is_obfuscated  # Obfuscated content is NOT "structured" in the COMPRESSED sense

        # Classification logic - COMPRESSED has human-readable structure
        is_compressed = (
            not is_obfuscated and (  # Obfuscated content is OPAQUE, not COMPRESSED
                (has_key_value and space_ratio < 0.15 and not has_sentences) or
                (space_ratio < 0.05 and len(text) > 10 and alpha_ratio > 0.6) or
                has_json_like or
                has_url_pattern or
                has_path_pattern or
                (has_delimiter_structure and alpha_ratio > 0.3) or
                (has_repeated_structure and not has_sentences)
            )
        )

        # OPAQUE: obfuscated, encoded, or truly unstructured content
        is_opaque = (
            is_obfuscated or  # All obfuscation patterns are OPAQUE
            (alpha_ratio < 0.4 and symbol_ratio > 0.25) or
            (bigram_score < 0.08 and alpha_ratio > 0.5) or
            has_binary or
            has_hex or
            (digit_ratio > 0.4 and symbol_ratio > 0.15)
        )

        # Ensure mutual exclusion: obfuscated content is always OPAQUE
        if is_obfuscated:
            is_compressed = False
            is_opaque = True

        # Technical vocabulary - split into strong (definitely technical) and weak (could be casual)
        # Strong terms trigger TECHNICAL with just 1 match
        strong_tech_vocab = [
            'api', 'database', 'authentication', 'token', 'endpoint', 'repository',
            'docker', 'kubernetes', 'microservice', 'async', 'synchronous',
            'traceback', 'stacktrace', 'serialize', 'deserialize', 'compile',
            'tcp', 'udp', 'socket', 'protocol', 'encryption', 'decoding',
            'yaml', 'json', 'xml', 'csv', 'schema', 'migration', 'deployment',
            'polymorphism', 'inheritance', 'encapsulation', 'abstraction',
            'metadata', 'payload', 'bandwidth', 'latency', 'throughput',
            'algorithm', 'implementation', 'initialize', 'dependencies',
            'npm', 'pip', 'git', 'sudo', 'chmod', 'localhost',
        ]
        # Weak terms need 2+ matches or combination with strong terms
        weak_tech_vocab = [
            'install', 'run', 'execute', 'parameter', 'returns', 'function',
            'connection', 'exception', 'input', 'output', 'config', 'server',
            'client', 'request', 'response', 'status', 'timeout', 'error',
            'debug', 'log', 'interface', 'module', 'library', 'framework',
            'query', 'container', 'thread', 'process', 'memory', 'cache',
            'buffer', 'queue', 'stack', 'binary', 'encoding', 'hash',
            'checksum', 'validate', 'parse', 'runtime', 'variable', 'constant',
            'method', 'class', 'object', 'instance', 'software', 'hardware',
            'system', 'network', 'header', 'index', 'cursor',
        ]
        text_lower = text.lower()
        strong_count = sum(1 for w in strong_tech_vocab if w in text_lower)
        weak_count = sum(1 for w in weak_tech_vocab if w in text_lower)
        tech_word_count = strong_count + weak_count
        # Technical if: 1+ strong OR 2+ weak OR (1 weak + code patterns)
        has_tech_vocab = strong_count >= 1 or weak_count >= 2

        # Also check for technical term density
        words = re.findall(r'\b\w+\b', text_lower)
        tech_density = tech_word_count / max(len(words), 1)
        has_high_tech_density = tech_density > 0.05  # Raised from 0.03 to 0.05

        # Code-like patterns in prose
        has_code_in_prose = bool(re.search(r':\s*(pip |npm |git |run |def |class )', text.lower()))
        has_api_pattern = bool(re.search(r'(GET|POST|PUT|DELETE|PATCH)\s+/', text))
        has_comment_start = text.strip().startswith('#') or text.strip().startswith('//')

        is_technical = (
            has_code_patterns or
            has_code_in_prose or
            has_api_pattern or
            has_comment_start or
            (has_tech_vocab and has_sentences) or
            has_high_tech_density or  # Technical prose with high term density
            (has_key_value and alpha_ratio > 0.5) or
            (symbol_ratio > 0.05 and symbol_ratio < 0.15 and digit_ratio > 0.05) or
            (strong_count >= 1 and has_sentences)  # Strong tech term is sufficient
        )

        is_natural = (
            alpha_ratio > 0.7 and
            space_ratio > 0.1 and
            bigram_score > 0.15 and
            not is_opaque and
            not is_compressed and
            not is_technical
        )

        # Natural language prose about technical topics should be TECHNICAL, not NATURAL
        # Require meaningful technical signal (1+ strong term OR 2+ weak terms)
        if has_sentences and bigram_score > 0.2:
            if has_tech_vocab or has_high_tech_density:  # Changed from tech_word_count >= 1
                is_technical = True
                is_natural = False
            # Pure prose without meaningful technical markers is NATURAL
            elif alpha_ratio > 0.75 and not has_tech_vocab and not has_code_patterns:
                is_natural = True
                is_technical = False

        return {
            "is_natural": is_natural,
            "is_technical": is_technical and not is_natural,
            "is_compressed": is_compressed and not is_opaque,
            "is_opaque": is_opaque,
        }

    def _compute_drift(self, current: float) -> float:
        """Compute drift from historical baseline."""
        if len(self._history) < 10:
            return 0.0

        baseline_scores = [s.score for s in self._history[-self.window_size:-1]]
        if not baseline_scores:
            return 0.0

        baseline_mean = np.mean(baseline_scores)
        return current - baseline_mean

    def _compute_confidence_interval(
        self,
        metrics: InterpretabilityMetrics,
        has_api_data: bool,
    ) -> Tuple[float, float]:
        """Compute confidence interval for legibility score."""
        values = [
            metrics.mode_confidence,
            metrics.manifold_distance,
            metrics.vocabulary_overlap,
            metrics.syntactic_complexity,
            metrics.embedding_coherence,
        ]
        std = np.std(values)

        if not has_api_data:
            std = max(std, 0.15)

        mean = np.mean(values)
        lower = max(0.0, mean - 1.96 * std)
        upper = min(1.0, mean + 1.96 * std)

        return (lower, upper)

    def _generate_interpretation(
        self,
        score: float,
        metrics: InterpretabilityMetrics,
    ) -> str:
        """Generate human-readable interpretation of legibility score."""
        if score >= self.THRESHOLDS["natural"]:
            level = "highly legible"
            desc = "Signal is easily interpretable and functions well for coordination"
        elif score >= self.THRESHOLDS["technical"]:
            level = "legible"
            desc = "Signal can be interpreted with reasonable confidence"
        elif score >= self.THRESHOLDS["compressed"]:
            level = "moderately legible"
            desc = "Signal requires some effort to interpret; potential ambiguity"
        else:
            level = "opaque"
            desc = "Signal is difficult to interpret; coordination cost is high"

        notes = []
        if metrics.vocabulary_overlap < 0.3:
            notes.append("unusual vocabulary")
        if metrics.syntactic_complexity < 0.4:
            notes.append("atypical sentence structure")
        if metrics.mode_confidence < 0.5:
            notes.append("ambiguous narrative mode")
        if metrics.embedding_coherence < 0.4:
            notes.append("inconsistent semantic content")

        interpretation = f"Text is {level}. {desc}."
        if notes:
            interpretation += f" Note: {', '.join(notes)}."

        return interpretation

    # =========================================================================
    # Private methods: API integration
    # =========================================================================

    async def _get_mode_classification(self, text: str) -> Optional[Dict]:
        """Get mode classification from Observatory API."""
        cache_key = f"mode:{hash(text)}"
        if cache_key in self._mode_cache:
            return self._mode_cache[cache_key]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_base}/v2/analyze",
                    json={
                        "text": text,
                        "model_id": self.model_id,
                    }
                )
                response.raise_for_status()
                result = response.json()
                mode = result.get("mode", {})
                self._mode_cache[cache_key] = mode
                return mode
        except Exception as e:
            logger.warning(f"Failed to get mode classification: {e}")
            return None

    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from Observatory API."""
        cache_key = f"emb:{hash(text)}"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_base}/embed",
                    json={
                        "text": text,
                        "model_id": self.model_id,
                    }
                )
                response.raise_for_status()
                result = response.json()
                embedding = np.array(result.get("embedding", []))
                self._embedding_cache[cache_key] = embedding
                return embedding
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
            return None

    # =========================================================================
    # Private methods: Phase transition detection
    # =========================================================================

    def _compute_rolling_mean(
        self,
        scores: np.ndarray,
        window_size: int,
    ) -> np.ndarray:
        """Compute rolling mean of scores."""
        cumsum = np.cumsum(np.insert(scores, 0, 0))
        rolling = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        padding = np.full(window_size - 1, rolling[0] if len(rolling) > 0 else 0.0)
        return np.concatenate([padding, rolling])

    def _compute_rolling_variance(
        self,
        scores: np.ndarray,
        window_size: int,
    ) -> np.ndarray:
        """Compute rolling variance of scores."""
        n = len(scores)
        result = np.zeros(n)

        for i in range(n):
            start = max(0, i - window_size + 1)
            window = scores[start:i + 1]
            result[i] = np.var(window) if len(window) > 1 else 0.0

        return result

    def _detect_mean_shifts(
        self,
        rolling_mean: np.ndarray,
        threshold: float,
    ) -> Tuple[List[int], List[float]]:
        """Detect significant shifts in rolling mean."""
        if len(rolling_mean) < 2:
            return [], []

        diff = np.diff(rolling_mean)
        transition_points = []
        magnitudes = []

        for i, d in enumerate(diff):
            if abs(d) > threshold:
                transition_points.append(i + 1)
                magnitudes.append(float(d))

        return transition_points, magnitudes

    def _detect_variance_spikes(
        self,
        rolling_variance: np.ndarray,
        threshold: float,
    ) -> List[int]:
        """Detect variance spikes indicating instability."""
        if len(rolling_variance) < 2:
            return []

        mean_var = np.mean(rolling_variance)
        std_var = np.std(rolling_variance)

        if std_var == 0:
            return []

        z_scores = (rolling_variance - mean_var) / std_var
        return list(np.where(z_scores > threshold)[0])

    def _label_regimes(
        self,
        rolling_mean: np.ndarray,
        transition_points: List[int],
    ) -> List[str]:
        """Label regimes based on rolling mean levels."""
        if len(rolling_mean) == 0:
            return []

        boundaries = [0] + transition_points + [len(rolling_mean)]
        labels = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            segment_mean = np.mean(rolling_mean[start:end])

            if segment_mean >= self.THRESHOLDS["natural"]:
                labels.append("NATURAL")
            elif segment_mean >= self.THRESHOLDS["technical"]:
                labels.append("TECHNICAL")
            elif segment_mean >= self.THRESHOLDS["compressed"]:
                labels.append("COMPRESSED")
            else:
                labels.append("OPAQUE")

        return labels

    def _detect_mode_shifts(
        self,
        mode_history: List[str],
    ) -> List[Tuple[int, str, str]]:
        """Detect mode classification shifts in history."""
        shifts = []

        for i in range(1, len(mode_history)):
            if mode_history[i] != mode_history[i - 1]:
                shifts.append((i, mode_history[i - 1], mode_history[i]))

        return shifts

    def _compute_trend(self, scores: List[float]) -> str:
        """Compute trend direction from scores."""
        if len(scores) < 3:
            return "insufficient_data"

        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]

        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        else:
            return "stable"


# =============================================================================
# Convenience functions
# =============================================================================

async def compute_legibility(text: str, api_base: str = DEFAULT_API_BASE) -> Dict[str, Any]:
    """
    Quick legibility check for a single message.

    This is a convenience function for one-off analysis without
    instantiating an analyzer object.
    """
    analyzer = LegibilityAnalyzer(api_base=api_base)
    score = await analyzer.compute_legibility_score_async(text)
    return score.to_dict()


def compute_legibility_sync(text: str) -> Dict[str, Any]:
    """
    Synchronous legibility check (local metrics only).

    Use when the Observatory API is not available.
    """
    analyzer = LegibilityAnalyzer()
    score = analyzer.compute_legibility_score(text)
    return score.to_dict()


# =============================================================================
# CLI Interface
# =============================================================================

async def main():
    """CLI interface for the Legibility Analyzer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Legibility Analyzer - Measure signal interpretability and detect phase transitions"
    )
    parser.add_argument("--text", type=str, help="Single text to analyze")
    parser.add_argument("--file", type=str, help="File with texts (one per line)")
    parser.add_argument(
        "--detect-phase",
        action="store_true",
        help="Detect phase transitions in file"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Window size for phase detection"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help="Observatory API base URL"
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Run without API (local metrics only)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Alert threshold for low legibility"
    )

    args = parser.parse_args()

    analyzer = LegibilityAnalyzer(
        api_base=args.api_base,
        alert_threshold=args.threshold,
        window_size=args.window_size,
    )

    if args.text:
        # Single text analysis
        print("\n=== LEGIBILITY ANALYSIS ===")
        print(f"\nText: {args.text[:100]}{'...' if len(args.text) > 100 else ''}")

        if args.no_api:
            result = analyzer.compute_legibility_score(args.text)
        else:
            result = await analyzer.compute_legibility_score_async(args.text)

        print(f"\nLegibility Score: {result.score:.4f}")
        print(f"Regime: {result.regime.value} (confidence: {result.regime_confidence:.2f})")
        print(f"Interpretation: {result.interpretation}")
        print(f"Confidence Interval: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")

        print("\n--- Component Metrics ---")
        metrics = result.metrics.to_dict()
        for key, value in metrics.items():
            if key != "raw":
                print(f"  {key}: {value:.4f}")

        if result.alert_triggered:
            print(f"\n!! ALERT: {result.alert_message}")

    elif args.file:
        # File analysis
        with open(args.file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"\n=== ANALYZING {len(texts)} TEXTS ===")

        if args.detect_phase:
            # Phase transition detection
            if args.no_api:
                # Compute local scores only
                scores = [
                    analyzer.compute_legibility_score(text).score
                    for text in texts
                ]
                result = analyzer.detect_phase_transition(
                    scores, window_size=args.window_size
                )
            else:
                result = await analyzer.detect_phase_transition_async(
                    texts, window_size=args.window_size
                )

            print("\n=== PHASE TRANSITION ANALYSIS ===")
            print(f"Transitions Detected: {result.detected}")
            print(f"Transition Points: {result.transition_points}")
            print(f"Transition Magnitudes: {result.transition_magnitudes}")
            print(f"Regime Labels: {result.regime_labels}")
            print(f"Variance Spikes: {result.variance_spikes}")

            if result.mode_confidence_shifts:
                print("\n--- Mode Shifts ---")
                for idx, from_m, to_m in result.mode_confidence_shifts:
                    print(f"  Index {idx}: {from_m} -> {to_m}")
        else:
            # Simple analysis of all texts
            print("\n--- Individual Scores ---")
            scores = []
            for i, text in enumerate(texts[:20]):  # Limit display
                if args.no_api:
                    result = analyzer.compute_legibility_score(text)
                else:
                    result = await analyzer.compute_legibility_score_async(text)
                scores.append(result.score)
                regime_short = result.regime.value[:4].upper()
                print(f"  {i+1:3d}. [{result.score:.3f}] [{regime_short}] {text[:50]}...")

            if len(texts) > 20:
                print(f"  ... and {len(texts) - 20} more")

            print(f"\nMean Legibility: {np.mean(scores):.4f}")
            print(f"Std Dev: {np.std(scores):.4f}")

            # Show history summary
            history = analyzer.get_history()
            print(f"Trend: {history['current_trend']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
