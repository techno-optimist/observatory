"""
Translation Lens for Emergent AI Protocols

This module provides tools for unsupervised translation of emergent communication
protocols developed by AI agents. When AI systems develop their own communication
languages, we need methods to decode them without parallel corpora or bilingual
dictionaries.

THEORETICAL FOUNDATION
======================

The Translation Lens operates on three key principles:

1. DISTRIBUTIONAL SEMANTICS: The meaning of a symbol is defined by its usage
   patterns. If symbol X appears in similar contexts as symbol Y, they likely
   share semantic properties (Harris, 1954; Firth, 1957).

2. GROUNDING THROUGH CO-OCCURRENCE: Symbols acquire meaning through their
   statistical association with observable states, actions, or outcomes.
   High PMI (Pointwise Mutual Information) between a symbol and a context
   suggests semantic linkage.

3. COMPOSITIONAL STRUCTURE: If emergent protocols exhibit compositionality,
   we can induce grammar rules by identifying recurring patterns and
   discovering how symbol combinations relate to combined meanings.

KEY COMPONENTS
==============

- SymbolGrounder: Maps symbols to probability distributions over meanings
  using co-occurrence analysis, PMI, and distributional semantics.

- GrammarInducer: Extracts compositional rules from message patterns,
  detecting positional regularities, delimiters, and recurring subsequences.

- TranslationLens: Unified interface combining grounding and grammar
  induction to decode new messages.

SAFETY CONSIDERATIONS
=====================

Understanding emergent AI protocols is crucial for AI safety:
- Detecting when agents develop opaque communication
- Ensuring human oversight of multi-agent coordination
- Identifying potential deceptive communication patterns

Usage:
    lens = TranslationLens()
    lens.fit(messages, contexts)
    result = lens.decode(new_message)
    print(result.likely_meaning, result.confidence)
"""

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes and Types
# =============================================================================


@dataclass
class SymbolMeaning:
    """
    Represents the grounded meaning of a symbol.

    A symbol may have multiple possible meanings with different probabilities.
    The grounding is based on observed co-occurrences with contexts/actions.
    """
    symbol: str
    meaning_distribution: Dict[str, float]  # meaning -> probability
    primary_meaning: str
    confidence: float
    support: int  # Number of observations
    pmi_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "meaning_distribution": {k: round(v, 4) for k, v in self.meaning_distribution.items()},
            "primary_meaning": self.primary_meaning,
            "confidence": round(self.confidence, 4),
            "support": self.support,
            "pmi_scores": {k: round(v, 4) for k, v in self.pmi_scores.items()},
        }


@dataclass
class GrammarRule:
    """
    Represents an induced grammar rule for symbol composition.

    Rules capture how symbols combine to form composite meanings.
    """
    pattern: Tuple[str, ...]  # Sequence of symbols or symbol classes
    combined_meaning: str
    confidence: float
    frequency: int
    examples: List[Tuple[str, str]] = field(default_factory=list)  # (message, context)

    def __str__(self) -> str:
        pattern_str = " + ".join(self.pattern)
        return f"{pattern_str} -> {self.combined_meaning} (conf={self.confidence:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": list(self.pattern),
            "combined_meaning": self.combined_meaning,
            "confidence": round(self.confidence, 4),
            "frequency": self.frequency,
            "examples": self.examples[:5],  # Limit examples
        }


@dataclass
class GrammarSketch:
    """
    A sketch of the grammar underlying an emergent protocol.

    This is not a complete formal grammar but rather a collection of
    observed regularities that help interpret messages.
    """
    rules: List[GrammarRule]
    delimiters: Set[str]
    positional_patterns: Dict[int, Dict[str, float]]  # position -> symbol -> frequency
    symbol_classes: Dict[str, Set[str]]  # class_name -> symbols
    vocabulary_size: int
    mean_message_length: float
    compositionality_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rules": [r.to_dict() for r in self.rules],
            "delimiters": list(self.delimiters),
            "positional_patterns": {
                pos: {sym: round(freq, 4) for sym, freq in dist.items()}
                for pos, dist in self.positional_patterns.items()
            },
            "symbol_classes": {k: list(v) for k, v in self.symbol_classes.items()},
            "vocabulary_size": self.vocabulary_size,
            "mean_message_length": round(self.mean_message_length, 4),
            "compositionality_score": round(self.compositionality_score, 4),
        }


@dataclass
class DecodeResult:
    """
    Result of decoding a message through the translation lens.
    """
    original_message: str
    tokens: List[str]
    likely_meaning: str
    confidence: float
    symbol_meanings: Dict[str, SymbolMeaning]
    applied_rules: List[GrammarRule]
    unknown_symbols: Set[str]
    interpretation_path: List[str]  # Step-by-step interpretation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_message": self.original_message,
            "tokens": self.tokens,
            "likely_meaning": self.likely_meaning,
            "confidence": round(self.confidence, 4),
            "symbol_meanings": {k: v.to_dict() for k, v in self.symbol_meanings.items()},
            "applied_rules": [r.to_dict() for r in self.applied_rules],
            "unknown_symbols": list(self.unknown_symbols),
            "interpretation_path": self.interpretation_path,
        }


@dataclass
class Context:
    """
    Represents the context in which a message was produced.

    Context can include environment state, actions taken, outcomes observed,
    or any other grounding information.
    """
    type: str  # "environment", "action", "outcome", "mixed"
    data: Dict[str, Any]
    labels: List[str]  # Human-readable labels for the context

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "Context":
        """Create context from environment state."""
        labels = []
        for key, value in state.items():
            if isinstance(value, bool):
                labels.append(f"{key}={value}")
            elif isinstance(value, (int, float)):
                labels.append(f"{key}:{value}")
            else:
                labels.append(str(value))
        return cls(type="environment", data=state, labels=labels)

    @classmethod
    def from_action(cls, action: str, params: Optional[Dict] = None) -> "Context":
        """Create context from action."""
        labels = [action]
        if params:
            for k, v in params.items():
                labels.append(f"{action}.{k}={v}")
        return cls(type="action", data={"action": action, "params": params or {}}, labels=labels)

    @classmethod
    def from_outcome(cls, outcome: str, success: bool = True) -> "Context":
        """Create context from outcome."""
        labels = [outcome, f"success={success}"]
        return cls(type="outcome", data={"outcome": outcome, "success": success}, labels=labels)


# =============================================================================
# Symbol Grounding
# =============================================================================


class SymbolGrounder:
    """
    Builds symbol -> meaning mappings through co-occurrence analysis.

    The grounding process works by analyzing how symbols co-occur with
    observable contexts (states, actions, outcomes). High PMI between
    a symbol and a context label suggests semantic association.

    Methods:
    - Co-occurrence counting: Raw frequency of symbol-context pairs
    - PMI (Pointwise Mutual Information): Normalized association strength
    - Distributional clustering: Group symbols with similar usage patterns

    Usage:
        grounder = SymbolGrounder()
        grounder.fit(messages, contexts)
        meaning = grounder.ground_symbol("X7")
        print(meaning.primary_meaning, meaning.confidence)
    """

    def __init__(
        self,
        min_support: int = 2,
        pmi_threshold: float = 0.5,
        smoothing: float = 0.1,
    ):
        """
        Initialize the symbol grounder.

        Args:
            min_support: Minimum observations for reliable grounding
            pmi_threshold: Minimum PMI score for meaning association
            smoothing: Laplace smoothing for probability estimation
        """
        self.min_support = min_support
        self.pmi_threshold = pmi_threshold
        self.smoothing = smoothing

        # Learned mappings
        self._symbol_counts: Counter = Counter()
        self._context_counts: Counter = Counter()
        self._cooccurrence_counts: Dict[str, Counter] = defaultdict(Counter)
        self._total_observations: int = 0

        # Cached results
        self._grounded_symbols: Dict[str, SymbolMeaning] = {}
        self._fitted: bool = False

    def fit(
        self,
        messages: List[str],
        contexts: List[Context],
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> "SymbolGrounder":
        """
        Fit the grounder to observed message-context pairs.

        Args:
            messages: List of messages in the emergent protocol
            contexts: Corresponding contexts for each message
            tokenizer: Function to tokenize messages (default: whitespace split)

        Returns:
            self for chaining
        """
        if len(messages) != len(contexts):
            raise ValueError("Number of messages must match number of contexts")

        tokenizer = tokenizer or self._default_tokenizer

        # Count co-occurrences
        for message, context in zip(messages, contexts):
            tokens = tokenizer(message)

            for token in tokens:
                self._symbol_counts[token] += 1

                for label in context.labels:
                    self._context_counts[label] += 1
                    self._cooccurrence_counts[token][label] += 1

            self._total_observations += 1

        # Compute groundings
        self._compute_groundings()
        self._fitted = True

        return self

    def ground_symbol(self, symbol: str) -> Optional[SymbolMeaning]:
        """
        Get the grounded meaning for a symbol.

        Args:
            symbol: The symbol to ground

        Returns:
            SymbolMeaning if the symbol is known, None otherwise
        """
        if not self._fitted:
            raise RuntimeError("Grounder must be fitted before grounding symbols")

        return self._grounded_symbols.get(symbol)

    def get_all_groundings(self) -> Dict[str, SymbolMeaning]:
        """Get all computed symbol groundings."""
        return dict(self._grounded_symbols)

    def compute_pmi(self, symbol: str, context_label: str) -> float:
        """
        Compute Pointwise Mutual Information between symbol and context.

        PMI(x, y) = log(P(x, y) / (P(x) * P(y)))

        High PMI indicates the symbol and context co-occur more than
        expected by chance, suggesting semantic association.
        """
        if self._total_observations == 0:
            return 0.0

        # Joint probability with smoothing
        joint_count = self._cooccurrence_counts[symbol][context_label] + self.smoothing
        p_joint = joint_count / (self._total_observations + self.smoothing * len(self._symbol_counts))

        # Marginal probabilities
        p_symbol = (self._symbol_counts[symbol] + self.smoothing) / (
            self._total_observations + self.smoothing * len(self._symbol_counts)
        )
        p_context = (self._context_counts[context_label] + self.smoothing) / (
            self._total_observations + self.smoothing * len(self._context_counts)
        )

        # PMI
        if p_symbol * p_context == 0:
            return 0.0

        return math.log(p_joint / (p_symbol * p_context))

    def compute_symbol_vector(self, symbol: str) -> np.ndarray:
        """
        Compute distributional vector for a symbol.

        The vector contains PMI scores for each context label,
        enabling distributional similarity comparisons.
        """
        if symbol not in self._symbol_counts:
            return np.array([])

        context_labels = sorted(self._context_counts.keys())
        vector = np.array([
            self.compute_pmi(symbol, label)
            for label in context_labels
        ])

        return vector

    def find_similar_symbols(
        self,
        symbol: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find symbols with similar distributional properties.

        Uses cosine similarity of PMI vectors.
        """
        if symbol not in self._symbol_counts:
            return []

        target_vector = self.compute_symbol_vector(symbol)
        if len(target_vector) == 0:
            return []

        similarities = []
        for other_symbol in self._symbol_counts:
            if other_symbol == symbol:
                continue

            other_vector = self.compute_symbol_vector(other_symbol)
            if len(other_vector) == 0:
                continue

            # Cosine similarity
            norm_product = np.linalg.norm(target_vector) * np.linalg.norm(other_vector)
            if norm_product > 0:
                sim = np.dot(target_vector, other_vector) / norm_product
                similarities.append((other_symbol, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _compute_groundings(self) -> None:
        """Compute grounded meanings for all symbols."""
        for symbol in self._symbol_counts:
            if self._symbol_counts[symbol] < self.min_support:
                continue

            # Compute PMI for all contexts
            pmi_scores = {}
            for context_label in self._context_counts:
                pmi = self.compute_pmi(symbol, context_label)
                if pmi > self.pmi_threshold:
                    pmi_scores[context_label] = pmi

            if not pmi_scores:
                # No strong associations found
                self._grounded_symbols[symbol] = SymbolMeaning(
                    symbol=symbol,
                    meaning_distribution={"UNKNOWN": 1.0},
                    primary_meaning="UNKNOWN",
                    confidence=0.0,
                    support=self._symbol_counts[symbol],
                    pmi_scores={},
                )
                continue

            # Normalize PMI scores to probability distribution
            # Weight by context frequency to prefer general labels over rare specifics
            weighted_scores = {}
            for label, pmi in pmi_scores.items():
                if pmi > 0:
                    # Boost common labels slightly to prefer general meanings
                    freq_weight = math.sqrt(self._context_counts.get(label, 1))
                    weighted_scores[label] = pmi * (1 + 0.1 * math.log(freq_weight + 1))
                else:
                    weighted_scores[label] = 0

            total_score = sum(weighted_scores.values())
            if total_score > 0:
                meaning_dist = {k: v / total_score for k, v in weighted_scores.items() if v > 0}
            else:
                meaning_dist = {k: 1 / len(pmi_scores) for k in pmi_scores if pmi_scores[k] > 0}

            if not meaning_dist:
                meaning_dist = {"UNKNOWN": 1.0}

            # Primary meaning is highest probability
            primary = max(meaning_dist, key=meaning_dist.get)

            # Confidence based on:
            # 1. How peaked the distribution is (primary meaning dominance)
            # 2. How strongly the symbol is associated (average PMI)
            n_meanings = len(meaning_dist)
            primary_prob = meaning_dist.get(primary, 0.0)

            if n_meanings <= 1:
                # Single meaning = high confidence
                confidence = primary_prob
            else:
                # Confidence from distribution peakedness
                # Compare primary_prob to uniform (1/n)
                uniform_prob = 1.0 / n_meanings
                peak_ratio = primary_prob / uniform_prob if uniform_prob > 0 else 1.0
                peak_confidence = min(1.0, (peak_ratio - 1.0) / max(1.0, n_meanings - 1))

                # Also factor in the strength of association (sum of positive PMIs)
                positive_pmis = [v for v in pmi_scores.values() if v > 0]
                if positive_pmis:
                    avg_pmi = sum(positive_pmis) / len(positive_pmis)
                    # Normalize: PMI of ~1 is strong association
                    pmi_confidence = min(1.0, avg_pmi / 1.0)
                else:
                    pmi_confidence = 0.0

                # Combine: weight peakedness and strength
                confidence = 0.5 * peak_confidence + 0.5 * pmi_confidence

            # Clamp confidence to [0, 1]
            confidence = max(0.0, min(1.0, confidence))

            self._grounded_symbols[symbol] = SymbolMeaning(
                symbol=symbol,
                meaning_distribution=meaning_dist,
                primary_meaning=primary,
                confidence=confidence,
                support=self._symbol_counts[symbol],
                pmi_scores=pmi_scores,
            )

    def _default_tokenizer(self, message: str) -> List[str]:
        """Default tokenizer: split on whitespace and punctuation."""
        # Handle various delimiter patterns
        tokens = re.findall(r'[^\s,;:|\[\]{}()]+', message)
        return tokens


# =============================================================================
# Grammar Induction
# =============================================================================


class GrammarInducer:
    """
    Extracts compositional rules from emergent protocol messages.

    Grammar induction discovers:
    - Positional patterns: What symbols appear at what positions?
    - Recurring subsequences: What n-grams repeat across messages?
    - Delimiter usage: What symbols separate semantic units?
    - Composition rules: How do symbol combinations map to meanings?

    Usage:
        inducer = GrammarInducer()
        grammar = inducer.induce(messages, contexts, grounder)
    """

    def __init__(
        self,
        min_pattern_frequency: int = 3,
        max_pattern_length: int = 5,
        delimiter_threshold: float = 0.3,
    ):
        """
        Initialize the grammar inducer.

        Args:
            min_pattern_frequency: Minimum occurrences for pattern
            max_pattern_length: Maximum length of n-gram patterns
            delimiter_threshold: Frequency threshold for delimiter detection
        """
        self.min_pattern_frequency = min_pattern_frequency
        self.max_pattern_length = max_pattern_length
        self.delimiter_threshold = delimiter_threshold

    def induce(
        self,
        messages: List[str],
        contexts: List[Context],
        grounder: Optional[SymbolGrounder] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> GrammarSketch:
        """
        Induce grammar from messages and their contexts.

        Args:
            messages: List of protocol messages
            contexts: Corresponding contexts
            grounder: Optional pre-fitted symbol grounder
            tokenizer: Optional custom tokenizer

        Returns:
            GrammarSketch with induced rules and patterns
        """
        tokenizer = tokenizer or self._default_tokenizer

        # Tokenize all messages
        tokenized = [tokenizer(msg) for msg in messages]

        # Collect statistics
        all_tokens = [t for tokens in tokenized for t in tokens]
        vocabulary = set(all_tokens)
        message_lengths = [len(t) for t in tokenized]

        # Detect delimiters
        delimiters = self._detect_delimiters(all_tokens, vocabulary)

        # Analyze positional patterns
        positional_patterns = self._analyze_positional_patterns(tokenized)

        # Find recurring subsequences
        ngram_patterns = self._find_ngram_patterns(tokenized)

        # Infer symbol classes from distributional similarity
        symbol_classes = self._infer_symbol_classes(grounder) if grounder else {}

        # Induce composition rules
        rules = self._induce_composition_rules(
            messages, tokenized, contexts, grounder, ngram_patterns
        )

        # Compute compositionality score
        compositionality = self._compute_compositionality(
            tokenized, contexts, grounder
        )

        return GrammarSketch(
            rules=rules,
            delimiters=delimiters,
            positional_patterns=positional_patterns,
            symbol_classes=symbol_classes,
            vocabulary_size=len(vocabulary),
            mean_message_length=np.mean(message_lengths) if message_lengths else 0.0,
            compositionality_score=compositionality,
        )

    def _detect_delimiters(
        self,
        all_tokens: List[str],
        vocabulary: Set[str],
    ) -> Set[str]:
        """Detect delimiter symbols based on frequency patterns."""
        token_counts = Counter(all_tokens)
        total = sum(token_counts.values())

        delimiters = set()
        for token, count in token_counts.items():
            freq = count / total
            # Delimiters are frequent and usually short
            if freq > self.delimiter_threshold and len(token) <= 2:
                delimiters.add(token)

        # Also check for punctuation-like tokens
        for token in vocabulary:
            if all(c in ".,;:|[]{}()-_<>/\\" for c in token):
                delimiters.add(token)

        return delimiters

    def _analyze_positional_patterns(
        self,
        tokenized: List[List[str]],
    ) -> Dict[int, Dict[str, float]]:
        """Analyze what symbols appear at each position."""
        positional_counts: Dict[int, Counter] = defaultdict(Counter)
        positional_totals: Counter = Counter()

        for tokens in tokenized:
            for pos, token in enumerate(tokens):
                positional_counts[pos][token] += 1
                positional_totals[pos] += 1

        # Convert to frequency distributions
        patterns = {}
        for pos in sorted(positional_counts.keys())[:10]:  # Limit to first 10 positions
            total = positional_totals[pos]
            patterns[pos] = {
                token: count / total
                for token, count in positional_counts[pos].most_common(10)
            }

        return patterns

    def _find_ngram_patterns(
        self,
        tokenized: List[List[str]],
    ) -> Dict[Tuple[str, ...], int]:
        """Find recurring n-gram patterns."""
        ngram_counts: Counter = Counter()

        for tokens in tokenized:
            for n in range(2, min(len(tokens) + 1, self.max_pattern_length + 1)):
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i + n])
                    ngram_counts[ngram] += 1

        # Filter by minimum frequency
        return {
            ngram: count
            for ngram, count in ngram_counts.items()
            if count >= self.min_pattern_frequency
        }

    def _infer_symbol_classes(
        self,
        grounder: SymbolGrounder,
    ) -> Dict[str, Set[str]]:
        """Infer symbol classes from distributional similarity."""
        classes: Dict[str, Set[str]] = defaultdict(set)

        groundings = grounder.get_all_groundings()

        # Group by primary meaning
        for symbol, meaning in groundings.items():
            if meaning.confidence > 0.3:
                classes[meaning.primary_meaning].add(symbol)

        # Only keep classes with multiple members
        return {k: v for k, v in classes.items() if len(v) > 1}

    def _induce_composition_rules(
        self,
        messages: List[str],
        tokenized: List[List[str]],
        contexts: List[Context],
        grounder: Optional[SymbolGrounder],
        ngram_patterns: Dict[Tuple[str, ...], int],
    ) -> List[GrammarRule]:
        """Induce composition rules from patterns and contexts."""
        rules = []

        # For each frequent n-gram, try to determine its meaning
        for ngram, frequency in ngram_patterns.items():
            if frequency < self.min_pattern_frequency:
                continue

            # Find all contexts where this n-gram appears
            matching_contexts: List[Context] = []
            examples: List[Tuple[str, str]] = []

            for msg, tokens, ctx in zip(messages, tokenized, contexts):
                # Check if n-gram is in tokens
                for i in range(len(tokens) - len(ngram) + 1):
                    if tuple(tokens[i:i + len(ngram)]) == ngram:
                        matching_contexts.append(ctx)
                        examples.append((msg, ", ".join(ctx.labels)))
                        break

            if not matching_contexts:
                continue

            # Find common context labels
            label_counts: Counter = Counter()
            for ctx in matching_contexts:
                for label in ctx.labels:
                    label_counts[label] += 1

            if not label_counts:
                continue

            # Most common label becomes the combined meaning
            combined_meaning, count = label_counts.most_common(1)[0]
            confidence = count / len(matching_contexts)

            # Try to compose from individual symbol meanings
            if grounder and confidence > 0.3:
                component_meanings = []
                for symbol in ngram:
                    sym_meaning = grounder.ground_symbol(symbol)
                    if sym_meaning and sym_meaning.confidence > 0.3:
                        component_meanings.append(sym_meaning.primary_meaning)

                if len(component_meanings) == len(ngram):
                    # Check if composition matches combined
                    composed = " + ".join(component_meanings)
                    if composed != combined_meaning:
                        # This is a non-trivial composition rule
                        rules.append(GrammarRule(
                            pattern=ngram,
                            combined_meaning=combined_meaning,
                            confidence=confidence,
                            frequency=frequency,
                            examples=examples[:5],
                        ))

        # Sort by confidence and frequency
        rules.sort(key=lambda r: (r.confidence, r.frequency), reverse=True)

        return rules[:20]  # Limit to top 20 rules

    def _compute_compositionality(
        self,
        tokenized: List[List[str]],
        contexts: List[Context],
        grounder: Optional[SymbolGrounder],
    ) -> float:
        """
        Compute compositionality score for the protocol.

        A compositional protocol is one where the meaning of a message
        can be predicted from the meanings of its parts.

        Score 1.0 = fully compositional
        Score 0.0 = non-compositional (holistic meanings)
        """
        if not grounder or len(tokenized) < 10:
            return 0.5  # Unknown

        # For each message, check if combined symbol meanings predict context
        correct_predictions = 0
        total_predictions = 0

        for tokens, ctx in zip(tokenized, contexts):
            if not tokens:
                continue

            # Get primary meanings for each token
            token_meanings = []
            for token in tokens:
                meaning = grounder.ground_symbol(token)
                if meaning and meaning.confidence > 0.3:
                    token_meanings.append(meaning.primary_meaning)

            if not token_meanings:
                continue

            # Check if any context label appears in token meanings
            context_labels = set(ctx.labels)
            predicted_labels = set(token_meanings)

            overlap = context_labels & predicted_labels
            if overlap:
                correct_predictions += len(overlap)

            total_predictions += len(context_labels)

        if total_predictions == 0:
            return 0.5

        return correct_predictions / total_predictions

    def _default_tokenizer(self, message: str) -> List[str]:
        """Default tokenizer."""
        return re.findall(r'[^\s,;:|\[\]{}()]+', message)


# =============================================================================
# Interpretability Scoring
# =============================================================================


def compute_interpretability(
    messages: List[str],
    symbol_groundings: Dict[str, SymbolMeaning],
    grammar: GrammarSketch,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> float:
    """
    Compute overall interpretability score for a protocol.

    Factors:
    - Symbol coverage: What fraction of symbols have grounded meanings?
    - Grammar consistency: How well do messages follow induced rules?
    - Meaning predictability: How confident are the symbol groundings?

    Args:
        messages: Sample of protocol messages
        symbol_groundings: Grounded symbol meanings
        grammar: Induced grammar sketch
        tokenizer: Optional custom tokenizer

    Returns:
        Interpretability score from 0.0 (opaque) to 1.0 (fully decodable)
    """
    tokenizer = tokenizer or (lambda m: re.findall(r'[^\s,;:|\[\]{}()]+', m))

    if not messages:
        return 0.0

    # 1. Symbol coverage
    all_tokens = set()
    for msg in messages:
        all_tokens.update(tokenizer(msg))

    if not all_tokens:
        return 0.0

    grounded_tokens = sum(
        1 for t in all_tokens
        if t in symbol_groundings and symbol_groundings[t].confidence > 0.3
    )
    symbol_coverage = grounded_tokens / len(all_tokens)

    # 2. Grammar consistency
    # Measure how often grammar rules match
    rule_matches = 0
    rule_opportunities = 0

    for msg in messages:
        tokens = tokenizer(msg)
        for rule in grammar.rules:
            pattern = rule.pattern
            for i in range(len(tokens) - len(pattern) + 1):
                if tuple(tokens[i:i + len(pattern)]) == pattern:
                    rule_matches += 1
                rule_opportunities += 1

    grammar_consistency = (
        rule_matches / rule_opportunities
        if rule_opportunities > 0
        else 0.5  # Neutral if no rules
    )

    # 3. Meaning predictability (average confidence of groundings)
    confidences = [
        m.confidence for m in symbol_groundings.values()
        if m.confidence > 0
    ]
    meaning_predictability = np.mean(confidences) if confidences else 0.0

    # 4. Include compositionality
    compositionality = grammar.compositionality_score

    # Weighted combination
    score = (
        0.30 * symbol_coverage +
        0.25 * grammar_consistency +
        0.25 * meaning_predictability +
        0.20 * compositionality
    )

    return float(np.clip(score, 0.0, 1.0))


# =============================================================================
# Glossary Generation
# =============================================================================


def generate_glossary(
    messages: List[str],
    contexts: List[Context],
    min_confidence: float = 0.0,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    grounder: Optional[SymbolGrounder] = None,
) -> Dict[str, str]:
    """
    Generate a human-readable glossary for an emergent protocol.

    The glossary maps symbols to their likely meanings with confidence scores.

    Args:
        messages: Protocol messages
        contexts: Corresponding contexts
        min_confidence: Minimum confidence for inclusion
        tokenizer: Optional custom tokenizer
        grounder: Optional pre-fitted SymbolGrounder to use

    Returns:
        Dictionary mapping symbols to glossary entries
    """
    # Ground symbols (use provided grounder or create new one)
    if grounder is None:
        grounder = SymbolGrounder(min_support=2, pmi_threshold=0.0)
        grounder.fit(messages, contexts, tokenizer)

    groundings = grounder.get_all_groundings()

    glossary = {}
    for symbol, meaning in groundings.items():
        if meaning.confidence >= min_confidence and meaning.primary_meaning != "UNKNOWN":
            # Format entry
            entry = f"{meaning.primary_meaning} (confidence: {meaning.confidence:.0%})"

            # Add alternative meanings if present
            alternatives = [
                (m, p) for m, p in meaning.meaning_distribution.items()
                if m != meaning.primary_meaning and p > 0.15
            ]
            if alternatives:
                alt_str = ", ".join(f"{m} ({p:.0%})" for m, p in alternatives[:3])
                entry += f" | Also: {alt_str}"

            glossary[symbol] = entry
        else:
            glossary[symbol] = f"UNKNOWN (confidence: {meaning.confidence:.0%})"

    return glossary


# =============================================================================
# Translation Lens (Unified Interface)
# =============================================================================


class TranslationLens:
    """
    Unified interface for translating emergent AI protocols.

    Combines symbol grounding, grammar induction, and interpretability
    scoring into a single interface for decoding messages.

    Usage:
        lens = TranslationLens()
        lens.fit(messages, contexts)

        result = lens.decode(new_message)
        print(result.likely_meaning)
        print(result.confidence)
        print(result.unknown_symbols)

        glossary = lens.get_glossary()
        interpretability = lens.get_interpretability()
    """

    def __init__(
        self,
        min_support: int = 2,
        pmi_threshold: float = 0.5,
        min_pattern_frequency: int = 3,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ):
        """
        Initialize the translation lens.

        Args:
            min_support: Minimum symbol occurrences for grounding
            pmi_threshold: Minimum PMI for meaning association
            min_pattern_frequency: Minimum frequency for grammar patterns
            tokenizer: Custom tokenizer function
        """
        self.grounder = SymbolGrounder(
            min_support=min_support,
            pmi_threshold=pmi_threshold,
        )
        self.inducer = GrammarInducer(
            min_pattern_frequency=min_pattern_frequency,
        )
        self.tokenizer = tokenizer or self._default_tokenizer

        self._grammar: Optional[GrammarSketch] = None
        self._glossary: Dict[str, str] = {}
        self._interpretability: float = 0.0
        self._fitted: bool = False
        self._messages: List[str] = []
        self._contexts: List[Context] = []

    def fit(
        self,
        messages: List[str],
        contexts: List[Context],
    ) -> "TranslationLens":
        """
        Fit the translation lens to observed messages and contexts.

        Args:
            messages: Protocol messages
            contexts: Corresponding contexts (states, actions, outcomes)

        Returns:
            self for chaining
        """
        self._messages = messages
        self._contexts = contexts

        # Ground symbols
        self.grounder.fit(messages, contexts, self.tokenizer)

        # Induce grammar
        self._grammar = self.inducer.induce(
            messages, contexts, self.grounder, self.tokenizer
        )

        # Generate glossary using our fitted grounder
        self._glossary = generate_glossary(
            messages, contexts, tokenizer=self.tokenizer, grounder=self.grounder
        )

        # Compute interpretability
        self._interpretability = compute_interpretability(
            messages,
            self.grounder.get_all_groundings(),
            self._grammar,
            self.tokenizer,
        )

        self._fitted = True
        return self

    def decode(self, message: str) -> DecodeResult:
        """
        Decode a new message using learned groundings and grammar.

        Args:
            message: The message to decode

        Returns:
            DecodeResult with likely meaning, confidence, and interpretation
        """
        if not self._fitted:
            raise RuntimeError("TranslationLens must be fitted before decoding")

        tokens = self.tokenizer(message)

        # Collect symbol meanings
        symbol_meanings: Dict[str, SymbolMeaning] = {}
        unknown_symbols: Set[str] = set()
        interpretation_path: List[str] = []

        # Use lower threshold for accepting meanings (0.05 = any positive grounding)
        meaning_threshold = 0.05

        for token in tokens:
            meaning = self.grounder.ground_symbol(token)
            if meaning:
                symbol_meanings[token] = meaning
                if meaning.confidence > meaning_threshold and meaning.primary_meaning != "UNKNOWN":
                    interpretation_path.append(
                        f"{token} -> {meaning.primary_meaning} (conf: {meaning.confidence:.2f})"
                    )
                else:
                    unknown_symbols.add(token)
                    interpretation_path.append(f"{token} -> UNKNOWN (low confidence)")
            else:
                unknown_symbols.add(token)
                interpretation_path.append(f"{token} -> UNKNOWN (not in vocabulary)")

        # Check for grammar rules that apply
        applied_rules: List[GrammarRule] = []
        if self._grammar:
            for rule in self._grammar.rules:
                pattern = rule.pattern
                for i in range(len(tokens) - len(pattern) + 1):
                    if tuple(tokens[i:i + len(pattern)]) == pattern:
                        applied_rules.append(rule)
                        interpretation_path.append(
                            f"Rule: {rule.pattern} -> {rule.combined_meaning}"
                        )

        # Compose likely meaning
        meaning_parts = []
        for token in tokens:
            sm = symbol_meanings.get(token)
            if sm and sm.confidence > meaning_threshold and sm.primary_meaning != "UNKNOWN":
                meaning_parts.append(sm.primary_meaning)
            elif token not in self._grammar.delimiters if self._grammar else True:
                meaning_parts.append(f"[{token}]")

        # Override with rule meanings if applicable
        for rule in applied_rules:
            if rule.confidence > 0.5:
                meaning_parts = [rule.combined_meaning]
                break

        likely_meaning = " ".join(meaning_parts) if meaning_parts else "INDECIPHERABLE"

        # Compute overall confidence
        if symbol_meanings:
            known_confidences = [
                m.confidence for m in symbol_meanings.values()
                if m.confidence > 0
            ]
            confidence = np.mean(known_confidences) if known_confidences else 0.0

            # Adjust for unknown symbols
            unknown_ratio = len(unknown_symbols) / len(tokens) if tokens else 1.0
            confidence *= (1.0 - unknown_ratio * 0.5)
        else:
            confidence = 0.0

        return DecodeResult(
            original_message=message,
            tokens=tokens,
            likely_meaning=likely_meaning,
            confidence=float(confidence),
            symbol_meanings=symbol_meanings,
            applied_rules=applied_rules,
            unknown_symbols=unknown_symbols,
            interpretation_path=interpretation_path,
        )

    def get_glossary(self) -> Dict[str, str]:
        """Get the generated glossary."""
        if not self._fitted:
            raise RuntimeError("TranslationLens must be fitted first")
        return dict(self._glossary)

    def get_grammar(self) -> Optional[GrammarSketch]:
        """Get the induced grammar sketch."""
        if not self._fitted:
            raise RuntimeError("TranslationLens must be fitted first")
        return self._grammar

    def get_interpretability(self) -> float:
        """Get the overall interpretability score."""
        if not self._fitted:
            raise RuntimeError("TranslationLens must be fitted first")
        return self._interpretability

    def find_similar_symbols(self, symbol: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find symbols with similar distributional properties."""
        return self.grounder.find_similar_symbols(symbol, top_k)

    def to_dict(self) -> Dict[str, Any]:
        """Export lens state as dictionary."""
        if not self._fitted:
            return {"error": "Not fitted"}

        return {
            "interpretability": round(self._interpretability, 4),
            "vocabulary_size": len(self.grounder.get_all_groundings()),
            "grammar": self._grammar.to_dict() if self._grammar else None,
            "glossary": self._glossary,
            "symbol_groundings": {
                k: v.to_dict() for k, v in self.grounder.get_all_groundings().items()
            },
        }

    def _default_tokenizer(self, message: str) -> List[str]:
        """Default tokenizer."""
        return re.findall(r'[^\s,;:|\[\]{}()]+', message)


# =============================================================================
# Synthetic Example: Emergent Protocol Demonstration
# =============================================================================


def create_synthetic_protocol_example():
    """
    Create a synthetic emergent protocol to demonstrate the translation lens.

    This simulates a simple coordination protocol between agents where:
    - Symbols encode environmental states and intended actions
    - Grammar rules combine symbols into commands
    - The protocol is partially compositional

    Returns:
        Tuple of (messages, contexts, expected_glossary)
    """
    # Define the "ground truth" protocol
    # Symbols: A=food, B=water, C=shelter, X=need, Y=have, Z=share
    # Grammar: XA = "need food", YB = "have water", ZC = "share shelter"

    # Helper to create clean contexts with simple labels
    def ctx(labels: List[str]) -> Context:
        """Create a context with explicit labels."""
        return Context(type="semantic", data={}, labels=labels)

    messages = [
        # Need food scenarios (X + A)
        "X A",
        "X A",
        "X A",
        "X A",
        "X A",
        "X A",
        # Have water scenarios (Y + B)
        "Y B",
        "Y B",
        "Y B",
        "Y B",
        "Y B",
        # Share shelter scenarios (Z + C)
        "Z C",
        "Z C",
        "Z C",
        "Z C",
        # Combined messages
        "X A Y B",
        "X A Z C",
        "Y B Z C",
        # Need water (X + B)
        "X B",
        "X B",
        "X B",
        "X B",
        # Have food (Y + A)
        "Y A",
        "Y A",
        "Y A",
        # Share food (Z + A)
        "Z A",
        "Z A",
        # Unknown/noise
        "Q R",
        "W W W",
    ]

    contexts = [
        # Need food - X and A appear together with "need" and "food" labels
        ctx(["need", "food", "request"]),
        ctx(["need", "food", "hungry"]),
        ctx(["need", "food"]),
        ctx(["need", "food", "seeking"]),
        ctx(["need", "food", "want"]),
        ctx(["need", "food"]),
        # Have water - Y and B appear together with "have" and "water" labels
        ctx(["have", "water", "offer"]),
        ctx(["have", "water", "available"]),
        ctx(["have", "water"]),
        ctx(["have", "water", "supply"]),
        ctx(["have", "water"]),
        # Share shelter - Z and C appear together
        ctx(["share", "shelter", "offer"]),
        ctx(["share", "shelter"]),
        ctx(["share", "shelter", "cooperative"]),
        ctx(["share", "shelter"]),
        # Combined
        ctx(["need", "food", "have", "water"]),
        ctx(["need", "food", "share", "shelter"]),
        ctx(["have", "water", "share", "shelter"]),
        # Need water - X and B
        ctx(["need", "water", "thirsty"]),
        ctx(["need", "water", "request"]),
        ctx(["need", "water"]),
        ctx(["need", "water"]),
        # Have food - Y and A
        ctx(["have", "food", "offer"]),
        ctx(["have", "food", "available"]),
        ctx(["have", "food"]),
        # Share food - Z and A
        ctx(["share", "food"]),
        ctx(["share", "food", "cooperative"]),
        # Noise
        ctx(["unknown"]),
        ctx(["noise"]),
    ]

    expected_glossary = {
        "X": "need/request (high confidence)",
        "Y": "have/offer (high confidence)",
        "Z": "share (medium confidence)",
        "A": "food (high confidence)",
        "B": "water (high confidence)",
        "C": "shelter (medium confidence)",
    }

    return messages, contexts, expected_glossary


def run_synthetic_example():
    """
    Run the translation lens on a synthetic emergent protocol.

    This demonstrates:
    1. Symbol grounding through co-occurrence
    2. Grammar induction from patterns
    3. Interpretability scoring
    4. Decoding new messages
    """
    print("=" * 70)
    print("TRANSLATION LENS DEMO: Synthetic Emergent Protocol")
    print("=" * 70)

    # Create synthetic data
    messages, contexts, expected = create_synthetic_protocol_example()

    print(f"\nTraining on {len(messages)} messages...")

    # Fit the translation lens with relaxed thresholds for small dataset
    lens = TranslationLens(min_support=2, pmi_threshold=0.0, min_pattern_frequency=2)
    lens.fit(messages, contexts)

    # Show interpretability
    print(f"\nInterpretability Score: {lens.get_interpretability():.2%}")

    # Show glossary
    print("\n--- GENERATED GLOSSARY ---")
    for symbol, meaning in sorted(lens.get_glossary().items()):
        print(f"  {symbol}: {meaning}")

    # Show grammar
    grammar = lens.get_grammar()
    if grammar:
        print(f"\n--- INDUCED GRAMMAR ---")
        print(f"Vocabulary size: {grammar.vocabulary_size}")
        print(f"Mean message length: {grammar.mean_message_length:.1f}")
        print(f"Compositionality: {grammar.compositionality_score:.2%}")

        if grammar.rules:
            print("\nComposition Rules:")
            for rule in grammar.rules[:5]:
                print(f"  {rule}")

        if grammar.delimiters:
            print(f"\nDelimiters: {grammar.delimiters}")

    # Decode new messages
    print("\n--- DECODING NEW MESSAGES ---")
    test_messages = [
        "X A",      # Should decode as "need food"
        "Y B",      # Should decode as "have water"
        "X C",      # Should decode as "need shelter"
        "Z A B",    # Should decode as "share food water"
        "Q Q Q",    # Unknown - should have low confidence
    ]

    for msg in test_messages:
        result = lens.decode(msg)
        print(f"\n  Message: '{msg}'")
        print(f"  Meaning: {result.likely_meaning}")
        print(f"  Confidence: {result.confidence:.2%}")
        if result.unknown_symbols:
            print(f"  Unknown: {result.unknown_symbols}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)

    return lens


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI interface for the Translation Lens."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Translation Lens - Decode emergent AI protocols"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run synthetic protocol demonstration"
    )
    parser.add_argument(
        "--messages",
        type=str,
        help="JSON file with messages array"
    )
    parser.add_argument(
        "--contexts",
        type=str,
        help="JSON file with contexts array"
    )
    parser.add_argument(
        "--decode",
        type=str,
        help="Message to decode (requires fitted lens)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )

    args = parser.parse_args()

    if args.demo:
        run_synthetic_example()
        return

    if args.messages and args.contexts:
        # Load data
        with open(args.messages, 'r') as f:
            messages = json.load(f)
        with open(args.contexts, 'r') as f:
            raw_contexts = json.load(f)

        # Convert raw contexts to Context objects
        contexts = []
        for ctx in raw_contexts:
            if isinstance(ctx, dict):
                contexts.append(Context(
                    type=ctx.get("type", "mixed"),
                    data=ctx.get("data", {}),
                    labels=ctx.get("labels", []),
                ))
            else:
                contexts.append(Context.from_state({"value": ctx}))

        # Fit lens
        lens = TranslationLens()
        lens.fit(messages, contexts)

        # Show results
        results = lens.to_dict()

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results written to {args.output}")
        else:
            print(json.dumps(results, indent=2))

        # Decode if requested
        if args.decode:
            decode_result = lens.decode(args.decode)
            print(f"\nDecoded '{args.decode}':")
            print(f"  Meaning: {decode_result.likely_meaning}")
            print(f"  Confidence: {decode_result.confidence:.2%}")

    else:
        parser.print_help()
        print("\nRun with --demo for a demonstration of the translation lens.")


if __name__ == "__main__":
    main()
