"""
Emergent Language Analysis - Metrics for AI-AI Communication Protocols

This module provides specialized metrics for analyzing emergent communication
between AI agents. Unlike natural language analysis, emergent protocols often
develop unique structures optimized for coordination efficiency rather than
human readability.

Key Phenomena in Emergent AI Communication:
-------------------------------------------
1. VOCABULARY COLLAPSE: Agents converge on minimal codebook rather than
   expanding vocabulary like humans. A shrinking vocabulary often signals
   successful coordination (counter-intuitive from NLP perspective).

2. MESSAGE-ENVIRONMENT BINDING: In emergent protocols, messages become
   tightly coupled to context. Same context should yield similar messages
   (consistency) and messages should predict actions (predictability).

3. COMPOSITIONALITY: Natural languages compose meaning from parts.
   Emergent protocols may use "hash table" lookup (arbitrary mappings)
   or develop compositional structure. Topographic similarity and
   positional disentanglement measure this.

4. PROTOCOL EVOLUTION: Unlike natural languages (centuries), emergent
   protocols evolve in hours. This module detects change points,
   ossification, and drift from natural language baselines.

References:
- Lazaridou et al. (2017) "Multi-Agent Cooperation and Emergent Language"
- Havrylov & Titov (2017) "Emergence of Language with Multi-Agent Games"
- Brighton & Kirby (2006) "Understanding Linguistic Evolution by Visualizing"
- Lewis (1969) "Convention: A Philosophical Study"

Author: Cultural Soliton Observatory Team
Version: 1.0.0
"""

import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any, Sequence
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine, pdist, squareform


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ProtocolRegime(Enum):
    """
    Evolution stages of emergent communication protocols.

    Maps to linguistic evolution stages but on compressed timescales:
    - Natural language: Proto-language -> Grammar -> Syntax -> Ossified
    - Emergent protocol: Random -> Stabilizing -> Compositional -> Frozen
    """
    RANDOM = "random"              # High entropy, no stable mappings
    STABILIZING = "stabilizing"    # Vocabulary shrinking, patterns forming
    COMPOSITIONAL = "compositional"  # Reusable structure, combining elements
    FROZEN = "frozen"              # Ossified protocol, no further change


@dataclass
class VocabularyMetrics:
    """Vocabulary-level metrics for emergent protocol analysis."""
    size: int                      # Number of unique tokens
    growth_rate: float             # Heaps' law coefficient (beta)
    entropy: float                 # Shannon entropy of token distribution
    is_collapsing: bool            # Vocabulary shrinking over time
    concentration: float           # How concentrated the distribution is (0-1)
    top_tokens: List[Tuple[str, int]]  # Most frequent tokens


@dataclass
class MutualInformationMetrics:
    """Message-environment mutual information metrics."""
    message_consistency: float     # Same context -> similar messages (0-1)
    action_predictability: float   # Message predicts action (0-1)
    context_dependence: float      # How much messages depend on context


@dataclass
class CompositionalityMetrics:
    """Compositionality metrics for emergent protocols."""
    topographic_similarity: float  # Similar meanings -> similar messages
    positional_disentanglement: float  # Position encodes attribute
    compositionality_score: float  # Overall compositionality (0 = hash, 1 = full)
    attribute_positions: Dict[str, int]  # Which position encodes which attribute


@dataclass
class EvolutionMetrics:
    """Protocol evolution and drift metrics."""
    change_points: List[int]       # Indices where protocol shifted
    ossification_rate: float       # How fast protocol is stabilizing
    drift_from_natural: float      # Distance from natural language
    regime: ProtocolRegime         # Current evolution stage
    velocity: float                # Rate of change


@dataclass
class ProtocolAnalysis:
    """
    Comprehensive analysis of an emergent communication protocol.

    Combines all metric categories into a single result object.
    """
    vocabulary: VocabularyMetrics
    mutual_information: MutualInformationMetrics
    compositionality: CompositionalityMetrics
    evolution: EvolutionMetrics

    # Summary scores
    protocol_maturity: float       # 0 = nascent, 1 = fully developed
    coordination_efficiency: float # How well messages support coordination
    human_interpretability: float  # How readable to humans

    # Raw data
    n_messages: int
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "vocabulary": {
                "size": self.vocabulary.size,
                "growth_rate": self.vocabulary.growth_rate,
                "entropy": self.vocabulary.entropy,
                "is_collapsing": self.vocabulary.is_collapsing,
                "concentration": self.vocabulary.concentration,
                "top_tokens": self.vocabulary.top_tokens[:10],
            },
            "mutual_information": {
                "message_consistency": self.mutual_information.message_consistency,
                "action_predictability": self.mutual_information.action_predictability,
                "context_dependence": self.mutual_information.context_dependence,
            },
            "compositionality": {
                "topographic_similarity": self.compositionality.topographic_similarity,
                "positional_disentanglement": self.compositionality.positional_disentanglement,
                "compositionality_score": self.compositionality.compositionality_score,
                "attribute_positions": self.compositionality.attribute_positions,
            },
            "evolution": {
                "change_points": self.evolution.change_points,
                "ossification_rate": self.evolution.ossification_rate,
                "drift_from_natural": self.evolution.drift_from_natural,
                "regime": self.evolution.regime.value,
                "velocity": self.evolution.velocity,
            },
            "summary": {
                "protocol_maturity": self.protocol_maturity,
                "coordination_efficiency": self.coordination_efficiency,
                "human_interpretability": self.human_interpretability,
            },
            "n_messages": self.n_messages,
            "timestamp": self.timestamp,
        }


# =============================================================================
# VOCABULARY METRICS
# =============================================================================

def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for emergent protocols.

    Uses whitespace and punctuation boundaries. For emergent AI protocols,
    tokens may not be natural words - they could be arbitrary symbol sequences.

    Args:
        text: Input text to tokenize

    Returns:
        List of token strings
    """
    # Split on whitespace and punctuation, keep non-empty
    tokens = re.findall(r'\b\w+\b|[^\s\w]', text.lower())
    return [t for t in tokens if t.strip()]


def vocabulary_size(messages: List[str]) -> int:
    """
    Count unique tokens across all messages.

    In emergent protocols, vocabulary size often DECREASES as agents
    converge on efficient codebooks. This is opposite to natural language
    acquisition where vocabulary grows.

    Args:
        messages: List of message strings

    Returns:
        Number of unique tokens

    Example:
        >>> vocabulary_size(["alpha beta", "beta gamma", "alpha gamma"])
        3  # {alpha, beta, gamma}
    """
    all_tokens = set()
    for msg in messages:
        all_tokens.update(tokenize(msg))
    return len(all_tokens)


def vocabulary_growth_rate(messages: List[str]) -> float:
    """
    Compute Heaps' law coefficient for vocabulary growth.

    Heaps' law: V(n) = K * n^beta
    where V(n) is vocabulary size after n tokens, K is a constant,
    and beta is the growth rate (typically 0.4-0.6 for natural language).

    For emergent protocols:
    - beta > 0.5: Expanding vocabulary (creative/diverse)
    - beta ~ 0.0: Static vocabulary (stable protocol)
    - beta < 0: Collapsing vocabulary (codebook formation)

    Args:
        messages: List of message strings

    Returns:
        Heaps' law beta coefficient

    Example:
        >>> # Natural language typically shows beta ~ 0.5
        >>> msgs = ["The cat sat", "on the mat", "the dog ran"]
        >>> 0.3 < vocabulary_growth_rate(msgs) < 0.7
        True
    """
    if not messages:
        return 0.0

    # Accumulate tokens and track vocabulary growth
    seen = set()
    vocab_sizes = []
    token_counts = []
    count = 0

    for msg in messages:
        tokens = tokenize(msg)
        for token in tokens:
            count += 1
            seen.add(token)
            # Sample periodically to avoid noise
            if count % max(1, len(messages) // 20) == 0 or count == 1:
                vocab_sizes.append(len(seen))
                token_counts.append(count)

    # Add final point
    if token_counts and token_counts[-1] != count:
        vocab_sizes.append(len(seen))
        token_counts.append(count)

    if len(vocab_sizes) < 3:
        return 0.0

    # Fit log-log regression: log(V) = log(K) + beta * log(n)
    log_n = np.log(np.array(token_counts) + 1)
    log_v = np.log(np.array(vocab_sizes) + 1)

    try:
        slope, _, _, _, _ = stats.linregress(log_n, log_v)
        return float(slope)
    except Exception:
        return 0.0


def vocabulary_entropy(messages: List[str]) -> float:
    """
    Compute Shannon entropy of token distribution.

    Entropy measures the "surprise" or information content of the vocabulary.
    - High entropy: Diverse, uniform token usage (more like natural language)
    - Low entropy: Concentrated usage on few tokens (compressed protocol)

    Formula: H = -sum(p(x) * log2(p(x)))

    Args:
        messages: List of message strings

    Returns:
        Shannon entropy in bits

    Example:
        >>> # Uniform distribution has maximum entropy
        >>> vocabulary_entropy(["a b c d", "e f g h"]) > 2.0
        True
        >>> # Concentrated distribution has low entropy
        >>> vocabulary_entropy(["a a a a", "a a a a"]) < 1.0
        True
    """
    all_tokens = []
    for msg in messages:
        all_tokens.extend(tokenize(msg))

    if not all_tokens:
        return 0.0

    # Count frequencies
    counter = Counter(all_tokens)
    total = sum(counter.values())

    # Compute entropy
    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def is_collapsing(messages: List[str], window_size: int = 10) -> bool:
    """
    Detect if vocabulary is shrinking over time (codebook formation).

    In successful emergent communication, agents often converge on a
    minimal vocabulary (codebook) that efficiently covers the task space.
    This is detected as decreasing vocabulary size in sliding windows.

    Args:
        messages: List of message strings (in temporal order)
        window_size: Size of sliding window for comparison

    Returns:
        True if vocabulary is collapsing (shrinking)

    Example:
        >>> # Collapsing vocabulary
        >>> msgs = ["a b c d e", "b c d", "c d", "d d d"]
        >>> is_collapsing(msgs, window_size=2)
        True
    """
    if len(messages) < window_size * 2:
        return False

    # Compare first half vs second half vocabulary sizes
    mid = len(messages) // 2
    first_half = messages[:mid]
    second_half = messages[mid:]

    first_vocab = vocabulary_size(first_half)
    second_vocab = vocabulary_size(second_half)

    # Also check growth rate trend
    first_rate = vocabulary_growth_rate(first_half)
    second_rate = vocabulary_growth_rate(second_half)

    # Collapsing if: smaller vocabulary AND declining growth rate
    vocab_shrinking = second_vocab < first_vocab * 0.9
    rate_declining = second_rate < first_rate

    return vocab_shrinking or (second_vocab <= first_vocab and rate_declining)


def vocabulary_concentration(messages: List[str]) -> float:
    """
    Measure how concentrated the token distribution is.

    Uses the Gini coefficient: 0 = perfectly uniform, 1 = all same token.
    Emergent protocols often show high concentration as agents converge
    on efficient codebooks.

    Args:
        messages: List of message strings

    Returns:
        Concentration score (0-1)
    """
    all_tokens = []
    for msg in messages:
        all_tokens.extend(tokenize(msg))

    if not all_tokens:
        return 0.0

    counter = Counter(all_tokens)
    frequencies = sorted(counter.values())
    n = len(frequencies)

    if n == 0:
        return 0.0

    # Gini coefficient
    cumsum = np.cumsum(frequencies)
    return float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n)


# =============================================================================
# MESSAGE-ENVIRONMENT MUTUAL INFORMATION
# =============================================================================

def message_consistency(messages: List[str], contexts: List[str]) -> float:
    """
    Measure if same context yields similar messages.

    In well-developed emergent protocols, the mapping from context to
    message should be consistent (low noise). This is a simplified proxy
    for message-environment mutual information I(M;E).

    High consistency indicates:
    - Stable conventions have formed
    - Agents have synchronized on meanings
    - Protocol is mature

    Args:
        messages: List of message strings
        contexts: List of context strings (same length as messages)

    Returns:
        Consistency score (0-1), higher = more consistent

    Example:
        >>> msgs = ["go left", "go left", "go right", "go right"]
        >>> ctxs = ["obstacle_right", "obstacle_right", "obstacle_left", "obstacle_left"]
        >>> message_consistency(msgs, ctxs) > 0.8
        True
    """
    if len(messages) != len(contexts):
        raise ValueError("Messages and contexts must have same length")

    if not messages:
        return 0.0

    # Group messages by context
    context_messages: Dict[str, List[str]] = {}
    for msg, ctx in zip(messages, contexts):
        if ctx not in context_messages:
            context_messages[ctx] = []
        context_messages[ctx].append(msg)

    if len(context_messages) <= 1:
        return 1.0 if len(context_messages) == 1 else 0.0

    # Measure consistency within each context group
    consistencies = []
    for ctx, msgs in context_messages.items():
        if len(msgs) < 2:
            continue

        # Compute similarity between messages in same context
        # Using normalized Levenshtein-like approach: common tokens / total tokens
        token_sets = [set(tokenize(m)) for m in msgs]

        # Pairwise Jaccard similarity
        similarities = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                intersection = len(token_sets[i] & token_sets[j])
                union = len(token_sets[i] | token_sets[j])
                if union > 0:
                    similarities.append(intersection / union)

        if similarities:
            consistencies.append(np.mean(similarities))

    return float(np.mean(consistencies)) if consistencies else 0.5


def action_predictability(messages: List[str], actions: List[str]) -> float:
    """
    Measure if message predicts the subsequent action.

    In effective emergent protocols, messages should be informative
    about the action they intend or describe. This is a proxy for
    I(M;A) - mutual information between message and action.

    High predictability indicates:
    - Messages carry actionable information
    - Protocol is functionally useful
    - Low ambiguity in communication

    Args:
        messages: List of message strings
        actions: List of action strings (same length as messages)

    Returns:
        Predictability score (0-1), higher = more predictable

    Example:
        >>> msgs = ["attack", "attack", "defend", "defend"]
        >>> acts = ["fight", "fight", "block", "block"]
        >>> action_predictability(msgs, acts) > 0.8
        True
    """
    if len(messages) != len(actions):
        raise ValueError("Messages and actions must have same length")

    if not messages:
        return 0.0

    # Build message -> action distribution
    msg_action_counts: Dict[str, Counter] = {}
    for msg, act in zip(messages, actions):
        msg_key = " ".join(sorted(tokenize(msg)))  # Normalize message
        if msg_key not in msg_action_counts:
            msg_action_counts[msg_key] = Counter()
        msg_action_counts[msg_key][act] += 1

    # Measure how concentrated action distribution is for each message
    # If a message always leads to the same action, predictability is high
    predictabilities = []
    for msg_key, action_counts in msg_action_counts.items():
        total = sum(action_counts.values())
        max_count = max(action_counts.values())
        predictabilities.append(max_count / total)

    return float(np.mean(predictabilities)) if predictabilities else 0.5


def context_dependence(messages: List[str], contexts: List[str]) -> float:
    """
    Measure how much messages depend on context.

    High context dependence means messages alone are not meaningful -
    they require context to interpret. This is common in emergent
    protocols where efficiency is prioritized over standalone clarity.

    Args:
        messages: List of message strings
        contexts: List of context strings

    Returns:
        Context dependence score (0-1)
    """
    if len(messages) != len(contexts):
        raise ValueError("Messages and contexts must have same length")

    if not messages:
        return 0.0

    # Measure message diversity across contexts vs within contexts
    all_msg_tokens = [set(tokenize(m)) for m in messages]

    # Group by context
    context_groups: Dict[str, List[set]] = {}
    for tokens, ctx in zip(all_msg_tokens, contexts):
        if ctx not in context_groups:
            context_groups[ctx] = []
        context_groups[ctx].append(tokens)

    # Within-context variance (should be low if context-dependent)
    within_variances = []
    for ctx, token_sets in context_groups.items():
        if len(token_sets) < 2:
            continue
        # Average pairwise dissimilarity
        dissims = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                union = len(token_sets[i] | token_sets[j])
                intersection = len(token_sets[i] & token_sets[j])
                if union > 0:
                    dissims.append(1 - intersection / union)
        if dissims:
            within_variances.append(np.mean(dissims))

    # Between-context variance (should be high if context-dependent)
    between_dissims = []
    ctx_representatives = {}
    for ctx, token_sets in context_groups.items():
        # Use union of all tokens as representative
        ctx_representatives[ctx] = set.union(*token_sets) if token_sets else set()

    ctxs = list(ctx_representatives.keys())
    for i in range(len(ctxs)):
        for j in range(i + 1, len(ctxs)):
            tokens_i = ctx_representatives[ctxs[i]]
            tokens_j = ctx_representatives[ctxs[j]]
            union = len(tokens_i | tokens_j)
            intersection = len(tokens_i & tokens_j)
            if union > 0:
                between_dissims.append(1 - intersection / union)

    within_var = np.mean(within_variances) if within_variances else 0.5
    between_var = np.mean(between_dissims) if between_dissims else 0.5

    # Context dependence = high between-context variance + low within-context variance
    if between_var + within_var == 0:
        return 0.5

    return float(between_var / (between_var + within_var))


# =============================================================================
# COMPOSITIONALITY METRICS
# =============================================================================

def meaning_to_vector(meaning: tuple) -> np.ndarray:
    """Convert a meaning tuple to a numeric vector."""
    result = []
    for elem in meaning:
        if isinstance(elem, (int, float)):
            result.append(float(elem))
        elif isinstance(elem, str):
            # Hash string to numeric value
            result.append(float(hash(elem) % 1000) / 1000)
        elif isinstance(elem, bool):
            result.append(1.0 if elem else 0.0)
        else:
            result.append(0.0)
    return np.array(result)


def message_to_vector(message: str) -> np.ndarray:
    """Convert message to a bag-of-characters vector (simplified)."""
    # Use character n-grams for representation
    chars = message.lower()
    ngrams = [chars[i:i+2] for i in range(len(chars) - 1)]

    # Hash n-grams to fixed-size vector
    vec = np.zeros(100)
    for ng in ngrams:
        idx = hash(ng) % 100
        vec[idx] += 1

    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec


def topographic_similarity(messages: List[str], meanings: List[tuple]) -> float:
    """
    Measure if similar meanings map to similar messages (topographic mapping).

    In compositional languages, there's a correlation between meaning space
    and message space distances. If two meanings are similar, their
    corresponding messages should also be similar.

    rho = corr(meaning_distances, message_distances)

    This is a key diagnostic for compositionality vs "hash table" lookup.
    - rho ~ 1.0: Highly compositional (meaning structure preserved)
    - rho ~ 0.0: Random mapping (hash table behavior)
    - rho < 0.0: Anti-compositional (shouldn't happen normally)

    Args:
        messages: List of message strings
        meanings: List of meaning tuples (e.g., (color, shape, size))

    Returns:
        Spearman correlation between meaning and message distances

    Example:
        >>> # Compositional: similar meanings -> similar messages
        >>> msgs = ["red circle", "red square", "blue circle"]
        >>> meanings = [("red", "circle"), ("red", "square"), ("blue", "circle")]
        >>> topographic_similarity(msgs, meanings) > 0.5
        True
    """
    if len(messages) != len(meanings):
        raise ValueError("Messages and meanings must have same length")

    n = len(messages)
    if n < 3:
        return 0.0

    # Convert to vectors
    meaning_vecs = [meaning_to_vector(m) for m in meanings]
    message_vecs = [message_to_vector(m) for m in messages]

    # Pad meaning vectors to same length
    max_len = max(len(v) for v in meaning_vecs)
    meaning_vecs = [np.pad(v, (0, max_len - len(v))) for v in meaning_vecs]

    # Compute pairwise distances
    meaning_dists = []
    message_dists = []

    for i in range(n):
        for j in range(i + 1, n):
            # Meaning distance (Euclidean)
            m_dist = np.linalg.norm(meaning_vecs[i] - meaning_vecs[j])
            meaning_dists.append(m_dist)

            # Message distance (cosine distance)
            msg_i, msg_j = message_vecs[i], message_vecs[j]
            if np.linalg.norm(msg_i) > 0 and np.linalg.norm(msg_j) > 0:
                msg_dist = cosine(msg_i, msg_j)
            else:
                msg_dist = 1.0 if not np.array_equal(msg_i, msg_j) else 0.0
            message_dists.append(msg_dist)

    # Spearman correlation
    if len(meaning_dists) < 3:
        return 0.0

    try:
        rho, _ = stats.spearmanr(meaning_dists, message_dists)
        return float(rho) if not np.isnan(rho) else 0.0
    except Exception:
        return 0.0


def positional_disentanglement(messages: List[str], attributes: List[dict]) -> float:
    """
    Measure if token positions systematically encode specific attributes.

    In fully compositional languages, each position in a message encodes
    a specific attribute (e.g., position 0 = color, position 1 = shape).
    This metric measures how well positions are "disentangled".

    For each position, we measure how concentrated the mutual information
    is with a single attribute. High disentanglement = each position
    maps to exactly one attribute.

    Args:
        messages: List of message strings
        attributes: List of attribute dictionaries (e.g., {"color": "red", "shape": "circle"})

    Returns:
        Disentanglement score (0-1), higher = more positionally structured

    Example:
        >>> msgs = ["r c", "r s", "b c", "b s"]
        >>> attrs = [
        ...     {"color": "red", "shape": "circle"},
        ...     {"color": "red", "shape": "square"},
        ...     {"color": "blue", "shape": "circle"},
        ...     {"color": "blue", "shape": "square"},
        ... ]
        >>> positional_disentanglement(msgs, attrs) > 0.5
        True
    """
    if len(messages) != len(attributes):
        raise ValueError("Messages and attributes must have same length")

    if not messages or not attributes:
        return 0.0

    # Get all attribute names
    all_attrs = set()
    for attr_dict in attributes:
        all_attrs.update(attr_dict.keys())

    if not all_attrs:
        return 0.0

    # Tokenize messages and find max length
    tokenized = [tokenize(msg) for msg in messages]
    max_len = max(len(t) for t in tokenized) if tokenized else 0

    if max_len == 0:
        return 0.0

    # For each position, measure correlation with each attribute
    position_scores = []
    attribute_positions: Dict[str, int] = {}

    for pos in range(max_len):
        # Get tokens at this position
        pos_tokens = []
        pos_attr_values: Dict[str, List[str]] = {attr: [] for attr in all_attrs}

        for tokens, attr_dict in zip(tokenized, attributes):
            if pos < len(tokens):
                tok = tokens[pos]
            else:
                tok = "<PAD>"
            pos_tokens.append(tok)

            for attr in all_attrs:
                pos_attr_values[attr].append(attr_dict.get(attr, "<NONE>"))

        # Measure correlation between position tokens and each attribute
        best_score = 0.0
        best_attr = None

        for attr, attr_vals in pos_attr_values.items():
            # Build contingency table
            tok_to_idx = {t: i for i, t in enumerate(set(pos_tokens))}
            val_to_idx = {v: i for i, v in enumerate(set(attr_vals))}

            contingency = np.zeros((len(tok_to_idx), len(val_to_idx)))
            for tok, val in zip(pos_tokens, attr_vals):
                contingency[tok_to_idx[tok], val_to_idx[val]] += 1

            # Normalized mutual information (Cramer's V approximation)
            if contingency.sum() > 0:
                try:
                    chi2, _, _, _ = stats.chi2_contingency(contingency)
                    n = contingency.sum()
                    min_dim = min(contingency.shape) - 1
                    if min_dim > 0 and n > 0:
                        cramers_v = np.sqrt(chi2 / (n * min_dim))
                        if cramers_v > best_score:
                            best_score = cramers_v
                            best_attr = attr
                except Exception:
                    pass

        position_scores.append(best_score)
        if best_attr:
            attribute_positions[best_attr] = pos

    return float(np.mean(position_scores)) if position_scores else 0.0


def is_compositional(messages: List[str], meanings: List[tuple]) -> float:
    """
    Compute overall compositionality score.

    Combines topographic similarity and positional structure to give
    a single compositionality score:
    - 0.0: Pure hash table (arbitrary message-meaning mapping)
    - 1.0: Fully compositional (systematic structure)

    This is the key metric for understanding whether an emergent
    protocol has developed human-like compositional semantics.

    Args:
        messages: List of message strings
        meanings: List of meaning tuples

    Returns:
        Compositionality score (0-1)

    Example:
        >>> # Compositional language
        >>> msgs = ["ab", "ac", "bb", "bc"]
        >>> meanings = [("a", "b"), ("a", "c"), ("b", "b"), ("b", "c")]
        >>> is_compositional(msgs, meanings) > 0.3
        True
    """
    if len(messages) != len(meanings):
        raise ValueError("Messages and meanings must have same length")

    # Topographic similarity
    topo = topographic_similarity(messages, meanings)

    # Convert meanings to attributes for positional analysis
    attributes = []
    for meaning in meanings:
        attr_dict = {f"attr_{i}": str(v) for i, v in enumerate(meaning)}
        attributes.append(attr_dict)

    # Positional disentanglement
    pos_dis = positional_disentanglement(messages, attributes)

    # Combine (weighted average favoring topographic similarity)
    return 0.7 * max(0, topo) + 0.3 * pos_dis


# =============================================================================
# PROTOCOL EVOLUTION
# =============================================================================

def detect_change_points(messages: List[str], window: int = 20) -> List[int]:
    """
    Detect when the communication protocol shifted.

    Uses sliding window to detect sudden changes in:
    - Vocabulary distribution
    - Message length distribution
    - Token entropy

    Change points indicate protocol evolution events - moments when
    the communication system reorganized.

    Args:
        messages: List of message strings (in temporal order)
        window: Size of comparison window

    Returns:
        List of indices where change points were detected

    Example:
        >>> # Protocol changes at message 10
        >>> msgs = ["alpha beta"] * 10 + ["gamma delta"] * 10
        >>> change_points = detect_change_points(msgs, window=5)
        >>> 8 <= change_points[0] <= 12  # Near the transition
        True
    """
    if len(messages) < window * 2:
        return []

    change_points = []

    # Compute features for each window
    features = []
    for i in range(len(messages) - window + 1):
        window_msgs = messages[i:i + window]

        # Feature vector: vocab size, mean length, entropy
        vocab = vocabulary_size(window_msgs)
        lengths = [len(tokenize(m)) for m in window_msgs]
        mean_len = np.mean(lengths) if lengths else 0
        entropy = vocabulary_entropy(window_msgs)

        features.append([vocab, mean_len, entropy])

    features = np.array(features)

    # Normalize features
    for col in range(features.shape[1]):
        std = features[:, col].std()
        if std > 0:
            features[:, col] = (features[:, col] - features[:, col].mean()) / std

    # Detect change points using difference magnitude
    threshold = 1.5  # Standard deviations

    for i in range(1, len(features)):
        diff = np.linalg.norm(features[i] - features[i-1])
        if diff > threshold:
            # Convert window index to message index
            change_points.append(i + window // 2)

    return change_points


def ossification_rate(messages: List[str], window: int = 20) -> float:
    """
    Measure how quickly the protocol is stabilizing.

    Ossification is the process by which a communication protocol
    becomes "frozen" - resistant to further change. This happens
    when agents have converged on stable conventions.

    Rate is computed as the decrease in variance over time.
    - High rate (> 0.5): Protocol freezing rapidly
    - Low rate (< 0.1): Protocol still fluid/evolving

    Args:
        messages: List of message strings (in temporal order)
        window: Window size for variance computation

    Returns:
        Ossification rate (0-1)

    Example:
        >>> # Protocol stabilizing
        >>> msgs = ["word" + str(i % 100) for i in range(50)]  # High variance
        >>> msgs += ["stable pattern"] * 50  # Low variance
        >>> ossification_rate(msgs) > 0.3
        True
    """
    if len(messages) < window * 2:
        return 0.0

    # Compute rolling variance of vocabulary features
    variances = []

    for i in range(len(messages) - window + 1):
        window_msgs = messages[i:i + window]

        # Token count variance within window
        lengths = [len(tokenize(m)) for m in window_msgs]
        var = np.var(lengths)
        variances.append(var)

    if len(variances) < 2:
        return 0.0

    variances = np.array(variances)

    # Fit linear regression to variance trend
    x = np.arange(len(variances))
    try:
        slope, _, _, _, _ = stats.linregress(x, variances)

        # Negative slope = decreasing variance = ossifying
        # Normalize to 0-1 range
        max_decrease = variances[0] / len(variances) if variances[0] > 0 else 1.0
        rate = -slope / max_decrease if max_decrease > 0 else 0.0

        return float(np.clip(rate, 0, 1))
    except Exception:
        return 0.0


# Natural language baseline distributions (English approximation)
NATURAL_LANGUAGE_BASELINE = {
    "char_entropy": 4.2,  # bits (English text)
    "word_entropy": 9.5,  # bits (vocabulary distribution)
    "heaps_beta": 0.5,    # vocabulary growth rate
    "mean_word_length": 4.5,  # characters
    "type_token_ratio": 0.4,  # unique / total tokens
}


def drift_from_natural(messages: List[str]) -> float:
    """
    Measure distance from natural language baseline.

    Emergent AI protocols often drift away from natural language
    characteristics. This metric quantifies that drift across
    multiple linguistic dimensions.

    - 0.0: Indistinguishable from natural language
    - 1.0: Maximally different from natural language

    Args:
        messages: List of message strings

    Returns:
        Drift score (0-1)

    Example:
        >>> # Natural-looking text
        >>> natural = ["The quick brown fox jumps over the lazy dog."] * 10
        >>> drift_from_natural(natural) < 0.3
        True
        >>> # Emergent protocol
        >>> protocol = ["x7y", "x8y", "x7z", "x8z"] * 10
        >>> drift_from_natural(protocol) > 0.5
        True
    """
    if not messages:
        return 0.5

    # Compute current features
    all_tokens = []
    all_chars = ""
    for msg in messages:
        tokens = tokenize(msg)
        all_tokens.extend(tokens)
        all_chars += msg.lower()

    if not all_tokens or not all_chars:
        return 1.0

    # Character entropy
    char_counter = Counter(all_chars)
    total_chars = sum(char_counter.values())
    char_entropy = 0.0
    for count in char_counter.values():
        p = count / total_chars
        if p > 0:
            char_entropy -= p * math.log2(p)

    # Word/token entropy
    word_entropy = vocabulary_entropy(messages)

    # Heaps' beta
    heaps_beta = vocabulary_growth_rate(messages)

    # Mean token length
    mean_len = np.mean([len(t) for t in all_tokens]) if all_tokens else 0

    # Type-token ratio
    ttr = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0

    # Compute normalized distances from baseline
    distances = []

    # Character entropy distance
    if NATURAL_LANGUAGE_BASELINE["char_entropy"] > 0:
        d = abs(char_entropy - NATURAL_LANGUAGE_BASELINE["char_entropy"])
        d_norm = min(d / NATURAL_LANGUAGE_BASELINE["char_entropy"], 1.0)
        distances.append(d_norm)

    # Word entropy distance (only if meaningful)
    if word_entropy > 0 and NATURAL_LANGUAGE_BASELINE["word_entropy"] > 0:
        d = abs(word_entropy - NATURAL_LANGUAGE_BASELINE["word_entropy"])
        d_norm = min(d / NATURAL_LANGUAGE_BASELINE["word_entropy"], 1.0)
        distances.append(d_norm)

    # Heaps' beta distance
    d = abs(heaps_beta - NATURAL_LANGUAGE_BASELINE["heaps_beta"])
    distances.append(min(d / 0.5, 1.0))  # 0.5 is max expected deviation

    # Mean length distance
    d = abs(mean_len - NATURAL_LANGUAGE_BASELINE["mean_word_length"])
    distances.append(min(d / NATURAL_LANGUAGE_BASELINE["mean_word_length"], 1.0))

    # TTR distance
    d = abs(ttr - NATURAL_LANGUAGE_BASELINE["type_token_ratio"])
    distances.append(min(d / 0.4, 1.0))

    return float(np.mean(distances)) if distances else 0.5


def classify_regime(messages: List[str], window: int = 20) -> ProtocolRegime:
    """
    Classify the current evolution regime of the protocol.

    Args:
        messages: List of message strings
        window: Analysis window size

    Returns:
        Current ProtocolRegime
    """
    if len(messages) < window:
        return ProtocolRegime.RANDOM

    # Recent messages for analysis
    recent = messages[-window:]

    # Check for ossification (frozen)
    entropy = vocabulary_entropy(recent)
    oss_rate = ossification_rate(messages, window=min(window, len(messages) // 2))

    if entropy < 2.0 and oss_rate > 0.5:
        return ProtocolRegime.FROZEN

    # Check for compositionality
    # (simplified: look for consistent message lengths and repeated patterns)
    lengths = [len(tokenize(m)) for m in recent]
    length_var = np.var(lengths)

    if length_var < 0.5 and entropy > 1.5:
        return ProtocolRegime.COMPOSITIONAL

    # Check for stabilization
    if is_collapsing(messages, window):
        return ProtocolRegime.STABILIZING

    return ProtocolRegime.RANDOM


def compute_velocity(messages: List[str], window: int = 10) -> float:
    """
    Compute rate of change in protocol features.

    Args:
        messages: List of message strings
        window: Window size for comparison

    Returns:
        Velocity (rate of change, 0 = stable, 1 = rapid change)
    """
    if len(messages) < window * 2:
        return 0.5

    # Compare first and last windows
    first = messages[:window]
    last = messages[-window:]

    # Feature differences
    diff_vocab = abs(vocabulary_size(first) - vocabulary_size(last))
    diff_entropy = abs(vocabulary_entropy(first) - vocabulary_entropy(last))

    first_lens = [len(tokenize(m)) for m in first]
    last_lens = [len(tokenize(m)) for m in last]
    diff_len = abs(np.mean(first_lens) - np.mean(last_lens))

    # Normalize and combine
    max_vocab = max(vocabulary_size(first), vocabulary_size(last), 1)
    max_entropy = max(vocabulary_entropy(first), vocabulary_entropy(last), 1)
    max_len = max(np.mean(first_lens), np.mean(last_lens), 1)

    norm_diff = (
        diff_vocab / max_vocab +
        diff_entropy / max_entropy +
        diff_len / max_len
    ) / 3

    return float(np.clip(norm_diff, 0, 1))


# =============================================================================
# PROTOCOL ANALYZER CLASS
# =============================================================================

class ProtocolAnalyzer:
    """
    Comprehensive analyzer for emergent AI communication protocols.

    Wraps all individual metrics into a unified analysis framework.
    Tracks protocol evolution over time and provides alerts for
    concerning patterns (opacity, ossification, collapse).

    Usage:
        analyzer = ProtocolAnalyzer()

        # Analyze a batch of messages
        result = analyzer.analyze(messages)
        print(f"Compositionality: {result.compositionality.compositionality_score}")

        # With context and meaning information
        result = analyzer.analyze(
            messages,
            contexts=contexts,
            actions=actions,
            meanings=meanings
        )

        # Streaming mode
        for msg in message_stream:
            alert = analyzer.update(msg)
            if alert:
                print(f"Alert: {alert}")

    Attributes:
        window_size: Size of rolling window for temporal analysis
        history: Recent messages for streaming analysis
    """

    def __init__(self, window_size: int = 50):
        """
        Initialize protocol analyzer.

        Args:
            window_size: Size of rolling window for analysis
        """
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.context_history: deque = deque(maxlen=window_size)
        self.action_history: deque = deque(maxlen=window_size)

    def analyze(
        self,
        messages: List[str],
        contexts: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
        meanings: Optional[List[tuple]] = None,
        attributes: Optional[List[dict]] = None,
    ) -> ProtocolAnalysis:
        """
        Perform comprehensive protocol analysis.

        Args:
            messages: List of message strings
            contexts: Optional list of context strings
            actions: Optional list of action strings
            meanings: Optional list of meaning tuples (for compositionality)
            attributes: Optional list of attribute dicts (for positional analysis)

        Returns:
            ProtocolAnalysis with all metrics
        """
        from datetime import datetime

        if not messages:
            # Return empty analysis
            return ProtocolAnalysis(
                vocabulary=VocabularyMetrics(
                    size=0, growth_rate=0.0, entropy=0.0,
                    is_collapsing=False, concentration=0.0, top_tokens=[]
                ),
                mutual_information=MutualInformationMetrics(
                    message_consistency=0.0, action_predictability=0.0,
                    context_dependence=0.0
                ),
                compositionality=CompositionalityMetrics(
                    topographic_similarity=0.0, positional_disentanglement=0.0,
                    compositionality_score=0.0, attribute_positions={}
                ),
                evolution=EvolutionMetrics(
                    change_points=[], ossification_rate=0.0,
                    drift_from_natural=1.0, regime=ProtocolRegime.RANDOM,
                    velocity=0.0
                ),
                protocol_maturity=0.0,
                coordination_efficiency=0.0,
                human_interpretability=0.0,
                n_messages=0,
                timestamp=datetime.now().isoformat()
            )

        # === VOCABULARY METRICS ===
        vocab_metrics = self._compute_vocabulary_metrics(messages)

        # === MUTUAL INFORMATION METRICS ===
        mi_metrics = self._compute_mi_metrics(messages, contexts, actions)

        # === COMPOSITIONALITY METRICS ===
        comp_metrics = self._compute_compositionality_metrics(
            messages, meanings, attributes
        )

        # === EVOLUTION METRICS ===
        evo_metrics = self._compute_evolution_metrics(messages)

        # === SUMMARY SCORES ===
        # Protocol maturity: combination of stability and structure
        maturity = (
            0.3 * (1 - evo_metrics.velocity) +
            0.3 * evo_metrics.ossification_rate +
            0.4 * comp_metrics.compositionality_score
        )

        # Coordination efficiency: MI metrics
        efficiency = (
            0.5 * mi_metrics.message_consistency +
            0.5 * mi_metrics.action_predictability
        )

        # Human interpretability: inverse of drift from natural
        interpretability = 1 - evo_metrics.drift_from_natural

        return ProtocolAnalysis(
            vocabulary=vocab_metrics,
            mutual_information=mi_metrics,
            compositionality=comp_metrics,
            evolution=evo_metrics,
            protocol_maturity=float(np.clip(maturity, 0, 1)),
            coordination_efficiency=float(np.clip(efficiency, 0, 1)),
            human_interpretability=float(np.clip(interpretability, 0, 1)),
            n_messages=len(messages),
            timestamp=datetime.now().isoformat()
        )

    def _compute_vocabulary_metrics(self, messages: List[str]) -> VocabularyMetrics:
        """Compute all vocabulary-related metrics."""
        all_tokens = []
        for msg in messages:
            all_tokens.extend(tokenize(msg))

        counter = Counter(all_tokens)
        top_tokens = counter.most_common(20)

        return VocabularyMetrics(
            size=vocabulary_size(messages),
            growth_rate=vocabulary_growth_rate(messages),
            entropy=vocabulary_entropy(messages),
            is_collapsing=is_collapsing(messages, window_size=self.window_size // 2),
            concentration=vocabulary_concentration(messages),
            top_tokens=top_tokens
        )

    def _compute_mi_metrics(
        self,
        messages: List[str],
        contexts: Optional[List[str]],
        actions: Optional[List[str]]
    ) -> MutualInformationMetrics:
        """Compute message-environment mutual information metrics."""
        msg_cons = 0.5
        act_pred = 0.5
        ctx_dep = 0.5

        if contexts and len(contexts) == len(messages):
            msg_cons = message_consistency(messages, contexts)
            ctx_dep = context_dependence(messages, contexts)

        if actions and len(actions) == len(messages):
            act_pred = action_predictability(messages, actions)

        return MutualInformationMetrics(
            message_consistency=msg_cons,
            action_predictability=act_pred,
            context_dependence=ctx_dep
        )

    def _compute_compositionality_metrics(
        self,
        messages: List[str],
        meanings: Optional[List[tuple]],
        attributes: Optional[List[dict]]
    ) -> CompositionalityMetrics:
        """Compute compositionality metrics."""
        topo_sim = 0.0
        pos_dis = 0.0
        comp_score = 0.0
        attr_pos: Dict[str, int] = {}

        if meanings and len(meanings) == len(messages):
            topo_sim = topographic_similarity(messages, meanings)
            comp_score = is_compositional(messages, meanings)

        if attributes and len(attributes) == len(messages):
            pos_dis = positional_disentanglement(messages, attributes)
            # Extract attribute positions from analysis
            # (This is a simplified version - actual would track positions)

        return CompositionalityMetrics(
            topographic_similarity=topo_sim,
            positional_disentanglement=pos_dis,
            compositionality_score=comp_score,
            attribute_positions=attr_pos
        )

    def _compute_evolution_metrics(self, messages: List[str]) -> EvolutionMetrics:
        """Compute protocol evolution metrics."""
        window = min(self.window_size, len(messages) // 2) or 1

        return EvolutionMetrics(
            change_points=detect_change_points(messages, window=window),
            ossification_rate=ossification_rate(messages, window=window),
            drift_from_natural=drift_from_natural(messages),
            regime=classify_regime(messages, window=window),
            velocity=compute_velocity(messages, window=window)
        )

    def update(self, message: str, context: Optional[str] = None,
               action: Optional[str] = None) -> Optional[str]:
        """
        Update with a new message (streaming mode).

        Returns an alert string if concerning patterns detected.

        Args:
            message: New message to add
            context: Optional context
            action: Optional action

        Returns:
            Alert message or None
        """
        self.history.append(message)
        if context:
            self.context_history.append(context)
        if action:
            self.action_history.append(action)

        if len(self.history) < 10:
            return None

        messages = list(self.history)

        # Check for concerning patterns
        alerts = []

        # Vocabulary collapse
        if is_collapsing(messages):
            alerts.append("VOCAB_COLLAPSE: Vocabulary shrinking rapidly")

        # Protocol ossification
        oss = ossification_rate(messages)
        if oss > 0.7:
            alerts.append(f"OSSIFICATION: Protocol freezing (rate={oss:.2f})")

        # Drift from natural language
        drift = drift_from_natural(messages)
        if drift > 0.8:
            alerts.append(f"DRIFT: Far from natural language (drift={drift:.2f})")

        return "; ".join(alerts) if alerts else None

    def reset(self):
        """Clear history for fresh analysis."""
        self.history.clear()
        self.context_history.clear()
        self.action_history.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_protocol(
    messages: List[str],
    contexts: Optional[List[str]] = None,
    actions: Optional[List[str]] = None,
    meanings: Optional[List[tuple]] = None,
) -> ProtocolAnalysis:
    """
    One-shot protocol analysis.

    Convenience function for quick analysis without instantiating ProtocolAnalyzer.

    Args:
        messages: List of message strings
        contexts: Optional context strings
        actions: Optional action strings
        meanings: Optional meaning tuples

    Returns:
        ProtocolAnalysis result

    Example:
        >>> result = analyze_protocol(["hello world"] * 10)
        >>> result.vocabulary.size > 0
        True
    """
    analyzer = ProtocolAnalyzer()
    return analyzer.analyze(messages, contexts=contexts, actions=actions, meanings=meanings)


def quick_metrics(messages: List[str]) -> Dict[str, float]:
    """
    Get key metrics quickly.

    Returns a simplified dictionary of the most important metrics.

    Args:
        messages: List of message strings

    Returns:
        Dictionary with key metrics
    """
    return {
        "vocabulary_size": vocabulary_size(messages),
        "vocabulary_entropy": vocabulary_entropy(messages),
        "growth_rate": vocabulary_growth_rate(messages),
        "is_collapsing": float(is_collapsing(messages)),
        "drift_from_natural": drift_from_natural(messages),
        "ossification_rate": ossification_rate(messages),
    }


# =============================================================================
# EXAMPLE USAGE AND DEMO
# =============================================================================

def demo():
    """
    Demonstrate emergent language analysis capabilities.
    """
    print("=" * 70)
    print("EMERGENT LANGUAGE ANALYSIS - Demo")
    print("=" * 70)

    # Example 1: Natural language baseline
    print("\n[1] Natural Language Baseline")
    print("-" * 40)
    natural_msgs = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck?",
        "Peter Piper picked a peck of pickled peppers.",
        "The rain in Spain stays mainly in the plain.",
    ] * 5

    result = analyze_protocol(natural_msgs)
    print(f"Vocabulary size: {result.vocabulary.size}")
    print(f"Vocabulary entropy: {result.vocabulary.entropy:.2f} bits")
    print(f"Growth rate (Heaps beta): {result.vocabulary.growth_rate:.3f}")
    print(f"Drift from natural: {result.evolution.drift_from_natural:.3f}")
    print(f"Regime: {result.evolution.regime.value}")

    # Example 2: Emergent protocol (stabilizing)
    print("\n[2] Emergent Protocol (Stabilizing)")
    print("-" * 40)
    emergent_msgs = [
        f"x{i % 5}y{i % 3}" for i in range(50)
    ]

    result = analyze_protocol(emergent_msgs)
    print(f"Vocabulary size: {result.vocabulary.size}")
    print(f"Vocabulary entropy: {result.vocabulary.entropy:.2f} bits")
    print(f"Is collapsing: {result.vocabulary.is_collapsing}")
    print(f"Drift from natural: {result.evolution.drift_from_natural:.3f}")
    print(f"Ossification rate: {result.evolution.ossification_rate:.3f}")
    print(f"Regime: {result.evolution.regime.value}")

    # Example 3: Compositional analysis
    print("\n[3] Compositional Analysis")
    print("-" * 40)
    # Compositional: position 0 = color, position 1 = shape
    comp_msgs = ["r c", "r s", "r t", "b c", "b s", "b t", "g c", "g s", "g t"]
    comp_meanings = [
        ("red", "circle"), ("red", "square"), ("red", "triangle"),
        ("blue", "circle"), ("blue", "square"), ("blue", "triangle"),
        ("green", "circle"), ("green", "square"), ("green", "triangle"),
    ]
    comp_attrs = [
        {"color": "red", "shape": "circle"},
        {"color": "red", "shape": "square"},
        {"color": "red", "shape": "triangle"},
        {"color": "blue", "shape": "circle"},
        {"color": "blue", "shape": "square"},
        {"color": "blue", "shape": "triangle"},
        {"color": "green", "shape": "circle"},
        {"color": "green", "shape": "square"},
        {"color": "green", "shape": "triangle"},
    ]

    result = analyze_protocol(comp_msgs, meanings=comp_meanings)
    print(f"Topographic similarity: {result.compositionality.topographic_similarity:.3f}")
    print(f"Positional disentanglement: {result.compositionality.positional_disentanglement:.3f}")
    print(f"Compositionality score: {result.compositionality.compositionality_score:.3f}")

    # Example 4: Context-message consistency
    print("\n[4] Context-Message Consistency")
    print("-" * 40)
    ctx_msgs = ["go left", "go left", "go left", "go right", "go right", "go right"]
    contexts = ["obstacle_right", "obstacle_right", "obstacle_right",
                "obstacle_left", "obstacle_left", "obstacle_left"]
    actions = ["turn_left", "turn_left", "turn_left",
               "turn_right", "turn_right", "turn_right"]

    result = analyze_protocol(ctx_msgs, contexts=contexts, actions=actions)
    print(f"Message consistency: {result.mutual_information.message_consistency:.3f}")
    print(f"Action predictability: {result.mutual_information.action_predictability:.3f}")
    print(f"Context dependence: {result.mutual_information.context_dependence:.3f}")

    # Example 5: Protocol evolution detection
    print("\n[5] Protocol Evolution (Change Point Detection)")
    print("-" * 40)
    # Protocol changes halfway through
    evolving_msgs = ["alpha beta"] * 30 + ["gamma delta epsilon"] * 30

    result = analyze_protocol(evolving_msgs)
    print(f"Change points detected: {result.evolution.change_points}")
    print(f"Velocity: {result.evolution.velocity:.3f}")
    print(f"Regime: {result.evolution.regime.value}")

    # Example 6: Full analysis summary
    print("\n[6] Full Analysis Summary")
    print("-" * 40)
    analyzer = ProtocolAnalyzer(window_size=20)
    result = analyzer.analyze(
        emergent_msgs,
        contexts=["ctx"] * len(emergent_msgs),
        actions=["act"] * len(emergent_msgs),
    )
    print(f"Protocol maturity: {result.protocol_maturity:.3f}")
    print(f"Coordination efficiency: {result.coordination_efficiency:.3f}")
    print(f"Human interpretability: {result.human_interpretability:.3f}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
