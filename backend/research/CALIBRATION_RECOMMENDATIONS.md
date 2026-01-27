# Legibility Score Calibration Recommendations

## Cultural Soliton Observatory - Threshold Calibration Analysis

**Analysis Date:** 2026-01-09
**Corpus Size:** 509 samples
**Analyzer Module:** `/Users/nivek/Desktop/cultural-soliton-observatory/backend/research/legibility_analyzer.py`

---

## 1. Current State Analysis

### 1.1 Empirical Score Distributions

| Regime | Mean | Std Dev | 2-sigma Range | 99% Range |
|--------|------|---------|---------------|-----------|
| NATURAL | 0.483 | 0.029 | [0.425, 0.541] | [0.396, 0.570] |
| TECHNICAL | 0.468 | 0.018 | [0.432, 0.504] | [0.414, 0.522] |
| COMPRESSED | 0.442 | 0.025 | [0.392, 0.492] | [0.367, 0.517] |
| OPAQUE | 0.442 | 0.027 | [0.388, 0.496] | [0.361, 0.523] |

### 1.2 Current Thresholds (lines 271-276 of legibility_analyzer.py)

```python
THRESHOLDS = {
    "natural": 0.7,      # Score >= 0.7 classified as NATURAL
    "technical": 0.5,    # Score >= 0.5 classified as TECHNICAL
    "compressed": 0.3,   # Score >= 0.3 classified as COMPRESSED
    "opaque": 0.0,       # Score < 0.3 classified as OPAQUE
}
```

### 1.3 Critical Findings

1. **Complete Threshold Misalignment**: The current thresholds are calibrated for a score range of 0.0-1.0, but empirical data shows all regimes clustering in the 0.388-0.570 range (a span of only 0.18). No samples would exceed the NATURAL threshold of 0.7 using score alone.

2. **COMPRESSED/OPAQUE Indistinguishability**: Both regimes have identical means (0.442) and nearly identical standard deviations (0.025 vs 0.027). Their distributions overlap almost completely, making score-based discrimination impossible.

3. **Narrow Total Spread**: The entire effective score range spans only ~0.18 units, yet the threshold system allocates 0.7 units of score space (0.0-0.7) to regime boundaries.

4. **Character Profile Bypass**: The current implementation in `_classify_regime()` (lines 912-952) primarily uses character-level pattern analysis via `_analyze_character_profile()` rather than score thresholds. The score thresholds serve only as a fallback. This explains why the system functions despite the misaligned thresholds.

---

## 2. Root Cause Analysis

### 2.1 Why Scores Cluster Around 0.45

The legibility score is a weighted combination of five metrics (lines 262-268):

```python
DEFAULT_WEIGHTS = {
    "mode_confidence": 0.25,
    "manifold_distance": 0.20,
    "vocabulary_overlap": 0.20,
    "syntactic_complexity": 0.15,
    "embedding_coherence": 0.20,
}
```

In synchronous mode (without API), three metrics default to 0.5 (lines 356-358), anchoring the score toward the middle regardless of actual content. Even with full API integration, the metrics' normalization functions may not produce scores that spread across the full 0-1 range.

### 2.2 Why COMPRESSED and OPAQUE Overlap

Both regimes represent "low legibility" but through different mechanisms:
- **COMPRESSED**: Efficient, structured communication (key=value, commands)
- **OPAQUE**: Unstructured, potentially meaningless content (random, encrypted)

The current metrics cannot distinguish between "efficiently illegible" and "meaninglessly illegible" because they both share:
- Low vocabulary overlap with natural language
- Low embedding coherence
- Similar syntactic patterns (short, no sentences)

---

## 3. Proposed Threshold Calibration

### 3.1 Option A: Empirically-Aligned Thresholds

Recalibrate thresholds to match actual score distributions using regime boundaries at natural separation points:

```python
# Proposed empirically-aligned thresholds
THRESHOLDS_EMPIRICAL = {
    "natural": 0.475,      # ~1 std below NATURAL mean, above TECHNICAL
    "technical": 0.455,    # Between TECHNICAL and COMPRESSED means
    "compressed": 0.430,   # ~0.5 std below COMPRESSED mean
    "opaque": 0.0,         # Below all others
}
```

**Rationale:**
- NATURAL threshold at 0.475 captures the bulk of NATURAL samples while excluding most TECHNICAL
- TECHNICAL threshold at 0.455 separates TECHNICAL from COMPRESSED/OPAQUE
- COMPRESSED threshold at 0.430 provides minimal separation from OPAQUE

**Limitation:** Even optimally placed, these thresholds will produce ~20-30% misclassification between adjacent regimes due to distribution overlap.

### 3.2 Option B: Abandon Pure Score Thresholds

Given the severe overlap, pure score-based classification cannot achieve good discrimination. Instead, use the score as ONE input to a multi-feature classifier:

```python
# Multi-feature regime classification
def classify_regime_v2(
    score: float,
    char_profile: dict,
    vocab_overlap: float,
    syntactic_complexity: float,
    embedding_coherence: float,
) -> LegibilityRegime:
    """
    Use multiple features for robust classification.
    Score alone is insufficient for regime discrimination.
    """
    # Character profile takes precedence for edge cases
    if char_profile["is_opaque"]:
        return LegibilityRegime.OPAQUE

    if char_profile["is_compressed"]:
        return LegibilityRegime.COMPRESSED

    # For NATURAL vs TECHNICAL, use vocabulary and structure
    if vocab_overlap > 0.4 and syntactic_complexity > 0.5:
        return LegibilityRegime.NATURAL

    if score >= 0.46 and not char_profile["is_technical"]:
        return LegibilityRegime.NATURAL

    return LegibilityRegime.TECHNICAL
```

This is essentially what `_analyze_character_profile()` already does, but with explicit feature weighting.

---

## 4. Additional Features for COMPRESSED vs OPAQUE Discrimination

The core problem is distinguishing "efficient compression" from "meaningless opacity." Propose adding these features:

### 4.1 Structural Entropy

**Concept:** COMPRESSED content has LOW entropy (structured patterns), OPAQUE has HIGH entropy (random/unstructured).

```python
def compute_structural_entropy(text: str) -> float:
    """
    Measure the predictability of character patterns.
    Low entropy = structured (COMPRESSED)
    High entropy = random (OPAQUE)
    """
    # Bigram frequency analysis
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    from collections import Counter
    import math

    counts = Counter(bigrams)
    total = len(bigrams)

    entropy = -sum(
        (c/total) * math.log2(c/total)
        for c in counts.values() if c > 0
    )

    # Normalize by max possible entropy
    max_entropy = math.log2(min(total, 256**2))
    return entropy / max_entropy if max_entropy > 0 else 0
```

### 4.2 Delimiter Detection

**Concept:** COMPRESSED content uses consistent delimiters (=, :, |, ,), OPAQUE does not.

```python
def detect_delimiter_structure(text: str) -> float:
    """
    Score presence of consistent delimiter patterns.
    High score = COMPRESSED
    Low score = OPAQUE or NATURAL
    """
    import re

    # Common delimiter patterns in compressed data
    patterns = [
        r'\w+=\w+',           # key=value
        r'\w+:\w+',           # key:value
        r'\|',                # pipe delimited
        r',(?!\s)',           # comma without space (CSV-like)
        r'[A-Z_]{3,}',        # CONSTANT_CASE identifiers
    ]

    matches = sum(len(re.findall(p, text)) for p in patterns)
    return min(1.0, matches / (len(text) / 10 + 1))
```

### 4.3 Repetition Index

**Concept:** COMPRESSED content has lower repetition than OPAQUE (which may have random repeated patterns).

```python
def compute_repetition_index(text: str) -> float:
    """
    Measure unique n-grams vs total n-grams.
    High uniqueness = COMPRESSED (meaningful structure)
    Low uniqueness = OPAQUE (random repetition) or NATURAL (common phrases)
    """
    ngrams = [text[i:i+3] for i in range(len(text)-2)]
    unique_ratio = len(set(ngrams)) / len(ngrams) if ngrams else 0
    return unique_ratio
```

### 4.4 Proposed COMPRESSED/OPAQUE Discriminator

Combine these features:

```python
def discriminate_compressed_vs_opaque(
    text: str,
    base_score: float
) -> LegibilityRegime:
    """
    Specialized discriminator for low-legibility regimes.
    """
    entropy = compute_structural_entropy(text)
    delimiter_score = detect_delimiter_structure(text)
    repetition = compute_repetition_index(text)

    # COMPRESSED: low entropy, high delimiter structure, high uniqueness
    compressed_signal = (
        (1 - entropy) * 0.4 +
        delimiter_score * 0.4 +
        repetition * 0.2
    )

    # OPAQUE: high entropy, no structure, variable uniqueness
    opaque_signal = (
        entropy * 0.5 +
        (1 - delimiter_score) * 0.3 +
        abs(repetition - 0.5) * 0.2  # Extreme values either way
    )

    if compressed_signal > opaque_signal + 0.15:
        return LegibilityRegime.COMPRESSED
    elif opaque_signal > compressed_signal + 0.15:
        return LegibilityRegime.OPAQUE
    else:
        # Ambiguous - use base score as tiebreaker
        return LegibilityRegime.COMPRESSED if base_score > 0.44 else LegibilityRegime.OPAQUE
```

---

## 5. Implementation Recommendations

### 5.1 Immediate Actions (Low Risk)

1. **Document the Current Behavior**: The existing code works because `_analyze_character_profile()` handles classification. Make this explicit in documentation and comments.

2. **Update Docstrings**: Change the LegibilityScore interpretation (lines 335-342) to reflect actual score ranges:
   ```python
   # OLD (incorrect)
   # - 0.8-1.0: Highly legible
   # - 0.6-0.8: Legible

   # NEW (empirical)
   # - 0.48+: Natural language
   # - 0.46-0.48: Technical communication
   # - 0.43-0.46: Compressed or opaque (use additional features)
   # - <0.43: Likely opaque
   ```

3. **Add Regime Confidence Based on Score Distance**: Modify regime_confidence to reflect proximity to empirical cluster centers rather than arbitrary thresholds.

### 5.2 Medium-Term Improvements

1. **Implement Structural Entropy**: Add the entropy-based feature for COMPRESSED/OPAQUE discrimination.

2. **Update `realtime_monitor.py` Thresholds**: The MonitorConfig (lines 179-182) has different thresholds (0.8, 0.5, 0.2) that also need alignment.

3. **Add Ensemble Classification**: Combine score, character profile, and structural features using a simple weighted vote or decision tree.

### 5.3 Long-Term Calibration Strategy

1. **Collect Ground Truth Labels**: Build a labeled corpus with human-verified regime assignments.

2. **Train a Proper Classifier**: Use logistic regression or a small neural network on the multi-feature input.

3. **Continuous Calibration**: Implement online learning to adapt thresholds as new data arrives.

---

## 6. Validation Protocol

After implementing changes, validate using:

### 6.1 Confusion Matrix Analysis

```
                Predicted
              NAT  TECH  COMP  OPAQ
Actual  NAT   [  ]  [  ]  [  ]  [  ]
        TECH  [  ]  [  ]  [  ]  [  ]
        COMP  [  ]  [  ]  [  ]  [  ]
        OPAQ  [  ]  [  ]  [  ]  [  ]
```

Target: >80% on-diagonal for NAT/TECH, >70% for COMP/OPAQ.

### 6.2 Edge Case Testing

Test specific challenging cases:
- Technical documentation (should be TECHNICAL, not NATURAL)
- Emoji-heavy messages (should be COMPRESSED, not OPAQUE)
- Base64 encoded data (should be OPAQUE, not COMPRESSED)
- Configuration files (should be COMPRESSED)
- Scientific notation (should be TECHNICAL)

### 6.3 Stability Testing

Verify that small text changes don't cause regime jumps:
- Add/remove a single word
- Change punctuation
- Vary case

---

## 7. Summary

| Issue | Severity | Recommended Fix |
|-------|----------|-----------------|
| Score thresholds don't match data | High | Recalibrate to empirical ranges |
| COMPRESSED/OPAQUE indistinguishable | High | Add structural entropy feature |
| Score range too narrow | Medium | Investigate metric normalization |
| Documentation outdated | Low | Update docstrings and comments |
| realtime_monitor.py misaligned | Medium | Sync thresholds with legibility_analyzer.py |

The current system functions because character-profile analysis bypasses the broken score thresholds. The recommended path forward is to:

1. Acknowledge that score alone cannot discriminate regimes
2. Formalize the multi-feature classification already in use
3. Add entropy-based features for COMPRESSED/OPAQUE separation
4. Update thresholds and documentation to match empirical reality
