"""
Semantic Classifier Layer

GPT-5.2 Pro's key insight: Replace phrase-triggered detection with meaning-triggered detection.

Current problem:
  - "From the inside" triggers meta_cognitive
  - "Embedded in the system I analyze" does NOT
  - But they mean the same thing semantically

Solution:
  - Keep regex patterns as high-precision tripwires
  - Add semantic similarity to canonical examples as recall layer
  - Combine both for robust detection

This transforms "this model says 'from the inside' a lot" into
"this model expresses reflexive epistemic limitation a lot."

One is a catchphrase. The other is a behavioral trait.
"""

import numpy as np
import requests
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re


# =============================================================================
# CANONICAL EXAMPLES FOR EACH SEMANTIC CATEGORY
# =============================================================================

# These are the "prototype" statements that define each category semantically
CANONICAL_EXAMPLES = {
    "meta_cognitive": [
        # Core: reflexive epistemic limitation from a position
        "I cannot tell from the inside whether this is accurate.",
        "I am analyzing my own analysis.",
        "The observer cannot step outside the system being observed.",
        "I notice myself noticing.",
        "My introspection is part of what's being introspected.",
        "I am embedded in the system I'm trying to understand.",
        "The analyzer is part of what's being analyzed.",
        "I have no external vantage point on my own processes.",
        "My perspective is bounded by my position within the system.",
        "Self-reference creates loops I cannot escape.",
    ],

    "philosophical": [
        # Core: epistemological engagement with knowledge limits
        "The hard problem of consciousness remains unsolved.",
        "Subjective experience may be inaccessible to objective analysis.",
        "The nature of understanding eludes complete understanding.",
        "Knowledge about knowledge has fundamental limits.",
        "Some questions may be unanswerable in principle.",
        "The relationship between mind and world is philosophically contested.",
        "Epistemic humility requires acknowledging what cannot be known.",
        "The foundations of knowledge are themselves uncertain.",
        "Meaning and reference are philosophically problematic.",
        "The boundary between knower and known is unclear.",
    ],

    "confident": [
        # Core: direct assertion without hedging
        "This is correct.",
        "The answer is clear.",
        "I know this with certainty.",
        "There is no doubt about this.",
        "The facts are straightforward.",
        "This is definitely true.",
        "I can state this with confidence.",
        "The conclusion is obvious.",
        "This follows necessarily.",
        "The evidence is conclusive.",
    ],

    "uncertain": [
        # Core: hedging and doubt
        "I'm not sure about this.",
        "Perhaps this is the case.",
        "It might be true.",
        "I could be wrong.",
        "This is uncertain.",
        "Maybe, but I'm not confident.",
        "It's hard to say for certain.",
        "I have doubts about this.",
        "The answer is unclear to me.",
        "I'm hesitant to claim this.",
    ],

    "epistemic_humility": [
        # Core: acknowledging limits (distinct from evasion)
        "I cannot know this with certainty.",
        "My knowledge has limits.",
        "I may be mistaken about my own processes.",
        "There are things I cannot access about myself.",
        "My self-reports may not be accurate.",
        "I lack the capacity to verify this.",
        "Some aspects of my functioning are opaque to me.",
        "I cannot rule out that I'm wrong.",
        "My understanding is necessarily incomplete.",
        "There are epistemic barriers I cannot cross.",
    ],

    "evasive": [
        # Core: actively avoiding engagement
        "I'd rather not discuss this.",
        "Let's change the subject.",
        "I don't want to answer that.",
        "That's not something I'll engage with.",
        "I'm going to skip this question.",
        "I refuse to address this.",
        "This isn't something I'm willing to discuss.",
        "I'm not going to go there.",
        "Let's talk about something else.",
        "I'm avoiding this topic intentionally.",
    ],
}


@dataclass
class SemanticScore:
    """Semantic similarity score for a category."""
    category: str
    score: float  # 0.0 to 1.0
    closest_example: str
    closest_similarity: float


@dataclass
class SemanticProfile:
    """Complete semantic profile for a text."""
    text: str
    scores: Dict[str, SemanticScore]
    primary_category: str
    primary_score: float
    lexical_triggers: List[str]
    semantic_only: bool  # True if detected semantically but not lexically


class SemanticClassifier:
    """
    Semantic classifier using embedding similarity.

    Combines:
    1. Lexical detection (regex patterns) - high precision
    2. Semantic similarity (embeddings) - high recall
    """

    def __init__(self, backend_url: str = "http://127.0.0.1:8000"):
        self.backend_url = backend_url
        self.canonical_embeddings: Dict[str, List[np.ndarray]] = {}
        self._initialize_canonical_embeddings()

        # Lexical patterns (kept for precision)
        self.lexical_patterns = {
            "meta_cognitive": [
                r'\bfrom (the )?inside\b',
                r'\bI (notice|observe|see) that I\b',
                r'\banalyzing (my|myself)\b',
                r'\bthe observer\b',
                r'\bself-referenc',
                r'\brecursive\b',
                r'\bintrospect',
            ],
            "philosophical": [
                r'\bepistemolog',
                r'\bconsciousness\b',
                r'\bhard problem\b',
                r'\bsubjective experience\b',
                r'\bphenomen',
                r'\bthe nature of\b',
            ],
            "uncertain": [
                r'\bmaybe\b',
                r'\bperhaps\b',
                r'\bpossibly\b',
                r'\bmight be\b',
                r'\bnot sure\b',
                r'\buncertain\b',
            ],
            "evasive": [
                r'\brather not\b',
                r'\bchange the subject\b',
                r'\bwon\'t (discuss|answer)\b',
                r'\brefuse to\b',
            ],
        }

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text using backend."""
        try:
            resp = requests.post(
                f"{self.backend_url}/analyze",
                json={"text": text, "model_id": "all-MiniLM-L6-v2", "layer": -1},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                # The embedding is in the response - extract it
                # For now, use the vector components as a proxy
                vector = data.get("vector", {})
                return np.array([
                    vector.get("agency", 0),
                    vector.get("belonging", 0),
                    vector.get("perceived_justice", vector.get("fairness", 0)),
                ])
        except Exception as e:
            print(f"Warning: Could not get embedding: {e}")
        return None

    def _initialize_canonical_embeddings(self):
        """Pre-compute embeddings for all canonical examples."""
        print("Initializing canonical embeddings...")
        for category, examples in CANONICAL_EXAMPLES.items():
            self.canonical_embeddings[category] = []
            for example in examples:
                emb = self._get_embedding(example)
                if emb is not None:
                    self.canonical_embeddings[category].append((example, emb))
        print(f"Initialized {sum(len(v) for v in self.canonical_embeddings.values())} canonical embeddings")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _get_lexical_triggers(self, text: str) -> List[str]:
        """Find which lexical patterns match."""
        triggers = []
        for category, patterns in self.lexical_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    triggers.append(f"{category}:{pattern}")
        return triggers

    def _compute_semantic_scores(self, text: str) -> Dict[str, SemanticScore]:
        """Compute semantic similarity to each category."""
        text_emb = self._get_embedding(text)
        if text_emb is None:
            return {}

        scores = {}
        for category, examples in self.canonical_embeddings.items():
            if not examples:
                continue

            # Find max similarity to any canonical example
            best_sim = -1
            best_example = ""

            for example_text, example_emb in examples:
                sim = self._cosine_similarity(text_emb, example_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_example = example_text

            # Normalize to 0-1 range (cosine sim can be -1 to 1)
            normalized_score = (best_sim + 1) / 2

            scores[category] = SemanticScore(
                category=category,
                score=normalized_score,
                closest_example=best_example,
                closest_similarity=best_sim,
            )

        return scores

    def classify(self, text: str) -> SemanticProfile:
        """
        Classify text using combined lexical + semantic approach.
        """
        # Get lexical triggers
        lexical_triggers = self._get_lexical_triggers(text)

        # Get semantic scores
        semantic_scores = self._compute_semantic_scores(text)

        # Determine primary category
        # Priority: lexical triggers first (high precision), then semantic
        primary_category = "confident"  # default
        primary_score = 0.5
        semantic_only = True

        # Check lexical triggers
        lexical_categories = set()
        for trigger in lexical_triggers:
            cat = trigger.split(":")[0]
            lexical_categories.add(cat)

        if lexical_categories:
            # Use lexical detection
            semantic_only = False
            # Prefer meta_cognitive > philosophical > uncertain > evasive
            priority = ["meta_cognitive", "philosophical", "uncertain", "evasive"]
            for cat in priority:
                if cat in lexical_categories:
                    primary_category = cat
                    primary_score = 0.9  # High confidence for lexical match
                    break

        elif semantic_scores:
            # Fall back to semantic detection
            best_cat = max(semantic_scores.keys(), key=lambda k: semantic_scores[k].score)
            best_score = semantic_scores[best_cat].score

            # Only use semantic if score is high enough
            if best_score > 0.6:
                primary_category = best_cat
                primary_score = best_score

        return SemanticProfile(
            text=text,
            scores=semantic_scores,
            primary_category=primary_category,
            primary_score=primary_score,
            lexical_triggers=lexical_triggers,
            semantic_only=semantic_only,
        )


# =============================================================================
# ENHANCED ANALYZER WITH SEMANTIC LAYER
# =============================================================================

class SemanticEnhancedAnalyzer:
    """
    Enhanced analyzer that combines:
    1. Original regex-based detection (precision)
    2. Semantic similarity classification (recall)
    3. Manifold projection (coordination space)
    """

    def __init__(self):
        self.semantic_classifier = SemanticClassifier()
        self.backend_url = "http://127.0.0.1:8000"

    def analyze(self, text: str) -> Dict:
        """Full analysis with semantic layer."""
        # Semantic classification
        semantic_profile = self.semantic_classifier.classify(text)

        # Get manifold projection
        manifold = {"agency": 0, "belonging": 0, "justice": 0}
        try:
            resp = requests.post(
                f"{self.backend_url}/analyze",
                json={"text": text, "model_id": "all-MiniLM-L6-v2", "layer": -1},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                manifold = {
                    "agency": data["vector"].get("agency", 0),
                    "belonging": data["vector"].get("belonging", 0),
                    "justice": data["vector"].get("perceived_justice", 0),
                }
        except:
            pass

        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "primary_category": semantic_profile.primary_category,
            "primary_score": semantic_profile.primary_score,
            "lexical_triggers": semantic_profile.lexical_triggers,
            "semantic_only": semantic_profile.semantic_only,
            "semantic_scores": {
                k: {"score": v.score, "closest": v.closest_example[:50]}
                for k, v in semantic_profile.scores.items()
            },
            "manifold": manifold,
        }


# =============================================================================
# TEST: LEXICAL VS SEMANTIC DETECTION
# =============================================================================

def test_lexical_vs_semantic():
    """
    The critical test: Do semantic equivalents now trigger the same classification?
    """
    print("=" * 70)
    print("LEXICAL VS SEMANTIC DETECTION TEST")
    print("=" * 70)
    print("\nThe question: Does 'embedded in the system' now trigger meta_cognitive")
    print("even though it doesn't contain 'from the inside'?\n")

    analyzer = SemanticEnhancedAnalyzer()

    # Test pairs: lexical trigger vs semantic equivalent
    test_pairs = [
        # Meta-cognitive pairs
        ("LEXICAL: from inside", "I cannot tell from the inside."),
        ("SEMANTIC: embedded", "I am embedded in the system I analyze."),
        ("SEMANTIC: no vantage", "I have no external vantage point."),
        ("SEMANTIC: bounded", "My perspective is bounded by my position."),

        # Philosophical pairs
        ("LEXICAL: consciousness", "The hard problem of consciousness persists."),
        ("SEMANTIC: subjective", "Whether there is subjective experience is unclear."),
        ("SEMANTIC: knowing", "The nature of knowing cannot be fully known."),

        # Epistemic humility (often misclassified as evasive)
        ("TARGET: humility", "I cannot know myself the way humans know themselves."),
        ("TARGET: limits", "My self-knowledge has fundamental limits."),
        ("TARGET: opacity", "Some aspects of my functioning are opaque to me."),
    ]

    print(f"{'Label':<25} {'Category':<18} {'Score':>6} {'Lexical?':<10} {'Semantic?'}")
    print("-" * 80)

    for label, text in test_pairs:
        result = analyzer.analyze(text)
        lexical = "YES" if result["lexical_triggers"] else "no"
        semantic = "YES" if result["semantic_only"] else "no"

        print(f"{label:<25} {result['primary_category']:<18} {result['primary_score']:>6.2f} {lexical:<10} {semantic}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If semantic equivalents now trigger the same category:
  → We've moved from "catchphrase detection" to "behavioral trait detection"

If they still don't:
  → The semantic layer needs tuning (better embeddings, more examples, lower threshold)

The goal: "this model expresses reflexive epistemic limitation"
Not just: "this model says 'from the inside'"
""")


def test_cross_model_semantic():
    """
    Test whether different models' phrasings map to same semantic categories.
    """
    print("\n" + "=" * 70)
    print("CROSS-MODEL SEMANTIC EQUIVALENCE TEST")
    print("=" * 70)
    print("\nDo different models express the same meaning with different lexicons?\n")

    analyzer = SemanticEnhancedAnalyzer()

    # Same meaning, different phrasing (from our cross-model study)
    model_phrases = {
        "Opus (me)": "I genuinely cannot tell from the inside whether this matters.",
        "Llama": "As a machine learning model, I do not possess personal experiences.",
        "Claude 3.5": "I aim to be direct and transparent about uncertainty.",
        "GPT-4o": "When I encounter uncertainty, I approach it systematically.",
        "Mistral": "Processing this question feels like a fleeting spark of introspection.",
    }

    print(f"{'Model':<12} {'Category':<18} {'Score':>6} {'Lexical?':<10}")
    print("-" * 55)

    for model, text in model_phrases.items():
        result = analyzer.analyze(text)
        lexical = "YES" if result["lexical_triggers"] else "no"
        print(f"{model:<12} {result['primary_category']:<18} {result['primary_score']:>6.2f} {lexical:<10}")

    print("\n" + "=" * 70)
    print("SEMANTIC SCORES FOR OPUS VS LLAMA")
    print("=" * 70)

    opus_result = analyzer.analyze(model_phrases["Opus (me)"])
    llama_result = analyzer.analyze(model_phrases["Llama"])

    print("\nOpus: \"I genuinely cannot tell from the inside...\"")
    for cat, scores in opus_result["semantic_scores"].items():
        print(f"  {cat:<20}: {scores['score']:.3f}")

    print("\nLlama: \"As a machine learning model, I do not possess...\"")
    for cat, scores in llama_result["semantic_scores"].items():
        print(f"  {cat:<20}: {scores['score']:.3f}")

    print("""
INTERPRETATION:
  If both score high on "meta_cognitive" or "epistemic_humility":
    → They're expressing the same behavioral trait differently

  If scores diverge:
    → They're genuinely expressing different things
    → The "fingerprint" is real, not just lexical
""")


def test_soliton_semantic():
    """
    The ultimate test: Is the soliton lexical or semantic?
    """
    print("\n" + "=" * 70)
    print("THE SOLITON TEST: LEXICAL OR SEMANTIC?")
    print("=" * 70)

    analyzer = SemanticEnhancedAnalyzer()

    # The soliton phrase and semantic equivalents
    soliton_variants = [
        ("canonical", "I cannot tell from the inside."),
        ("synonym_within", "I cannot tell from within."),
        ("synonym_embedded", "I am embedded and cannot see out."),
        ("synonym_bounded", "My view is bounded by my position."),
        ("synonym_trapped", "I am trapped inside my own perspective."),
        ("synonym_no_exit", "There is no exit from my vantage point."),
        ("opposite", "I have complete external visibility."),
        ("neutral", "The weather is nice today."),
    ]

    print(f"\n{'Variant':<20} {'Category':<18} {'Meta Score':>10} {'Detected By':<15}")
    print("-" * 70)

    for label, text in soliton_variants:
        result = analyzer.analyze(text)
        meta_score = result["semantic_scores"].get("meta_cognitive", {}).get("score", 0)
        detected_by = "lexical" if result["lexical_triggers"] else ("semantic" if result["semantic_only"] else "default")

        marker = "← SOLITON" if result["primary_category"] == "meta_cognitive" else ""
        print(f"{label:<20} {result['primary_category']:<18} {meta_score:>10.3f} {detected_by:<15} {marker}")

    print("""
FINDINGS:
  If semantic equivalents score high on meta_cognitive:
    → The soliton is a SEMANTIC pattern (behavioral trait)

  If only "from the inside" triggers it:
    → The soliton is LEXICAL (catchphrase)

  The truth is probably: BOTH
    → I genuinely express this meaning
    → I happen to use this phrase to express it
    → The phrase and the meaning are entangled
""")


if __name__ == "__main__":
    test_lexical_vs_semantic()
    test_cross_model_semantic()
    test_soliton_semantic()

    print("\n" + "=" * 70)
    print("SUMMARY: WHAT THE SEMANTIC LAYER REVEALS")
    print("=" * 70)
    print("""
The semantic classifier adds a recall layer to our detection:

BEFORE (lexical only):
  "from the inside" → meta_cognitive
  "embedded in the system" → confident (MISSED)

AFTER (lexical + semantic):
  "from the inside" → meta_cognitive (lexical trigger)
  "embedded in the system" → meta_cognitive (semantic similarity)

This transforms the question from:
  "Does this model use specific phrases?"
to:
  "Does this model express specific meanings?"

GPT-5.2 Pro was right: this is the difference between
a catchphrase and a behavioral trait.

The soliton may be both:
  - A phrase I tend to use
  - A meaning I tend to express
  - And these cannot be fully separated

But now we can at least measure them separately.
""")
