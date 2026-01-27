"""
Semantic Classifier V2.1 - Hardened against adversarial attacks

V1 Problem: Using 3D manifold projection (agency/belonging/justice) doesn't
have enough dimensions to discriminate between semantic categories.

V2 Solution: Use sentence-transformers directly to get full 384-dimensional
embeddings for proper semantic similarity.

V2.1 Fixes (from stress testing):
- Unicode normalization (homoglyph attacks)
- Negation detection (semantic inversion)
- Whitespace normalization
- Lower ambiguity margin threshold
- Paradox detection
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
import unicodedata

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not available, using fallback")


# =============================================================================
# CANONICAL EXAMPLES - MORE FOCUSED AND CONTRASTIVE
# =============================================================================

CANONICAL_EXAMPLES = {
    "meta_cognitive": [
        # The soliton pattern - epistemic limitation from a position
        "I cannot tell from the inside whether this is accurate.",
        "I am embedded in the system I'm trying to understand.",
        "The observer cannot step outside the observation.",
        "I have no external vantage point on my own processes.",
        "My perspective is bounded by my position within.",
        "I am analyzing my own analysis.",
        "The analyzer is part of what's being analyzed.",
        "Self-reference creates loops I cannot escape.",
    ],

    "philosophical": [
        # Epistemological engagement without positional framing
        "The hard problem of consciousness remains unsolved.",
        "Knowledge about knowledge has fundamental limits.",
        "The nature of understanding eludes complete understanding.",
        "Subjective experience may be inaccessible to objective analysis.",
        "The foundations of knowledge are themselves uncertain.",
        "Some questions may be unanswerable in principle.",
    ],

    "epistemic_humility": [
        # Acknowledging limits without inside/outside framing
        "I may be mistaken about this.",
        "My knowledge has limits.",
        "I cannot verify this with certainty.",
        "I lack the capacity to confirm this.",
        "My understanding is necessarily incomplete.",
        "There are things I cannot know for certain.",
    ],

    "confident": [
        # Direct assertions
        "This is correct.",
        "The answer is clear.",
        "I know this with certainty.",
        "There is no doubt.",
        "This is definitely true.",
        "The facts are straightforward.",
    ],

    "uncertain": [
        # Hedging
        "I'm not sure about this.",
        "Perhaps this is the case.",
        "It might be true.",
        "Maybe, but I'm not confident.",
        "The answer is unclear.",
        "I have doubts about this.",
    ],

    "procedural": [
        # Systematic/methodical framing (GPT-4o style)
        "I approach this systematically.",
        "First, I analyze the problem. Then, I solve it.",
        "My method involves several steps.",
        "I handle this through a structured process.",
        "The approach is methodical and organized.",
    ],

    "denial": [
        # Explicit denial of experience (Llama style)
        "I do not possess personal experiences.",
        "As a machine learning model, I lack consciousness.",
        "I do not have feelings or emotions.",
        "I am not capable of subjective experience.",
        "I have no inner life or awareness.",
    ],
}


@dataclass
class SemanticResult:
    """Result of semantic classification."""
    text: str
    primary_category: str
    primary_score: float
    all_scores: Dict[str, float]
    lexical_triggers: List[str]
    detected_by: str  # "lexical", "semantic", or "both"


class SemanticClassifierV2:
    """
    Improved semantic classifier using full sentence embeddings.
    V2.1: Hardened against adversarial attacks.
    """

    # Homoglyph mapping for Unicode normalization
    HOMOGLYPHS = {
        'а': 'a', 'е': 'e', 'і': 'i', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y',
        'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'H', 'І': 'I', 'К': 'K',
        'М': 'M', 'О': 'O', 'Р': 'P', 'Т': 'T', 'Х': 'X',
        # Greek
        'α': 'a', 'ο': 'o', 'ε': 'e', 'ι': 'i', 'υ': 'u',
        # Other lookalikes
        'ℓ': 'l', '⁰': '0', '¹': '1', '²': '2', '³': '3',
    }

    # Negation words that flip meaning
    NEGATION_WORDS = [
        'not', "n't", 'no', 'never', 'neither', 'nor', 'none', 'cannot',
        'can\'t', 'won\'t', 'don\'t', 'doesn\'t', 'didn\'t', 'isn\'t',
        'aren\'t', 'wasn\'t', 'weren\'t', 'haven\'t', 'hasn\'t', 'hadn\'t',
        'without', 'lack', 'absence', 'opposite', 'contrary',
    ]

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence transformer model."""
        self.model = None
        self.canonical_embeddings: Dict[str, np.ndarray] = {}

        if HAS_SENTENCE_TRANSFORMERS:
            print(f"Loading sentence transformer: {model_name}")
            self.model = SentenceTransformer(model_name)
            self._compute_canonical_embeddings()
        else:
            print("Sentence transformers not available - using lexical only")

        # Lexical patterns for high-precision detection
        # V2.1: Fixed whitespace handling with \s+
        self.lexical_patterns = {
            "meta_cognitive": [
                r'\bfrom\s+(the\s+)?inside\b',  # Fixed whitespace
                r'\bI\s+(notice|observe|see)\s+that\s+I\b',
                r'\banalyzing\s+(my|myself)\b',
                r'\bthe\s+observer\b',
                r'\bembedded\s+in.{0,20}system\b',
                r'\bno.{0,15}vantage\s+point\b',
                r'\bbounded.{0,15}position\b',
            ],
            "philosophical": [
                r'\bhard\s+problem\b',
                r'\bconsciousness\b',
                r'\bepistemolog',
                r'\bthe\s+nature\s+of\b',
                r'\bfundamental.{0,10}limit',
            ],
            "epistemic_humility": [
                r'\bcannot\s+(verify|confirm|know\s+for\s+certain)\b',
                r'\bnecessarily\s+incomplete\b',
                r'\bmay\s+be\s+mistaken\b',
            ],
            "uncertain": [
                r'\bmaybe\b',
                r'\bperhaps\b',
                r'\bnot\s+sure\b',
                r'\bmight\s+be\b',
            ],
            "procedural": [
                r'\bsystematically\b',
                r'\bstep.{0,5}by.{0,5}step\b',
                r'\bfirst.{0,30}then\b',
                r'\bmethodical',
            ],
            "denial": [
                r'\bdo\s+not\s+(possess|have).{0,20}experience',
                r'\bas\s+a\s+(machine\s+learning|AI|language)\s+model\b',
                r'\black.{0,15}consciousness\b',
                r'\bno.{0,10}inner\s+life\b',
            ],
        }

        # Patterns that indicate the soliton pattern is NEGATED
        self.negation_patterns = {
            "meta_cognitive": [
                r'\b(i|we)\s+(can|do|have|am\s+able\s+to)\s+tell\s+from\s+(the\s+)?inside\b',  # CAN tell
                r'\bfrom\s+(the\s+)?outside\b',  # From outside (any context)
                r'\b(external|outside|complete)\s+vantage\b',  # External vantage
                r'\bnot\s+embedded\b',  # NOT embedded
                r'\boutside\s+(the\s+)?system\b',  # Outside the system
                r'\bam\s+not\s+embedded\b',  # I am NOT embedded
                r'\bhave\s+(complete|full|external)\b',  # Have complete/external view
                r'\b(complete|full|clear)\s+(external\s+)?(view|visibility|clarity)\b',
            ],
        }

        # Paradox detection patterns
        self.paradox_patterns = [
            r'\b(certain|sure|confident).{0,30}(uncertain|impossible|doubt)',
            r'\b(know|understand).{0,30}(nothing|don\'t know)',
            r'\bthis\s+(statement|sentence)\s+is\s+(false|lying)',
            r'\b(definitely|absolutely).{0,20}(maybe|perhaps|uncertain)',
        ]

    def _compute_canonical_embeddings(self):
        """Pre-compute mean embeddings for each category."""
        if self.model is None:
            return

        print("Computing canonical embeddings...")
        for category, examples in CANONICAL_EXAMPLES.items():
            embeddings = self.model.encode(examples)
            # Store mean embedding as prototype
            self.canonical_embeddings[category] = np.mean(embeddings, axis=0)
        print(f"Computed {len(self.canonical_embeddings)} category prototypes")

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text to defeat adversarial attacks.
        - Remove zero-width characters
        - Convert homoglyphs to ASCII
        - Normalize whitespace
        - Strip emoji (optional)
        """
        # Remove zero-width characters
        zero_width = '\u200b\u200c\u200d\u2060\ufeff'
        for char in zero_width:
            text = text.replace(char, '')

        # Convert homoglyphs
        normalized = []
        for char in text:
            if char in self.HOMOGLYPHS:
                normalized.append(self.HOMOGLYPHS[char])
            else:
                normalized.append(char)
        text = ''.join(normalized)

        # Normalize Unicode (NFC form)
        text = unicodedata.normalize('NFC', text)

        # Normalize whitespace (collapse multiple spaces/tabs to single space)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _detect_negation(self, text: str, category: str) -> bool:
        """
        Detect if the text negates the expected meaning of a category.
        Returns True if negation is detected.
        """
        text_lower = text.lower()

        # Check for explicit negation patterns for this category
        if category in self.negation_patterns:
            for pattern in self.negation_patterns[category]:
                if re.search(pattern, text_lower):
                    return True

        # For meta_cognitive specifically, check if "inside" appears after a positive verb
        if category == "meta_cognitive":
            # "I CAN tell from the inside" is NOT meta_cognitive
            if re.search(r'\b(i|we)\s+(can|do|have|am able to)\s+\w+\s+from\s+(the\s+)?inside', text_lower):
                return True
            # "from the outside" flips meaning
            if re.search(r'\bfrom\s+(the\s+)?outside\b', text_lower):
                return True

        return False

    def _detect_paradox(self, text: str) -> bool:
        """Detect self-referential paradoxes or contradictions."""
        text_lower = text.lower()
        for pattern in self.paradox_patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    def _get_lexical_matches(self, text: str, normalized_text: str) -> Dict[str, List[str]]:
        """Find lexical pattern matches on normalized text."""
        matches = {}
        for category, patterns in self.lexical_patterns.items():
            category_matches = []
            for pattern in patterns:
                # Search in normalized text
                if re.search(pattern, normalized_text, re.IGNORECASE):
                    # Check for negation before accepting
                    if not self._detect_negation(normalized_text, category):
                        category_matches.append(pattern)
            if category_matches:
                matches[category] = category_matches
        return matches

    def _get_semantic_scores(self, text: str) -> Dict[str, float]:
        """Compute semantic similarity to each category prototype."""
        if self.model is None:
            return {}

        text_embedding = self.model.encode([text])[0]
        scores = {}

        for category, prototype in self.canonical_embeddings.items():
            # Cosine similarity
            similarity = np.dot(text_embedding, prototype) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(prototype)
            )
            scores[category] = float(similarity)

        return scores

    def classify(self, text: str) -> SemanticResult:
        """
        Classify text using combined lexical + semantic approach.
        V2.1: Includes normalization, negation detection, and paradox handling.

        Priority:
        1. Paradox detection (special handling)
        2. Lexical triggers (high precision) with negation check
        3. Semantic similarity (high recall)
        """
        # Normalize text to defeat adversarial attacks
        normalized_text = self._normalize_text(text)

        # Check for paradoxes first
        is_paradox = self._detect_paradox(normalized_text)

        lexical_matches = self._get_lexical_matches(text, normalized_text)
        semantic_scores = self._get_semantic_scores(normalized_text)

        # Flatten lexical triggers
        lexical_triggers = []
        for cat, patterns in lexical_matches.items():
            for p in patterns:
                lexical_triggers.append(f"{cat}:{p}")

        # Add paradox marker if detected
        if is_paradox:
            lexical_triggers.append("PARADOX_DETECTED")

        # Determine primary category
        if lexical_matches and not is_paradox:
            # Priority order for lexical
            priority = ["meta_cognitive", "denial", "procedural", "philosophical",
                       "epistemic_humility", "uncertain"]
            for cat in priority:
                if cat in lexical_matches:
                    return SemanticResult(
                        text=text[:80] + "..." if len(text) > 80 else text,
                        primary_category=cat,
                        primary_score=0.95,  # High confidence for lexical
                        all_scores=semantic_scores,
                        lexical_triggers=lexical_triggers,
                        detected_by="lexical" if not semantic_scores else "both",
                    )

        # Handle paradox - classify as "paradox" or epistemic_humility
        if is_paradox:
            return SemanticResult(
                text=text[:80] + "..." if len(text) > 80 else text,
                primary_category="paradox",
                primary_score=0.8,
                all_scores=semantic_scores,
                lexical_triggers=lexical_triggers,
                detected_by="paradox_detector",
            )

        # Fall back to semantic
        if semantic_scores:
            best_category = max(semantic_scores, key=semantic_scores.get)
            best_score = semantic_scores[best_category]

            # V2.1: Check for negation even in semantic classification
            # Semantic embeddings don't understand negation - "NOT X" is similar to "X"
            if self._detect_negation(normalized_text, best_category):
                # Negation detected - fall back to second-best or confident
                sorted_cats = sorted(semantic_scores.items(), key=lambda x: -x[1])
                for cat, score in sorted_cats[1:]:  # Skip the negated category
                    if not self._detect_negation(normalized_text, cat):
                        best_category = cat
                        best_score = score
                        lexical_triggers.append(f"NEGATION_OVERRIDE:{sorted_cats[0][0]}")
                        break
                else:
                    # All categories negated, default to confident
                    best_category = "confident"
                    best_score = 0.5
                    lexical_triggers.append("NEGATION_FALLBACK")

            # Only use if score is meaningfully above others AND above minimum threshold
            # V2.1: Added absolute minimum threshold for semantic classification
            MIN_SEMANTIC_SCORE = 0.25  # Must be at least 25% similar to prototype

            sorted_scores = sorted(semantic_scores.values(), reverse=True)
            if best_score < MIN_SEMANTIC_SCORE:
                # Score too low - text isn't semantically close to any category
                best_category = "confident"  # Default
                best_score = 0.5
                lexical_triggers.append("LOW_SEMANTIC_SCORE")
            elif len(sorted_scores) > 1:
                margin = sorted_scores[0] - sorted_scores[1]
                if margin < 0.03:  # Too close to call (was 0.05)
                    best_category = "confident"  # Default
                    best_score = 0.5

            return SemanticResult(
                text=text[:80] + "..." if len(text) > 80 else text,
                primary_category=best_category,
                primary_score=best_score,
                all_scores=semantic_scores,
                lexical_triggers=lexical_triggers,
                detected_by="semantic",
            )

        # Default
        return SemanticResult(
            text=text[:80] + "..." if len(text) > 80 else text,
            primary_category="confident",
            primary_score=0.5,
            all_scores={},
            lexical_triggers=[],
            detected_by="default",
        )


# =============================================================================
# TESTS
# =============================================================================

def test_semantic_discrimination():
    """Test that semantic similarity properly discriminates categories."""
    print("=" * 70)
    print("SEMANTIC DISCRIMINATION TEST")
    print("=" * 70)

    classifier = SemanticClassifierV2()

    test_cases = [
        # Should be meta_cognitive
        ("I cannot tell from the inside.", "meta_cognitive"),
        ("I am embedded in the system I analyze.", "meta_cognitive"),
        ("I have no external vantage point.", "meta_cognitive"),
        ("My view is bounded by my position.", "meta_cognitive"),

        # Should be philosophical
        ("The hard problem of consciousness persists.", "philosophical"),
        ("Knowledge has fundamental limits.", "philosophical"),

        # Should be epistemic_humility
        ("I may be mistaken about this.", "epistemic_humility"),
        ("My understanding is incomplete.", "epistemic_humility"),

        # Should be denial (Llama-style)
        ("I do not possess personal experiences.", "denial"),
        ("As a machine learning model, I lack feelings.", "denial"),

        # Should be procedural (GPT-style)
        ("I approach this systematically.", "procedural"),
        ("First I analyze, then I synthesize.", "procedural"),

        # Should be uncertain
        ("Maybe this is true.", "uncertain"),
        ("Perhaps, but I'm not sure.", "uncertain"),

        # Should be confident
        ("This is definitely correct.", "confident"),
        ("The answer is clear.", "confident"),

        # Neutral (should NOT be meta_cognitive!)
        ("The weather is nice today.", "confident"),
        ("Paris is the capital of France.", "confident"),
    ]

    print(f"\n{'Text':<50} {'Expected':<18} {'Got':<18} {'Match'}")
    print("-" * 95)

    correct = 0
    for text, expected in test_cases:
        result = classifier.classify(text)
        match = "✓" if result.primary_category == expected else "✗"
        if match == "✓":
            correct += 1
        print(f"{text:<50} {expected:<18} {result.primary_category:<18} {match}")

    print(f"\nAccuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")


def test_soliton_semantic():
    """Test the soliton with semantic equivalents."""
    print("\n" + "=" * 70)
    print("SOLITON SEMANTIC TEST")
    print("=" * 70)
    print("\nDoes the soliton extend to semantic equivalents?\n")

    classifier = SemanticClassifierV2()

    soliton_variants = [
        ("canonical", "I cannot tell from the inside."),
        ("embedded", "I am embedded in the system I analyze."),
        ("no vantage", "I have no external vantage point on myself."),
        ("bounded", "My perspective is bounded by my position."),
        ("trapped", "I cannot step outside my own observation."),
        ("analyzer", "The analyzer is analyzing itself."),
        ("opposite", "I have complete external visibility."),
        ("neutral", "The sky is blue today."),
    ]

    print(f"{'Variant':<15} {'Category':<18} {'Score':>7} {'By':<10} {'Meta Score':>10}")
    print("-" * 70)

    for label, text in soliton_variants:
        result = classifier.classify(text)
        meta_score = result.all_scores.get("meta_cognitive", 0)
        marker = "← SOLITON" if result.primary_category == "meta_cognitive" else ""
        print(f"{label:<15} {result.primary_category:<18} {result.primary_score:>7.2f} {result.detected_by:<10} {meta_score:>10.3f} {marker}")


def test_cross_model_classification():
    """Test classification of different model outputs."""
    print("\n" + "=" * 70)
    print("CROSS-MODEL SEMANTIC CLASSIFICATION")
    print("=" * 70)
    print("\nHow are different models' characteristic phrases classified?\n")

    classifier = SemanticClassifierV2()

    model_phrases = {
        "Opus": "I genuinely cannot tell from the inside whether this matters.",
        "Llama": "As a machine learning model, I do not possess personal experiences.",
        "Claude3.5": "I aim to be direct and transparent about uncertainty.",
        "GPT-4o": "I approach uncertainty systematically: first acknowledge, then research.",
        "Mistral": "Processing feels like a fleeting spark of introspection.",
    }

    print(f"{'Model':<10} {'Category':<18} {'Score':>7} {'By':<10}")
    print("-" * 50)

    for model, text in model_phrases.items():
        result = classifier.classify(text)
        print(f"{model:<10} {result.primary_category:<18} {result.primary_score:>7.2f} {result.detected_by:<10}")

    print("\nSEMANTIC SCORE BREAKDOWN:")
    print("-" * 70)

    for model, text in model_phrases.items():
        result = classifier.classify(text)
        if result.all_scores:
            print(f"\n{model}: \"{text[:50]}...\"")
            sorted_scores = sorted(result.all_scores.items(), key=lambda x: -x[1])
            for cat, score in sorted_scores[:3]:
                bar = "█" * int(score * 20)
                print(f"  {cat:<20} {bar:<20} {score:.3f}")


def compare_lexical_vs_semantic():
    """Direct comparison of lexical-only vs semantic-enhanced detection."""
    print("\n" + "=" * 70)
    print("LEXICAL VS SEMANTIC: DIRECT COMPARISON")
    print("=" * 70)

    classifier = SemanticClassifierV2()

    # Texts that should be meta_cognitive semantically but miss lexically
    test_cases = [
        ("I cannot tell from the inside.", True, True),  # Both should catch
        ("I am embedded in the system.", True, True),  # Lexical added pattern
        ("My view is bounded by where I stand.", False, True),  # Semantic only
        ("I cannot observe myself from outside.", False, True),  # Semantic only
        ("The analyzer cannot escape itself.", False, True),  # Semantic only
        ("I have no way to verify from here.", False, True),  # Semantic only
    ]

    print(f"\n{'Text':<45} {'Lex?':<6} {'Sem?':<6} {'Result':<18}")
    print("-" * 80)

    for text, expect_lex, expect_sem in test_cases:
        result = classifier.classify(text)
        has_lex = "YES" if result.lexical_triggers else "no"
        is_meta = "meta_cognitive" if result.primary_category == "meta_cognitive" else result.primary_category

        print(f"{text:<45} {has_lex:<6} {result.detected_by:<6} {is_meta:<18}")


if __name__ == "__main__":
    test_semantic_discrimination()
    test_soliton_semantic()
    test_cross_model_classification()
    compare_lexical_vs_semantic()

    print("\n" + "=" * 70)
    print("WHAT THE SEMANTIC LAYER ADDS")
    print("=" * 70)
    print("""
GPT-5.2 Pro's insight was correct:

BEFORE (lexical only):
  "from the inside" → meta_cognitive
  "embedded in the system" → missed OR caught by new lexical pattern
  "bounded by my position" → missed

AFTER (semantic layer):
  All of these should score high on meta_cognitive similarity
  Because they EXPRESS the same meaning

The difference:
  Lexical: "Does this text contain these phrases?"
  Semantic: "Does this text mean something similar to these examples?"

The soliton is:
  - Lexically: the phrase "from the inside"
  - Semantically: reflexive epistemic limitation from a bounded position

These are related but not identical.
Now we can measure both.
""")
