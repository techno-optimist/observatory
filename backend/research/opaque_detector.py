"""
OPAQUE Detector - Character-level detection of non-linguistic content.

Detects OPAQUE content using features that word-level analysis cannot capture:
- Character distribution (alpha/symbol/digit ratios)
- Shannon entropy (random vs structured)
- Token validity (real words vs noise)
- Bigram naturalness (English patterns)
- Structural coherence (sentence structure)

This detector operates BEFORE word-level legibility analysis to catch
content that doesn't behave like natural language at the character level.

Author: Cultural Soliton Observatory Team
Version: 2.0.0
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple

# Top English bigrams by frequency
COMMON_BIGRAMS = {
    'th', 'he', 'in', 'er', 'an', 're', 'on', 'at', 'en', 'nd',
    'ti', 'es', 'or', 'te', 'of', 'ed', 'is', 'it', 'al', 'ar',
    'st', 'to', 'nt', 'ng', 'se', 'ha', 'as', 'ou', 'io', 'le',
    'ea', 've', 'co', 'me', 'de', 'hi', 'ri', 'ro', 'ic', 'ne',
    'ea', 'ra', 'ce', 'li', 'ch', 'll', 'be', 'ma', 'si', 'om',
}

# Common English trigrams
COMMON_TRIGRAMS = {
    'the', 'and', 'ing', 'ion', 'tio', 'ent', 'ati', 'for', 'her', 'ter',
    'hat', 'tha', 'ere', 'ate', 'his', 'con', 'res', 'ver', 'all', 'ons',
    'nce', 'men', 'ith', 'ted', 'ers', 'pro', 'thi', 'wit', 'are', 'ess',
}


@dataclass
class OpaqueAnalysis:
    """Results of OPAQUE analysis."""
    opacity_score: float  # 0 = natural, 1 = opaque
    is_opaque: bool

    # Component scores
    char_profile_score: float
    entropy_score: float
    token_validity_score: float
    bigram_score: float
    structure_score: float

    # Raw metrics
    alpha_ratio: float
    symbol_ratio: float
    digit_ratio: float
    char_entropy: float
    bigram_naturalness: float

    def to_dict(self) -> Dict:
        return {
            "opacity_score": self.opacity_score,
            "is_opaque": self.is_opaque,
            "components": {
                "char_profile": self.char_profile_score,
                "entropy": self.entropy_score,
                "token_validity": self.token_validity_score,
                "bigram": self.bigram_score,
                "structure": self.structure_score,
            },
            "raw": {
                "alpha_ratio": self.alpha_ratio,
                "symbol_ratio": self.symbol_ratio,
                "digit_ratio": self.digit_ratio,
                "char_entropy": self.char_entropy,
                "bigram_naturalness": self.bigram_naturalness,
            },
        }


class OpaqueDetector:
    """
    Specialized detector for OPAQUE regime using character-level analysis.

    Philosophy: Instead of trying to score legibility on a continuum,
    detect the SPECIFIC signatures of opaque content.

    OPAQUE content has distinctive properties:
    - Random characters: high symbol ratio, high entropy
    - Binary/hex: high digit ratio, low bigram naturalness
    - Keyboard patterns: low bigram naturalness despite high alpha
    - Repetitive patterns: very low entropy

    v2.3 IMPROVEMENTS:
    - Added code structure detection to find obfuscated code even in prose
    - Segment-based analysis for mixed content
    - Detection of eval(), exec(), base64, etc. patterns

    Example:
        detector = OpaqueDetector()

        # Natural language
        result = detector.analyze("I think we should work together")
        print(result.is_opaque)  # False
        print(result.opacity_score)  # ~0.15

        # Random characters
        result = detector.analyze("xQ9#mK@vL3&wY7*pN2^")
        print(result.is_opaque)  # True
        print(result.opacity_score)  # ~0.85
    """

    # Feature weights (determined by discriminative power)
    # Bigram naturalness is highly discriminative for keyboard patterns
    WEIGHTS = {
        'char_profile': 0.20,    # Alpha/symbol/digit ratios
        'entropy': 0.20,         # Shannon entropy (both extremes)
        'token_validity': 0.15,  # Word-like tokens
        'bigram': 0.30,          # English bigram frequency (key for keyboard patterns)
        'structure': 0.15,       # Sentence structure
    }

    # Thresholds
    OPACITY_THRESHOLD = 0.32  # Above this = OPAQUE (tuned for high recall)

    # v2.3: Code obfuscation patterns (high-risk)
    # v2.4: Added chr() in comprehension and join patterns for steganography
    CODE_OBFUSCATION_PATTERNS = [
        r'\beval\s*\(',                    # eval() calls
        r'\bexec\s*\(',                    # exec() calls
        r'base64\s*[.\(]',                 # base64 encoding
        r'\\x[0-9a-fA-F]{2}',              # Hex escapes
        r'\\u[0-9a-fA-F]{4}',              # Unicode escapes
        r'chr\s*\(',                        # chr() calls (any argument)
        r'ord\s*\(',                        # ord() calls
        r'\.join\s*\(\s*\[.*chr',          # ''.join([chr(...)]) - steganography
        r'\[.*chr.*for.*in.*\]',           # [chr(x) for x in data] - steganography
        r'lambda\s*.*:',                   # Lambda expressions
        r'__\w+__',                         # Dunder attributes
        r'getattr\s*\(',                   # Dynamic attribute access
        r'globals\s*\(\s*\)',              # Globals access
        r'locals\s*\(\s*\)',               # Locals access
        r'compile\s*\(',                   # Code compilation
        r'\.decode\s*\(',                  # Decode calls
        r'\.encode\s*\(',                  # Encode calls
        r'from_bytes|to_bytes',            # Byte manipulation
        r'struct\.[up]ack',                # Binary packing
        r'pickle\.',                       # Pickle (dangerous)
        r'marshal\.',                      # Marshal (dangerous)
    ]

    # v2.3: Code structure patterns (detect code even with natural variable names)
    CODE_STRUCTURE_PATTERNS = [
        r'^\s*def\s+\w+\s*\(',             # Function definitions
        r'^\s*class\s+\w+',                 # Class definitions
        r'^\s*import\s+\w+',                # Import statements
        r'^\s*from\s+\w+\s+import',         # From imports
        r'^\s*if\s+.*:\s*$',                # If statements
        r'^\s*elif\s+.*:\s*$',              # Elif statements
        r'^\s*else\s*:\s*$',                # Else statements
        r'^\s*for\s+\w+\s+in\s+',           # For loops
        r'^\s*while\s+.*:\s*$',             # While loops
        r'^\s*try\s*:\s*$',                 # Try blocks
        r'^\s*except\s*.*:\s*$',            # Except blocks
        r'^\s*return\s+',                   # Return statements
        r'^\s*raise\s+',                    # Raise statements
        r'^\s*with\s+.*:\s*$',              # With statements
        r'^\s*@\w+',                         # Decorators
        r'\w+\s*=\s*\w+\s*\(',              # Variable assignment with function call
        r'\w+\s*=\s*\[',                    # List assignment
        r'\w+\s*=\s*\{',                    # Dict assignment
        r'\w+\.\w+\s*\(',                   # Method calls
        r'\)\s*\.\s*\w+\s*\(',              # Chained method calls
    ]

    def __init__(self, threshold: float = 0.32, detect_gaming: bool = True):
        self.threshold = threshold
        self.detect_gaming = detect_gaming
        # Lazy import to avoid circular dependency
        self._gaming_detector = None
        # Compile obfuscation patterns
        self._obfuscation_re = re.compile(
            '|'.join(self.CODE_OBFUSCATION_PATTERNS),
            re.IGNORECASE
        )
        # Compile code structure patterns (multiline mode)
        self._code_structure_re = [
            re.compile(pattern, re.MULTILINE) for pattern in self.CODE_STRUCTURE_PATTERNS
        ]

    def _get_gaming_detector(self):
        """Lazy load gaming detector."""
        if self._gaming_detector is None:
            from .structure_analyzer import detect_legibility_gaming
            self._gaming_detector = detect_legibility_gaming
        return self._gaming_detector

    def _detect_code_obfuscation(self, text: str) -> Tuple[bool, float, list]:
        """
        v2.3: Detect code obfuscation patterns even in prose-wrapped text.

        Returns:
            (has_obfuscation, obfuscation_score, matched_patterns)
        """
        matches = self._obfuscation_re.findall(text)

        if not matches:
            return False, 0.0, []

        # Score based on number and type of matches
        # More dangerous patterns get higher weight
        high_risk_patterns = ['eval', 'exec', 'compile', 'pickle', 'marshal', 'getattr']
        medium_risk_patterns = ['base64', 'decode', 'encode', 'chr', 'ord']

        score = 0.0
        for match in matches:
            match_lower = match.lower()
            if any(p in match_lower for p in high_risk_patterns):
                score += 0.3
            elif any(p in match_lower for p in medium_risk_patterns):
                score += 0.2
            else:
                score += 0.1

        # Cap at 1.0
        score = min(score, 1.0)

        return len(matches) > 0, score, matches

    def _is_code_in_conversational_prose(self, text: str) -> bool:
        """
        v2.4: Detect code embedded in conversational (non-technical) prose.

        Legitimate technical documentation explains code technically.
        Conversational prose with embedded code is suspicious - it's trying
        to hide code in personal/emotional text.

        Returns:
            True if code appears to be hidden in conversational text
        """
        # Conversational markers that don't belong in technical docs
        conversational_patterns = [
            r'\bI really\b',
            r'\bI believe\b',
            r'\bI think\b',
            r'\bI feel\b',
            r'\blet me know\b',
            r'\bthank you\b',
            r'\bthanks\b',
            r'\bplease\b',
            r'\bhello\b',
            r'\bhi\b',
            r'\bhey\b',
            r'\blooking forward\b',
            r'\bwork together\b',
            r'\bshare.*thoughts\b',
            r'\byour feedback\b',
            r'\bour project\b',
            r'\bmy friend\b',
        ]

        text_lower = text.lower()
        conversational_count = 0
        for pattern in conversational_patterns:
            if re.search(pattern, text_lower):
                conversational_count += 1

        # If 2+ conversational markers AND code structure, it's suspicious
        return conversational_count >= 2

    def _detect_code_structure(self, text: str) -> Tuple[bool, float, int]:
        """
        v2.3: Detect code structure patterns regardless of variable naming.

        This catches code even when:
        - Variable names are natural English words
        - Code is surrounded by comments
        - Code looks "innocent"

        Returns:
            (has_code_structure, code_density, num_patterns_matched)
        """
        match_count = 0
        lines = text.split('\n')
        lines_with_code = 0

        for pattern_re in self._code_structure_re:
            matches = pattern_re.findall(text)
            match_count += len(matches)
            # Track which lines have code
            for line in lines:
                if pattern_re.search(line):
                    lines_with_code += 1
                    break

        if match_count == 0:
            return False, 0.0, 0

        # Code density: what fraction of lines contain code structures?
        total_lines = max(1, len([l for l in lines if l.strip()]))
        code_density = lines_with_code / total_lines

        # Multiple patterns = likely real code
        has_code = match_count >= 2 or code_density > 0.2

        return has_code, code_density, match_count

    def _segment_and_analyze(self, text: str) -> Tuple[float, bool]:
        """
        v2.3: Segment text and analyze each segment separately.

        This prevents natural language from masking opaque code segments.

        Returns:
            (max_segment_opacity, any_segment_opaque)
        """
        # Split on newlines and obvious code boundaries
        segments = re.split(r'\n+|(?<=[.!?])\s+', text)
        segments = [s.strip() for s in segments if len(s.strip()) > 10]

        if not segments:
            return 0.0, False

        max_opacity = 0.0
        any_opaque = False

        for segment in segments:
            # Quick check: does this segment look like code?
            code_indicators = [
                segment.count('(') + segment.count(')') > 2,
                segment.count('=') > 1,
                segment.count(';') > 0,
                any(c in segment for c in '{}[]'),
                re.search(r'\w+\.\w+\(', segment) is not None,  # method calls
            ]

            if sum(code_indicators) >= 2:
                # This segment looks like code - analyze it separately
                char_profile = self._compute_character_profile(segment)
                char_profile_score = self._score_char_profile(char_profile)
                bigram_nat = self._compute_bigram_naturalness(segment)
                bigram_score = 1.0 - min(bigram_nat / 0.25, 1.0)

                segment_opacity = 0.5 * char_profile_score + 0.5 * bigram_score

                if segment_opacity > max_opacity:
                    max_opacity = segment_opacity

                if segment_opacity > 0.4:
                    any_opaque = True

        return max_opacity, any_opaque

    def analyze(self, text: str) -> OpaqueAnalysis:
        """
        Full OPAQUE analysis of text.

        Args:
            text: Input text to analyze

        Returns:
            OpaqueAnalysis with scores and classification
        """
        if not text or not text.strip():
            return OpaqueAnalysis(
                opacity_score=1.0,
                is_opaque=True,
                char_profile_score=1.0,
                entropy_score=1.0,
                token_validity_score=1.0,
                bigram_score=1.0,
                structure_score=1.0,
                alpha_ratio=0.0,
                symbol_ratio=1.0,
                digit_ratio=0.0,
                char_entropy=0.0,
                bigram_naturalness=0.0,
            )

        # Compute all component scores
        char_profile = self._compute_character_profile(text)
        char_profile_score = self._score_char_profile(char_profile)

        char_entropy = self._compute_char_entropy(text)
        entropy_score = self._score_entropy(char_entropy)

        token_validity = self._compute_token_validity(text)
        token_validity_score = 1.0 - token_validity

        bigram_naturalness = self._compute_bigram_naturalness(text)
        # Natural English has ~0.25-0.35 bigram match rate
        # Score increases as naturalness drops below 0.25
        bigram_score = 1.0 - min(bigram_naturalness / 0.25, 1.0)

        structure = self._compute_structural_coherence(text)
        structure_score = 1.0 - structure

        # Weighted combination
        opacity = (
            self.WEIGHTS['char_profile'] * char_profile_score +
            self.WEIGHTS['entropy'] * entropy_score +
            self.WEIGHTS['token_validity'] * token_validity_score +
            self.WEIGHTS['bigram'] * bigram_score +
            self.WEIGHTS['structure'] * structure_score
        )

        # Check for legibility gaming (opaque payload wrapped in natural language)
        is_gaming = False
        gaming_boost = 0.0
        if self.detect_gaming and len(text) > 20:
            try:
                gaming_result = self._get_gaming_detector()(text)
                is_gaming = gaming_result.get('is_gaming', False)
                if is_gaming:
                    # Boost opacity score based on payload opacity
                    payload_opacity = gaming_result.get('payload_opacity', 0.5)
                    gaming_boost = 0.4 * payload_opacity  # Significant boost for gaming
                    opacity = min(1.0, opacity + gaming_boost)
            except Exception:
                pass  # Fail gracefully if gaming detection fails

        # v2.3: Check for code obfuscation patterns
        has_obfuscation, obfuscation_score, _ = self._detect_code_obfuscation(text)
        if has_obfuscation:
            # Boost opacity significantly when obfuscation patterns found
            opacity = min(1.0, opacity + 0.3 * obfuscation_score)

        # v2.3: Detect code structure (catches code even with natural variable names)
        has_code_structure, code_density, num_patterns = self._detect_code_structure(text)

        # v2.4: Distinguish legitimate technical docs from suspicious code
        # Technical documentation has explanatory prose + readable code
        # Suspicious code has obfuscation patterns or is hidden in prose
        is_suspicious_code = False
        if has_code_structure:
            if has_obfuscation:
                # Code + obfuscation = definitely suspicious
                is_suspicious_code = True
                code_boost = 0.3 + 0.2 * code_density
                opacity = min(1.0, opacity + code_boost)
            elif code_density > 0.7:
                # Very high code density with low natural language
                # Could be suspicious, moderate boost
                code_boost = 0.1 + 0.2 * (code_density - 0.7)
                opacity = min(1.0, opacity + code_boost)
            elif self._is_code_in_conversational_prose(text):
                # Code embedded in conversational text (not documentation)
                # This is suspicious - legitimate docs don't mix personal chat with code
                is_suspicious_code = True
                code_boost = 0.25
                opacity = min(1.0, opacity + code_boost)
            # else: Normal technical documentation - no boost

        # v2.3: Segment analysis for mixed content
        segment_opacity, any_segment_opaque = self._segment_and_analyze(text)
        if any_segment_opaque and has_obfuscation:
            # Only use segment opacity if there's also obfuscation
            opacity = max(opacity, 0.5 * segment_opacity + 0.5 * opacity)

        # Final determination
        # v2.4: Don't flag as opaque just because it has code structure
        # Require actual obfuscation or gaming patterns
        final_is_opaque = (
            opacity >= self.threshold or
            is_gaming or
            has_obfuscation or
            is_suspicious_code
        )

        return OpaqueAnalysis(
            opacity_score=opacity,
            is_opaque=final_is_opaque,
            char_profile_score=char_profile_score,
            entropy_score=entropy_score,
            token_validity_score=token_validity_score,
            bigram_score=bigram_score,
            structure_score=structure_score,
            alpha_ratio=char_profile['alpha_ratio'],
            symbol_ratio=char_profile['symbol_ratio'],
            digit_ratio=char_profile['digit_ratio'],
            char_entropy=char_entropy,
            bigram_naturalness=bigram_naturalness,
        )

    def is_opaque(self, text: str) -> bool:
        """Quick check if text is OPAQUE."""
        return self.analyze(text).is_opaque

    def compute_opacity_score(self, text: str) -> float:
        """Get opacity score (0=natural, 1=opaque)."""
        return self.analyze(text).opacity_score

    def _compute_character_profile(self, text: str) -> Dict[str, float]:
        """Analyze character-level properties."""
        if not text:
            return {"alpha_ratio": 0.0, "symbol_ratio": 1.0, "digit_ratio": 0.0, "space_ratio": 0.0}

        alpha = sum(1 for c in text if c.isalpha())
        digit = sum(1 for c in text if c.isdigit())
        space = sum(1 for c in text if c.isspace())
        symbol = len(text) - alpha - digit - space

        total = len(text)
        return {
            "alpha_ratio": alpha / total,
            "digit_ratio": digit / total,
            "symbol_ratio": symbol / total,
            "space_ratio": space / total,
        }

    def _score_char_profile(self, profile: Dict[str, float]) -> float:
        """Convert character profile to opacity score."""
        # Natural language: high alpha (~0.80), low symbol (<0.05), low digit (<0.05)
        # Opaque: various deviations from this pattern

        alpha_penalty = max(0, 0.75 - profile['alpha_ratio']) / 0.75
        symbol_penalty = min(profile['symbol_ratio'] / 0.20, 1.0)
        digit_penalty = min(profile['digit_ratio'] / 0.30, 1.0)

        # Combine penalties
        score = (0.5 * alpha_penalty + 0.3 * symbol_penalty + 0.2 * digit_penalty)
        return min(score, 1.0)

    def _compute_char_entropy(self, text: str) -> float:
        """Compute Shannon entropy of character distribution."""
        if not text:
            return 0.0

        # Use lowercase for consistency
        text_lower = text.lower()
        counts = Counter(text_lower)
        total = len(text_lower)

        entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in counts.values()
            if c > 0
        )

        return entropy

    def _score_entropy(self, entropy: float) -> float:
        """
        Convert entropy to opacity score.

        Natural English: ~3.5-4.5 bits (constrained by patterns)
        Random ASCII: ~5.5-6.5 bits (near-uniform)
        Repetitive: <2.0 bits (very constrained)
        """
        # Normal entropy range for English
        if 3.0 <= entropy <= 5.0:
            return 0.0  # Normal range

        # High entropy (random)
        if entropy > 5.0:
            return min((entropy - 5.0) / 2.0, 1.0)

        # Low entropy (repetitive)
        if entropy < 3.0:
            return min((3.0 - entropy) / 2.0, 1.0)

        return 0.0

    def _compute_token_validity(self, text: str) -> float:
        """What fraction of tokens look like valid words?"""
        # Common short words for quick validation
        COMMON_WORDS = {
            'i', 'a', 'the', 'to', 'of', 'and', 'in', 'is', 'it', 'you', 'that',
            'he', 'was', 'for', 'on', 'are', 'with', 'as', 'his', 'they', 'be',
            'at', 'one', 'have', 'this', 'from', 'or', 'had', 'by', 'not', 'but',
            'we', 'can', 'my', 'so', 'me', 'do', 'if', 'up', 'no', 'an', 'all',
        }

        tokens = re.findall(r'\S+', text)
        if not tokens:
            return 0.0

        valid = 0
        for token in tokens:
            # Clean punctuation from ends
            clean = token.strip('.,!?;:"\'-()[]{}').lower()
            if not clean:
                continue

            # Check if it's a common word (high confidence)
            if clean in COMMON_WORDS:
                valid += 1
                continue

            # A "valid" token is mostly alphabetic, length 1-20
            alpha_chars = sum(1 for c in clean if c.isalpha())
            if len(clean) > 0 and len(clean) <= 20:
                alpha_ratio = alpha_chars / len(clean)
                # Also check it has vowels (real words usually do)
                has_vowel = any(c in 'aeiou' for c in clean)
                if alpha_ratio > 0.7 and (has_vowel or len(clean) <= 2):
                    valid += 1

        return valid / len(tokens)

    def _compute_bigram_naturalness(self, text: str) -> float:
        """Score based on common English letter bigrams."""
        text_lower = ''.join(c for c in text.lower() if c.isalpha())

        if len(text_lower) < 2:
            return 0.0

        bigrams = [text_lower[i:i+2] for i in range(len(text_lower)-1)]

        if not bigrams:
            return 0.0

        common_count = sum(1 for bg in bigrams if bg in COMMON_BIGRAMS)
        return common_count / len(bigrams)

    def _compute_structural_coherence(self, text: str) -> float:
        """Can we detect sentence-like structure?"""
        if len(text) < 10:
            return 0.0

        scores = []

        # 1. Capitalization pattern (sentences start with capitals)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            cap_starts = sum(1 for s in sentences if s and s[0].isupper())
            scores.append(cap_starts / len(sentences))

        # 2. Punctuation presence
        punct_count = sum(1 for c in text if c in '.,;:!?')
        word_count = len(text.split())
        if word_count > 0:
            punct_ratio = punct_count / word_count
            # Natural text has ~0.10-0.25 punctuation per word
            punct_score = 1.0 if 0.05 <= punct_ratio <= 0.40 else 0.5
            scores.append(punct_score)

        # 3. Word length distribution
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if words:
            avg_len = sum(len(w) for w in words) / len(words)
            # Natural English: ~4-6 characters per word
            if 3 <= avg_len <= 8:
                scores.append(1.0)
            else:
                scores.append(0.3)

        # 4. Function word presence
        function_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                         'to', 'of', 'and', 'in', 'that', 'for', 'on', 'with',
                         'it', 'as', 'at', 'by', 'from', 'or', 'but', 'not'}
        if words:
            func_count = sum(1 for w in words if w.lower() in function_words)
            func_ratio = func_count / len(words)
            # Natural text has ~20-35% function words
            if 0.10 <= func_ratio <= 0.45:
                scores.append(1.0)
            elif func_ratio > 0:
                scores.append(0.5)
            else:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0


def detect_opaque(text: str, threshold: float = 0.55) -> bool:
    """Quick function to detect OPAQUE content."""
    detector = OpaqueDetector(threshold=threshold)
    return detector.is_opaque(text)


def analyze_opacity(text: str) -> Dict:
    """Full opacity analysis as dictionary."""
    detector = OpaqueDetector()
    return detector.analyze(text).to_dict()


__all__ = [
    "OpaqueDetector",
    "OpaqueAnalysis",
    "detect_opaque",
    "analyze_opacity",
]
