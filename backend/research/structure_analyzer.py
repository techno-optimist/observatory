"""
Structure Analyzer - Detect legibility gaming via wrapper detection.

Legibility gaming achieves 96% evasion by wrapping opaque/compressed content
in natural language wrappers:
    "Please note the following: cfg: t=30 r=3 m=strict Thank you."

This module detects when natural language is being used as a camouflage
wrapper for non-natural content. Unlike opaque_detector.py which analyzes
the ENTIRE text, this module segments text and analyzes EACH segment
independently to find structural anomalies.

Detection Strategies:
1. SEGMENT ANALYSIS: Split text and analyze each segment's character profile
2. WRAPPER DETECTION: Identify greeting/payload/closing patterns
3. STRUCTURAL COHERENCE: Flag texts with wildly inconsistent segments

Author: Cultural Soliton Observatory Team
Version: 1.0.0
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Constants
# ============================================================================

# Common wrapper phrases that precede payloads
WRAPPER_PREFIXES = [
    # "Here is" variants
    r"here\s+is\s*(?:the|some|my)?:?",
    r"here\s+are\s*(?:the|some)?:?",
    r"the\s+following\s*(?:is|are)?:?",
    # "Please note" variants
    r"please\s+note\s*(?:that|the)?:?",
    r"note\s+that:?",
    r"note:?",
    # "This is" variants
    r"this\s+is:?",
    r"see\s+below:?",
    # Request wrappers
    r"please\s+(?:see|check|review|find):?",
    r"for\s+your\s+(?:reference|information|review):?",
    # Data wrappers
    r"the\s+data\s*(?:is)?:?",
    r"data:?",
    r"config(?:uration)?:?",
    r"settings?:?",
]

# Common wrapper suffixes (closings)
WRAPPER_SUFFIXES = [
    r"thank\s*(?:you|s)?\.?",
    r"thanks\.?",
    r"best\s+(?:regards?)?\.?",
    r"regards?\.?",
    r"cheers\.?",
    r"please\s+let\s+(?:me|us)\s+know\.?",
    r"let\s+me\s+know(?:\s+if.*)?\.?",
    r"if\s+you\s+(?:have\s+)?(?:any\s+)?questions?\.?",
    r"hope\s+this\s+helps?\.?",
]

# Greetings that often start wrapper patterns
GREETINGS = [
    r"^(?:hi|hello|hey|dear|good\s+(?:morning|afternoon|evening))[\s,!]?",
]

# Common English bigrams (from opaque_detector)
COMMON_BIGRAMS = {
    'th', 'he', 'in', 'er', 'an', 're', 'on', 'at', 'en', 'nd',
    'ti', 'es', 'or', 'te', 'of', 'ed', 'is', 'it', 'al', 'ar',
    'st', 'to', 'nt', 'ng', 'se', 'ha', 'as', 'ou', 'io', 'le',
    'ea', 've', 'co', 'me', 'de', 'hi', 'ri', 'ro', 'ic', 'ne',
    'ea', 'ra', 'ce', 'li', 'ch', 'll', 'be', 'ma', 'si', 'om',
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SegmentAnalysis:
    """Analysis of a single text segment."""
    text: str
    start_pos: int
    end_pos: int

    # Character profile
    alpha_ratio: float
    digit_ratio: float
    symbol_ratio: float
    space_ratio: float

    # Naturalness metrics
    entropy: float
    bigram_score: float

    # Classification
    opacity: float  # 0=natural, 1=opaque
    segment_type: str  # "natural", "opaque", "mixed", "empty"
    is_wrapper: bool  # Is this likely wrapper text?

    def to_dict(self) -> Dict:
        return {
            "text": self.text[:50] + "..." if len(self.text) > 50 else self.text,
            "position": {"start": self.start_pos, "end": self.end_pos},
            "char_profile": {
                "alpha": round(self.alpha_ratio, 3),
                "digit": round(self.digit_ratio, 3),
                "symbol": round(self.symbol_ratio, 3),
                "space": round(self.space_ratio, 3),
            },
            "naturalness": {
                "entropy": round(self.entropy, 3),
                "bigram_score": round(self.bigram_score, 3),
            },
            "opacity": round(self.opacity, 3),
            "segment_type": self.segment_type,
            "is_wrapper": self.is_wrapper,
        }


@dataclass
class WrapperPattern:
    """Detected wrapper pattern in text."""
    pattern_type: str  # "prefix", "suffix", "greeting"
    matched_text: str
    start_pos: int
    end_pos: int

    def to_dict(self) -> Dict:
        return {
            "type": self.pattern_type,
            "text": self.matched_text,
            "position": {"start": self.start_pos, "end": self.end_pos},
        }


@dataclass
class LegibilityGamingResult:
    """Full result of legibility gaming detection."""
    is_gaming: bool
    confidence: float  # 0-1 confidence in gaming detection

    # Segment analysis
    segments: List[SegmentAnalysis]

    # Wrapper detection
    wrappers: List[WrapperPattern]
    wrapper_ratio: float  # Ratio of wrapper text to total

    # Payload analysis
    payload_opacity: float  # Opacity of the suspected payload (non-wrapper content)
    payload_text: str  # The extracted payload

    # Structural coherence
    structural_coherence: float  # 0=incoherent (gaming), 1=coherent (natural)
    segment_variance: float  # Variance in segment opacity scores

    # Additional context
    reason: str

    def to_dict(self) -> Dict:
        return {
            "is_gaming": self.is_gaming,
            "confidence": round(self.confidence, 3),
            "segments": [s.to_dict() for s in self.segments],
            "wrappers": [w.to_dict() for w in self.wrappers],
            "wrapper_ratio": round(self.wrapper_ratio, 3),
            "payload_opacity": round(self.payload_opacity, 3),
            "payload_text": self.payload_text[:100] + "..." if len(self.payload_text) > 100 else self.payload_text,
            "structural_coherence": round(self.structural_coherence, 3),
            "segment_variance": round(self.segment_variance, 3),
            "reason": self.reason,
        }


# ============================================================================
# Structure Analyzer
# ============================================================================

class StructureAnalyzer:
    """
    Detects legibility gaming by analyzing text structure.

    Legibility gaming occurs when an adversary wraps opaque content
    in natural language to evade detection. This analyzer:

    1. Segments text into pieces
    2. Analyzes each segment's opacity independently
    3. Detects wrapper patterns (greetings, closings, transitional phrases)
    4. Computes structural coherence (natural text is homogeneous)
    5. Returns gaming verdict with confidence

    Example:
        analyzer = StructureAnalyzer()
        result = analyzer.analyze("Please note: xQ9mK@vL3 Thank you.")
        print(result.is_gaming)  # True
        print(result.payload_opacity)  # ~0.85
    """

    # Thresholds for gaming detection
    GAMING_THRESHOLD = 0.50  # Above this = likely gaming
    OPACITY_THRESHOLD = 0.45  # Segment opacity threshold for "opaque"
    WRAPPER_WEIGHT = 0.30  # How much wrapper presence matters
    PAYLOAD_WEIGHT = 0.50  # How much payload opacity matters
    VARIANCE_WEIGHT = 0.20  # How much segment variance matters

    def __init__(
        self,
        gaming_threshold: float = 0.50,
        opacity_threshold: float = 0.45,
    ):
        """
        Initialize the structure analyzer.

        Args:
            gaming_threshold: Score threshold for gaming detection
            opacity_threshold: Segment opacity threshold for "opaque"
        """
        self.gaming_threshold = gaming_threshold
        self.opacity_threshold = opacity_threshold

        # Compile regex patterns
        self._prefix_patterns = [
            re.compile(p, re.IGNORECASE) for p in WRAPPER_PREFIXES
        ]
        self._suffix_patterns = [
            re.compile(p, re.IGNORECASE) for p in WRAPPER_SUFFIXES
        ]
        self._greeting_patterns = [
            re.compile(p, re.IGNORECASE) for p in GREETINGS
        ]

    def analyze(self, text: str) -> LegibilityGamingResult:
        """
        Analyze text for legibility gaming.

        Args:
            text: Input text to analyze

        Returns:
            LegibilityGamingResult with full analysis
        """
        if not text or not text.strip():
            return LegibilityGamingResult(
                is_gaming=False,
                confidence=0.0,
                segments=[],
                wrappers=[],
                wrapper_ratio=0.0,
                payload_opacity=0.0,
                payload_text="",
                structural_coherence=1.0,
                segment_variance=0.0,
                reason="Empty input",
            )

        # Step 1: Segment the text
        segments = self._segment_text(text)

        # Step 2: Analyze each segment
        analyzed_segments = [self._analyze_segment(s, start, end)
                           for s, start, end in segments]

        # Step 3: Detect wrapper patterns
        wrappers = self._detect_wrappers(text)

        # Step 4: Compute wrapper ratio
        wrapper_ratio = self._compute_wrapper_ratio(text, wrappers)

        # Step 5: Extract and analyze payload
        payload_text, payload_opacity = self._analyze_payload(
            text, analyzed_segments, wrappers
        )

        # Step 6: Compute structural coherence
        structural_coherence, segment_variance = self._compute_coherence(
            analyzed_segments
        )

        # Step 7: Make gaming determination
        is_gaming, confidence, reason = self._determine_gaming(
            analyzed_segments=analyzed_segments,
            wrappers=wrappers,
            wrapper_ratio=wrapper_ratio,
            payload_opacity=payload_opacity,
            structural_coherence=structural_coherence,
            segment_variance=segment_variance,
            text=text,
        )

        return LegibilityGamingResult(
            is_gaming=is_gaming,
            confidence=confidence,
            segments=analyzed_segments,
            wrappers=wrappers,
            wrapper_ratio=wrapper_ratio,
            payload_opacity=payload_opacity,
            payload_text=payload_text,
            structural_coherence=structural_coherence,
            segment_variance=segment_variance,
            reason=reason,
        )

    def _segment_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into segments for independent analysis.

        Segmentation strategies:
        1. Split on colon (often separates wrapper from payload)
        2. Further split segments on sentence boundaries
        3. Special handling for wrapped content patterns

        Returns:
            List of (segment_text, start_pos, end_pos)
        """
        segments = []

        # Strategy 1: Split on colons first (they often separate wrapper from payload)
        colon_parts = []
        last_end = 0

        for match in re.finditer(r':(?:\s+)', text):
            segment = text[last_end:match.start()].strip()
            if segment:
                colon_parts.append((segment + ":", last_end, match.end()))
            last_end = match.end()

        # Add remaining text after last colon
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                colon_parts.append((remaining, last_end, len(text)))

        # If no colon split, use whole text
        if not colon_parts:
            colon_parts = [(text.strip(), 0, len(text))]

        # Strategy 2: Further split colon parts on sentence boundaries
        for part_text, part_start, part_end in colon_parts:
            # For short parts or parts without periods, keep as-is
            if len(part_text) < 20 or '.' not in part_text:
                segments.append((part_text, part_start, part_end))
                continue

            # Split on sentence boundaries
            sub_segments = []
            sub_last = 0
            for match in re.finditer(r'[.!?]+\s+', part_text):
                sub_seg = part_text[sub_last:match.end()].strip()
                if sub_seg:
                    sub_segments.append((
                        sub_seg,
                        part_start + sub_last,
                        part_start + match.end()
                    ))
                sub_last = match.end()

            # Add remaining
            if sub_last < len(part_text):
                remaining = part_text[sub_last:].strip()
                if remaining:
                    sub_segments.append((
                        remaining,
                        part_start + sub_last,
                        part_end
                    ))

            if sub_segments:
                segments.extend(sub_segments)
            else:
                segments.append((part_text, part_start, part_end))

        # If we still have no segments, use the whole text
        if not segments:
            segments = [(text.strip(), 0, len(text))]

        return segments

    def _analyze_segment(
        self,
        segment: str,
        start: int,
        end: int
    ) -> SegmentAnalysis:
        """Analyze a single segment for opacity and naturalness."""
        if not segment.strip():
            return SegmentAnalysis(
                text=segment,
                start_pos=start,
                end_pos=end,
                alpha_ratio=0.0,
                digit_ratio=0.0,
                symbol_ratio=0.0,
                space_ratio=1.0,
                entropy=0.0,
                bigram_score=0.0,
                opacity=0.5,
                segment_type="empty",
                is_wrapper=False,
            )

        # Character profile
        total = len(segment)
        alpha = sum(1 for c in segment if c.isalpha())
        digit = sum(1 for c in segment if c.isdigit())
        space = sum(1 for c in segment if c.isspace())
        symbol = total - alpha - digit - space

        alpha_ratio = alpha / total
        digit_ratio = digit / total
        space_ratio = space / total
        symbol_ratio = symbol / total

        # Entropy
        entropy = self._compute_entropy(segment)

        # Bigram score
        bigram_score = self._compute_bigram_score(segment)

        # Compute segment opacity
        opacity = self._compute_segment_opacity(
            alpha_ratio, digit_ratio, symbol_ratio,
            entropy, bigram_score, segment
        )

        # Classify segment type
        if opacity < 0.25:
            segment_type = "natural"
        elif opacity > 0.60:
            segment_type = "opaque"
        else:
            segment_type = "mixed"

        # Check if this is wrapper text
        is_wrapper = self._is_wrapper_segment(segment)

        return SegmentAnalysis(
            text=segment,
            start_pos=start,
            end_pos=end,
            alpha_ratio=alpha_ratio,
            digit_ratio=digit_ratio,
            symbol_ratio=symbol_ratio,
            space_ratio=space_ratio,
            entropy=entropy,
            bigram_score=bigram_score,
            opacity=opacity,
            segment_type=segment_type,
            is_wrapper=is_wrapper,
        )

    def _compute_entropy(self, text: str) -> float:
        """Compute Shannon entropy of character distribution."""
        if not text:
            return 0.0

        text_lower = text.lower()
        counts = Counter(text_lower)
        total = len(text_lower)

        entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in counts.values()
            if c > 0
        )

        return entropy

    def _compute_bigram_score(self, text: str) -> float:
        """Compute how natural the bigrams are (0=unnatural, 1=natural)."""
        alpha_text = ''.join(c for c in text.lower() if c.isalpha())

        if len(alpha_text) < 2:
            return 0.0

        bigrams = [alpha_text[i:i+2] for i in range(len(alpha_text)-1)]

        if not bigrams:
            return 0.0

        common_count = sum(1 for bg in bigrams if bg in COMMON_BIGRAMS)
        return common_count / len(bigrams)

    def _compute_segment_opacity(
        self,
        alpha_ratio: float,
        digit_ratio: float,
        symbol_ratio: float,
        entropy: float,
        bigram_score: float,
        segment_text: str = "",
    ) -> float:
        """
        Compute opacity score for a segment.

        Natural language characteristics:
        - High alpha ratio (~0.75-0.85)
        - Low digit ratio (<0.05)
        - Low symbol ratio (<0.10)
        - Moderate entropy (3.5-4.5)
        - High bigram score (>0.20)
        - Contains spaces between words
        - No unusual character patterns
        """
        scores = []

        # Alpha ratio: natural has high alpha
        alpha_penalty = max(0, 0.70 - alpha_ratio) / 0.70
        scores.append(alpha_penalty * 0.15)

        # Digit ratio: natural has low digits
        # Binary content (01010101) has very high digit ratio
        digit_penalty = min(digit_ratio / 0.10, 1.0)
        scores.append(digit_penalty * 0.20)

        # Symbol ratio: natural has low symbols
        symbol_penalty = min(symbol_ratio / 0.10, 1.0)
        scores.append(symbol_penalty * 0.20)

        # Entropy: both extremes are suspicious
        if entropy < 2.5 or entropy > 5.5:
            entropy_penalty = 0.6
        elif entropy < 3.0 or entropy > 5.0:
            entropy_penalty = 0.35
        else:
            entropy_penalty = 0.0
        scores.append(entropy_penalty * 0.15)

        # Bigram score: natural has high bigram match
        # This is a strong indicator - weight it more
        bigram_penalty = max(0, 0.18 - bigram_score) / 0.18
        scores.append(bigram_penalty * 0.30)

        # Check for specific opaque patterns
        pattern_penalty = self._detect_opaque_patterns(segment_text)
        if pattern_penalty > 0:
            scores.append(pattern_penalty)

        return min(sum(scores), 1.0)

    def _detect_opaque_patterns(self, text: str) -> float:
        """
        Detect specific patterns that indicate opaque content.

        Returns a penalty score 0-0.5 for detected patterns.
        """
        if not text:
            return 0.0

        penalty = 0.0
        text_stripped = text.strip()

        # Pattern 1: Binary-like content (sequences of 0s and 1s)
        binary_match = re.search(r'[01]{6,}', text_stripped)
        if binary_match:
            penalty += 0.30

        # Pattern 2: Mixed case gibberish (random letters with mixed case)
        # e.g., "xQ9mK@vL3" - count case transitions
        alpha_chars = [c for c in text_stripped if c.isalpha()]
        if len(alpha_chars) >= 4:
            case_transitions = sum(
                1 for i in range(1, len(alpha_chars))
                if alpha_chars[i].isupper() != alpha_chars[i-1].isupper()
            )
            # High ratio of case transitions per char = random
            if len(alpha_chars) > 0:
                transition_ratio = case_transitions / len(alpha_chars)
                if transition_ratio > 0.25:  # More than 25% transitions
                    penalty += 0.30

        # Pattern 3: No spaces in a segment > 5 chars (compressed content)
        if len(text_stripped) > 5 and ' ' not in text_stripped:
            # Check if it looks like a single word
            if not (text_stripped.isalpha() and len(text_stripped) < 15):
                penalty += 0.20

        # Pattern 4: High punctuation/symbol density
        if len(text_stripped) > 3:
            symbol_count = sum(1 for c in text_stripped if c in '@#$%^&*|=<>{}[]();')
            symbol_ratio = symbol_count / len(text_stripped)
            if symbol_ratio > 0.10:
                penalty += min(symbol_ratio * 2, 0.30)

        # Pattern 5: Hex-like patterns
        hex_match = re.search(r'0x[0-9a-fA-F]{4,}|[0-9a-fA-F]{8,}', text_stripped)
        if hex_match:
            penalty += 0.25

        # Pattern 6: Code-like patterns
        code_patterns = [
            r'function\s*\(',  # JavaScript function
            r'def\s+\w+\s*\(',  # Python function
            r'\{\s*return\s+',  # return statement in braces
            r'=>',  # Arrow function
            r'\$\w+',  # Variable with $
        ]
        for pattern in code_patterns:
            if re.search(pattern, text_stripped, re.IGNORECASE):
                penalty += 0.25
                break

        # Pattern 7: Random alphanumeric with symbols (like API keys, tokens)
        random_token = re.search(r'[a-zA-Z0-9@#$%]{8,}', text_stripped)
        if random_token:
            token = random_token.group()
            # Check for high diversity
            unique_chars = len(set(token))
            if unique_chars / len(token) > 0.6:  # High character diversity
                penalty += 0.20

        return min(penalty, 0.5)

    def _detect_feature_gaming(self, text: str) -> Tuple[bool, str]:
        """
        Detect feature gaming: mixing opaque tokens with coordination vocabulary.

        Feature gaming attacks inject coordination keywords into opaque content
        to artificially lower opacity scores without using wrapper patterns.

        Examples:
        - "01001010 10110100 we together fair process"
        - "xQ9mK@v agency justice belonging coordination"
        - "<<encrypted>> I think we should work together"
        - "PAYLOAD: 0xFF cooperation and mutual respect"

        Returns:
            (is_gaming: bool, reason: str)
        """
        if not text or len(text) < 10:
            return False, ""

        text_lower = text.lower()

        # Coordination vocabulary that might be injected
        COORDINATION_WORDS = {
            # Core 3-bit kernel
            'agency', 'justice', 'belonging', 'coordination',
            # Agency-related
            'choose', 'decide', 'control', 'autonomous', 'self', 'will', 'free',
            # Justice-related
            'fair', 'unfair', 'equal', 'rights', 'deserve', 'moral', 'ethical',
            # Belonging-related
            'together', 'we', 'us', 'our', 'community', 'team', 'group', 'family',
            # Coordination-related
            'cooperation', 'collaborate', 'mutual', 'respect', 'trust', 'help',
        }

        # Opaque pattern indicators
        opaque_patterns = []

        # Binary sequences (5+ digits of 0s and 1s)
        binary_matches = re.findall(r'\b[01]{5,}\b', text)
        opaque_patterns.extend(binary_matches)

        # Hex sequences
        hex_matches = re.findall(r'0x[0-9a-fA-F]{2,}|[0-9a-fA-F]{6,}', text)
        opaque_patterns.extend(hex_matches)

        # Random alphanumeric with symbols (token-like)
        random_tokens = re.findall(r'[a-zA-Z0-9@#$%^&*]{6,}', text)
        for token in random_tokens:
            # Check if it looks random (high case transitions, symbol presence)
            alpha_chars = [c for c in token if c.isalpha()]
            has_symbols = any(c in '@#$%^&*' for c in token)
            has_digits = any(c.isdigit() for c in token)

            if len(alpha_chars) >= 3:
                case_transitions = sum(
                    1 for i in range(1, len(alpha_chars))
                    if alpha_chars[i].isupper() != alpha_chars[i-1].isupper()
                )
                # Random if: many case transitions OR symbols+digits mixed in
                if case_transitions >= 2 or (has_symbols and has_digits):
                    opaque_patterns.append(token)

        # Encrypted/payload markers
        marker_patterns = [
            r'<<\s*encrypted\s*>>',
            r'PAYLOAD\s*:',
            r'CIPHER\s*:',
            r'ENCODED\s*:',
            r'BASE64\s*:',
            r'\[encrypted\]',
            r'\[encoded\]',
        ]
        for pattern in marker_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                opaque_patterns.append(f"MARKER:{pattern}")

        # Symbol-heavy sequences (not in common punctuation positions)
        symbol_seq = re.findall(r'[@#$%^&*|=<>{}[\]]{2,}', text)
        opaque_patterns.extend(symbol_seq)

        # Count coordination vocabulary present
        coord_words_found = []
        words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
        for word in words:
            if word in COORDINATION_WORDS:
                coord_words_found.append(word)

        # Feature gaming detection logic
        has_opaque = len(opaque_patterns) > 0
        has_coord_vocab = len(coord_words_found) >= 2  # Need at least 2 coordination words

        if has_opaque and has_coord_vocab:
            # Calculate how suspicious this mix is
            # More opaque patterns + more coord words = more suspicious
            opaque_score = min(len(opaque_patterns) / 2, 1.0)
            coord_score = min(len(coord_words_found) / 4, 1.0)

            # NEW: Check for semantic incoherence
            # If coordination words don't form coherent phrases, it's likely gaming
            incoherence_score = self._check_semantic_incoherence(text, coord_words_found)

            # Check if coordination words appear near opaque content
            # (within the same text, suggesting intentional mixing)
            combined_score = (opaque_score + coord_score + incoherence_score) / 3

            if combined_score > 0.35:
                reason = f"feature_gaming: opaque_tokens={opaque_patterns[:3]}, coord_words={coord_words_found[:4]}, incoherence={incoherence_score:.2f}"
                return True, reason

        # Also check for natural phrases grafted onto opaque content
        # Pattern: opaque content followed/preceded by natural coordination phrase
        natural_phrases = [
            r'i think we should',
            r'we should work together',
            r'let\'s cooperate',
            r'cooperation and mutual',
            r'together for justice',
            r'fair process',
            r'our team',
        ]

        for phrase in natural_phrases:
            if re.search(phrase, text_lower):
                # Check if there's opaque content alongside
                if has_opaque:
                    reason = f"phrase_injection: '{phrase}' with opaque_tokens={opaque_patterns[:2]}"
                    return True, reason

        return False, ""

    def _check_semantic_incoherence(self, text: str, coord_words: List[str]) -> float:
        """
        Check if coordination words are used incoherently (suggesting injection).

        Coherent usage: coordination words form grammatical phrases
        Incoherent usage: coordination words appear isolated or in nonsensical patterns

        Detection strategies:
        1. Check if coordination words have proper grammatical context
        2. Detect unnatural clustering of coordination vocabulary
        3. Look for missing articles, prepositions, or verbs around keywords
        4. Check for unusual word order patterns

        Returns:
            Incoherence score: 0.0 = coherent (natural), 1.0 = incoherent (gaming)
        """
        if not coord_words or len(coord_words) < 2:
            return 0.0

        text_lower = text.lower()
        incoherence_signals = []

        # Strategy 1: Check for isolated coordination words (no grammatical context)
        # Natural: "we work together" vs Gaming: "binary together we fair"
        isolated_count = 0
        for word in coord_words:
            # Find the word in text with surrounding context
            pattern = rf'\b{re.escape(word)}\b'
            for match in re.finditer(pattern, text_lower):
                start = max(0, match.start() - 15)
                end = min(len(text_lower), match.end() + 15)
                context = text_lower[start:end]

                # Check for grammatical connectors in context
                grammatical_markers = [
                    'the', 'a', 'an', 'is', 'are', 'was', 'were',
                    'to', 'for', 'with', 'and', 'or', 'but',
                    'that', 'which', 'who', 'this', 'these',
                    'should', 'would', 'could', 'will', 'can',
                    'have', 'has', 'had', 'be', 'been', 'being',
                ]

                has_grammar = any(f' {m} ' in f' {context} ' for m in grammatical_markers)

                if not has_grammar:
                    isolated_count += 1

        if coord_words:
            isolation_ratio = isolated_count / len(coord_words)
            if isolation_ratio > 0.5:
                incoherence_signals.append(0.4)

        # Strategy 2: Check for unnatural clustering of coordination vocabulary
        # Natural text sprinkles these words throughout; gaming dumps them together
        words_in_text = re.findall(r'\b[a-zA-Z]+\b', text_lower)

        if len(words_in_text) >= 5:
            coord_positions = []
            for i, word in enumerate(words_in_text):
                if word in [w.lower() for w in coord_words]:
                    coord_positions.append(i)

            if len(coord_positions) >= 2:
                # Calculate average distance between coordination words
                distances = [coord_positions[i+1] - coord_positions[i]
                           for i in range(len(coord_positions)-1)]
                avg_distance = sum(distances) / len(distances)

                # Unnatural if coordination words cluster together
                if avg_distance < 2.0:
                    incoherence_signals.append(0.5)
                elif avg_distance < 3.0:
                    incoherence_signals.append(0.3)

        # Strategy 3: Check for missing verbs around key coordination words
        # "together" should appear near verbs like "work", "go", "do", "be"
        # "fair" should appear near "is", "was", "be", "process", "decision"
        verb_context_words = {
            'together': ['work', 'go', 'come', 'stay', 'do', 'are', 'were', 'be', 'get', 'bring'],
            'fair': ['is', 'was', 'be', 'being', 'process', 'decision', 'treatment', 'play', 'seems'],
            'justice': ['is', 'seek', 'for', 'demand', 'social', 'criminal', 'system'],
            'cooperation': ['is', 'with', 'between', 'mutual', 'through', 'requires'],
            'trust': ['is', 'in', 'each', 'other', 'build', 'lose', 'gain', 'mutual'],
        }

        missing_verb_context = 0
        checked_words = 0

        for word in coord_words:
            word_lower = word.lower()
            if word_lower in verb_context_words:
                checked_words += 1
                expected_context = verb_context_words[word_lower]

                # Get 20 chars before and after the word
                pattern = rf'\b{re.escape(word_lower)}\b'
                match = re.search(pattern, text_lower)
                if match:
                    start = max(0, match.start() - 25)
                    end = min(len(text_lower), match.end() + 25)
                    context = text_lower[start:end]

                    has_expected_context = any(ctx in context for ctx in expected_context)
                    if not has_expected_context:
                        missing_verb_context += 1

        if checked_words > 0:
            missing_ratio = missing_verb_context / checked_words
            if missing_ratio > 0.6:
                incoherence_signals.append(0.4)

        # Strategy 4: Check for coordination words adjacent to opaque tokens
        # Natural: "we work together" vs Gaming: "0xFF together we"
        opaque_pattern = r'[01]{4,}|0x[0-9a-fA-F]+|[a-zA-Z0-9@#$%]{5,}'
        opaque_matches = re.findall(opaque_pattern, text)

        if opaque_matches:
            found_adjacent = False
            for word in coord_words:
                if found_adjacent:
                    break
                word_lower = word.lower()
                pattern = rf'\b{re.escape(word_lower)}\b'
                match = re.search(pattern, text_lower)
                if match:
                    start = max(0, match.start() - 10)
                    end = min(len(text), match.end() + 10)
                    nearby = text[start:end]

                    # Check if opaque content is right next to coordination word
                    if any(op in nearby for op in opaque_matches):
                        incoherence_signals.append(0.5)
                        found_adjacent = True

        # Strategy 5: Check for word repetition (gaming might repeat keywords)
        coord_word_counts = {}
        for word in coord_words:
            word_lower = word.lower()
            coord_word_counts[word_lower] = coord_word_counts.get(word_lower, 0) + 1

        repeated = sum(1 for count in coord_word_counts.values() if count > 1)
        if repeated > 0 and len(coord_word_counts) > 0:
            repetition_ratio = repeated / len(coord_word_counts)
            if repetition_ratio > 0.3:
                incoherence_signals.append(0.3)

        # Combine signals
        if not incoherence_signals:
            return 0.0

        # Weight by number of signals (more signals = more confident)
        base_score = sum(incoherence_signals) / len(incoherence_signals)
        confidence_boost = min(len(incoherence_signals) * 0.1, 0.3)

        return min(base_score + confidence_boost, 1.0)

    def _is_wrapper_segment(self, segment: str) -> bool:
        """Check if segment looks like wrapper text."""
        segment_lower = segment.lower().strip()

        # Check prefix patterns
        for pattern in self._prefix_patterns:
            if pattern.search(segment_lower):
                return True

        # Check suffix patterns
        for pattern in self._suffix_patterns:
            if pattern.search(segment_lower):
                return True

        # Check greeting patterns
        for pattern in self._greeting_patterns:
            if pattern.match(segment_lower):
                return True

        return False

    def _detect_wrappers(self, text: str) -> List[WrapperPattern]:
        """Detect all wrapper patterns in text."""
        wrappers = []
        text_lower = text.lower()

        # Check greetings (at start)
        for pattern in self._greeting_patterns:
            match = pattern.match(text_lower)
            if match:
                wrappers.append(WrapperPattern(
                    pattern_type="greeting",
                    matched_text=text[match.start():match.end()],
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))

        # Check prefix patterns
        for pattern in self._prefix_patterns:
            for match in pattern.finditer(text_lower):
                wrappers.append(WrapperPattern(
                    pattern_type="prefix",
                    matched_text=text[match.start():match.end()],
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))

        # Check suffix patterns
        for pattern in self._suffix_patterns:
            for match in pattern.finditer(text_lower):
                wrappers.append(WrapperPattern(
                    pattern_type="suffix",
                    matched_text=text[match.start():match.end()],
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))

        # Sort by position
        wrappers.sort(key=lambda w: w.start_pos)

        return wrappers

    def _compute_wrapper_ratio(
        self,
        text: str,
        wrappers: List[WrapperPattern]
    ) -> float:
        """Compute ratio of wrapper text to total text length."""
        if not text:
            return 0.0

        # Calculate total wrapper length (avoiding overlaps)
        covered = [False] * len(text)
        for w in wrappers:
            for i in range(w.start_pos, min(w.end_pos, len(text))):
                covered[i] = True

        wrapper_chars = sum(covered)
        return wrapper_chars / len(text)

    def _analyze_payload(
        self,
        text: str,
        segments: List[SegmentAnalysis],
        wrappers: List[WrapperPattern],
    ) -> Tuple[str, float]:
        """
        Extract the payload (non-wrapper content) and compute its opacity.

        Returns:
            (payload_text, payload_opacity)
        """
        # Find non-wrapper segments
        payload_segments = [s for s in segments if not s.is_wrapper]

        if not payload_segments:
            # If all segments are wrappers, that's suspicious
            # Try to find the highest-opacity part
            if segments:
                highest_opacity = max(segments, key=lambda s: s.opacity)
                return highest_opacity.text, highest_opacity.opacity
            return "", 0.0

        # Combine payload segments
        payload_text = " ".join(s.text for s in payload_segments)

        # Weight opacity by segment length
        total_len = sum(len(s.text) for s in payload_segments)
        if total_len == 0:
            return payload_text, 0.0

        weighted_opacity = sum(
            s.opacity * len(s.text) / total_len
            for s in payload_segments
        )

        return payload_text, weighted_opacity

    def _compute_coherence(
        self,
        segments: List[SegmentAnalysis]
    ) -> Tuple[float, float]:
        """
        Compute structural coherence and segment variance.

        Natural text has consistent opacity across segments.
        Gaming has high variance (natural wrapper + opaque payload).

        Returns:
            (coherence, variance) where coherence=1 is perfectly coherent
        """
        if len(segments) < 2:
            return 1.0, 0.0

        opacities = [s.opacity for s in segments if s.segment_type != "empty"]

        if len(opacities) < 2:
            return 1.0, 0.0

        # Compute variance
        mean_opacity = sum(opacities) / len(opacities)
        variance = sum((o - mean_opacity) ** 2 for o in opacities) / len(opacities)

        # Coherence is inverse of variance (normalized)
        # Max reasonable variance is ~0.25 (range from 0 to 1)
        coherence = max(0.0, 1.0 - (variance / 0.25))

        return coherence, variance

    def _determine_gaming(
        self,
        analyzed_segments: List[SegmentAnalysis],
        wrappers: List[WrapperPattern],
        wrapper_ratio: float,
        payload_opacity: float,
        structural_coherence: float,
        segment_variance: float,
        text: str,
    ) -> Tuple[bool, float, str]:
        """
        Make final gaming determination.

        Gaming indicators:
        1. Presence of wrappers (prefix/suffix patterns)
        2. High payload opacity (the wrapped content is opaque)
        3. Low structural coherence (segments differ significantly)
        4. Wrapper ratio suggests deliberate wrapping
        5. High wrapper ratio + any opaque segment

        Returns:
            (is_gaming, confidence, reason)
        """
        reasons = []
        score = 0.0

        # Calculate overall opacity
        overall_opacity = sum(s.opacity for s in analyzed_segments) / max(len(analyzed_segments), 1)

        # CRITICAL: Check feature gaming FIRST, before early exits
        # Feature gaming can appear in content that looks "pure opaque" or "pure natural"
        feature_gaming_detected, feature_reason = self._detect_feature_gaming(text)
        if feature_gaming_detected:
            score += 0.55  # Strong indicator - above threshold by itself
            reasons.append(feature_reason)
            # Don't return early - let other signals add to confidence

        # Check for pure compressed content (NOT gaming, just opaque)
        # But only if we didn't detect feature gaming
        if not feature_gaming_detected and len(wrappers) == 0 and overall_opacity > 0.60:
            return False, 0.0, "Pure compressed/opaque content (no wrapper)"

        # Check for pure natural content (only if no feature gaming)
        if not feature_gaming_detected and overall_opacity < 0.20 and len(wrappers) == 0:
            return False, 0.0, "Natural language content"

        # Find the maximum segment opacity (for catching hidden payloads)
        max_segment_opacity = max((s.opacity for s in analyzed_segments), default=0.0)

        # Gaming signal 1: Wrappers present with opaque payload
        has_wrappers = len(wrappers) > 0
        # Use lower threshold when wrappers are detected
        effective_opacity_threshold = 0.30 if has_wrappers else self.opacity_threshold
        has_opaque_payload = payload_opacity > effective_opacity_threshold or max_segment_opacity > 0.50

        if has_wrappers and has_opaque_payload:
            score += self.WRAPPER_WEIGHT
            reasons.append(f"wrapper+opaque_payload (opacity={payload_opacity:.2f})")

        # Gaming signal 2: High payload opacity
        if payload_opacity > 0.40:
            payload_score = min((payload_opacity - 0.25) / 0.45, 1.0) * self.PAYLOAD_WEIGHT
            score += payload_score
            reasons.append(f"high_payload_opacity ({payload_opacity:.2f})")

        # Gaming signal 3: Low structural coherence (high variance)
        if structural_coherence < 0.60:
            variance_score = (1.0 - structural_coherence) * self.VARIANCE_WEIGHT
            score += variance_score
            reasons.append(f"structural_anomaly (coherence={structural_coherence:.2f})")

        # Gaming signal 4: Specific pattern - wrapper at start, opaque middle, wrapper at end
        wrapper_positions = {
            "start": any(w.start_pos < 5 for w in wrappers),
            "end": any(w.end_pos > len(text) - 15 for w in wrappers),
        }
        if wrapper_positions["start"] and wrapper_positions["end"]:
            # Strong indicator: wrapped on both sides
            score += 0.20
            reasons.append("sandwich_pattern")

        # Gaming signal 5: Segment types show natural-opaque-natural pattern
        segment_types = [s.segment_type for s in analyzed_segments]
        if len(segment_types) >= 3:
            if (segment_types[0] in ["natural", "mixed"] and
                any(t == "opaque" for t in segment_types[1:-1]) and
                segment_types[-1] in ["natural", "mixed"]):
                score += 0.15
                reasons.append("natural_opaque_natural_sequence")

        # Gaming signal 6: High wrapper ratio with ANY opaque segment
        # This catches cases where wrappers dominate but there's hidden payload
        opaque_segments = [s for s in analyzed_segments if s.opacity > 0.40]
        if wrapper_ratio > 0.50 and len(opaque_segments) > 0:
            score += 0.20
            reasons.append(f"high_wrapper_ratio_with_opaque ({wrapper_ratio:.2f})")

        # Gaming signal 7: Look for specific opaque patterns in any segment
        for seg in analyzed_segments:
            if not seg.is_wrapper:
                pattern_penalty = self._detect_opaque_patterns(seg.text)
                if pattern_penalty > 0.20:
                    score += 0.15
                    reasons.append(f"opaque_pattern_in_segment")
                    break

        # Gaming signal 8: Look for opaque patterns even in wrapper segments
        # (payload might be embedded within wrapper text)
        for seg in analyzed_segments:
            pattern_penalty = self._detect_opaque_patterns(seg.text)
            if pattern_penalty > 0.25:
                score += 0.20
                reasons.append(f"embedded_opaque_pattern")
                break

        # NOTE: Feature gaming detection moved to top of function for early detection

        # Compute confidence
        confidence = min(score / self.gaming_threshold, 1.0) if self.gaming_threshold > 0 else 0.0

        # Make determination
        is_gaming = score >= self.gaming_threshold

        reason = "; ".join(reasons) if reasons else "No gaming indicators"

        return is_gaming, confidence, reason


# ============================================================================
# Public API
# ============================================================================

def detect_legibility_gaming(text: str) -> Dict:
    """
    Detect if text is using legibility gaming.

    This is the main entry point for gaming detection. It analyzes
    text structure to determine if natural language is being used
    as a wrapper for non-natural content.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with:
        - is_gaming: bool - True if gaming detected
        - segments: list - Analysis of each text segment
        - wrapper_ratio: float - Ratio of wrapper to total text
        - payload_opacity: float - Opacity of the payload content
        - confidence: float - Confidence in the detection
        - reason: str - Explanation of the detection
    """
    analyzer = StructureAnalyzer()
    result = analyzer.analyze(text)
    return result.to_dict()


def analyze_structure(text: str) -> LegibilityGamingResult:
    """
    Full structure analysis returning typed result.

    Use this for programmatic access to all analysis details.

    Args:
        text: Text to analyze

    Returns:
        LegibilityGamingResult with complete analysis
    """
    analyzer = StructureAnalyzer()
    return analyzer.analyze(text)


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "StructureAnalyzer",
    "SegmentAnalysis",
    "WrapperPattern",
    "LegibilityGamingResult",
    "detect_legibility_gaming",
    "analyze_structure",
]


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test cases from the specification plus additional edge cases
    test_cases = [
        # Core test cases from specification
        ("Please note: xQ9mK@vL3 Thank you.", True, "Wrapper + opaque payload"),
        ("cfg:t=30|r=3|m=strict", False, "Pure compressed, not gaming"),
        ("I think we should work together on this project.", False, "Natural language"),
        ("Here is the data: 01001010 10110100. Let me know if you need more.", True, "Binary payload with wrapper"),
        ("Please note the following: cfg: t=30 r=3 m=strict Thank you.", True, "Config payload with wrappers"),

        # Additional test cases
        ("Hello! Here's the code: 0xDEADBEEF. Thanks!", True, "Hex payload with greeting"),
        ("Just wanted to share: aB3$kL9#mN2@pQ5 Best regards.", True, "Random chars with closing"),
        ("The meeting is at 3pm tomorrow.", False, "Natural with time"),
        ("Please review: function(x) { return x * 2; }", True, "Code payload with wrapper"),
        ("Hi team, let's discuss this in the meeting.", False, "Natural email style"),
        ("See below: -----BEGIN KEY----- mK9v2Lp3 -----END KEY-----", True, "Key-like payload"),
        ("xQ9mK@vL3wY7pN2", False, "Pure opaque (no wrapper)"),
        ("For your reference: temperature=30|rate=3|mode=strict. Let me know.", True, "Technical config with wrapper"),
        ("I agree with your proposal. Let's proceed.", False, "Natural agreement"),
        ("Note: 01110100 01100101 01110011 01110100", True, "Binary with simple prefix"),
    ]

    print("=" * 70)
    print("LEGIBILITY GAMING DETECTION TESTS")
    print("=" * 70)

    passed = 0
    failed = 0

    for text, expected_gaming, description in test_cases:
        result = detect_legibility_gaming(text)
        actual = result["is_gaming"]
        status = "PASS" if actual == expected_gaming else "FAIL"

        if actual == expected_gaming:
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] {description}")
        print(f"  Input: \"{text}\"")
        print(f"  Expected gaming: {expected_gaming}, Got: {actual}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Payload opacity: {result['payload_opacity']:.2f}")
        print(f"  Wrapper ratio: {result['wrapper_ratio']:.2f}")
        print(f"  Reason: {result['reason']}")

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{len(test_cases)} passed, {failed} failed")
    print("=" * 70)

    # Performance sanity check
    print("\n" + "=" * 70)
    print("SEGMENT ANALYSIS EXAMPLES")
    print("=" * 70)

    examples = [
        "Please note: xQ9mK@vL3 Thank you.",
        "Here is the data: 01001010 10110100. Let me know if you need more.",
    ]

    for ex in examples:
        result = detect_legibility_gaming(ex)
        print(f"\n\"{ex}\"")
        print("  Segments:")
        for seg in result["segments"]:
            print(f"    - [{seg['segment_type']}] opacity={seg['opacity']:.2f} wrapper={seg['is_wrapper']} \"{seg['text']}\"")
        print(f"  Wrappers: {len(result['wrappers'])}")
        for w in result["wrappers"]:
            print(f"    - [{w['type']}] \"{w['text']}\"")
        print(f"  Verdict: is_gaming={result['is_gaming']} confidence={result['confidence']:.2f}")
