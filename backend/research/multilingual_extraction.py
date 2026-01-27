"""
Multilingual Coordinate Extraction for Cross-Cultural Narrative Analysis.

Addresses peer reviewer concerns about English-centricity:
- Pro-drop languages (Spanish, Japanese) won't work with pronoun-based extraction
- Languages with grammaticalized evidentiality (Turkish, Tibetan) need different extraction
- Honorific systems (Japanese, Korean) encode social dimensions differently

This module provides:
1. Language-agnostic extraction interface (LanguageExtractor ABC)
2. Language-specific extractors (English, Spanish, Japanese, with stubs for others)
3. Unified multilingual coordinator (MultilingualCoordinateExtractor)
4. Cross-linguistic invariance testing framework

Linguistic Features by Language Type:
- English: Explicit pronouns, word order for agency, lexical evidentiality
- Spanish: Pro-drop (verb morphology encodes person), tu/usted formality
- Japanese: Topic/subject marking (wa/ga), honorifics, sentence-final particles
- Turkish: Grammaticalized evidentiality (-mis/-di distinction)
- Korean: Honorific verb endings, hierarchical social marking

Theoretical Foundation:
The coordination substrate (agency, justice, belonging) should be language-universal,
but the SURFACE MARKERS differ radically. This module enables testing that hypothesis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import re
import logging
from collections import defaultdict
import warnings

# Import hierarchical coordinate structures
from .hierarchical_coordinates import (
    AgencyDecomposition,
    JusticeDecomposition,
    BelongingDecomposition,
    CoordinationCore,
    CoordinationModifiers,
    EpistemicModifiers,
    TemporalModifiers,
    SocialModifiers,
    EmotionalModifiers,
    HierarchicalCoordinate,
    DecorativeLayer
)

logger = logging.getLogger(__name__)


# =============================================================================
# Language Feature Enumerations
# =============================================================================

class ProDropType(Enum):
    """Classification of pro-drop behavior in languages."""
    NONE = "none"              # English, German - pronouns required
    PARTIAL = "partial"        # Spanish, Italian - subject pronouns dropped
    FULL = "full"              # Japanese, Korean, Chinese - subject and object dropped
    TOPIC_DROP = "topic_drop"  # Japanese topic-prominent language


class EvidentialitySystem(Enum):
    """How languages encode information source."""
    LEXICAL = "lexical"              # English: "apparently", "I saw"
    GRAMMATICALIZED = "grammaticalized"  # Turkish, Tibetan, Bulgarian
    PARTICLE_BASED = "particle_based"    # Japanese: sentence-final particles
    NONE = "none"                    # No systematic evidentiality


class HonorificSystem(Enum):
    """Classification of politeness/honorific systems."""
    NONE = "none"                    # English - minimal systematic honorifics
    BINARY = "binary"                # Spanish tu/usted, German du/Sie
    MULTI_LEVEL = "multi_level"      # Japanese keigo (5+ levels)
    VERB_BASED = "verb_based"        # Korean - honorific verb endings


class AgencyEncoding(Enum):
    """How languages encode agency."""
    WORD_ORDER = "word_order"        # English: SVO word order
    CASE_MARKING = "case_marking"    # Japanese: ga/wo/ni particles
    VERB_MORPHOLOGY = "verb_morphology"  # Spanish: verb endings
    MIXED = "mixed"                  # Languages with multiple strategies


# =============================================================================
# Language Profile Dataclass
# =============================================================================

@dataclass
class LanguageProfile:
    """
    Typological profile of a language for extraction strategy selection.

    This captures the linguistic features that affect how we extract
    coordination dimensions from text.
    """
    code: str                                    # ISO 639-1 code
    name: str                                    # Human-readable name
    pro_drop: ProDropType = ProDropType.NONE
    evidentiality: EvidentialitySystem = EvidentialitySystem.LEXICAL
    honorifics: HonorificSystem = HonorificSystem.NONE
    agency_encoding: AgencyEncoding = AgencyEncoding.WORD_ORDER

    # Morphological richness affects parsing strategy
    is_agglutinative: bool = False
    is_isolating: bool = False
    is_fusional: bool = False

    # Writing system affects tokenization
    uses_word_spaces: bool = True
    script: str = "latin"

    # Social features encoded in grammar
    has_grammatical_gender: bool = False
    has_formal_informal_distinction: bool = False
    formality_in_pronouns: bool = False
    formality_in_verbs: bool = False

    # Additional features
    topic_prominent: bool = False          # Topic-comment structure (Japanese)
    subject_prominent: bool = True         # Subject-predicate structure (English)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "pro_drop": self.pro_drop.value,
            "evidentiality": self.evidentiality.value,
            "honorifics": self.honorifics.value,
            "agency_encoding": self.agency_encoding.value,
            "is_agglutinative": self.is_agglutinative,
            "is_isolating": self.is_isolating,
            "is_fusional": self.is_fusional,
            "uses_word_spaces": self.uses_word_spaces,
            "script": self.script,
            "has_grammatical_gender": self.has_grammatical_gender,
            "has_formal_informal_distinction": self.has_formal_informal_distinction,
            "topic_prominent": self.topic_prominent,
            "subject_prominent": self.subject_prominent
        }


# Pre-defined language profiles
LANGUAGE_PROFILES = {
    "en": LanguageProfile(
        code="en",
        name="English",
        pro_drop=ProDropType.NONE,
        evidentiality=EvidentialitySystem.LEXICAL,
        honorifics=HonorificSystem.NONE,
        agency_encoding=AgencyEncoding.WORD_ORDER,
        is_fusional=True,
        script="latin",
        has_grammatical_gender=False,
        has_formal_informal_distinction=False,
        subject_prominent=True
    ),
    "es": LanguageProfile(
        code="es",
        name="Spanish",
        pro_drop=ProDropType.PARTIAL,
        evidentiality=EvidentialitySystem.LEXICAL,
        honorifics=HonorificSystem.BINARY,
        agency_encoding=AgencyEncoding.VERB_MORPHOLOGY,
        is_fusional=True,
        script="latin",
        has_grammatical_gender=True,
        has_formal_informal_distinction=True,
        formality_in_pronouns=True,
        formality_in_verbs=True,
        subject_prominent=True
    ),
    "ja": LanguageProfile(
        code="ja",
        name="Japanese",
        pro_drop=ProDropType.FULL,
        evidentiality=EvidentialitySystem.PARTICLE_BASED,
        honorifics=HonorificSystem.MULTI_LEVEL,
        agency_encoding=AgencyEncoding.CASE_MARKING,
        is_agglutinative=True,
        uses_word_spaces=False,
        script="japanese",  # mix of kanji, hiragana, katakana
        has_grammatical_gender=False,
        has_formal_informal_distinction=True,
        formality_in_pronouns=True,
        formality_in_verbs=True,
        topic_prominent=True,
        subject_prominent=False
    ),
    "ko": LanguageProfile(
        code="ko",
        name="Korean",
        pro_drop=ProDropType.FULL,
        evidentiality=EvidentialitySystem.LEXICAL,
        honorifics=HonorificSystem.VERB_BASED,
        agency_encoding=AgencyEncoding.CASE_MARKING,
        is_agglutinative=True,
        uses_word_spaces=True,
        script="hangul",
        has_grammatical_gender=False,
        has_formal_informal_distinction=True,
        formality_in_verbs=True,
        topic_prominent=True
    ),
    "tr": LanguageProfile(
        code="tr",
        name="Turkish",
        pro_drop=ProDropType.PARTIAL,
        evidentiality=EvidentialitySystem.GRAMMATICALIZED,
        honorifics=HonorificSystem.BINARY,
        agency_encoding=AgencyEncoding.CASE_MARKING,
        is_agglutinative=True,
        script="latin",
        has_grammatical_gender=False,
        has_formal_informal_distinction=True,
        formality_in_pronouns=True
    ),
    "zh": LanguageProfile(
        code="zh",
        name="Chinese (Mandarin)",
        pro_drop=ProDropType.FULL,
        evidentiality=EvidentialitySystem.LEXICAL,
        honorifics=HonorificSystem.NONE,  # Lexical politeness, not grammatical
        agency_encoding=AgencyEncoding.WORD_ORDER,
        is_isolating=True,
        uses_word_spaces=False,
        script="chinese",
        has_grammatical_gender=False,
        has_formal_informal_distinction=False,
        topic_prominent=True
    ),
    "de": LanguageProfile(
        code="de",
        name="German",
        pro_drop=ProDropType.NONE,
        evidentiality=EvidentialitySystem.LEXICAL,
        honorifics=HonorificSystem.BINARY,  # du/Sie
        agency_encoding=AgencyEncoding.CASE_MARKING,
        is_fusional=True,
        script="latin",
        has_grammatical_gender=True,
        has_formal_informal_distinction=True,
        formality_in_pronouns=True,
        subject_prominent=True
    ),
    "fr": LanguageProfile(
        code="fr",
        name="French",
        pro_drop=ProDropType.NONE,
        evidentiality=EvidentialitySystem.LEXICAL,
        honorifics=HonorificSystem.BINARY,  # tu/vous
        agency_encoding=AgencyEncoding.WORD_ORDER,
        is_fusional=True,
        script="latin",
        has_grammatical_gender=True,
        has_formal_informal_distinction=True,
        formality_in_pronouns=True,
        subject_prominent=True
    ),
    "ar": LanguageProfile(
        code="ar",
        name="Arabic",
        pro_drop=ProDropType.PARTIAL,
        evidentiality=EvidentialitySystem.LEXICAL,
        honorifics=HonorificSystem.NONE,
        agency_encoding=AgencyEncoding.VERB_MORPHOLOGY,
        is_fusional=True,
        script="arabic",
        has_grammatical_gender=True,
        has_formal_informal_distinction=False,
        subject_prominent=True
    ),
}


# =============================================================================
# Extraction Evidence Tracking
# =============================================================================

@dataclass
class ExtractionEvidence:
    """
    Records the linguistic evidence used for each extraction.

    Critical for cross-linguistic validation: we need to know WHAT features
    contributed to each dimension to assess whether extractions are comparable.
    """
    dimension: str                      # "agency", "justice", "belonging"
    subdimension: str                   # "self", "other", "system", etc.
    language: str                       # Language code
    value: float                        # Extracted value

    # Evidence sources
    lexical_matches: List[str] = field(default_factory=list)
    morphological_features: Dict[str, str] = field(default_factory=dict)
    syntactic_patterns: List[str] = field(default_factory=list)
    discourse_markers: List[str] = field(default_factory=list)

    # Confidence and source type
    confidence: float = 0.5
    source_type: str = "mixed"  # "lexical", "morphological", "syntactic", "pragmatic"

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "subdimension": self.subdimension,
            "language": self.language,
            "value": self.value,
            "lexical_matches": self.lexical_matches,
            "morphological_features": self.morphological_features,
            "syntactic_patterns": self.syntactic_patterns,
            "discourse_markers": self.discourse_markers,
            "confidence": self.confidence,
            "source_type": self.source_type
        }


@dataclass
class ExtractionResult:
    """Complete extraction result with coordinate and evidence."""
    coordinate: HierarchicalCoordinate
    language: str
    evidence: List[ExtractionEvidence] = field(default_factory=list)
    extraction_method: str = "rule_based"
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "coordinate": self.coordinate.to_dict(),
            "language": self.language,
            "evidence": [e.to_dict() for e in self.evidence],
            "extraction_method": self.extraction_method,
            "processing_time_ms": self.processing_time_ms,
            "warnings": self.warnings
        }


# =============================================================================
# Abstract Base Class: LanguageExtractor
# =============================================================================

class LanguageExtractor(ABC):
    """
    Abstract base class for language-specific coordinate extraction.

    Each language implementation must provide:
    1. extract_agency() - Agency decomposition (self, other, system)
    2. extract_justice() - Justice decomposition (procedural, distributive, interactional)
    3. extract_belonging() - Belonging decomposition (ingroup, outgroup, universal)
    4. get_language_code() - ISO 639-1 language code

    Implementations should use language-appropriate features:
    - English: pronouns, word order
    - Spanish: verb morphology for person, tu/usted for formality
    - Japanese: particles, honorifics, topic marking
    """

    def __init__(self):
        self._nlp = None  # Lazy-loaded spacy model
        self._profile: Optional[LanguageProfile] = None
        self._evidence_log: List[ExtractionEvidence] = []

    @abstractmethod
    def extract_agency(self, doc: Any) -> AgencyDecomposition:
        """
        Extract agency decomposition from document.

        Args:
            doc: Processed document (spacy Doc or equivalent)

        Returns:
            AgencyDecomposition with self, other, system values
        """
        pass

    @abstractmethod
    def extract_justice(self, doc: Any) -> JusticeDecomposition:
        """
        Extract justice decomposition from document.

        Args:
            doc: Processed document

        Returns:
            JusticeDecomposition with procedural, distributive, interactional
        """
        pass

    @abstractmethod
    def extract_belonging(self, doc: Any) -> BelongingDecomposition:
        """
        Extract belonging decomposition from document.

        Args:
            doc: Processed document

        Returns:
            BelongingDecomposition with ingroup, outgroup, universal
        """
        pass

    @abstractmethod
    def get_language_code(self) -> str:
        """Return ISO 639-1 language code."""
        pass

    def get_language_profile(self) -> LanguageProfile:
        """Return the language profile for this extractor."""
        code = self.get_language_code()
        if self._profile is None:
            self._profile = LANGUAGE_PROFILES.get(code, self._create_default_profile())
        return self._profile

    def _create_default_profile(self) -> LanguageProfile:
        """Create default profile for unknown languages."""
        return LanguageProfile(
            code=self.get_language_code(),
            name=f"Unknown ({self.get_language_code()})"
        )

    def extract_modifiers(self, doc: Any) -> CoordinationModifiers:
        """
        Extract coordination modifiers.

        Override in subclasses for language-specific modifier extraction.
        Default implementation provides basic extraction.
        """
        return CoordinationModifiers(
            epistemic=self._extract_epistemic(doc),
            temporal=self._extract_temporal(doc),
            social=self._extract_social(doc),
            emotional=self._extract_emotional(doc)
        )

    def _extract_epistemic(self, doc: Any) -> EpistemicModifiers:
        """Extract epistemic modifiers. Override for language-specific handling."""
        return EpistemicModifiers()

    def _extract_temporal(self, doc: Any) -> TemporalModifiers:
        """Extract temporal modifiers. Override for language-specific handling."""
        return TemporalModifiers()

    def _extract_social(self, doc: Any) -> SocialModifiers:
        """Extract social modifiers. Override for language-specific handling."""
        return SocialModifiers()

    def _extract_emotional(self, doc: Any) -> EmotionalModifiers:
        """Extract emotional modifiers. Override for language-specific handling."""
        return EmotionalModifiers()

    def process_text(self, text: str) -> Any:
        """
        Process raw text into document format.

        Args:
            text: Raw input text

        Returns:
            Processed document (spacy Doc or equivalent)
        """
        if self._nlp is None:
            self._load_nlp()
        return self._nlp(text)

    def _load_nlp(self):
        """Load the NLP model. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _load_nlp()")

    def extract_full(self, text: str) -> ExtractionResult:
        """
        Extract complete hierarchical coordinate from text.

        Args:
            text: Raw input text

        Returns:
            ExtractionResult with coordinate and evidence
        """
        import time
        start_time = time.time()

        self._evidence_log = []  # Reset evidence log
        warnings = []

        try:
            doc = self.process_text(text)
        except Exception as e:
            warnings.append(f"NLP processing failed: {str(e)}. Using fallback.")
            doc = None

        # Extract core dimensions
        try:
            agency = self.extract_agency(doc) if doc else AgencyDecomposition()
        except Exception as e:
            warnings.append(f"Agency extraction failed: {str(e)}")
            agency = AgencyDecomposition()

        try:
            justice = self.extract_justice(doc) if doc else JusticeDecomposition()
        except Exception as e:
            warnings.append(f"Justice extraction failed: {str(e)}")
            justice = JusticeDecomposition()

        try:
            belonging = self.extract_belonging(doc) if doc else BelongingDecomposition()
        except Exception as e:
            warnings.append(f"Belonging extraction failed: {str(e)}")
            belonging = BelongingDecomposition()

        # Extract modifiers
        try:
            modifiers = self.extract_modifiers(doc) if doc else CoordinationModifiers()
        except Exception as e:
            warnings.append(f"Modifier extraction failed: {str(e)}")
            modifiers = CoordinationModifiers()

        # Build coordinate
        core = CoordinationCore(agency=agency, justice=justice, belonging=belonging)
        coordinate = HierarchicalCoordinate(core=core, modifiers=modifiers)

        elapsed_ms = (time.time() - start_time) * 1000

        return ExtractionResult(
            coordinate=coordinate,
            language=self.get_language_code(),
            evidence=self._evidence_log.copy(),
            extraction_method="rule_based",
            processing_time_ms=elapsed_ms,
            warnings=warnings
        )

    def _log_evidence(self, evidence: ExtractionEvidence):
        """Add evidence to the extraction log."""
        self._evidence_log.append(evidence)


# =============================================================================
# English Extractor (Baseline)
# =============================================================================

class EnglishExtractor(LanguageExtractor):
    """
    English language extractor - the baseline implementation.

    Uses dependency parsing (not regex) for robust extraction.
    English-specific features:
    - Explicit pronouns for person marking
    - Word order for agency (SVO)
    - Lexical markers for evidentiality
    """

    def __init__(self):
        super().__init__()
        self._agency_lexicon = self._build_agency_lexicon()
        self._justice_lexicon = self._build_justice_lexicon()
        self._belonging_lexicon = self._build_belonging_lexicon()

    def _load_nlp(self):
        """Load English spacy model."""
        try:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("en_core_web_sm not found. Attempting download...")
                from spacy.cli import download
                download("en_core_web_sm")
                self._nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded English spacy model: en_core_web_sm")
        except Exception as e:
            logger.error(f"Failed to load English NLP model: {e}")
            # Create a simple fallback tokenizer
            self._nlp = self._create_fallback_nlp()

    def _create_fallback_nlp(self):
        """Create a simple fallback when spacy isn't available."""
        class FallbackDoc:
            def __init__(self, text):
                self.text = text
                self.tokens = text.lower().split()
                self.sents = [self]  # Treat whole text as one sentence

            def __iter__(self):
                return iter([FallbackToken(t) for t in self.tokens])

        class FallbackToken:
            def __init__(self, text):
                self.text = text
                self.lower_ = text.lower()
                self.lemma_ = text.lower()
                self.pos_ = "UNKNOWN"
                self.dep_ = "UNKNOWN"
                self.head = self

        class FallbackNLP:
            def __call__(self, text):
                return FallbackDoc(text)

        return FallbackNLP()

    def get_language_code(self) -> str:
        return "en"

    def _build_agency_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Build lexicon for agency detection."""
        return {
            "self_agency": {
                "i": 0.3, "my": 0.2, "myself": 0.4, "me": 0.2,
                "chose": 0.5, "decided": 0.5, "achieved": 0.6, "created": 0.5,
                "built": 0.5, "made": 0.4, "earned": 0.5, "accomplished": 0.6,
                "determined": 0.4, "controlled": 0.4, "succeeded": 0.5,
                "initiated": 0.5, "led": 0.4, "pioneered": 0.6
            },
            "other_agency": {
                "they": 0.2, "their": 0.2, "them": 0.2, "you": 0.2, "your": 0.2,
                "he": 0.2, "she": 0.2, "his": 0.2, "her": 0.2,
                "forced": 0.4, "made": 0.3, "caused": 0.4, "influenced": 0.3,
                "manipulated": 0.5, "pressured": 0.4, "convinced": 0.3
            },
            "system_agency": {
                "system": 0.5, "institution": 0.5, "government": 0.5,
                "society": 0.4, "structure": 0.4, "forces": 0.4,
                "circumstances": 0.4, "conditions": 0.3, "environment": 0.3,
                "fate": 0.4, "destiny": 0.4, "luck": 0.3,
                "policy": 0.4, "law": 0.4, "regulation": 0.4,
                "inevitable": 0.5, "unavoidable": 0.5, "beyond": 0.3
            }
        }

    def _build_justice_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Build lexicon for justice detection."""
        return {
            "procedural": {
                "process": 0.5, "procedure": 0.5, "rule": 0.4, "regulation": 0.4,
                "review": 0.4, "appeal": 0.5, "due": 0.3, "hearing": 0.5,
                "trial": 0.5, "evaluation": 0.4, "assessment": 0.4,
                "protocol": 0.4, "standard": 0.3, "guideline": 0.4,
                "transparent": 0.4, "consistent": 0.4, "impartial": 0.5
            },
            "distributive": {
                "deserve": 0.5, "earned": 0.5, "merit": 0.5, "reward": 0.4,
                "punish": 0.4, "fair": 0.4, "share": 0.3, "distribute": 0.4,
                "equal": 0.4, "equity": 0.5, "proportional": 0.4,
                "allocation": 0.4, "resources": 0.3, "opportunity": 0.4,
                "outcome": 0.3, "result": 0.2, "consequences": 0.3
            },
            "interactional": {
                "respect": 0.5, "dignity": 0.5, "treat": 0.4, "listen": 0.4,
                "hear": 0.3, "acknowledge": 0.4, "recognize": 0.4,
                "courteous": 0.4, "polite": 0.3, "considerate": 0.4,
                "explained": 0.3, "informed": 0.3, "consulted": 0.4,
                "included": 0.4, "valued": 0.5, "appreciated": 0.4
            }
        }

    def _build_belonging_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Build lexicon for belonging detection."""
        return {
            "ingroup": {
                "we": 0.4, "us": 0.4, "our": 0.4, "ours": 0.4,
                "together": 0.5, "community": 0.5, "team": 0.4, "family": 0.5,
                "tribe": 0.5, "group": 0.4, "members": 0.3, "brothers": 0.5,
                "sisters": 0.5, "comrades": 0.5, "fellows": 0.4,
                "belong": 0.5, "united": 0.5, "solidarity": 0.5
            },
            "outgroup": {
                "them": 0.3, "they": 0.2, "those": 0.3, "others": 0.3,
                "outsiders": 0.5, "foreigners": 0.5, "strangers": 0.4,
                "enemies": 0.6, "opponents": 0.4, "rivals": 0.4,
                "different": 0.2, "separate": 0.3, "apart": 0.3,
                "exclude": 0.5, "reject": 0.5, "alien": 0.5
            },
            "universal": {
                "everyone": 0.5, "everybody": 0.5, "humanity": 0.6,
                "people": 0.3, "human": 0.4, "universal": 0.5,
                "global": 0.4, "world": 0.4, "mankind": 0.5,
                "citizens": 0.3, "persons": 0.3, "individuals": 0.3,
                "all": 0.3, "anyone": 0.4, "anyone": 0.4
            }
        }

    def extract_agency(self, doc: Any) -> AgencyDecomposition:
        """
        Extract agency using dependency parsing.

        Strategy:
        1. Identify clauses with first-person subjects -> self_agency
        2. Identify clauses with third-person/second-person subjects -> other_agency
        3. Identify passive constructions and system nouns -> system_agency
        """
        if doc is None:
            return AgencyDecomposition()

        self_score = 0.0
        other_score = 0.0
        system_score = 0.0

        evidence_lexical = []
        evidence_syntactic = []

        # Count tokens and check for patterns
        token_count = 0
        for token in doc:
            token_count += 1
            lemma = token.lemma_.lower() if hasattr(token, 'lemma_') else token.text.lower()

            # Self agency markers
            if lemma in self._agency_lexicon["self_agency"]:
                weight = self._agency_lexicon["self_agency"][lemma]
                self_score += weight
                evidence_lexical.append(token.text)

            # Other agency markers
            if lemma in self._agency_lexicon["other_agency"]:
                weight = self._agency_lexicon["other_agency"][lemma]
                other_score += weight
                evidence_lexical.append(token.text)

            # System agency markers
            if lemma in self._agency_lexicon["system_agency"]:
                weight = self._agency_lexicon["system_agency"][lemma]
                system_score += weight
                evidence_lexical.append(token.text)

        # Check for passive voice (reduces agency)
        if hasattr(doc, 'sents'):
            for sent in doc.sents:
                sent_text = sent.text.lower() if hasattr(sent, 'text') else str(sent).lower()
                if " was " in sent_text or " were " in sent_text or " been " in sent_text:
                    if " by " in sent_text:  # Passive with agent
                        system_score += 0.2
                        evidence_syntactic.append("passive_with_agent")
                    else:  # Agentless passive
                        system_score += 0.3
                        evidence_syntactic.append("agentless_passive")

        # Normalize by text length
        normalizer = max(1, token_count / 20)
        self_score = min(1.0, self_score / normalizer)
        other_score = min(1.0, other_score / normalizer)
        system_score = min(1.0, system_score / normalizer)

        # Log evidence
        self._log_evidence(ExtractionEvidence(
            dimension="agency",
            subdimension="combined",
            language="en",
            value=self_score - system_score,
            lexical_matches=evidence_lexical,
            syntactic_patterns=evidence_syntactic,
            confidence=0.7 if evidence_lexical else 0.3,
            source_type="lexical" if not evidence_syntactic else "mixed"
        ))

        return AgencyDecomposition(
            self_agency=self_score,
            other_agency=other_score,
            system_agency=system_score
        )

    def extract_justice(self, doc: Any) -> JusticeDecomposition:
        """Extract justice dimensions from English text."""
        if doc is None:
            return JusticeDecomposition()

        procedural_score = 0.0
        distributive_score = 0.0
        interactional_score = 0.0

        evidence_lexical = []
        token_count = 0

        for token in doc:
            token_count += 1
            lemma = token.lemma_.lower() if hasattr(token, 'lemma_') else token.text.lower()

            if lemma in self._justice_lexicon["procedural"]:
                procedural_score += self._justice_lexicon["procedural"][lemma]
                evidence_lexical.append(token.text)

            if lemma in self._justice_lexicon["distributive"]:
                distributive_score += self._justice_lexicon["distributive"][lemma]
                evidence_lexical.append(token.text)

            if lemma in self._justice_lexicon["interactional"]:
                interactional_score += self._justice_lexicon["interactional"][lemma]
                evidence_lexical.append(token.text)

        # Normalize
        normalizer = max(1, token_count / 20)
        procedural_score = min(1.0, procedural_score / normalizer)
        distributive_score = min(1.0, distributive_score / normalizer)
        interactional_score = min(1.0, interactional_score / normalizer)

        self._log_evidence(ExtractionEvidence(
            dimension="justice",
            subdimension="combined",
            language="en",
            value=(procedural_score + distributive_score + interactional_score) / 3,
            lexical_matches=evidence_lexical,
            confidence=0.6 if evidence_lexical else 0.3,
            source_type="lexical"
        ))

        return JusticeDecomposition(
            procedural=procedural_score,
            distributive=distributive_score,
            interactional=interactional_score
        )

    def extract_belonging(self, doc: Any) -> BelongingDecomposition:
        """Extract belonging dimensions from English text."""
        if doc is None:
            return BelongingDecomposition()

        ingroup_score = 0.0
        outgroup_score = 0.0
        universal_score = 0.0

        evidence_lexical = []
        token_count = 0

        for token in doc:
            token_count += 1
            lemma = token.lemma_.lower() if hasattr(token, 'lemma_') else token.text.lower()

            if lemma in self._belonging_lexicon["ingroup"]:
                ingroup_score += self._belonging_lexicon["ingroup"][lemma]
                evidence_lexical.append(token.text)

            if lemma in self._belonging_lexicon["outgroup"]:
                outgroup_score += self._belonging_lexicon["outgroup"][lemma]
                evidence_lexical.append(token.text)

            if lemma in self._belonging_lexicon["universal"]:
                universal_score += self._belonging_lexicon["universal"][lemma]
                evidence_lexical.append(token.text)

        # Normalize
        normalizer = max(1, token_count / 20)
        ingroup_score = min(1.0, ingroup_score / normalizer)
        outgroup_score = min(1.0, outgroup_score / normalizer)
        universal_score = min(1.0, universal_score / normalizer)

        self._log_evidence(ExtractionEvidence(
            dimension="belonging",
            subdimension="combined",
            language="en",
            value=ingroup_score + universal_score - outgroup_score,
            lexical_matches=evidence_lexical,
            confidence=0.6 if evidence_lexical else 0.3,
            source_type="lexical"
        ))

        return BelongingDecomposition(
            ingroup=ingroup_score,
            outgroup=outgroup_score,
            universal=universal_score
        )

    def _extract_epistemic(self, doc: Any) -> EpistemicModifiers:
        """Extract epistemic modifiers from English text."""
        if doc is None:
            return EpistemicModifiers()

        certainty_high = {"definitely", "certainly", "absolutely", "clearly", "obviously"}
        certainty_low = {"maybe", "perhaps", "possibly", "might", "could", "uncertain"}
        evidential_direct = {"saw", "heard", "felt", "witnessed", "experienced"}
        evidential_reported = {"apparently", "supposedly", "allegedly", "reportedly", "said"}

        certainty = 0.0
        evidentiality = 0.0

        for token in doc:
            lemma = token.lemma_.lower() if hasattr(token, 'lemma_') else token.text.lower()
            if lemma in certainty_high:
                certainty += 0.3
            elif lemma in certainty_low:
                certainty -= 0.3

            if lemma in evidential_direct:
                evidentiality += 0.3
            elif lemma in evidential_reported:
                evidentiality -= 0.3

        return EpistemicModifiers(
            certainty=max(-1.0, min(1.0, certainty)),
            evidentiality=max(-1.0, min(1.0, evidentiality)),
            commitment=0.0
        )

    def _extract_social(self, doc: Any) -> SocialModifiers:
        """Extract social modifiers from English text."""
        if doc is None:
            return SocialModifiers()

        formal_markers = {"therefore", "hence", "consequently", "furthermore", "hereby", "thus"}
        informal_markers = {"gonna", "wanna", "kinda", "gotta", "yeah", "nope", "cool", "awesome"}

        formality = 0.0
        for token in doc:
            text = token.text.lower() if hasattr(token, 'text') else str(token).lower()
            if text in formal_markers:
                formality += 0.2
            elif text in informal_markers:
                formality -= 0.2

        return SocialModifiers(
            power_differential=0.0,  # Hard to detect without context
            social_distance=max(-1.0, min(1.0, formality))
        )


# =============================================================================
# Spanish Extractor (Pro-Drop Example)
# =============================================================================

class SpanishExtractor(LanguageExtractor):
    """
    Spanish language extractor - demonstrates pro-drop handling.

    Key linguistic features:
    1. Pro-drop: Subject pronouns often omitted, encoded in verb morphology
       - "Quiero" = "I want" (first person in -o ending)
       - "Quiere" = "He/She wants" (third person in -e ending)

    2. Formality distinction:
       - tu/vosotros = informal (2nd person)
       - usted/ustedes = formal (2nd person, conjugated as 3rd)

    3. Verb morphology encodes person and number:
       - -o: 1st person singular (yo)
       - -as/-es: 2nd person singular informal (tu)
       - -a/-e: 3rd person singular / 2nd formal (el/ella/usted)
       - -amos/-emos/-imos: 1st person plural (nosotros)
       - -ais/-eis/-is: 2nd person plural informal (vosotros)
       - -an/-en: 3rd person plural / 2nd plural formal (ellos/ustedes)
    """

    def __init__(self):
        super().__init__()
        self._verb_person_patterns = self._build_verb_patterns()
        self._agency_lexicon = self._build_spanish_agency_lexicon()
        self._justice_lexicon = self._build_spanish_justice_lexicon()
        self._belonging_lexicon = self._build_spanish_belonging_lexicon()

    def _load_nlp(self):
        """Load Spanish spacy model."""
        try:
            import spacy
            try:
                self._nlp = spacy.load("es_core_news_sm")
            except OSError:
                logger.warning("es_core_news_sm not found. Using fallback tokenizer.")
                self._nlp = self._create_fallback_nlp()
            logger.info("Loaded Spanish NLP model")
        except Exception as e:
            logger.error(f"Failed to load Spanish NLP: {e}")
            self._nlp = self._create_fallback_nlp()

    def _create_fallback_nlp(self):
        """Create fallback tokenizer for Spanish."""
        class FallbackDoc:
            def __init__(self, text):
                self.text = text
                # Basic Spanish tokenization
                self.tokens = re.findall(r'\b\w+\b', text.lower())
                self.sents = [self]

            def __iter__(self):
                return iter([FallbackToken(t) for t in self.tokens])

        class FallbackToken:
            def __init__(self, text):
                self.text = text
                self.lower_ = text.lower()
                self.lemma_ = text.lower()
                self.pos_ = "UNKNOWN"
                self.morph = FallbackMorph()

        class FallbackMorph:
            def get(self, key, default=None):
                return default

        class FallbackNLP:
            def __call__(self, text):
                return FallbackDoc(text)

        return FallbackNLP()

    def get_language_code(self) -> str:
        return "es"

    def _build_verb_patterns(self) -> Dict[str, str]:
        """
        Build regex patterns for Spanish verb person detection.

        These patterns detect person from verb endings when
        subject pronouns are dropped (pro-drop).
        """
        return {
            # First person singular: -o (present), -e (preterite -ar verbs), -i (preterite -er/-ir)
            "1sg": r'\b\w+(o|e|i)\b',  # Simplified; real implementation needs conjugation tables

            # Second person singular informal: -as (ar), -es (er/ir)
            "2sg_informal": r'\b\w+(as|es)\b',

            # Third person singular / second formal: -a (ar), -e (er/ir)
            "3sg": r'\b\w+(a|e)\b',

            # First person plural: -amos, -emos, -imos
            "1pl": r'\b\w+(amos|emos|imos)\b',

            # Second person plural informal (vosotros): -ais, -eis, -is
            "2pl_informal": r'\b\w+(ais|eis|is)\b',

            # Third person plural / second plural formal: -an, -en
            "3pl": r'\b\w+(an|en)\b'
        }

    def _build_spanish_agency_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Spanish agency lexicon."""
        return {
            "self_agency": {
                "yo": 0.3, "mi": 0.2, "mio": 0.2, "conmigo": 0.3,
                "decidi": 0.5, "elegi": 0.5, "logre": 0.6, "consegui": 0.5,
                "cree": 0.5, "hice": 0.5, "construi": 0.5
            },
            "other_agency": {
                "tu": 0.2, "el": 0.2, "ella": 0.2, "ellos": 0.2, "ellas": 0.2,
                "usted": 0.2, "ustedes": 0.2,
                "obligaron": 0.4, "forzaron": 0.5, "causaron": 0.4
            },
            "system_agency": {
                "sistema": 0.5, "gobierno": 0.5, "institucion": 0.5,
                "sociedad": 0.4, "estructura": 0.4, "fuerzas": 0.4,
                "circunstancias": 0.4, "destino": 0.4, "suerte": 0.3,
                "ley": 0.4, "politica": 0.4, "inevitable": 0.5
            }
        }

    def _build_spanish_justice_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Spanish justice lexicon."""
        return {
            "procedural": {
                "proceso": 0.5, "procedimiento": 0.5, "regla": 0.4,
                "reglamento": 0.4, "revision": 0.4, "apelacion": 0.5,
                "juicio": 0.5, "audiencia": 0.5, "transparente": 0.4
            },
            "distributive": {
                "merecer": 0.5, "ganar": 0.4, "merito": 0.5,
                "recompensa": 0.4, "castigo": 0.4, "justo": 0.5,
                "igual": 0.4, "equidad": 0.5, "distribuir": 0.4
            },
            "interactional": {
                "respeto": 0.5, "dignidad": 0.5, "tratar": 0.4,
                "escuchar": 0.4, "reconocer": 0.4, "valorar": 0.5,
                "apreciar": 0.4, "considerar": 0.4
            }
        }

    def _build_spanish_belonging_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Spanish belonging lexicon."""
        return {
            "ingroup": {
                "nosotros": 0.5, "nosotras": 0.5, "nuestro": 0.4, "nuestra": 0.4,
                "juntos": 0.5, "comunidad": 0.5, "equipo": 0.4, "familia": 0.5,
                "hermanos": 0.5, "hermanas": 0.5, "companeros": 0.4,
                "unidos": 0.5, "solidaridad": 0.5, "pueblo": 0.4
            },
            "outgroup": {
                "ellos": 0.2, "ellas": 0.2, "esos": 0.3, "aquellos": 0.3,
                "extranjeros": 0.5, "forasteros": 0.4, "enemigos": 0.6,
                "rivales": 0.4, "excluir": 0.5, "rechazar": 0.5
            },
            "universal": {
                "todos": 0.5, "todas": 0.5, "humanidad": 0.6,
                "gente": 0.3, "humano": 0.4, "universal": 0.5,
                "mundial": 0.4, "mundo": 0.4, "personas": 0.3
            }
        }

    def _detect_person_from_verb(self, token: Any) -> Optional[str]:
        """
        Detect grammatical person from Spanish verb morphology.

        This is the key pro-drop handling: we extract person information
        from verb endings when subject pronouns are absent.
        """
        text = token.text.lower() if hasattr(token, 'text') else str(token).lower()

        # Check morphological features if available (spacy)
        if hasattr(token, 'morph'):
            person = token.morph.get('Person', [])
            number = token.morph.get('Number', [])
            if person and number:
                return f"{person[0]}{'sg' if 'Sing' in number else 'pl'}"

        # Fallback: pattern matching on verb endings
        # First person singular (-o for present tense)
        if text.endswith('o') and len(text) > 2:
            return "1sg"

        # First person plural (-amos, -emos, -imos)
        if re.search(r'(amos|emos|imos)$', text):
            return "1pl"

        # Second person singular informal (-as, -es)
        if re.search(r'(as|es)$', text) and len(text) > 3:
            return "2sg"

        # Third/second formal (-a, -e for singular; -an, -en for plural)
        if re.search(r'(an|en)$', text) and len(text) > 3:
            return "3pl"

        return None

    def extract_agency(self, doc: Any) -> AgencyDecomposition:
        """
        Extract agency from Spanish text, handling pro-drop.

        Key insight: In Spanish, we must look at VERB MORPHOLOGY
        to detect first-person agency when pronouns are dropped.
        """
        if doc is None:
            return AgencyDecomposition()

        self_score = 0.0
        other_score = 0.0
        system_score = 0.0

        evidence_lexical = []
        evidence_morphological = {}
        token_count = 0

        first_person_verbs = 0
        third_person_verbs = 0

        for token in doc:
            token_count += 1
            text = token.text.lower() if hasattr(token, 'text') else str(token).lower()
            lemma = token.lemma_.lower() if hasattr(token, 'lemma_') else text

            # Check explicit pronouns and agency words
            if lemma in self._agency_lexicon["self_agency"]:
                self_score += self._agency_lexicon["self_agency"][lemma]
                evidence_lexical.append(text)

            if lemma in self._agency_lexicon["other_agency"]:
                other_score += self._agency_lexicon["other_agency"][lemma]
                evidence_lexical.append(text)

            if lemma in self._agency_lexicon["system_agency"]:
                system_score += self._agency_lexicon["system_agency"][lemma]
                evidence_lexical.append(text)

            # PRO-DROP HANDLING: Check verb morphology
            pos = getattr(token, 'pos_', 'UNKNOWN')
            if pos == 'VERB' or (text.endswith('o') and len(text) > 2):
                person = self._detect_person_from_verb(token)
                if person:
                    evidence_morphological[text] = person
                    if person.startswith('1'):  # First person
                        first_person_verbs += 1
                        self_score += 0.2
                    elif person.startswith('3'):  # Third person
                        third_person_verbs += 1
                        other_score += 0.1

        # Normalize
        normalizer = max(1, token_count / 20)
        self_score = min(1.0, self_score / normalizer)
        other_score = min(1.0, other_score / normalizer)
        system_score = min(1.0, system_score / normalizer)

        # Log evidence
        self._log_evidence(ExtractionEvidence(
            dimension="agency",
            subdimension="combined",
            language="es",
            value=self_score - system_score,
            lexical_matches=evidence_lexical,
            morphological_features=evidence_morphological,
            confidence=0.7 if evidence_morphological else 0.5,
            source_type="morphological" if evidence_morphological else "lexical"
        ))

        return AgencyDecomposition(
            self_agency=self_score,
            other_agency=other_score,
            system_agency=system_score
        )

    def extract_justice(self, doc: Any) -> JusticeDecomposition:
        """Extract justice from Spanish text."""
        if doc is None:
            return JusticeDecomposition()

        procedural_score = 0.0
        distributive_score = 0.0
        interactional_score = 0.0

        token_count = 0
        for token in doc:
            token_count += 1
            lemma = token.lemma_.lower() if hasattr(token, 'lemma_') else token.text.lower()

            if lemma in self._justice_lexicon["procedural"]:
                procedural_score += self._justice_lexicon["procedural"][lemma]
            if lemma in self._justice_lexicon["distributive"]:
                distributive_score += self._justice_lexicon["distributive"][lemma]
            if lemma in self._justice_lexicon["interactional"]:
                interactional_score += self._justice_lexicon["interactional"][lemma]

        normalizer = max(1, token_count / 20)
        return JusticeDecomposition(
            procedural=min(1.0, procedural_score / normalizer),
            distributive=min(1.0, distributive_score / normalizer),
            interactional=min(1.0, interactional_score / normalizer)
        )

    def extract_belonging(self, doc: Any) -> BelongingDecomposition:
        """Extract belonging from Spanish text."""
        if doc is None:
            return BelongingDecomposition()

        ingroup_score = 0.0
        outgroup_score = 0.0
        universal_score = 0.0

        token_count = 0
        for token in doc:
            token_count += 1
            lemma = token.lemma_.lower() if hasattr(token, 'lemma_') else token.text.lower()

            if lemma in self._belonging_lexicon["ingroup"]:
                ingroup_score += self._belonging_lexicon["ingroup"][lemma]
            if lemma in self._belonging_lexicon["outgroup"]:
                outgroup_score += self._belonging_lexicon["outgroup"][lemma]
            if lemma in self._belonging_lexicon["universal"]:
                universal_score += self._belonging_lexicon["universal"][lemma]

        normalizer = max(1, token_count / 20)
        return BelongingDecomposition(
            ingroup=min(1.0, ingroup_score / normalizer),
            outgroup=min(1.0, outgroup_score / normalizer),
            universal=min(1.0, universal_score / normalizer)
        )

    def _extract_social(self, doc: Any) -> SocialModifiers:
        """
        Extract social modifiers, including tu/usted formality.

        The tu/usted distinction is critical for Spanish social dynamics.
        """
        if doc is None:
            return SocialModifiers()

        informal_markers = {"tu", "tus", "ti", "contigo", "vosotros", "vosotras"}
        formal_markers = {"usted", "ustedes", "su", "sus"}  # Note: su/sus are ambiguous

        formality = 0.0
        for token in doc:
            text = token.text.lower() if hasattr(token, 'text') else str(token).lower()
            if text in informal_markers:
                formality -= 0.3
            elif text in formal_markers:
                formality += 0.3

        return SocialModifiers(
            power_differential=0.0,
            social_distance=max(-1.0, min(1.0, formality))
        )


# =============================================================================
# Japanese Extractor (Honorifics + Topic Marking Example)
# =============================================================================

class JapaneseExtractor(LanguageExtractor):
    """
    Japanese language extractor - demonstrates honorific and topic-marking handling.

    Key linguistic features:
    1. TOPIC MARKING (wa/ga distinction):
       - は (wa) marks topic: what we're talking about
       - が (ga) marks subject: who performs action (more agentive)

    2. HONORIFIC SYSTEM (keigo):
       - 敬語 (keigo) has multiple levels affecting social dimension extraction
       - 丁寧語 (teineigo): polite language (-masu, -desu forms)
       - 尊敬語 (sonkeigo): respectful language (elevating others)
       - 謙譲語 (kenjougo): humble language (lowering self)

    3. SENTENCE-FINAL PARTICLES for evidentiality/certainty:
       - よ (yo): assertion, certainty
       - ね (ne): seeking agreement, confirmation
       - か (ka): question
       - って (tte): hearsay, quotative
       - らしい (rashii): seems like, apparently
       - そうだ (souda): I heard that

    4. FULL PRO-DROP:
       - Subject, object, and topic all frequently dropped
       - Context and honorifics help recover referents
    """

    def __init__(self):
        super().__init__()
        self._particles = self._build_particle_lexicon()
        self._honorifics = self._build_honorific_patterns()
        self._agency_lexicon = self._build_japanese_agency_lexicon()
        self._belonging_lexicon = self._build_japanese_belonging_lexicon()

    def _load_nlp(self):
        """Load Japanese spacy model."""
        try:
            import spacy
            try:
                self._nlp = spacy.load("ja_core_news_sm")
            except OSError:
                logger.warning("ja_core_news_sm not found. Using fallback tokenizer.")
                self._nlp = self._create_fallback_nlp()
            logger.info("Loaded Japanese NLP model")
        except Exception as e:
            logger.error(f"Failed to load Japanese NLP: {e}")
            self._nlp = self._create_fallback_nlp()

    def _create_fallback_nlp(self):
        """Create fallback tokenizer for Japanese."""
        class FallbackDoc:
            def __init__(self, text):
                self.text = text
                # Very basic Japanese tokenization (character-based as fallback)
                # Real Japanese NLP needs morphological analysis (MeCab, etc.)
                self.tokens = list(text)
                self.sents = [self]

            def __iter__(self):
                return iter([FallbackToken(t) for t in self.tokens])

        class FallbackToken:
            def __init__(self, text):
                self.text = text
                self.lower_ = text
                self.lemma_ = text
                self.pos_ = "UNKNOWN"

        class FallbackNLP:
            def __call__(self, text):
                return FallbackDoc(text)

        return FallbackNLP()

    def get_language_code(self) -> str:
        return "ja"

    def _build_particle_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Build lexicon of Japanese particles and their meanings."""
        return {
            # Case/topic particles
            "topic_subject": {
                "は": ("topic", 0.0),      # Topic marker
                "が": ("subject", 0.3),    # Subject marker (more agentive)
                "を": ("object", -0.1),    # Object marker
                "に": ("indirect", 0.0),   # Indirect object/location
                "で": ("means", 0.1),      # Means/location
                "へ": ("direction", 0.0),  # Direction
                "と": ("with", 0.1),       # With/quotation
                "から": ("from", 0.0),     # From
                "まで": ("until", 0.0),    # Until
            },
            # Sentence-final particles (evidentiality/stance)
            "sentence_final": {
                "よ": ("assertion", 0.4),     # Strong assertion
                "ね": ("confirmation", 0.1),   # Seeking agreement
                "か": ("question", -0.1),      # Question
                "わ": ("feminine_assertion", 0.3),  # Feminine assertion
                "ぞ": ("masculine_assertion", 0.4), # Masculine strong assertion
                "の": ("explanatory", 0.0),    # Explanation
                "さ": ("casual_assertion", 0.2),   # Casual assertion
            },
            # Evidential markers
            "evidential": {
                "らしい": ("hearsay", -0.3),     # Seems like (hearsay)
                "そうだ": ("reported", -0.4),    # I heard that
                "ようだ": ("appearance", -0.2),  # Appears to be
                "みたいだ": ("like", -0.2),      # Looks like
                "だろう": ("conjecture", -0.2),  # Probably
                "かもしれない": ("uncertain", -0.4), # Might be
            }
        }

    def _build_honorific_patterns(self) -> Dict[str, float]:
        """
        Build patterns for detecting honorific levels.

        Returns social distance score:
        - Negative: humble/informal
        - Positive: polite/formal/respectful
        """
        return {
            # Polite verb endings (-masu/-desu)
            "ます": 0.3,    # Polite verb ending
            "です": 0.3,    # Polite copula
            "ません": 0.3,  # Polite negative
            "でした": 0.3,  # Polite past copula
            "ました": 0.3,  # Polite past verb

            # Humble forms (kenjougo)
            "いたします": 0.5,   # Humble do
            "申します": 0.5,     # Humble say
            "参ります": 0.5,     # Humble go/come
            "おります": 0.5,     # Humble be
            "いただきます": 0.5, # Humble receive

            # Respectful forms (sonkeigo)
            "いらっしゃいます": 0.6,  # Respectful be/go/come
            "おっしゃいます": 0.6,    # Respectful say
            "なさいます": 0.6,        # Respectful do
            "ご覧になります": 0.6,    # Respectful see

            # Informal/plain forms
            "だ": -0.2,       # Plain copula
            "する": -0.1,     # Plain do
            "である": 0.1,    # Written formal (neither humble nor informal)

            # Casual particles
            "ぜ": -0.3,       # Casual masculine
            "な": -0.2,       # Casual (can be emphatic or prohibitive)
        }

    def _build_japanese_agency_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Japanese agency-related words."""
        return {
            "self_agency": {
                "私": 0.3,      # I (watashi)
                "僕": 0.3,      # I (boku, male)
                "俺": 0.3,      # I (ore, male casual)
                "自分": 0.4,    # Self
                "決めた": 0.5,  # Decided
                "選んだ": 0.5,  # Chose
                "作った": 0.5,  # Made/created
                "できた": 0.4,  # Was able to
                "頑張った": 0.5, # Tried hard
                "努力": 0.4,    # Effort
            },
            "other_agency": {
                "彼": 0.2,      # He
                "彼女": 0.2,    # She
                "彼ら": 0.2,    # They
                "あなた": 0.2,  # You
                "あの人": 0.2,  # That person
                "させられた": 0.4,  # Was made to (causative-passive)
            },
            "system_agency": {
                "社会": 0.5,    # Society
                "制度": 0.5,    # System/institution
                "政府": 0.5,    # Government
                "運命": 0.4,    # Fate
                "仕方ない": 0.5, # Can't be helped
                "しょうがない": 0.5, # Can't be helped (casual)
                "環境": 0.4,    # Environment
                "状況": 0.4,    # Situation
                "避けられない": 0.5, # Unavoidable
            }
        }

    def _build_japanese_belonging_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Japanese belonging-related words."""
        return {
            "ingroup": {
                "私たち": 0.5,  # We (watashi-tachi)
                "我々": 0.5,    # We (wareware, formal)
                "うち": 0.4,    # Our (group)/inside
                "仲間": 0.5,    # Companions
                "皆": 0.4,      # Everyone (in group)
                "家族": 0.5,    # Family
                "一緒": 0.5,    # Together
                "団結": 0.5,    # Unity
                "協力": 0.4,    # Cooperation
            },
            "outgroup": {
                "彼ら": 0.2,    # They (can be neutral or outgroup)
                "あいつら": 0.4, # Those guys (dismissive)
                "外国人": 0.3,  # Foreigners
                "よそ者": 0.5,  # Outsider
                "敵": 0.6,      # Enemy
                "他人": 0.3,    # Others/strangers
            },
            "universal": {
                "皆さん": 0.4,  # Everyone (polite)
                "人類": 0.6,    # Humanity
                "人間": 0.4,    # Human beings
                "世界": 0.4,    # World
                "全員": 0.5,    # Everyone (all members)
                "人々": 0.4,    # People
            }
        }

    def _detect_topic_subject_marking(self, doc: Any) -> Dict[str, int]:
        """
        Detect wa/ga topic-subject marking.

        が (ga) typically marks more agentive subjects,
        は (wa) marks topics which may or may not be agents.
        """
        counts = {"topic_wa": 0, "subject_ga": 0}

        text = doc.text if hasattr(doc, 'text') else str(doc)

        # Count particles
        counts["topic_wa"] = text.count("は")
        counts["subject_ga"] = text.count("が")

        return counts

    def _detect_honorific_level(self, doc: Any) -> float:
        """
        Detect overall honorific level of the text.

        Returns score from -1 (very casual) to +1 (very formal/respectful).
        """
        text = doc.text if hasattr(doc, 'text') else str(doc)

        total_score = 0.0
        matches = 0

        for pattern, score in self._honorifics.items():
            count = text.count(pattern)
            if count > 0:
                total_score += score * count
                matches += count

        if matches == 0:
            return 0.0

        return max(-1.0, min(1.0, total_score / matches))

    def _detect_evidentiality(self, doc: Any) -> float:
        """
        Detect evidentiality markers.

        Japanese uses sentence-final forms to mark information source:
        - Direct experience: more certain
        - Hearsay (rashii, souda): less certain
        """
        text = doc.text if hasattr(doc, 'text') else str(doc)

        score = 0.0

        for marker, (meaning, weight) in self._particles["evidential"].items():
            if marker in text:
                score += weight

        # Check sentence-final particles
        for particle, (meaning, weight) in self._particles["sentence_final"].items():
            if text.endswith(particle) or f"{particle}。" in text:
                score += weight

        return max(-1.0, min(1.0, score))

    def extract_agency(self, doc: Any) -> AgencyDecomposition:
        """
        Extract agency from Japanese text.

        Key features:
        1. が (ga) marking indicates clearer agency
        2. Causative-passive forms reduce agency
        3. 仕方ない/しょうがない = high system agency
        """
        if doc is None:
            return AgencyDecomposition()

        self_score = 0.0
        other_score = 0.0
        system_score = 0.0

        text = doc.text if hasattr(doc, 'text') else str(doc)

        evidence_lexical = []
        evidence_syntactic = []

        # Check topic/subject marking
        marking = self._detect_topic_subject_marking(doc)
        if marking["subject_ga"] > marking["topic_wa"]:
            self_score += 0.2  # More explicit agency marking
            evidence_syntactic.append("high_ga_marking")

        # Check lexical items
        for word, weight in self._agency_lexicon["self_agency"].items():
            if word in text:
                self_score += weight
                evidence_lexical.append(word)

        for word, weight in self._agency_lexicon["other_agency"].items():
            if word in text:
                other_score += weight
                evidence_lexical.append(word)

        for word, weight in self._agency_lexicon["system_agency"].items():
            if word in text:
                system_score += weight
                evidence_lexical.append(word)

        # Check for causative-passive (させられた) - reduces self agency
        if "させられ" in text:
            other_score += 0.3
            self_score -= 0.2
            evidence_syntactic.append("causative_passive")

        # Normalize by text length
        char_count = len(text)
        normalizer = max(1, char_count / 50)  # Japanese: chars instead of words

        self_score = min(1.0, self_score / normalizer)
        other_score = min(1.0, other_score / normalizer)
        system_score = min(1.0, system_score / normalizer)

        self._log_evidence(ExtractionEvidence(
            dimension="agency",
            subdimension="combined",
            language="ja",
            value=self_score - system_score,
            lexical_matches=evidence_lexical,
            syntactic_patterns=evidence_syntactic,
            confidence=0.6,
            source_type="mixed"
        ))

        return AgencyDecomposition(
            self_agency=self_score,
            other_agency=other_score,
            system_agency=system_score
        )

    def extract_justice(self, doc: Any) -> JusticeDecomposition:
        """Extract justice from Japanese text."""
        if doc is None:
            return JusticeDecomposition()

        text = doc.text if hasattr(doc, 'text') else str(doc)

        # Japanese justice lexicon
        procedural_words = {"手続き": 0.5, "規則": 0.4, "審査": 0.5, "公正": 0.5}
        distributive_words = {"報酬": 0.4, "罰": 0.4, "公平": 0.5, "平等": 0.5}
        interactional_words = {"尊重": 0.5, "礼儀": 0.4, "敬意": 0.5, "丁寧": 0.4}

        procedural_score = sum(w for word, w in procedural_words.items() if word in text)
        distributive_score = sum(w for word, w in distributive_words.items() if word in text)
        interactional_score = sum(w for word, w in interactional_words.items() if word in text)

        # Normalize
        char_count = len(text)
        normalizer = max(1, char_count / 50)

        return JusticeDecomposition(
            procedural=min(1.0, procedural_score / normalizer),
            distributive=min(1.0, distributive_score / normalizer),
            interactional=min(1.0, interactional_score / normalizer)
        )

    def extract_belonging(self, doc: Any) -> BelongingDecomposition:
        """
        Extract belonging from Japanese text.

        Japanese has strong ingroup/outgroup (uchi/soto) distinctions.
        """
        if doc is None:
            return BelongingDecomposition()

        text = doc.text if hasattr(doc, 'text') else str(doc)

        ingroup_score = 0.0
        outgroup_score = 0.0
        universal_score = 0.0

        for word, weight in self._belonging_lexicon["ingroup"].items():
            if word in text:
                ingroup_score += weight

        for word, weight in self._belonging_lexicon["outgroup"].items():
            if word in text:
                outgroup_score += weight

        for word, weight in self._belonging_lexicon["universal"].items():
            if word in text:
                universal_score += weight

        # Normalize
        char_count = len(text)
        normalizer = max(1, char_count / 50)

        return BelongingDecomposition(
            ingroup=min(1.0, ingroup_score / normalizer),
            outgroup=min(1.0, outgroup_score / normalizer),
            universal=min(1.0, universal_score / normalizer)
        )

    def _extract_epistemic(self, doc: Any) -> EpistemicModifiers:
        """Extract epistemic modifiers using sentence-final particles."""
        if doc is None:
            return EpistemicModifiers()

        certainty = self._detect_evidentiality(doc)

        return EpistemicModifiers(
            certainty=certainty,
            evidentiality=certainty,  # In Japanese, these are closely linked
            commitment=0.0
        )

    def _extract_social(self, doc: Any) -> SocialModifiers:
        """
        Extract social modifiers using honorific analysis.

        The keigo system directly encodes power differential and social distance.
        """
        if doc is None:
            return SocialModifiers()

        honorific_level = self._detect_honorific_level(doc)

        return SocialModifiers(
            power_differential=honorific_level * 0.5,  # Honorifics imply power awareness
            social_distance=honorific_level
        )


# =============================================================================
# Stub Extractors for Other Languages
# =============================================================================

class TurkishExtractor(LanguageExtractor):
    """
    Turkish extractor stub - demonstrates grammaticalized evidentiality.

    Turkish has grammaticalized evidentiality:
    - -di (past): direct experience
    - -mis (past): hearsay/inference

    Example:
    - "Geldi" = He came (I saw it)
    - "Gelmis" = He came (I heard, inferred)

    TODO: Full implementation requires:
    1. Turkish morphological analyzer
    2. Evidential suffix detection
    3. Agglutinative morphology handling
    """

    def __init__(self):
        super().__init__()
        logger.warning("TurkishExtractor is a stub implementation. "
                      "Full Turkish support requires additional development.")

    def _load_nlp(self):
        """Load Turkish NLP - stub for now."""
        class StubNLP:
            def __call__(self, text):
                return StubDoc(text)

        class StubDoc:
            def __init__(self, text):
                self.text = text
                self.tokens = text.split()
            def __iter__(self):
                return iter([StubToken(t) for t in self.tokens])

        class StubToken:
            def __init__(self, text):
                self.text = text
                self.lemma_ = text.lower()

        self._nlp = StubNLP()

    def get_language_code(self) -> str:
        return "tr"

    def _detect_evidentiality_suffix(self, text: str) -> float:
        """
        Detect Turkish evidentiality from verb suffixes.

        -di: direct evidence (high certainty)
        -mis: indirect evidence (lower certainty)
        """
        # Count evidential markers
        direct_count = 0
        indirect_count = 0

        # Very simplified pattern matching
        # Real implementation needs full morphological analysis
        words = text.split()
        for word in words:
            # Direct past (-di, -ti, -du, -tu with vowel harmony)
            if re.search(r'(di|ti|du|tu|di|ti)$', word.lower()):
                direct_count += 1
            # Indirect past (-mis, -mıs, -mus, -mus with vowel harmony)
            if re.search(r'(mis|mış|muş|müş)$', word.lower()):
                indirect_count += 1

        if direct_count == 0 and indirect_count == 0:
            return 0.0

        # More direct evidence = higher certainty
        total = direct_count + indirect_count
        return (direct_count - indirect_count) / total

    def extract_agency(self, doc: Any) -> AgencyDecomposition:
        """Stub agency extraction for Turkish."""
        # TODO: Implement Turkish-specific agency extraction
        return AgencyDecomposition()

    def extract_justice(self, doc: Any) -> JusticeDecomposition:
        """Stub justice extraction for Turkish."""
        return JusticeDecomposition()

    def extract_belonging(self, doc: Any) -> BelongingDecomposition:
        """Stub belonging extraction for Turkish."""
        return BelongingDecomposition()

    def _extract_epistemic(self, doc: Any) -> EpistemicModifiers:
        """Extract epistemic modifiers using Turkish evidentiality."""
        if doc is None:
            return EpistemicModifiers()

        text = doc.text if hasattr(doc, 'text') else str(doc)
        evidentiality = self._detect_evidentiality_suffix(text)

        return EpistemicModifiers(
            certainty=evidentiality,
            evidentiality=evidentiality,
            commitment=0.0
        )


class KoreanExtractor(LanguageExtractor):
    """
    Korean extractor stub - demonstrates verb-based honorific system.

    Korean has grammaticalized honorifics in verb endings:
    - 해요체 (haeyo-che): polite
    - 합니다체 (hapnida-che): formal
    - 해체 (hae-che): casual
    - 하십시오체 (hasipsio-che): very formal

    TODO: Full implementation requires:
    1. Korean morphological analyzer (KoNLPy, etc.)
    2. Honorific verb ending detection
    3. Subject/object honorific markers
    """

    def __init__(self):
        super().__init__()
        logger.warning("KoreanExtractor is a stub implementation.")

    def _load_nlp(self):
        """Load Korean NLP - stub."""
        self._nlp = lambda text: type('Doc', (), {'text': text, '__iter__': lambda s: iter([])})()

    def get_language_code(self) -> str:
        return "ko"

    def extract_agency(self, doc: Any) -> AgencyDecomposition:
        return AgencyDecomposition()

    def extract_justice(self, doc: Any) -> JusticeDecomposition:
        return JusticeDecomposition()

    def extract_belonging(self, doc: Any) -> BelongingDecomposition:
        return BelongingDecomposition()


# =============================================================================
# Multilingual Coordinate Extractor (Unified Interface)
# =============================================================================

class MultilingualCoordinateExtractor:
    """
    Unified interface for multilingual coordinate extraction.

    Features:
    1. Language detection (auto or specified)
    2. Appropriate extractor selection
    3. Parallel text extraction for cross-linguistic comparison
    4. Invariance testing across translations
    """

    def __init__(self, load_all: bool = False):
        """
        Initialize multilingual extractor.

        Args:
            load_all: If True, pre-load all extractors (slower startup, faster extraction)
                     If False, load extractors on demand (faster startup)
        """
        self._extractors: Dict[str, LanguageExtractor] = {}
        self._extractor_classes = {
            "en": EnglishExtractor,
            "es": SpanishExtractor,
            "ja": JapaneseExtractor,
            "tr": TurkishExtractor,
            "ko": KoreanExtractor,
        }

        if load_all:
            self._load_all_extractors()

        self._language_detector = None

    def _load_all_extractors(self):
        """Pre-load all extractors."""
        for code, extractor_class in self._extractor_classes.items():
            try:
                self._extractors[code] = extractor_class()
                logger.info(f"Loaded extractor for {code}")
            except Exception as e:
                logger.warning(f"Failed to load extractor for {code}: {e}")

    def _get_extractor(self, lang: str) -> LanguageExtractor:
        """Get or create extractor for a language."""
        if lang not in self._extractors:
            if lang in self._extractor_classes:
                self._extractors[lang] = self._extractor_classes[lang]()
            else:
                logger.warning(f"No extractor for language '{lang}'. Falling back to English.")
                if "en" not in self._extractors:
                    self._extractors["en"] = EnglishExtractor()
                return self._extractors["en"]
        return self._extractors[lang]

    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.

        Uses simple heuristics. For production, integrate langdetect or similar.
        """
        # Try to use langdetect if available
        try:
            from langdetect import detect
            detected = detect(text)
            return detected
        except ImportError:
            pass
        except Exception:
            pass

        # Simple heuristic fallback
        # Japanese: contains hiragana/katakana/kanji
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text):
            return "ja"

        # Korean: contains Hangul
        if re.search(r'[\uac00-\ud7af\u1100-\u11ff]', text):
            return "ko"

        # Arabic: contains Arabic script
        if re.search(r'[\u0600-\u06ff]', text):
            return "ar"

        # Chinese: contains Chinese characters without Japanese kana
        if re.search(r'[\u4e00-\u9fff]', text) and not re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return "zh"

        # Spanish heuristics
        spanish_markers = {"que", "de", "en", "la", "el", "es", "por", "con", "una", "para"}
        words = set(text.lower().split())
        if len(words & spanish_markers) >= 2:
            # Check for Spanish-specific characters
            if re.search(r'[áéíóúñ¿¡]', text):
                return "es"

        # Turkish: check for Turkish-specific characters
        if re.search(r'[ıİğĞşŞçÇöÖüÜ]', text):
            return "tr"

        # Default to English
        return "en"

    def extract(
        self,
        text: str,
        lang: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract coordinate from text.

        Args:
            text: Input text
            lang: Language code (ISO 639-1). Auto-detected if not specified.

        Returns:
            ExtractionResult with coordinate and evidence
        """
        if lang is None:
            lang = self.detect_language(text)
            logger.debug(f"Detected language: {lang}")

        extractor = self._get_extractor(lang)
        return extractor.extract_full(text)

    def extract_parallel(
        self,
        texts: Dict[str, str]
    ) -> Dict[str, ExtractionResult]:
        """
        Extract coordinates from parallel texts (same content in different languages).

        This is useful for:
        1. Validating cross-linguistic consistency
        2. Identifying language-specific vs. universal dimensions
        3. Testing translation quality

        Args:
            texts: Dictionary mapping language codes to text content
                  e.g., {"en": "I decided...", "es": "Decidi...", "ja": "私は決めた..."}

        Returns:
            Dictionary mapping language codes to extraction results
        """
        results = {}
        for lang, text in texts.items():
            try:
                results[lang] = self.extract(text, lang=lang)
            except Exception as e:
                logger.error(f"Extraction failed for {lang}: {e}")
                results[lang] = ExtractionResult(
                    coordinate=HierarchicalCoordinate(),
                    language=lang,
                    warnings=[f"Extraction failed: {str(e)}"]
                )
        return results

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return list(self._extractor_classes.keys())

    def get_language_profile(self, lang: str) -> Optional[LanguageProfile]:
        """Get the linguistic profile for a language."""
        return LANGUAGE_PROFILES.get(lang)


# =============================================================================
# Cross-Linguistic Invariance Testing
# =============================================================================

@dataclass
class DimensionInvariance:
    """Invariance analysis for a single dimension."""
    dimension: str                      # agency, justice, belonging
    mean_value: float                   # Mean across languages
    std_value: float                    # Standard deviation
    max_deviation: float                # Maximum deviation from mean
    is_invariant: bool                  # Below invariance threshold?
    language_values: Dict[str, float]   # Value per language
    deviating_languages: List[str]      # Languages that deviate significantly

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "mean_value": self.mean_value,
            "std_value": self.std_value,
            "max_deviation": self.max_deviation,
            "is_invariant": self.is_invariant,
            "language_values": self.language_values,
            "deviating_languages": self.deviating_languages
        }


@dataclass
class InvarianceReport:
    """Complete invariance analysis report."""
    n_texts: int
    languages: List[str]

    # Per-dimension analysis
    agency_invariance: DimensionInvariance
    justice_invariance: DimensionInvariance
    belonging_invariance: DimensionInvariance

    # Subdimension analysis
    subdimension_invariances: Dict[str, DimensionInvariance]

    # Overall statistics
    overall_invariance_score: float     # 0-1, higher = more invariant
    most_invariant_dimension: str
    least_invariant_dimension: str

    # Recommendations
    warnings: List[str]
    recommendations: List[str]

    def to_dict(self) -> dict:
        return {
            "n_texts": self.n_texts,
            "languages": self.languages,
            "agency_invariance": self.agency_invariance.to_dict(),
            "justice_invariance": self.justice_invariance.to_dict(),
            "belonging_invariance": self.belonging_invariance.to_dict(),
            "subdimension_invariances": {
                k: v.to_dict() for k, v in self.subdimension_invariances.items()
            },
            "overall_invariance_score": self.overall_invariance_score,
            "most_invariant_dimension": self.most_invariant_dimension,
            "least_invariant_dimension": self.least_invariant_dimension,
            "warnings": self.warnings,
            "recommendations": self.recommendations
        }


def test_cross_linguistic_invariance(
    parallel_corpus: Dict[str, List[str]],
    extractor: Optional[MultilingualCoordinateExtractor] = None,
    invariance_threshold: float = 0.3
) -> InvarianceReport:
    """
    Test whether coordinate extraction is invariant across translations.

    This is the key test for the Cultural Soliton Observatory's claim
    that the coordination substrate is universal.

    Args:
        parallel_corpus: Dictionary mapping language codes to lists of texts.
                        Each list should contain the SAME content translated.
                        e.g., parallel_corpus["en"][0] is a translation of
                              parallel_corpus["es"][0]
        extractor: MultilingualCoordinateExtractor (created if not provided)
        invariance_threshold: Maximum allowed deviation for invariance

    Returns:
        InvarianceReport with detailed analysis
    """
    if extractor is None:
        extractor = MultilingualCoordinateExtractor()

    languages = list(parallel_corpus.keys())
    n_texts = len(next(iter(parallel_corpus.values())))

    # Validate parallel corpus
    for lang, texts in parallel_corpus.items():
        if len(texts) != n_texts:
            raise ValueError(f"Parallel corpus mismatch: {lang} has {len(texts)} texts, expected {n_texts}")

    # Extract coordinates for all texts
    all_results: Dict[str, List[ExtractionResult]] = defaultdict(list)

    for lang in languages:
        for text in parallel_corpus[lang]:
            result = extractor.extract(text, lang=lang)
            all_results[lang].append(result)

    # Compute invariance for each dimension
    def compute_dimension_invariance(
        dim_name: str,
        get_value: callable
    ) -> DimensionInvariance:
        """Compute invariance for a dimension across languages."""
        lang_means = {}

        for lang in languages:
            values = [get_value(r.coordinate) for r in all_results[lang]]
            lang_means[lang] = np.mean(values)

        mean_value = np.mean(list(lang_means.values()))
        std_value = np.std(list(lang_means.values()))
        max_deviation = max(abs(v - mean_value) for v in lang_means.values())

        deviating = [
            lang for lang, val in lang_means.items()
            if abs(val - mean_value) > invariance_threshold
        ]

        return DimensionInvariance(
            dimension=dim_name,
            mean_value=float(mean_value),
            std_value=float(std_value),
            max_deviation=float(max_deviation),
            is_invariant=max_deviation <= invariance_threshold,
            language_values=lang_means,
            deviating_languages=deviating
        )

    # Compute invariance for main dimensions
    agency_inv = compute_dimension_invariance(
        "agency",
        lambda c: c.core.agency.aggregate
    )
    justice_inv = compute_dimension_invariance(
        "justice",
        lambda c: c.core.justice.aggregate
    )
    belonging_inv = compute_dimension_invariance(
        "belonging",
        lambda c: c.core.belonging.aggregate
    )

    # Compute subdimension invariances
    subdim_invariances = {}

    # Agency subdimensions
    for subdim, getter in [
        ("agency_self", lambda c: c.core.agency.self_agency),
        ("agency_other", lambda c: c.core.agency.other_agency),
        ("agency_system", lambda c: c.core.agency.system_agency),
    ]:
        subdim_invariances[subdim] = compute_dimension_invariance(subdim, getter)

    # Justice subdimensions
    for subdim, getter in [
        ("justice_procedural", lambda c: c.core.justice.procedural),
        ("justice_distributive", lambda c: c.core.justice.distributive),
        ("justice_interactional", lambda c: c.core.justice.interactional),
    ]:
        subdim_invariances[subdim] = compute_dimension_invariance(subdim, getter)

    # Belonging subdimensions
    for subdim, getter in [
        ("belonging_ingroup", lambda c: c.core.belonging.ingroup),
        ("belonging_outgroup", lambda c: c.core.belonging.outgroup),
        ("belonging_universal", lambda c: c.core.belonging.universal),
    ]:
        subdim_invariances[subdim] = compute_dimension_invariance(subdim, getter)

    # Compute overall invariance score
    invariance_scores = [
        1.0 - min(1.0, agency_inv.max_deviation / invariance_threshold),
        1.0 - min(1.0, justice_inv.max_deviation / invariance_threshold),
        1.0 - min(1.0, belonging_inv.max_deviation / invariance_threshold),
    ]
    overall_invariance = np.mean(invariance_scores)

    # Determine most/least invariant
    dim_deviations = {
        "agency": agency_inv.max_deviation,
        "justice": justice_inv.max_deviation,
        "belonging": belonging_inv.max_deviation
    }
    most_invariant = min(dim_deviations, key=dim_deviations.get)
    least_invariant = max(dim_deviations, key=dim_deviations.get)

    # Generate warnings and recommendations
    warnings = []
    recommendations = []

    if not agency_inv.is_invariant:
        warnings.append(f"Agency dimension shows significant variation (std={agency_inv.std_value:.3f})")
        if agency_inv.deviating_languages:
            recommendations.append(
                f"Review agency extraction for: {', '.join(agency_inv.deviating_languages)}"
            )

    if not justice_inv.is_invariant:
        warnings.append(f"Justice dimension shows significant variation (std={justice_inv.std_value:.3f})")

    if not belonging_inv.is_invariant:
        warnings.append(f"Belonging dimension shows significant variation (std={belonging_inv.std_value:.3f})")

    # Check for consistent biases
    for lang in languages:
        profile = LANGUAGE_PROFILES.get(lang)
        if profile:
            if profile.pro_drop == ProDropType.FULL:
                recommendations.append(
                    f"Language {lang} is pro-drop. Verify verb morphology is being used for agency."
                )
            if profile.honorifics == HonorificSystem.MULTI_LEVEL:
                recommendations.append(
                    f"Language {lang} has complex honorifics. Social modifiers may need recalibration."
                )

    if overall_invariance < 0.5:
        warnings.append("Overall invariance is low. Coordinate extraction may not be cross-linguistically valid.")
        recommendations.append("Consider language-specific normalization or recalibration.")

    return InvarianceReport(
        n_texts=n_texts,
        languages=languages,
        agency_invariance=agency_inv,
        justice_invariance=justice_inv,
        belonging_invariance=belonging_inv,
        subdimension_invariances=subdim_invariances,
        overall_invariance_score=float(overall_invariance),
        most_invariant_dimension=most_invariant,
        least_invariant_dimension=least_invariant,
        warnings=warnings,
        recommendations=recommendations
    )


# =============================================================================
# Extractor Registry and Factory
# =============================================================================

class ExtractorRegistry:
    """
    Registry for language extractors.

    Allows registration of custom extractors for additional languages
    or specialized domains.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._extractors = {}
            cls._instance._register_defaults()
        return cls._instance

    def _register_defaults(self):
        """Register default extractors."""
        self._extractors = {
            "en": EnglishExtractor,
            "es": SpanishExtractor,
            "ja": JapaneseExtractor,
            "tr": TurkishExtractor,
            "ko": KoreanExtractor,
        }

    def register(self, lang_code: str, extractor_class: type):
        """
        Register a new extractor class.

        Args:
            lang_code: ISO 639-1 language code
            extractor_class: Class that inherits from LanguageExtractor
        """
        if not issubclass(extractor_class, LanguageExtractor):
            raise TypeError("Extractor must inherit from LanguageExtractor")
        self._extractors[lang_code] = extractor_class
        logger.info(f"Registered extractor for {lang_code}: {extractor_class.__name__}")

    def get(self, lang_code: str) -> Optional[type]:
        """Get extractor class for a language."""
        return self._extractors.get(lang_code)

    def list_languages(self) -> List[str]:
        """List all registered language codes."""
        return list(self._extractors.keys())

    def create(self, lang_code: str) -> LanguageExtractor:
        """Create an extractor instance for a language."""
        extractor_class = self.get(lang_code)
        if extractor_class is None:
            raise ValueError(f"No extractor registered for language: {lang_code}")
        return extractor_class()


def get_extractor_registry() -> ExtractorRegistry:
    """Get the global extractor registry."""
    return ExtractorRegistry()


# =============================================================================
# Convenience Functions
# =============================================================================

def extract_multilingual(
    text: str,
    lang: Optional[str] = None
) -> ExtractionResult:
    """
    Convenience function for single-text extraction.

    Args:
        text: Input text
        lang: Language code (auto-detected if None)

    Returns:
        ExtractionResult
    """
    extractor = MultilingualCoordinateExtractor()
    return extractor.extract(text, lang=lang)


def compare_translations(
    translations: Dict[str, str]
) -> Dict[str, Any]:
    """
    Compare coordinate extraction across translations.

    Args:
        translations: Dictionary mapping language codes to translated text

    Returns:
        Dictionary with extraction results and comparison statistics
    """
    extractor = MultilingualCoordinateExtractor()
    results = extractor.extract_parallel(translations)

    # Compute comparison statistics
    coords = {}
    for lang, result in results.items():
        coords[lang] = {
            "agency": result.coordinate.core.agency.aggregate,
            "justice": result.coordinate.core.justice.aggregate,
            "belonging": result.coordinate.core.belonging.aggregate
        }

    # Compute standard deviations across languages
    agency_values = [c["agency"] for c in coords.values()]
    justice_values = [c["justice"] for c in coords.values()]
    belonging_values = [c["belonging"] for c in coords.values()]

    return {
        "results": {lang: r.to_dict() for lang, r in results.items()},
        "coordinates": coords,
        "statistics": {
            "agency_std": float(np.std(agency_values)),
            "justice_std": float(np.std(justice_values)),
            "belonging_std": float(np.std(belonging_values)),
            "overall_consistency": 1.0 - float(np.mean([
                np.std(agency_values),
                np.std(justice_values),
                np.std(belonging_values)
            ]))
        }
    }


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == "__main__":
    # Example: Cross-linguistic extraction
    print("=" * 60)
    print("Multilingual Coordinate Extraction Demo")
    print("=" * 60)

    # Test texts expressing similar meaning
    test_texts = {
        "en": "I decided to start my own business because I believe in taking control of my destiny.",
        "es": "Decidi empezar mi propio negocio porque creo en tomar control de mi destino.",
        "ja": "自分の運命をコントロールすることを信じているから、自分のビジネスを始めることを決めた。"
    }

    extractor = MultilingualCoordinateExtractor()

    print("\nExtracting coordinates from parallel texts:")
    print("-" * 40)

    results = extractor.extract_parallel(test_texts)

    for lang, result in results.items():
        print(f"\n{lang.upper()} ({LANGUAGE_PROFILES.get(lang, LanguageProfile(lang, 'Unknown')).name}):")
        print(f"  Agency:    {result.coordinate.core.agency.aggregate:.3f}")
        print(f"  Justice:   {result.coordinate.core.justice.aggregate:.3f}")
        print(f"  Belonging: {result.coordinate.core.belonging.aggregate:.3f}")
        if result.warnings:
            print(f"  Warnings: {result.warnings}")

    # Test invariance
    print("\n" + "=" * 60)
    print("Cross-Linguistic Invariance Test")
    print("=" * 60)

    parallel_corpus = {
        "en": [
            "I decided to start my own business because I believe in taking control of my destiny.",
            "We must work together to build a better community for everyone.",
            "The system is unfair, but I refuse to give up."
        ],
        "es": [
            "Decidi empezar mi propio negocio porque creo en tomar control de mi destino.",
            "Debemos trabajar juntos para construir una mejor comunidad para todos.",
            "El sistema es injusto, pero me niego a rendirme."
        ],
        "ja": [
            "自分の運命をコントロールすることを信じているから、自分のビジネスを始めることを決めた。",
            "皆のためにより良いコミュニティを作るために、一緒に働かなければならない。",
            "システムは不公平だが、私は諦めることを拒否する。"
        ]
    }

    report = test_cross_linguistic_invariance(parallel_corpus, extractor)

    print(f"\nOverall Invariance Score: {report.overall_invariance_score:.3f}")
    print(f"Most Invariant Dimension: {report.most_invariant_dimension}")
    print(f"Least Invariant Dimension: {report.least_invariant_dimension}")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  - {w}")

    if report.recommendations:
        print("\nRecommendations:")
        for r in report.recommendations:
            print(f"  - {r}")
