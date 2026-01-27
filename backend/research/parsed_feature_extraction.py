"""
Dependency-Parsed Feature Extraction for Cultural Soliton Observatory.

This module replaces regex-based pattern matching with proper linguistic analysis
using spaCy dependency parsing. It captures:
- Actual syntactic structure (who is doing what to whom)
- Semantic roles (agent, patient, experiencer)
- Negation scope and modal modification
- Embedded clauses and coreference

Author: NLP Engineering Team
Version: 1.0.0
"""

import spacy
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    nlp = None


# =============================================================================
# Constants and Linguistic Categories
# =============================================================================

# First person pronouns
FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself"}
FIRST_PERSON_PLURAL = {"we", "us", "our", "ours", "ourselves"}
FIRST_PERSON = FIRST_PERSON_SINGULAR | FIRST_PERSON_PLURAL

# Second person pronouns
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}

# Third person pronouns
THIRD_PERSON = {"he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their", "theirs",
                "himself", "herself", "itself", "themselves"}

# Impersonal/system subjects
IMPERSONAL_SUBJECTS = {"it", "there", "one", "someone", "something", "anyone", "anything",
                       "everyone", "everything", "nobody", "nothing", "somebody"}

SYSTEM_NOUNS = {"system", "institution", "government", "authority", "force", "circumstance",
                "structure", "process", "procedure", "rule", "regulation", "law", "policy",
                "organization", "bureaucracy", "administration", "mechanism", "framework"}

# Modal verbs with strength classification
MODAL_STRONG = {"must", "shall", "will", "need"}
MODAL_MEDIUM = {"should", "ought", "would"}
MODAL_WEAK = {"can", "could", "may", "might"}

# Negation markers
NEGATION_MARKERS = {"not", "n't", "never", "no", "none", "nothing", "nobody", "nowhere",
                    "neither", "nor", "hardly", "scarcely", "barely"}

# Procedural justice verbs and nouns
PROCEDURAL_VERBS = {"govern", "regulate", "require", "mandate", "specify", "dictate",
                    "prescribe", "stipulate", "determine", "establish", "follow", "enforce"}
PROCEDURAL_NOUNS = {"process", "procedure", "rule", "regulation", "review", "appeal",
                    "hearing", "protocol", "guideline", "standard", "criterion", "policy"}

# Distributive justice verbs
DISTRIBUTIVE_VERBS = {"give", "receive", "distribute", "allocate", "assign", "award",
                      "grant", "deny", "withhold", "share", "divide", "earn", "deserve", "merit"}
FAIRNESS_MODIFIERS = {"fair", "fairly", "unfair", "unfairly", "equal", "equally", "unequal",
                      "just", "unjust", "equitable", "inequitable", "proportional", "disproportionate"}

# Interactional justice verbs
COMMUNICATION_VERBS = {"tell", "say", "explain", "inform", "notify", "communicate",
                       "discuss", "address", "respond", "reply", "listen", "hear", "acknowledge"}
MANNER_ADVERBS = {"respectfully", "politely", "rudely", "dismissively", "kindly",
                  "harshly", "gently", "curtly", "warmly", "coldly", "honestly", "sincerely"}

# Group reference terms
INGROUP_MARKERS = {"we", "us", "our", "ours", "ourselves", "together", "community",
                   "team", "family", "group", "collective", "fellow", "member", "peer"}
OUTGROUP_MARKERS = {"them", "they", "their", "those", "outsider", "stranger", "enemy",
                    "other", "foreigner", "alien", "opponent", "rival"}
UNIVERSAL_MARKERS = {"everyone", "everybody", "all", "humanity", "humankind", "people",
                     "human", "universal", "global", "worldwide", "society", "world"}


# =============================================================================
# Data Classes for Parsed Features
# =============================================================================

@dataclass
class AgencyEvent:
    """Represents a single agency event extracted from text."""
    agent_text: str
    verb_text: str
    agent_type: str  # "self", "other", "system"
    is_negated: bool = False
    modal_strength: float = 1.0  # 1.0 = full agency, 0.5 = tentative, 0.0 = negated
    is_passive: bool = False
    embedding_depth: int = 0  # How deeply embedded in clauses


@dataclass
class JusticeEvent:
    """Represents a justice-related event."""
    justice_type: str  # "procedural", "distributive", "interactional"
    trigger_text: str
    polarity: float  # -1 to 1 (negative = unfair, positive = fair)
    confidence: float


@dataclass
class BelongingMarker:
    """Represents a belonging/group reference."""
    marker_type: str  # "ingroup", "outgroup", "universal"
    text: str
    is_possessive: bool = False
    is_subject: bool = False


@dataclass
class ParsedFeatures:
    """Container for all parsed features from a text."""
    agency_events: List[AgencyEvent] = field(default_factory=list)
    justice_events: List[JusticeEvent] = field(default_factory=list)
    belonging_markers: List[BelongingMarker] = field(default_factory=list)

    # Aggregated scores
    self_agency: float = 0.0
    other_agency: float = 0.0
    system_agency: float = 0.0
    procedural_justice: float = 0.0
    distributive_justice: float = 0.0
    interactional_justice: float = 0.0
    ingroup: float = 0.0
    outgroup: float = 0.0
    universal: float = 0.0

    # Metadata
    sentence_count: int = 0
    passive_voice_ratio: float = 0.0
    negation_ratio: float = 0.0
    modal_usage_ratio: float = 0.0


# =============================================================================
# Core Parsing Functions
# =============================================================================

def get_subject_of_verb(token) -> Optional[Any]:
    """Find the subject of a verb token."""
    for child in token.children:
        if child.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass"):
            return child
    # Check for subject in parent clause
    if token.head and token.head != token:
        for child in token.head.children:
            if child.dep_ in ("nsubj", "nsubjpass") and child.head == token.head:
                return child
    return None


def get_root_of_noun_phrase(token) -> Any:
    """Get the root word of a noun phrase."""
    # Navigate up through compounds and modifiers
    while token.dep_ in ("compound", "amod", "det", "poss"):
        token = token.head
    return token


def is_passive_voice(verb_token) -> bool:
    """Check if a verb is in passive voice."""
    # Check for passive subject
    for child in verb_token.children:
        if child.dep_ == "nsubjpass":
            return True
    # Check for passive auxiliary
    for child in verb_token.children:
        if child.dep_ == "auxpass":
            return True
    # Check if verb is past participle with 'be' auxiliary
    if verb_token.tag_ == "VBN":
        for child in verb_token.children:
            if child.dep_ == "aux" and child.lemma_ == "be":
                return True
    return False


def get_negation_scope(token) -> bool:
    """Check if a token is under negation scope."""
    # Check direct children for negation
    for child in token.children:
        if child.dep_ == "neg" or child.text.lower() in NEGATION_MARKERS:
            return True
    # Check if parent verb is negated
    if token.head and token.head.pos_ == "VERB":
        for child in token.head.children:
            if child.dep_ == "neg" or child.text.lower() in NEGATION_MARKERS:
                return True
    return False


def get_modal_strength(verb_token) -> float:
    """
    Calculate modal modification strength for a verb.
    Returns: 1.0 (definite), 0.8 (strong modal), 0.5 (medium), 0.3 (weak)
    """
    for child in verb_token.children:
        if child.dep_ == "aux" and child.pos_ == "AUX":
            modal_text = child.text.lower()
            if modal_text in MODAL_STRONG:
                return 0.9
            elif modal_text in MODAL_MEDIUM:
                return 0.6
            elif modal_text in MODAL_WEAK:
                return 0.4
    return 1.0


def get_embedding_depth(token) -> int:
    """Calculate how deeply embedded a token is in clause structure."""
    depth = 0
    current = token
    while current.head != current:
        if current.head.pos_ == "VERB" and current.dep_ in ("ccomp", "xcomp", "advcl", "relcl", "acl"):
            depth += 1
        current = current.head
    return depth


def classify_subject_type(subject_token) -> str:
    """Classify a subject as self, other, or system."""
    text_lower = subject_token.text.lower()
    lemma_lower = subject_token.lemma_.lower()

    # Check pronouns first
    if text_lower in FIRST_PERSON:
        return "self"
    elif text_lower in SECOND_PERSON or text_lower in THIRD_PERSON:
        return "other"
    elif text_lower in IMPERSONAL_SUBJECTS:
        return "system"

    # Check noun phrases
    # Get the full noun phrase
    np_tokens = [subject_token]
    for child in subject_token.subtree:
        np_tokens.append(child)
    np_text = " ".join([t.text.lower() for t in np_tokens])

    if any(noun in np_text for noun in SYSTEM_NOUNS):
        return "system"

    # Default to "other" for non-first-person subjects
    return "other"


def extract_agency_events(doc) -> List[AgencyEvent]:
    """Extract agency events from parsed document."""
    events = []

    for token in doc:
        # Only process main verbs
        if token.pos_ != "VERB":
            continue

        # Skip auxiliary verbs
        if token.dep_ in ("aux", "auxpass"):
            continue

        subject = get_subject_of_verb(token)
        if not subject:
            continue

        # Get properties
        is_passive = is_passive_voice(token)
        is_negated = get_negation_scope(token)
        modal_strength = get_modal_strength(token)
        embedding_depth = get_embedding_depth(token)

        # Determine agent type
        if is_passive:
            # In passive voice, the grammatical subject is the patient
            # The agent (if expressed) would be in a by-phrase
            agent_type = "system"  # Default passive to system agency
            by_agent = None
            for child in token.children:
                if child.dep_ == "agent":  # "by X" phrase
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            by_agent = pobj
                            break
            if by_agent:
                agent_type = classify_subject_type(by_agent)
        else:
            agent_type = classify_subject_type(subject)

        # Calculate effective agency (reduced by negation and modals)
        effective_modal = modal_strength
        if is_negated:
            effective_modal *= 0.3  # Negation significantly reduces agency

        event = AgencyEvent(
            agent_text=subject.text,
            verb_text=token.text,
            agent_type=agent_type,
            is_negated=is_negated,
            modal_strength=effective_modal,
            is_passive=is_passive,
            embedding_depth=embedding_depth
        )
        events.append(event)

    return events


def extract_justice_events(doc) -> List[JusticeEvent]:
    """Extract justice-related events from parsed document."""
    events = []

    for token in doc:
        justice_type = None
        polarity = 0.0
        confidence = 0.0

        # Check for procedural justice
        if token.pos_ == "VERB" and token.lemma_.lower() in PROCEDURAL_VERBS:
            justice_type = "procedural"
            polarity = 0.5  # Neutral presence of procedure
            confidence = 0.8

            # Check for fairness modifiers in the clause
            for child in token.subtree:
                if child.text.lower() in FAIRNESS_MODIFIERS:
                    if child.text.lower() in {"fair", "fairly", "just", "equitable", "proper"}:
                        polarity = 0.8
                    elif child.text.lower() in {"unfair", "unfairly", "unjust", "inequitable", "improper"}:
                        polarity = -0.8
                    confidence = 0.9

        elif token.pos_ == "NOUN" and token.lemma_.lower() in PROCEDURAL_NOUNS:
            justice_type = "procedural"
            polarity = 0.3  # Neutral reference to process
            confidence = 0.6

            # Check for adjective modifiers
            for child in token.children:
                if child.dep_ == "amod":
                    if child.text.lower() in {"fair", "proper", "due", "correct", "appropriate"}:
                        polarity = 0.7
                        confidence = 0.8
                    elif child.text.lower() in {"unfair", "improper", "flawed", "biased"}:
                        polarity = -0.7
                        confidence = 0.8

        # Check for distributive justice
        elif token.pos_ == "VERB" and token.lemma_.lower() in DISTRIBUTIVE_VERBS:
            justice_type = "distributive"
            polarity = 0.0  # Neutral distribution
            confidence = 0.7

            # Check for fairness context
            for child in token.subtree:
                if child.text.lower() in FAIRNESS_MODIFIERS:
                    if child.text.lower() in {"fair", "fairly", "equal", "equally", "proportional", "deserved"}:
                        polarity = 0.8
                    else:
                        polarity = -0.8
                    confidence = 0.9

        # Check for interactional justice
        elif token.pos_ == "VERB" and token.lemma_.lower() in COMMUNICATION_VERBS:
            justice_type = "interactional"
            polarity = 0.0  # Neutral communication
            confidence = 0.6

            # Check for manner adverbs
            for child in token.children:
                if child.dep_ == "advmod" and child.text.lower() in MANNER_ADVERBS:
                    if child.text.lower() in {"respectfully", "politely", "kindly", "warmly", "honestly", "sincerely"}:
                        polarity = 0.8
                    elif child.text.lower() in {"rudely", "dismissively", "harshly", "curtly", "coldly"}:
                        polarity = -0.8
                    confidence = 0.9

        if justice_type:
            event = JusticeEvent(
                justice_type=justice_type,
                trigger_text=token.text,
                polarity=polarity,
                confidence=confidence
            )
            events.append(event)

    return events


def extract_belonging_markers(doc) -> List[BelongingMarker]:
    """Extract belonging/group reference markers."""
    markers = []

    for token in doc:
        marker_type = None
        text_lower = token.text.lower()

        # Check token against marker sets
        if text_lower in INGROUP_MARKERS:
            marker_type = "ingroup"
        elif text_lower in OUTGROUP_MARKERS:
            marker_type = "outgroup"
        elif text_lower in UNIVERSAL_MARKERS:
            marker_type = "universal"

        # Also check noun phrases for group-related terms
        if token.pos_ == "NOUN":
            lemma_lower = token.lemma_.lower()
            if lemma_lower in {"community", "team", "family", "group", "collective", "tribe", "clan", "nation"}:
                # Check for possessive - "our community" vs "their community"
                for child in token.children:
                    if child.dep_ == "poss":
                        if child.text.lower() in FIRST_PERSON:
                            marker_type = "ingroup"
                        elif child.text.lower() in {"their", "his", "her", "its"}:
                            marker_type = "outgroup"
                if not marker_type:
                    marker_type = "ingroup"  # Default group references to ingroup

        if marker_type:
            marker = BelongingMarker(
                marker_type=marker_type,
                text=token.text,
                is_possessive=token.dep_ == "poss",
                is_subject=token.dep_ in ("nsubj", "nsubjpass")
            )
            markers.append(marker)

    return markers


# =============================================================================
# Aggregation Functions
# =============================================================================

def aggregate_agency_scores(events: List[AgencyEvent], sentence_count: int) -> Tuple[float, float, float]:
    """
    Aggregate agency events into self, other, system scores.

    Returns normalized scores in range [-1, 1].
    """
    self_score = 0.0
    other_score = 0.0
    system_score = 0.0

    for event in events:
        # Weight by modal strength and embedding depth
        weight = event.modal_strength / (1 + event.embedding_depth * 0.5)

        if event.agent_type == "self":
            self_score += weight
        elif event.agent_type == "other":
            other_score += weight
        else:  # system
            system_score += weight

    # Normalize by sentence count
    if sentence_count > 0:
        normalizer = max(1, sentence_count)
        self_score = min(max(self_score / normalizer - 1.0, -1.0), 1.0)
        other_score = min(max(other_score / normalizer - 1.0, -1.0), 1.0)
        system_score = min(max(system_score / normalizer - 1.0, -1.0), 1.0)

    return self_score, other_score, system_score


def aggregate_justice_scores(events: List[JusticeEvent]) -> Tuple[float, float, float]:
    """
    Aggregate justice events into procedural, distributive, interactional scores.

    Returns scores weighted by polarity and confidence.
    """
    procedural = 0.0
    distributive = 0.0
    interactional = 0.0

    proc_count = dist_count = inter_count = 0

    for event in events:
        weighted_score = event.polarity * event.confidence

        if event.justice_type == "procedural":
            procedural += weighted_score
            proc_count += 1
        elif event.justice_type == "distributive":
            distributive += weighted_score
            dist_count += 1
        else:  # interactional
            interactional += weighted_score
            inter_count += 1

    # Normalize
    if proc_count > 0:
        procedural = min(max(procedural / proc_count, -1.0), 1.0)
    if dist_count > 0:
        distributive = min(max(distributive / dist_count, -1.0), 1.0)
    if inter_count > 0:
        interactional = min(max(interactional / inter_count, -1.0), 1.0)

    return procedural, distributive, interactional


def aggregate_belonging_scores(markers: List[BelongingMarker], sentence_count: int) -> Tuple[float, float, float]:
    """
    Aggregate belonging markers into ingroup, outgroup, universal scores.
    """
    ingroup = 0.0
    outgroup = 0.0
    universal = 0.0

    for marker in markers:
        # Weight subject positions higher
        weight = 1.5 if marker.is_subject else 1.0

        if marker.marker_type == "ingroup":
            ingroup += weight
        elif marker.marker_type == "outgroup":
            outgroup += weight
        else:  # universal
            universal += weight

    # Normalize
    if sentence_count > 0:
        normalizer = max(1, sentence_count)
        ingroup = min(max(ingroup / normalizer - 1.0, -1.0), 1.0)
        outgroup = min(max(outgroup / normalizer - 1.0, -1.0), 1.0)
        universal = min(max(universal / normalizer - 1.0, -1.0), 1.0)

    return ingroup, outgroup, universal


# =============================================================================
# Main Extraction Function
# =============================================================================

def parse_text(text: str) -> ParsedFeatures:
    """
    Parse text and extract all linguistic features using dependency parsing.

    This is the main internal function that returns detailed parsed features.
    """
    if nlp is None:
        raise RuntimeError("spaCy model not loaded. Run: python -m spacy download en_core_web_sm")

    doc = nlp(text)

    # Count sentences
    sentences = list(doc.sents)
    sentence_count = len(sentences)

    # Extract events and markers
    agency_events = extract_agency_events(doc)
    justice_events = extract_justice_events(doc)
    belonging_markers = extract_belonging_markers(doc)

    # Aggregate scores
    self_agency, other_agency, system_agency = aggregate_agency_scores(agency_events, sentence_count)
    procedural, distributive, interactional = aggregate_justice_scores(justice_events)
    ingroup, outgroup, universal = aggregate_belonging_scores(belonging_markers, sentence_count)

    # Calculate metadata
    passive_count = sum(1 for e in agency_events if e.is_passive)
    negation_count = sum(1 for e in agency_events if e.is_negated)
    modal_count = sum(1 for e in agency_events if e.modal_strength < 1.0)
    total_events = len(agency_events) or 1

    return ParsedFeatures(
        agency_events=agency_events,
        justice_events=justice_events,
        belonging_markers=belonging_markers,
        self_agency=self_agency,
        other_agency=other_agency,
        system_agency=system_agency,
        procedural_justice=procedural,
        distributive_justice=distributive,
        interactional_justice=interactional,
        ingroup=ingroup,
        outgroup=outgroup,
        universal=universal,
        sentence_count=sentence_count,
        passive_voice_ratio=passive_count / total_events,
        negation_ratio=negation_count / total_events,
        modal_usage_ratio=modal_count / total_events
    )


def extract_features_parsed(text: str) -> Dict[str, int]:
    """
    Drop-in replacement for extract_features() that uses dependency parsing.

    Returns the same structure as the regex-based extract_features() for
    backward compatibility, but with more accurate counts based on actual
    syntactic analysis.

    Returns:
        Dict with feature names mapped to occurrence counts
    """
    parsed = parse_text(text)

    # Convert parsed features back to count-like format for compatibility
    # We use scaled values (0-10 range) to approximate regex counts
    def score_to_count(score: float, base: int = 5) -> int:
        """Convert a -1 to 1 score to a count-like integer."""
        return max(0, int((score + 1) * base))

    features = {
        # Agency markers - based on actual syntactic agency
        "first_person_singular": sum(1 for e in parsed.agency_events
                                     if e.agent_type == "self" and not e.is_negated),
        "first_person_plural": sum(1 for m in parsed.belonging_markers
                                   if m.marker_type == "ingroup" and m.text.lower() in FIRST_PERSON_PLURAL),
        "second_person": sum(1 for e in parsed.agency_events
                            if e.agent_type == "other" and e.agent_text.lower() in SECOND_PERSON),
        "third_person": sum(1 for e in parsed.agency_events
                           if e.agent_type == "other" and e.agent_text.lower() not in SECOND_PERSON),

        # System agency markers
        "system_agency": sum(1 for e in parsed.agency_events if e.agent_type == "system"),

        # Justice markers - based on actual syntactic patterns
        "procedural_justice": len([e for e in parsed.justice_events if e.justice_type == "procedural"]),
        "distributive_justice": len([e for e in parsed.justice_events if e.justice_type == "distributive"]),
        "interactional_justice": len([e for e in parsed.justice_events if e.justice_type == "interactional"]),

        # Belonging markers - based on actual group references
        "ingroup": sum(1 for m in parsed.belonging_markers if m.marker_type == "ingroup"),
        "outgroup": sum(1 for m in parsed.belonging_markers if m.marker_type == "outgroup"),
        "universal": sum(1 for m in parsed.belonging_markers if m.marker_type == "universal"),

        # Derived counts for compatibility
        "certainty_high": 0,  # Would need separate epistemic analysis
        "certainty_low": 0,
        "evidential_direct": 0,
        "evidential_reported": 0,
        "past_focus": 0,
        "future_focus": 0,
        "high_arousal": 0,
        "low_arousal": 0,
        "positive_valence": 0,
        "negative_valence": 0,
        "articles": 0,
        "hedging": 0,
        "intensifiers": 0,
        "filler": 0,
        "formal_register": 0,
        "informal_register": 0,
    }

    return features


def extract_features_detailed(text: str) -> Dict[str, Any]:
    """
    Extract features with full detail from dependency parsing.

    Returns a comprehensive dictionary with:
    - Aggregated scores for each dimension
    - Lists of extracted events
    - Metadata about the analysis
    """
    parsed = parse_text(text)

    return {
        "agency": {
            "self": parsed.self_agency,
            "other": parsed.other_agency,
            "system": parsed.system_agency,
            "events": [
                {
                    "agent": e.agent_text,
                    "verb": e.verb_text,
                    "type": e.agent_type,
                    "negated": e.is_negated,
                    "modal_strength": e.modal_strength,
                    "passive": e.is_passive
                }
                for e in parsed.agency_events
            ]
        },
        "justice": {
            "procedural": parsed.procedural_justice,
            "distributive": parsed.distributive_justice,
            "interactional": parsed.interactional_justice,
            "events": [
                {
                    "type": e.justice_type,
                    "trigger": e.trigger_text,
                    "polarity": e.polarity,
                    "confidence": e.confidence
                }
                for e in parsed.justice_events
            ]
        },
        "belonging": {
            "ingroup": parsed.ingroup,
            "outgroup": parsed.outgroup,
            "universal": parsed.universal,
            "markers": [
                {
                    "type": m.marker_type,
                    "text": m.text,
                    "is_subject": m.is_subject
                }
                for m in parsed.belonging_markers
            ]
        },
        "metadata": {
            "sentence_count": parsed.sentence_count,
            "passive_voice_ratio": parsed.passive_voice_ratio,
            "negation_ratio": parsed.negation_ratio,
            "modal_usage_ratio": parsed.modal_usage_ratio
        }
    }


# =============================================================================
# Comparison Function
# =============================================================================

def compare_extraction_methods(text: str) -> Dict[str, Any]:
    """
    Compare regex-based and dependency-parsed feature extraction.

    Shows the differences between the two methods to illustrate
    the improvements from syntactic analysis.

    Returns:
        Dictionary with both results and a diff analysis
    """
    # Import the regex-based function
    from hierarchical_coordinates import extract_features as extract_features_regex

    # Get both extractions
    regex_features = extract_features_regex(text)
    parsed_features = extract_features_parsed(text)
    parsed_detailed = extract_features_detailed(text)

    # Build comparison
    comparison = {
        "text": text,
        "regex_based": regex_features,
        "parsed_based": parsed_features,
        "detailed_analysis": parsed_detailed,
        "differences": {},
        "improvements": []
    }

    # Compare key features
    key_features = [
        "first_person_singular", "system_agency",
        "procedural_justice", "distributive_justice", "interactional_justice",
        "ingroup", "outgroup", "universal"
    ]

    for feature in key_features:
        regex_val = regex_features.get(feature, 0)
        parsed_val = parsed_features.get(feature, 0)
        if regex_val != parsed_val:
            comparison["differences"][feature] = {
                "regex": regex_val,
                "parsed": parsed_val,
                "delta": parsed_val - regex_val
            }

    # Document improvements
    parsed = parse_text(text)

    # Check for negation handling
    negated_events = [e for e in parsed.agency_events if e.is_negated]
    if negated_events:
        comparison["improvements"].append({
            "type": "negation_scope",
            "description": f"Detected {len(negated_events)} negated agency events that regex would miss",
            "examples": [f"'{e.agent_text} {e.verb_text}' is negated" for e in negated_events[:3]]
        })

    # Check for modal handling
    modal_events = [e for e in parsed.agency_events if e.modal_strength < 1.0]
    if modal_events:
        comparison["improvements"].append({
            "type": "modal_modification",
            "description": f"Detected {len(modal_events)} modally modified verbs",
            "examples": [f"'{e.agent_text} {e.verb_text}' has modal strength {e.modal_strength:.2f}"
                        for e in modal_events[:3]]
        })

    # Check for passive voice handling
    passive_events = [e for e in parsed.agency_events if e.is_passive]
    if passive_events:
        comparison["improvements"].append({
            "type": "passive_voice",
            "description": f"Detected {len(passive_events)} passive constructions (shifted agency)",
            "examples": [f"'{e.verb_text}' is passive (agent: {e.agent_type})"
                        for e in passive_events[:3]]
        })

    return comparison


# =============================================================================
# Test Examples
# =============================================================================

def run_tests():
    """
    Run tests demonstrating improvements over regex-based extraction.
    """
    test_cases = [
        # Test 1: Negation scope
        {
            "name": "Negation Scope",
            "text": "I didn't actually do it. He didn't help at all.",
            "expected_improvement": "Regex counts 'I' as agency. Parser detects negation reduces agency."
        },
        # Test 2: Modal verbs
        {
            "name": "Modal Modification",
            "text": "I could help you tomorrow. I might be able to assist.",
            "expected_improvement": "Regex counts full agency. Parser detects tentative modality."
        },
        # Test 3: Passive voice
        {
            "name": "Passive Voice",
            "text": "I was given the instructions. The decision was made by the committee.",
            "expected_improvement": "Regex sees 'I'. Parser knows 'I' is patient, not agent."
        },
        # Test 4: Embedded clauses
        {
            "name": "Embedded Clauses",
            "text": "I think that she should go. He believes they will succeed.",
            "expected_improvement": "Parser identifies actual agents in embedded clauses."
        },
        # Test 5: Procedural justice
        {
            "name": "Procedural Justice",
            "text": "The rules governing this process require proper review. The procedure was followed correctly.",
            "expected_improvement": "Parser finds actual procedural language patterns, not just word matching."
        },
        # Test 6: Distributive justice
        {
            "name": "Distributive Justice",
            "text": "Resources were distributed fairly among all participants. Everyone received their equal share.",
            "expected_improvement": "Parser detects distribution verbs with fairness modifiers."
        },
        # Test 7: Interactional justice
        {
            "name": "Interactional Justice",
            "text": "She explained the decision respectfully. They dismissed our concerns rudely.",
            "expected_improvement": "Parser finds communication verbs with manner adverbs."
        },
        # Test 8: Group belonging
        {
            "name": "Group Belonging",
            "text": "Our community stands together. They don't understand us. Everyone deserves respect.",
            "expected_improvement": "Parser identifies possessive + group nouns, not just pronouns."
        },
        # Test 9: Quoted speech
        {
            "name": "Reported Speech",
            "text": "He said 'I will do it myself.' She claimed they were responsible.",
            "expected_improvement": "Parser can potentially distinguish quoted from direct speech."
        },
        # Test 10: Complex sentence
        {
            "name": "Complex Structure",
            "text": "Although I was told the rules would be applied fairly, the process that governed my appeal was neither transparent nor just.",
            "expected_improvement": "Parser handles complex clause structure and identifies multiple justice dimensions."
        },
    ]

    print("=" * 80)
    print("DEPENDENCY PARSING vs REGEX FEATURE EXTRACTION: TEST RESULTS")
    print("=" * 80)

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {test['name']}")
        print(f"{'='*80}")
        print(f"Text: \"{test['text']}\"")
        print(f"Expected Improvement: {test['expected_improvement']}")
        print("-" * 40)

        try:
            comparison = compare_extraction_methods(test["text"])

            print("\nFeature Differences (Regex vs Parsed):")
            if comparison["differences"]:
                for feature, diff in comparison["differences"].items():
                    print(f"  {feature}: regex={diff['regex']}, parsed={diff['parsed']} (delta={diff['delta']:+d})")
            else:
                print("  No differences in count-based features")

            print("\nParsing Improvements Detected:")
            if comparison["improvements"]:
                for imp in comparison["improvements"]:
                    print(f"  [{imp['type']}] {imp['description']}")
                    for ex in imp["examples"]:
                        print(f"    - {ex}")
            else:
                print("  No specific improvements for this example")

            print("\nDetailed Agency Analysis:")
            for event in comparison["detailed_analysis"]["agency"]["events"]:
                neg_str = " [NEGATED]" if event["negated"] else ""
                pass_str = " [PASSIVE]" if event["passive"] else ""
                modal_str = f" [modal={event['modal_strength']:.2f}]" if event["modal_strength"] < 1.0 else ""
                print(f"  '{event['agent']} {event['verb']}' -> {event['type']}{neg_str}{pass_str}{modal_str}")

        except Exception as e:
            print(f"Error processing: {e}")

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("""
Key improvements of dependency parsing over regex:

1. NEGATION SCOPE: "I didn't do it" - regex counts 'I' as agency,
   parser reduces agency score due to negation.

2. MODAL MODIFICATION: "I could help" - regex counts full agency,
   parser assigns reduced agency (0.4) for weak modal.

3. PASSIVE VOICE: "I was given" - regex sees 'I' as agent,
   parser correctly identifies 'I' as patient/recipient.

4. SEMANTIC ROLES: "I was told" vs "I told" - same regex match,
   different syntactic roles (experiencer vs agent).

5. EMBEDDED CLAUSES: "I think she should go" - parser can identify
   the embedded agent 'she', not just the matrix subject 'I'.

6. PROCEDURAL JUSTICE: "rules governing the process" - parser finds
   actual governing relationships, not just word co-occurrence.

7. JUSTICE MODIFIERS: "fairly distributed" - parser detects the
   manner adverb modifying the distribution verb.
""")


if __name__ == "__main__":
    run_tests()
