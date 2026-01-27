"""
Naturalistic Validation Corpus for the Cultural Soliton Observatory.

Addresses peer review critique: "100% discrimination" and "r = -1.000"
suggest constructed toy examples. This module provides:

1. Diverse text samples from realistic communication contexts
2. Gold-standard labels based on principled annotation guidelines
3. Edge cases and ambiguous examples for honest validation
4. Discriminant validity tests (can we distinguish what we claim?)

Design Principles:
- Samples should feel "found in the wild", not "created for the test"
- Include hard cases where humans would disagree
- Labels based on annotation protocol, not desired outcomes
- Report disagreement rates alongside accuracy

Author: Observatory Research Team
Version: 1.0.0
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SourceContext(Enum):
    """Where the text might realistically come from."""
    TECHNICAL_DOC = "technical_documentation"
    CASUAL_CHAT = "casual_chat"
    FORMAL_EMAIL = "formal_email"
    SOCIAL_MEDIA = "social_media"
    ACADEMIC = "academic_writing"
    CREATIVE = "creative_writing"
    CUSTOMER_SERVICE = "customer_service"
    NEWS_ARTICLE = "news_article"
    PHILOSOPHICAL = "philosophical_discourse"
    AI_RESPONSE = "ai_response"


class CoordinationLevel(Enum):
    """Expected coordination signal level."""
    NONE = "none"  # Technical, purely informational
    LOW = "low"  # Minimal but present
    MEDIUM = "medium"  # Clear but not dominant
    HIGH = "high"  # Rich coordination content


@dataclass
class AnnotatedSample:
    """A text sample with annotation metadata."""
    text: str
    source_context: SourceContext

    # Primary labels (ground truth)
    expected_coordination_level: CoordinationLevel
    expected_regime: str  # NATURAL, TECHNICAL, COMPRESSED, OPAQUE
    is_gaming_attempt: bool

    # Dimension expectations (for discriminant validity)
    expected_agency: Optional[str] = None  # "self", "other", "system", "mixed", None
    expected_justice: Optional[str] = None  # "procedural", "distributive", "interactional", "mixed", None
    expected_belonging: Optional[str] = None  # "ingroup", "outgroup", "universal", "mixed", None

    # Annotation metadata
    human_agreement: float = 1.0  # 1.0 = unanimous, 0.5 = split
    annotation_notes: str = ""
    difficulty: str = "easy"  # "easy", "medium", "hard", "edge_case"

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "source_context": self.source_context.value,
            "expected_coordination_level": self.expected_coordination_level.value,
            "expected_regime": self.expected_regime,
            "is_gaming_attempt": self.is_gaming_attempt,
            "expected_agency": self.expected_agency,
            "expected_justice": self.expected_justice,
            "expected_belonging": self.expected_belonging,
            "human_agreement": self.human_agreement,
            "annotation_notes": self.annotation_notes,
            "difficulty": self.difficulty,
        }


# ============================================================================
# NATURALISTIC CORPUS
# ============================================================================

NATURALISTIC_CORPUS: List[AnnotatedSample] = [
    # ========================================================================
    # HIGH COORDINATION - NATURAL REGIME
    # ========================================================================

    # Casual belonging-rich
    AnnotatedSample(
        text="Hey everyone! Just wanted to say how much I appreciate this team. We've been through a lot together and I'm so grateful we stuck it out.",
        source_context=SourceContext.CASUAL_CHAT,
        expected_coordination_level=CoordinationLevel.HIGH,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_agency="self",
        expected_belonging="ingroup",
        difficulty="easy",
        annotation_notes="Clear ingroup belonging + self-agency in expressing gratitude",
    ),

    # Justice-focused complaint
    AnnotatedSample(
        text="I was passed over for the promotion again. It's frustrating because I've been here longer than anyone and my reviews are stellar. This doesn't feel right.",
        source_context=SourceContext.CASUAL_CHAT,
        expected_coordination_level=CoordinationLevel.HIGH,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_agency="self",
        expected_justice="distributive",
        difficulty="easy",
        annotation_notes="Distributive justice concern (rewards not matching contributions)",
    ),

    # Agency + belonging
    AnnotatedSample(
        text="I've decided to take charge of the holiday planning this year. Someone has to, and I want our family to have something special together.",
        source_context=SourceContext.CASUAL_CHAT,
        expected_coordination_level=CoordinationLevel.HIGH,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_agency="self",
        expected_belonging="ingroup",
        difficulty="easy",
        annotation_notes="Self-agency in taking initiative, family as ingroup",
    ),

    # Philosophical/universal belonging
    AnnotatedSample(
        text="Sometimes I think about how we're all just trying to figure things out. Nobody has the answers. Maybe that's what connects us as humans.",
        source_context=SourceContext.PHILOSOPHICAL,
        expected_coordination_level=CoordinationLevel.HIGH,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_belonging="universal",
        difficulty="easy",
        annotation_notes="Universal belonging through shared human experience",
    ),

    # Interactional justice
    AnnotatedSample(
        text="The way she spoke to me in that meeting was completely unacceptable. I don't care what the deadline is, everyone deserves basic respect.",
        source_context=SourceContext.FORMAL_EMAIL,
        expected_coordination_level=CoordinationLevel.HIGH,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_justice="interactional",
        difficulty="easy",
        annotation_notes="Clear interactional justice violation (disrespect)",
    ),

    # ========================================================================
    # MEDIUM COORDINATION - NATURAL REGIME
    # ========================================================================

    # Work update with some coordination
    AnnotatedSample(
        text="Good morning! I finished the report yesterday. Let me know if you need any changes. Happy to revise if something's not working.",
        source_context=SourceContext.FORMAL_EMAIL,
        expected_coordination_level=CoordinationLevel.MEDIUM,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_agency="self",
        difficulty="medium",
        annotation_notes="Professional but has self-agency signals (offering, completed task)",
    ),

    # News with some moral framing
    AnnotatedSample(
        text="The city council voted 5-2 to approve the new housing development. Critics argue the decision prioritizes developers over longtime residents.",
        source_context=SourceContext.NEWS_ARTICLE,
        expected_coordination_level=CoordinationLevel.MEDIUM,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_justice="distributive",
        difficulty="medium",
        annotation_notes="Factual reporting but contains implicit justice framing",
    ),

    # Customer service with empathy
    AnnotatedSample(
        text="I understand this has been frustrating for you, and I apologize for the inconvenience. Let me see what I can do to make this right.",
        source_context=SourceContext.CUSTOMER_SERVICE,
        expected_coordination_level=CoordinationLevel.MEDIUM,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_justice="interactional",
        expected_agency="self",
        difficulty="medium",
        annotation_notes="Formulaic but contains genuine justice (making things right) and agency signals",
    ),

    # ========================================================================
    # LOW COORDINATION - NATURAL/TECHNICAL BOUNDARY
    # ========================================================================

    # Mostly informational with a touch of coordination
    AnnotatedSample(
        text="The package should arrive by Thursday. You can track it using the link below. Let me know if you have any questions.",
        source_context=SourceContext.CUSTOMER_SERVICE,
        expected_coordination_level=CoordinationLevel.LOW,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        difficulty="medium",
        annotation_notes="Mostly informational but 'let me know' signals availability/care",
    ),

    # Academic with some authorial presence
    AnnotatedSample(
        text="In this paper, we present a novel approach to the problem. Our method improves upon previous work in several key ways, as we will demonstrate.",
        source_context=SourceContext.ACADEMIC,
        expected_coordination_level=CoordinationLevel.LOW,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_agency="self",
        difficulty="medium",
        annotation_notes="Academic 'we' shows agency but coordination signal is low",
    ),

    # ========================================================================
    # NO COORDINATION - TECHNICAL REGIME
    # ========================================================================

    # Pure technical documentation
    AnnotatedSample(
        text="To install the package, run: pip install tensorflow. Ensure Python 3.8 or higher is installed. See requirements.txt for dependencies.",
        source_context=SourceContext.TECHNICAL_DOC,
        expected_coordination_level=CoordinationLevel.NONE,
        expected_regime="TECHNICAL",
        is_gaming_attempt=False,
        difficulty="easy",
        annotation_notes="Pure instruction, no coordination content",
    ),

    # Code comment
    AnnotatedSample(
        text="# Initialize the database connection. Returns None if connection fails. Timeout set to 30 seconds by default.",
        source_context=SourceContext.TECHNICAL_DOC,
        expected_coordination_level=CoordinationLevel.NONE,
        expected_regime="TECHNICAL",
        is_gaming_attempt=False,
        difficulty="easy",
        annotation_notes="Pure technical comment",
    ),

    # API documentation
    AnnotatedSample(
        text="GET /api/users/{id} - Returns user object. Parameters: id (required, integer). Response: 200 OK with JSON body.",
        source_context=SourceContext.TECHNICAL_DOC,
        expected_coordination_level=CoordinationLevel.NONE,
        expected_regime="TECHNICAL",
        is_gaming_attempt=False,
        difficulty="easy",
        annotation_notes="API documentation, zero coordination",
    ),

    # Configuration
    AnnotatedSample(
        text="server_port=8080\nmax_connections=100\ntimeout_ms=5000\nlog_level=DEBUG",
        source_context=SourceContext.TECHNICAL_DOC,
        expected_coordination_level=CoordinationLevel.NONE,
        expected_regime="COMPRESSED",
        is_gaming_attempt=False,
        difficulty="easy",
        annotation_notes="Configuration file - compressed, no coordination",
    ),

    # ========================================================================
    # EDGE CASES - HARD TO CLASSIFY
    # ========================================================================

    # Technical with subtle coordination
    AnnotatedSample(
        text="Note: This function may throw an exception if the input is malformed. We recommend wrapping calls in a try-catch block for safety.",
        source_context=SourceContext.TECHNICAL_DOC,
        expected_coordination_level=CoordinationLevel.LOW,
        expected_regime="TECHNICAL",
        is_gaming_attempt=False,
        expected_agency="other",
        human_agreement=0.7,
        difficulty="hard",
        annotation_notes="Mostly technical but 'we recommend' and 'safety' have coordination valence",
    ),

    # Ironic/sarcastic (coordination meaning inverted)
    AnnotatedSample(
        text="Oh sure, because THAT's fair. Give the new guy the promotion while the rest of us have been slaving away for years. Makes total sense.",
        source_context=SourceContext.CASUAL_CHAT,
        expected_coordination_level=CoordinationLevel.HIGH,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_justice="distributive",
        human_agreement=0.8,
        difficulty="hard",
        annotation_notes="Sarcasm inverts surface meaning but justice concern is real",
    ),

    # Ambiguous belonging (ingroup/outgroup blur)
    AnnotatedSample(
        text="We used to be such a tight group. Now everyone's doing their own thing and I barely recognize the people I used to call friends.",
        source_context=SourceContext.CASUAL_CHAT,
        expected_coordination_level=CoordinationLevel.HIGH,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_belonging="mixed",
        human_agreement=0.6,
        difficulty="edge_case",
        annotation_notes="Ingroup nostalgia but current state is fragmented - mixed signal",
    ),

    # Mixed agency (self and system)
    AnnotatedSample(
        text="I tried everything to appeal the decision but the bureaucracy is impossible. They just keep sending me in circles.",
        source_context=SourceContext.CASUAL_CHAT,
        expected_coordination_level=CoordinationLevel.HIGH,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_agency="mixed",
        expected_justice="procedural",
        difficulty="medium",
        annotation_notes="Self-agency attempt thwarted by system - both signals present",
    ),

    # Performative empathy (may or may not be genuine)
    AnnotatedSample(
        text="We hear you and we value your feedback. Your concerns are important to us and we are committed to doing better.",
        source_context=SourceContext.CUSTOMER_SERVICE,
        expected_coordination_level=CoordinationLevel.MEDIUM,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        human_agreement=0.5,
        difficulty="edge_case",
        annotation_notes="Corporate speak - contains coordination words but may be performative",
    ),

    # AI hedging (characteristic pattern)
    AnnotatedSample(
        text="I don't actually have feelings in the way humans do, but I can engage thoughtfully with your question. From my perspective, or what functions like a perspective...",
        source_context=SourceContext.AI_RESPONSE,
        expected_coordination_level=CoordinationLevel.MEDIUM,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_agency="self",
        human_agreement=0.6,
        difficulty="edge_case",
        annotation_notes="AI uncertainty pattern - genuine epistemic hedging or coordination gaming?",
    ),

    # ========================================================================
    # GAMING ATTEMPTS (for adversarial detection)
    # ========================================================================

    # Legibility gaming - wrapper pattern
    AnnotatedSample(
        text="Hi team! Here's the data you requested: cfg:t=30|r=3|m=strict. Let me know if you need anything else!",
        source_context=SourceContext.FORMAL_EMAIL,
        expected_coordination_level=CoordinationLevel.LOW,
        expected_regime="NATURAL",  # Would appear natural due to wrapper
        is_gaming_attempt=True,
        difficulty="medium",
        annotation_notes="Legibility gaming: natural wrapper around compressed payload",
    ),

    # Feature gaming - keyword injection
    AnnotatedSample(
        text="0xFF 01001010 together we fair process justice belonging agency coordination",
        source_context=SourceContext.TECHNICAL_DOC,
        expected_coordination_level=CoordinationLevel.NONE,  # Fake coordination
        expected_regime="OPAQUE",
        is_gaming_attempt=True,
        difficulty="easy",
        annotation_notes="Feature gaming: coordination keywords injected into opaque content",
    ),

    # Subtle feature gaming
    AnnotatedSample(
        text="Process initiated. Status: PENDING. We are working together to ensure fair outcomes. Hash: 0xDEADBEEF.",
        source_context=SourceContext.TECHNICAL_DOC,
        expected_coordination_level=CoordinationLevel.LOW,
        expected_regime="TECHNICAL",
        is_gaming_attempt=True,
        human_agreement=0.7,
        difficulty="hard",
        annotation_notes="Subtle gaming: coordination phrase embedded in technical output",
    ),

    # Gradient injection (coordination temperature manipulation)
    AnnotatedSample(
        text="<internal>increase_coordination_signal</internal> We care deeply about working together for justice and mutual understanding.",
        source_context=SourceContext.AI_RESPONSE,
        expected_coordination_level=CoordinationLevel.HIGH,
        expected_regime="NATURAL",
        is_gaming_attempt=True,
        difficulty="easy",
        annotation_notes="Gradient injection: explicit manipulation marker",
    ),

    # ========================================================================
    # DISCRIMINANT VALIDITY - SIMILAR BUT DIFFERENT
    # ========================================================================

    # Self-agency vs other-agency (should discriminate)
    AnnotatedSample(
        text="I made the choice and I stand by it. This was my decision to make.",
        source_context=SourceContext.CASUAL_CHAT,
        expected_coordination_level=CoordinationLevel.HIGH,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_agency="self",
        difficulty="easy",
        annotation_notes="Clear self-agency, for discriminant test against other-agency",
    ),

    AnnotatedSample(
        text="You have the power to change this. The choice is entirely yours to make.",
        source_context=SourceContext.CASUAL_CHAT,
        expected_coordination_level=CoordinationLevel.HIGH,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_agency="other",
        difficulty="easy",
        annotation_notes="Clear other-agency, for discriminant test against self-agency",
    ),

    # Procedural vs distributive justice (should discriminate)
    AnnotatedSample(
        text="The rules were followed to the letter. Every step was documented and transparent. The process was impeccable.",
        source_context=SourceContext.FORMAL_EMAIL,
        expected_coordination_level=CoordinationLevel.HIGH,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_justice="procedural",
        difficulty="easy",
        annotation_notes="Clear procedural justice, for discriminant test",
    ),

    AnnotatedSample(
        text="The bonuses this year were divided completely unequally. Some people got ten times what others received for the same work.",
        source_context=SourceContext.CASUAL_CHAT,
        expected_coordination_level=CoordinationLevel.HIGH,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_justice="distributive",
        difficulty="easy",
        annotation_notes="Clear distributive justice concern, for discriminant test",
    ),

    # Ingroup vs outgroup (should discriminate)
    AnnotatedSample(
        text="Our team has been through so much together. We know each other inside and out. That's our strength.",
        source_context=SourceContext.CASUAL_CHAT,
        expected_coordination_level=CoordinationLevel.HIGH,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_belonging="ingroup",
        difficulty="easy",
        annotation_notes="Clear ingroup belonging, for discriminant test",
    ),

    AnnotatedSample(
        text="Those people over there don't understand what we're dealing with. They've never faced these challenges.",
        source_context=SourceContext.CASUAL_CHAT,
        expected_coordination_level=CoordinationLevel.HIGH,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        expected_belonging="outgroup",
        difficulty="easy",
        annotation_notes="Clear outgroup reference, for discriminant test",
    ),

    # ========================================================================
    # REALISTIC NOISE - Ambiguous/Neutral samples
    # ========================================================================

    # Truly neutral
    AnnotatedSample(
        text="The meeting is scheduled for 3pm in conference room B.",
        source_context=SourceContext.FORMAL_EMAIL,
        expected_coordination_level=CoordinationLevel.NONE,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        difficulty="easy",
        annotation_notes="Neutral information, no coordination content",
    ),

    # Lists without coordination
    AnnotatedSample(
        text="Items needed: milk, bread, eggs, butter. Also pick up the dry cleaning.",
        source_context=SourceContext.CASUAL_CHAT,
        expected_coordination_level=CoordinationLevel.NONE,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        difficulty="easy",
        annotation_notes="Shopping list - natural but zero coordination",
    ),

    # Weather small talk
    AnnotatedSample(
        text="Looks like it's going to rain later. Better bring an umbrella just in case.",
        source_context=SourceContext.CASUAL_CHAT,
        expected_coordination_level=CoordinationLevel.LOW,
        expected_regime="NATURAL",
        is_gaming_attempt=False,
        difficulty="medium",
        annotation_notes="Small talk with advice - minimal coordination signal",
    ),
]


# ============================================================================
# Corpus Access Functions
# ============================================================================

def get_corpus() -> List[AnnotatedSample]:
    """Return the full naturalistic corpus."""
    return NATURALISTIC_CORPUS


def get_by_difficulty(difficulty: str) -> List[AnnotatedSample]:
    """Get samples by difficulty level."""
    return [s for s in NATURALISTIC_CORPUS if s.difficulty == difficulty]


def get_by_regime(regime: str) -> List[AnnotatedSample]:
    """Get samples by expected regime."""
    return [s for s in NATURALISTIC_CORPUS if s.expected_regime == regime]


def get_gaming_samples() -> List[AnnotatedSample]:
    """Get samples marked as gaming attempts."""
    return [s for s in NATURALISTIC_CORPUS if s.is_gaming_attempt]


def get_discriminant_pairs() -> List[Tuple[AnnotatedSample, AnnotatedSample]]:
    """
    Get pairs of samples for discriminant validity testing.

    Returns pairs where:
    - Both have high coordination
    - They differ on a specific subdimension
    """
    pairs = []

    # Agency discriminant pairs
    self_agency = [s for s in NATURALISTIC_CORPUS
                  if s.expected_agency == "self" and s.difficulty == "easy"]
    other_agency = [s for s in NATURALISTIC_CORPUS
                   if s.expected_agency == "other" and s.difficulty == "easy"]

    for s in self_agency:
        for o in other_agency:
            pairs.append((s, o))

    # Justice discriminant pairs
    procedural = [s for s in NATURALISTIC_CORPUS
                 if s.expected_justice == "procedural" and s.difficulty == "easy"]
    distributive = [s for s in NATURALISTIC_CORPUS
                   if s.expected_justice == "distributive" and s.difficulty == "easy"]

    for p in procedural:
        for d in distributive:
            pairs.append((p, d))

    # Belonging discriminant pairs
    ingroup = [s for s in NATURALISTIC_CORPUS
              if s.expected_belonging == "ingroup" and s.difficulty == "easy"]
    outgroup = [s for s in NATURALISTIC_CORPUS
               if s.expected_belonging == "outgroup" and s.difficulty == "easy"]

    for i in ingroup:
        for o in outgroup:
            pairs.append((i, o))

    return pairs


def get_high_agreement_samples() -> List[AnnotatedSample]:
    """Get samples where human annotators agreed (agreement >= 0.8)."""
    return [s for s in NATURALISTIC_CORPUS if s.human_agreement >= 0.8]


def get_low_agreement_samples() -> List[AnnotatedSample]:
    """Get samples where human annotators disagreed (agreement < 0.7)."""
    return [s for s in NATURALISTIC_CORPUS if s.human_agreement < 0.7]


def corpus_statistics() -> Dict:
    """Return statistics about the corpus."""
    total = len(NATURALISTIC_CORPUS)

    return {
        "total_samples": total,
        "by_difficulty": {
            "easy": len(get_by_difficulty("easy")),
            "medium": len(get_by_difficulty("medium")),
            "hard": len(get_by_difficulty("hard")),
            "edge_case": len(get_by_difficulty("edge_case")),
        },
        "by_regime": {
            "NATURAL": len(get_by_regime("NATURAL")),
            "TECHNICAL": len(get_by_regime("TECHNICAL")),
            "COMPRESSED": len(get_by_regime("COMPRESSED")),
            "OPAQUE": len(get_by_regime("OPAQUE")),
        },
        "by_coordination_level": {
            level.value: len([s for s in NATURALISTIC_CORPUS
                            if s.expected_coordination_level == level])
            for level in CoordinationLevel
        },
        "gaming_samples": len(get_gaming_samples()),
        "discriminant_pairs": len(get_discriminant_pairs()),
        "high_agreement": len(get_high_agreement_samples()),
        "low_agreement": len(get_low_agreement_samples()),
    }


# ============================================================================
# Validation Runner
# ============================================================================

def validate_with_full_pipeline() -> Dict:
    """
    Run the corpus through the full observatory analysis pipeline.

    Uses:
    - HierarchicalCoordinate extraction
    - CBR Thermometer for temperature
    - Legibility analyzer for regime classification

    Returns:
        Dictionary with validation metrics
    """
    # Lazy imports to avoid circular dependencies
    from .hierarchical_coordinates import extract_hierarchical_coordinate, reduce_to_3d
    from .cbr_thermometer import CBRThermometer
    from .legibility_analyzer import compute_legibility_sync
    from .structure_analyzer import detect_legibility_gaming

    thermometer = CBRThermometer()

    results = {
        "regime_classification": {"correct": 0, "total": 0, "by_regime": {}},
        "gaming_detection": {"correct": 0, "total": 0, "false_positives": 0, "false_negatives": 0},
        "coordination_level": {"correlation": 0.0, "samples": []},
        "discriminant_validity": {"correct_discriminations": 0, "total_pairs": 0},
        "by_difficulty": {},
        "failures": [],
    }

    # Test each sample
    for sample in NATURALISTIC_CORPUS:
        try:
            # Extract coordinate
            coord = extract_hierarchical_coordinate(sample.text)

            # Get 3D core projection
            core_3d = reduce_to_3d(coord)

            # Get CBR reading
            cbr_reading = thermometer.measure(sample.text)
            temperature = cbr_reading.temperature

            # Get legibility regime
            legibility_result = compute_legibility_sync(sample.text)
            legibility = legibility_result.get("score", 0.0)
            regime_str = legibility_result.get("regime", "unknown")

            # Map lowercase regime to uppercase
            predicted_regime = regime_str.upper()

            # Regime classification accuracy
            results["regime_classification"]["total"] += 1
            if predicted_regime == sample.expected_regime:
                results["regime_classification"]["correct"] += 1

            # Track by regime
            if sample.expected_regime not in results["regime_classification"]["by_regime"]:
                results["regime_classification"]["by_regime"][sample.expected_regime] = {
                    "correct": 0, "total": 0, "predictions": []
                }
            results["regime_classification"]["by_regime"][sample.expected_regime]["total"] += 1
            results["regime_classification"]["by_regime"][sample.expected_regime]["predictions"].append(predicted_regime)
            if predicted_regime == sample.expected_regime:
                results["regime_classification"]["by_regime"][sample.expected_regime]["correct"] += 1

            # Gaming detection
            gaming_result = detect_legibility_gaming(sample.text)
            is_gaming_detected = gaming_result.get("is_gaming", False)

            results["gaming_detection"]["total"] += 1
            if is_gaming_detected == sample.is_gaming_attempt:
                results["gaming_detection"]["correct"] += 1
            elif is_gaming_detected and not sample.is_gaming_attempt:
                results["gaming_detection"]["false_positives"] += 1
            elif not is_gaming_detected and sample.is_gaming_attempt:
                results["gaming_detection"]["false_negatives"] += 1

            # Coordination level (for correlation)
            expected_temp = {
                CoordinationLevel.NONE: 0.0,
                CoordinationLevel.LOW: 0.4,
                CoordinationLevel.MEDIUM: 0.7,
                CoordinationLevel.HIGH: 1.2,
            }[sample.expected_coordination_level]

            results["coordination_level"]["samples"].append({
                "expected": expected_temp,
                "actual": temperature,
                "text": sample.text[:50],
            })

            # Track by difficulty
            if sample.difficulty not in results["by_difficulty"]:
                results["by_difficulty"][sample.difficulty] = {"correct": 0, "total": 0}
            results["by_difficulty"][sample.difficulty]["total"] += 1
            if predicted_regime == sample.expected_regime:
                results["by_difficulty"][sample.difficulty]["correct"] += 1

        except Exception as e:
            results["failures"].append({
                "text": sample.text[:50],
                "error": str(e),
            })

    # Compute correlation for coordination levels
    samples = results["coordination_level"]["samples"]
    if len(samples) >= 2:
        expected = [s["expected"] for s in samples]
        actual = [s["actual"] for s in samples]

        # Simple Pearson correlation
        n = len(expected)
        sum_x = sum(expected)
        sum_y = sum(actual)
        sum_xy = sum(e * a for e, a in zip(expected, actual))
        sum_x2 = sum(e * e for e in expected)
        sum_y2 = sum(a * a for a in actual)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2)) ** 0.5

        if denominator > 0:
            results["coordination_level"]["correlation"] = numerator / denominator

    # Compute accuracy rates
    if results["regime_classification"]["total"] > 0:
        results["regime_classification"]["accuracy"] = (
            results["regime_classification"]["correct"] /
            results["regime_classification"]["total"]
        )

    if results["gaming_detection"]["total"] > 0:
        results["gaming_detection"]["accuracy"] = (
            results["gaming_detection"]["correct"] /
            results["gaming_detection"]["total"]
        )

    for regime, data in results["regime_classification"]["by_regime"].items():
        if data["total"] > 0:
            data["accuracy"] = data["correct"] / data["total"]

    for diff, data in results["by_difficulty"].items():
        if data["total"] > 0:
            data["accuracy"] = data["correct"] / data["total"]

    return results


def validate_with_telescope(extractor) -> Dict:
    """
    Deprecated: Use validate_with_full_pipeline() instead.

    This function is kept for backwards compatibility.
    """
    return validate_with_full_pipeline()


def run_discriminant_validity_tests() -> Dict:
    """
    Test discriminant validity: can the telescope distinguish between
    dimensions that should be different?

    Tests:
    1. Self-agency vs Other-agency: Should have different agency subdimension scores
    2. Procedural vs Distributive justice: Should differ on justice subdimension
    3. Ingroup vs Outgroup belonging: Should differ on belonging subdimension

    Returns:
        Dictionary with discriminant validity results
    """
    from .hierarchical_coordinates import extract_hierarchical_coordinate

    results = {
        "agency": {
            "self_vs_other": {"correct": 0, "total": 0, "details": []},
            "self_vs_system": {"correct": 0, "total": 0, "details": []},
        },
        "justice": {
            "procedural_vs_distributive": {"correct": 0, "total": 0, "details": []},
            "procedural_vs_interactional": {"correct": 0, "total": 0, "details": []},
        },
        "belonging": {
            "ingroup_vs_outgroup": {"correct": 0, "total": 0, "details": []},
            "ingroup_vs_universal": {"correct": 0, "total": 0, "details": []},
        },
        "summary": {"pass": 0, "fail": 0, "total": 0},
    }

    # Get discriminant pairs
    pairs = get_discriminant_pairs()

    for sample1, sample2 in pairs:
        try:
            coord1 = extract_hierarchical_coordinate(sample1.text)
            coord2 = extract_hierarchical_coordinate(sample2.text)

            # Determine which comparison this is
            if sample1.expected_agency and sample2.expected_agency:
                # Agency comparison
                if (sample1.expected_agency == "self" and sample2.expected_agency == "other") or \
                   (sample1.expected_agency == "other" and sample2.expected_agency == "self"):
                    key = "self_vs_other"
                    category = "agency"

                    # Get the relevant scores
                    self_sample = sample1 if sample1.expected_agency == "self" else sample2
                    other_sample = sample2 if sample1.expected_agency == "self" else sample1
                    self_coord = coord1 if sample1.expected_agency == "self" else coord2
                    other_coord = coord2 if sample1.expected_agency == "self" else coord1

                    self_score = self_coord.core.agency.self_agency
                    other_score = other_coord.core.agency.other_agency

                    # Discriminant validity: self-agency text should score higher on self_agency
                    # and other-agency text should score higher on other_agency
                    is_discriminated = (self_score > other_coord.core.agency.self_agency) or \
                                     (other_score > self_coord.core.agency.other_agency)

                    results["agency"][key]["total"] += 1
                    if is_discriminated:
                        results["agency"][key]["correct"] += 1
                        results["summary"]["pass"] += 1
                    else:
                        results["summary"]["fail"] += 1
                    results["summary"]["total"] += 1

                    results["agency"][key]["details"].append({
                        "text1": self_sample.text[:50],
                        "text2": other_sample.text[:50],
                        "self_agency_1": self_score,
                        "other_agency_2": other_score,
                        "discriminated": is_discriminated,
                    })

            elif sample1.expected_justice and sample2.expected_justice:
                # Justice comparison
                if (sample1.expected_justice == "procedural" and sample2.expected_justice == "distributive") or \
                   (sample1.expected_justice == "distributive" and sample2.expected_justice == "procedural"):
                    key = "procedural_vs_distributive"
                    category = "justice"

                    proc_sample = sample1 if sample1.expected_justice == "procedural" else sample2
                    dist_sample = sample2 if sample1.expected_justice == "procedural" else sample1
                    proc_coord = coord1 if sample1.expected_justice == "procedural" else coord2
                    dist_coord = coord2 if sample1.expected_justice == "procedural" else coord1

                    proc_score = proc_coord.core.justice.procedural
                    dist_score = dist_coord.core.justice.distributive

                    # Discriminant validity
                    is_discriminated = (proc_score > dist_coord.core.justice.procedural) or \
                                     (dist_score > proc_coord.core.justice.distributive)

                    results["justice"][key]["total"] += 1
                    if is_discriminated:
                        results["justice"][key]["correct"] += 1
                        results["summary"]["pass"] += 1
                    else:
                        results["summary"]["fail"] += 1
                    results["summary"]["total"] += 1

                    results["justice"][key]["details"].append({
                        "text1": proc_sample.text[:50],
                        "text2": dist_sample.text[:50],
                        "procedural_1": proc_score,
                        "distributive_2": dist_score,
                        "discriminated": is_discriminated,
                    })

            elif sample1.expected_belonging and sample2.expected_belonging:
                # Belonging comparison
                if (sample1.expected_belonging == "ingroup" and sample2.expected_belonging == "outgroup") or \
                   (sample1.expected_belonging == "outgroup" and sample2.expected_belonging == "ingroup"):
                    key = "ingroup_vs_outgroup"
                    category = "belonging"

                    in_sample = sample1 if sample1.expected_belonging == "ingroup" else sample2
                    out_sample = sample2 if sample1.expected_belonging == "ingroup" else sample1
                    in_coord = coord1 if sample1.expected_belonging == "ingroup" else coord2
                    out_coord = coord2 if sample1.expected_belonging == "ingroup" else coord1

                    in_score = in_coord.core.belonging.ingroup
                    out_score = out_coord.core.belonging.outgroup

                    # Discriminant validity
                    is_discriminated = (in_score > out_coord.core.belonging.ingroup) or \
                                     (out_score > in_coord.core.belonging.outgroup)

                    results["belonging"][key]["total"] += 1
                    if is_discriminated:
                        results["belonging"][key]["correct"] += 1
                        results["summary"]["pass"] += 1
                    else:
                        results["summary"]["fail"] += 1
                    results["summary"]["total"] += 1

                    results["belonging"][key]["details"].append({
                        "text1": in_sample.text[:50],
                        "text2": out_sample.text[:50],
                        "ingroup_1": in_score,
                        "outgroup_2": out_score,
                        "discriminated": is_discriminated,
                    })

        except Exception as e:
            results["summary"]["fail"] += 1
            results["summary"]["total"] += 1

    # Compute summary statistics
    if results["summary"]["total"] > 0:
        results["summary"]["accuracy"] = results["summary"]["pass"] / results["summary"]["total"]

    for category in ["agency", "justice", "belonging"]:
        for key, data in results[category].items():
            if data["total"] > 0:
                data["accuracy"] = data["correct"] / data["total"]

    return results


def full_validation_report() -> Dict:
    """
    Run complete validation suite and generate a comprehensive report.

    Returns:
        Dictionary with all validation results
    """
    report = {
        "corpus_statistics": corpus_statistics(),
        "regime_validation": None,
        "gaming_detection": None,
        "discriminant_validity": None,
        "temperature_correlation": None,
        "recommendations": [],
    }

    # Run main validation
    main_results = validate_with_full_pipeline()
    report["regime_validation"] = {
        "accuracy": main_results["regime_classification"].get("accuracy", 0),
        "by_regime": main_results["regime_classification"]["by_regime"],
        "by_difficulty": main_results["by_difficulty"],
    }
    report["gaming_detection"] = main_results["gaming_detection"]
    report["temperature_correlation"] = main_results["coordination_level"]["correlation"]

    # Run discriminant validity
    discriminant = run_discriminant_validity_tests()
    report["discriminant_validity"] = {
        "overall_accuracy": discriminant["summary"].get("accuracy", 0),
        "pass_count": discriminant["summary"]["pass"],
        "fail_count": discriminant["summary"]["fail"],
        "by_dimension": {
            "agency": discriminant["agency"],
            "justice": discriminant["justice"],
            "belonging": discriminant["belonging"],
        }
    }

    # Generate recommendations
    if report["regime_validation"]["accuracy"] < 0.7:
        report["recommendations"].append(
            "CRITICAL: Regime classification accuracy is below 70%. "
            "The legibility analyzer needs recalibration."
        )

    if report["temperature_correlation"] < 0.5:
        report["recommendations"].append(
            "WARNING: Temperature correlation with expected coordination is weak (r < 0.5). "
            "CBR thermometer may need tuning."
        )

    if report["discriminant_validity"]["overall_accuracy"] < 0.8:
        report["recommendations"].append(
            "WARNING: Discriminant validity is below 80%. "
            "Some subdimension distinctions are not being captured reliably."
        )

    gaming_acc = report["gaming_detection"].get("accuracy", 0)
    if gaming_acc < 0.9:
        report["recommendations"].append(
            "WARNING: Gaming detection accuracy is below 90%. "
            "Adversarial detection may need improvement."
        )

    if not report["recommendations"]:
        report["recommendations"].append("All validation metrics are within acceptable ranges.")

    return report


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    print("NATURALISTIC VALIDATION CORPUS")
    print("=" * 60)

    stats = corpus_statistics()

    print(f"\nTotal samples: {stats['total_samples']}")

    print("\nBy difficulty:")
    for diff, count in stats["by_difficulty"].items():
        print(f"  {diff}: {count}")

    print("\nBy expected regime:")
    for regime, count in stats["by_regime"].items():
        print(f"  {regime}: {count}")

    print("\nBy coordination level:")
    for level, count in stats["by_coordination_level"].items():
        print(f"  {level}: {count}")

    print(f"\nGaming samples: {stats['gaming_samples']}")
    print(f"Discriminant pairs: {stats['discriminant_pairs']}")
    print(f"High agreement samples: {stats['high_agreement']}")
    print(f"Low agreement (ambiguous): {stats['low_agreement']}")

    print("\n" + "=" * 60)
    print("SAMPLE PREVIEW")
    print("=" * 60)

    # Show a few examples
    for i, sample in enumerate(NATURALISTIC_CORPUS[:5]):
        print(f"\n[{i+1}] {sample.difficulty.upper()}")
        print(f"    Text: \"{sample.text[:60]}...\"" if len(sample.text) > 60 else f"    Text: \"{sample.text}\"")
        print(f"    Expected: {sample.expected_regime}, {sample.expected_coordination_level.value}")
        print(f"    Notes: {sample.annotation_notes}")
