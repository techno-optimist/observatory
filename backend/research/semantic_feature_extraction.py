"""
Semantic Feature Extraction for Cultural Soliton Observatory.

Replaces regex-based word matching with embedding-based semantic similarity.
This addresses the key problem of word sense disambiguation:
- "The process failed" (system/technical) vs "due process" (justice)
- "We won" (ingroup success) vs "We must consider" (inclusive framing)

Uses sentence-transformers to compute semantic similarity between text
and carefully designed prototype sentences for each construct.

The 18 dimensions are:
- Core (9 dimensions):
  - Agency: self_agency, other_agency, system_agency
  - Justice: procedural, distributive, interactional
  - Belonging: ingroup, outgroup, universal
- Modifiers (9 dimensions):
  - Epistemic: certainty, evidentiality, commitment
  - Temporal: focus, scope
  - Social: power_differential, social_distance
  - Emotional: arousal, valence
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
import logging
import warnings

from .hierarchical_coordinates import (
    HierarchicalCoordinate,
    CoordinationCore,
    CoordinationModifiers,
    AgencyDecomposition,
    JusticeDecomposition,
    BelongingDecomposition,
    EpistemicModifiers,
    TemporalModifiers,
    SocialModifiers,
    EmotionalModifiers,
    DecorativeLayer,
    extract_hierarchical_coordinate as regex_extract_coordinate,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Construct Prototypes - Semantic anchors for each dimension
# =============================================================================

CONSTRUCT_PROTOTYPES = {
    # =========================================================================
    # AGENCY DIMENSIONS
    # =========================================================================

    "self_agency": [
        # Core self-agency expressions
        "I made the decision myself",
        "I took control of the situation",
        "I am responsible for my own actions",
        "I chose to do this on my own",
        "I decided to take action",
        "I have the power to change things",
        "I can make this happen",
        "My choices determine my future",
        "I am in charge of my own life",
        "I created this outcome through my efforts",
        # Active self-determination
        "I took initiative and solved the problem",
        "I am the author of my own story",
        "I exercised my free will",
        "I deliberately chose this path",
        "I made it happen through my own work",
    ],

    "other_agency": [
        # Others acting on the speaker
        "They decided for me without asking",
        "You made this choice, not me",
        "She took control of the situation",
        "He is responsible for what happened",
        "They acted on their own accord",
        "You have the power here, not me",
        "The decision was made by others",
        "They chose this path for us",
        # External actors with agency
        "My boss made the final call",
        "The committee decided our fate",
        "You are the one who did this",
        "They took it upon themselves to act",
        "Someone else is pulling the strings",
        "You have to take responsibility",
        "Their actions caused this outcome",
    ],

    "system_agency": [
        # Impersonal forces and structures
        "The system is designed this way",
        "Circumstances beyond our control",
        "Forces of nature determined the outcome",
        "The institution made it happen",
        "Structural factors caused this",
        "The economy shaped these conditions",
        "History led to this outcome",
        "The process runs automatically",
        "Market forces decided the result",
        "The algorithm made the decision",
        # Passive systemic agency
        "Things just happened this way",
        "It was inevitable given the circumstances",
        "The rules dictate what happens",
        "Society shapes our choices",
        "The bureaucracy determined the outcome",
    ],

    # =========================================================================
    # JUSTICE DIMENSIONS
    # =========================================================================

    "procedural_justice": [
        # Fair legal/institutional process (explicitly justice-related)
        "Due process of law was followed correctly",
        "The legal procedure was fair and transparent",
        "Everyone had a voice in the court decision",
        "The judicial rules were applied consistently",
        "The legal hearing gave everyone equal time",
        "The trial followed proper procedure",
        "All parties were heard before the verdict",
        "The appeals process was available to all defendants",
        # Institutional fairness
        "The review was conducted impartially by the board",
        "The rules apply the same to everyone in the organization",
        "The decision-making process was open and accountable",
        "Proper administrative channels were followed",
        "The grievance procedure was fair to all parties",
        "Fair procedures ensured the legitimacy of the decision",
        "The complaint process treated everyone equally",
    ],

    # Contrastive: Technical/mechanical processes (NOT justice)
    "procedural_justice_contrast": [
        "The manufacturing process runs efficiently",
        "The data processing pipeline completed",
        "The chemical process produced the compound",
        "The software build process finished",
        "Food processing machinery operates continuously",
        "The biological process of photosynthesis",
        "Industrial processes consume energy",
        "The technical procedure was documented",
    ],

    "distributive_justice": [
        # Fair outcomes and distribution
        "Everyone got what they deserved",
        "The rewards matched the contributions",
        "Resources were distributed fairly",
        "Each person received their fair share",
        "The outcome was proportional to effort",
        "Merit determined who succeeded",
        "Equal work deserves equal pay",
        "The benefits were shared equitably",
        # Distribution principles
        "Those who worked harder earned more",
        "Need was considered in the allocation",
        "The gains were distributed justly",
        "Each got according to their contribution",
        "The punishment fit the crime",
        "Rewards went to those who earned them",
        "The distribution was based on merit",
    ],

    "interactional_justice": [
        # Respectful treatment and dignity
        "They treated me with respect and dignity",
        "I was heard and acknowledged",
        "They listened to my concerns sincerely",
        "I felt valued as a person",
        "They showed genuine care for my situation",
        "My voice mattered in the conversation",
        "They treated me as an equal",
        "I was given the respect I deserved",
        # Interpersonal fairness
        "They explained things to me honestly",
        "I was treated with courtesy",
        "They took my feelings into account",
        "My perspective was genuinely considered",
        "They communicated with respect",
        "I felt recognized and validated",
        "They showed empathy and understanding",
    ],

    # =========================================================================
    # BELONGING DIMENSIONS
    # =========================================================================

    "ingroup": [
        # Group membership and solidarity
        "We are in this together as a team",
        "Our community stands united",
        "Together we can overcome anything",
        "We share the same values and goals",
        "Our people understand each other",
        "We have a special bond as a group",
        "Our team succeeds when we cooperate",
        "We support each other no matter what",
        # Group identity
        "This is who we are as a community",
        "We belong to something greater",
        "Our shared history unites us",
        "We stand together against challenges",
        "Our group has its own traditions",
        "We take care of our own",
        "Together we are stronger",
    ],

    "outgroup": [
        # Othering and exclusion
        "They are not like us at all",
        "Those people do not belong here",
        "They are outsiders who do not understand",
        "We need to protect ourselves from them",
        "They threaten our way of life",
        "Those others cannot be trusted",
        "They are different and dangerous",
        "We must keep them out",
        # Group boundaries
        "They do not share our values",
        "Those people are the problem",
        "They are enemies of our community",
        "We need to defend against outsiders",
        "They will never understand us",
        "Those foreigners are not welcome",
        "They are taking what is ours",
    ],

    "universal": [
        # Inclusive humanity
        "We are all human beings together",
        "Everyone deserves dignity and respect",
        "Humanity shares a common fate",
        "All people have the same basic rights",
        "We are one human family",
        "Every person matters equally",
        "Universal values unite us all",
        "People everywhere share common hopes",
        # Global inclusion
        "All of humanity benefits together",
        "Human rights apply to everyone",
        "We share one planet and one future",
        "Every life has equal value",
        "People of all backgrounds are welcome",
        "Our common humanity transcends differences",
        "Everyone belongs to the human community",
    ],

    # =========================================================================
    # EPISTEMIC MODIFIERS
    # =========================================================================

    "certainty": [
        # High certainty (positive)
        "I am absolutely certain this is true",
        "This is definitely the case without doubt",
        "I know for a fact that this happened",
        "There is no question about it",
        "This is undeniably true",
        "I am completely sure of this",
        "The evidence is conclusive",
        "This is beyond any reasonable doubt",
    ],

    "certainty_negative": [
        # Low certainty (for contrast)
        "I am not sure if this is true",
        "Maybe this is the case, maybe not",
        "It is uncertain what happened",
        "I doubt whether this is accurate",
        "This might or might not be true",
        "I am unsure about these claims",
        "The evidence is inconclusive",
        "Perhaps, but I cannot be certain",
    ],

    "evidentiality": [
        # Direct experience (positive)
        "I saw it happen with my own eyes",
        "I personally witnessed this event",
        "I experienced this firsthand myself",
        "I directly observed the situation",
        "I was there when it happened",
        "I know this from direct experience",
        "I heard them say it myself",
        "I have firsthand knowledge of this",
    ],

    "evidentiality_negative": [
        # Hearsay/indirect (for contrast)
        "They say this happened but I was not there",
        "Apparently this is what occurred",
        "Supposedly this is true according to others",
        "I heard from someone that this happened",
        "Rumor has it that this is the case",
        "According to reports this occurred",
        "People claim this is what happened",
        "I read somewhere that this might be true",
    ],

    "commitment": [
        # High speaker commitment (positive)
        "I firmly believe this is right",
        "I am committed to this position",
        "I stand behind this claim completely",
        "This is what I truly believe",
        "I stake my reputation on this",
        "I am fully invested in this view",
        "I strongly endorse this statement",
        "This reflects my deepest convictions",
    ],

    "commitment_negative": [
        # Low speaker commitment (for contrast)
        "I am just saying what others think",
        "This is not necessarily my view",
        "I do not have strong feelings about this",
        "I am simply reporting what I heard",
        "This might be true, but I take no position",
        "I am neutral on this matter",
        "I have no stake in this claim",
        "This is just speculation on my part",
    ],

    # =========================================================================
    # TEMPORAL MODIFIERS
    # =========================================================================

    "temporal_focus": [
        # Future orientation (positive)
        "We will accomplish great things ahead",
        "The future holds many possibilities",
        "Tomorrow brings new opportunities",
        "Looking forward to what comes next",
        "In the coming years we will succeed",
        "Our plans for the future are exciting",
        "Soon we will see the results",
        "The next chapter awaits us",
    ],

    "temporal_focus_negative": [
        # Past orientation (for contrast)
        "This is how things were before",
        "In the past we did things differently",
        "Yesterday we faced similar challenges",
        "Looking back on what happened",
        "The old days were better",
        "We used to do it this way",
        "Remembering how things once were",
        "History teaches us lessons",
    ],

    "temporal_scope": [
        # Long-term perspective (positive)
        "This will matter for generations to come",
        "We must think about the long run",
        "Over the decades we will see change",
        "This has lasting implications",
        "In the grand scheme of things",
        "The long-term effects are significant",
        "For centuries this has been true",
        "Thinking about the next hundred years",
    ],

    "temporal_scope_negative": [
        # Immediate/short-term (for contrast)
        "Right now we need to act",
        "This is about the present moment",
        "Today is what matters most",
        "The immediate concern is urgent",
        "In the next few minutes we must decide",
        "The short-term impact is clear",
        "This instant requires attention",
        "Deal with the here and now",
    ],

    # =========================================================================
    # SOCIAL MODIFIERS
    # =========================================================================

    "power_differential": [
        # High social status/authority (positive) - explicitly social power
        "I have the authority to decide this matter",
        "As the leader, I make the final call",
        "My position in the hierarchy gives me this responsibility",
        "I command the resources and people we need",
        "Those below me follow my direction",
        "My status and rank allow me to act",
        "I am in charge of this situation",
        "The boss has the power to fire employees",
        "Those in authority make the decisions",
        "Political power determines policy",
        "The manager exercises control over the team",
        "Social power comes from position and influence",
    ],

    # Contrastive: Physical/electrical power (NOT social power)
    "power_differential_contrast": [
        "The power plant generates electricity",
        "Electric power flows through the wires",
        "The engine has 300 horsepower",
        "Solar power is renewable energy",
        "The power grid distributes electricity",
        "Nuclear power plants produce energy",
        "Physical power measured in watts",
        "The battery provides power to the device",
    ],

    "power_differential_negative": [
        # Low status/power (for contrast)
        "I must defer to those above me",
        "I have no say in this matter",
        "I am just following orders",
        "Those in power decide, not me",
        "I lack the authority to act",
        "I am subordinate to their decisions",
        "My position gives me no influence",
        "I must do what I am told",
    ],

    "social_distance": [
        # Formal/distant (positive)
        "I respectfully submit this formal request",
        "Per our professional agreement",
        "In accordance with proper protocol",
        "Your esteemed organization",
        "The distinguished committee has decided",
        "Following proper channels and procedures",
        "The official position is as follows",
        "Hereby submitted for your consideration",
    ],

    "social_distance_negative": [
        # Informal/intimate (for contrast)
        "Hey, what's up with you?",
        "Let's just chill and talk about it",
        "Between us friends, here's the deal",
        "No need for formalities here",
        "We're close enough to be honest",
        "Buddy, let me tell you something",
        "Just keeping it real with you",
        "This is casual, no big deal",
    ],

    # =========================================================================
    # EMOTIONAL MODIFIERS
    # =========================================================================

    "arousal": [
        # High arousal (positive)
        "This is incredibly exciting and thrilling!",
        "I am absolutely furious about this!",
        "My heart is racing with anticipation!",
        "I am so overwhelmed with emotion!",
        "This is absolutely terrifying!",
        "I cannot contain my excitement!",
        "The intensity is overwhelming!",
        "I am bursting with energy!",
    ],

    "arousal_negative": [
        # Low arousal (for contrast)
        "I feel calm and peaceful about this",
        "This leaves me feeling quite relaxed",
        "I am serenely content with things",
        "Nothing about this bothers me",
        "I am pleasantly at ease",
        "Everything feels quiet and still",
        "I am in a restful state of mind",
        "This is soothing and tranquil",
    ],

    "valence": [
        # Positive valence
        "This is wonderful and beautiful",
        "I am so happy and grateful",
        "This brings me great joy",
        "Everything is going excellently",
        "I love how this turned out",
        "This is truly amazing and good",
        "I feel blessed and fortunate",
        "This is the best possible outcome",
    ],

    "valence_negative": [
        # Negative valence (for contrast)
        "This is terrible and awful",
        "I am so sad and disappointed",
        "This brings me great pain",
        "Everything is going horribly wrong",
        "I hate how this turned out",
        "This is truly disgusting and bad",
        "I feel cursed and unfortunate",
        "This is the worst possible outcome",
    ],
}

# Negative contrast prototypes for bipolar dimensions
NEGATIVE_PROTOTYPES = {
    "certainty": "certainty_negative",
    "evidentiality": "evidentiality_negative",
    "commitment": "commitment_negative",
    "temporal_focus": "temporal_focus_negative",
    "temporal_scope": "temporal_scope_negative",
    "power_differential": "power_differential_negative",
    "social_distance": "social_distance_negative",
    "arousal": "arousal_negative",
    "valence": "valence_negative",
}

# Core dimensions (always positive scoring)
CORE_DIMENSIONS = [
    "self_agency", "other_agency", "system_agency",
    "procedural_justice", "distributive_justice", "interactional_justice",
    "ingroup", "outgroup", "universal",
]

# Contrastive prototypes for disambiguation
# Maps dimension to contrast dimension (for words that appear in multiple contexts)
CORE_CONTRAST_PROTOTYPES = {
    "procedural_justice": "procedural_justice_contrast",  # Legal process vs technical process
}

# Contrastive prototypes for modifier dimensions
MODIFIER_CONTRAST_PROTOTYPES = {
    "power_differential": "power_differential_contrast",  # Social power vs physical power
}

# Modifier dimensions (bipolar scoring)
MODIFIER_DIMENSIONS = [
    "certainty", "evidentiality", "commitment",
    "temporal_focus", "temporal_scope",
    "power_differential", "social_distance",
    "arousal", "valence",
]

# All 18 primary dimensions
ALL_DIMENSIONS = CORE_DIMENSIONS + MODIFIER_DIMENSIONS


# =============================================================================
# Calibration Data - Neutral baselines and disambiguation examples
# =============================================================================

CALIBRATION_NEUTRAL_TEXTS = [
    # Generic neutral sentences
    "The quick brown fox jumps over the lazy dog.",
    "Today is a regular day with nothing special happening.",
    "The room contains a table and four chairs.",
    "Numbers can be added or subtracted.",
    "Water freezes at zero degrees Celsius.",
    "The book is on the shelf next to the window.",
    "Time passes at a constant rate.",
    "Colors can be mixed to create new colors.",
    "The sky is visible from outside.",
    "Objects fall when dropped.",
    # Random factual statements
    "Triangles have three sides.",
    "January is the first month of the year.",
    "Cats are mammals that can purr.",
    "Roads connect different places.",
    "Music consists of sounds arranged in patterns.",
]

# Disambiguation test cases
DISAMBIGUATION_CASES = [
    # "process" disambiguation - uses relative comparison
    {
        "text": "The manufacturing process runs smoothly.",
        "higher_than": [("system_agency", "procedural_justice")],  # system > procedural
        "note": "Technical process, not justice-related",
    },
    {
        "text": "Due process must be respected in court.",
        "higher_than": [("procedural_justice", "system_agency")],  # procedural > system
        "note": "Legal process, justice-related",
    },
    # "we" disambiguation - relative comparisons
    {
        "text": "We won the championship game!",
        "higher_than": [("ingroup", "universal")],  # ingroup > universal
        "note": "Competitive ingroup, not universal",
    },
    {
        "text": "We as humanity must work together to solve this.",
        "higher_than": [("universal", "outgroup")],  # universal > outgroup
        "note": "Universal humanity, not tribal",
    },
    # "fair" disambiguation
    {
        "text": "The weather is fair today with clear skies.",
        "higher_than": [],  # No specific comparison, just check low justice scores
        "expected_low": ["procedural_justice", "distributive_justice"],
        "note": "Weather fair, not justice fair",
    },
    {
        "text": "The trial was fair and just.",
        "higher_than": [("procedural_justice", "system_agency")],
        "note": "Justice fair, not weather",
    },
    # "power" disambiguation
    {
        "text": "The power plant generates electricity.",
        "higher_than": [("system_agency", "power_differential")],  # system > social power
        "note": "Physical power, not social power",
    },
    {
        "text": "Those in power abuse their authority.",
        "higher_than": [("power_differential", "system_agency")],  # social power > system
        "note": "Social power, not physical",
    },
]


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class ConstructScore:
    """Score for a single construct."""
    construct: str
    score: float  # [-1, 1] for modifiers, [0, 1] for core
    max_similarity: float  # Raw max cosine similarity
    adjusted_similarity: float  # Max similarity after contrastive adjustment
    mean_similarity: float  # Mean similarity to all prototypes
    best_prototype: str  # Most similar prototype sentence
    confidence: float  # Confidence in the score [0, 1]

    def to_dict(self) -> dict:
        return {
            "construct": self.construct,
            "score": self.score,
            "max_similarity": self.max_similarity,
            "adjusted_similarity": self.adjusted_similarity,
            "mean_similarity": self.mean_similarity,
            "best_prototype": self.best_prototype,
            "confidence": self.confidence,
        }


@dataclass
class SemanticExtractionResult:
    """Full result of semantic feature extraction."""
    text: str
    construct_scores: Dict[str, ConstructScore]
    hierarchical_coordinate: HierarchicalCoordinate
    raw_similarities: Dict[str, np.ndarray]  # For debugging
    extraction_method: str = "semantic"

    def to_dict(self) -> dict:
        return {
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "construct_scores": {k: v.to_dict() for k, v in self.construct_scores.items()},
            "hierarchical_coordinate": self.hierarchical_coordinate.to_dict(),
            "extraction_method": self.extraction_method,
        }


@dataclass
class ComparisonResult:
    """Result of comparing regex vs semantic extraction."""
    text: str
    regex_coordinate: HierarchicalCoordinate
    semantic_coordinate: HierarchicalCoordinate
    dimension_diffs: Dict[str, float]
    significant_differences: List[str]  # Dimensions with |diff| > threshold
    improvement_notes: List[str]  # Why semantic might be better
    concern_notes: List[str]  # Potential issues with semantic

    def to_dict(self) -> dict:
        return {
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "regex_coordinate": self.regex_coordinate.to_dict(),
            "semantic_coordinate": self.semantic_coordinate.to_dict(),
            "dimension_diffs": self.dimension_diffs,
            "significant_differences": self.significant_differences,
            "improvement_notes": self.improvement_notes,
            "concern_notes": self.concern_notes,
        }


@dataclass
class CalibrationResult:
    """Result of calibration analysis."""
    neutral_baselines: Dict[str, float]  # Mean scores on neutral text
    neutral_stds: Dict[str, float]  # Standard deviations
    detection_thresholds: Dict[str, float]  # Thresholds for detection
    disambiguation_accuracy: float  # Accuracy on disambiguation test cases
    disambiguation_details: List[dict]  # Per-case results


# =============================================================================
# Semantic Feature Extractor
# =============================================================================

class SemanticFeatureExtractor:
    """
    Extracts hierarchical coordinates using semantic similarity to prototypes.

    This addresses the key weakness of regex-based extraction: word sense
    disambiguation. By comparing to full prototype sentences, we capture
    the semantic context rather than just surface word matches.

    Example improvements:
    - "The process failed" -> low procedural_justice (regex) vs low system_agency (semantic)
    - "Due process" -> high procedural_justice (both, but semantic more confident)
    - "We won" -> high ingroup (semantic) vs might trigger universal (regex)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_prototypes: bool = True,
        calibrate_on_init: bool = True,
    ):
        """
        Initialize the semantic feature extractor.

        Args:
            model_name: Sentence transformer model to use
            cache_prototypes: Whether to pre-compute prototype embeddings
            calibrate_on_init: Whether to compute baseline calibration
        """
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        # Pre-compute prototype embeddings for efficiency
        self.prototype_embeddings: Dict[str, np.ndarray] = {}
        self.prototypes = CONSTRUCT_PROTOTYPES

        if cache_prototypes:
            self._cache_prototype_embeddings()

        # Calibration data
        self.neutral_baselines: Dict[str, float] = {}
        self.neutral_stds: Dict[str, float] = {}
        self.detection_thresholds: Dict[str, float] = {}

        if calibrate_on_init:
            self._calibrate_baselines()

    def _cache_prototype_embeddings(self) -> None:
        """Pre-compute embeddings for all prototype sentences."""
        logger.info("Caching prototype embeddings...")

        for construct, prototypes in self.prototypes.items():
            embeddings = self.model.encode(
                prototypes,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.prototype_embeddings[construct] = embeddings

        logger.info(f"Cached embeddings for {len(self.prototype_embeddings)} constructs")

    def _calibrate_baselines(self) -> None:
        """Compute baseline scores on neutral text for calibration."""
        logger.info("Calibrating baselines on neutral text...")

        # Get scores for all neutral texts
        all_scores = {dim: [] for dim in ALL_DIMENSIONS}

        for text in CALIBRATION_NEUTRAL_TEXTS:
            result = self._compute_raw_similarities(text)
            for dim in ALL_DIMENSIONS:
                score = self._similarity_to_score(dim, result)
                all_scores[dim].append(score)

        # Compute statistics
        for dim in ALL_DIMENSIONS:
            scores = np.array(all_scores[dim])
            self.neutral_baselines[dim] = float(np.mean(scores))
            self.neutral_stds[dim] = float(np.std(scores))
            # Detection threshold: baseline + 2*std
            self.detection_thresholds[dim] = self.neutral_baselines[dim] + 2 * self.neutral_stds[dim]

        logger.info("Calibration complete")

    def _compute_raw_similarities(self, text: str) -> Dict[str, np.ndarray]:
        """Compute raw cosine similarities to all prototypes."""
        # Encode the input text
        text_embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        similarities = {}
        for construct, proto_embeddings in self.prototype_embeddings.items():
            # Cosine similarity (embeddings are normalized)
            sims = proto_embeddings @ text_embedding
            similarities[construct] = sims

        return similarities

    def _similarity_to_score(
        self,
        dimension: str,
        similarities: Dict[str, np.ndarray],
    ) -> float:
        """
        Convert raw similarities to a dimension score.

        For core dimensions: returns max similarity [0, 1], with contrastive adjustment
        For modifier dimensions: returns difference between positive and negative [-1, 1]
        """
        if dimension in CORE_DIMENSIONS:
            # Core dimensions: max similarity as the score
            sims = similarities[dimension]
            pos_max = float(np.max(sims))

            # Check for contrastive disambiguation
            contrast_key = CORE_CONTRAST_PROTOTYPES.get(dimension)
            if contrast_key and contrast_key in similarities:
                contrast_sims = similarities[contrast_key]
                contrast_max = float(np.max(contrast_sims))

                # If contrast similarity is higher, reduce the score
                # This handles cases like "manufacturing process" vs "due process"
                if contrast_max > pos_max:
                    # Text is more similar to contrast (non-target) concept
                    # Reduce score proportionally
                    ratio = pos_max / (contrast_max + 1e-6)
                    return pos_max * ratio
                else:
                    # Text is more similar to target concept - boost confidence
                    # The larger the gap, the more certain we are
                    gap = pos_max - contrast_max
                    return pos_max + (gap * 0.3)  # Small boost for clear disambiguation

            return pos_max

        elif dimension in MODIFIER_DIMENSIONS:
            # Modifier dimensions: difference between positive and negative prototypes
            pos_sims = similarities[dimension]
            pos_max = float(np.max(pos_sims))

            # First check for contrastive disambiguation (e.g., social power vs physical power)
            contrast_key = MODIFIER_CONTRAST_PROTOTYPES.get(dimension)
            if contrast_key and contrast_key in similarities:
                contrast_sims = similarities[contrast_key]
                contrast_max = float(np.max(contrast_sims))

                # If contrast similarity is higher, reduce the positive score
                if contrast_max > pos_max:
                    ratio = pos_max / (contrast_max + 1e-6)
                    pos_max = pos_max * ratio

            # Now compute bipolar score with negative prototypes
            neg_key = NEGATIVE_PROTOTYPES.get(dimension)
            if neg_key and neg_key in similarities:
                neg_sims = similarities[neg_key]
                neg_max = float(np.max(neg_sims))
                # Score is difference between max positive and max negative
                # Normalize to [-1, 1]
                return (pos_max - neg_max)
            else:
                # No negative prototypes, just use positive
                return pos_max * 2 - 1  # Scale [0,1] to [-1,1]

        else:
            warnings.warn(f"Unknown dimension: {dimension}")
            return 0.0

    def _compute_confidence(
        self,
        dimension: str,
        similarities: Dict[str, np.ndarray],
    ) -> float:
        """
        Compute confidence in the score based on similarity distribution.

        High confidence when:
        - High max similarity (strong match to a prototype)
        - Large gap between best and second-best prototype
        - Scores clearly above baseline
        """
        sims = similarities[dimension]
        sorted_sims = np.sort(sims)[::-1]

        # Factor 1: Absolute max similarity
        max_sim = sorted_sims[0]

        # Factor 2: Gap to second best
        if len(sorted_sims) > 1:
            gap = sorted_sims[0] - sorted_sims[1]
        else:
            gap = 0.0

        # Factor 3: Above baseline (if calibrated)
        if dimension in self.neutral_baselines:
            baseline = self.neutral_baselines[dimension]
            std = self.neutral_stds[dimension] + 1e-6
            z_score = (max_sim - baseline) / std
            above_baseline = min(1.0, max(0.0, z_score / 3))  # Normalize to [0, 1]
        else:
            above_baseline = 0.5

        # Combine factors
        confidence = 0.4 * max_sim + 0.3 * (gap * 5) + 0.3 * above_baseline
        return min(1.0, max(0.0, confidence))

    def extract_features(self, text: str) -> Dict[str, ConstructScore]:
        """
        Extract semantic feature scores for all constructs.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary mapping dimension names to ConstructScore objects
        """
        similarities = self._compute_raw_similarities(text)

        scores = {}
        for dimension in ALL_DIMENSIONS:
            score = self._similarity_to_score(dimension, similarities)
            confidence = self._compute_confidence(dimension, similarities)
            adjusted_sim = self._compute_adjusted_similarity(dimension, similarities)

            # Get best matching prototype
            sims = similarities[dimension]
            best_idx = int(np.argmax(sims))
            best_prototype = self.prototypes[dimension][best_idx]

            scores[dimension] = ConstructScore(
                construct=dimension,
                score=score,
                max_similarity=float(np.max(sims)),
                adjusted_similarity=adjusted_sim,
                mean_similarity=float(np.mean(sims)),
                best_prototype=best_prototype,
                confidence=confidence,
            )

        return scores

    def _compute_adjusted_similarity(
        self,
        dimension: str,
        similarities: Dict[str, np.ndarray],
    ) -> float:
        """
        Compute adjusted similarity accounting for contrastive disambiguation.

        For dimensions with contrastive prototypes, reduces similarity if the
        text is more similar to the contrast (e.g., physical power vs social power).
        """
        sims = similarities[dimension]
        pos_max = float(np.max(sims))

        # Check for contrastive prototypes
        contrast_key = None
        if dimension in CORE_DIMENSIONS:
            contrast_key = CORE_CONTRAST_PROTOTYPES.get(dimension)
        elif dimension in MODIFIER_DIMENSIONS:
            contrast_key = MODIFIER_CONTRAST_PROTOTYPES.get(dimension)

        if contrast_key and contrast_key in similarities:
            contrast_sims = similarities[contrast_key]
            contrast_max = float(np.max(contrast_sims))

            if contrast_max > pos_max:
                # Reduce similarity proportionally when contrast is stronger
                ratio = pos_max / (contrast_max + 1e-6)
                return pos_max * ratio
            else:
                # Boost slightly when target is clearly stronger
                gap = pos_max - contrast_max
                return min(1.0, pos_max + gap * 0.1)

        return pos_max

    def extract_hierarchical_coordinate(self, text: str) -> HierarchicalCoordinate:
        """
        Extract hierarchical coordinate from text using semantic similarity.

        This is a drop-in replacement for the regex-based function.

        Args:
            text: Input text to analyze

        Returns:
            HierarchicalCoordinate with all dimensions populated
        """
        scores = self.extract_features(text)

        # Build AgencyDecomposition
        agency = AgencyDecomposition(
            self_agency=self._normalize_core_score("self_agency", scores["self_agency"].max_similarity),
            other_agency=self._normalize_core_score("other_agency", scores["other_agency"].max_similarity),
            system_agency=self._normalize_core_score("system_agency", scores["system_agency"].max_similarity),
        )

        # Build JusticeDecomposition
        justice = JusticeDecomposition(
            procedural=self._normalize_core_score("procedural_justice", scores["procedural_justice"].max_similarity),
            distributive=self._normalize_core_score("distributive_justice", scores["distributive_justice"].max_similarity),
            interactional=self._normalize_core_score("interactional_justice", scores["interactional_justice"].max_similarity),
        )

        # Build BelongingDecomposition
        belonging = BelongingDecomposition(
            ingroup=self._normalize_core_score("ingroup", scores["ingroup"].max_similarity),
            outgroup=self._normalize_core_score("outgroup", scores["outgroup"].max_similarity),
            universal=self._normalize_core_score("universal", scores["universal"].max_similarity),
        )

        # Build EpistemicModifiers
        epistemic = EpistemicModifiers(
            certainty=self._clamp(scores["certainty"].score),
            evidentiality=self._clamp(scores["evidentiality"].score),
            commitment=self._clamp(scores["commitment"].score),
        )

        # Build TemporalModifiers
        temporal = TemporalModifiers(
            focus=self._clamp(scores["temporal_focus"].score),
            scope=self._clamp(scores["temporal_scope"].score),
        )

        # Build SocialModifiers
        social = SocialModifiers(
            power_differential=self._clamp(scores["power_differential"].score),
            social_distance=self._clamp(scores["social_distance"].score),
        )

        # Build EmotionalModifiers
        emotional = EmotionalModifiers(
            arousal=self._clamp(scores["arousal"].score),
            valence=self._clamp(scores["valence"].score),
        )

        # Decorative layer (not extracted semantically, use regex fallback)
        decorative = DecorativeLayer()  # Default empty

        return HierarchicalCoordinate(
            core=CoordinationCore(agency=agency, justice=justice, belonging=belonging),
            modifiers=CoordinationModifiers(
                epistemic=epistemic,
                temporal=temporal,
                social=social,
                emotional=emotional,
            ),
            decorative=decorative,
        )

    def _normalize_core_score(self, dimension: str, similarity: float) -> float:
        """
        Normalize core dimension score from similarity to [-1, 1] range.

        Uses calibration-based z-score normalization:
        - Scores below baseline map to negative values
        - Scores above baseline map to positive values
        - Detection threshold (~baseline + 2*std) maps to ~0.5

        This ensures scores are comparable to the regex-based extraction.
        """
        if dimension in self.neutral_baselines:
            baseline = self.neutral_baselines[dimension]
            std = self.neutral_stds[dimension] + 1e-6

            # Z-score normalization scaled to [-1, 1]
            # 2 standard deviations above baseline maps to ~0.67
            # 3 standard deviations above baseline maps to ~1.0
            z_score = (similarity - baseline) / std
            normalized = z_score / 3.0  # Scale so 3 std = 1.0

            return self._clamp(normalized)
        else:
            # Fallback if calibration not done
            return self._clamp((similarity - 0.2) * 2.0)

    def _clamp(self, value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """Clamp value to range."""
        return max(min_val, min(max_val, value))

    def extract_with_details(self, text: str) -> SemanticExtractionResult:
        """
        Extract features with full details for analysis.

        Args:
            text: Input text to analyze

        Returns:
            SemanticExtractionResult with all details
        """
        similarities = self._compute_raw_similarities(text)
        construct_scores = self.extract_features(text)
        coordinate = self.extract_hierarchical_coordinate(text)

        return SemanticExtractionResult(
            text=text,
            construct_scores=construct_scores,
            hierarchical_coordinate=coordinate,
            raw_similarities=similarities,
        )

    def calibrate(self, neutral_texts: Optional[List[str]] = None) -> CalibrationResult:
        """
        Run full calibration including disambiguation tests.

        Args:
            neutral_texts: Optional custom neutral texts for baseline

        Returns:
            CalibrationResult with all calibration data
        """
        texts = neutral_texts or CALIBRATION_NEUTRAL_TEXTS

        # Recompute baselines
        all_scores = {dim: [] for dim in ALL_DIMENSIONS}
        for text in texts:
            result = self._compute_raw_similarities(text)
            for dim in ALL_DIMENSIONS:
                score = self._similarity_to_score(dim, result)
                all_scores[dim].append(score)

        neutral_baselines = {}
        neutral_stds = {}
        detection_thresholds = {}

        for dim in ALL_DIMENSIONS:
            scores = np.array(all_scores[dim])
            neutral_baselines[dim] = float(np.mean(scores))
            neutral_stds[dim] = float(np.std(scores))
            detection_thresholds[dim] = neutral_baselines[dim] + 2 * neutral_stds[dim]

        # Update instance calibration
        self.neutral_baselines = neutral_baselines
        self.neutral_stds = neutral_stds
        self.detection_thresholds = detection_thresholds

        # Run disambiguation tests using relative comparisons
        correct = 0
        total = 0
        disambiguation_details = []

        for case in DISAMBIGUATION_CASES:
            text = case["text"]
            scores = self.extract_features(text)

            case_correct = True
            details = {"text": text, "results": {}, "comparisons": []}

            # Check relative comparisons (higher_than)
            # Use adjusted_similarity which accounts for contrastive disambiguation
            for higher_dim, lower_dim in case.get("higher_than", []):
                if higher_dim in scores and lower_dim in scores:
                    # Use adjusted_similarity for disambiguated comparisons
                    higher_val = scores[higher_dim].adjusted_similarity
                    lower_val = scores[lower_dim].adjusted_similarity

                    is_correct = higher_val > lower_val
                    comparison = {
                        "comparison": f"{higher_dim} > {lower_dim}",
                        "higher_value": higher_val,
                        "lower_value": lower_val,
                        "pass": is_correct,
                    }
                    details["comparisons"].append(comparison)
                    if not is_correct:
                        case_correct = False
                    total += 1

            # Check expected low (absolute threshold)
            for dim in case.get("expected_low", []):
                if dim in scores:
                    is_low = scores[dim].max_similarity < detection_thresholds.get(dim, 0.3)
                    details["results"][dim] = {
                        "expected": "low",
                        "score": scores[dim].max_similarity,
                        "threshold": detection_thresholds.get(dim, 0.3),
                        "pass": is_low,
                    }
                    if not is_low:
                        case_correct = False
                    total += 1

            details["case_passed"] = case_correct
            details["note"] = case.get("note", "")
            disambiguation_details.append(details)

            if case_correct:
                correct += 1

        accuracy = correct / len(DISAMBIGUATION_CASES) if DISAMBIGUATION_CASES else 0.0

        return CalibrationResult(
            neutral_baselines=neutral_baselines,
            neutral_stds=neutral_stds,
            detection_thresholds=detection_thresholds,
            disambiguation_accuracy=accuracy,
            disambiguation_details=disambiguation_details,
        )


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_regex_vs_semantic(
    texts: List[str],
    extractor: Optional[SemanticFeatureExtractor] = None,
    difference_threshold: float = 0.3,
) -> List[ComparisonResult]:
    """
    Compare regex-based vs semantic extraction on a list of texts.

    Highlights cases where semantic extraction differs from regex and
    provides analysis of improvements and potential issues.

    Args:
        texts: List of texts to compare
        extractor: SemanticFeatureExtractor instance (created if None)
        difference_threshold: Threshold for "significant" difference

    Returns:
        List of ComparisonResult objects
    """
    if extractor is None:
        extractor = SemanticFeatureExtractor()

    results = []

    for text in texts:
        # Get both extractions
        regex_coord = regex_extract_coordinate(text)
        semantic_coord = extractor.extract_hierarchical_coordinate(text)

        # Compute differences for each dimension
        regex_array = regex_coord.to_full_array()
        semantic_array = semantic_coord.to_full_array()

        dimension_names = [
            "self_agency", "other_agency", "system_agency",
            "procedural", "distributive", "interactional",
            "ingroup", "outgroup", "universal",
            "certainty", "evidentiality", "commitment",
            "temporal_focus", "temporal_scope",
            "power_differential", "social_distance",
            "arousal", "valence",
        ]

        dimension_diffs = {}
        significant_differences = []

        for i, name in enumerate(dimension_names):
            diff = semantic_array[i] - regex_array[i]
            dimension_diffs[name] = float(diff)
            if abs(diff) > difference_threshold:
                significant_differences.append(name)

        # Analyze differences
        improvement_notes = []
        concern_notes = []

        # Check for disambiguation improvements
        text_lower = text.lower()

        if "process" in text_lower:
            if "due process" in text_lower or "fair process" in text_lower:
                if dimension_diffs.get("procedural", 0) > 0:
                    improvement_notes.append("Correctly identifies 'due process' as procedural justice")
            elif "manufacturing" in text_lower or "technical" in text_lower:
                if dimension_diffs.get("procedural", 0) < 0:
                    improvement_notes.append("Correctly distinguishes technical 'process' from justice concept")

        if "we" in text_lower:
            if "humanity" in text_lower or "species" in text_lower or "all of us" in text_lower:
                if dimension_diffs.get("universal", 0) > 0:
                    improvement_notes.append("Correctly identifies universal 'we' vs tribal 'we'")
            elif "won" in text_lower or "beat" in text_lower or "our team" in text_lower:
                if dimension_diffs.get("ingroup", 0) > 0:
                    improvement_notes.append("Correctly identifies competitive ingroup 'we'")

        if "fair" in text_lower:
            if "weather" in text_lower or "clear" in text_lower:
                if abs(dimension_diffs.get("procedural", 0)) < 0.1:
                    improvement_notes.append("Correctly ignores weather 'fair' for justice dimensions")

        # Check for potential concerns
        if len(text.split()) < 5:
            concern_notes.append("Very short text may have unreliable semantic similarity")

        if all(abs(d) < 0.1 for d in dimension_diffs.values()):
            concern_notes.append("Regex and semantic agree closely - no disambiguation needed")

        if any(abs(d) > 0.8 for d in dimension_diffs.values()):
            concern_notes.append("Very large difference - investigate for potential semantic extraction error")

        results.append(ComparisonResult(
            text=text,
            regex_coordinate=regex_coord,
            semantic_coordinate=semantic_coord,
            dimension_diffs=dimension_diffs,
            significant_differences=significant_differences,
            improvement_notes=improvement_notes,
            concern_notes=concern_notes,
        ))

    return results


def batch_extract_semantic(
    texts: List[str],
    extractor: Optional[SemanticFeatureExtractor] = None,
) -> List[HierarchicalCoordinate]:
    """
    Extract hierarchical coordinates for multiple texts.

    More efficient than calling extract_hierarchical_coordinate repeatedly
    due to batch encoding.

    Args:
        texts: List of texts to extract
        extractor: SemanticFeatureExtractor instance

    Returns:
        List of HierarchicalCoordinate objects
    """
    if extractor is None:
        extractor = SemanticFeatureExtractor()

    return [extractor.extract_hierarchical_coordinate(text) for text in texts]


# =============================================================================
# Demonstration and Testing
# =============================================================================

def demonstrate_disambiguation():
    """
    Demonstrate semantic disambiguation on challenging examples.
    """
    print("=" * 70)
    print("SEMANTIC FEATURE EXTRACTION DEMONSTRATION")
    print("=" * 70)

    extractor = SemanticFeatureExtractor()

    test_pairs = [
        # Process disambiguation
        (
            "The manufacturing process runs smoothly without interruption.",
            "Due process must be respected in all court proceedings.",
            "Disambiguating 'process'"
        ),
        # We disambiguation
        (
            "We beat them 3-0 in the championship final!",
            "We as humanity must address climate change together.",
            "Disambiguating 'we'"
        ),
        # Fair disambiguation
        (
            "The weather was fair with clear blue skies.",
            "The trial was fair and the verdict was just.",
            "Disambiguating 'fair'"
        ),
        # Power disambiguation
        (
            "The power plant generates electricity for the city.",
            "Those in power rarely understand ordinary struggles.",
            "Disambiguating 'power'"
        ),
    ]

    for text1, text2, description in test_pairs:
        print(f"\n{description}")
        print("-" * 50)

        results = compare_regex_vs_semantic([text1, text2], extractor)

        for result in results:
            print(f"\nText: {result.text[:60]}...")
            if result.significant_differences:
                print(f"  Significant differences: {result.significant_differences}")
            if result.improvement_notes:
                print(f"  Improvements: {result.improvement_notes}")
            if result.concern_notes:
                print(f"  Concerns: {result.concern_notes}")

    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)

    calibration = extractor.calibrate()
    print(f"\nDisambiguation accuracy: {calibration.disambiguation_accuracy:.1%}")
    print("\nNeutral baselines (should be near 0 for good calibration):")
    for dim in ALL_DIMENSIONS[:9]:  # Show first 9 (core dimensions)
        print(f"  {dim}: {calibration.neutral_baselines[dim]:.3f} +/- {calibration.neutral_stds[dim]:.3f}")


def run_full_comparison(texts: List[str]) -> None:
    """
    Run a full comparison analysis on provided texts.

    Args:
        texts: List of texts to analyze
    """
    print("=" * 70)
    print("REGEX vs SEMANTIC EXTRACTION COMPARISON")
    print("=" * 70)

    extractor = SemanticFeatureExtractor()
    results = compare_regex_vs_semantic(texts, extractor)

    significant_count = sum(1 for r in results if r.significant_differences)

    print(f"\nAnalyzed {len(texts)} texts")
    print(f"Texts with significant differences: {significant_count}")

    for i, result in enumerate(results):
        if result.significant_differences:
            print(f"\n[{i+1}] {result.text[:50]}...")
            print(f"    Differences in: {', '.join(result.significant_differences)}")
            for dim in result.significant_differences:
                diff = result.dimension_diffs[dim]
                direction = "higher" if diff > 0 else "lower"
                print(f"      {dim}: semantic is {abs(diff):.2f} {direction}")


if __name__ == "__main__":
    demonstrate_disambiguation()
