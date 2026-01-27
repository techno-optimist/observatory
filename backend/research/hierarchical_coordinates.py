"""
Hierarchical Coordinate Structure for High-Resolution Manifold Analysis.

Implements the three-tier structure from V2.0 design:

1. CoordinationCore (9 dimensions) - The irreducible coordination substrate
   - Agency: self, other, system
   - Justice: procedural, distributive, interactional
   - Belonging: ingroup, outgroup, universal

2. CoordinationModifiers (9 dimensions) - Features that modulate coordination
   - Epistemic: certainty, evidentiality, commitment
   - Temporal: focus, scope
   - Social: power_differential, social_distance
   - Emotional: arousal, valence

3. DecorativeLayer - Features stripped during core extraction
   - Articles, hedging, intensifiers, filler, style markers

Based on expert synthesis:
- Cognitive scientist: First-person = deictic anchor for perspectival content
- Linguistics expert: Deixis types (person, spatial, temporal)
- Mathematical physicist: Fiber bundle structure E = B x F
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Hierarchical Coordinate Dataclasses
# =============================================================================

@dataclass
class AgencyDecomposition:
    """Decomposed agency along three sub-dimensions."""

    self_agency: float = 0.0      # "I did", "I chose", "my action"
    other_agency: float = 0.0     # "they did", "you chose", "their action"
    system_agency: float = 0.0    # "the system", "forces", "circumstances"

    def to_array(self) -> np.ndarray:
        return np.array([self.self_agency, self.other_agency, self.system_agency])

    def to_dict(self) -> dict:
        return {
            "self_agency": self.self_agency,
            "other_agency": self.other_agency,
            "system_agency": self.system_agency
        }

    @property
    def aggregate(self) -> float:
        """Compute aggregate agency score (-1 to 1)."""
        # Self agency positive, system agency negative, other neutral
        return self.self_agency - self.system_agency

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "AgencyDecomposition":
        return cls(self_agency=arr[0], other_agency=arr[1], system_agency=arr[2])


@dataclass
class JusticeDecomposition:
    """Decomposed justice along three sub-dimensions."""

    procedural: float = 0.0       # Fair process: "proper review", "due process"
    distributive: float = 0.0     # Fair outcomes: "deserved", "earned"
    interactional: float = 0.0    # Fair treatment: "respected", "heard"

    def to_array(self) -> np.ndarray:
        return np.array([self.procedural, self.distributive, self.interactional])

    def to_dict(self) -> dict:
        return {
            "procedural": self.procedural,
            "distributive": self.distributive,
            "interactional": self.interactional
        }

    @property
    def aggregate(self) -> float:
        """Compute aggregate justice score (-1 to 1)."""
        return (self.procedural + self.distributive + self.interactional) / 3

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "JusticeDecomposition":
        return cls(procedural=arr[0], distributive=arr[1], interactional=arr[2])


@dataclass
class BelongingDecomposition:
    """Decomposed belonging along three sub-dimensions."""

    ingroup: float = 0.0          # "we", "our community", "together"
    outgroup: float = 0.0         # "them", "those people", "outsiders"
    universal: float = 0.0        # "everyone", "humanity", "all of us"

    def to_array(self) -> np.ndarray:
        return np.array([self.ingroup, self.outgroup, self.universal])

    def to_dict(self) -> dict:
        return {
            "ingroup": self.ingroup,
            "outgroup": self.outgroup,
            "universal": self.universal
        }

    @property
    def aggregate(self) -> float:
        """Compute aggregate belonging score (-1 to 1)."""
        # Ingroup and universal positive, outgroup creates tension
        return (self.ingroup + self.universal - self.outgroup) / 2

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "BelongingDecomposition":
        return cls(ingroup=arr[0], outgroup=arr[1], universal=arr[2])


@dataclass
class CoordinationCore:
    """
    The irreducible coordination substrate - 9 dimensions.

    This is the "base manifold" (B) in the fiber bundle E = B x F.
    Removing any of these dimensions causes coordination drift.
    """

    agency: AgencyDecomposition = field(default_factory=AgencyDecomposition)
    justice: JusticeDecomposition = field(default_factory=JusticeDecomposition)
    belonging: BelongingDecomposition = field(default_factory=BelongingDecomposition)

    def to_array(self) -> np.ndarray:
        """Return 9D array: [agency_3, justice_3, belonging_3]."""
        return np.concatenate([
            self.agency.to_array(),
            self.justice.to_array(),
            self.belonging.to_array()
        ])

    def to_dict(self) -> dict:
        return {
            "agency": self.agency.to_dict(),
            "justice": self.justice.to_dict(),
            "belonging": self.belonging.to_dict()
        }

    def to_legacy_3d(self) -> Tuple[float, float, float]:
        """Convert to legacy 3D coordinates for backward compatibility."""
        return (
            self.agency.aggregate,
            self.justice.aggregate,
            self.belonging.aggregate
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "CoordinationCore":
        """Create from 9D array."""
        return cls(
            agency=AgencyDecomposition.from_array(arr[0:3]),
            justice=JusticeDecomposition.from_array(arr[3:6]),
            belonging=BelongingDecomposition.from_array(arr[6:9])
        )

    @classmethod
    def from_legacy_3d(
        cls,
        agency: float,
        justice: float,
        belonging: float
    ) -> "CoordinationCore":
        """
        Create from legacy 3D coordinates.

        Distributes aggregate scores evenly across sub-dimensions.
        """
        return cls(
            agency=AgencyDecomposition(
                self_agency=agency * 0.6,
                other_agency=0.0,
                system_agency=-agency * 0.4 if agency < 0 else 0.0
            ),
            justice=JusticeDecomposition(
                procedural=justice * 0.4,
                distributive=justice * 0.4,
                interactional=justice * 0.2
            ),
            belonging=BelongingDecomposition(
                ingroup=belonging * 0.5,
                outgroup=0.0,
                universal=belonging * 0.5
            )
        )


@dataclass
class EpistemicModifiers:
    """Epistemic stance modifiers."""

    certainty: float = 0.0         # "definitely" (+1) vs "maybe" (-1)
    evidentiality: float = 0.0     # Direct experience (+1) vs hearsay (-1)
    commitment: float = 0.0        # Speaker stake in claim (-1 to +1)

    def to_array(self) -> np.ndarray:
        return np.array([self.certainty, self.evidentiality, self.commitment])

    def to_dict(self) -> dict:
        return {
            "certainty": self.certainty,
            "evidentiality": self.evidentiality,
            "commitment": self.commitment
        }


@dataclass
class TemporalModifiers:
    """Temporal orientation modifiers."""

    focus: float = 0.0            # Past (-1) vs future (+1)
    scope: float = 0.0            # Immediate (-1) vs long-term (+1)

    def to_array(self) -> np.ndarray:
        return np.array([self.focus, self.scope])

    def to_dict(self) -> dict:
        return {
            "focus": self.focus,
            "scope": self.scope
        }


@dataclass
class SocialModifiers:
    """Social relationship modifiers."""

    power_differential: float = 0.0    # Low status (-1) vs high status (+1)
    social_distance: float = 0.0       # Intimate (-1) vs formal (+1)

    def to_array(self) -> np.ndarray:
        return np.array([self.power_differential, self.social_distance])

    def to_dict(self) -> dict:
        return {
            "power_differential": self.power_differential,
            "social_distance": self.social_distance
        }


@dataclass
class EmotionalModifiers:
    """Emotional/affective modifiers."""

    arousal: float = 0.0          # Low arousal (-1) vs high arousal (+1)
    valence: float = 0.0          # Negative (-1) vs positive (+1)

    def to_array(self) -> np.ndarray:
        return np.array([self.arousal, self.valence])

    def to_dict(self) -> dict:
        return {
            "arousal": self.arousal,
            "valence": self.valence
        }


@dataclass
class CoordinationModifiers:
    """
    Features that modulate but don't determine coordination.

    These are the "fiber" (F) in the fiber bundle structure.
    Different texts can have the same base coordinates but different modifiers.
    """

    epistemic: EpistemicModifiers = field(default_factory=EpistemicModifiers)
    temporal: TemporalModifiers = field(default_factory=TemporalModifiers)
    social: SocialModifiers = field(default_factory=SocialModifiers)
    emotional: EmotionalModifiers = field(default_factory=EmotionalModifiers)

    def to_array(self) -> np.ndarray:
        """Return 9D array of all modifiers."""
        return np.concatenate([
            self.epistemic.to_array(),
            self.temporal.to_array(),
            self.social.to_array(),
            self.emotional.to_array()
        ])

    def to_dict(self) -> dict:
        return {
            "epistemic": self.epistemic.to_dict(),
            "temporal": self.temporal.to_dict(),
            "social": self.social.to_dict(),
            "emotional": self.emotional.to_dict()
        }

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "CoordinationModifiers":
        """Create from 9D array."""
        return cls(
            epistemic=EpistemicModifiers(
                certainty=arr[0], evidentiality=arr[1], commitment=arr[2]
            ),
            temporal=TemporalModifiers(focus=arr[3], scope=arr[4]),
            social=SocialModifiers(power_differential=arr[5], social_distance=arr[6]),
            emotional=EmotionalModifiers(arousal=arr[7], valence=arr[8])
        )


class DecorativeFeature(Enum):
    """Features stripped during core extraction."""

    ARTICLES = "articles"           # a, an, the
    HEDGING = "hedging"             # maybe, perhaps, sort of
    INTENSIFIERS = "intensifiers"   # very, really, extremely
    FILLER = "filler"               # like, you know, basically
    STYLE_MARKERS = "style_markers" # formal/informal register


@dataclass
class DecorativeLayer:
    """
    Decorative features that don't affect coordination.

    These can vary without changing the base manifold position.
    """

    article_count: int = 0
    hedge_count: int = 0
    intensifier_count: int = 0
    filler_count: int = 0
    formality_score: float = 0.0  # -1 = informal, +1 = formal

    def to_dict(self) -> dict:
        return {
            "article_count": self.article_count,
            "hedge_count": self.hedge_count,
            "intensifier_count": self.intensifier_count,
            "filler_count": self.filler_count,
            "formality_score": self.formality_score
        }


@dataclass
class HierarchicalCoordinate:
    """
    Full hierarchical coordinate in the narrative manifold.

    Structure mirrors fiber bundle: E = B x F
    - core: Base manifold position (coordination substrate)
    - modifiers: Fiber at that base point
    - decorative: Additional features not affecting coordination
    """

    core: CoordinationCore = field(default_factory=CoordinationCore)
    modifiers: CoordinationModifiers = field(default_factory=CoordinationModifiers)
    decorative: DecorativeLayer = field(default_factory=DecorativeLayer)

    def to_full_array(self) -> np.ndarray:
        """Return 18D array: [core_9, modifiers_9]."""
        return np.concatenate([
            self.core.to_array(),
            self.modifiers.to_array()
        ])

    def to_dict(self) -> dict:
        return {
            "core": self.core.to_dict(),
            "modifiers": self.modifiers.to_dict(),
            "decorative": self.decorative.to_dict(),
            "legacy_3d": {
                "agency": self.core.agency.aggregate,
                "perceived_justice": self.core.justice.aggregate,
                "belonging": self.core.belonging.aggregate
            }
        }

    @classmethod
    def from_legacy_3d(
        cls,
        agency: float,
        justice: float,
        belonging: float
    ) -> "HierarchicalCoordinate":
        """Create from legacy 3D coordinates."""
        return cls(
            core=CoordinationCore.from_legacy_3d(agency, justice, belonging)
        )


# =============================================================================
# Feature Detection Patterns
# =============================================================================

# Patterns for detecting linguistic features
FEATURE_PATTERNS = {
    # Agency markers - pronouns
    "first_person_singular": r"\b(I|me|my|mine|myself)\b",
    "first_person_plural": r"\b(we|us|our|ours|ourselves)\b",
    "second_person": r"\b(you|your|yours|yourself|yourselves)\b",
    "third_person": r"\b(he|she|they|them|his|her|their|theirs|it|its)\b",

    # Agency markers - ENHANCED: volitional verbs (high self-agency)
    "self_agency_volitional": r"\b(I (chose|decided|determined|created|initiated|made|started|built|accomplished)|my (choice|decision|doing|initiative|effort)|took (control|charge|initiative|action)|on my own|by myself)\b",

    # Agency markers - ENHANCED: control/autonomy language
    "self_agency_control": r"\b(I can|I will|I'm able|in control|my responsibility|I own|I lead|I manage|I direct)\b",

    # Agency markers - ENHANCED: helplessness/low agency
    "low_agency_markers": r"\b((forced|had|made|required|compelled) to|no (choice|control|say|option)|beyond my control|nothing I (can|could) do|powerless|helpless|victim of|at the mercy)\b",

    # System agency markers - ENHANCED
    "system_agency": r"\b(system|institution|government|authority|force|circumstance|structure|policy|bureaucracy|organization|corporation|society|the powers|external forces)\b",

    # Justice markers - ENHANCED: procedural with fairness polarity
    "procedural_justice": r"\b(fair (process|procedure|hearing|trial|review)|due process|proper (review|procedure|hearing)|transparent|impartial|unbiased|consistent (procedures?|rules?)|equal (access|opportunity)|right to (appeal|review|hearing))\b",
    "procedural_injustice": r"\b(unfair (process|procedure|hearing|trial)|rigged|biased|corrupt|kangaroo court|fixed|no (appeal|recourse|due process)|denied (hearing|review|appeal))\b",

    # Justice markers - ENHANCED: distributive with equity/equality
    "distributive_justice": r"\b(fair (share|distribution|outcome)|deserve[ds]?|earned?|merit|equal (pay|treatment|share)|proportional|equitable|just (reward|outcome)|what('s| is) fair)\b",
    "distributive_injustice": r"\b(unfair (share|distribution|outcome)|undeserved|unearned|inequity|inequality|disproportionate|unjust|cheated|robbed|exploited)\b",

    # Justice markers - ENHANCED: interactional
    "interactional_justice": r"\b(respect(ed|ful)?|dignity|treat(ed)? (fairly|well|with respect)|listen(ed)? to|heard|acknowledge[d]?|recognize[d]?|valued|appreciated|considerate)\b",
    "interactional_injustice": r"\b(disrespect(ed)?|undignified|treat(ed)? (badly|poorly|unfairly)|ignored|dismissed|dehumanized|degraded|humiliated|belittled)\b",

    # Belonging markers
    "ingroup": r"\b(we|us|our|together|community|team|family|tribe)\b",
    "outgroup": r"\b(them|they|those people|outsiders|others|enemies)\b",
    "universal": r"\b(everyone|everybody|humanity|people|human|universal|all of us)\b",

    # Epistemic markers
    "certainty_high": r"\b(definitely|certainly|absolutely|clearly|obviously|undoubtedly)\b",
    "certainty_low": r"\b(maybe|perhaps|possibly|might|could|uncertain|unclear)\b",
    "evidential_direct": r"\b(I saw|I heard|I felt|I experienced|witnessed)\b",
    "evidential_reported": r"\b(they say|apparently|supposedly|allegedly|reportedly)\b",

    # Temporal markers
    "past_focus": r"\b(was|were|had|did|used to|back then|before|yesterday|ago)\b",
    "future_focus": r"\b(will|shall|going to|tomorrow|soon|eventually|future|ahead)\b",

    # Emotional markers
    "high_arousal": r"\b(excited|angry|terrified|thrilled|furious|ecstatic|enraged)\b",
    "low_arousal": r"\b(calm|peaceful|relaxed|serene|bored|tired|content)\b",
    "positive_valence": r"\b(happy|joy|love|wonderful|great|excellent|beautiful)\b",
    "negative_valence": r"\b(sad|hate|terrible|awful|horrible|disgusting|painful)\b",

    # Decorative features
    "articles": r"\b(a|an|the)\b",
    "hedging": r"\b(maybe|perhaps|sort of|kind of|somewhat|rather|quite)\b",
    "intensifiers": r"\b(very|really|extremely|absolutely|totally|completely|utterly)\b",
    "filler": r"\b(like|you know|basically|actually|literally|I mean|well)\b",

    # Social markers
    "formal_register": r"\b(therefore|hence|consequently|furthermore|moreover|hereby)\b",
    "informal_register": r"\b(gonna|wanna|kinda|gotta|dunno|yeah|nope|cool)\b",
}


# =============================================================================
# Feature Extraction
# =============================================================================

def count_pattern_matches(text: str, pattern: str) -> int:
    """Count regex pattern matches in text."""
    return len(re.findall(pattern, text, re.IGNORECASE))


def extract_features(text: str) -> Dict[str, int]:
    """Extract all feature counts from text."""
    features = {}
    for name, pattern in FEATURE_PATTERNS.items():
        features[name] = count_pattern_matches(text, pattern)
    return features


def extract_hierarchical_coordinate(
    text: str,
    base_embedding: Optional[np.ndarray] = None
) -> HierarchicalCoordinate:
    """
    Extract hierarchical coordinate from text using linguistic patterns.

    This is a rule-based extraction. For production use, combine with
    embedding-based projection.

    Args:
        text: Input text to analyze
        base_embedding: Optional embedding for hybrid extraction

    Returns:
        HierarchicalCoordinate with all dimensions populated
    """
    features = extract_features(text)
    word_count = len(text.split()) + 1  # Avoid division by zero

    # Normalize counts by text length
    # FIXED: Returns 0.0 for no matches (neutral), scales to [-1, 1]
    def norm(count: int) -> float:
        if count == 0:
            return 0.0
        scaled = count / max(word_count / 10, 1.0)
        return max(-1.0, min(scaled, 1.0))

    # Agency decomposition - ENHANCED with volitional verbs and control language
    # High self-agency: pronouns + volitional verbs + control language - helplessness
    self_agency_positive = (
        features["first_person_singular"] +
        features.get("self_agency_volitional", 0) * 2 +  # Weight volitional verbs
        features.get("self_agency_control", 0) * 1.5
    )
    self_agency_negative = features.get("low_agency_markers", 0) * 2
    self_agency_net = self_agency_positive - self_agency_negative

    agency = AgencyDecomposition(
        self_agency=norm(self_agency_net),
        other_agency=norm(features["second_person"] + features["third_person"]),
        system_agency=norm(features["system_agency"])
    )

    # Justice decomposition - ENHANCED with polarity awareness
    # Procedural: positive - negative
    procedural_net = (
        features["procedural_justice"] -
        features.get("procedural_injustice", 0)
    )
    # Distributive: positive - negative
    distributive_net = (
        features["distributive_justice"] -
        features.get("distributive_injustice", 0)
    )
    # Interactional: positive - negative
    interactional_net = (
        features["interactional_justice"] -
        features.get("interactional_injustice", 0)
    )

    justice = JusticeDecomposition(
        procedural=norm(procedural_net),
        distributive=norm(distributive_net),
        interactional=norm(interactional_net)
    )

    # Belonging decomposition
    belonging = BelongingDecomposition(
        ingroup=norm(features["ingroup"]),
        outgroup=norm(features["outgroup"]),
        universal=norm(features["universal"])
    )

    # Epistemic modifiers
    epistemic = EpistemicModifiers(
        certainty=norm(features["certainty_high"]) - norm(features["certainty_low"]),
        evidentiality=norm(features["evidential_direct"]) - norm(features["evidential_reported"]),
        commitment=0.0  # Requires deeper analysis
    )

    # Temporal modifiers
    temporal = TemporalModifiers(
        focus=norm(features["future_focus"]) - norm(features["past_focus"]),
        scope=0.0  # Requires context analysis
    )

    # Social modifiers
    social = SocialModifiers(
        power_differential=0.0,  # Requires context
        social_distance=norm(features["formal_register"]) - norm(features["informal_register"])
    )

    # Emotional modifiers
    emotional = EmotionalModifiers(
        arousal=norm(features["high_arousal"]) - norm(features["low_arousal"]),
        valence=norm(features["positive_valence"]) - norm(features["negative_valence"])
    )

    # Decorative layer
    decorative = DecorativeLayer(
        article_count=features["articles"],
        hedge_count=features["hedging"],
        intensifier_count=features["intensifiers"],
        filler_count=features["filler"],
        formality_score=norm(features["formal_register"]) - norm(features["informal_register"])
    )

    return HierarchicalCoordinate(
        core=CoordinationCore(agency=agency, justice=justice, belonging=belonging),
        modifiers=CoordinationModifiers(
            epistemic=epistemic,
            temporal=temporal,
            social=social,
            emotional=emotional
        ),
        decorative=decorative
    )


# =============================================================================
# Fiber Bundle Operations
# =============================================================================

def project_to_base(coord: HierarchicalCoordinate) -> np.ndarray:
    """
    Project full coordinate to base manifold (strip fiber).

    This is the projection map pi: E -> B in the fiber bundle.
    """
    return coord.core.to_array()


def project_to_fiber(coord: HierarchicalCoordinate) -> np.ndarray:
    """
    Extract fiber coordinates at the base point.
    """
    return coord.modifiers.to_array()


def reconstruct_from_bundle(
    base: np.ndarray,
    fiber: np.ndarray
) -> HierarchicalCoordinate:
    """
    Reconstruct full coordinate from base + fiber.

    This is the inverse of the projection (local trivialization).
    """
    return HierarchicalCoordinate(
        core=CoordinationCore.from_array(base),
        modifiers=CoordinationModifiers.from_array(fiber)
    )


def parallel_transport(
    coord: HierarchicalCoordinate,
    target_base: np.ndarray
) -> HierarchicalCoordinate:
    """
    Transport fiber to a new base position.

    In a flat connection, the fiber stays constant during transport.
    More sophisticated connections could modify the fiber based on curvature.
    """
    return HierarchicalCoordinate(
        core=CoordinationCore.from_array(target_base),
        modifiers=coord.modifiers,
        decorative=coord.decorative
    )


def compute_bundle_distance(
    coord1: HierarchicalCoordinate,
    coord2: HierarchicalCoordinate,
    base_weight: float = 0.7,
    fiber_weight: float = 0.3
) -> float:
    """
    Compute distance in the total bundle space.

    Args:
        coord1, coord2: Coordinates to compare
        base_weight: Weight for base manifold distance (default 0.7)
        fiber_weight: Weight for fiber distance (default 0.3)

    Returns:
        Weighted distance in bundle
    """
    base_dist = np.linalg.norm(coord1.core.to_array() - coord2.core.to_array())
    fiber_dist = np.linalg.norm(coord1.modifiers.to_array() - coord2.modifiers.to_array())

    return base_weight * base_dist + fiber_weight * fiber_dist


# =============================================================================
# Coordination Necessity Classification
# =============================================================================

@dataclass
class FeatureNecessityResult:
    """Result of testing whether a feature is coordination-necessary."""

    feature_name: str
    drift_magnitude: float              # How much base coords change when feature removed
    is_necessary: bool                  # True if drift > threshold
    classification: str                 # "decorative", "modifying", "necessary", "critical"
    confidence: float                   # Statistical confidence

    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "drift_magnitude": self.drift_magnitude,
            "is_necessary": self.is_necessary,
            "classification": self.classification,
            "confidence": self.confidence
        }


def classify_feature_necessity(
    drift_magnitude: float,
    thresholds: Dict[str, float] = None
) -> str:
    """
    Classify feature based on coordination drift magnitude.

    Default thresholds based on experimental findings:
    - decorative: drift < 0.1
    - modifying: 0.1 <= drift < 0.3
    - necessary: 0.3 <= drift < 0.5
    - critical: drift >= 0.5
    """
    if thresholds is None:
        thresholds = {
            "decorative": 0.1,
            "modifying": 0.3,
            "necessary": 0.5
        }

    if drift_magnitude < thresholds["decorative"]:
        return "decorative"
    elif drift_magnitude < thresholds["modifying"]:
        return "modifying"
    elif drift_magnitude < thresholds["necessary"]:
        return "necessary"
    else:
        return "critical"


# =============================================================================
# Dimensionality Reduction for Visualization
# =============================================================================

def reduce_to_3d(
    coord: HierarchicalCoordinate,
    method: str = "aggregate"
) -> Tuple[float, float, float]:
    """
    Reduce hierarchical coordinate to 3D for visualization.

    Args:
        coord: Full hierarchical coordinate
        method:
            - "aggregate": Use aggregate scores (default)
            - "pca": Use PCA on full 18D
            - "umap": Use UMAP projection

    Returns:
        Tuple of (agency, justice, belonging) in [-1, 1]
    """
    if method == "aggregate":
        return coord.core.to_legacy_3d()
    elif method == "pca":
        # Would require fitted PCA model
        logger.warning("PCA reduction not implemented, using aggregate")
        return coord.core.to_legacy_3d()
    elif method == "umap":
        # Would require fitted UMAP model
        logger.warning("UMAP reduction not implemented, using aggregate")
        return coord.core.to_legacy_3d()
    else:
        raise ValueError(f"Unknown reduction method: {method}")


def batch_reduce_to_3d(
    coords: List[HierarchicalCoordinate],
    method: str = "aggregate"
) -> np.ndarray:
    """
    Reduce multiple coordinates to 3D array.

    Returns:
        (N, 3) array of reduced coordinates
    """
    return np.array([reduce_to_3d(c, method) for c in coords])
