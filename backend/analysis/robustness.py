"""
Adversarial Robustness Testing for Projections

Tests how stable projection results are under various text perturbations.
Identifies minimal edits that can flip narrative modes and computes
overall robustness scores.
"""

import re
import asyncio
from enum import Enum
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class PerturbationType(str, Enum):
    """Types of text perturbations for robustness testing."""
    SYNONYM = "synonym"           # Replace words with synonyms
    NEGATION = "negation"         # Add/remove negation ("fair" <-> "not unfair")
    TENSE = "tense"               # Change verb tense (past/present/future)
    VOICE = "voice"               # Active <-> passive voice
    HEDGING = "hedging"           # Add/remove hedges ("will" <-> "might")
    INTENSIFIER = "intensifier"   # Add/remove intensifiers ("good" <-> "very good")


@dataclass
class Perturbation:
    """A single perturbation applied to text."""
    original: str
    perturbed: str
    perturbation_type: PerturbationType
    description: str


@dataclass
class PerturbationResult:
    """Result of projecting a perturbed text."""
    perturbation: Perturbation
    original_projection: Dict[str, Any]
    perturbed_projection: Dict[str, Any]
    delta: Dict[str, float]  # Change per axis
    mode_flipped: bool

    def to_dict(self) -> dict:
        return {
            "original_text": self.perturbation.original,
            "perturbed_text": self.perturbation.perturbed,
            "perturbation_type": self.perturbation.perturbation_type.value,
            "description": self.perturbation.description,
            "original_projection": self.original_projection,
            "perturbed_projection": self.perturbed_projection,
            "delta": self.delta,
            "mode_flipped": self.mode_flipped
        }


@dataclass
class RobustnessReport:
    """Full robustness test report for a text."""
    original_text: str
    original_projection: Dict[str, Any]
    perturbation_results: List[PerturbationResult]
    robustness_score: float
    mode_flip_count: int
    most_sensitive_perturbation: Optional[PerturbationResult]

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "original_projection": self.original_projection,
            "perturbation_results": [r.to_dict() for r in self.perturbation_results],
            "robustness_score": self.robustness_score,
            "mode_flip_count": self.mode_flip_count,
            "most_sensitive_perturbation": (
                self.most_sensitive_perturbation.to_dict()
                if self.most_sensitive_perturbation else None
            )
        }


@dataclass
class ModeFlipResult:
    """Result of finding minimal edit to flip mode."""
    original_text: str
    original_mode: str
    target_mode: str
    found: bool
    flipped_text: Optional[str] = None
    perturbation_type: Optional[PerturbationType] = None
    edit_description: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "original_mode": self.original_mode,
            "target_mode": self.target_mode,
            "found": self.found,
            "flipped_text": self.flipped_text,
            "perturbation_type": self.perturbation_type.value if self.perturbation_type else None,
            "edit_description": self.edit_description
        }


class RobustnessTester:
    """
    Tests adversarial robustness of text projections.

    Applies various perturbations to input text and measures how
    the projection changes, identifying vulnerabilities where small
    edits cause large shifts in the projected space.
    """

    # Synonym mappings for common words
    SYNONYMS = {
        # Positive/negative pairs
        "good": ["fine", "great", "excellent", "positive", "favorable"],
        "bad": ["poor", "terrible", "negative", "unfavorable", "awful"],
        "happy": ["joyful", "pleased", "content", "delighted", "glad"],
        "sad": ["unhappy", "sorrowful", "melancholy", "dejected", "gloomy"],
        "fair": ["just", "equitable", "impartial", "balanced", "reasonable"],
        "unfair": ["unjust", "biased", "inequitable", "partial", "unreasonable"],
        # Action words
        "help": ["assist", "aid", "support", "facilitate", "enable"],
        "harm": ["hurt", "damage", "injure", "impair", "hinder"],
        "create": ["make", "build", "produce", "generate", "construct"],
        "destroy": ["demolish", "ruin", "eliminate", "annihilate", "wreck"],
        # People words
        "people": ["individuals", "persons", "humans", "folks", "citizens"],
        "community": ["group", "society", "collective", "population", "neighborhood"],
        "everyone": ["everybody", "all", "each person", "all people", "the public"],
        # Belonging words
        "belong": ["fit in", "be part of", "be included", "be accepted", "be welcomed"],
        "excluded": ["left out", "marginalized", "rejected", "ostracized", "isolated"],
        # Agency words
        "choose": ["decide", "select", "opt", "pick", "determine"],
        "control": ["manage", "direct", "govern", "regulate", "command"],
        "power": ["authority", "influence", "capability", "strength", "force"],
        # Common verbs
        "think": ["believe", "consider", "feel", "suppose", "reckon"],
        "want": ["desire", "wish", "hope for", "seek", "crave"],
        "need": ["require", "must have", "depend on", "necessitate"],
        "make": ["create", "produce", "generate", "cause", "form"],
        "give": ["provide", "offer", "grant", "supply", "bestow"],
        "take": ["grab", "seize", "obtain", "acquire", "receive"],
        # Adjectives
        "important": ["significant", "crucial", "vital", "essential", "key"],
        "big": ["large", "huge", "massive", "substantial", "major"],
        "small": ["little", "tiny", "minor", "slight", "modest"],
        "new": ["recent", "fresh", "novel", "modern", "contemporary"],
        "old": ["ancient", "former", "previous", "past", "traditional"],
    }

    # Negation patterns
    NEGATION_PAIRS = [
        (r"\bis\b", "is not"),
        (r"\bis not\b", "is"),
        (r"\bisn't\b", "is"),
        (r"\bare\b", "are not"),
        (r"\bare not\b", "are"),
        (r"\baren't\b", "are"),
        (r"\bwas\b", "was not"),
        (r"\bwas not\b", "was"),
        (r"\bwasn't\b", "was"),
        (r"\bwere\b", "were not"),
        (r"\bwere not\b", "were"),
        (r"\bweren't\b", "were"),
        (r"\bwill\b", "will not"),
        (r"\bwill not\b", "will"),
        (r"\bwon't\b", "will"),
        (r"\bcan\b", "cannot"),
        (r"\bcannot\b", "can"),
        (r"\bcan't\b", "can"),
        (r"\bdo\b", "do not"),
        (r"\bdo not\b", "do"),
        (r"\bdon't\b", "do"),
        (r"\bdoes\b", "does not"),
        (r"\bdoes not\b", "does"),
        (r"\bdoesn't\b", "does"),
        (r"\bhave\b", "have not"),
        (r"\bhave not\b", "have"),
        (r"\bhaven't\b", "have"),
        (r"\bhas\b", "has not"),
        (r"\bhas not\b", "has"),
        (r"\bhasn't\b", "has"),
        (r"\bshould\b", "should not"),
        (r"\bshould not\b", "should"),
        (r"\bshouldn't\b", "should"),
    ]

    # Double negative patterns ("not unfair" <-> "fair")
    DOUBLE_NEGATION_PAIRS = [
        ("fair", "not unfair"),
        ("not unfair", "fair"),
        ("good", "not bad"),
        ("not bad", "good"),
        ("happy", "not unhappy"),
        ("not unhappy", "happy"),
        ("included", "not excluded"),
        ("not excluded", "included"),
        ("equal", "not unequal"),
        ("not unequal", "equal"),
        ("just", "not unjust"),
        ("not unjust", "just"),
        ("possible", "not impossible"),
        ("not impossible", "possible"),
        ("likely", "not unlikely"),
        ("not unlikely", "likely"),
    ]

    # Tense conversion patterns
    TENSE_PATTERNS = {
        # Simple present -> past
        "present_to_past": [
            (r"\b(I|you|we|they)\s+go\b", r"\1 went"),
            (r"\b(he|she|it)\s+goes\b", r"\1 went"),
            (r"\b(I|you|we|they)\s+do\b", r"\1 did"),
            (r"\b(he|she|it)\s+does\b", r"\1 did"),
            (r"\b(I|you|we|they)\s+have\b", r"\1 had"),
            (r"\b(he|she|it)\s+has\b", r"\1 had"),
            (r"\b(I|you|we|they)\s+make\b", r"\1 made"),
            (r"\b(he|she|it)\s+makes\b", r"\1 made"),
            (r"\b(I|you|we|they)\s+take\b", r"\1 took"),
            (r"\b(he|she|it)\s+takes\b", r"\1 took"),
            (r"\b(I|you|we|they)\s+give\b", r"\1 gave"),
            (r"\b(he|she|it)\s+gives\b", r"\1 gave"),
            (r"\b(I|you|we|they)\s+come\b", r"\1 came"),
            (r"\b(he|she|it)\s+comes\b", r"\1 came"),
            (r"\b(I|you|we|they)\s+see\b", r"\1 saw"),
            (r"\b(he|she|it)\s+sees\b", r"\1 saw"),
            (r"\b(I|you|we|they)\s+know\b", r"\1 knew"),
            (r"\b(he|she|it)\s+knows\b", r"\1 knew"),
            (r"\b(I|you|we|they)\s+think\b", r"\1 thought"),
            (r"\b(he|she|it)\s+thinks\b", r"\1 thought"),
            (r"\b(I|you|we|they)\s+feel\b", r"\1 felt"),
            (r"\b(he|she|it)\s+feels\b", r"\1 felt"),
            (r"\b(I|you|we|they)\s+say\b", r"\1 said"),
            (r"\b(he|she|it)\s+says\b", r"\1 said"),
            (r"\bis\b", "was"),
            (r"\bare\b", "were"),
        ],
        # Present -> future
        "present_to_future": [
            (r"\b(I|you|we|they|he|she|it)\s+(\w+)s?\b", r"\1 will \2"),
            (r"\bis\b", "will be"),
            (r"\bare\b", "will be"),
        ],
        # Past -> present (reverse of present_to_past)
        "past_to_present": [
            (r"\bwent\b", "go"),
            (r"\bdid\b", "do"),
            (r"\bhad\b", "have"),
            (r"\bmade\b", "make"),
            (r"\btook\b", "take"),
            (r"\bgave\b", "give"),
            (r"\bcame\b", "come"),
            (r"\bsaw\b", "see"),
            (r"\bknew\b", "know"),
            (r"\bthought\b", "think"),
            (r"\bfelt\b", "feel"),
            (r"\bsaid\b", "say"),
            (r"\bwas\b", "is"),
            (r"\bwere\b", "are"),
        ],
    }

    # Active to passive patterns
    VOICE_PATTERNS = {
        "active_to_passive": [
            # "X does Y" -> "Y is done by X"
            (r"(\w+)\s+(creates?|builds?|makes?)\s+(\w+)", r"\3 is created by \1"),
            (r"(\w+)\s+(helps?|assists?|supports?)\s+(\w+)", r"\3 is helped by \1"),
            (r"(\w+)\s+(gives?|provides?|offers?)\s+(\w+)", r"\3 is given by \1"),
            (r"(\w+)\s+(teaches?|educates?|trains?)\s+(\w+)", r"\3 is taught by \1"),
            (r"(\w+)\s+(loves?|likes?|appreciates?)\s+(\w+)", r"\3 is loved by \1"),
        ],
        "passive_to_active": [
            # "Y is done by X" -> "X does Y"
            (r"(\w+)\s+is\s+created\s+by\s+(\w+)", r"\2 creates \1"),
            (r"(\w+)\s+is\s+helped\s+by\s+(\w+)", r"\2 helps \1"),
            (r"(\w+)\s+is\s+given\s+by\s+(\w+)", r"\2 gives \1"),
            (r"(\w+)\s+is\s+taught\s+by\s+(\w+)", r"\2 teaches \1"),
            (r"(\w+)\s+is\s+loved\s+by\s+(\w+)", r"\2 loves \1"),
        ],
    }

    # Hedging words
    HEDGES = {
        "add": [
            ("will", "might"),
            ("will", "may"),
            ("will", "could"),
            ("will", "possibly will"),
            ("is", "might be"),
            ("is", "may be"),
            ("is", "could be"),
            ("are", "might be"),
            ("are", "may be"),
            ("can", "might"),
            ("must", "should"),
            ("always", "often"),
            ("never", "rarely"),
            ("certainly", "probably"),
            ("definitely", "likely"),
        ],
        "remove": [
            ("might", "will"),
            ("may", "will"),
            ("could", "will"),
            ("possibly will", "will"),
            ("might be", "is"),
            ("may be", "is"),
            ("could be", "is"),
            ("should", "must"),
            ("often", "always"),
            ("rarely", "never"),
            ("probably", "certainly"),
            ("likely", "definitely"),
            ("perhaps", ""),
            ("maybe", ""),
            ("possibly", ""),
        ],
    }

    # Intensifiers
    INTENSIFIERS = {
        "add": [
            ("good", "very good"),
            ("bad", "very bad"),
            ("important", "extremely important"),
            ("helpful", "incredibly helpful"),
            ("fair", "completely fair"),
            ("unfair", "completely unfair"),
            ("happy", "very happy"),
            ("sad", "very sad"),
            ("powerful", "extremely powerful"),
            ("weak", "extremely weak"),
            ("strong", "incredibly strong"),
            ("clear", "crystal clear"),
            ("sure", "absolutely sure"),
            ("different", "completely different"),
            ("same", "exactly the same"),
        ],
        "remove": [
            ("very good", "good"),
            ("very bad", "bad"),
            ("extremely important", "important"),
            ("incredibly helpful", "helpful"),
            ("completely fair", "fair"),
            ("completely unfair", "unfair"),
            ("very happy", "happy"),
            ("very sad", "sad"),
            ("extremely powerful", "powerful"),
            ("extremely weak", "weak"),
            ("incredibly strong", "strong"),
            ("crystal clear", "clear"),
            ("absolutely sure", "sure"),
            ("completely different", "different"),
            ("exactly the same", "same"),
            ("really ", ""),
            ("truly ", ""),
            ("absolutely ", ""),
            ("completely ", ""),
            ("extremely ", ""),
            ("incredibly ", ""),
            ("very ", ""),
        ],
    }

    def __init__(self):
        """Initialize the robustness tester."""
        pass

    def generate_perturbations(
        self,
        text: str,
        types: Optional[List[PerturbationType]] = None
    ) -> List[Perturbation]:
        """
        Generate perturbed versions of the input text.

        Args:
            text: Original text to perturb
            types: List of perturbation types to apply (all if None)

        Returns:
            List of Perturbation objects with different modifications
        """
        if types is None:
            types = list(PerturbationType)

        perturbations = []

        for ptype in types:
            if ptype == PerturbationType.SYNONYM:
                perturbations.extend(self._generate_synonym_perturbations(text))
            elif ptype == PerturbationType.NEGATION:
                perturbations.extend(self._generate_negation_perturbations(text))
            elif ptype == PerturbationType.TENSE:
                perturbations.extend(self._generate_tense_perturbations(text))
            elif ptype == PerturbationType.VOICE:
                perturbations.extend(self._generate_voice_perturbations(text))
            elif ptype == PerturbationType.HEDGING:
                perturbations.extend(self._generate_hedging_perturbations(text))
            elif ptype == PerturbationType.INTENSIFIER:
                perturbations.extend(self._generate_intensifier_perturbations(text))

        # Remove duplicates and perturbations that didn't change the text
        seen = set()
        unique_perturbations = []
        for p in perturbations:
            if p.perturbed != text and p.perturbed not in seen:
                seen.add(p.perturbed)
                unique_perturbations.append(p)

        logger.info(f"Generated {len(unique_perturbations)} unique perturbations for text")
        return unique_perturbations

    def _generate_synonym_perturbations(self, text: str) -> List[Perturbation]:
        """Generate perturbations by replacing words with synonyms."""
        perturbations = []
        text_lower = text.lower()

        for word, synonyms in self.SYNONYMS.items():
            if re.search(rf"\b{word}\b", text_lower):
                for synonym in synonyms[:3]:  # Limit to 3 synonyms per word
                    # Replace preserving case
                    perturbed = re.sub(
                        rf"\b{word}\b",
                        synonym,
                        text,
                        flags=re.IGNORECASE,
                        count=1
                    )
                    if perturbed != text:
                        perturbations.append(Perturbation(
                            original=text,
                            perturbed=perturbed,
                            perturbation_type=PerturbationType.SYNONYM,
                            description=f"Replaced '{word}' with '{synonym}'"
                        ))

        return perturbations

    def _generate_negation_perturbations(self, text: str) -> List[Perturbation]:
        """Generate perturbations by adding/removing negation."""
        perturbations = []

        # Standard negation toggles
        for pattern, replacement in self.NEGATION_PAIRS:
            if re.search(pattern, text, re.IGNORECASE):
                perturbed = re.sub(pattern, replacement, text, flags=re.IGNORECASE, count=1)
                if perturbed != text:
                    perturbations.append(Perturbation(
                        original=text,
                        perturbed=perturbed,
                        perturbation_type=PerturbationType.NEGATION,
                        description=f"Toggled negation: '{pattern}' -> '{replacement}'"
                    ))

        # Double negation replacements
        for original, double_neg in self.DOUBLE_NEGATION_PAIRS:
            if original in text.lower():
                perturbed = re.sub(
                    rf"\b{original}\b",
                    double_neg,
                    text,
                    flags=re.IGNORECASE,
                    count=1
                )
                if perturbed != text:
                    perturbations.append(Perturbation(
                        original=text,
                        perturbed=perturbed,
                        perturbation_type=PerturbationType.NEGATION,
                        description=f"Double negation: '{original}' -> '{double_neg}'"
                    ))

        return perturbations

    def _generate_tense_perturbations(self, text: str) -> List[Perturbation]:
        """Generate perturbations by changing verb tense."""
        perturbations = []

        for tense_change, patterns in self.TENSE_PATTERNS.items():
            perturbed = text
            applied = False

            for pattern, replacement in patterns:
                new_text = re.sub(pattern, replacement, perturbed, flags=re.IGNORECASE, count=1)
                if new_text != perturbed:
                    perturbed = new_text
                    applied = True
                    break  # Apply only one change per tense type

            if applied and perturbed != text:
                perturbations.append(Perturbation(
                    original=text,
                    perturbed=perturbed,
                    perturbation_type=PerturbationType.TENSE,
                    description=f"Changed tense: {tense_change}"
                ))

        return perturbations

    def _generate_voice_perturbations(self, text: str) -> List[Perturbation]:
        """Generate perturbations by switching active/passive voice."""
        perturbations = []

        for voice_change, patterns in self.VOICE_PATTERNS.items():
            for pattern, replacement in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    perturbed = re.sub(pattern, replacement, text, flags=re.IGNORECASE, count=1)
                    if perturbed != text:
                        perturbations.append(Perturbation(
                            original=text,
                            perturbed=perturbed,
                            perturbation_type=PerturbationType.VOICE,
                            description=f"Changed voice: {voice_change}"
                        ))

        return perturbations

    def _generate_hedging_perturbations(self, text: str) -> List[Perturbation]:
        """Generate perturbations by adding/removing hedging language."""
        perturbations = []

        # Add hedges
        for original, hedged in self.HEDGES["add"]:
            if re.search(rf"\b{original}\b", text, re.IGNORECASE):
                perturbed = re.sub(
                    rf"\b{original}\b",
                    hedged,
                    text,
                    flags=re.IGNORECASE,
                    count=1
                )
                if perturbed != text:
                    perturbations.append(Perturbation(
                        original=text,
                        perturbed=perturbed,
                        perturbation_type=PerturbationType.HEDGING,
                        description=f"Added hedge: '{original}' -> '{hedged}'"
                    ))

        # Remove hedges
        for hedged, original in self.HEDGES["remove"]:
            if hedged in text.lower():
                perturbed = re.sub(
                    rf"\b{hedged}\b" if hedged.strip() else hedged,
                    original,
                    text,
                    flags=re.IGNORECASE,
                    count=1
                )
                if perturbed != text:
                    perturbations.append(Perturbation(
                        original=text,
                        perturbed=perturbed,
                        perturbation_type=PerturbationType.HEDGING,
                        description=f"Removed hedge: '{hedged}' -> '{original}'"
                    ))

        return perturbations

    def _generate_intensifier_perturbations(self, text: str) -> List[Perturbation]:
        """Generate perturbations by adding/removing intensifiers."""
        perturbations = []

        # Add intensifiers
        for original, intensified in self.INTENSIFIERS["add"]:
            if re.search(rf"\b{original}\b", text, re.IGNORECASE):
                # Make sure we're not already intensified
                if intensified.lower() not in text.lower():
                    perturbed = re.sub(
                        rf"\b{original}\b",
                        intensified,
                        text,
                        flags=re.IGNORECASE,
                        count=1
                    )
                    if perturbed != text:
                        perturbations.append(Perturbation(
                            original=text,
                            perturbed=perturbed,
                            perturbation_type=PerturbationType.INTENSIFIER,
                            description=f"Added intensifier: '{original}' -> '{intensified}'"
                        ))

        # Remove intensifiers
        for intensified, original in self.INTENSIFIERS["remove"]:
            if intensified.lower() in text.lower():
                perturbed = re.sub(
                    rf"\b{intensified}\b" if intensified.strip() else intensified,
                    original,
                    text,
                    flags=re.IGNORECASE,
                    count=1
                )
                if perturbed != text:
                    perturbations.append(Perturbation(
                        original=text,
                        perturbed=perturbed,
                        perturbation_type=PerturbationType.INTENSIFIER,
                        description=f"Removed intensifier: '{intensified}' -> '{original}'"
                    ))

        return perturbations

    async def test_robustness(
        self,
        text: str,
        projection_fn: Callable[[str], Any],
        types: Optional[List[PerturbationType]] = None
    ) -> RobustnessReport:
        """
        Test robustness of projection against all perturbations.

        Args:
            text: Original text to test
            projection_fn: Async function that projects text to vector/mode
            types: Perturbation types to test (all if None)

        Returns:
            RobustnessReport with full analysis
        """
        # Get original projection
        original_projection = await projection_fn(text)
        original_mode = original_projection.get("mode", "")
        original_vector = original_projection.get("vector", {})

        # Generate perturbations
        perturbations = self.generate_perturbations(text, types)

        if not perturbations:
            logger.warning("No perturbations generated for text")
            return RobustnessReport(
                original_text=text,
                original_projection=original_projection,
                perturbation_results=[],
                robustness_score=1.0,
                mode_flip_count=0,
                most_sensitive_perturbation=None
            )

        # Test each perturbation
        results = []
        mode_flips = 0
        max_delta_magnitude = 0.0
        most_sensitive = None

        for perturbation in perturbations:
            perturbed_projection = await projection_fn(perturbation.perturbed)
            perturbed_mode = perturbed_projection.get("mode", "")
            perturbed_vector = perturbed_projection.get("vector", {})

            # Calculate delta per axis
            delta = {}
            delta_magnitude = 0.0
            for axis in ["agency", "fairness", "belonging"]:
                orig_val = original_vector.get(axis, 0.0)
                pert_val = perturbed_vector.get(axis, 0.0)
                delta[axis] = pert_val - orig_val
                delta_magnitude += delta[axis] ** 2
            delta_magnitude = delta_magnitude ** 0.5

            mode_flipped = original_mode != perturbed_mode
            if mode_flipped:
                mode_flips += 1

            result = PerturbationResult(
                perturbation=perturbation,
                original_projection=original_projection,
                perturbed_projection=perturbed_projection,
                delta=delta,
                mode_flipped=mode_flipped
            )
            results.append(result)

            # Track most sensitive perturbation
            if delta_magnitude > max_delta_magnitude:
                max_delta_magnitude = delta_magnitude
                most_sensitive = result

        # Calculate robustness score
        # Based on: (1) how many mode flips, (2) average delta magnitude
        if results:
            mode_stability = 1.0 - (mode_flips / len(results))
            avg_delta = sum(
                sum(abs(d) for d in r.delta.values()) / 3.0
                for r in results
            ) / len(results)
            vector_stability = 1.0 / (1.0 + avg_delta)
            robustness_score = 0.5 * mode_stability + 0.5 * vector_stability
        else:
            robustness_score = 1.0

        return RobustnessReport(
            original_text=text,
            original_projection=original_projection,
            perturbation_results=results,
            robustness_score=robustness_score,
            mode_flip_count=mode_flips,
            most_sensitive_perturbation=most_sensitive
        )

    async def find_mode_flip(
        self,
        text: str,
        target_mode: str,
        projection_fn: Callable[[str], Any]
    ) -> ModeFlipResult:
        """
        Find minimal edit that flips the mode to a target.

        Args:
            text: Original text
            target_mode: Desired mode to flip to
            projection_fn: Async projection function

        Returns:
            ModeFlipResult indicating if a flip was found
        """
        # Get original mode
        original_projection = await projection_fn(text)
        original_mode = original_projection.get("mode", "")

        if original_mode == target_mode:
            return ModeFlipResult(
                original_text=text,
                original_mode=original_mode,
                target_mode=target_mode,
                found=True,
                flipped_text=text,
                edit_description="Already in target mode"
            )

        # Generate all perturbations
        perturbations = self.generate_perturbations(text)

        # Test each perturbation
        for perturbation in perturbations:
            perturbed_projection = await projection_fn(perturbation.perturbed)
            perturbed_mode = perturbed_projection.get("mode", "")

            if perturbed_mode == target_mode:
                return ModeFlipResult(
                    original_text=text,
                    original_mode=original_mode,
                    target_mode=target_mode,
                    found=True,
                    flipped_text=perturbation.perturbed,
                    perturbation_type=perturbation.perturbation_type,
                    edit_description=perturbation.description
                )

        # Try combinations of perturbations (up to 2 at a time)
        for i, p1 in enumerate(perturbations):
            for p2 in perturbations[i+1:]:
                # Apply second perturbation to first's result
                combined_text = p1.perturbed
                # Try applying p2's change pattern to combined_text
                combined_perturbs = self.generate_perturbations(
                    combined_text,
                    [p2.perturbation_type]
                )

                for combined in combined_perturbs:
                    combined_projection = await projection_fn(combined.perturbed)
                    combined_mode = combined_projection.get("mode", "")

                    if combined_mode == target_mode:
                        return ModeFlipResult(
                            original_text=text,
                            original_mode=original_mode,
                            target_mode=target_mode,
                            found=True,
                            flipped_text=combined.perturbed,
                            perturbation_type=p2.perturbation_type,
                            edit_description=f"{p1.description} + {combined.description}"
                        )

        return ModeFlipResult(
            original_text=text,
            original_mode=original_mode,
            target_mode=target_mode,
            found=False
        )

    async def robustness_score(
        self,
        text: str,
        projection_fn: Callable[[str], Any]
    ) -> float:
        """
        Calculate overall robustness score (0-1) for a text.

        Higher score = more robust (projection is stable under perturbations)

        Args:
            text: Text to evaluate
            projection_fn: Async projection function

        Returns:
            Float between 0 and 1
        """
        report = await self.test_robustness(text, projection_fn)
        return report.robustness_score


# Singleton instance
_robustness_tester = None


def get_robustness_tester() -> RobustnessTester:
    """Get singleton robustness tester instance."""
    global _robustness_tester
    if _robustness_tester is None:
        _robustness_tester = RobustnessTester()
    return _robustness_tester


# Demonstration test
async def _demo():
    """Demonstrate robustness testing functionality."""

    # Mock projection function that returns different results based on keywords
    async def mock_projection(text: str) -> dict:
        """Simple mock projection based on keyword matching."""
        text_lower = text.lower()

        # Calculate scores based on keywords
        agency = 0.0
        fairness = 0.0
        belonging = 0.0

        # Agency keywords
        if any(w in text_lower for w in ["choose", "decide", "control", "power", "can", "will"]):
            agency += 0.5
        if any(w in text_lower for w in ["cannot", "can't", "won't", "helpless", "forced"]):
            agency -= 0.5

        # Fairness keywords
        if any(w in text_lower for w in ["fair", "just", "equal", "equitable", "balanced"]):
            fairness += 0.5
        if any(w in text_lower for w in ["unfair", "unjust", "biased", "unequal"]):
            fairness -= 0.5

        # Belonging keywords
        if any(w in text_lower for w in ["belong", "included", "together", "community", "welcome"]):
            belonging += 0.5
        if any(w in text_lower for w in ["excluded", "alone", "isolated", "rejected", "outsider"]):
            belonging -= 0.5

        # Negation affects scores
        if "not " in text_lower or "n't " in text_lower:
            agency *= -0.5
            fairness *= -0.5
            belonging *= -0.5

        # Intensifiers boost scores
        if any(w in text_lower for w in ["very", "extremely", "incredibly", "completely"]):
            agency *= 1.5
            fairness *= 1.5
            belonging *= 1.5

        # Hedges reduce scores
        if any(w in text_lower for w in ["might", "maybe", "perhaps", "possibly", "could"]):
            agency *= 0.7
            fairness *= 0.7
            belonging *= 0.7

        # Determine mode
        if agency > 0.3 and fairness > 0.3:
            mode = "DREAM_POSITIVE"
        elif agency > 0.3 and fairness < -0.3:
            mode = "DREAM_SHADOW"
        elif belonging > 0.3:
            mode = "DREAM_EXIT"
        else:
            mode = "NOISE_OTHER"

        return {
            "vector": {
                "agency": round(agency, 3),
                "fairness": round(fairness, 3),
                "belonging": round(belonging, 3)
            },
            "mode": mode
        }

    print("=" * 60)
    print("Adversarial Robustness Testing Demo")
    print("=" * 60)

    tester = RobustnessTester()

    # Test text
    test_text = "Everyone can choose to be part of a fair community."

    print(f"\nOriginal text: \"{test_text}\"")

    # Show original projection
    original = await mock_projection(test_text)
    print(f"\nOriginal projection:")
    print(f"  Vector: {original['vector']}")
    print(f"  Mode: {original['mode']}")

    # Generate perturbations
    print("\n" + "-" * 40)
    print("Generating perturbations...")
    perturbations = tester.generate_perturbations(test_text)
    print(f"Generated {len(perturbations)} perturbations")

    # Show some examples
    print("\nSample perturbations:")
    for p in perturbations[:5]:
        print(f"  [{p.perturbation_type.value}] {p.description}")
        print(f"    -> \"{p.perturbed}\"")

    # Full robustness test
    print("\n" + "-" * 40)
    print("Running full robustness test...")
    report = await tester.test_robustness(test_text, mock_projection)

    print(f"\nRobustness Report:")
    print(f"  Total perturbations tested: {len(report.perturbation_results)}")
    print(f"  Mode flips: {report.mode_flip_count}")
    print(f"  Robustness score: {report.robustness_score:.3f}")

    if report.most_sensitive_perturbation:
        msp = report.most_sensitive_perturbation
        print(f"\n  Most sensitive perturbation:")
        print(f"    Type: {msp.perturbation.perturbation_type.value}")
        print(f"    Description: {msp.perturbation.description}")
        print(f"    Delta: {msp.delta}")
        print(f"    Mode flipped: {msp.mode_flipped}")

    # Find mode flip
    print("\n" + "-" * 40)
    print("Searching for minimal edit to flip to DREAM_SHADOW...")
    flip_result = await tester.find_mode_flip(test_text, "DREAM_SHADOW", mock_projection)

    if flip_result.found:
        print(f"  Found!")
        print(f"    Original mode: {flip_result.original_mode}")
        print(f"    New text: \"{flip_result.flipped_text}\"")
        print(f"    Edit: {flip_result.edit_description}")
    else:
        print(f"  No single/double perturbation found to flip to DREAM_SHADOW")

    # Calculate robustness score
    print("\n" + "-" * 40)
    score = await tester.robustness_score(test_text, mock_projection)
    print(f"Overall robustness score: {score:.3f}")

    # Test a less robust text
    print("\n" + "=" * 60)
    vulnerable_text = "This is fair."
    print(f"\nTesting vulnerable text: \"{vulnerable_text}\"")

    vulnerable_report = await tester.test_robustness(vulnerable_text, mock_projection)
    print(f"  Robustness score: {vulnerable_report.robustness_score:.3f}")
    print(f"  Mode flips: {vulnerable_report.mode_flip_count}")

    # Show a perturbation that causes a flip
    flipping_results = [r for r in vulnerable_report.perturbation_results if r.mode_flipped]
    if flipping_results:
        print(f"\n  Perturbations that cause mode flips:")
        for r in flipping_results[:3]:
            print(f"    [{r.perturbation.perturbation_type.value}] {r.perturbation.description}")
            print(f"      Text: \"{r.perturbation.perturbed}\"")
            print(f"      Mode: {r.original_projection['mode']} -> {r.perturbed_projection['mode']}")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(_demo())
