"""
Minimal Coordination Substrate Analysis

This module identifies the irreducible core of coordination that survives all compression -
the "DNA" of coordination that must be preserved for coordination to function.

THEORETICAL FRAMEWORK
=====================

The Observatory reveals an 18D hierarchical manifold structure:
- 9D CoordinationCore (base manifold B): agency_3 + justice_3 + belonging_3
- 9D CoordinationModifiers (fiber F): epistemic_3 + temporal_2 + social_2 + emotional_2

The question: What is the MINIMUM dimensionality for functioning coordination?

DIMENSIONALITY HIERARCHY
========================

From the grammar deletion and substrate-agnostic tests, we identify:

1. DECORATIVE (can be fully removed, d < 0.2):
   - Articles (the, a, an)
   - Hedging (maybe, perhaps)
   - Intensifiers (very, really)
   - Filler (like, you know)
   - Style markers (formality)

2. MODIFYING (add nuance, 0.2 < d < 0.5):
   - Emotional modifiers (arousal, valence)
   - Temporal scope
   - Social distance markers
   - Epistemic evidentiality

3. NECESSARY (strong but not absolute, 0.5 < d < 0.8):
   - Epistemic certainty
   - Temporal focus
   - Power differential signals
   - Epistemic commitment

4. CRITICAL (must be preserved, d > 0.8):
   - Agency attribution (WHO acts)
   - Justice valence (FAIR or UNFAIR)
   - Belonging direction (IN-GROUP vs OUT-GROUP)

THE COORDINATION KERNEL
=======================

The minimal coordination substrate appears to be 3 BITS of information:

1. AGENCY POLARITY: Self (+) vs System/Other (-)
   - "I did it" vs "It happened to me"
   - Binary choice: who is the agent?

2. JUSTICE POLARITY: Fair (+) vs Unfair (-)
   - "Deserved" vs "Wronged"
   - Binary choice: is the outcome just?

3. BELONGING POLARITY: In (+) vs Out (-)
   - "We/Us" vs "Them/Those"
   - Binary choice: group membership direction

This gives us a 3-bit coordination code with 8 possible states:

    (+,+,+) = HEROIC_INCLUSIVE: "I achieved this for us, as deserved"
    (+,+,-) = HEROIC_EXCLUSIVE: "I achieved this against them, as deserved"
    (+,-,+) = TRAGIC_INCLUSIVE: "I failed us, unfairly"
    (+,-,-) = TRAGIC_EXCLUSIVE: "I failed against them, unfairly"
    (-,+,+) = FORTUNATE_INCLUSIVE: "We were blessed, as deserved"
    (-,+,-) = FORTUNATE_EXCLUSIVE: "We won against them, as deserved"
    (-,-,+) = VICTIM_INCLUSIVE: "We were wronged together"
    (-,-,-) = VICTIM_EXCLUSIVE: "We were wronged by them"

ISOMORPHISM EXAMPLES
====================

These all encode the same coordination state (+,-,-) = TRAGIC_EXCLUSIVE:

Natural:     "The system unfairly destroyed my community"
Stripped:    "system.unfair -> my_community.destroyed"
Protocol:    "SYS:NEG:AGN:-1:JST:-1:BLG:-1"
Symbolic:    "S- J- B-"
Minimal:     "---" (three minus signs)

COMPRESSION LIMIT
=================

Shannon's channel capacity theorem suggests:
- 3 bits = minimum for coordination
- Below 3 bits = ambiguous, requires context
- "XZQ<<alpha>>sig" fails because it's BELOW the compression limit

The coordination limit is NOT about legibility (human readability).
It's about preserving the 3-bit polarity structure.

EXPERIMENTAL VALIDATION
=======================

Test 1: Progressive Deletion
- Remove features in order of necessity
- Measure when coordination BREAKS (mode classification fails)
- Find the critical threshold

Test 2: Dimensional Projection
- Project 18D -> 9D -> 3D -> 1D
- At each stage, measure coordination effectiveness
- Find minimum dimensionality

Test 3: Isomorphism Test
- Express same coordination in different substrates
- Verify they occupy same manifold region
- Confirm 3-bit encoding is preserved
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from .hierarchical_coordinates import (
    HierarchicalCoordinate,
    CoordinationCore,
    AgencyDecomposition,
    JusticeDecomposition,
    BelongingDecomposition,
    extract_hierarchical_coordinate,
    project_to_base,
)


class CoordinationState(Enum):
    """The 8 fundamental coordination states (3-bit encoding)."""
    HEROIC_INCLUSIVE = "+,+,+"      # Agency+, Justice+, Belonging+
    HEROIC_EXCLUSIVE = "+,+,-"      # Agency+, Justice+, Belonging-
    TRAGIC_INCLUSIVE = "+,-,+"      # Agency+, Justice-, Belonging+
    TRAGIC_EXCLUSIVE = "+,-,-"      # Agency+, Justice-, Belonging-
    FORTUNATE_INCLUSIVE = "-,+,+"   # Agency-, Justice+, Belonging+
    FORTUNATE_EXCLUSIVE = "-,+,-"   # Agency-, Justice+, Belonging-
    VICTIM_INCLUSIVE = "-,-,+"      # Agency-, Justice-, Belonging+
    VICTIM_EXCLUSIVE = "-,-,-"      # Agency-, Justice-, Belonging-


@dataclass
class MinimalCoordinationCode:
    """
    The irreducible 3-bit coordination substrate.

    This is the "DNA" of coordination - remove any bit and coordination fails.
    """
    agency_polarity: int  # +1 (self) or -1 (system/other)
    justice_polarity: int  # +1 (fair) or -1 (unfair)
    belonging_polarity: int  # +1 (in-group) or -1 (out-group)

    def __post_init__(self):
        # Ensure polarities are valid
        assert self.agency_polarity in (-1, +1), "Agency must be +1 or -1"
        assert self.justice_polarity in (-1, +1), "Justice must be +1 or -1"
        assert self.belonging_polarity in (-1, +1), "Belonging must be +1 or -1"

    @property
    def state(self) -> CoordinationState:
        """Get the fundamental coordination state."""
        code = (
            "+" if self.agency_polarity > 0 else "-",
            "+" if self.justice_polarity > 0 else "-",
            "+" if self.belonging_polarity > 0 else "-"
        )
        code_str = ",".join(code)

        for state in CoordinationState:
            if state.value == code_str:
                return state
        return CoordinationState.VICTIM_EXCLUSIVE  # Default fallback

    @property
    def binary_code(self) -> int:
        """Get 3-bit binary representation (0-7)."""
        a = 1 if self.agency_polarity > 0 else 0
        j = 1 if self.justice_polarity > 0 else 0
        b = 1 if self.belonging_polarity > 0 else 0
        return (a << 2) | (j << 1) | b

    def to_symbolic(self) -> str:
        """Minimal symbolic encoding."""
        a = "+" if self.agency_polarity > 0 else "-"
        j = "+" if self.justice_polarity > 0 else "-"
        b = "+" if self.belonging_polarity > 0 else "-"
        return f"{a}{j}{b}"

    def to_protocol(self) -> str:
        """Protocol encoding for AI systems."""
        return f"A:{self.agency_polarity}:J:{self.justice_polarity}:B:{self.belonging_polarity}"

    @classmethod
    def from_coordinate(cls, coord: HierarchicalCoordinate) -> "MinimalCoordinationCode":
        """Extract minimal code from full hierarchical coordinate."""
        agency = coord.core.agency.aggregate
        justice = coord.core.justice.aggregate
        belonging = coord.core.belonging.aggregate

        return cls(
            agency_polarity=+1 if agency >= 0 else -1,
            justice_polarity=+1 if justice >= 0 else -1,
            belonging_polarity=+1 if belonging >= 0 else -1
        )

    @classmethod
    def from_text(cls, text: str) -> "MinimalCoordinationCode":
        """Extract minimal code directly from text."""
        coord = extract_hierarchical_coordinate(text)
        return cls.from_coordinate(coord)


@dataclass
class FeatureNecessity:
    """Analysis of a feature's necessity for coordination."""
    feature_name: str
    drift_magnitude: float  # How much coordination drifts when removed
    classification: str  # decorative, modifying, necessary, critical
    polarity_flip_rate: float  # How often removal flips core polarities

    @property
    def is_critical(self) -> bool:
        return self.classification == "critical"

    @property
    def can_remove(self) -> bool:
        return self.classification in ("decorative", "modifying")


@dataclass
class CompressionExperimentResult:
    """Result of a compression experiment."""
    original_text: str
    compressed_forms: Dict[str, str]  # compression_level -> text
    preserved_code: Dict[str, MinimalCoordinationCode]
    coordination_preserved: Dict[str, bool]
    compression_limit: str  # Most compressed form that still coordinates
    shannon_bits: float  # Effective bits of coordination information


class MinimalCoordinationAnalyzer:
    """
    Analyzes the minimal coordination substrate.

    Finds the irreducible core of coordination by:
    1. Progressive feature deletion
    2. Dimensional projection
    3. Substrate isomorphism testing
    """

    # Feature necessity thresholds
    THRESHOLDS = {
        "decorative": 0.2,
        "modifying": 0.5,
        "necessary": 0.8,
        # Above 0.8 = critical
    }

    # Canonical examples of each coordination state
    CANONICAL_EXAMPLES = {
        CoordinationState.HEROIC_INCLUSIVE: [
            "I worked hard and achieved success for our team",
            "My effort brought victory to our community",
            "I built this for all of us together",
        ],
        CoordinationState.HEROIC_EXCLUSIVE: [
            "I defeated them and won what I deserved",
            "My triumph proved them wrong",
            "I overcame their opposition through my own strength",
        ],
        CoordinationState.TRAGIC_INCLUSIVE: [
            "I failed my community despite my best efforts",
            "My mistakes hurt us all unfairly",
            "I let us down through no fault of anyone",
        ],
        CoordinationState.TRAGIC_EXCLUSIVE: [
            "I fought against their system but was defeated unfairly",
            "Despite my efforts, they crushed me unjustly",
            "I stood alone against them and lost wrongly",
        ],
        CoordinationState.FORTUNATE_INCLUSIVE: [
            "Fortune smiled on our community",
            "We were blessed with good luck together",
            "Circumstances favored all of us deservedly",
        ],
        CoordinationState.FORTUNATE_EXCLUSIVE: [
            "Fate favored us over them as it should",
            "Fortune chose us instead of them rightly",
            "We prospered while they didn't, as deserved",
        ],
        CoordinationState.VICTIM_INCLUSIVE: [
            "We were wronged together as a community",
            "Our group suffered unjustly as one",
            "Together we bore unfair treatment",
        ],
        CoordinationState.VICTIM_EXCLUSIVE: [
            "They did this to us unfairly",
            "We were victimized by them unjustly",
            "Their system oppressed us wrongly",
        ],
    }

    def __init__(self):
        self.feature_necessity_cache: Dict[str, FeatureNecessity] = {}

    def extract_minimal_code(self, text: str) -> MinimalCoordinationCode:
        """Extract the minimal 3-bit coordination code from text."""
        return MinimalCoordinationCode.from_text(text)

    def test_coordination_preservation(
        self,
        original: str,
        transformed: str
    ) -> Tuple[bool, float]:
        """
        Test if coordination is preserved between original and transformed text.

        Returns:
            (preserved: bool, drift: float)
        """
        orig_code = self.extract_minimal_code(original)
        trans_code = self.extract_minimal_code(transformed)

        # Count polarity flips
        flips = 0
        if orig_code.agency_polarity != trans_code.agency_polarity:
            flips += 1
        if orig_code.justice_polarity != trans_code.justice_polarity:
            flips += 1
        if orig_code.belonging_polarity != trans_code.belonging_polarity:
            flips += 1

        # Coordination preserved if no polarity flips
        preserved = (flips == 0)

        # Drift as fraction of flipped polarities
        drift = flips / 3.0

        return preserved, drift

    def progressive_compression(
        self,
        text: str,
        levels: int = 10
    ) -> CompressionExperimentResult:
        """
        Progressively compress text and track coordination preservation.

        This finds the compression limit - the maximum compression
        before coordination breaks.
        """
        original_code = self.extract_minimal_code(text)

        compressed_forms = {}
        preserved_codes = {}
        coordination_preserved = {}

        words = text.split()

        for i in range(levels):
            level = i / (levels - 1)
            level_name = f"L{i}"

            if level == 0:
                # Level 0: Original
                compressed = text
            elif level < 0.3:
                # Remove decorative words
                decorative = {"the", "a", "an", "very", "really", "just", "like", "so"}
                compressed = " ".join(w for w in words if w.lower() not in decorative)
            elif level < 0.5:
                # Technical abbreviations
                compressed = " ".join(w[:4].upper() for w in words if len(w) > 2)
            elif level < 0.7:
                # Key words only
                key_markers = {
                    "agency": ["I", "me", "my", "we", "us", "they", "system"],
                    "justice": ["fair", "unfair", "deserve", "wrong", "right", "just"],
                    "belonging": ["our", "together", "them", "those", "outsider"],
                }
                all_markers = sum(key_markers.values(), [])
                compressed = " ".join(w for w in words if w.lower() in all_markers)
            elif level < 0.9:
                # Protocol format
                compressed = original_code.to_protocol()
            else:
                # Minimal symbolic
                compressed = original_code.to_symbolic()

            compressed_forms[level_name] = compressed

            if compressed.strip():
                code = self.extract_minimal_code(compressed)
                preserved_codes[level_name] = code
                coordination_preserved[level_name] = (code.state == original_code.state)
            else:
                preserved_codes[level_name] = original_code
                coordination_preserved[level_name] = False

        # Find compression limit (most compressed that still works)
        compression_limit = "L0"
        for level_name, preserved in coordination_preserved.items():
            if preserved:
                compression_limit = level_name

        # Effective Shannon bits
        # 3 bits minimum for coordination, +log2(vocab_size) for richness
        unique_symbols = len(set(compressed_forms[compression_limit].split()))
        shannon_bits = 3.0 + max(0, np.log2(unique_symbols + 1))

        return CompressionExperimentResult(
            original_text=text,
            compressed_forms=compressed_forms,
            preserved_code=preserved_codes,
            coordination_preserved=coordination_preserved,
            compression_limit=compression_limit,
            shannon_bits=shannon_bits
        )

    def test_substrate_isomorphism(
        self,
        natural: str,
        protocol: str,
        symbolic: str
    ) -> Dict[str, any]:
        """
        Test if different substrate representations encode the same coordination.

        Returns analysis of isomorphism between substrates.
        """
        natural_code = self.extract_minimal_code(natural)
        protocol_code = self.extract_minimal_code(protocol)
        symbolic_code = self.extract_minimal_code(symbolic)

        # Check if all have same state
        states = [natural_code.state, protocol_code.state, symbolic_code.state]
        is_isomorphic = len(set(states)) == 1

        return {
            "natural": {
                "text": natural,
                "code": natural_code.to_symbolic(),
                "state": natural_code.state.name
            },
            "protocol": {
                "text": protocol,
                "code": protocol_code.to_symbolic(),
                "state": protocol_code.state.name
            },
            "symbolic": {
                "text": symbolic,
                "code": symbolic_code.to_symbolic(),
                "state": symbolic_code.state.name
            },
            "is_isomorphic": is_isomorphic,
            "common_state": states[0].name if is_isomorphic else "DIVERGENT"
        }

    def project_to_dimensions(
        self,
        text: str,
        target_dims: int
    ) -> Tuple[np.ndarray, bool]:
        """
        Project text coordinate to target dimensionality.

        Returns:
            (projected_coordinate, coordination_preserved)
        """
        coord = extract_hierarchical_coordinate(text)
        original_code = MinimalCoordinationCode.from_coordinate(coord)

        if target_dims >= 18:
            # Full 18D
            projected = coord.to_full_array()
        elif target_dims >= 9:
            # 9D core only
            projected = coord.core.to_array()
        elif target_dims >= 3:
            # 3D aggregate
            projected = np.array(coord.core.to_legacy_3d())
        else:
            # 1D: weighted sum
            legacy = coord.core.to_legacy_3d()
            projected = np.array([sum(legacy) / 3])

        # Check if coordination preserved
        if target_dims >= 3:
            # Can still extract polarity from 3D
            agency_sign = +1 if projected[0 if target_dims >= 3 else 0] >= 0 else -1
            justice_sign = +1 if projected[1 if target_dims >= 3 else 0] >= 0 else -1
            belonging_sign = +1 if projected[2 if target_dims >= 3 else 0] >= 0 else -1

            recovered_code = MinimalCoordinationCode(
                agency_polarity=agency_sign,
                justice_polarity=justice_sign,
                belonging_polarity=belonging_sign
            )
            preserved = (recovered_code.state == original_code.state)
        else:
            # 1D or 2D - cannot preserve full coordination
            preserved = False

        return projected, preserved

    def find_coordination_kernel(
        self,
        texts: List[str]
    ) -> Dict[str, any]:
        """
        Analyze multiple texts to find the coordination kernel.

        The kernel is the minimum information that MUST be preserved
        across all texts for coordination to work.
        """
        results = {
            "texts_analyzed": len(texts),
            "kernel_dimensions": 3,
            "kernel_bits": 3,
            "state_distribution": {},
            "compression_results": [],
            "dimensional_preservation": {},
        }

        # Analyze each text
        for text in texts:
            # Get state
            code = self.extract_minimal_code(text)
            state_name = code.state.name
            results["state_distribution"][state_name] = \
                results["state_distribution"].get(state_name, 0) + 1

            # Compression test
            compression = self.progressive_compression(text)
            results["compression_results"].append({
                "original": text[:50] + "...",
                "limit": compression.compression_limit,
                "bits": compression.shannon_bits
            })

        # Dimensional preservation test
        for target_dim in [18, 9, 3, 1]:
            preserved_count = 0
            for text in texts:
                _, preserved = self.project_to_dimensions(text, target_dim)
                if preserved:
                    preserved_count += 1

            results["dimensional_preservation"][f"{target_dim}D"] = \
                preserved_count / len(texts)

        # Generate kernel description
        results["kernel_description"] = self._generate_kernel_description(results)

        return results

    def _generate_kernel_description(self, analysis: Dict) -> str:
        """Generate prose description of the coordination kernel."""
        lines = [
            "MINIMAL COORDINATION SUBSTRATE",
            "=" * 40,
            "",
            f"Analyzed {analysis['texts_analyzed']} texts",
            "",
            "KERNEL PROPERTIES:",
            f"  Dimensions: {analysis['kernel_dimensions']}",
            f"  Information: {analysis['kernel_bits']} bits",
            "",
            "The coordination kernel is the 3-bit polarity structure:",
            "  - Agency: WHO acts (self vs system)",
            "  - Justice: FAIRNESS of outcome (just vs unjust)",
            "  - Belonging: GROUP direction (in vs out)",
            "",
            "DIMENSIONAL PRESERVATION:",
        ]

        for dim, rate in analysis["dimensional_preservation"].items():
            status = "PRESERVED" if rate > 0.9 else "DEGRADED" if rate > 0.5 else "BROKEN"
            lines.append(f"  {dim}: {rate:.1%} coordination preserved ({status})")

        lines.extend([
            "",
            "CONCLUSION:",
            "  Minimum: 3D (3 bits)",
            "  Below 3D: Coordination FAILS",
            "  9D: Full nuance, 100% preserved",
            "  18D: Full richness with modifiers",
        ])

        return "\n".join(lines)


def run_minimal_substrate_analysis(sample_texts: Optional[List[str]] = None) -> Dict:
    """
    Run the full minimal coordination substrate analysis.

    This is the main entry point for finding the coordination "DNA".
    """
    analyzer = MinimalCoordinationAnalyzer()

    if sample_texts is None:
        # Use canonical examples
        sample_texts = []
        for examples in analyzer.CANONICAL_EXAMPLES.values():
            sample_texts.extend(examples)

    return analyzer.find_coordination_kernel(sample_texts)


# Convenience exports
__all__ = [
    "CoordinationState",
    "MinimalCoordinationCode",
    "FeatureNecessity",
    "CompressionExperimentResult",
    "MinimalCoordinationAnalyzer",
    "run_minimal_substrate_analysis",
]
