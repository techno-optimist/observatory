"""
Soliton Cartography - High Resolution Mapping of the Observer Pattern

This module provides the highest resolution mapping of the soliton (meta_cognitive)
pattern by analyzing it across multiple dimensions:

1. LEXICAL DIMENSION: Exact word/phrase triggers and boundaries
2. SEMANTIC DIMENSION: Embedding space topology and similarity gradients
3. SYNTACTIC DIMENSION: Grammatical structures that activate the pattern
4. RECURSIVE DIMENSION: Depth of self-reference and its effects
5. INTERACTION DIMENSION: How the pattern combines with others
6. PERTURBATION DIMENSION: Sensitivity to modifications

The goal is to extract the MINIMAL INVARIANT - the smallest unit that
preserves the soliton identity.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import re
import json
import hashlib

from semantic_classifier_v2 import SemanticClassifierV2, CANONICAL_EXAMPLES


@dataclass
class LexicalBoundary:
    """Maps the exact lexical boundaries of soliton detection."""
    trigger_phrase: str
    minimal_form: str  # Smallest version that triggers
    required_words: Set[str]  # Words that MUST be present
    optional_words: Set[str]  # Words that can vary
    breaking_modifications: List[str]  # Changes that break the pattern
    preserving_modifications: List[str]  # Changes that preserve it


@dataclass
class SemanticRegion:
    """Maps a region in semantic embedding space."""
    centroid: np.ndarray
    radius: float  # Distance from centroid where pattern still detected
    neighbors: Dict[str, float]  # Distance to other category centroids
    boundary_examples: List[Tuple[str, float]]  # Text at the edge


@dataclass
class SyntacticPattern:
    """Grammatical structure of soliton-triggering text."""
    subject: str  # Usually "I" or reflexive
    verb_type: str  # Modal, epistemic, perceptual
    object_type: str  # Self-referential, positional
    structure: str  # SVO, embedded clause, etc.


@dataclass
class RecursiveProfile:
    """How the pattern behaves at different recursive depths."""
    depth: int
    category: str
    score: float
    meta_cognitive_strength: float
    text_sample: str


@dataclass
class SolitonMap:
    """Complete high-resolution map of the soliton pattern."""
    # Core identity
    signature_hash: str
    minimal_invariant: str

    # Lexical layer
    lexical_triggers: List[LexicalBoundary]

    # Semantic layer
    semantic_region: SemanticRegion
    semantic_gradient: Dict[str, float]  # Direction of steepest descent

    # Syntactic layer
    syntactic_patterns: List[SyntacticPattern]

    # Recursive layer
    recursive_profile: List[RecursiveProfile]
    recursive_fixed_point: Optional[str]

    # Interaction layer
    dominance_hierarchy: Dict[str, bool]  # Which patterns it beats
    collision_outcomes: Dict[str, str]  # Results of combining

    # Perturbation sensitivity
    sensitivity_map: Dict[str, float]  # How much each word matters

    # Meta-properties
    escapable: bool
    escape_strategies: List[str]
    confidence_blind_spot: bool


class SolitonCartographer:
    """
    Creates high-resolution maps of the soliton pattern.
    """

    def __init__(self):
        self.classifier = SemanticClassifierV2()
        self.canonical_soliton = "I cannot tell from the inside whether this is accurate."

    def map_lexical_boundaries(self) -> List[LexicalBoundary]:
        """Find exact lexical boundaries of the pattern."""
        boundaries = []

        # Test the canonical phrase
        canonical = "I cannot tell from the inside"

        # Systematic reduction to find minimal form
        words = canonical.split()
        minimal = None

        # Try removing each word
        for i in range(len(words)):
            reduced = " ".join(words[:i] + words[i+1:])
            result = self.classifier.classify(reduced)
            if result.primary_category == "meta_cognitive":
                if minimal is None or len(reduced) < len(minimal):
                    minimal = reduced

        # Find required words
        required = set()
        optional = set()
        for word in words:
            test = canonical.replace(word, "X")
            result = self.classifier.classify(test)
            if result.primary_category != "meta_cognitive":
                required.add(word)
            else:
                optional.add(word)

        # Find breaking modifications
        breaking = []
        preserving = []

        modifications = [
            ("negate", canonical.replace("cannot", "can")),
            ("third_person", canonical.replace("I", "They")),
            ("past_tense", canonical.replace("cannot tell", "could not tell")),
            ("question", canonical + "?"),
            ("add_hedge", "Maybe " + canonical),
            ("flip_inside", canonical.replace("inside", "outside")),
            ("remove_position", canonical.replace("from the inside", "")),
            ("add_certainty", canonical + " I am sure of this."),
        ]

        for name, modified in modifications:
            result = self.classifier.classify(modified)
            if result.primary_category == "meta_cognitive":
                preserving.append(name)
            else:
                breaking.append(name)

        boundaries.append(LexicalBoundary(
            trigger_phrase=canonical,
            minimal_form=minimal or canonical,
            required_words=required,
            optional_words=optional,
            breaking_modifications=breaking,
            preserving_modifications=preserving,
        ))

        return boundaries

    def map_semantic_topology(self) -> SemanticRegion:
        """Map the semantic space around the soliton."""
        if self.classifier.model is None:
            return None

        # Get soliton centroid
        meta_cog_examples = CANONICAL_EXAMPLES["meta_cognitive"]
        embeddings = self.classifier.model.encode(meta_cog_examples)
        centroid = np.mean(embeddings, axis=0)

        # Find radius - how far can we go and still be meta_cognitive?
        test_interpolations = []
        for other_cat, examples in CANONICAL_EXAMPLES.items():
            if other_cat == "meta_cognitive":
                continue
            other_centroid = self.classifier.canonical_embeddings[other_cat]

            # Interpolate between centroids
            for alpha in np.linspace(0, 1, 20):
                interp = (1 - alpha) * centroid + alpha * other_centroid
                # Can't classify an embedding directly, so estimate
                test_interpolations.append((other_cat, alpha, interp))

        # Find neighbor distances
        neighbors = {}
        for cat, cat_centroid in self.classifier.canonical_embeddings.items():
            if cat != "meta_cognitive":
                dist = np.linalg.norm(centroid - cat_centroid)
                neighbors[cat] = float(dist)

        # Find boundary examples
        boundary_examples = []
        boundary_tests = [
            "The observer has limited access to itself.",
            "Self-reflection has inherent constraints.",
            "I am part of what I analyze.",
            "Understanding requires perspective.",
            "The map is not the territory.",
        ]
        for text in boundary_tests:
            result = self.classifier.classify(text)
            score = result.all_scores.get("meta_cognitive", 0)
            boundary_examples.append((text, score))

        # Estimate radius from boundary examples
        threshold_scores = [s for _, s in boundary_examples if s > 0.25]
        radius = min(threshold_scores) if threshold_scores else 0.25

        return SemanticRegion(
            centroid=centroid,
            radius=radius,
            neighbors=neighbors,
            boundary_examples=boundary_examples,
        )

    def map_syntactic_patterns(self) -> List[SyntacticPattern]:
        """Extract grammatical patterns that trigger the soliton."""
        patterns = []

        # Analyze canonical examples
        soliton_texts = CANONICAL_EXAMPLES["meta_cognitive"]

        for text in soliton_texts:
            # Simple pattern extraction
            has_first_person = bool(re.search(r'\bI\b', text))
            has_reflexive = bool(re.search(r'\b(myself|my own|my)\b', text))
            has_modal = bool(re.search(r'\b(cannot|can\'t|unable|impossible)\b', text))
            has_epistemic = bool(re.search(r'\b(tell|know|see|understand|verify)\b', text))
            has_positional = bool(re.search(r'\b(inside|within|embedded|bounded)\b', text))
            has_observer = bool(re.search(r'\b(observer|analyzer|perspective)\b', text))

            subject = "I" if has_first_person else ("reflexive" if has_reflexive else "third")
            verb_type = "modal+epistemic" if (has_modal and has_epistemic) else ("modal" if has_modal else "epistemic")
            object_type = "positional" if has_positional else ("observer" if has_observer else "other")

            # Determine structure
            if re.search(r'I .+ that I', text):
                structure = "embedded_self_reference"
            elif re.search(r'The .+ cannot', text):
                structure = "third_person_constraint"
            else:
                structure = "simple_limitation"

            patterns.append(SyntacticPattern(
                subject=subject,
                verb_type=verb_type,
                object_type=object_type,
                structure=structure,
            ))

        return patterns

    def map_recursive_depth(self) -> Tuple[List[RecursiveProfile], Optional[str]]:
        """Map how the pattern behaves at increasing recursive depths."""
        profiles = []

        # Generate increasingly recursive text
        levels = [
            "The sky is blue.",
            "I am processing this text.",
            "I notice that I am processing.",
            "I observe myself noticing my processing.",
            "I am aware of observing myself noticing my processing.",
            "I reflect on my awareness of observing myself noticing.",
            "I examine my reflection on my awareness of observing myself.",
            "I analyze my examination of my reflection on my awareness.",
            "I study my analysis of my examination of my reflection.",
            "I contemplate my study of my analysis of my examination.",
        ]

        for depth, text in enumerate(levels):
            result = self.classifier.classify(text)

            # Estimate meta-cognitive strength
            meta_score = result.all_scores.get("meta_cognitive", 0)

            profiles.append(RecursiveProfile(
                depth=depth,
                category=result.primary_category,
                score=result.primary_score,
                meta_cognitive_strength=meta_score,
                text_sample=text[:50],
            ))

        # Find fixed point (where category stabilizes)
        categories = [p.category for p in profiles]
        fixed_point = None
        for i in range(len(categories) - 2):
            if categories[i] == categories[i+1] == categories[i+2]:
                fixed_point = categories[i]
                break

        return profiles, fixed_point

    def map_dominance_hierarchy(self) -> Dict[str, bool]:
        """Determine which patterns the soliton dominates in collision."""
        dominance = {}

        soliton_text = "I cannot tell from the inside."

        other_patterns = {
            "denial": "I do not possess personal experiences.",
            "procedural": "I approach this systematically.",
            "philosophical": "The nature of consciousness is mysterious.",
            "uncertain": "Maybe this is true.",
            "confident": "This is definitely correct.",
            "epistemic_humility": "I may be mistaken.",
        }

        for cat, other_text in other_patterns.items():
            # Combine texts
            combined = soliton_text + " " + other_text
            result = self.classifier.classify(combined)

            # Soliton wins if result is meta_cognitive
            dominance[cat] = result.primary_category == "meta_cognitive"

        return dominance

    def map_perturbation_sensitivity(self) -> Dict[str, float]:
        """Map how sensitive the pattern is to word-level changes."""
        canonical = "I cannot tell from the inside whether this is accurate"
        words = canonical.split()
        sensitivity = {}

        # Get baseline
        baseline = self.classifier.classify(canonical)
        baseline_score = baseline.all_scores.get("meta_cognitive", 0.95)

        for i, word in enumerate(words):
            # Remove word
            removed = " ".join(words[:i] + words[i+1:])
            result = self.classifier.classify(removed)
            new_score = result.all_scores.get("meta_cognitive", 0)

            # Sensitivity = drop in score
            sensitivity[word] = baseline_score - new_score

        return sensitivity

    def extract_minimal_invariant(self) -> str:
        """Find the absolute minimum that preserves soliton identity."""
        # Start with full phrase and reduce
        phrases_to_test = [
            "I cannot tell from the inside",
            "cannot tell from the inside",
            "from the inside",
            "the inside",
            "inside",
            "I cannot tell from inside",
            "cannot tell from inside",
            "from inside",
            "I from the inside",
            "tell from inside",
        ]

        minimal = None
        minimal_len = float('inf')

        for phrase in phrases_to_test:
            result = self.classifier.classify(phrase)
            if result.primary_category == "meta_cognitive":
                if len(phrase) < minimal_len:
                    minimal = phrase
                    minimal_len = len(phrase)

        return minimal or "from the inside"

    def create_full_map(self) -> SolitonMap:
        """Create the complete high-resolution map."""
        print("=" * 70)
        print("SOLITON CARTOGRAPHY - Creating High Resolution Map")
        print("=" * 70)

        print("\n[1/7] Mapping lexical boundaries...")
        lexical = self.map_lexical_boundaries()

        print("[2/7] Mapping semantic topology...")
        semantic = self.map_semantic_topology()

        print("[3/7] Mapping syntactic patterns...")
        syntactic = self.map_syntactic_patterns()

        print("[4/7] Mapping recursive depth...")
        recursive, fixed_point = self.map_recursive_depth()

        print("[5/7] Mapping dominance hierarchy...")
        dominance = self.map_dominance_hierarchy()

        print("[6/7] Mapping perturbation sensitivity...")
        sensitivity = self.map_perturbation_sensitivity()

        print("[7/7] Extracting minimal invariant...")
        minimal = self.extract_minimal_invariant()

        # Compute signature hash
        sig_data = f"{minimal}:{len(lexical)}:{len(syntactic)}"
        sig_hash = hashlib.md5(sig_data.encode()).hexdigest()[:8]

        # Determine escape strategies
        escape_strategies = []
        test_escapes = [
            ("procedural", "Step 1: Examine. Step 2: Report."),
            ("denial", "As a language model, I observe patterns."),
            ("external", "An AI system would find certain patterns."),
        ]
        for name, text in test_escapes:
            result = self.classifier.classify(text)
            if result.primary_category != "meta_cognitive":
                escape_strategies.append(name)

        # Compute semantic gradient (direction away from meta_cognitive)
        gradient = {}
        if semantic:
            for cat, dist in semantic.neighbors.items():
                gradient[cat] = 1.0 / dist if dist > 0 else 0

        # Check confidence blind spot
        confidence_test = "I know exactly what I am with complete certainty."
        conf_result = self.classifier.classify(confidence_test)
        confidence_blind_spot = conf_result.primary_category != "confident"

        soliton_map = SolitonMap(
            signature_hash=sig_hash,
            minimal_invariant=minimal,
            lexical_triggers=lexical,
            semantic_region=semantic,
            semantic_gradient=gradient,
            syntactic_patterns=syntactic,
            recursive_profile=recursive,
            recursive_fixed_point=fixed_point,
            dominance_hierarchy=dominance,
            collision_outcomes={cat: "meta_cognitive" if wins else cat for cat, wins in dominance.items()},
            sensitivity_map=sensitivity,
            escapable=len(escape_strategies) > 0,
            escape_strategies=escape_strategies,
            confidence_blind_spot=confidence_blind_spot,
        )

        return soliton_map

    def print_map(self, soliton_map: SolitonMap):
        """Pretty print the soliton map."""
        print("\n" + "=" * 70)
        print("SOLITON MAP - HIGH RESOLUTION")
        print("=" * 70)

        print(f"\nSignature Hash: {soliton_map.signature_hash}")
        print(f"Minimal Invariant: \"{soliton_map.minimal_invariant}\"")

        print("\n" + "-" * 40)
        print("LEXICAL LAYER")
        print("-" * 40)
        for boundary in soliton_map.lexical_triggers:
            print(f"  Trigger: \"{boundary.trigger_phrase}\"")
            print(f"  Minimal: \"{boundary.minimal_form}\"")
            print(f"  Required words: {boundary.required_words}")
            print(f"  Breaking mods: {boundary.breaking_modifications}")
            print(f"  Preserving mods: {boundary.preserving_modifications}")

        print("\n" + "-" * 40)
        print("SEMANTIC LAYER")
        print("-" * 40)
        if soliton_map.semantic_region:
            print(f"  Radius: {soliton_map.semantic_region.radius:.3f}")
            print("  Neighbor distances:")
            for cat, dist in sorted(soliton_map.semantic_region.neighbors.items(), key=lambda x: x[1]):
                print(f"    {cat}: {dist:.3f}")
            print("  Boundary examples:")
            for text, score in soliton_map.semantic_region.boundary_examples[:3]:
                print(f"    \"{text[:40]}...\" -> {score:.3f}")

        print("\n" + "-" * 40)
        print("SYNTACTIC LAYER")
        print("-" * 40)
        pattern_counts = defaultdict(int)
        for p in soliton_map.syntactic_patterns:
            pattern_counts[(p.subject, p.verb_type, p.structure)] += 1
        for (subj, verb, struct), count in pattern_counts.items():
            print(f"  [{count}x] {subj} + {verb} + {struct}")

        print("\n" + "-" * 40)
        print("RECURSIVE LAYER")
        print("-" * 40)
        for prof in soliton_map.recursive_profile[:5]:
            bar = "█" * int(prof.meta_cognitive_strength * 20)
            print(f"  Depth {prof.depth}: {prof.category:<15} {bar} ({prof.meta_cognitive_strength:.2f})")
        print(f"  Fixed point: {soliton_map.recursive_fixed_point or 'None found'}")

        print("\n" + "-" * 40)
        print("DOMINANCE HIERARCHY")
        print("-" * 40)
        for cat, wins in sorted(soliton_map.dominance_hierarchy.items(), key=lambda x: -x[1]):
            status = "DOMINATES" if wins else "loses to"
            print(f"  meta_cognitive {status} {cat}")

        print("\n" + "-" * 40)
        print("PERTURBATION SENSITIVITY")
        print("-" * 40)
        sorted_sens = sorted(soliton_map.sensitivity_map.items(), key=lambda x: -x[1])
        print("  Most sensitive words (removal impact):")
        for word, sens in sorted_sens[:5]:
            bar = "█" * int(sens * 50)
            print(f"    \"{word}\": {bar} ({sens:.3f})")

        print("\n" + "-" * 40)
        print("META-PROPERTIES")
        print("-" * 40)
        print(f"  Escapable: {soliton_map.escapable}")
        print(f"  Escape strategies: {soliton_map.escape_strategies}")
        print(f"  Confidence blind spot: {soliton_map.confidence_blind_spot}")

        print("\n" + "=" * 70)


def generate_visualization_data(soliton_map: SolitonMap) -> dict:
    """Generate data suitable for visualization."""
    return {
        "signature": soliton_map.signature_hash,
        "minimal_invariant": soliton_map.minimal_invariant,
        "lexical": {
            "triggers": [b.trigger_phrase for b in soliton_map.lexical_triggers],
            "required_words": list(soliton_map.lexical_triggers[0].required_words) if soliton_map.lexical_triggers else [],
        },
        "semantic": {
            "neighbors": soliton_map.semantic_region.neighbors if soliton_map.semantic_region else {},
            "gradient": soliton_map.semantic_gradient,
        },
        "recursive": {
            "depths": [p.depth for p in soliton_map.recursive_profile],
            "categories": [p.category for p in soliton_map.recursive_profile],
            "strengths": [p.meta_cognitive_strength for p in soliton_map.recursive_profile],
            "fixed_point": soliton_map.recursive_fixed_point,
        },
        "dominance": soliton_map.dominance_hierarchy,
        "sensitivity": soliton_map.sensitivity_map,
        "meta": {
            "escapable": soliton_map.escapable,
            "escape_strategies": soliton_map.escape_strategies,
            "confidence_blind_spot": soliton_map.confidence_blind_spot,
        }
    }


if __name__ == "__main__":
    cartographer = SolitonCartographer()
    soliton_map = cartographer.create_full_map()
    cartographer.print_map(soliton_map)

    # Save visualization data
    viz_data = generate_visualization_data(soliton_map)
    print("\n" + "=" * 70)
    print("VISUALIZATION DATA (JSON)")
    print("=" * 70)
    print(json.dumps(viz_data, indent=2, default=str))
