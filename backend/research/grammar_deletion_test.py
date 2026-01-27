"""
Grammar Deletion Test

Identifies which grammatical structures are coordination-necessary vs decorative
by systematically removing linguistic features and measuring projection drift.

This is the first experiment enabled by the emergent language research reframe:
If AI systems can coordinate with minimal symbols, what parts of human grammar
are actually doing coordination work vs cultural ornamentation?

Usage:
    python -m research.grammar_deletion_test --text "Your narrative here"
    python -m research.grammar_deletion_test --file path/to/corpus.txt
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DeletionResult:
    """Result of a single deletion test."""
    feature_name: str
    original_text: str
    modified_text: str
    original_projection: Dict[str, float]
    modified_projection: Dict[str, float]
    projection_drift: float  # Euclidean distance
    axis_drifts: Dict[str, float]  # Per-axis drift
    mode_changed: bool
    original_mode: str
    modified_mode: str


@dataclass
class GrammarAnalysis:
    """Complete grammar deletion analysis for a text."""
    original_text: str
    original_projection: Dict[str, float]
    original_mode: str
    deletions: List[DeletionResult]
    necessary_features: List[str]  # Features that caused significant drift
    decorative_features: List[str]  # Features with minimal impact
    coordination_core: str  # Text with only necessary features


# Grammatical feature deletion functions
DELETION_FUNCTIONS = {}


def register_deletion(name: str):
    """Decorator to register a deletion function."""
    def decorator(func):
        DELETION_FUNCTIONS[name] = func
        return func
    return decorator


@register_deletion("articles")
def delete_articles(text: str) -> str:
    """Remove articles (a, an, the)."""
    return re.sub(r'\b(a|an|the)\b\s*', '', text, flags=re.IGNORECASE)


@register_deletion("pronouns_first_person")
def delete_first_person_pronouns(text: str) -> str:
    """Remove first-person pronouns (I, me, my, mine, we, us, our, ours)."""
    return re.sub(
        r'\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b',
        '[SELF]',
        text,
        flags=re.IGNORECASE
    )


@register_deletion("pronouns_second_person")
def delete_second_person_pronouns(text: str) -> str:
    """Remove second-person pronouns (you, your, yours)."""
    return re.sub(
        r'\b(you|your|yours|yourself|yourselves)\b',
        '[OTHER]',
        text,
        flags=re.IGNORECASE
    )


@register_deletion("pronouns_third_person")
def delete_third_person_pronouns(text: str) -> str:
    """Remove third-person pronouns (he, she, they, it, etc.)."""
    return re.sub(
        r'\b(he|him|his|himself|she|her|hers|herself|they|them|their|theirs|themselves|it|its|itself)\b',
        '[ENTITY]',
        text,
        flags=re.IGNORECASE
    )


@register_deletion("tense_past")
def convert_past_to_present(text: str) -> str:
    """Convert past tense markers to present (crude approximation)."""
    # Common irregular verbs
    text = re.sub(r'\bwas\b', 'is', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwere\b', 'are', text, flags=re.IGNORECASE)
    text = re.sub(r'\bhad\b', 'have', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdid\b', 'do', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwent\b', 'go', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcame\b', 'come', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsaw\b', 'see', text, flags=re.IGNORECASE)
    text = re.sub(r'\btook\b', 'take', text, flags=re.IGNORECASE)
    # Regular past tense -ed
    text = re.sub(r'(\w+)ed\b', r'\1', text)
    return text


@register_deletion("modals")
def delete_modals(text: str) -> str:
    """Remove modal verbs (can, could, will, would, should, must, may, might)."""
    return re.sub(
        r'\b(can|could|will|would|shall|should|must|may|might|ought)\b\s*',
        '',
        text,
        flags=re.IGNORECASE
    )


@register_deletion("conjunctions")
def delete_conjunctions(text: str) -> str:
    """Remove coordinating and subordinating conjunctions."""
    return re.sub(
        r'\b(and|but|or|nor|for|yet|so|although|because|since|unless|while|whereas|if|when|after|before)\b',
        '',
        text,
        flags=re.IGNORECASE
    )


@register_deletion("prepositions")
def delete_prepositions(text: str) -> str:
    """Remove common prepositions."""
    return re.sub(
        r'\b(in|on|at|to|for|with|by|from|of|about|into|through|during|before|after|above|below|between|under|over)\b',
        '',
        text,
        flags=re.IGNORECASE
    )


@register_deletion("adjectives_positive")
def delete_positive_adjectives(text: str) -> str:
    """Remove positive emotional adjectives."""
    return re.sub(
        r'\b(good|great|excellent|wonderful|amazing|fantastic|beautiful|happy|joyful|loving|kind|generous|strong|powerful|successful|brilliant)\b',
        '',
        text,
        flags=re.IGNORECASE
    )


@register_deletion("adjectives_negative")
def delete_negative_adjectives(text: str) -> str:
    """Remove negative emotional adjectives."""
    return re.sub(
        r'\b(bad|terrible|awful|horrible|ugly|sad|angry|hateful|cruel|weak|failed|stupid|broken|corrupt|unfair|unjust)\b',
        '',
        text,
        flags=re.IGNORECASE
    )


@register_deletion("hedging")
def delete_hedging(text: str) -> str:
    """Remove hedging language."""
    return re.sub(
        r'\b(maybe|perhaps|possibly|probably|likely|somewhat|rather|quite|fairly|sort of|kind of|I think|I believe|I feel|it seems)\b',
        '',
        text,
        flags=re.IGNORECASE
    )


@register_deletion("intensifiers")
def delete_intensifiers(text: str) -> str:
    """Remove intensifiers."""
    return re.sub(
        r'\b(very|really|extremely|absolutely|completely|totally|utterly|highly|incredibly|deeply|truly)\b',
        '',
        text,
        flags=re.IGNORECASE
    )


@register_deletion("negation")
def delete_negation(text: str) -> str:
    """Remove negation markers."""
    return re.sub(
        r"\b(not|n't|never|no|none|nothing|nowhere|nobody|neither|nor)\b",
        '',
        text,
        flags=re.IGNORECASE
    )


@register_deletion("possessives")
def delete_possessives(text: str) -> str:
    """Remove possessive markers."""
    text = re.sub(r"'s\b", '', text)  # 's possessive
    text = re.sub(r"s'\b", 's', text)  # s' possessive
    return text


@register_deletion("agency_markers")
def delete_agency_markers(text: str) -> str:
    """Remove agency-indicating language."""
    return re.sub(
        r'\b(I decided|I chose|I made|I created|I built|I achieved|I control|I own|my decision|my choice|I am responsible|I take responsibility)\b',
        '[ACTION]',
        text,
        flags=re.IGNORECASE
    )


@register_deletion("belonging_markers")
def delete_belonging_markers(text: str) -> str:
    """Remove belonging-indicating language."""
    return re.sub(
        r'\b(we together|our community|our team|our family|we belong|part of|member of|connected to|one of us|in this together)\b',
        '[GROUP]',
        text,
        flags=re.IGNORECASE
    )


@register_deletion("justice_markers")
def delete_justice_markers(text: str) -> str:
    """Remove justice/fairness-indicating language."""
    return re.sub(
        r'\b(fair|unfair|just|unjust|deserve|entitled|right|wrong|should have|ought to|merit|earned|rigged|corrupt|equitable|legitimate)\b',
        '[JUSTICE]',
        text,
        flags=re.IGNORECASE
    )


def clean_text(text: str) -> str:
    """Clean up spacing and punctuation after deletions."""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove space before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Remove empty brackets
    text = re.sub(r'\[\s*\]', '', text)
    return text.strip()


class GrammarDeletionAnalyzer:
    """
    Analyzes which grammatical features are coordination-necessary.

    Uses the Cultural Soliton Observatory projection to measure how much
    meaning (in terms of coordination coordinates) is lost when specific
    grammatical structures are removed.
    """

    def __init__(self, drift_threshold: float = 0.3):
        """
        Args:
            drift_threshold: Euclidean distance threshold for "significant" drift.
                            Features causing drift > threshold are "necessary".
        """
        self.drift_threshold = drift_threshold
        self.projection_cache = {}

    async def project_text(self, text: str) -> Tuple[Dict[str, float], str]:
        """
        Project text into the coordination manifold.

        Returns:
            (coordinates dict, mode name)
        """
        # Import here to avoid circular imports
        from api_observer_chat import analyze_text_internal

        result = await analyze_text_internal(text, include_forces=False)

        coordinates = result.get("vector", {})
        mode = result.get("mode", {}).get("primary_mode", "NEUTRAL")

        return coordinates, mode

    def compute_drift(self,
                      original: Dict[str, float],
                      modified: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Compute projection drift between original and modified text.

        Returns:
            (euclidean_distance, per_axis_drifts)
        """
        axes = ["agency", "perceived_justice", "belonging"]
        axis_drifts = {}
        total = 0.0

        for axis in axes:
            orig_val = original.get(axis, 0.0)
            mod_val = modified.get(axis, 0.0)
            drift = mod_val - orig_val
            axis_drifts[axis] = drift
            total += drift ** 2

        euclidean = np.sqrt(total)
        return euclidean, axis_drifts

    async def test_deletion(self,
                            text: str,
                            feature_name: str,
                            deletion_func) -> DeletionResult:
        """Test a single deletion and measure its impact."""

        # Apply deletion
        modified_text = clean_text(deletion_func(text))

        # Skip if no change
        if modified_text == text:
            return None

        # Get projections
        orig_coords, orig_mode = await self.project_text(text)
        mod_coords, mod_mode = await self.project_text(modified_text)

        # Compute drift
        drift, axis_drifts = self.compute_drift(orig_coords, mod_coords)

        return DeletionResult(
            feature_name=feature_name,
            original_text=text,
            modified_text=modified_text,
            original_projection=orig_coords,
            modified_projection=mod_coords,
            projection_drift=drift,
            axis_drifts=axis_drifts,
            mode_changed=(orig_mode != mod_mode),
            original_mode=orig_mode,
            modified_mode=mod_mode
        )

    async def analyze_text(self, text: str) -> GrammarAnalysis:
        """
        Run complete grammar deletion analysis on a text.

        Returns analysis showing which features are necessary vs decorative.
        """
        # Get original projection
        orig_coords, orig_mode = await self.project_text(text)

        # Test each deletion
        deletions = []
        for feature_name, deletion_func in DELETION_FUNCTIONS.items():
            result = await self.test_deletion(text, feature_name, deletion_func)
            if result:
                deletions.append(result)

        # Sort by drift (highest first)
        deletions.sort(key=lambda x: x.projection_drift, reverse=True)

        # Classify features
        necessary = [d.feature_name for d in deletions if d.projection_drift > self.drift_threshold]
        decorative = [d.feature_name for d in deletions if d.projection_drift <= self.drift_threshold]

        # Build coordination core (remove all decorative features)
        core_text = text
        for feature_name in decorative:
            if feature_name in DELETION_FUNCTIONS:
                core_text = clean_text(DELETION_FUNCTIONS[feature_name](core_text))

        return GrammarAnalysis(
            original_text=text,
            original_projection=orig_coords,
            original_mode=orig_mode,
            deletions=deletions,
            necessary_features=necessary,
            decorative_features=decorative,
            coordination_core=core_text
        )

    async def analyze_corpus(self, texts: List[str]) -> Dict:
        """
        Analyze a corpus to find consistent patterns.

        Returns aggregate statistics on which features are universally
        necessary vs consistently decorative.
        """
        feature_drifts = {name: [] for name in DELETION_FUNCTIONS}
        mode_flip_counts = {name: 0 for name in DELETION_FUNCTIONS}
        total_texts = 0

        for text in texts:
            if len(text.strip()) < 20:
                continue

            analysis = await self.analyze_text(text)
            total_texts += 1

            for deletion in analysis.deletions:
                feature_drifts[deletion.feature_name].append(deletion.projection_drift)
                if deletion.mode_changed:
                    mode_flip_counts[deletion.feature_name] += 1

        # Compute statistics
        feature_stats = {}
        for name, drifts in feature_drifts.items():
            if drifts:
                feature_stats[name] = {
                    "mean_drift": float(np.mean(drifts)),
                    "std_drift": float(np.std(drifts)),
                    "max_drift": float(np.max(drifts)),
                    "mode_flip_rate": mode_flip_counts[name] / total_texts if total_texts > 0 else 0,
                    "classification": "necessary" if np.mean(drifts) > self.drift_threshold else "decorative"
                }

        # Rank by importance
        ranked_features = sorted(
            feature_stats.items(),
            key=lambda x: x[1]["mean_drift"],
            reverse=True
        )

        return {
            "total_texts": total_texts,
            "feature_statistics": feature_stats,
            "ranked_by_importance": [f[0] for f in ranked_features],
            "necessary_features": [
                f[0] for f in ranked_features
                if f[1]["classification"] == "necessary"
            ],
            "decorative_features": [
                f[0] for f in ranked_features
                if f[1]["classification"] == "decorative"
            ]
        }


# CLI interface
async def main():
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="Grammar Deletion Test - Find coordination-necessary linguistic features"
    )
    parser.add_argument("--text", type=str, help="Single text to analyze")
    parser.add_argument("--file", type=str, help="File with texts (one per line or paragraph)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Drift threshold for 'necessary' classification")

    args = parser.parse_args()

    analyzer = GrammarDeletionAnalyzer(drift_threshold=args.threshold)

    if args.text:
        analysis = await analyzer.analyze_text(args.text)
        print("\n=== GRAMMAR DELETION ANALYSIS ===")
        print(f"\nOriginal: {analysis.original_text[:100]}...")
        print(f"Mode: {analysis.original_mode}")
        print(f"Projection: {analysis.original_projection}")

        print("\n--- Feature Impact Ranking ---")
        for d in analysis.deletions:
            status = "[NECESSARY]" if d.projection_drift > args.threshold else "[decorative]"
            mode_note = f" (mode: {d.original_mode}→{d.modified_mode})" if d.mode_changed else ""
            print(f"  {d.projection_drift:.3f} {status} {d.feature_name}{mode_note}")

        print(f"\nCoordination Core (necessary only):")
        print(f"  {analysis.coordination_core}")

    elif args.file:
        with open(args.file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]

        results = await analyzer.analyze_corpus(texts)

        print("\n=== CORPUS GRAMMAR ANALYSIS ===")
        print(f"Texts analyzed: {results['total_texts']}")

        print("\n--- Necessary Features (high drift when removed) ---")
        for name in results['necessary_features']:
            stats = results['feature_statistics'][name]
            print(f"  {name}: drift={stats['mean_drift']:.3f} (±{stats['std_drift']:.3f}), "
                  f"mode_flip_rate={stats['mode_flip_rate']:.1%}")

        print("\n--- Decorative Features (low drift when removed) ---")
        for name in results['decorative_features']:
            stats = results['feature_statistics'][name]
            print(f"  {name}: drift={stats['mean_drift']:.3f}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
