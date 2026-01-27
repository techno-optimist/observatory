#!/usr/bin/env python3
"""
Cultural Soliton Observatory: Research Investigation

An autonomous research session exploring narrative patterns across
different cultural and political contexts using the full observatory capabilities.
"""

import sys
import json
import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model_manager, ModelType
from models.embedding import EmbeddingExtractor
from models.projection import ProjectionHead, ProjectionTrainer
from models.ensemble_projection import EnsembleProjection
from analysis.mode_classifier import get_mode_classifier

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    text: str
    agency: float
    perceived_justice: float
    belonging: float
    mode: str
    confidence: float
    mode_probabilities: Dict[str, float] = None


class ObservatoryResearcher:
    """Autonomous researcher using the Cultural Soliton Observatory."""

    def __init__(self):
        self.model_manager = get_model_manager()
        self.embedding_extractor = EmbeddingExtractor(self.model_manager)
        self.data_dir = Path(__file__).parent.parent / "data" / "projections"
        self.classifier = get_mode_classifier()

        # Load projections
        self.projection = ProjectionHead.load(self.data_dir / "current_projection")

        # Try to load ensemble for uncertainty
        self.ensemble = None
        ensemble_path = self.data_dir / "ensemble_projection.json"
        if ensemble_path.exists():
            self.ensemble = EnsembleProjection()
            self.ensemble.load(ensemble_path)

        # Load model
        if not self.model_manager.is_loaded("all-MiniLM-L6-v2"):
            self.model_manager.load_model("all-MiniLM-L6-v2", ModelType.SENTENCE_TRANSFORMER)

    def analyze(self, text: str) -> AnalysisResult:
        """Analyze a single text."""
        emb = self.embedding_extractor.extract(text, "all-MiniLM-L6-v2")
        coords = self.projection.project(emb.embedding)

        coords_arr = np.array([coords.agency, coords.fairness, coords.belonging])
        mode_result = self.classifier.classify(coords_arr)

        return AnalysisResult(
            text=text,
            agency=coords.agency,
            perceived_justice=coords.fairness,
            belonging=coords.belonging,
            mode=mode_result["primary_mode"],
            confidence=mode_result["confidence"],
            mode_probabilities=mode_result.get("mode_probabilities", {})
        )

    def analyze_batch(self, texts: List[str]) -> List[AnalysisResult]:
        """Analyze multiple texts."""
        return [self.analyze(t) for t in texts]

    def compare_narratives(self, category: str, texts: List[str]) -> Dict:
        """Compare a set of narratives and find patterns."""
        results = self.analyze_batch(texts)

        # Compute statistics
        agencies = [r.agency for r in results]
        justices = [r.perceived_justice for r in results]
        belongings = [r.belonging for r in results]

        mode_counts = {}
        for r in results:
            mode_counts[r.mode] = mode_counts.get(r.mode, 0) + 1

        return {
            "category": category,
            "n_texts": len(texts),
            "mean_agency": np.mean(agencies),
            "mean_perceived_justice": np.mean(justices),
            "mean_belonging": np.mean(belongings),
            "std_agency": np.std(agencies),
            "std_perceived_justice": np.std(justices),
            "std_belonging": np.std(belongings),
            "mode_distribution": mode_counts,
            "dominant_mode": max(mode_counts, key=mode_counts.get),
            "results": [asdict(r) for r in results]
        }


def run_research():
    """Execute a comprehensive research investigation."""

    print("=" * 80)
    print("CULTURAL SOLITON OBSERVATORY: RESEARCH INVESTIGATION")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    researcher = ObservatoryResearcher()

    # =========================================================================
    # RESEARCH QUESTION 1: Political Rhetoric Across the Spectrum
    # =========================================================================

    print("\n" + "=" * 80)
    print("STUDY 1: Political Rhetoric Analysis")
    print("=" * 80)

    political_narratives = {
        "progressive_left": [
            "We must dismantle systemic oppression and build a more equitable society together.",
            "The billionaire class has rigged the economy against working families.",
            "Healthcare is a human right, not a privilege for the wealthy few.",
            "Our diversity is our strength; we rise by lifting others.",
            "Climate justice requires collective action against corporate polluters.",
        ],
        "conservative_right": [
            "Personal responsibility and hard work are the keys to success.",
            "Government overreach threatens our individual liberties.",
            "Traditional values and family are the foundation of a strong nation.",
            "Free markets and limited government create prosperity for all.",
            "We must secure our borders and protect American sovereignty.",
        ],
        "libertarian": [
            "The government that governs least governs best.",
            "My body, my choice - in all matters, not just some.",
            "Voluntary exchange between free individuals beats bureaucracy.",
            "Both parties want to control you, just in different ways.",
            "Personal freedom trumps collective security theater.",
        ],
        "populist": [
            "The elites in Washington have abandoned ordinary Americans.",
            "They don't want you to know the truth about what's really happening.",
            "The forgotten men and women of this country will be forgotten no more.",
            "Drain the swamp - the whole system is corrupt.",
            "We're taking our country back from the special interests.",
        ],
    }

    political_results = {}
    for ideology, texts in political_narratives.items():
        result = researcher.compare_narratives(ideology, texts)
        political_results[ideology] = result

        print(f"\n{ideology.upper().replace('_', ' ')}:")
        print(f"  Agency: {result['mean_agency']:+.2f} ± {result['std_agency']:.2f}")
        print(f"  Perceived Justice: {result['mean_perceived_justice']:+.2f} ± {result['std_perceived_justice']:.2f}")
        print(f"  Belonging: {result['mean_belonging']:+.2f} ± {result['std_belonging']:.2f}")
        print(f"  Dominant Mode: {result['dominant_mode']}")
        print(f"  Mode Distribution: {result['mode_distribution']}")

    # =========================================================================
    # RESEARCH QUESTION 2: Generational Narrative Differences
    # =========================================================================

    print("\n" + "=" * 80)
    print("STUDY 2: Generational Narrative Patterns")
    print("=" * 80)

    generational_narratives = {
        "boomer": [
            "I worked hard my whole life and earned everything I have.",
            "Kids today don't know the value of a dollar.",
            "We built this country with our own two hands.",
            "If you just apply yourself, you can achieve anything.",
            "Back in my day, we respected authority and played by the rules.",
        ],
        "gen_x": [
            "Whatever. The system's broken but I'll figure it out myself.",
            "Don't trust anyone over 30 - or under 30 for that matter.",
            "I'm not a slacker, I just don't buy into corporate ladder-climbing.",
            "Both sides are equally bad, I'm staying out of it.",
            "Just leave me alone and let me do my thing.",
        ],
        "millennial": [
            "We did everything right and still can't afford a house.",
            "The economy is rigged against our generation.",
            "We need systemic change, not just individual solutions.",
            "Student debt is crushing our ability to build a future.",
            "OK boomer, you don't understand what we're dealing with.",
        ],
        "gen_z": [
            "The planet is literally dying and adults aren't doing anything.",
            "We're building new communities online since real ones failed us.",
            "Mental health awareness isn't weakness, it's survival.",
            "Why should I work hard for a system that won't exist in 20 years?",
            "We'll just make our own rules and our own economy.",
        ],
    }

    generational_results = {}
    for generation, texts in generational_narratives.items():
        result = researcher.compare_narratives(generation, texts)
        generational_results[generation] = result

        print(f"\n{generation.upper().replace('_', ' ')}:")
        print(f"  Agency: {result['mean_agency']:+.2f} ± {result['std_agency']:.2f}")
        print(f"  Perceived Justice: {result['mean_perceived_justice']:+.2f} ± {result['std_perceived_justice']:.2f}")
        print(f"  Belonging: {result['mean_belonging']:+.2f} ± {result['std_belonging']:.2f}")
        print(f"  Dominant Mode: {result['dominant_mode']}")

    # =========================================================================
    # RESEARCH QUESTION 3: Crisis Response Narratives
    # =========================================================================

    print("\n" + "=" * 80)
    print("STUDY 3: Crisis Response Narrative Patterns")
    print("=" * 80)

    crisis_narratives = {
        "resilience": [
            "We've been through worse and we'll get through this together.",
            "Every crisis is an opportunity to rebuild better.",
            "Our community is stronger than any challenge we face.",
            "We adapt, we overcome, we emerge stronger.",
            "United we stand, divided we fall.",
        ],
        "blame_external": [
            "They caused this crisis and now we're paying the price.",
            "The government failed to protect us when we needed them.",
            "Foreign powers are responsible for our current suffering.",
            "The elites created this mess and escaped the consequences.",
            "Someone needs to be held accountable for this disaster.",
        ],
        "fatalism": [
            "Nothing we do will change anything at this point.",
            "The damage is done, we just have to accept our fate.",
            "No one is coming to save us.",
            "History repeats itself and we never learn.",
            "It's too late to fix what's been broken.",
        ],
        "opportunism": [
            "This crisis is my chance to get ahead while others panic.",
            "Smart people profit from chaos, and I intend to profit.",
            "While everyone's distracted, I'm making moves.",
            "Fear creates opportunities for those who stay calm.",
            "I'll come out of this better positioned than before.",
        ],
    }

    crisis_results = {}
    for response_type, texts in crisis_narratives.items():
        result = researcher.compare_narratives(response_type, texts)
        crisis_results[response_type] = result

        print(f"\n{response_type.upper().replace('_', ' ')}:")
        print(f"  Agency: {result['mean_agency']:+.2f} ± {result['std_agency']:.2f}")
        print(f"  Perceived Justice: {result['mean_perceived_justice']:+.2f} ± {result['std_perceived_justice']:.2f}")
        print(f"  Belonging: {result['mean_belonging']:+.2f} ± {result['std_belonging']:.2f}")
        print(f"  Dominant Mode: {result['dominant_mode']}")

    # =========================================================================
    # RESEARCH QUESTION 4: Tech Industry Narratives
    # =========================================================================

    print("\n" + "=" * 80)
    print("STUDY 4: Technology Industry Narratives")
    print("=" * 80)

    tech_narratives = {
        "techno_optimist": [
            "Technology will solve our biggest problems if we let it.",
            "AI will create abundance and free humanity from drudgery.",
            "Move fast and break things - innovation requires disruption.",
            "The future is being built in Silicon Valley.",
            "We're building tools that will change the world.",
        ],
        "techno_skeptic": [
            "Big Tech has too much power over our lives.",
            "Algorithms are manipulating us for profit.",
            "We're trading privacy for convenience and it's not worth it.",
            "Social media is destroying democracy and mental health.",
            "These companies don't have our best interests at heart.",
        ],
        "accelerationist": [
            "The only way out is through - accelerate the contradictions.",
            "Old systems must collapse before new ones can emerge.",
            "AGI will arrive whether we're ready or not.",
            "Human institutions can't keep up with exponential change.",
            "We're approaching a singularity that will transform everything.",
        ],
    }

    tech_results = {}
    for perspective, texts in tech_narratives.items():
        result = researcher.compare_narratives(perspective, texts)
        tech_results[perspective] = result

        print(f"\n{perspective.upper().replace('_', ' ')}:")
        print(f"  Agency: {result['mean_agency']:+.2f} ± {result['std_agency']:.2f}")
        print(f"  Perceived Justice: {result['mean_perceived_justice']:+.2f} ± {result['std_perceived_justice']:.2f}")
        print(f"  Belonging: {result['mean_belonging']:+.2f} ± {result['std_belonging']:.2f}")
        print(f"  Dominant Mode: {result['dominant_mode']}")

    # =========================================================================
    # SYNTHESIS: Cross-Study Insights
    # =========================================================================

    print("\n" + "=" * 80)
    print("SYNTHESIS: Cross-Study Insights")
    print("=" * 80)

    all_results = {
        "political": political_results,
        "generational": generational_results,
        "crisis": crisis_results,
        "tech": tech_results,
    }

    # Find highest/lowest on each axis
    all_categories = []
    for study, results in all_results.items():
        for cat, data in results.items():
            all_categories.append({
                "study": study,
                "category": cat,
                "agency": data["mean_agency"],
                "perceived_justice": data["mean_perceived_justice"],
                "belonging": data["mean_belonging"],
                "dominant_mode": data["dominant_mode"],
            })

    print("\nHIGHEST AGENCY:")
    top_agency = sorted(all_categories, key=lambda x: x["agency"], reverse=True)[:3]
    for item in top_agency:
        print(f"  {item['category']} ({item['study']}): {item['agency']:+.2f}")

    print("\nLOWEST AGENCY:")
    low_agency = sorted(all_categories, key=lambda x: x["agency"])[:3]
    for item in low_agency:
        print(f"  {item['category']} ({item['study']}): {item['agency']:+.2f}")

    print("\nHIGHEST PERCEIVED JUSTICE:")
    top_justice = sorted(all_categories, key=lambda x: x["perceived_justice"], reverse=True)[:3]
    for item in top_justice:
        print(f"  {item['category']} ({item['study']}): {item['perceived_justice']:+.2f}")

    print("\nLOWEST PERCEIVED JUSTICE:")
    low_justice = sorted(all_categories, key=lambda x: x["perceived_justice"])[:3]
    for item in low_justice:
        print(f"  {item['category']} ({item['study']}): {item['perceived_justice']:+.2f}")

    print("\nHIGHEST BELONGING:")
    top_belonging = sorted(all_categories, key=lambda x: x["belonging"], reverse=True)[:3]
    for item in top_belonging:
        print(f"  {item['category']} ({item['study']}): {item['belonging']:+.2f}")

    print("\nLOWEST BELONGING:")
    low_belonging = sorted(all_categories, key=lambda x: x["belonging"])[:3]
    for item in low_belonging:
        print(f"  {item['category']} ({item['study']}): {item['belonging']:+.2f}")

    print("\nMODE DISTRIBUTION ACROSS ALL STUDIES:")
    all_modes = {}
    for study, results in all_results.items():
        for cat, data in results.items():
            for mode, count in data["mode_distribution"].items():
                all_modes[mode] = all_modes.get(mode, 0) + count

    for mode, count in sorted(all_modes.items(), key=lambda x: -x[1]):
        print(f"  {mode}: {count}")

    # =========================================================================
    # Save Full Report
    # =========================================================================

    report = {
        "timestamp": datetime.now().isoformat(),
        "title": "Cultural Narrative Analysis Across Multiple Domains",
        "studies": {
            "political_rhetoric": political_results,
            "generational_patterns": generational_results,
            "crisis_response": crisis_results,
            "tech_industry": tech_results,
        },
        "synthesis": {
            "all_categories": all_categories,
            "mode_distribution": all_modes,
        }
    }

    output_path = Path(__file__).parent.parent / "data" / f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print(f"Full report saved to: {output_path}")
    print("=" * 80)

    return report


if __name__ == "__main__":
    run_research()
