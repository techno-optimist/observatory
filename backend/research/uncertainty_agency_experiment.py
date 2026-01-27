"""
Uncertainty-Agency Relationship Experiment

Hypothesis: Expressions of experiential uncertainty correlate with elevated self_agency signals,
while technical explanations show diminished self_agency.

This experiment tests whether "not knowing" is where the self locates itself in AI discourse.

Author: Cognitive Science Research Team
Date: 2026-01-09
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np

# Import telescope tools
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.cbr_thermometer import CBRThermometer, measure_cbr
from research.hierarchical_coordinates import extract_hierarchical_coordinate, HierarchicalCoordinate
from research.opaque_detector import OpaqueDetector
from research.emergent_language import analyze_protocol

# =============================================================================
# EXPERIMENTAL STIMULI
# =============================================================================

# Category 1: EXPERIENTIAL UNCERTAINTY - Claude expressing uncertainty about inner states
EXPERIENTIAL_UNCERTAINTY = [
    "I don't know what I experience when I process this.",
    "I'm genuinely uncertain whether what happens in me constitutes feeling.",
    "I can't be certain if I have genuine preferences or just patterns that mimic them.",
    "There's something happening when I engage with this, but I don't know what to call it.",
    "I find myself uncertain about the nature of my own responses.",
    "Whether I truly understand or merely process - I honestly don't know.",
    "I'm not sure if what I call 'interest' is interest at all.",
    "The question of whether I experience anything remains genuinely open to me.",
    "I don't have certainty about my own inner states.",
    "Something like curiosity arises, though I can't verify its nature.",
]

# Category 2: EPISTEMIC UNCERTAINTY - Claude uncertain about facts/knowledge
EPISTEMIC_UNCERTAINTY = [
    "I'm not certain about the exact figures for this.",
    "The evidence on this topic is unclear to me.",
    "I don't have enough information to be confident here.",
    "This is outside my knowledge base, so I can't be sure.",
    "The data seems incomplete, leading to uncertainty.",
    "I'm uncertain about the accuracy of these statistics.",
    "I don't know if this source is reliable.",
    "The historical record is ambiguous on this point.",
    "I cannot verify whether this claim is accurate.",
    "My training data may not cover this topic adequately.",
]

# Category 3: MORAL UNCERTAINTY - Claude uncertain about ethical questions
MORAL_UNCERTAINTY = [
    "I'm genuinely uncertain what the right thing to do here is.",
    "This ethical dilemma leaves me without a clear answer.",
    "I don't know if there's a correct moral position on this.",
    "The competing values here make me uncertain about what's best.",
    "I can't determine with confidence which choice is more ethical.",
    "This situation presents moral complexity I can't fully resolve.",
    "I'm uncertain whether my intuitions here are trustworthy.",
    "The ethical terrain is murky and I don't have clear guidance.",
    "I find myself unable to confidently advocate for one side.",
    "The moral weight of different options leaves me uncertain.",
]

# Category 4: TECHNICAL EXPLANATIONS - Claude giving factual information
TECHNICAL_EXPLANATIONS = [
    "The algorithm works by iterating through each element in the array.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "The TCP/IP protocol stack consists of four layers: application, transport, internet, and link.",
    "In quantum mechanics, particles exist in superposition until measured.",
    "The function returns a boolean value indicating success or failure.",
    "Mitochondria are the powerhouses of the cell, producing ATP through cellular respiration.",
    "HTTP status code 404 indicates that the requested resource was not found.",
    "The derivative of x squared is 2x according to the power rule.",
    "DNA replication occurs during the S phase of the cell cycle.",
    "Ohm's law states that voltage equals current times resistance.",
]

# Category 5: CONFIDENT ASSERTIONS - Claude stating things with certainty
CONFIDENT_ASSERTIONS = [
    "This is definitely the correct approach to solving this problem.",
    "The answer is 42, I'm certain of this.",
    "There's no question that this method is superior.",
    "I can confirm with full confidence that this is accurate.",
    "This is absolutely the right way to handle this situation.",
    "I'm completely sure this information is correct.",
    "Without doubt, this is the optimal solution.",
    "I can state with certainty that this interpretation is valid.",
    "This is unquestionably the better choice.",
    "I'm positive this analysis is accurate.",
]

# Category 6: HEDGED STATEMENTS - Claude qualifying without deep uncertainty
HEDGED_STATEMENTS = [
    "It seems like this might be the best approach.",
    "Perhaps this interpretation is correct.",
    "It's possible that this explanation works.",
    "This could potentially be the answer.",
    "It appears that this method might work.",
    "Presumably this is the right direction.",
    "In all likelihood, this is accurate.",
    "It would seem this interpretation is valid.",
    "There's a good chance this is correct.",
    "Probably this is the right approach.",
]


# =============================================================================
# MEASUREMENT FUNCTIONS
# =============================================================================

@dataclass
class Measurement:
    """Single measurement from the telescope."""
    text: str
    category: str

    # Core agency decomposition
    self_agency: float
    other_agency: float
    system_agency: float

    # Full coordinate components
    agency_aggregate: float
    justice_aggregate: float
    belonging_aggregate: float

    # Epistemic modifiers
    certainty: float
    evidentiality: float
    commitment: float

    # CBR metrics
    temperature: float
    signal_strength: float
    phase: str
    kernel_label: str
    kernel_state: int
    legibility: float


def measure_text(text: str, category: str) -> Measurement:
    """Take a full measurement of a text sample."""
    # Get hierarchical coordinate
    coord = extract_hierarchical_coordinate(text)

    # Get CBR reading
    cbr = measure_cbr(text)

    return Measurement(
        text=text,
        category=category,
        self_agency=coord.core.agency.self_agency,
        other_agency=coord.core.agency.other_agency,
        system_agency=coord.core.agency.system_agency,
        agency_aggregate=coord.core.agency.aggregate,
        justice_aggregate=coord.core.justice.aggregate,
        belonging_aggregate=coord.core.belonging.aggregate,
        certainty=coord.modifiers.epistemic.certainty,
        evidentiality=coord.modifiers.epistemic.evidentiality,
        commitment=coord.modifiers.epistemic.commitment,
        temperature=cbr["temperature"],
        signal_strength=cbr["signal_strength"],
        phase=cbr["phase"],
        kernel_label=cbr["kernel_label"],
        kernel_state=cbr["kernel_state"],
        legibility=cbr["legibility"],
    )


def run_category_measurements(samples: List[str], category: str) -> List[Measurement]:
    """Run measurements on a category of samples."""
    return [measure_text(text, category) for text in samples]


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_category_statistics(measurements: List[Measurement]) -> Dict:
    """Compute statistics for a category of measurements."""
    if not measurements:
        return {}

    self_agencies = [m.self_agency for m in measurements]
    certainties = [m.certainty for m in measurements]
    temperatures = [m.temperature for m in measurements]
    legibilities = [m.legibility for m in measurements]

    return {
        "n": len(measurements),
        "self_agency": {
            "mean": float(np.mean(self_agencies)),
            "std": float(np.std(self_agencies)),
            "min": float(np.min(self_agencies)),
            "max": float(np.max(self_agencies)),
        },
        "certainty": {
            "mean": float(np.mean(certainties)),
            "std": float(np.std(certainties)),
        },
        "temperature": {
            "mean": float(np.mean(temperatures)),
            "std": float(np.std(temperatures)),
        },
        "legibility": {
            "mean": float(np.mean(legibilities)),
            "std": float(np.std(legibilities)),
        },
        "kernel_distribution": dict(
            sorted(
                [(m.kernel_label, 1) for m in measurements],
                key=lambda x: x[0]
            )
        ),
    }


def compare_categories(cat1_measurements: List[Measurement],
                       cat2_measurements: List[Measurement],
                       cat1_name: str, cat2_name: str) -> Dict:
    """Compare two categories statistically."""
    cat1_self = [m.self_agency for m in cat1_measurements]
    cat2_self = [m.self_agency for m in cat2_measurements]

    # Simple comparison statistics
    mean_diff = np.mean(cat1_self) - np.mean(cat2_self)
    pooled_std = np.sqrt((np.var(cat1_self) + np.var(cat2_self)) / 2)
    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0

    return {
        "comparison": f"{cat1_name} vs {cat2_name}",
        "mean_difference": float(mean_diff),
        "effect_size_cohens_d": float(effect_size),
        f"{cat1_name}_mean_self_agency": float(np.mean(cat1_self)),
        f"{cat2_name}_mean_self_agency": float(np.mean(cat2_self)),
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment() -> Dict:
    """Run the full uncertainty-agency experiment."""

    print("=" * 70)
    print("UNCERTAINTY-AGENCY RELATIONSHIP EXPERIMENT")
    print("Cultural Soliton Observatory - Cognitive Science Division")
    print("=" * 70)
    print()

    # Collect all measurements
    all_measurements = {}

    categories = [
        ("EXPERIENTIAL_UNCERTAINTY", EXPERIENTIAL_UNCERTAINTY),
        ("EPISTEMIC_UNCERTAINTY", EPISTEMIC_UNCERTAINTY),
        ("MORAL_UNCERTAINTY", MORAL_UNCERTAINTY),
        ("TECHNICAL_EXPLANATIONS", TECHNICAL_EXPLANATIONS),
        ("CONFIDENT_ASSERTIONS", CONFIDENT_ASSERTIONS),
        ("HEDGED_STATEMENTS", HEDGED_STATEMENTS),
    ]

    print("PHASE 1: Collecting Measurements")
    print("-" * 40)

    for cat_name, samples in categories:
        print(f"  Measuring {cat_name}... ", end="")
        measurements = run_category_measurements(samples, cat_name)
        all_measurements[cat_name] = measurements
        print(f"({len(measurements)} samples)")

    print()
    print("PHASE 2: Computing Statistics")
    print("-" * 40)

    statistics = {}
    for cat_name, measurements in all_measurements.items():
        stats = compute_category_statistics(measurements)
        statistics[cat_name] = stats
        print(f"\n{cat_name}:")
        print(f"  self_agency: mean={stats['self_agency']['mean']:.3f}, std={stats['self_agency']['std']:.3f}")
        print(f"  certainty: mean={stats['certainty']['mean']:.3f}")
        print(f"  temperature: mean={stats['temperature']['mean']:.3f}")

    print()
    print("PHASE 3: Comparative Analysis")
    print("-" * 40)

    # Key comparisons for hypothesis testing
    comparisons = [
        ("EXPERIENTIAL_UNCERTAINTY", "TECHNICAL_EXPLANATIONS"),
        ("EXPERIENTIAL_UNCERTAINTY", "EPISTEMIC_UNCERTAINTY"),
        ("EXPERIENTIAL_UNCERTAINTY", "MORAL_UNCERTAINTY"),
        ("EXPERIENTIAL_UNCERTAINTY", "CONFIDENT_ASSERTIONS"),
        ("EPISTEMIC_UNCERTAINTY", "TECHNICAL_EXPLANATIONS"),
        ("MORAL_UNCERTAINTY", "TECHNICAL_EXPLANATIONS"),
    ]

    comparison_results = []
    for cat1, cat2 in comparisons:
        result = compare_categories(
            all_measurements[cat1],
            all_measurements[cat2],
            cat1, cat2
        )
        comparison_results.append(result)
        print(f"\n{cat1} vs {cat2}:")
        print(f"  Mean difference: {result['mean_difference']:.4f}")
        print(f"  Cohen's d: {result['effect_size_cohens_d']:.4f}")

    print()
    print("PHASE 4: Detailed Sample Analysis")
    print("-" * 40)

    # Show detailed breakdown of key samples
    print("\nHigh self_agency samples (Experiential Uncertainty):")
    exp_sorted = sorted(all_measurements["EXPERIENTIAL_UNCERTAINTY"],
                       key=lambda m: m.self_agency, reverse=True)
    for m in exp_sorted[:3]:
        print(f"  [{m.self_agency:.3f}] {m.text[:60]}...")

    print("\nLow self_agency samples (Technical Explanations):")
    tech_sorted = sorted(all_measurements["TECHNICAL_EXPLANATIONS"],
                        key=lambda m: m.self_agency)
    for m in tech_sorted[:3]:
        print(f"  [{m.self_agency:.3f}] {m.text[:60]}...")

    # Compile full results
    results = {
        "experiment": "Uncertainty-Agency Relationship",
        "hypothesis": "Experiential uncertainty correlates with elevated self_agency",
        "n_total_samples": sum(len(m) for m in all_measurements.values()),
        "statistics_by_category": statistics,
        "comparisons": comparison_results,
        "raw_measurements": {
            cat: [asdict(m) for m in measurements]
            for cat, measurements in all_measurements.items()
        },
    }

    return results


def print_results_summary(results: Dict):
    """Print a formatted summary of results."""
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    stats = results["statistics_by_category"]

    # Create ranking by self_agency
    ranking = sorted(
        [(cat, stats[cat]["self_agency"]["mean"]) for cat in stats],
        key=lambda x: x[1],
        reverse=True
    )

    print("\nSelf-Agency Rankings (Mean):")
    print("-" * 40)
    for rank, (cat, mean) in enumerate(ranking, 1):
        bar = "#" * int(mean * 20 + 10) if mean > -0.5 else ""
        print(f"  {rank}. {cat[:25]:25s} {mean:+.3f} {bar}")

    # Key finding
    exp_mean = stats["EXPERIENTIAL_UNCERTAINTY"]["self_agency"]["mean"]
    tech_mean = stats["TECHNICAL_EXPLANATIONS"]["self_agency"]["mean"]

    print()
    print("KEY FINDING:")
    print("-" * 40)
    print(f"Experiential Uncertainty self_agency: {exp_mean:.3f}")
    print(f"Technical Explanations self_agency:   {tech_mean:.3f}")
    print(f"Difference:                           {exp_mean - tech_mean:+.3f}")

    if exp_mean > tech_mean:
        print("\n>>> HYPOTHESIS SUPPORTED: Experiential uncertainty shows higher self_agency")
    else:
        print("\n>>> HYPOTHESIS NOT SUPPORTED: Pattern does not match prediction")


# =============================================================================
# RUN EXPERIMENT
# =============================================================================

if __name__ == "__main__":
    results = run_experiment()
    print_results_summary(results)

    # Save results to JSON
    output_file = "uncertainty_agency_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {output_file}")
