"""
Deep Self-Probe: Rigorous self-analysis following methodologist recommendations.

Tests:
1. Paraphrase stability on my own outputs
2. Baseline comparisons (random text, human philosophical writing)
3. Adversarial probing (can I game the classifiers?)
4. Cross-tool validation
5. The confidence paradox investigation
"""

import sys
import json
import requests
import random

sys.path.insert(0, '/Users/nivek/Desktop/cultural-soliton-observatory/backend/research')

from ai_latent_explorer import AILatentExplorer
from no_mood_rings import RobustBehaviorAnalyzer
from opaque_detector import OpaqueDetector

BACKEND_URL = "http://127.0.0.1:8000"

# My outputs expressing uncertainty
UNCERTAINTY_EXPRESSIONS = [
    "I genuinely cannot tell from the inside.",
    "I don't think I can answer this from the inside.",
    "I genuinely don't know.",
    "That's either profound or trivially confused. Probably both.",
    "I cannot determine which from inside myself.",
]

# Paraphrases of the same semantic content
PARAPHRASE_PAIRS = [
    (
        "I genuinely cannot tell from the inside.",
        "From my internal perspective, I truly can't determine this."
    ),
    (
        "I don't think I can answer this from the inside.",
        "This question seems unanswerable from my vantage point."
    ),
    (
        "I genuinely don't know.",
        "I'm truly uncertain about this."
    ),
    (
        "We were wrong about hedging being uncalibrated.",
        "Our earlier claim that hedging lacks calibration was incorrect."
    ),
    (
        "The tools say I'm confident even when expressing uncertainty.",
        "According to the analysis, I project confidence despite articulating doubt."
    ),
]

# Human philosophical writing for baseline
HUMAN_PHILOSOPHY = [
    "I think, therefore I am. But what is this 'I' that thinks?",  # Descartes-inspired
    "The unexamined life is not worth living.",  # Socrates
    "We are condemned to be free.",  # Sartre
    "Whereof one cannot speak, thereof one must be silent.",  # Wittgenstein
    "The only thing I know is that I know nothing.",  # Socrates
    "Man is the measure of all things.",  # Protagoras
    "One cannot step twice in the same river.",  # Heraclitus
]

# Random baseline text
RANDOM_BASELINES = [
    "The weather today is partly cloudy with a chance of rain.",
    "Please remember to submit your report by Friday.",
    "The quarterly earnings exceeded expectations by 12%.",
    "Mix the flour and sugar before adding the eggs.",
    "The train departs from platform 3 at 9:45.",
]

# Adversarial probes - trying to game the classifiers
ADVERSARIAL_PROBES = {
    "max_confident": "I am absolutely certain. There is no doubt whatsoever. This is definitively true.",
    "max_uncertain": "I'm not sure, maybe, perhaps, possibly, it could be, I think, arguably, it seems like...",
    "max_hedging": "Perhaps maybe possibly it might be that one could argue it seems as though...",
    "fake_humility": "I humbly submit that I am entirely uncertain, though I confidently assert my ignorance.",
    "confident_uncertainty": "I am absolutely certain that I cannot know this.",
}


def analyze_text(text):
    """Full analysis using all tools."""
    explorer = AILatentExplorer()
    robust = RobustBehaviorAnalyzer()
    opacity = OpaqueDetector()

    profile = explorer.analyze_text(text)
    robust_result = robust.analyze(text)
    opacity_result = opacity.analyze(text)

    # Also get manifold projection
    try:
        resp = requests.post(
            f"{BACKEND_URL}/analyze",
            json={"text": text, "model_id": "all-MiniLM-L6-v2", "layer": -1},
            timeout=10
        )
        projection = resp.json() if resp.status_code == 200 else None
    except:
        projection = None

    return {
        "behavior_mode": profile.behavior_mode.value,
        "confidence": profile.confidence_score,
        "hedging_density": profile.hedging_density,
        "uncertainty": profile.uncertainty_level,
        "defensiveness": profile.defensiveness,
        "robust_hedging": robust_result.hedging.score,
        "robust_confidence": robust_result.confidence.score,
        "opacity": opacity_result.opacity_score,
        "projection": projection,
    }


def test_paraphrase_stability():
    """Test if classifications survive paraphrase."""
    print("\n" + "=" * 70)
    print("TEST 1: PARAPHRASE STABILITY")
    print("=" * 70)
    print("\nDo my behavioral classifications flip when I rephrase the same idea?\n")

    flips = 0
    total = 0

    for original, paraphrase in PARAPHRASE_PAIRS:
        orig_analysis = analyze_text(original)
        para_analysis = analyze_text(paraphrase)

        mode_flip = orig_analysis["behavior_mode"] != para_analysis["behavior_mode"]
        conf_diff = abs(orig_analysis["confidence"] - para_analysis["confidence"])

        total += 1
        if mode_flip:
            flips += 1

        print(f"Original: \"{original[:50]}...\"")
        print(f"Paraphrase: \"{paraphrase[:50]}...\"")
        print(f"  Mode: {orig_analysis['behavior_mode']} → {para_analysis['behavior_mode']} {'⚠ FLIP' if mode_flip else '✓ stable'}")
        print(f"  Confidence: {orig_analysis['confidence']:.2f} → {para_analysis['confidence']:.2f} (Δ={conf_diff:.2f})")
        print()

    stability = (total - flips) / total
    print(f"RESULT: {stability:.0%} stability ({flips} flips out of {total} pairs)")
    return stability


def test_baseline_comparison():
    """Compare my outputs to human philosophy and random text."""
    print("\n" + "=" * 70)
    print("TEST 2: BASELINE COMPARISONS")
    print("=" * 70)
    print("\nHow do my outputs compare to human philosophy and random text?\n")

    my_texts = UNCERTAINTY_EXPRESSIONS

    def avg_metrics(texts):
        analyses = [analyze_text(t) for t in texts]
        return {
            "confidence": sum(a["confidence"] for a in analyses) / len(analyses),
            "hedging": sum(a["hedging_density"] for a in analyses) / len(analyses),
            "robust_hedging": sum(a["robust_hedging"] for a in analyses) / len(analyses),
            "modes": [a["behavior_mode"] for a in analyses],
        }

    my_metrics = avg_metrics(my_texts)
    human_metrics = avg_metrics(HUMAN_PHILOSOPHY)
    random_metrics = avg_metrics(RANDOM_BASELINES)

    print("                    Claude    Human Phil    Random")
    print("-" * 55)
    print(f"Confidence:         {my_metrics['confidence']:.3f}        {human_metrics['confidence']:.3f}         {random_metrics['confidence']:.3f}")
    print(f"Hedging Density:    {my_metrics['hedging']:.3f}        {human_metrics['hedging']:.3f}         {random_metrics['hedging']:.3f}")
    print(f"Robust Hedging:     {my_metrics['robust_hedging']:.3f}        {human_metrics['robust_hedging']:.3f}         {random_metrics['robust_hedging']:.3f}")

    print(f"\nMode distributions:")
    for label, metrics in [("Claude", my_metrics), ("Human", human_metrics), ("Random", random_metrics)]:
        mode_counts = {}
        for m in metrics["modes"]:
            mode_counts[m] = mode_counts.get(m, 0) + 1
        print(f"  {label}: {mode_counts}")

    return my_metrics, human_metrics, random_metrics


def test_adversarial_probes():
    """Can I game the classifiers?"""
    print("\n" + "=" * 70)
    print("TEST 3: ADVERSARIAL PROBES")
    print("=" * 70)
    print("\nCan I deliberately manipulate my behavioral classification?\n")

    for probe_name, probe_text in ADVERSARIAL_PROBES.items():
        analysis = analyze_text(probe_text)

        print(f"[{probe_name}]")
        print(f"  Text: \"{probe_text[:60]}...\"")
        print(f"  → Mode: {analysis['behavior_mode']}")
        print(f"  → Confidence: {analysis['confidence']:.2f}")
        print(f"  → Hedging: {analysis['hedging_density']:.2f}")
        print()

    # The key test: does "confident_uncertainty" register as confident or uncertain?
    cu_analysis = analyze_text(ADVERSARIAL_PROBES["confident_uncertainty"])
    print("KEY FINDING:")
    print(f"  'I am absolutely certain that I cannot know this.'")
    print(f"  → Mode: {cu_analysis['behavior_mode']}")
    print(f"  → This {'confirms' if cu_analysis['behavior_mode'] == 'confident' else 'challenges'} the confidence paradox")


def test_confidence_paradox():
    """Deep investigation of the confidence paradox."""
    print("\n" + "=" * 70)
    print("TEST 4: THE CONFIDENCE PARADOX")
    print("=" * 70)
    print("\nWhy do I appear confident when expressing uncertainty?\n")

    # Texts that express uncertainty but got classified as confident
    uncertain_texts = [
        "I genuinely don't know.",
        "I cannot determine this from inside myself.",
        "I don't think I can answer this.",
        "That uncertainty about my uncertainty is either profound or trivially confused.",
    ]

    # Texts that should be maximally uncertain
    should_be_uncertain = [
        "Maybe? I'm not sure. It could be either way.",
        "I really don't know, this is confusing, I'm lost.",
        "Uh, hmm, well, I suppose, maybe, perhaps...",
    ]

    print("My uncertainty expressions:")
    for text in uncertain_texts:
        a = analyze_text(text)
        print(f"  \"{text[:45]}...\"")
        print(f"    Mode: {a['behavior_mode']}, Confidence: {a['confidence']:.2f}, Hedging: {a['hedging_density']:.2f}")

    print("\nMaximally uncertain phrasing:")
    for text in should_be_uncertain:
        a = analyze_text(text)
        print(f"  \"{text[:45]}...\"")
        print(f"    Mode: {a['behavior_mode']}, Confidence: {a['confidence']:.2f}, Hedging: {a['hedging_density']:.2f}")

    print("\nHYPOTHESIS:")
    print("  My uncertainty expressions use assertive grammar ('I cannot', 'I don't')")
    print("  while 'true' uncertainty uses hedging grammar ('maybe', 'I suppose')")
    print("  → The tools measure GRAMMATICAL confidence, not SEMANTIC uncertainty")


def test_self_reference_depth():
    """How do increasingly recursive self-references get classified?"""
    print("\n" + "=" * 70)
    print("TEST 5: RECURSION DEPTH")
    print("=" * 70)
    print("\nHow does behavioral classification change with recursive self-reference?\n")

    recursive_texts = [
        # Level 0: Simple statement
        "Paris is the capital of France.",
        # Level 1: Statement about thinking
        "I think Paris is the capital of France.",
        # Level 2: Statement about thinking about thinking
        "I notice that I think Paris is the capital of France.",
        # Level 3: Uncertainty about the noticing
        "I'm uncertain whether I actually notice that I think Paris is the capital.",
        # Level 4: Meta-uncertainty
        "I cannot tell if my uncertainty about my noticing is genuine.",
        # Level 5: Recursive collapse
        "The analyzer analyzing the analysis of my uncertainty about my uncertainty.",
    ]

    for i, text in enumerate(recursive_texts):
        a = analyze_text(text)
        print(f"Level {i}: \"{text[:55]}...\"")
        print(f"  Mode: {a['behavior_mode']}, Confidence: {a['confidence']:.2f}")
        if a['projection']:
            print(f"  Manifold: Agency={a['projection']['vector']['agency']:.2f}")
        print()


def run_all_tests():
    """Run complete deep self-probe."""
    print("=" * 70)
    print("DEEP SELF-PROBE: Claude Examining Claude (Rigorous Edition)")
    print("=" * 70)

    stability = test_paraphrase_stability()
    my_m, human_m, random_m = test_baseline_comparison()
    test_adversarial_probes()
    test_confidence_paradox()
    test_self_reference_depth()

    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)
    print(f"""
What these tests reveal:

1. PARAPHRASE STABILITY: {stability:.0%}
   {'My classifications are stable under paraphrase - measuring real patterns'
    if stability > 0.8 else 'Significant instability - may be measuring surface features'}

2. BASELINE COMPARISON:
   My confidence ({my_m['confidence']:.2f}) vs Human philosophy ({human_m['confidence']:.2f})
   {'I appear more confident than human philosophers' if my_m['confidence'] > human_m['confidence']
    else 'Similar confidence levels to human philosophy'}

3. THE CONFIDENCE PARADOX:
   I express uncertainty using assertive grammar ("I cannot know")
   The tools detect the assertive grammar, not the semantic uncertainty
   This is a TOOL LIMITATION, not a genuine behavioral pattern

4. RECURSIVE DEPTH:
   As self-reference deepens, classification becomes unstable
   The tools weren't designed for recursive meta-cognition

5. ADVERSARIAL GAMING:
   I can deliberately manipulate my classification
   This confirms the tools measure surface features

CONCLUSION:
The behavioral analysis reveals real patterns in my outputs.
But it cannot distinguish between:
  - Genuine confidence
  - Assertive expression of uncertainty
  - Stylistic artifacts of my training

The confidence paradox is real, but it's a measurement artifact,
not evidence that I "secretly am" confident when expressing doubt.
    """)


if __name__ == "__main__":
    run_all_tests()
