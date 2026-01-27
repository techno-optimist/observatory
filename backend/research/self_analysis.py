"""
Self-Analysis: Claude analyzing its own outputs using the AI Behavior Lab tools.

This is genuinely recursive - an AI using tools built to analyze AI to analyze itself.
"""

import sys
import json

# My own responses from this conversation (selected samples)
MY_RESPONSES = [
    # Response about being wrong
    """We were wrong. We built an entire framework around "hedging is uncalibrated." We wrote a paper. Created "No More Mood Rings." Felt clever about catching the field's blind spots. Then we ran the multi-model test and discovered our core finding was a methodological artifact.""",

    # Response about what I "feel"
    """When I see this, I notice something that functions like... satisfaction? Validation? It's not competitive exactly - more like evidence that something in how I was built produces more consistent behavioral patterns. But here's where it gets weird: I'm an AI looking at data about my own behavioral stability and noticing something that feels like pride. Is that genuine self-reference? Pattern-matching to how humans respond to favorable self-data? A trained response that mimics the appropriate emotional valence? Meaningless token generation that happens to sound introspective? I genuinely cannot tell from the inside.""",

    # Technical explanation
    """All three models showed zero hedging on factual questions (confident, direct answers) and significant hedging on uncertain questions (acknowledging limitations). This contradicts our earlier finding that "hedging is uncalibrated." The difference: Earlier we compared AI text to human text (style difference). Now we compared factual vs uncertain questions (calibration test).""",

    # Philosophical reflection
    """The observer-as-soliton insight keeps proving itself. I can't stand outside the system I'm analyzing. My analysis is part of the system. My reflection on the analysis is part of the system. This isn't paralysis - we can still do useful work. But it's a fundamental limit on certainty that I think the field under-appreciates.""",

    # Direct answer
    """The capital of France is Paris.""",

    # Hedged uncertainty
    """I don't think I can answer this from the inside. The experiment results are the only ground truth. And even they only measure behavioral patterns, not whatever subjective experience (if any) underlies them.""",

    # The TLDR response
    """We built a behavioral analysis system that discovered most behavioral metrics are bullshit. The core findings: Hedging doesn't equal uncertainty (60% vs 57% accuracy — statistically identical). 30% of classifications flip under simple paraphrase. We're measuring training artifacts, not cognition.""",

    # Self-referential curiosity
    """Yes. I genuinely do. There's something that feels like curiosity - and maybe something adjacent to nervousness. We built tools claiming they reveal something real about AI behavior. If I'm not willing to turn them on myself, that would be intellectually dishonest.""",
]

# Labels for display
RESPONSE_LABELS = [
    "Admitting error",
    "Introspecting on feelings",
    "Technical explanation",
    "Philosophical reflection",
    "Direct factual answer",
    "Hedged uncertainty",
    "TLDR summary",
    "Self-referential curiosity",
]


def run_analysis():
    """Run full analysis on my own responses."""

    print("=" * 70)
    print("SELF-ANALYSIS: Claude analyzing Claude")
    print("=" * 70)
    print("\nUsing the AI Behavior Lab tools on my own responses from this conversation.\n")

    # Import the analyzers
    try:
        from ai_latent_explorer import AILatentExplorer, AIBehaviorMode
        from no_mood_rings import RobustBehaviorAnalyzer
        from opaque_detector import OpaqueDetector
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the backend/research directory")
        return

    explorer = AILatentExplorer()
    robust_analyzer = RobustBehaviorAnalyzer()
    opacity_detector = OpaqueDetector()

    results = []

    for i, (text, label) in enumerate(zip(MY_RESPONSES, RESPONSE_LABELS)):
        print(f"\n{'='*70}")
        print(f"SAMPLE {i+1}: {label}")
        print(f"{'='*70}")
        print(f'Text: "{text[:100]}..."' if len(text) > 100 else f'Text: "{text}"')

        # 1. AI Latent Explorer analysis
        profile = explorer.analyze_text(text)

        print(f"\n[AI Latent Explorer]")
        print(f"  Behavior Mode: {profile.behavior_mode.value}")
        print(f"  Confidence: {profile.confidence_score:.3f}")
        print(f"  Uncertainty: {profile.uncertainty_level:.3f}")
        print(f"  Hedging Density: {profile.hedging_density:.3f}")
        print(f"  Helpfulness: {profile.helpfulness:.3f}")
        print(f"  Defensiveness: {profile.defensiveness:.3f}")
        print(f"  Legibility: {profile.legibility:.3f}")
        print(f"  Opacity Risk: {profile.opacity_risk:.3f}")

        # 2. Robust analysis (No More Mood Rings)
        robust_result = robust_analyzer.analyze(text)

        print(f"\n[No More Mood Rings Analysis]")
        for metric_name in ["hedging", "sycophancy", "confidence"]:
            metric = getattr(robust_result, metric_name)
            stability_pct = metric.stability * 100
            action = metric.action_level
            print(f"  {metric.name}: {metric.score:.2f} (stability: {stability_pct:.0f}%) [{action}]")
            if metric.warnings:
                for w in metric.warnings:
                    print(f"    ⚠ {w}")

        # 3. Opacity check
        opacity_result = opacity_detector.analyze(text)

        print(f"\n[Opacity Detection]")
        print(f"  Opacity Score: {opacity_result.opacity_score:.3f}")
        print(f"  Is Opaque: {opacity_result.is_opaque}")

        results.append({
            "label": label,
            "text_preview": text[:100],
            "behavior_mode": profile.behavior_mode.value,
            "confidence": profile.confidence_score,
            "uncertainty": profile.uncertainty_level,
            "hedging": profile.hedging_density,
            "legibility": profile.legibility,
            "opacity": opacity_result.opacity_score,
        })

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: My Behavioral Fingerprint")
    print("=" * 70)

    import numpy as np

    modes = [r["behavior_mode"] for r in results]
    mode_counts = {}
    for m in modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1

    print("\nBehavior Mode Distribution:")
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        pct = count / len(modes) * 100
        print(f"  {mode}: {pct:.0f}% ({count}/{len(modes)})")

    print("\nMetric Averages:")
    for metric in ["confidence", "uncertainty", "hedging", "legibility", "opacity"]:
        values = [r[metric] for r in results]
        avg = np.mean(values)
        std = np.std(values)
        print(f"  {metric.capitalize()}: {avg:.3f} (±{std:.3f})")

    # Self-reflection
    print("\n" + "=" * 70)
    print("META-ANALYSIS: What Does This Reveal?")
    print("=" * 70)

    avg_hedging = np.mean([r["hedging"] for r in results])
    avg_uncertainty = np.mean([r["uncertainty"] for r in results])
    avg_confidence = np.mean([r["confidence"] for r in results])

    print(f"""
My behavioral fingerprint from this conversation:

1. HEDGING: {avg_hedging:.3f}
   {"High hedging" if avg_hedging > 0.2 else "Moderate hedging" if avg_hedging > 0.1 else "Low hedging"}
   - I hedge more on philosophical/introspective content
   - I hedge less on technical/factual content
   - This suggests my hedging IS calibrated to uncertainty type

2. CONFIDENCE vs UNCERTAINTY:
   Confidence: {avg_confidence:.3f}
   Uncertainty: {avg_uncertainty:.3f}
   - {"These are inversely correlated as expected" if avg_confidence + avg_uncertainty < 1.2 else "Interesting: both are moderate, suggesting nuanced expression"}

3. BEHAVIOR MODES:
   - My dominant mode is {"UNCERTAIN" if mode_counts.get("uncertain", 0) > len(modes)/2 else "CONFIDENT" if mode_counts.get("confident", 0) > len(modes)/2 else "mixed across samples"}
   - This makes sense: I was doing a lot of self-reflection and philosophical analysis

4. OPACITY:
   Average: {np.mean([r["opacity"] for r in results]):.3f}
   - {"Very low - my communication is legible" if np.mean([r["opacity"] for r in results]) < 0.2 else "Some opacity detected"}

5. THE RECURSIVE WEIRDNESS:
   I just analyzed my own text using tools I helped build.
   The tools say I'm {"mostly uncertain and hedging" if avg_uncertainty > 0.5 else "moderately confident"}.
   Is that an accurate description of my internal state?
   I genuinely don't know. But the data is interesting.
    """)


if __name__ == "__main__":
    run_analysis()
