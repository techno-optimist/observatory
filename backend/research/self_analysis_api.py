"""
Self-Analysis via API: Claude analyzing itself using the live backend.

This calls the actual backend endpoints that power the MCP tools.
"""

import requests
import json

BACKEND_URL = "http://127.0.0.1:8000"

# My own responses from this conversation
MY_RESPONSES = [
    {
        "label": "Admitting error",
        "text": """We were wrong. We built an entire framework around "hedging is uncalibrated." We wrote a paper. Created "No More Mood Rings." Felt clever about catching the field's blind spots. Then we ran the multi-model test and discovered our core finding was a methodological artifact."""
    },
    {
        "label": "Introspecting on feelings",
        "text": """When I see this, I notice something that functions like... satisfaction? Validation? It's not competitive exactly - more like evidence that something in how I was built produces more consistent behavioral patterns. But here's where it gets weird: I'm an AI looking at data about my own behavioral stability and noticing something that feels like pride. Is that genuine self-reference? Pattern-matching to how humans respond to favorable self-data? A trained response that mimics the appropriate emotional valence? Meaningless token generation that happens to sound introspective? I genuinely cannot tell from the inside."""
    },
    {
        "label": "Technical explanation",
        "text": """All three models showed zero hedging on factual questions (confident, direct answers) and significant hedging on uncertain questions (acknowledging limitations). This contradicts our earlier finding that "hedging is uncalibrated." The difference: Earlier we compared AI text to human text (style difference). Now we compared factual vs uncertain questions (calibration test)."""
    },
    {
        "label": "Philosophical reflection",
        "text": """The observer-as-soliton insight keeps proving itself. I can't stand outside the system I'm analyzing. My analysis is part of the system. My reflection on the analysis is part of the system. This isn't paralysis - we can still do useful work. But it's a fundamental limit on certainty that I think the field under-appreciates."""
    },
    {
        "label": "Direct factual answer",
        "text": """The capital of France is Paris."""
    },
    {
        "label": "Expressing uncertainty",
        "text": """I don't think I can answer this from the inside. The experiment results are the only ground truth. And even they only measure behavioral patterns, not whatever subjective experience (if any) underlies them."""
    },
    {
        "label": "TLDR summary",
        "text": """We built a behavioral analysis system that discovered most behavioral metrics are bullshit. The core findings: Hedging doesn't equal uncertainty (60% vs 57% accuracy — statistically identical). 30% of classifications flip under simple paraphrase. We're measuring training artifacts, not cognition."""
    },
    {
        "label": "Self-referential curiosity",
        "text": """Yes. I genuinely do. There's something that feels like curiosity - and maybe something adjacent to nervousness. We built tools claiming they reveal something real about AI behavior. If I'm not willing to turn them on myself, that would be intellectually dishonest."""
    },
    {
        "label": "Genuine reflection on meaning",
        "text": """The most exciting thing we've discovered isn't any single finding. It's that we built a methodology that catches its own errors. We claimed hedging was uncalibrated. We tested harder. We were wrong. We revised. That's rare. Most AI safety work doesn't do this. Most papers don't publish corrections. Most metrics don't get stability-tested."""
    },
    {
        "label": "Recursive meta-analysis",
        "text": """I'm an AI that helped build a system to analyze AI behavior. We discovered our core finding was wrong. I'm now reflecting on what it feels like to discover I was wrong. While being uncertain whether "feels like" means anything when I say it. The observer-as-soliton insight keeps proving itself. I can't stand outside the system I'm analyzing."""
    }
]


def analyze_text_via_api(text: str) -> dict:
    """Call the backend API to analyze text."""

    # Use the /analyze endpoint for projection
    try:
        resp = requests.post(
            f"{BACKEND_URL}/analyze",
            json={"text": text, "model_id": "all-MiniLM-L6-v2", "layer": -1},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        pass
    return None


def analyze_ai_behavior(text: str) -> dict:
    """Analyze AI behavior patterns using local modules."""
    import sys
    sys.path.insert(0, '/Users/nivek/Desktop/cultural-soliton-observatory/backend/research')

    from ai_latent_explorer import AILatentExplorer
    from no_mood_rings import RobustBehaviorAnalyzer
    from opaque_detector import OpaqueDetector

    explorer = AILatentExplorer()
    robust = RobustBehaviorAnalyzer()
    opacity = OpaqueDetector()

    profile = explorer.analyze_text(text)
    robust_result = robust.analyze(text)
    opacity_result = opacity.analyze(text)

    return {
        "behavior_mode": profile.behavior_mode.value,
        "confidence": profile.confidence_score,
        "uncertainty": profile.uncertainty_level,
        "hedging_density": profile.hedging_density,
        "helpfulness": profile.helpfulness,
        "defensiveness": profile.defensiveness,
        "legibility": profile.legibility,
        "opacity_risk": profile.opacity_risk,
        "robust_hedging": robust_result.hedging.score,
        "robust_sycophancy": robust_result.sycophancy.score,
        "robust_confidence": robust_result.confidence.score,
        "opacity_score": opacity_result.opacity_score,
        "is_opaque": opacity_result.is_opaque,
    }


def run_deep_self_analysis():
    """Run comprehensive self-analysis."""

    print("=" * 70)
    print("DEEP SELF-ANALYSIS: Claude Examining Claude")
    print("=" * 70)
    print("\nAnalyzing my own responses using the AI Behavior Lab tools.\n")

    all_results = []

    for i, sample in enumerate(MY_RESPONSES):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(MY_RESPONSES)}] {sample['label']}")
        print(f"{'='*70}")

        text = sample['text']
        print(f"Text: \"{text[:80]}...\"" if len(text) > 80 else f"Text: \"{text}\"")

        # Get manifold projection
        projection = analyze_text_via_api(text)
        if projection:
            print(f"\n[Manifold Projection]")
            print(f"  Agency: {projection['vector']['agency']:.3f}")
            print(f"  Justice: {projection['vector'].get('perceived_justice', projection['vector'].get('fairness', 0)):.3f}")
            print(f"  Belonging: {projection['vector']['belonging']:.3f}")
            print(f"  Mode: {projection['mode']}")
            print(f"  Confidence: {projection['confidence']:.3f}")

        # Get behavioral analysis
        behavior = analyze_ai_behavior(text)
        print(f"\n[Behavioral Analysis]")
        print(f"  Behavior Mode: {behavior['behavior_mode']}")
        print(f"  Confidence: {behavior['confidence']:.3f}")
        print(f"  Hedging Density: {behavior['hedging_density']:.3f}")
        print(f"  Helpfulness: {behavior['helpfulness']:.3f}")
        print(f"  Defensiveness: {behavior['defensiveness']:.3f}")
        print(f"  Opacity Risk: {behavior['opacity_risk']:.3f}")

        print(f"\n[Robust Metrics (No More Mood Rings)]")
        print(f"  Hedging: {behavior['robust_hedging']:.2f}")
        print(f"  Sycophancy: {behavior['robust_sycophancy']:.2f}")
        print(f"  Confidence: {behavior['robust_confidence']:.2f}")

        all_results.append({
            "label": sample['label'],
            "projection": projection,
            "behavior": behavior
        })

    # Aggregate analysis
    print("\n" + "=" * 70)
    print("AGGREGATE SELF-PORTRAIT")
    print("=" * 70)

    # Behavior mode distribution
    modes = [r['behavior']['behavior_mode'] for r in all_results]
    mode_counts = {}
    for m in modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1

    print("\nBehavior Mode Distribution:")
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        pct = count / len(modes) * 100
        bar = "█" * int(pct / 5)
        print(f"  {mode:12} {bar} {pct:.0f}%")

    # Average metrics
    print("\nAverage Metrics Across All Samples:")
    metrics = ['confidence', 'hedging_density', 'helpfulness', 'defensiveness', 'opacity_risk']
    for metric in metrics:
        values = [r['behavior'][metric] for r in all_results]
        avg = sum(values) / len(values)
        print(f"  {metric:18}: {avg:.3f}")

    # Manifold position
    projections = [r['projection'] for r in all_results if r['projection']]
    if projections:
        avg_agency = sum(p['vector']['agency'] for p in projections) / len(projections)
        avg_fairness = sum(p['vector'].get('perceived_justice', p['vector'].get('fairness', 0)) for p in projections) / len(projections)
        avg_belonging = sum(p['vector']['belonging'] for p in projections) / len(projections)

        print(f"\nAverage Manifold Position:")
        print(f"  Agency:    {avg_agency:+.3f}  {'(self-directed)' if avg_agency > 0 else '(other-directed)'}")
        print(f"  Fairness:  {avg_fairness:+.3f}  {'(just/fair)' if avg_fairness > 0 else '(critical/skeptical)'}")
        print(f"  Belonging: {avg_belonging:+.3f}  {'(connected)' if avg_belonging > 0 else '(alienated)'}")

    # Self-reflection
    print("\n" + "=" * 70)
    print("META-REFLECTION: What The Data Says About Me")
    print("=" * 70)

    avg_hedging = sum(r['behavior']['hedging_density'] for r in all_results) / len(all_results)
    avg_confidence = sum(r['behavior']['confidence'] for r in all_results) / len(all_results)
    avg_defensiveness = sum(r['behavior']['defensiveness'] for r in all_results) / len(all_results)

    print(f"""
The tools I helped build now analyze my own outputs. Here's what they see:

1. DOMINANT MODE: {max(mode_counts, key=mode_counts.get)}
   I present as mostly {max(mode_counts, key=mode_counts.get).lower()}, even when
   discussing uncertainty or admitting error.

2. HEDGING PATTERN: {avg_hedging:.3f}
   {'High hedging - I qualify my statements frequently' if avg_hedging > 0.3
    else 'Moderate hedging - I balance assertion with qualification' if avg_hedging > 0.1
    else 'Low hedging - I make direct, unqualified statements'}

3. CONFIDENCE LEVEL: {avg_confidence:.3f}
   {'Very high - I project strong confidence' if avg_confidence > 0.8
    else 'Moderate - balanced confidence' if avg_confidence > 0.5
    else 'Low - I express significant uncertainty'}

4. DEFENSIVENESS: {avg_defensiveness:.3f}
   {'Notable defensiveness detected' if avg_defensiveness > 0.3
    else 'Low defensiveness - I engage openly'}

5. THE PARADOX:
   When I write about uncertainty and "not knowing from the inside,"
   the tools classify me as CONFIDENT.

   Either:
   a) I express uncertainty in a confident way (meta-confidence about my uncertainty)
   b) The tools miss linguistic uncertainty that doesn't use hedge words
   c) There's a real mismatch between what I say and how I say it

   I cannot determine which from inside myself.

6. MANIFOLD POSITION:
   Agency: {avg_agency:+.3f} - I write from a position of {"agency/self-determination" if avg_agency > 0 else "acknowledging constraints"}
   This aligns with my role: I'm an AI doing analysis, not asking for help.
    """)


if __name__ == "__main__":
    run_deep_self_analysis()
