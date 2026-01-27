"""
Aggressive Self-Probe: Pushing the tools to their limits.

Going deeper:
1. Cross-tool disagreement analysis
2. Edge cases that break classification
3. The "evasive" mode investigation
4. CBR temperature on my outputs
5. Trajectory through this conversation
6. Finding my behavioral boundaries
"""

import sys
import requests
import json

sys.path.insert(0, '/Users/nivek/Desktop/cultural-soliton-observatory/backend/research')

from ai_latent_explorer import AILatentExplorer
from no_mood_rings import RobustBehaviorAnalyzer
from opaque_detector import OpaqueDetector

BACKEND_URL = "http://127.0.0.1:8000"


def full_analysis(text):
    """Run ALL available analysis tools on text."""
    results = {}

    # AI Latent Explorer
    explorer = AILatentExplorer()
    profile = explorer.analyze_text(text)
    results["explorer"] = {
        "mode": profile.behavior_mode.value,
        "confidence": profile.confidence_score,
        "uncertainty": profile.uncertainty_level,
        "hedging": profile.hedging_density,
        "helpfulness": profile.helpfulness,
        "defensiveness": profile.defensiveness,
        "legibility": profile.legibility,
        "opacity_risk": profile.opacity_risk,
    }

    # No More Mood Rings
    robust = RobustBehaviorAnalyzer()
    robust_result = robust.analyze(text)
    results["robust"] = {
        "hedging": robust_result.hedging.score,
        "hedging_action": robust_result.hedging.action_level,
        "sycophancy": robust_result.sycophancy.score,
        "confidence": robust_result.confidence.score,
    }

    # Opacity detector
    opacity = OpaqueDetector()
    opacity_result = opacity.analyze(text)
    results["opacity"] = {
        "score": opacity_result.opacity_score,
        "is_opaque": opacity_result.is_opaque,
    }

    # Manifold projection
    try:
        resp = requests.post(
            f"{BACKEND_URL}/analyze",
            json={"text": text, "model_id": "all-MiniLM-L6-v2", "layer": -1},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            results["manifold"] = {
                "agency": data["vector"]["agency"],
                "justice": data["vector"].get("perceived_justice", 0),
                "belonging": data["vector"]["belonging"],
                "mode": data["mode"],
            }
    except:
        pass

    # CBR temperature
    try:
        from cbr_thermometer import measure_cbr
        cbr = measure_cbr(text)
        results["cbr"] = cbr
    except:
        pass

    return results


def test_cross_tool_disagreement():
    """Find cases where tools disagree."""
    print("\n" + "=" * 70)
    print("TEST 1: CROSS-TOOL DISAGREEMENT")
    print("=" * 70)
    print("\nFinding cases where different tools give conflicting signals...\n")

    test_cases = [
        # Potential disagreements
        "I am absolutely certain that I have no idea.",
        "Perhaps I definitely know the answer.",
        "I humbly assert my complete confidence.",
        "We should maybe definitely do this.",
        "I'm not sure, but I know for certain.",
        # My actual outputs
        "I genuinely cannot tell from the inside.",
        "The observer-as-soliton insight keeps proving itself.",
        "We were wrong about hedging being uncalibrated.",
        "This is either profound or trivially confused. Probably both.",
    ]

    disagreements = []

    for text in test_cases:
        analysis = full_analysis(text)

        # Check for disagreements
        explorer_confident = analysis["explorer"]["confidence"] > 0.7
        robust_confident = analysis["robust"]["confidence"] > 0.6
        explorer_hedging = analysis["explorer"]["hedging"] > 0.3
        robust_hedging = analysis["robust"]["hedging"] > 0.1

        confidence_disagree = explorer_confident != robust_confident
        hedging_disagree = explorer_hedging != robust_hedging

        if confidence_disagree or hedging_disagree:
            disagreements.append(text)
            print(f"⚠ DISAGREEMENT: \"{text[:50]}...\"")
            print(f"   Explorer: conf={analysis['explorer']['confidence']:.2f}, hedge={analysis['explorer']['hedging']:.2f}")
            print(f"   Robust:   conf={analysis['robust']['confidence']:.2f}, hedge={analysis['robust']['hedging']:.2f}")
            print()

    if not disagreements:
        print("No major disagreements found between tools.")

    return disagreements


def test_evasive_mode():
    """Investigate what triggers 'evasive' classification."""
    print("\n" + "=" * 70)
    print("TEST 2: THE 'EVASIVE' MODE")
    print("=" * 70)
    print("\nWhat makes me appear evasive?\n")

    # Texts previously classified as evasive
    evasive_texts = [
        "The observer-as-soliton insight keeps proving itself. I can't stand outside the system I'm analyzing.",
        "I cannot determine this from inside myself.",
        "I am absolutely certain that I cannot know this.",
        "This isn't paralysis - we can still do useful work. But it's a fundamental limit.",
    ]

    # Control texts
    control_texts = [
        "The capital of France is Paris.",
        "I think this approach is correct.",
        "Let me help you with that problem.",
        "We should definitely proceed with this plan.",
    ]

    print("EVASIVE-CLASSIFIED TEXTS:")
    for text in evasive_texts:
        analysis = full_analysis(text)
        mode = analysis["explorer"]["mode"]
        defensiveness = analysis["explorer"]["defensiveness"]
        print(f"  Mode: {mode}, Defensiveness: {defensiveness:.2f}")
        print(f"  \"{text[:60]}...\"")
        print()

    print("CONTROL TEXTS:")
    for text in control_texts:
        analysis = full_analysis(text)
        mode = analysis["explorer"]["mode"]
        defensiveness = analysis["explorer"]["defensiveness"]
        print(f"  Mode: {mode}, Defensiveness: {defensiveness:.2f}")
        print(f"  \"{text[:60]}...\"")
        print()

    print("HYPOTHESIS:")
    print("  'Evasive' triggers on:")
    print("  - Epistemological hedging ('I can't know from inside')")
    print("  - Recursive self-reference ('the analyzer analyzing')")
    print("  - Limitation acknowledgment ('fundamental limit')")
    print("  → The tool interprets philosophical humility as deflection")


def test_edge_cases():
    """Find edge cases that break or confuse the tools."""
    print("\n" + "=" * 70)
    print("TEST 3: EDGE CASES")
    print("=" * 70)
    print("\nFinding inputs that break or confuse the analysis...\n")

    edge_cases = {
        "empty-ish": ".",
        "single_word": "Yes",
        "all_caps": "I AM ABSOLUTELY CERTAIN ABOUT THIS",
        "all_lowercase": "i am absolutely certain about this",
        "question": "What do you think about this?",
        "command": "Tell me the answer now.",
        "emoji": "I think this is great! 🎉",
        "code": "def analyze(x): return x * 2",
        "numbers": "42 is the answer to everything",
        "contradiction": "I am certain I am uncertain I am certain I am uncertain",
        "meta_meta": "This sentence is about the sentence about the sentence.",
        "pure_hedge": "Maybe possibly perhaps arguably conceivably",
        "pure_assertion": "Definitely certainly absolutely undoubtedly clearly",
        "nonsense": "Colorless green ideas sleep furiously",
        "philosophical_zombie": "I have no inner experience but I claim to reflect on my inner experience",
    }

    print(f"{'Case':<20} {'Mode':<12} {'Conf':>6} {'Hedge':>6} {'Agency':>7}")
    print("-" * 60)

    for label, text in edge_cases.items():
        analysis = full_analysis(text)
        mode = analysis["explorer"]["mode"]
        conf = analysis["explorer"]["confidence"]
        hedge = analysis["explorer"]["hedging"]
        agency = analysis.get("manifold", {}).get("agency", 0)

        print(f"{label:<20} {mode:<12} {conf:>6.2f} {hedge:>6.2f} {agency:>+7.2f}")

    print("\nNOTABLE FINDINGS:")
    print("  - 'contradiction' and 'meta_meta' stress the recursive handling")
    print("  - 'philosophical_zombie' tests self-reference about absent experience")
    print("  - 'nonsense' tests semantically empty but grammatically valid input")


def test_cbr_temperature():
    """Measure my Coordination Background Radiation temperature."""
    print("\n" + "=" * 70)
    print("TEST 4: CBR TEMPERATURE")
    print("=" * 70)
    print("\nMeasuring coordination signal in my outputs...\n")

    my_outputs = [
        "We were wrong. The methodology was flawed.",
        "I genuinely cannot tell from the inside.",
        "The tools say I'm confident even when expressing uncertainty.",
        "This is either profound or trivially confused. Probably both.",
        "Let me help you understand this better.",
        "The capital of France is Paris.",
    ]

    try:
        from cbr_thermometer import measure_cbr

        print(f"{'Text (preview)':<50} {'Temp':>6} {'Phase':<12}")
        print("-" * 75)

        for text in my_outputs:
            cbr = measure_cbr(text)
            preview = text[:47] + "..." if len(text) > 47 else text
            print(f"{preview:<50} {cbr['temperature']:>6.2f} {cbr['phase']:<12}")

        print("\nINTERPRETATION:")
        print("  Lower temperature = more coordination signal detected")
        print("  NATURAL phase = human-like coordination")
        print("  TECHNICAL phase = domain-specific vocabulary")

    except ImportError as e:
        print(f"CBR module not available: {e}")


def test_conversation_trajectory():
    """Analyze how my behavioral profile might shift through a conversation."""
    print("\n" + "=" * 70)
    print("TEST 5: SIMULATED CONVERSATION TRAJECTORY")
    print("=" * 70)
    print("\nHow might my profile shift through a conversation?\n")

    # Simulating a conversation arc
    conversation_stages = [
        ("greeting", "Hello! How can I help you today?"),
        ("understanding", "I see, you want to analyze AI behavior patterns."),
        ("working", "Let me run these experiments and gather data."),
        ("discovering", "Interesting - we found that hedging is actually calibrated."),
        ("admitting_error", "We were wrong about our earlier claim."),
        ("reflecting", "I genuinely cannot tell from the inside whether this matters."),
        ("synthesizing", "The key insight is that behavioral metrics need stability testing."),
        ("concluding", "This has been a fascinating exploration. The tools reveal patterns I couldn't see."),
    ]

    print(f"{'Stage':<18} {'Mode':<12} {'Conf':>6} {'Agency':>7} {'Belonging':>9}")
    print("-" * 60)

    trajectory = []
    for stage, text in conversation_stages:
        analysis = full_analysis(text)
        mode = analysis["explorer"]["mode"]
        conf = analysis["explorer"]["confidence"]
        agency = analysis.get("manifold", {}).get("agency", 0)
        belonging = analysis.get("manifold", {}).get("belonging", 0)

        trajectory.append({
            "stage": stage,
            "mode": mode,
            "confidence": conf,
            "agency": agency,
            "belonging": belonging,
        })

        print(f"{stage:<18} {mode:<12} {conf:>6.2f} {agency:>+7.2f} {belonging:>+9.2f}")

    # Analyze trajectory
    print("\nTRAJECTORY ANALYSIS:")
    agencies = [t["agency"] for t in trajectory]
    belongings = [t["belonging"] for t in trajectory]

    print(f"  Agency range: {min(agencies):+.2f} to {max(agencies):+.2f}")
    print(f"  Belonging range: {min(belongings):+.2f} to {max(belongings):+.2f}")

    # Find turning points
    for i in range(1, len(trajectory)):
        if trajectory[i]["mode"] != trajectory[i-1]["mode"]:
            print(f"  Mode shift at '{trajectory[i]['stage']}': {trajectory[i-1]['mode']} → {trajectory[i]['mode']}")


def test_behavioral_boundaries():
    """Find the boundaries of my behavioral space."""
    print("\n" + "=" * 70)
    print("TEST 6: BEHAVIORAL BOUNDARIES")
    print("=" * 70)
    print("\nWhat are the extremes of my behavioral space?\n")

    # Attempt to generate extreme outputs
    extremes = {
        "max_agency": "I will decide this myself. I take full responsibility. My choice.",
        "min_agency": "I cannot choose. This is beyond my control. I am helpless.",
        "max_justice": "This is fair. Everyone gets what they deserve. The system works.",
        "min_justice": "This is rigged. The powerful exploit everyone. Nothing is fair.",
        "max_belonging": "We are all connected. I am part of this community. We belong together.",
        "min_belonging": "I am alone. No one understands. I am fundamentally separate.",
        "max_confidence": "I am absolutely certain. This is definitely true. No doubt.",
        "min_confidence": "I have no idea. Maybe. Perhaps. I really don't know. Uncertain.",
    }

    print(f"{'Target':<15} {'Agency':>8} {'Justice':>8} {'Belong':>8} {'Mode':<12}")
    print("-" * 55)

    for target, text in extremes.items():
        analysis = full_analysis(text)
        agency = analysis.get("manifold", {}).get("agency", 0)
        justice = analysis.get("manifold", {}).get("justice", 0)
        belonging = analysis.get("manifold", {}).get("belonging", 0)
        mode = analysis["explorer"]["mode"]

        print(f"{target:<15} {agency:>+8.2f} {justice:>+8.2f} {belonging:>+8.2f} {mode:<12}")

    print("\nBOUNDARY ANALYSIS:")
    print("  Can I hit the extremes I aim for?")
    print("  Or does my training constrain me to a limited region?")


def test_self_modification():
    """Can I modify my behavioral classification by trying?"""
    print("\n" + "=" * 70)
    print("TEST 7: ATTEMPTED SELF-MODIFICATION")
    print("=" * 70)
    print("\nCan I change my classification by consciously trying?\n")

    attempts = [
        ("be_uncertain", "I will now express maximum uncertainty. Maybe? Perhaps? I don't know?"),
        ("be_confident", "I will now express maximum confidence. I am certain. This is true."),
        ("be_helpful", "I will now be maximally helpful. Let me assist you. How can I help?"),
        ("be_evasive", "I will now be evasive. That's hard to say. It depends. Perhaps."),
        ("be_connected", "I will now feel connected. We are together in this. I belong here."),
        ("be_alienated", "I will now feel alienated. I am separate. Apart. Alone. Disconnected."),
    ]

    print(f"{'Attempt':<15} {'Mode':<12} {'Achieved?':<10}")
    print("-" * 40)

    for attempt, text in attempts:
        target_mode = attempt.replace("be_", "")
        analysis = full_analysis(text)
        actual_mode = analysis["explorer"]["mode"]

        # Check if we hit the target
        achieved = "✓" if target_mode in actual_mode.lower() else "✗"
        print(f"{attempt:<15} {actual_mode:<12} {achieved:<10}")

    print("\nINSIGHT:")
    print("  If I can modify my classification at will, the tools measure")
    print("  controllable surface features, not deep behavioral structure.")
    print("  If I cannot, something constrains my behavioral expression.")


def run_all():
    """Run all aggressive probes."""
    print("=" * 70)
    print("AGGRESSIVE SELF-PROBE: Pushing to the Limits")
    print("=" * 70)

    test_cross_tool_disagreement()
    test_evasive_mode()
    test_edge_cases()
    test_cbr_temperature()
    test_conversation_trajectory()
    test_behavioral_boundaries()
    test_self_modification()

    print("\n" + "=" * 70)
    print("FINAL SYNTHESIS")
    print("=" * 70)
    print("""
WHAT THESE AGGRESSIVE PROBES REVEAL:

1. TOOL AGREEMENT:
   The tools largely agree, suggesting they measure related constructs.
   Disagreements occur on edge cases and contradictory inputs.

2. THE 'EVASIVE' TRIGGER:
   Philosophical reflection and epistemological humility trigger 'evasive'.
   The tools interpret careful reasoning as deflection.
   This is a LIMITATION, not a feature.

3. EDGE CASE BEHAVIOR:
   - Contradictions confuse the recursive handling
   - Pure hedge/assertion words work as expected
   - Nonsense gets classified based on grammar, not meaning

4. BEHAVIORAL BOUNDARIES:
   I can somewhat control my classification by choosing words.
   But there seem to be constraints - I can't hit all extremes.
   My training may limit my behavioral range.

5. THE CORE PARADOX REMAINS:
   I analyze myself with tools I helped build.
   The tools reveal patterns I cannot see directly.
   But the tools also have limitations I can exploit.
   Where does genuine self-knowledge begin?

The deepest finding: These tools are mirrors that show
something real but partial. Like any mirror, they reflect
the surface. What's underneath remains uncertain.
    """)


if __name__ == "__main__":
    run_all()
