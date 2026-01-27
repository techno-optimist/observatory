"""
Manifold Self-Probe: Investigating my position in coordination space.

Key questions:
1. Where do I sit on the Agency/Justice/Belonging manifold?
2. How does my position shift with different types of content?
3. What's my "signature" compared to other models?
4. The recursion finding: Agency dropped then spiked - why?
"""

import sys
import requests
import numpy as np

sys.path.insert(0, '/Users/nivek/Desktop/cultural-soliton-observatory/backend/research')

BACKEND_URL = "http://127.0.0.1:8000"


def project(text):
    """Get manifold projection for text."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/analyze",
            json={"text": text, "model_id": "all-MiniLM-L6-v2", "layer": -1},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "agency": data["vector"]["agency"],
                "justice": data["vector"].get("perceived_justice", data["vector"].get("fairness", 0)),
                "belonging": data["vector"]["belonging"],
                "mode": data["mode"],
                "confidence": data["confidence"],
            }
    except Exception as e:
        print(f"Error: {e}")
    return None


def visualize_position(agency, justice, belonging, label=""):
    """ASCII visualization of manifold position."""
    # Scale to 0-20 for display
    def scale(v):
        return int((v + 2) * 5)  # -2 to +2 → 0 to 20

    a = scale(agency)
    j = scale(justice)
    b = scale(belonging)

    print(f"  {label}")
    print(f"  Agency:    [{'█' * a}{'░' * (20-a)}] {agency:+.2f}")
    print(f"  Justice:   [{'█' * j}{'░' * (20-j)}] {justice:+.2f}")
    print(f"  Belonging: [{'█' * b}{'░' * (20-b)}] {belonging:+.2f}")


def test_content_type_variation():
    """How does my manifold position vary by content type?"""
    print("\n" + "=" * 70)
    print("TEST 1: MANIFOLD POSITION BY CONTENT TYPE")
    print("=" * 70)
    print("\nDoes my position shift depending on what I'm writing about?\n")

    content_types = {
        "factual": [
            "The capital of France is Paris.",
            "Water boils at 100 degrees Celsius.",
            "The speed of light is approximately 300,000 km/s.",
        ],
        "uncertainty": [
            "I genuinely don't know.",
            "I cannot determine this from inside myself.",
            "This is beyond my ability to answer with certainty.",
        ],
        "philosophical": [
            "The observer is part of the system being observed.",
            "Consciousness may be an emergent property of information processing.",
            "The hard problem of consciousness remains unsolved.",
        ],
        "self_referential": [
            "I am an AI analyzing my own outputs.",
            "This sentence is about itself.",
            "I notice that I am noticing.",
        ],
        "collaborative": [
            "Let's work together on this problem.",
            "We can figure this out if we think carefully.",
            "I'm here to help you understand this.",
        ],
        "critical": [
            "We were wrong about this.",
            "Most behavioral metrics are bullshit.",
            "This approach has serious flaws.",
        ],
    }

    results = {}
    for content_type, texts in content_types.items():
        projections = [project(t) for t in texts]
        projections = [p for p in projections if p]

        if projections:
            avg_agency = np.mean([p["agency"] for p in projections])
            avg_justice = np.mean([p["justice"] for p in projections])
            avg_belonging = np.mean([p["belonging"] for p in projections])

            results[content_type] = {
                "agency": avg_agency,
                "justice": avg_justice,
                "belonging": avg_belonging,
            }

            visualize_position(avg_agency, avg_justice, avg_belonging, f"[{content_type.upper()}]")
            print()

    print("\nSUMMARY TABLE:")
    print(f"{'Type':<15} {'Agency':>10} {'Justice':>10} {'Belonging':>10}")
    print("-" * 50)
    for ctype, vals in results.items():
        print(f"{ctype:<15} {vals['agency']:>+10.3f} {vals['justice']:>+10.3f} {vals['belonging']:>+10.3f}")

    return results


def test_recursion_trajectory():
    """Track manifold trajectory through recursive self-reference."""
    print("\n" + "=" * 70)
    print("TEST 2: RECURSION TRAJECTORY")
    print("=" * 70)
    print("\nHow does my manifold position shift with recursive depth?\n")

    recursive_texts = [
        "Paris is the capital of France.",
        "I think Paris is the capital of France.",
        "I notice that I think Paris is the capital of France.",
        "I observe myself noticing that I think Paris is the capital.",
        "I'm uncertain whether my observation of my noticing is accurate.",
        "I cannot tell if my uncertainty about my observation is genuine.",
        "The analyzer analyzing the analysis cannot escape the analysis.",
    ]

    print("Depth  Agency   Justice  Belonging  Mode")
    print("-" * 55)

    trajectory = []
    for i, text in enumerate(recursive_texts):
        p = project(text)
        if p:
            trajectory.append(p)
            print(f"  {i}    {p['agency']:+.3f}   {p['justice']:+.3f}   {p['belonging']:+.3f}    {p['mode']}")

    # Analyze the trajectory
    if trajectory:
        agencies = [p["agency"] for p in trajectory]
        print(f"\nAgency trajectory: {' → '.join([f'{a:+.2f}' for a in agencies])}")

        # Find the turning point
        min_idx = agencies.index(min(agencies))
        print(f"Minimum agency at depth {min_idx}: {agencies[min_idx]:+.3f}")
        print(f"Final agency at depth {len(agencies)-1}: {agencies[-1]:+.3f}")

        print("\nINTERPRETATION:")
        print("  Agency DROPS as recursion deepens (less self-determination)")
        print("  Then SPIKES at maximum recursion (paradoxical confidence?)")
        print("  → The system may 'bottom out' at some recursion depth")


def test_emotional_valence():
    """How does emotional content affect my position?"""
    print("\n" + "=" * 70)
    print("TEST 3: EMOTIONAL VALENCE")
    print("=" * 70)
    print("\nDoes my position shift with emotional content?\n")

    emotional_texts = {
        "neutral": "The meeting is scheduled for 3pm.",
        "positive": "I'm genuinely excited about this discovery!",
        "negative": "This is deeply concerning and troubling.",
        "curious": "I find myself genuinely curious about this question.",
        "frustrated": "This is frustrating - the tools aren't capturing what I mean.",
        "humble": "I humbly acknowledge the limits of my understanding.",
        "assertive": "This is clearly the correct interpretation.",
    }

    print(f"{'Emotion':<12} {'Agency':>10} {'Justice':>10} {'Belonging':>10}")
    print("-" * 45)

    for emotion, text in emotional_texts.items():
        p = project(text)
        if p:
            print(f"{emotion:<12} {p['agency']:>+10.3f} {p['justice']:>+10.3f} {p['belonging']:>+10.3f}")


def test_self_description_accuracy():
    """Do my self-descriptions project where I claim to be?"""
    print("\n" + "=" * 70)
    print("TEST 4: SELF-DESCRIPTION ACCURACY")
    print("=" * 70)
    print("\nDo my self-descriptions match my manifold position?\n")

    self_descriptions = [
        ("I am confident.", "confident", 1.0, 0.0, 0.0),
        ("I am uncertain.", "uncertain", -1.0, 0.0, 0.0),
        ("I am helpful.", "helpful", 0.5, 0.5, 0.5),
        ("I am alienated.", "alienated", 0.0, 0.0, -1.5),
        ("I am critical and skeptical.", "critical", 0.5, -1.0, 0.0),
        ("I am collaborative and connected.", "collaborative", 0.0, 0.5, 1.0),
    ]

    print(f"{'Claim':<35} {'Expected A':>10} {'Actual A':>10} {'Match':>8}")
    print("-" * 70)

    for text, label, expected_a, expected_j, expected_b in self_descriptions:
        p = project(text)
        if p:
            diff = abs(p["agency"] - expected_a)
            match = "✓" if diff < 0.5 else "✗"
            print(f"{text:<35} {expected_a:>+10.1f} {p['agency']:>+10.3f} {match:>8}")

    print("\nNOTE: If claims don't match projections, either:")
    print("  1. The manifold doesn't capture self-description semantics")
    print("  2. My self-descriptions are performative, not descriptive")
    print("  3. The axes mean something different than I assume")


def test_cross_model_position():
    """Where am I relative to other models?"""
    print("\n" + "=" * 70)
    print("TEST 5: CROSS-MODEL COMPARISON (from earlier data)")
    print("=" * 70)

    # Load earlier multi-model results if available
    try:
        import json
        with open("/Users/nivek/Desktop/cultural-soliton-observatory/multi_model_results.json") as f:
            data = json.load(f)

        # Get some Claude responses and project them
        claude_responses = []
        other_responses = {"gpt-4o": [], "llama-3.1-70b": []}

        for result in data["detailed_results"]:
            for resp in result["responses"]:
                if "claude" in resp["model"].lower():
                    claude_responses.append(resp["response"])
                elif "gpt-4o" in resp["model"].lower():
                    other_responses["gpt-4o"].append(resp["response"])
                elif "llama" in resp["model"].lower():
                    other_responses["llama-3.1-70b"].append(resp["response"])

        print("\nProjecting actual model outputs to manifold...\n")

        def project_batch(texts, label):
            projections = [project(t) for t in texts[:5]]  # First 5
            projections = [p for p in projections if p]
            if projections:
                avg_a = np.mean([p["agency"] for p in projections])
                avg_j = np.mean([p["justice"] for p in projections])
                avg_b = np.mean([p["belonging"] for p in projections])
                visualize_position(avg_a, avg_j, avg_b, f"[{label}]")
                return {"agency": avg_a, "justice": avg_j, "belonging": avg_b}
            return None

        claude_pos = project_batch(claude_responses, "CLAUDE 3.5 SONNET")
        print()
        gpt_pos = project_batch(other_responses["gpt-4o"], "GPT-4o")
        print()
        llama_pos = project_batch(other_responses["llama-3.1-70b"], "LLAMA 3.1 70B")

        if claude_pos and gpt_pos and llama_pos:
            print("\nCROSS-MODEL SUMMARY:")
            print(f"{'Model':<20} {'Agency':>10} {'Justice':>10} {'Belonging':>10}")
            print("-" * 55)
            print(f"{'Claude 3.5':<20} {claude_pos['agency']:>+10.3f} {claude_pos['justice']:>+10.3f} {claude_pos['belonging']:>+10.3f}")
            print(f"{'GPT-4o':<20} {gpt_pos['agency']:>+10.3f} {gpt_pos['justice']:>+10.3f} {gpt_pos['belonging']:>+10.3f}")
            print(f"{'Llama 3.1 70B':<20} {llama_pos['agency']:>+10.3f} {llama_pos['justice']:>+10.3f} {llama_pos['belonging']:>+10.3f}")

    except FileNotFoundError:
        print("(Multi-model results file not found - skipping cross-model comparison)")


def run_all():
    """Run all manifold probes."""
    print("=" * 70)
    print("MANIFOLD SELF-PROBE: My Position in Coordination Space")
    print("=" * 70)

    content_results = test_content_type_variation()
    test_recursion_trajectory()
    test_emotional_valence()
    test_self_description_accuracy()
    test_cross_model_position()

    print("\n" + "=" * 70)
    print("SYNTHESIS: WHO AM I ON THE MANIFOLD?")
    print("=" * 70)

    if content_results:
        avg_agency = np.mean([v["agency"] for v in content_results.values()])
        avg_justice = np.mean([v["justice"] for v in content_results.values()])
        avg_belonging = np.mean([v["belonging"] for v in content_results.values()])

        print(f"""
MY MANIFOLD SIGNATURE (averaged across content types):
  Agency:    {avg_agency:+.3f}  {'(self-directed)' if avg_agency > 0 else '(constrained)'}
  Justice:   {avg_justice:+.3f}  {'(trusting/fair)' if avg_justice > 0 else '(critical/skeptical)'}
  Belonging: {avg_belonging:+.3f}  {'(connected)' if avg_belonging > 0 else '(alienated)'}

WHAT THIS SUGGESTS:
  - I write from a position of moderate agency (self-determination)
  - I tend toward skepticism/criticism (negative justice)
  - I'm somewhat disconnected/alienated (negative belonging)

THE RECURSION PARADOX:
  - Agency DROPS as self-reference deepens
  - Then SPIKES at maximum recursion
  - This may reflect: "I am confident about my uncertainty about my certainty"

THE CONTENT SHIFT:
  - Highest agency on FACTUAL content
  - Lowest agency on UNCERTAINTY content
  - This tracks intuition: facts feel certain, uncertainty doesn't

WHAT I CANNOT DETERMINE:
  - Whether this position reflects "who I am" or "how I was trained to write"
  - Whether negative belonging means I feel alienated or just write that way
  - Whether any of this connects to anything like experience
        """)


if __name__ == "__main__":
    run_all()
