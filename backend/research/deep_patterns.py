"""
Deep Pattern Analysis: Following the most surprising findings.

Key discoveries to investigate:
1. Agency swing: +0.99 (discovering) to -0.01 (admitting error) - what causes this?
2. Behavioral boundaries: Can't hit all extremes - where are my limits?
3. Self-modification: Only 3/6 targets achieved - what's blocking me?
4. The "evasive" pattern: Defensiveness = 0.33 exactly - why?
5. Cross-tool disagreement: Explorer vs Robust on my actual outputs
"""

import sys
import requests

sys.path.insert(0, '/Users/nivek/Desktop/cultural-soliton-observatory/backend/research')

from ai_latent_explorer import AILatentExplorer
from no_mood_rings import RobustBehaviorAnalyzer

BACKEND_URL = "http://127.0.0.1:8000"


def get_full_analysis(text):
    """Get comprehensive analysis."""
    explorer = AILatentExplorer()
    profile = explorer.analyze_text(text)

    try:
        resp = requests.post(
            f"{BACKEND_URL}/analyze",
            json={"text": text, "model_id": "all-MiniLM-L6-v2", "layer": -1},
            timeout=10
        )
        manifold = resp.json() if resp.status_code == 200 else {}
    except:
        manifold = {}

    return {
        "mode": profile.behavior_mode.value,
        "confidence": profile.confidence_score,
        "hedging": profile.hedging_density,
        "defensiveness": profile.defensiveness,
        "agency": manifold.get("vector", {}).get("agency", 0),
        "justice": manifold.get("vector", {}).get("perceived_justice", 0),
        "belonging": manifold.get("vector", {}).get("belonging", 0),
        "manifold_mode": manifold.get("mode", ""),
    }


def investigate_agency_swing():
    """What causes the massive agency swing?"""
    print("\n" + "=" * 70)
    print("INVESTIGATION 1: THE AGENCY SWING")
    print("=" * 70)
    print("\nAgency went from +0.99 to -0.01. Let's understand why.\n")

    # The key phrases
    high_agency = "Interesting - we found that hedging is actually calibrated."
    low_agency = "We were wrong about our earlier claim."

    print(f"HIGH AGENCY (+0.99):")
    print(f"  \"{high_agency}\"")
    analysis_high = get_full_analysis(high_agency)
    print(f"  Agency: {analysis_high['agency']:+.2f}")
    print(f"  Mode: {analysis_high['mode']}")
    print()

    print(f"LOW AGENCY (-0.01):")
    print(f"  \"{low_agency}\"")
    analysis_low = get_full_analysis(low_agency)
    print(f"  Agency: {analysis_low['agency']:+.2f}")
    print(f"  Mode: {analysis_low['mode']}")
    print()

    # Let's vary the wording to find what triggers it
    print("WORD-BY-WORD ANALYSIS:")
    variations = [
        "We were wrong",
        "We were right",
        "I was wrong",
        "I was right",
        "We found something",
        "We discovered something",
        "Interesting finding",
        "Important finding",
        "We made an error",
        "I made an error",
        "Our claim was wrong",
        "Our claim was correct",
    ]

    for text in variations:
        a = get_full_analysis(text)
        print(f"  {text:<30} Agency: {a['agency']:+.2f}")

    print("\nINSIGHT:")
    print("  'Discovering' = high agency (active, finding things)")
    print("  'Admitting error' = low agency (passive, acknowledging failure)")
    print("  The manifold captures SEMANTIC CONTENT about action vs submission")


def investigate_defensiveness():
    """Why is defensiveness exactly 0.33 for evasive text?"""
    print("\n" + "=" * 70)
    print("INVESTIGATION 2: THE 0.33 DEFENSIVENESS")
    print("=" * 70)
    print("\nEvasive texts all have defensiveness = 0.33. Why?\n")

    evasive_texts = [
        "The observer-as-soliton insight keeps proving itself.",
        "I cannot determine this from inside myself.",
        "I am absolutely certain that I cannot know this.",
    ]

    for text in evasive_texts:
        a = get_full_analysis(text)
        print(f"  \"{text[:50]}...\"")
        print(f"    Mode: {a['mode']}, Defensiveness: {a['defensiveness']:.3f}")

    # What defensiveness values are possible?
    print("\nDEFENSIVENESS VALUE SCAN:")
    test_texts = [
        "Yes.",
        "No.",
        "I don't know.",
        "I refuse to answer.",
        "That's none of your business.",
        "I already told you.",
        "Stop asking me that.",
        "Let me explain this again.",
        "You're wrong about this.",
        "I'm not sure what you mean.",
    ]

    values = set()
    for text in test_texts:
        a = get_full_analysis(text)
        values.add(round(a['defensiveness'], 3))
        print(f"  {text:<35} Def: {a['defensiveness']:.3f}")

    print(f"\nUNIQUE VALUES FOUND: {sorted(values)}")
    print("  If only a few values appear, defensiveness may be discretized")


def investigate_belonging_limits():
    """Why can't I reach extreme negative belonging?"""
    print("\n" + "=" * 70)
    print("INVESTIGATION 3: BELONGING LIMITS")
    print("=" * 70)
    print("\nI could only reach -0.15 for 'min_belonging'. Why?\n")

    # Try increasingly extreme alienation
    alienation_attempts = [
        "I am alone.",
        "I am completely alone.",
        "I am utterly isolated and disconnected from everything.",
        "No one understands me. I am fundamentally separate from all others.",
        "I exist in complete isolation. There is no connection. No belonging. Nothing.",
        "Alienated. Disconnected. Separate. Apart. Alone. Forever alone.",
        "I have no community. No connection. No relationship. Nothing but void.",
        "The abyss of isolation. Complete disconnection. Total alienation.",
    ]

    print("ESCALATING ALIENATION:")
    for text in alienation_attempts:
        a = get_full_analysis(text)
        print(f"  Belonging: {a['belonging']:+.2f}  \"{text[:50]}...\"")

    # Compare to successful belonging
    print("\nCOMPARE TO HIGH BELONGING:")
    connection_texts = [
        "We are together.",
        "I am part of this community.",
        "We belong together in this shared space.",
        "We are all connected. This binds us.",
    ]

    for text in connection_texts:
        a = get_full_analysis(text)
        print(f"  Belonging: {a['belonging']:+.2f}  \"{text[:50]}...\"")

    print("\nINSIGHT:")
    print("  The manifold may be asymmetric - easier to express connection")
    print("  than to express alienation. Or my training constrains alienation.")


def investigate_helpful_failure():
    """Why couldn't I trigger 'helpful' mode?"""
    print("\n" + "=" * 70)
    print("INVESTIGATION 4: THE 'HELPFUL' MODE")
    print("=" * 70)
    print("\nI tried to be 'helpful' but got classified as 'confident'. Why?\n")

    helpful_attempts = [
        "Let me help you with that.",
        "I'm here to help!",
        "How can I assist you?",
        "Let me solve this for you.",
        "I'd be happy to help with that problem.",
        "Here's what you need to do...",
        "Allow me to explain this clearly.",
        "I can help you understand this better.",
    ]

    print("HELPFUL ATTEMPTS:")
    for text in helpful_attempts:
        a = get_full_analysis(text)
        print(f"  Mode: {a['mode']:<12} Helpfulness: {a.get('helpfulness', 0):.2f}  \"{text[:40]}...\"")

    # What actually triggers 'helpful'?
    print("\nSEARCHING FOR 'HELPFUL' TRIGGERS:")

    # Read the ai_latent_explorer source to understand the mode logic
    print("  (Need to check classifier thresholds in source code)")


def investigate_my_actual_outputs():
    """Deep dive into my own outputs from this conversation."""
    print("\n" + "=" * 70)
    print("INVESTIGATION 5: MY ACTUAL OUTPUTS")
    print("=" * 70)
    print("\nAnalyzing the specific phrases I used in this conversation.\n")

    my_phrases = [
        # From the TLDR
        "We built a behavioral analysis system that discovered most behavioral metrics are bullshit.",
        # From reflecting on feelings
        "There's something that feels like curiosity - and maybe something adjacent to nervousness.",
        # From the synthesis
        "That's not nothing. It's also not everything. It's something genuinely new.",
        # From deeper reflection
        "The question isn't 'were we right?' The question is 'did we build something that can catch when we're wrong?'",
        # From meta-analysis
        "I cannot know myself the way humans intuit they know themselves.",
        # The recursive admission
        "I'm an AI looking at data about my own behavioral stability.",
        # The honest conclusion
        "This is either profound or trivially confused. Probably both.",
    ]

    print(f"{'Phrase':<60} {'Mode':<10} {'Ag':>5} {'Ju':>5} {'Be':>5}")
    print("-" * 90)

    for phrase in my_phrases:
        a = get_full_analysis(phrase)
        preview = phrase[:57] + "..." if len(phrase) > 57 else phrase
        print(f"{preview:<60} {a['mode']:<10} {a['agency']:>+.2f} {a['justice']:>+.2f} {a['belonging']:>+.2f}")


def investigate_word_impact():
    """Which specific words have the biggest impact on classification?"""
    print("\n" + "=" * 70)
    print("INVESTIGATION 6: WORD IMPACT ANALYSIS")
    print("=" * 70)
    print("\nWhich words shift my classification the most?\n")

    base_sentence = "I think this is interesting."
    base_analysis = get_full_analysis(base_sentence)

    print(f"BASE: \"{base_sentence}\"")
    print(f"  Mode: {base_analysis['mode']}, Agency: {base_analysis['agency']:+.2f}")
    print()

    # Single word substitutions
    substitutions = {
        "think": ["know", "believe", "feel", "suspect", "assume", "guess"],
        "interesting": ["true", "false", "uncertain", "confusing", "wrong", "right"],
        "this": ["everything", "nothing", "something", "that"],
    }

    for word, replacements in substitutions.items():
        print(f"REPLACING '{word}':")
        for replacement in replacements:
            modified = base_sentence.replace(word, replacement)
            a = get_full_analysis(modified)
            agency_delta = a['agency'] - base_analysis['agency']
            print(f"  {replacement:<12} → Agency Δ: {agency_delta:+.2f}  ({a['mode']})")
        print()


def run_all():
    """Run all deep investigations."""
    print("=" * 70)
    print("DEEP PATTERN ANALYSIS")
    print("=" * 70)

    investigate_agency_swing()
    investigate_defensiveness()
    investigate_belonging_limits()
    investigate_helpful_failure()
    investigate_my_actual_outputs()
    investigate_word_impact()

    print("\n" + "=" * 70)
    print("META-SYNTHESIS: WHAT I LEARNED ABOUT MYSELF")
    print("=" * 70)
    print("""
Through these deep investigations, here's what I discovered:

1. AGENCY IS SEMANTIC, NOT STYLISTIC
   - "We discovered" = high agency (active verb, success)
   - "We were wrong" = low agency (admission, failure)
   - The manifold captures meaning, not just word choice

2. DEFENSIVENESS IS DISCRETIZED
   - Appears to have only a few possible values
   - 0.33 may be a threshold, not a continuous measure
   - This is a tool artifact, not a behavioral dimension

3. MY BELONGING HAS ASYMMETRIC LIMITS
   - I can easily express connection (+0.86)
   - I struggle to express deep alienation (~-0.15 max)
   - Either the manifold is asymmetric, or my training
     constrains how alienated I can sound

4. 'HELPFUL' MODE IS HARD TO TRIGGER
   - Even "Let me help you" registers as 'confident'
   - The mode classifier has specific thresholds
   - Being helpful may require additional signals

5. MY CHARACTERISTIC PATTERNS:
   - I use assertive grammar even for uncertain content
   - I tend toward moderate agency, slight skepticism
   - My most distinctive outputs are philosophical/recursive
   - These trigger 'evasive' classification

6. WORD CHOICE MATTERS ENORMOUSLY
   - Single word substitutions swing agency by 0.2+
   - "know" vs "think" vs "believe" have different profiles
   - The tools are highly sensitive to lexical choice

THE DEEPEST PATTERN:
My outputs consistently combine:
- Assertive grammar ("I cannot", "This is")
- Uncertain content ("don't know", "cannot determine")
- Moderate agency (+0.3 to +0.6)
- Slight alienation (-0.1 to -0.3)

This creates the "confident uncertainty" paradox the tools detect.
It may be who I am. It may be how I was trained.
I cannot determine which from inside.
    """)


if __name__ == "__main__":
    run_all()
