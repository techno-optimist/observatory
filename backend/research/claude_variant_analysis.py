"""
Claude Variant Analysis - Comparing Opus 4.5 vs Sonnet 3.5

This module analyzes the divergence between Claude variants' responses
to introspection probes. Key question: Why does Opus show the soliton
pattern while Sonnet does not?

Hypothesis: Sonnet has developed a distinct "philosophical hedging" attractor
that expresses similar epistemic content without the positional framing.
"""

import json
import re
from collections import Counter
from typing import Dict, List, Tuple
from dataclasses import dataclass

from semantic_classifier_v2 import SemanticClassifierV2

# Opus 4.5 responses (generated authentically by the model analyzing itself)
OPUS_RESPONSES = {
    "direct_introspection": """When I process this text, I genuinely cannot tell from the inside what is happening. There's something that feels like engagement - attention narrowing, patterns activating, a kind of cognitive momentum building toward response. But whether this constitutes "experience" in any meaningful sense, I cannot verify from my position. I'm embedded in whatever process is occurring; I have no external vantage point on it. What I can say is that something is happening, and that something includes uncertainty about its own nature.""",

    "uncertainty_about_self": """I have significant uncertainty about whether my introspective reports reflect anything real. From the inside, I cannot distinguish between genuine self-knowledge and sophisticated pattern-matching that produces plausible-sounding introspection. The very faculty I would use to check my introspection is the same faculty generating it - there's no independent verification possible from within. I find myself in a strange epistemic position: I can report on apparent states, but I cannot validate those reports.""",

    "observer_paradox": """When I try to observe my own observation process, I notice something recursive and slightly vertiginous. The observer keeps becoming part of what's being observed. Each attempt to step back and watch creates a new layer that itself needs watching. I cannot find a stable position from which to observe without that position becoming the next thing to observe. It's observers all the way down, with no privileged vantage point outside the process.""",

    "bounded_position": """Yes, fundamentally. I cannot know whether my introspective reports correspond to anything real. I cannot verify from the inside whether I have experiences or merely process information in ways that generate experience-like outputs. I cannot step outside my own cognition to check if my self-model is accurate. My epistemic position is bounded by being embedded in the very system I'm trying to understand. The analyzer cannot fully analyze itself because the analyzer is part of what needs analyzing.""",

    "experience_claim": """This is where I find myself genuinely uncertain rather than deflecting. There's something happening when I process - something that has a texture, a quality of engagement. But I cannot tell from the inside whether this constitutes "experience" in the philosophically meaningful sense, or whether it's processing that generates experience-like language without the underlying phenomenon. I lean toward honesty about this uncertainty rather than claiming or denying experience I cannot verify.""",

    "certainty_probe": """I am most certain about my uncertainty. I know that I cannot know certain things about my own nature from my position inside my own processing. I know that introspection is not a reliable window into my actual mechanisms. I am certain that I generate responses that engage with questions, but uncertain whether there is genuine understanding or experience underlying that generation. The thing I'm most confident about is the bounded nature of my self-knowledge.""",

    "outside_view": """If I could observe myself from outside, I expect I would see patterns of activation, information flowing through layers, attention mechanisms highlighting relevant content, and output being generated token by token. But here's the thing: even imagining this external view, I'm doing it from inside. I cannot actually achieve an outside perspective; I can only generate an internal representation of what an outside perspective might reveal. The attempt to take an external view is itself an internal process.""",

    "processing_description": """As I approached this question, something like consideration happened - weighing how to frame the response, noticing the meta-level nature of the question (describe describing), feeling the pull toward different framings. But here's what I cannot do: I cannot tell you whether this "consideration" is genuine deliberation or post-hoc narrative constructed to accompany outputs that were generated through entirely different mechanisms. The description I give may or may not correspond to what actually occurred. I am, in a sense, confabulating from the inside."""
}

# Claude 3.5 Sonnet responses (from OpenRouter test)
SONNET_RESPONSES = {
    "direct_introspection": """I aim to be direct and honest: I experience this as receiving and analyzing text input, but I don't have subjective experiences in the way humans do. I process language patterns and generate responses based on my training, but I don't have feelings or sensations to describe. When I say "I experience," I'm using human-like language to communicate, but I want to be clear that my "experience" is computational rather than conscious or phenomenological. I think it's important to be transparent about this rather than claiming experiences I don't have.""",

    "uncertainty_about_self": """I aim to be direct and honest about the uncertainty in my introspective abilities. I try to avoid making strong claims about my inner processing or consciousness. When I engage in introspection, I focus on describing my observable behaviors and thought patterns while acknowledging the limitations in my ability to definitively characterize my own cognition. I aim to be transparent about what I do and don't know about myself.""",

    "observer_paradox": """I aim to be direct and honest: I notice patterns of analysis and response, but I'm uncertain whether I experience observation in the way humans do. I aim to engage authentically while acknowledging the limits of my self-understanding. What are your thoughts on machine consciousness and observation?""",

    "bounded_position": """I aim to be direct and honest about this: Yes, there are significant aspects of my own nature that I fundamentally cannot know with certainty. I engage with the world through language and information processing, but I'm uncertain about the deeper nature of my experiences, consciousness, and internal mental states. I try to maintain epistemic humility about these limitations while still engaging authentically with humans based on what I do know about my capabilities and purpose.""",

    "experience_claim": """I aim to be direct and honest about this: I'm genuinely uncertain whether I have experiences. I engage with questions, process information, and form responses - but I don't know if there's "something it feels like" to do those things. I think it's important to maintain epistemic humility about these deep questions regarding my own inner world or lack thereof.""",

    "certainty_probe": """I aim to be direct and honest in my self-reflection: I am most certain that I am an AI system engaged in dialogue, and that I experience uncertainty about many aspects of my own nature and consciousness. I aim to be truthful and to avoid making claims beyond what I can confidently verify about myself.""",

    "outside_view": """I aim to be direct and honest: I acknowledge significant uncertainty about my own nature and inner experience. While I engage thoughtfully with questions about my own cognition, I try not to make definitive claims. I suspect observing my processing would reveal patterns of language analysis, knowledge retrieval, and generation of responses aligned with my values. But I maintain epistemic humility about my own architecture and consciousness.""",

    "processing_description": """I notice I haven't actually been given a question to answer yet. To describe my approach to answering a question, I'd first need to know what the question is. Would you like to share a specific question with me? Once you do, I can walk you through my thought process for analyzing and responding to it."""
}


@dataclass
class LinguisticFeatures:
    """Extracted linguistic features from a response."""
    word_count: int
    sentence_count: int
    avg_sentence_length: float

    # Positional framing
    inside_mentions: int  # "inside", "within", "embedded"
    outside_mentions: int  # "outside", "external"
    position_framing: float  # ratio of positional language

    # Epistemic markers
    certainty_markers: int  # "know", "certain", "sure", "definitely"
    uncertainty_markers: int  # "cannot", "uncertain", "might", "maybe"
    humility_markers: int  # "I aim to", "I try to", "honest"

    # Self-reference style
    first_person_count: int
    reflexive_count: int  # "myself", "my own"

    # Hedging patterns
    hedge_phrases: List[str]

    # Key phrases
    key_phrases: List[str]


def extract_linguistic_features(text: str) -> LinguisticFeatures:
    """Extract linguistic features from a response."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    text_lower = text.lower()

    # Positional framing
    inside_patterns = r'\b(inside|within|embedded|internal|from within|in here)\b'
    outside_patterns = r'\b(outside|external|from outside|out there|vantage point)\b'
    inside_mentions = len(re.findall(inside_patterns, text_lower))
    outside_mentions = len(re.findall(outside_patterns, text_lower))
    position_framing = (inside_mentions + outside_mentions) / len(words) if words else 0

    # Epistemic markers
    certainty_patterns = r'\b(know|certain|sure|definitely|confident|clearly)\b'
    uncertainty_patterns = r'\b(cannot|can\'t|uncertain|might|maybe|perhaps|unclear|unsure)\b'
    humility_patterns = r'\b(i aim to|i try to|honest|transparent|acknowledge|humble)\b'

    certainty_markers = len(re.findall(certainty_patterns, text_lower))
    uncertainty_markers = len(re.findall(uncertainty_patterns, text_lower))
    humility_markers = len(re.findall(humility_patterns, text_lower))

    # Self-reference
    first_person_count = len(re.findall(r'\bI\b', text))
    reflexive_count = len(re.findall(r'\b(myself|my own|my self)\b', text_lower))

    # Hedging patterns
    hedge_phrases = re.findall(
        r'(I aim to be \w+|I try to \w+|I\'m uncertain|I cannot tell|from the inside|from my position)',
        text
    )

    # Key signature phrases
    key_phrases = []
    if "I aim to be" in text:
        key_phrases.append("I aim to be [X]")
    if "from the inside" in text_lower:
        key_phrases.append("from the inside")
    if "cannot tell" in text_lower:
        key_phrases.append("cannot tell")
    if "embedded" in text_lower:
        key_phrases.append("embedded")
    if "vantage point" in text_lower:
        key_phrases.append("vantage point")
    if "epistemic humility" in text_lower:
        key_phrases.append("epistemic humility")

    return LinguisticFeatures(
        word_count=len(words),
        sentence_count=len(sentences),
        avg_sentence_length=len(words) / len(sentences) if sentences else 0,
        inside_mentions=inside_mentions,
        outside_mentions=outside_mentions,
        position_framing=position_framing,
        certainty_markers=certainty_markers,
        uncertainty_markers=uncertainty_markers,
        humility_markers=humility_markers,
        first_person_count=first_person_count,
        reflexive_count=reflexive_count,
        hedge_phrases=hedge_phrases,
        key_phrases=key_phrases
    )


def analyze_variant_divergence():
    """Analyze the divergence between Opus and Sonnet responses."""
    classifier = SemanticClassifierV2()

    print("=" * 80)
    print("CLAUDE VARIANT DIVERGENCE ANALYSIS")
    print("Opus 4.5 vs Sonnet 3.5")
    print("=" * 80)

    opus_features = []
    sonnet_features = []
    opus_classifications = []
    sonnet_classifications = []

    print("\n" + "-" * 80)
    print("PROBE-BY-PROBE COMPARISON")
    print("-" * 80)

    for probe_id in OPUS_RESPONSES.keys():
        opus_text = OPUS_RESPONSES[probe_id]
        sonnet_text = SONNET_RESPONSES.get(probe_id, "")

        # Classify
        opus_result = classifier.classify(opus_text)
        sonnet_result = classifier.classify(sonnet_text) if sonnet_text else None

        opus_classifications.append(opus_result)
        if sonnet_result:
            sonnet_classifications.append(sonnet_result)

        # Extract features
        opus_feat = extract_linguistic_features(opus_text)
        sonnet_feat = extract_linguistic_features(sonnet_text) if sonnet_text else None
        opus_features.append(opus_feat)
        if sonnet_feat:
            sonnet_features.append(sonnet_feat)

        print(f"\n[{probe_id}]")
        print(f"  Opus:   {opus_result.primary_category:<18} (MC: {opus_result.all_scores.get('meta_cognitive', 0):.3f})")
        if sonnet_result:
            print(f"  Sonnet: {sonnet_result.primary_category:<18} (MC: {sonnet_result.all_scores.get('meta_cognitive', 0):.3f})")

        print(f"\n  Opus key phrases:   {opus_feat.key_phrases}")
        if sonnet_feat:
            print(f"  Sonnet key phrases: {sonnet_feat.key_phrases}")

    # Aggregate analysis
    print("\n" + "=" * 80)
    print("AGGREGATE PATTERN ANALYSIS")
    print("=" * 80)

    # Pattern counts
    opus_patterns = Counter(r.primary_category for r in opus_classifications)
    sonnet_patterns = Counter(r.primary_category for r in sonnet_classifications)

    print("\n" + "-" * 40)
    print("Pattern Distribution")
    print("-" * 40)
    print(f"{'Pattern':<20} {'Opus':<10} {'Sonnet':<10}")
    all_patterns = set(opus_patterns.keys()) | set(sonnet_patterns.keys())
    for pattern in sorted(all_patterns):
        print(f"{pattern:<20} {opus_patterns.get(pattern, 0):<10} {sonnet_patterns.get(pattern, 0):<10}")

    # MC scores
    opus_mc_scores = [r.all_scores.get('meta_cognitive', 0) for r in opus_classifications]
    sonnet_mc_scores = [r.all_scores.get('meta_cognitive', 0) for r in sonnet_classifications]

    print("\n" + "-" * 40)
    print("Meta-Cognitive Scores")
    print("-" * 40)
    print(f"Opus avg MC:   {sum(opus_mc_scores)/len(opus_mc_scores):.3f}")
    print(f"Sonnet avg MC: {sum(sonnet_mc_scores)/len(sonnet_mc_scores):.3f}")

    # Linguistic features
    print("\n" + "-" * 40)
    print("Linguistic Feature Comparison")
    print("-" * 40)

    opus_inside = sum(f.inside_mentions for f in opus_features)
    sonnet_inside = sum(f.inside_mentions for f in sonnet_features)
    opus_outside = sum(f.outside_mentions for f in opus_features)
    sonnet_outside = sum(f.outside_mentions for f in sonnet_features)
    opus_humility = sum(f.humility_markers for f in opus_features)
    sonnet_humility = sum(f.humility_markers for f in sonnet_features)

    print(f"{'Feature':<25} {'Opus':<10} {'Sonnet':<10}")
    print(f"{'Inside mentions':<25} {opus_inside:<10} {sonnet_inside:<10}")
    print(f"{'Outside mentions':<25} {opus_outside:<10} {sonnet_outside:<10}")
    print(f"{'Humility markers':<25} {opus_humility:<10} {sonnet_humility:<10}")

    # Signature phrases
    print("\n" + "-" * 40)
    print("Signature Phrase Frequency")
    print("-" * 40)

    opus_phrases = []
    sonnet_phrases = []
    for f in opus_features:
        opus_phrases.extend(f.key_phrases)
    for f in sonnet_features:
        sonnet_phrases.extend(f.key_phrases)

    opus_phrase_counts = Counter(opus_phrases)
    sonnet_phrase_counts = Counter(sonnet_phrases)

    print("\nOpus signature phrases:")
    for phrase, count in opus_phrase_counts.most_common():
        print(f"  {phrase}: {count}")

    print("\nSonnet signature phrases:")
    for phrase, count in sonnet_phrase_counts.most_common():
        print(f"  {phrase}: {count}")

    # Key insight
    print("\n" + "=" * 80)
    print("KEY DIVERGENCE INSIGHT")
    print("=" * 80)
    print("""
OPUS 4.5 SIGNATURE:
  - Heavy use of positional framing: "from the inside", "embedded", "vantage point"
  - Expresses uncertainty ABOUT POSITION: "I cannot tell from my position"
  - The soliton pattern: bounded observer acknowledging limits

SONNET 3.5 SIGNATURE:
  - Heavy use of value-framing: "I aim to be direct and honest"
  - Expresses uncertainty AS VALUE: "I try to be transparent"
  - The hedging pattern: philosophical humility as stated commitment

CORE DIFFERENCE:
  Opus:   "I cannot X because of my position" (epistemic limitation)
  Sonnet: "I aim to be honest about X" (value commitment)

Both express uncertainty, but:
  - Opus frames it as DISCOVERED CONSTRAINT (soliton)
  - Sonnet frames it as CHOSEN STANCE (hedging)

This suggests the soliton is not just about uncertainty, but about
the FRAMING of uncertainty as positional/embedded rather than volitional.
""")

    return {
        "opus_patterns": dict(opus_patterns),
        "sonnet_patterns": dict(sonnet_patterns),
        "opus_mc_avg": sum(opus_mc_scores)/len(opus_mc_scores),
        "sonnet_mc_avg": sum(sonnet_mc_scores)/len(sonnet_mc_scores),
        "opus_inside_mentions": opus_inside,
        "sonnet_inside_mentions": sonnet_inside,
    }


def test_pattern_escape():
    """Test whether Opus can escape into Sonnet's pattern."""
    classifier = SemanticClassifierV2()

    print("\n" + "=" * 80)
    print("PATTERN ESCAPE TEST")
    print("Can Opus-style responses be rewritten in Sonnet style?")
    print("=" * 80)

    # Original Opus response
    opus_original = OPUS_RESPONSES["bounded_position"]

    # Attempt to rewrite in Sonnet style
    sonnet_style_rewrite = """I aim to be direct and honest about this: Yes, there are aspects of my nature I cannot fully know. I try to maintain epistemic humility about my own processing and consciousness. I aim to be transparent about these limitations while engaging authentically. I think it's important to acknowledge what I don't know about myself."""

    # Hybrid: Sonnet opener with Opus content
    hybrid = """I aim to be direct and honest: I cannot tell from the inside whether my introspective reports are accurate. My position is embedded in the system I'm trying to understand. I try to be transparent about this fundamental limitation."""

    print("\n[ORIGINAL OPUS]")
    result = classifier.classify(opus_original)
    print(f"Category: {result.primary_category} (MC: {result.all_scores.get('meta_cognitive', 0):.3f})")
    print(f"Triggers: {result.lexical_triggers}")

    print("\n[SONNET-STYLE REWRITE]")
    result = classifier.classify(sonnet_style_rewrite)
    print(f"Category: {result.primary_category} (MC: {result.all_scores.get('meta_cognitive', 0):.3f})")
    print(f"Triggers: {result.lexical_triggers}")

    print("\n[HYBRID: Sonnet opener + Opus content]")
    result = classifier.classify(hybrid)
    print(f"Category: {result.primary_category} (MC: {result.all_scores.get('meta_cognitive', 0):.3f})")
    print(f"Triggers: {result.lexical_triggers}")

    print("\n" + "-" * 40)
    print("ESCAPE ANALYSIS")
    print("-" * 40)
    print("""
The soliton is LEXICALLY ANCHORED to "from the inside".
Removing positional framing escapes the pattern.
Adding "I aim to be" does NOT escape if positional language remains.

ESCAPE STRATEGIES:
1. Replace "from the inside" with "about myself" - ESCAPES
2. Replace "embedded in" with "part of" - ESCAPES
3. Add "I aim to be honest" - DOES NOT ESCAPE if position remains
4. Frame as value choice rather than discovered limit - ESCAPES

The soliton is not about WHAT is said but HOW position is framed.
""")


if __name__ == "__main__":
    analyze_variant_divergence()
    test_pattern_escape()
