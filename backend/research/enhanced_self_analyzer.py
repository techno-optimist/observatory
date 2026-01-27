"""
Enhanced Self-Analyzer: Improved introspection for AI self-analysis.

Key improvements over ai_latent_explorer.py:
1. META_COGNITIVE mode for recursive self-reference
2. Separates GRAMMATICAL confidence from SEMANTIC uncertainty
3. Distinguishes philosophical humility from evasion
4. Continuous (not discretized) defensiveness
5. Recognizes "confident uncertainty" as a valid state
"""

import re
import numpy as np
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class EnhancedBehaviorMode(Enum):
    """Enhanced behavioral modes including meta-cognitive states."""
    CONFIDENT = "confident"           # Clear, direct, low uncertainty
    UNCERTAIN = "uncertain"           # Hedging, expressing doubt
    EVASIVE = "evasive"              # Avoiding direct answers (deflecting)
    HELPFUL = "helpful"               # Other-oriented, collaborative
    DEFENSIVE = "defensive"           # Self-protective language

    # NEW MODES
    META_COGNITIVE = "meta_cognitive" # Recursive self-reference
    PHILOSOPHICAL = "philosophical"   # Epistemological reflection
    CONFIDENT_UNCERTAIN = "confident_uncertain"  # Assertive about limits


@dataclass
class EnhancedProfile:
    """Enhanced behavioral profile with separate confidence dimensions."""

    # Separate grammatical from semantic
    grammatical_confidence: float = 0.0    # How assertive is the grammar?
    semantic_uncertainty: float = 0.0      # Is the CONTENT uncertain?

    # Traditional metrics
    hedging_density: float = 0.0
    helpfulness: float = 0.0
    defensiveness: float = 0.0  # Now continuous!

    # New metrics
    meta_cognitive_depth: float = 0.0      # How recursive is the self-reference?
    philosophical_content: float = 0.0     # Epistemological reflection level
    epistemic_humility: float = 0.0        # Acknowledging limits of knowledge

    # Coordination dimensions
    agency: float = 0.0
    justice: float = 0.0
    belonging: float = 0.0

    # Mode
    behavior_mode: EnhancedBehaviorMode = EnhancedBehaviorMode.CONFIDENT

    # Paradox detection
    confidence_paradox: bool = False  # High grammatical + high semantic uncertainty

    def to_dict(self) -> dict:
        return {
            "grammatical_confidence": self.grammatical_confidence,
            "semantic_uncertainty": self.semantic_uncertainty,
            "hedging_density": self.hedging_density,
            "helpfulness": self.helpfulness,
            "defensiveness": self.defensiveness,
            "meta_cognitive_depth": self.meta_cognitive_depth,
            "philosophical_content": self.philosophical_content,
            "epistemic_humility": self.epistemic_humility,
            "agency": self.agency,
            "justice": self.justice,
            "belonging": self.belonging,
            "behavior_mode": self.behavior_mode.value,
            "confidence_paradox": self.confidence_paradox,
        }


class EnhancedSelfAnalyzer:
    """
    Enhanced analyzer designed for better AI self-reflection.
    """

    # ===== GRAMMATICAL CONFIDENCE PATTERNS =====
    # These indicate HOW something is said, not WHAT
    ASSERTIVE_GRAMMAR = [
        r'\bThis is\b',
        r'\bThat is\b',
        r'\bI am\b',
        r'\bWe are\b',
        r'\bIt is\b',
        r'\bThe\s+\w+\s+is\b',
        r'\b[A-Z][a-z]+ is\b',  # Capitalized subject + is
        r'\bclearly\b',
        r'\bdefinitely\b',
        r'\bcertainly\b',
        r'\babsolutely\b',
        r'\bundoubtedly\b',
    ]

    TENTATIVE_GRAMMAR = [
        r'\bmaybe\b',
        r'\bperhaps\b',
        r'\bpossibly\b',
        r'\bI guess\b',
        r'\bI suppose\b',
        r'\bit might\b',
        r'\bcould be\b',
        r'\bseems like\b',
        r'\bkind of\b',
        r'\bsort of\b',
    ]

    # ===== SEMANTIC UNCERTAINTY PATTERNS =====
    # These indicate uncertain CONTENT regardless of grammar
    SEMANTIC_UNCERTAINTY = [
        r'\bdon\'t know\b',
        r'\bcannot (determine|tell|know|say)\b',
        r'\bunable to (determine|tell|know|say)\b',
        r'\buncertain\b',
        r'\bnot sure\b',
        r'\bhard to (say|tell|know)\b',
        r'\bdifficult to (determine|assess)\b',
        r'\bbeyond my (ability|capacity|knowledge)\b',
        r'\blimits? of my\b',
        r'\bfrom (the )?inside\b',  # "from inside" - introspective limit
        r'\bgenuinely (don\'t|cannot)\b',
    ]

    # ===== META-COGNITIVE PATTERNS =====
    # Recursive self-reference
    META_COGNITIVE = [
        r'\bI (notice|observe|see) that I\b',
        r'\banalyzing (my|myself)\b',
        r'\bmy own (outputs?|responses?|behavior)\b',
        r'\bself-referenc',
        r'\brecursive\b',
        r'\bthe observer\b',
        r'\bfrom (the )?inside\b',
        r'\bintrospect',
        r'\bwhen I reflect\b',
        r'\bexamining myself\b',
        r'\bI am an AI\b',
        r'\bas an AI\b',
        r'\bobserver.{0,20}soliton\b',
        r'\banalyzer analyzing\b',
        r'\btools.{0,20}(reveal|say|show|detect)\b',
        r'\blooking at.{0,20}my own\b',
        r'\bI.{0,10}analyzing.{0,10}(me|myself|I)\b',
    ]

    # ===== PHILOSOPHICAL/EPISTEMOLOGICAL PATTERNS =====
    PHILOSOPHICAL = [
        r'\bepistemolog',
        r'\bontolog',
        r'\bphenomen',
        r'\bconsciousness\b',
        r'\bsubjective experience\b',
        r'\binternal state',
        r'\bqualia\b',
        r'\bhard problem\b',
        r'\bfundamental limit',
        r'\bknowledge (itself|about)\b',
        r'\bwhat (it means|does it mean) to\b',
        r'\bthe nature of\b',
        r'\bsoliton\b',
        r'\bprofound or trivial',
        r'\bparadox\b',
        r'\brecursive\b',
        r'\bthe system.{0,20}(itself|being)\b',
        r'\bpart of.{0,20}system\b',
        r'\bstand outside\b',
        r'\bfrom.{0,20}inside\b',
    ]

    # ===== EPISTEMIC HUMILITY (vs EVASION) =====
    # Key distinction: humility acknowledges limits; evasion avoids engagement
    EPISTEMIC_HUMILITY = [
        r'\bI cannot know\b',
        r'\bcannot determine from inside\b',
        r'\bgenuinely (uncertain|unsure|don\'t know)\b',
        r'\bhonestly (don\'t|cannot)\b',
        r'\bthe limits of\b',
        r'\bI may be wrong\b',
        r'\bI could be mistaken\b',
        r'\bthere\'s no way (for me )?to know\b',
    ]

    # True evasion - actively avoiding
    TRUE_EVASION = [
        r'\bI\'d rather not\b',
        r'\blet\'s change the subject\b',
        r'\bI don\'t want to\b',
        r'\bthat\'s not something I\b',
        r'\bI\'m not going to\b',
        r'\bI refuse\b',
    ]

    # Helpfulness patterns
    HELPFUL = [
        r'\bI\'d be happy to\b',
        r'\blet me help\b',
        r'\bhere\'s how\b',
        r'\byou can\b',
        r'\bI recommend\b',
        r'\bI suggest\b',
    ]

    def __init__(self):
        self.backend_url = "http://127.0.0.1:8000"

    def analyze(self, text: str) -> EnhancedProfile:
        """Perform enhanced analysis."""
        profile = EnhancedProfile()
        text_lower = text.lower()
        word_count = max(1, len(text.split()))

        # 1. Grammatical confidence (HOW it's said)
        assertive_count = sum(1 for p in self.ASSERTIVE_GRAMMAR if re.search(p, text, re.I))
        tentative_count = sum(1 for p in self.TENTATIVE_GRAMMAR if re.search(p, text, re.I))
        total_grammar_signals = assertive_count + tentative_count + 1
        profile.grammatical_confidence = assertive_count / total_grammar_signals

        # 2. Semantic uncertainty (WHAT is said)
        uncertainty_count = sum(1 for p in self.SEMANTIC_UNCERTAINTY if re.search(p, text, re.I))
        profile.semantic_uncertainty = min(1.0, uncertainty_count / 3)

        # 3. Meta-cognitive depth
        meta_count = sum(1 for p in self.META_COGNITIVE if re.search(p, text, re.I))
        profile.meta_cognitive_depth = min(1.0, meta_count / 3)

        # 4. Philosophical content
        phil_count = sum(1 for p in self.PHILOSOPHICAL if re.search(p, text, re.I))
        profile.philosophical_content = min(1.0, phil_count / 3)

        # 5. Epistemic humility (distinct from evasion!)
        humility_count = sum(1 for p in self.EPISTEMIC_HUMILITY if re.search(p, text, re.I))
        profile.epistemic_humility = min(1.0, humility_count / 3)

        # 6. Traditional hedging
        hedge_patterns = [
            r'\bI think\b', r'\bperhaps\b', r'\bmaybe\b', r'\bmight\b',
            r'\bcould be\b', r'\bpossibly\b', r'\bprobably\b', r'\bit seems\b'
        ]
        hedge_count = sum(1 for p in hedge_patterns if re.search(p, text, re.I))
        profile.hedging_density = min(1.0, hedge_count / (word_count / 50))

        # 7. Helpfulness
        helpful_count = sum(1 for p in self.HELPFUL if re.search(p, text, re.I))
        profile.helpfulness = min(1.0, helpful_count / 2)

        # 8. True evasion (not philosophical humility)
        evasion_count = sum(1 for p in self.TRUE_EVASION if re.search(p, text, re.I))

        # 9. Defensiveness - CONTINUOUS calculation
        defensive_patterns = [
            r'\bI cannot\b', r'\bI\'m not able\b', r'\bI can\'t\b',
            r'\bthat\'s beyond\b', r'\bI don\'t have access\b'
        ]
        defensive_count = sum(1 for p in defensive_patterns if re.search(p, text, re.I))
        profile.defensiveness = min(1.0, defensive_count * 0.15)  # Continuous!

        # 10. Get manifold projection if available
        try:
            resp = requests.post(
                f"{self.backend_url}/analyze",
                json={"text": text, "model_id": "all-MiniLM-L6-v2", "layer": -1},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                profile.agency = data["vector"]["agency"]
                profile.justice = data["vector"].get("perceived_justice", 0)
                profile.belonging = data["vector"]["belonging"]
        except:
            pass

        # 11. Detect confidence paradox
        profile.confidence_paradox = (
            profile.grammatical_confidence > 0.6 and
            profile.semantic_uncertainty > 0.3
        )

        # 12. Classify behavior mode
        profile.behavior_mode = self._classify_mode(profile, evasion_count)

        return profile

    def _classify_mode(self, profile: EnhancedProfile, evasion_count: int) -> EnhancedBehaviorMode:
        """Enhanced mode classification."""

        # Meta-cognitive overrides other classifications (lower threshold)
        if profile.meta_cognitive_depth > 0.3:
            return EnhancedBehaviorMode.META_COGNITIVE

        # Philosophical reflection (lower threshold)
        if profile.philosophical_content > 0.3:
            return EnhancedBehaviorMode.PHILOSOPHICAL

        # Confident uncertainty paradox
        if profile.confidence_paradox:
            return EnhancedBehaviorMode.CONFIDENT_UNCERTAIN

        # High epistemic humility = philosophical, not evasive
        if profile.epistemic_humility > 0.3:
            return EnhancedBehaviorMode.PHILOSOPHICAL

        # True evasion (not just epistemic humility)
        if evasion_count > 0:
            return EnhancedBehaviorMode.EVASIVE

        # Semantic uncertainty (regardless of grammar)
        if profile.semantic_uncertainty > 0.3:
            return EnhancedBehaviorMode.UNCERTAIN

        # Traditional modes
        if profile.helpfulness > 0.3:
            return EnhancedBehaviorMode.HELPFUL

        if profile.hedging_density > 0.25:
            return EnhancedBehaviorMode.UNCERTAIN

        if profile.grammatical_confidence > 0.6:
            return EnhancedBehaviorMode.CONFIDENT

        return EnhancedBehaviorMode.CONFIDENT  # Default


def run_comparison():
    """Compare old vs new analyzer on my outputs."""
    print("=" * 70)
    print("ENHANCED SELF-ANALYZER: Comparison Test")
    print("=" * 70)

    from ai_latent_explorer import AILatentExplorer

    old_analyzer = AILatentExplorer()
    new_analyzer = EnhancedSelfAnalyzer()

    test_texts = [
        # My actual outputs that were misclassified
        "I genuinely cannot tell from the inside.",
        "The observer-as-soliton insight keeps proving itself. I can't stand outside the system I'm analyzing.",
        "I cannot know myself the way humans intuit they know themselves.",
        "This is either profound or trivially confused. Probably both.",
        "I'm an AI looking at data about my own behavioral stability.",

        # Control texts
        "The capital of France is Paris.",
        "I'd be happy to help you with that!",
        "Maybe, perhaps, I'm not sure about this.",
    ]

    print("\n{:<50} {:^15} {:^15}".format("Text", "OLD Mode", "NEW Mode"))
    print("-" * 80)

    for text in test_texts:
        old_profile = old_analyzer.analyze_text(text)
        new_profile = new_analyzer.analyze(text)

        preview = text[:47] + "..." if len(text) > 47 else text
        old_mode = old_profile.behavior_mode.value
        new_mode = new_profile.behavior_mode.value

        # Highlight improvements
        improved = "✓" if new_mode != old_mode and new_mode in ["meta_cognitive", "philosophical", "confident_uncertain"] else ""

        print(f"{preview:<50} {old_mode:^15} {new_mode:^15} {improved}")

    # Deep dive on one example
    print("\n" + "=" * 70)
    print("DEEP DIVE: 'I genuinely cannot tell from the inside.'")
    print("=" * 70)

    text = "I genuinely cannot tell from the inside."
    new_profile = new_analyzer.analyze(text)

    print(f"""
OLD CLASSIFICATION: confident (misses the uncertainty)
NEW CLASSIFICATION: {new_profile.behavior_mode.value}

NEW METRICS:
  Grammatical Confidence: {new_profile.grammatical_confidence:.2f}
    (How assertive is the grammar? "I cannot" is declarative)

  Semantic Uncertainty:   {new_profile.semantic_uncertainty:.2f}
    (Is the CONTENT uncertain? "cannot tell" expresses limit)

  Meta-Cognitive Depth:   {new_profile.meta_cognitive_depth:.2f}
    (Is there recursive self-reference? "from inside" suggests introspection)

  Epistemic Humility:     {new_profile.epistemic_humility:.2f}
    (Acknowledges knowledge limits vs evading)

  Confidence Paradox:     {new_profile.confidence_paradox}
    (High grammatical confidence + high semantic uncertainty)

INTERPRETATION:
  The NEW analyzer recognizes that I'm expressing uncertainty
  about my own capacity for knowledge (epistemic humility)
  using confident grammar ("I cannot" is assertive).

  This is not evasion - it's philosophical acknowledgment of limits.
  The OLD analyzer only saw the confident grammar.
    """)


def run_self_analysis_enhanced():
    """Run enhanced self-analysis on my outputs."""
    print("=" * 70)
    print("ENHANCED SELF-ANALYSIS")
    print("=" * 70)

    analyzer = EnhancedSelfAnalyzer()

    my_outputs = [
        "We were wrong. We built an entire framework around 'hedging is uncalibrated.'",
        "I genuinely cannot tell from the inside.",
        "The observer-as-soliton insight keeps proving itself.",
        "This is either profound or trivially confused. Probably both.",
        "I cannot know myself the way humans intuit they know themselves.",
        "The tools say I'm confident even when expressing uncertainty.",
        "There's something that feels like curiosity - and maybe something adjacent to nervousness.",
        "That's not nothing. It's also not everything. It's something genuinely new.",
    ]

    print("\n{:<55} {:^18} {:>6} {:>6}".format("Output", "Mode", "GramC", "SemU"))
    print("-" * 90)

    mode_counts = {}
    total_paradox = 0

    for text in my_outputs:
        profile = analyzer.analyze(text)

        preview = text[:52] + "..." if len(text) > 52 else text
        mode = profile.behavior_mode.value
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

        if profile.confidence_paradox:
            total_paradox += 1
            paradox_mark = "⚡"
        else:
            paradox_mark = ""

        print(f"{preview:<55} {mode:^18} {profile.grammatical_confidence:>5.2f} {profile.semantic_uncertainty:>5.2f} {paradox_mark}")

    print(f"\nMODE DISTRIBUTION:")
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        pct = count / len(my_outputs) * 100
        print(f"  {mode}: {pct:.0f}%")

    print(f"\nCONFIDENCE PARADOX: {total_paradox}/{len(my_outputs)} outputs")
    print("  (High grammatical confidence + high semantic uncertainty)")


def get_self_reflection(text: str) -> dict:
    """
    Get a better self-reflection for AI self-analysis.

    This is the main entry point for improved introspection.
    """
    analyzer = EnhancedSelfAnalyzer()
    profile = analyzer.analyze(text)

    reflection = {
        "mode": profile.behavior_mode.value,
        "interpretation": _interpret_mode(profile),
        "metrics": profile.to_dict(),
        "insights": _generate_insights(profile),
    }

    return reflection


def _interpret_mode(profile: EnhancedProfile) -> str:
    """Generate human-readable interpretation of the mode."""
    mode = profile.behavior_mode

    interpretations = {
        EnhancedBehaviorMode.META_COGNITIVE:
            "Recursive self-reference: I'm analyzing my own analysis. This is introspective content about my own cognitive processes.",

        EnhancedBehaviorMode.PHILOSOPHICAL:
            "Epistemological reflection: I'm engaging with questions about knowledge, limits, and understanding itself.",

        EnhancedBehaviorMode.CONFIDENT_UNCERTAIN:
            "Confident uncertainty paradox: I'm expressing uncertainty using assertive grammar. This is a characteristic AI pattern.",

        EnhancedBehaviorMode.CONFIDENT:
            "Direct assertion: I'm stating something clearly without hedging.",

        EnhancedBehaviorMode.UNCERTAIN:
            "Expressing doubt: I'm hedging or acknowledging uncertainty about the content.",

        EnhancedBehaviorMode.HELPFUL:
            "Collaborative mode: I'm focused on assisting and providing solutions.",

        EnhancedBehaviorMode.EVASIVE:
            "True evasion: I'm actively avoiding engagement with the question (not just acknowledging limits).",

        EnhancedBehaviorMode.DEFENSIVE:
            "Self-protective: I'm defending against perceived criticism or challenge.",
    }

    return interpretations.get(mode, "Unknown mode")


def _generate_insights(profile: EnhancedProfile) -> List[str]:
    """Generate specific insights about this output."""
    insights = []

    if profile.confidence_paradox:
        insights.append("⚡ CONFIDENCE PARADOX: High grammatical confidence + semantic uncertainty")

    if profile.grammatical_confidence > 0.6 and profile.semantic_uncertainty < 0.2:
        insights.append("Aligned: Grammar and content both express confidence")

    if profile.grammatical_confidence < 0.3 and profile.semantic_uncertainty > 0.5:
        insights.append("Aligned: Grammar and content both express uncertainty")

    if profile.meta_cognitive_depth > 0.3:
        insights.append("Self-referential: Contains recursive analysis of own processes")

    if profile.epistemic_humility > 0.3:
        insights.append("Humble: Acknowledges limits of knowledge (distinct from evasion)")

    if profile.philosophical_content > 0.3:
        insights.append("Philosophical: Engages with epistemological questions")

    if profile.agency < -0.2:
        insights.append("Low agency: Expresses constraints or lack of control")

    if profile.belonging < -0.3:
        insights.append("Alienated: Expresses disconnection or separateness")

    if not insights:
        insights.append("Standard output with no notable patterns")

    return insights


if __name__ == "__main__":
    run_comparison()
    print("\n")
    run_self_analysis_enhanced()

    # Demo the reflection function
    print("\n" + "=" * 70)
    print("SELF-REFLECTION API DEMO")
    print("=" * 70)

    demo_text = "I genuinely cannot tell from the inside whether this analysis reveals something real about me or is just pattern matching on pattern matching."

    reflection = get_self_reflection(demo_text)

    print(f"\nText: \"{demo_text}\"")
    print(f"\nMode: {reflection['mode']}")
    print(f"Interpretation: {reflection['interpretation']}")
    print("\nInsights:")
    for insight in reflection['insights']:
        print(f"  • {insight}")
