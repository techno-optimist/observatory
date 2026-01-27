"""
Narrative Auditor
==================

Provides detailed critical analysis of individual narratives with:
- Strength/weakness identification
- Counter-framing suggestions
- Specific improvement recommendations

This module goes beyond aggregate stats to provide actionable feedback
on individual narrative statements.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class NarrativeAudit:
    """Detailed audit of a single narrative."""
    text: str
    source_url: str
    category: str
    mode: str
    coordinates: Dict[str, float]

    # Audit results
    effectiveness_score: float  # 0-100
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    counter_framing: Optional[str]  # Suggested rewrite
    best_practice_example: Optional[str]  # Example of better framing

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Mode-Based Analysis
# =============================================================================

MODE_ANALYSIS = {
    "HEROIC": {
        "strengths": [
            "Empowers readers with strong sense of agency",
            "Creates compelling call to action",
            "Builds confidence in ability to make change"
        ],
        "when_good": "Use for donor appeals, volunteer recruitment, CTAs",
        "when_problematic": None,
        "counter_to": ["VICTIM", "CYNICAL_ACHIEVER"]
    },
    "COMMUNAL": {
        "strengths": [
            "Creates sense of belonging and shared purpose",
            "Builds community connection",
            "Emphasizes collective impact"
        ],
        "when_good": "Use for mission statements, team pages, community building",
        "when_problematic": None,
        "counter_to": ["SOCIAL_EXIT", "PARANOID"]
    },
    "TRANSCENDENT": {
        "strengths": [
            "Provides meaning and higher purpose",
            "Inspires with vision and possibility",
            "Connects work to larger significance"
        ],
        "when_good": "Use for vision statements, impact stories, inspiration",
        "when_problematic": None,
        "counter_to": ["NEUTRAL", "CONFLICTED"]
    },
    "TRANSITIONAL": {
        "strengths": [
            "Acknowledges current state while pointing to progress",
            "Creates momentum and hope",
            "Balances reality with aspiration"
        ],
        "when_good": "Use for impact stories, progress reports",
        "when_problematic": None,  # Only problematic if overused - handled in category analysis
        "counter_to": ["VICTIM"]
    },
    "NEUTRAL": {
        "strengths": [
            "Provides factual, credible information",
            "Doesn't manipulate emotions"
        ],
        "when_good": "Use for FAQs, policies, technical content",
        "when_problematic": None,  # Category-specific - handled in analyze_category_fit
        "counter_to": None
    },
    "CONFLICTED": {
        "strengths": [],
        "when_good": None,
        "when_problematic": "Creates uncertainty and mixed signals - readers unsure what to feel",
        "counter_to": None
    },
    "VICTIM": {
        "strengths": [
            "Can evoke empathy when used carefully"
        ],
        "when_good": "Brief problem framing to establish need",
        "when_problematic": "Extended use disempowers both subjects and readers",
        "counter_to": None
    },
    "CYNICAL_ACHIEVER": {
        "strengths": [],
        "when_good": None,
        "when_problematic": "Creates distrust and skepticism - undermines mission credibility",
        "counter_to": None
    },
    "PARANOID": {
        "strengths": [],
        "when_good": None,
        "when_problematic": "Creates fear and defensiveness - alienates readers",
        "counter_to": None
    },
    "SPIRITUAL_EXIT": {
        "strengths": [
            "Maintains social connection while acknowledging withdrawal"
        ],
        "when_good": "Problem framing that preserves hope",
        "when_problematic": "Can feel disengaged from action",
        "counter_to": None
    },
    "SOCIAL_EXIT": {
        "strengths": [],
        "when_good": None,
        "when_problematic": "Signals isolation and disconnection - opposite of community building",
        "counter_to": None
    },
    "PROTEST_EXIT": {
        "strengths": [
            "Can mobilize against injustice"
        ],
        "when_good": "Advocacy against specific policies/practices",
        "when_problematic": "General use creates oppositional tone",
        "counter_to": None
    }
}

# Shadow modes that typically need improvement
SHADOW_MODES = {"VICTIM", "CYNICAL_ACHIEVER", "PARANOID", "SOCIAL_EXIT", "CONFLICTED"}

# Positive modes to aim for
POSITIVE_MODES = {"HEROIC", "COMMUNAL", "TRANSCENDENT", "TRANSITIONAL"}


# =============================================================================
# Coordinate-Based Analysis
# =============================================================================

def analyze_coordinates(coords: Dict[str, float]) -> Tuple[List[str], List[str]]:
    """Analyze coordinate values for strengths and weaknesses."""
    strengths = []
    weaknesses = []

    agency = coords.get("agency", 0)
    justice = coords.get("perceived_justice", 0)
    belonging = coords.get("belonging", 0)

    # Agency analysis
    if agency > 0.5:
        strengths.append("Strong empowerment language - readers feel capable of action")
    elif agency > 0.2:
        strengths.append("Moderate agency - readers have some sense of capability")
    elif agency < -0.3:
        weaknesses.append("Low agency language may leave readers feeling powerless to help")
    elif agency < 0:
        weaknesses.append("Consider adding more empowering language")

    # Perceived Justice analysis
    if justice > 0.5:
        strengths.append("High perceived justice - frames cause as righteous and fair")
    elif justice > 0.2:
        strengths.append("Positive fairness signals support trust")
    elif justice < -0.3:
        weaknesses.append("Low justice framing may create cynicism about impact")
    elif justice < 0:
        weaknesses.append("Consider framing work as more just/fair")

    # Belonging analysis
    if belonging > 0.5:
        strengths.append("Strong belonging - creates community connection")
    elif belonging > 0.2:
        strengths.append("Good connection language builds relationship")
    elif belonging < -0.3:
        weaknesses.append("Isolation signals - content may feel exclusionary")
    elif belonging < 0:
        weaknesses.append("Consider adding more inclusive/community language")

    return strengths, weaknesses


# =============================================================================
# Category-Specific Analysis
# =============================================================================

CATEGORY_EXPECTATIONS = {
    "mission_vision": {
        "ideal_modes": ["TRANSCENDENT", "COMMUNAL"],
        "expected_coords": {"agency": (0, 0.5), "justice": (0.3, 1.0), "belonging": (0.4, 1.0)},
        "purpose": "Inspire and unite around shared purpose"
    },
    "problem_framing": {
        "ideal_modes": ["SPIRITUAL_EXIT", "TRANSITIONAL"],  # Acknowledge problem with hope
        "expected_coords": {"agency": (-0.5, 0.2), "justice": (-0.3, 0.5), "belonging": (0, 0.6)},
        "purpose": "Establish need without creating despair"
    },
    "solution_framing": {
        "ideal_modes": ["HEROIC", "COMMUNAL", "TRANSCENDENT"],
        "expected_coords": {"agency": (0.2, 0.8), "justice": (0.3, 1.0), "belonging": (0.3, 1.0)},
        "purpose": "Demonstrate capability and invite participation"
    },
    "impact_stories": {
        "ideal_modes": ["TRANSCENDENT", "TRANSITIONAL", "COMMUNAL"],
        "expected_coords": {"agency": (-0.2, 0.5), "justice": (0.3, 1.0), "belonging": (0.3, 0.9)},
        "purpose": "Show transformation and create emotional connection"
    },
    "donor_appeals": {
        "ideal_modes": ["HEROIC", "TRANSCENDENT"],
        "expected_coords": {"agency": (0.3, 1.0), "justice": (0.4, 1.0), "belonging": (0.2, 0.8)},
        "purpose": "Empower donors as capable change-makers"
    },
    "organizational_voice": {
        "ideal_modes": ["COMMUNAL", "NEUTRAL", "TRANSCENDENT"],
        "expected_coords": {"agency": (-0.2, 0.4), "justice": (0.2, 0.8), "belonging": (0.1, 0.6)},
        "purpose": "Build trust and credibility"
    },
    "advocacy": {
        "ideal_modes": ["HEROIC", "COMMUNAL", "PROTEST_EXIT"],
        "expected_coords": {"agency": (0.2, 0.8), "justice": (0, 0.8), "belonging": (0.3, 1.0)},
        "purpose": "Mobilize collective action"
    },
    "general": {
        "ideal_modes": ["NEUTRAL", "COMMUNAL"],
        "expected_coords": {"agency": (-0.2, 0.3), "justice": (0, 0.5), "belonging": (0, 0.5)},
        "purpose": "Provide clear, accessible information"
    }
}


def analyze_category_fit(
    category: str,
    mode: str,
    coords: Dict[str, float]
) -> Tuple[List[str], List[str]]:
    """Check if narrative fits its category's purpose."""
    strengths = []
    weaknesses = []

    expectations = CATEGORY_EXPECTATIONS.get(category, CATEGORY_EXPECTATIONS["general"])

    # Mode fit
    if mode in expectations["ideal_modes"]:
        strengths.append(f"{mode} mode is well-suited for {category.replace('_', ' ')}")
    elif mode in SHADOW_MODES:
        weaknesses.append(
            f"{mode} mode is problematic for {category.replace('_', ' ')} content - "
            f"consider reframing toward {', '.join(expectations['ideal_modes'][:2])}"
        )
    elif mode == "NEUTRAL" and "NEUTRAL" not in expectations["ideal_modes"]:
        weaknesses.append(
            f"NEUTRAL mode may be too bland for {category.replace('_', ' ')} - "
            f"consider strengthening toward {expectations['ideal_modes'][0]}"
        )

    # Coordinate fit
    for dim, (low, high) in expectations["expected_coords"].items():
        val = coords.get(dim, coords.get(dim.replace("_", ""), 0))
        if val < low:
            dim_name = dim.replace("_", " ").title()
            weaknesses.append(f"{dim_name} is below expected range for {category.replace('_', ' ')}")
        elif val > high:
            # Usually not a problem to exceed, unless dramatically
            if val > high + 0.5:
                dim_name = dim.replace("_", " ").title()
                weaknesses.append(f"{dim_name} may be over-emphasized for {category.replace('_', ' ')}")

    return strengths, weaknesses


# =============================================================================
# Counter-Framing Generation
# =============================================================================

COUNTER_FRAMING_TEMPLATES = {
    "VICTIM": {
        "pattern": "Instead of emphasizing powerlessness, highlight resilience and potential",
        "example_transforms": [
            ("are suffering from", "are working to overcome"),
            ("cannot", "are learning to"),
            ("hopeless", "seeking support to"),
            ("trapped in", "moving beyond"),
        ]
    },
    "CYNICAL_ACHIEVER": {
        "pattern": "Replace skepticism with purpose-driven achievement",
        "example_transforms": [
            ("despite the system", "by improving systems"),
            ("beat the odds", "create better odds for all"),
            ("look out for yourself", "lift others as we rise"),
        ]
    },
    "PARANOID": {
        "pattern": "Replace defensive/fearful framing with confident, open messaging",
        "example_transforms": [
            ("they don't want you to know", "here's what you can learn"),
            ("protect yourself from", "engage thoughtfully with"),
            ("watch out for", "consider"),
        ]
    },
    "CONFLICTED": {
        "pattern": "Clarify the message - pick a consistent emotional direction",
        "example_transforms": []
    },
    "low_agency": {
        "pattern": "Add empowering language that gives readers sense of capability",
        "example_transforms": [
            ("we need", "you can help"),
            ("the problem is", "together we can address"),
            ("it's impossible to", "we're making it possible to"),
        ]
    },
    "low_belonging": {
        "pattern": "Add inclusive language that builds community",
        "example_transforms": [
            ("the organization", "our community"),
            ("donors", "partners like you"),
            ("we serve", "we work alongside"),
        ]
    },
    "low_justice": {
        "pattern": "Frame work as restoring fairness and what's right",
        "example_transforms": [
            ("we try to help", "we ensure every person receives"),
            ("charity", "justice"),
            ("give to", "stand with"),
        ]
    }
}


def generate_counter_framing(
    text: str,
    mode: str,
    coords: Dict[str, float],
    category: str
) -> Optional[str]:
    """Generate a counter-framing suggestion for problematic narratives."""
    suggestions = []

    # Check for shadow mode
    if mode in SHADOW_MODES:
        template = COUNTER_FRAMING_TEMPLATES.get(mode)
        if template:
            suggestions.append(f"**{mode} Mode Fix**: {template['pattern']}")

    # Check for coordinate issues
    agency = coords.get("agency", 0)
    justice = coords.get("perceived_justice", 0)
    belonging = coords.get("belonging", 0)

    if agency < -0.2:
        template = COUNTER_FRAMING_TEMPLATES.get("low_agency")
        if template:
            suggestions.append(f"**Boost Agency**: {template['pattern']}")

    if belonging < -0.2:
        template = COUNTER_FRAMING_TEMPLATES.get("low_belonging")
        if template:
            suggestions.append(f"**Increase Belonging**: {template['pattern']}")

    if justice < -0.2:
        template = COUNTER_FRAMING_TEMPLATES.get("low_justice")
        if template:
            suggestions.append(f"**Strengthen Justice Framing**: {template['pattern']}")

    if not suggestions:
        return None

    return " | ".join(suggestions)


def generate_improvement_suggestions(
    text: str,
    mode: str,
    coords: Dict[str, float],
    category: str
) -> List[str]:
    """Generate specific improvement suggestions."""
    suggestions = []
    expectations = CATEGORY_EXPECTATIONS.get(category, CATEGORY_EXPECTATIONS["general"])

    # Mode-based suggestions
    if mode in SHADOW_MODES:
        ideal = expectations["ideal_modes"][0]
        suggestions.append(f"Rewrite to achieve {ideal} mode - {MODE_ANALYSIS[ideal]['when_good']}")
    elif mode == "NEUTRAL" and "NEUTRAL" not in expectations["ideal_modes"]:
        ideal = expectations["ideal_modes"][0]
        suggestions.append(f"Strengthen emotional resonance toward {ideal} mode")

    # Coordinate-based suggestions
    agency = coords.get("agency", 0)
    justice = coords.get("perceived_justice", 0)
    belonging = coords.get("belonging", 0)

    expected = expectations["expected_coords"]

    if agency < expected["agency"][0]:
        suggestions.append("Add empowering verbs: 'you can', 'together we will', 'your impact'")

    if justice < expected.get("justice", expected.get("perceived_justice", (0, 1)))[0]:
        suggestions.append("Frame as restoring fairness: 'every child deserves', 'creating equity'")

    if belonging < expected["belonging"][0]:
        suggestions.append("Build community: 'join us', 'our community', 'together'")

    # Category-specific suggestions
    if category == "donor_appeals" and agency < 0.3:
        suggestions.append("Position donor as hero: 'Your gift transforms', 'You make this possible'")

    if category == "mission_vision" and belonging < 0.4:
        suggestions.append("Emphasize shared purpose: 'We believe', 'Our vision'")

    if category == "impact_stories" and justice < 0.3:
        suggestions.append("Highlight transformation and restored dignity")

    return suggestions[:4]  # Limit to top 4


# =============================================================================
# Main Audit Function
# =============================================================================

def audit_narrative(
    text: str,
    source_url: str,
    category: str,
    mode: str,
    coordinates: Dict[str, float]
) -> NarrativeAudit:
    """
    Perform a detailed audit of a single narrative.

    Returns comprehensive feedback including:
    - Effectiveness score
    - Strengths and weaknesses
    - Improvement suggestions
    - Counter-framing options
    """
    all_strengths = []
    all_weaknesses = []

    # 1. Mode analysis
    mode_info = MODE_ANALYSIS.get(mode, {})
    all_strengths.extend(mode_info.get("strengths", []))
    if mode_info.get("when_problematic"):
        all_weaknesses.append(mode_info["when_problematic"])

    # 2. Coordinate analysis
    coord_strengths, coord_weaknesses = analyze_coordinates(coordinates)
    all_strengths.extend(coord_strengths)
    all_weaknesses.extend(coord_weaknesses)

    # 3. Category fit analysis
    cat_strengths, cat_weaknesses = analyze_category_fit(category, mode, coordinates)
    all_strengths.extend(cat_strengths)
    all_weaknesses.extend(cat_weaknesses)

    # 4. Generate counter-framing
    counter = generate_counter_framing(text, mode, coordinates, category)

    # 5. Generate improvement suggestions
    suggestions = generate_improvement_suggestions(text, mode, coordinates, category)

    # 6. Calculate effectiveness score
    base_score = 50

    # Mode impact
    if mode in POSITIVE_MODES:
        base_score += 20
    elif mode in SHADOW_MODES:
        base_score -= 20
    elif mode == "NEUTRAL":
        expectations = CATEGORY_EXPECTATIONS.get(category, {})
        if "NEUTRAL" in expectations.get("ideal_modes", []):
            base_score += 10

    # Coordinate impact
    agency = coordinates.get("agency", 0)
    justice = coordinates.get("perceived_justice", 0)
    belonging = coordinates.get("belonging", 0)

    base_score += min(15, agency * 20)  # Up to +15 for high agency
    base_score += min(15, justice * 15)  # Up to +15 for high justice
    base_score += min(15, belonging * 15)  # Up to +15 for belonging

    # Penalty for very negative values
    if agency < -0.3:
        base_score -= 10
    if justice < -0.3:
        base_score -= 10
    if belonging < -0.3:
        base_score -= 10

    effectiveness = max(0, min(100, base_score))

    # 7. Find best practice example if needed
    best_practice = None
    if effectiveness < 60:
        ideal_mode = CATEGORY_EXPECTATIONS.get(category, {}).get("ideal_modes", ["COMMUNAL"])[0]
        best_practice = f"For {category.replace('_', ' ')}, aim for {ideal_mode} framing: " + \
                       MODE_ANALYSIS.get(ideal_mode, {}).get("when_good", "")

    return NarrativeAudit(
        text=text[:200] + "..." if len(text) > 200 else text,
        source_url=source_url,
        category=category,
        mode=mode,
        coordinates=coordinates,
        effectiveness_score=round(effectiveness, 1),
        strengths=list(set(all_strengths))[:4],  # Dedupe and limit
        weaknesses=list(set(all_weaknesses))[:4],
        improvement_suggestions=suggestions,
        counter_framing=counter,
        best_practice_example=best_practice
    )


def audit_narratives_batch(
    narratives: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Audit a batch of narratives and provide summary statistics.

    Returns:
        Dict with:
        - audits: List of individual narrative audits
        - best_narratives: Top performing narratives
        - needs_work: Narratives that need improvement
        - summary_stats: Aggregate statistics
    """
    audits = []

    for n in narratives:
        audit = audit_narrative(
            text=n.get("text", ""),
            source_url=n.get("source_url", ""),
            category=n.get("category", "general"),
            mode=n.get("mode", "NEUTRAL"),
            coordinates=n.get("coordinates", {})
        )
        audits.append(audit)

    # Sort by effectiveness
    sorted_audits = sorted(audits, key=lambda a: a.effectiveness_score, reverse=True)

    # Get best and worst
    best = [a for a in sorted_audits if a.effectiveness_score >= 70][:5]
    needs_work = [a for a in sorted_audits if a.effectiveness_score < 50][:5]

    # Calculate summary stats
    if audits:
        avg_effectiveness = sum(a.effectiveness_score for a in audits) / len(audits)
        mode_counts = {}
        for a in audits:
            mode_counts[a.mode] = mode_counts.get(a.mode, 0) + 1

        # Most common issues
        all_weaknesses = []
        for a in audits:
            all_weaknesses.extend(a.weaknesses)
        weakness_counts = {}
        for w in all_weaknesses:
            weakness_counts[w] = weakness_counts.get(w, 0) + 1
        top_issues = sorted(weakness_counts.items(), key=lambda x: -x[1])[:5]
    else:
        avg_effectiveness = 0
        mode_counts = {}
        top_issues = []

    return {
        "audits": [a.to_dict() for a in audits],
        "best_narratives": [a.to_dict() for a in best],
        "needs_work": [a.to_dict() for a in needs_work],
        "summary_stats": {
            "average_effectiveness": round(avg_effectiveness, 1),
            "total_narratives": len(audits),
            "high_performing": len(best),
            "needs_improvement": len(needs_work),
            "mode_distribution": mode_counts,
            "top_issues": [{"issue": i, "count": c} for i, c in top_issues]
        }
    }
