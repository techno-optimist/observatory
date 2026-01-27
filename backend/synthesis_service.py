"""
Synthesis Service for Comprehensive Report Generation
======================================================

This module uses the Claude API to generate narrative insights and synthesize
comprehensive reports from Cultural Soliton Observatory analyses.

The service transforms structured analysis data into human-readable narratives
following the format demonstrated in case studies like The Hope Effect analysis.

Features:
- Executive summary generation
- Category-specific deep dive narratives
- Strategic insights and pattern identification
- Actionable recommendations
- Full report orchestration

Usage:
    from synthesis_service import SynthesisService

    service = SynthesisService()
    report = await service.synthesize_full_report(site_data)

Environment:
    ANTHROPIC_API_KEY: Required API key for Claude access
"""

import os
import logging
from typing import Dict, List, Any, Optional, TypedDict
from collections import defaultdict
from statistics import mean, stdev

import anthropic

logger = logging.getLogger(__name__)


# ============================================================================
# Type Definitions
# ============================================================================

class ModeInfo(TypedDict):
    """Mode classification result."""
    primary_mode: str
    confidence: float
    secondary_mode: Optional[str]


class Coordinates(TypedDict):
    """3D coordinate position in cultural manifold."""
    agency: float
    perceived_justice: float
    belonging: float


class ForceField(TypedDict, total=False):
    """Force field analysis result."""
    attractor_strength: float
    detractor_strength: float
    primary_attractor: Optional[str]
    primary_detractor: Optional[str]
    force_quadrant: str


class Analysis(TypedDict, total=False):
    """Individual narrative analysis result."""
    text: str
    url: str
    category: str
    mode: ModeInfo
    coordinates: Coordinates
    force_field: ForceField


class CategoryMetrics(TypedDict):
    """Aggregated metrics for a category."""
    mean_agency: float
    mean_perceived_justice: float
    mean_belonging: float
    dominant_mode: str
    narrative_count: int


class CategoryAnalysis(TypedDict):
    """Full category analysis with narrative."""
    category: str
    metrics: CategoryMetrics
    sample_projections: List[Dict[str, Any]]
    analysis_narrative: str
    critical_insights: List[str]


class KeyFindings(TypedDict):
    """Key findings summary."""
    strengths: List[str]
    concerns: List[str]
    benchmarks: List[Dict[str, Any]]


class FullReport(TypedDict):
    """Complete synthesized report."""
    executive_summary: str
    organization_background: Dict[str, Any]
    overall_profile: Dict[str, Any]
    category_analyses: List[CategoryAnalysis]
    cross_category_comparisons: List[Dict[str, Any]]
    key_findings: KeyFindings
    recommendations: List[str]
    conclusion: str
    health_score: int


# ============================================================================
# Constants
# ============================================================================

# Category display names mapping
CATEGORY_DISPLAY_NAMES = {
    "mission": "Mission & Vision",
    "mission_vision": "Mission & Vision",
    "impact_stories": "Impact Stories",
    "donor_appeals": "Donor Appeals",
    "problem_framing": "Problem Framing",
    "solution_framing": "Solution Framing",
    "organizational_voice": "Organizational Voice",
    "advocacy": "Advocacy",
    "team": "Team & Leadership",
    "about": "About Us",
    "homepage": "Homepage",
    "services": "Services & Programs",
    "blog": "Blog & Updates",
}

# Mode quality scores for health calculation
MODE_QUALITY_SCORES = {
    "TRANSCENDENT": 1.0,
    "COMMUNAL": 0.95,
    "HEROIC": 0.9,
    "GROWTH_MINDSET": 0.85,
    "NEUTRAL": 0.7,
    "TRANSITIONAL": 0.6,
    "CONFLICTED": 0.5,
    "SPIRITUAL_EXIT": 0.45,
    "PROTEST_EXIT": 0.4,
    "SOCIAL_EXIT": 0.35,
    "VICTIM": 0.25,
    "PARANOID": 0.2,
    "NIHILIST": 0.1,
}

# Model configuration
DEFAULT_MODEL = "claude-3-5-haiku-20241022"
FALLBACK_MODEL = "claude-3-haiku-20240307"
MAX_TOKENS = 4096


# ============================================================================
# Utility Functions
# ============================================================================

def calculate_mean_coordinates(analyses: List[Analysis]) -> Coordinates:
    """Calculate mean coordinates across analyses."""
    if not analyses:
        return {"agency": 0.0, "perceived_justice": 0.0, "belonging": 0.0}

    agency_values = []
    justice_values = []
    belonging_values = []

    for analysis in analyses:
        coords = analysis.get("coordinates", {})
        if coords:
            agency_values.append(coords.get("agency", 0))
            justice_values.append(coords.get("perceived_justice", 0))
            belonging_values.append(coords.get("belonging", 0))

    if not agency_values:
        return {"agency": 0.0, "perceived_justice": 0.0, "belonging": 0.0}

    return {
        "agency": round(mean(agency_values), 3),
        "perceived_justice": round(mean(justice_values), 3),
        "belonging": round(mean(belonging_values), 3),
    }


def calculate_mode_distribution(analyses: List[Analysis]) -> Dict[str, float]:
    """Calculate mode distribution as percentages."""
    if not analyses:
        return {}

    mode_counts: Dict[str, int] = defaultdict(int)
    total = 0

    for analysis in analyses:
        mode_info = analysis.get("mode", {})
        if isinstance(mode_info, dict):
            mode = mode_info.get("primary_mode", "NEUTRAL")
        else:
            mode = str(mode_info) if mode_info else "NEUTRAL"
        mode_counts[mode] += 1
        total += 1

    if total == 0:
        return {}

    return {
        mode: round(count / total, 3)
        for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1])
    }


def get_dominant_mode(analyses: List[Analysis]) -> str:
    """Get the most common mode from analyses."""
    distribution = calculate_mode_distribution(analyses)
    if not distribution:
        return "NEUTRAL"
    return max(distribution, key=distribution.get)


def group_analyses_by_category(analyses: List[Analysis]) -> Dict[str, List[Analysis]]:
    """Group analyses by their category."""
    grouped: Dict[str, List[Analysis]] = defaultdict(list)
    for analysis in analyses:
        category = analysis.get("category", "uncategorized")
        grouped[category].append(analysis)
    return dict(grouped)


def format_coordinate_stats(analyses: List[Analysis]) -> str:
    """Format coordinate statistics for prompt context."""
    coords = calculate_mean_coordinates(analyses)

    # Calculate standard deviations if enough data
    if len(analyses) >= 2:
        agency_vals = [a.get("coordinates", {}).get("agency", 0) for a in analyses]
        justice_vals = [a.get("coordinates", {}).get("perceived_justice", 0) for a in analyses]
        belonging_vals = [a.get("coordinates", {}).get("belonging", 0) for a in analyses]

        agency_std = round(stdev(agency_vals), 3) if len(agency_vals) >= 2 else 0
        justice_std = round(stdev(justice_vals), 3) if len(justice_vals) >= 2 else 0
        belonging_std = round(stdev(belonging_vals), 3) if len(belonging_vals) >= 2 else 0

        return (
            f"Agency: {coords['agency']} (+/-{agency_std}), "
            f"Perceived Justice: {coords['perceived_justice']} (+/-{justice_std}), "
            f"Belonging: {coords['belonging']} (+/-{belonging_std})"
        )

    return (
        f"Agency: {coords['agency']}, "
        f"Perceived Justice: {coords['perceived_justice']}, "
        f"Belonging: {coords['belonging']}"
    )


def calculate_health_score(
    analyses: List[Analysis],
    mode_distribution: Dict[str, float]
) -> int:
    """
    Calculate overall narrative health score (0-100).

    Factors:
    - Mode quality distribution (40%)
    - Coordinate balance (35%)
    - Narrative consistency (25%)
    """
    if not analyses:
        return 50

    # Mode quality score (0-40)
    mode_score = 0
    for mode, proportion in mode_distribution.items():
        quality = MODE_QUALITY_SCORES.get(mode, 0.5)
        mode_score += proportion * quality * 40

    # Coordinate balance score (0-35)
    coords = calculate_mean_coordinates(analyses)
    # Transform from [-2, 2] range to [0, 1]
    agency_norm = (coords["agency"] + 2) / 4
    justice_norm = (coords["perceived_justice"] + 2) / 4
    belonging_norm = (coords["belonging"] + 2) / 4

    coord_score = ((agency_norm + justice_norm + belonging_norm) / 3) * 35

    # Consistency score based on mode diversity (0-25)
    # Lower diversity = higher consistency (but some diversity is healthy)
    mode_count = len(mode_distribution)
    if mode_count <= 3:
        consistency_score = 25
    elif mode_count <= 5:
        consistency_score = 20
    elif mode_count <= 7:
        consistency_score = 15
    else:
        consistency_score = 10

    total = mode_score + coord_score + consistency_score
    return min(100, max(0, int(round(total))))


# ============================================================================
# Synthesis Service Class
# ============================================================================

class SynthesisService:
    """
    Service for generating narrative synthesis using Claude API.

    This service transforms structured Cultural Soliton Observatory analysis
    data into human-readable comprehensive reports with insights and
    recommendations.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the synthesis service.

        Args:
            api_key: Optional API key. If not provided, reads from
                    ANTHROPIC_API_KEY environment variable.

        Raises:
            ValueError: If no API key is available.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = DEFAULT_MODEL
        logger.info(f"SynthesisService initialized with model: {self.model}")

    def _call_claude(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = MAX_TOKENS
    ) -> str:
        """
        Make a call to Claude API with error handling and fallback.

        Args:
            system_prompt: System context for the model
            user_prompt: User message/query
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response

        Raises:
            anthropic.APIError: If both primary and fallback models fail
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text

        except anthropic.NotFoundError:
            # Model not found, try fallback
            logger.warning(
                f"Model {self.model} not found, falling back to {FALLBACK_MODEL}"
            )
            self.model = FALLBACK_MODEL
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text

        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise

        except anthropic.APIStatusError as e:
            logger.error(f"API status error: {e.status_code} - {e.message}")
            raise

        except anthropic.APIConnectionError as e:
            logger.error(f"API connection error: {e}")
            raise

    def generate_executive_summary(
        self,
        analyses: List[Analysis],
        org_name: str
    ) -> str:
        """
        Generate a high-level executive summary of findings.

        Args:
            analyses: List of narrative analyses
            org_name: Name of the organization being analyzed

        Returns:
            Executive summary text (2-3 paragraphs)
        """
        coords = calculate_mean_coordinates(analyses)
        mode_dist = calculate_mode_distribution(analyses)
        dominant_mode = get_dominant_mode(analyses)

        system_prompt = """You are an expert analyst for the Cultural Soliton Observatory,
a system that analyzes organizational narratives through a 3D cultural manifold.
Your role is to synthesize analysis data into clear, actionable executive summaries.

Key concepts:
- Agency: Individual empowerment vs collective dependency (-2 to +2)
- Perceived Justice: Fairness and righteousness framing (-2 to +2)
- Belonging: Community connection vs isolation (-2 to +2)
- Modes: Narrative archetypes (HEROIC, COMMUNAL, TRANSCENDENT, VICTIM, etc.)

Write in a professional, analytical tone. Be specific about findings and their implications."""

        user_prompt = f"""Generate an executive summary for {org_name}'s narrative analysis.

Data:
- Total narratives analyzed: {len(analyses)}
- Categories covered: {len(group_analyses_by_category(analyses))}

Coordinate Averages:
- Agency: {coords['agency']} ({"high" if coords['agency'] > 0.3 else "low" if coords['agency'] < -0.3 else "neutral"})
- Perceived Justice: {coords['perceived_justice']} ({"high" if coords['perceived_justice'] > 0.3 else "low" if coords['perceived_justice'] < -0.3 else "moderate"})
- Belonging: {coords['belonging']} ({"high" if coords['belonging'] > 0.3 else "low" if coords['belonging'] < -0.3 else "moderate"})

Mode Distribution:
{chr(10).join(f"- {mode}: {pct*100:.1f}%" for mode, pct in list(mode_dist.items())[:5])}

Dominant Mode: {dominant_mode}

Write a 2-3 paragraph executive summary that:
1. Characterizes the overall narrative strategy
2. Highlights the most significant coordinate patterns
3. Notes any mode distribution patterns worth attention
4. Mentions the "hypocrisy gap" if there are inconsistencies across categories

Be specific and analytical, not generic. Reference the actual numbers."""

        return self._call_claude(system_prompt, user_prompt)

    def generate_category_analysis(
        self,
        category_name: str,
        analyses: List[Analysis]
    ) -> CategoryAnalysis:
        """
        Generate deep dive narrative analysis for a specific category.

        Args:
            category_name: Name of the category
            analyses: List of analyses for this category

        Returns:
            CategoryAnalysis with metrics, samples, narrative, and insights
        """
        if not analyses:
            return {
                "category": category_name,
                "metrics": {
                    "mean_agency": 0,
                    "mean_perceived_justice": 0,
                    "mean_belonging": 0,
                    "dominant_mode": "NEUTRAL",
                    "narrative_count": 0,
                },
                "sample_projections": [],
                "analysis_narrative": "No narratives available for analysis.",
                "critical_insights": [],
            }

        coords = calculate_mean_coordinates(analyses)
        dominant_mode = get_dominant_mode(analyses)

        # Get sample projections (up to 3)
        sample_projections = []
        for analysis in analyses[:3]:
            text = analysis.get("text", "")[:100]
            mode_info = analysis.get("mode", {})
            mode = mode_info.get("primary_mode", "NEUTRAL") if isinstance(mode_info, dict) else str(mode_info)
            analysis_coords = analysis.get("coordinates", {})

            sample_projections.append({
                "text_preview": text + "..." if len(analysis.get("text", "")) > 100 else text,
                "mode": mode,
                "coordinates": analysis_coords,
            })

        display_name = CATEGORY_DISPLAY_NAMES.get(category_name, category_name.replace("_", " ").title())

        system_prompt = """You are an expert analyst for the Cultural Soliton Observatory.
Generate insightful category-level analysis of narrative patterns.

Focus on:
1. What the coordinate values reveal about messaging strategy
2. Why certain modes dominate in this category
3. What works well and what might be improved
4. Specific patterns unique to this type of content

Be specific and reference the actual data provided."""

        user_prompt = f"""Analyze the "{display_name}" category narratives.

Metrics:
- Narrative count: {len(analyses)}
- Mean Agency: {coords['agency']}
- Mean Perceived Justice: {coords['perceived_justice']}
- Mean Belonging: {coords['belonging']}
- Dominant Mode: {dominant_mode}

Sample texts analyzed:
{chr(10).join(f'- "{s["text_preview"]}" -> {s["mode"]}' for s in sample_projections)}

Generate:
1. A 2-3 paragraph analysis narrative explaining what these patterns mean
2. 2-4 critical insights (bullet points) specific to this category

Format your response as:
ANALYSIS:
[Your narrative analysis here]

INSIGHTS:
- [Insight 1]
- [Insight 2]
- [etc]"""

        response = self._call_claude(system_prompt, user_prompt)

        # Parse response
        analysis_narrative = ""
        critical_insights = []

        if "ANALYSIS:" in response and "INSIGHTS:" in response:
            parts = response.split("INSIGHTS:")
            analysis_narrative = parts[0].replace("ANALYSIS:", "").strip()
            insights_text = parts[1].strip()
            critical_insights = [
                line.strip().lstrip("- ").lstrip("* ")
                for line in insights_text.split("\n")
                if line.strip() and line.strip().startswith(("-", "*", "1", "2", "3", "4"))
            ]
        else:
            analysis_narrative = response

        return {
            "category": display_name,
            "metrics": {
                "mean_agency": coords["agency"],
                "mean_perceived_justice": coords["perceived_justice"],
                "mean_belonging": coords["belonging"],
                "dominant_mode": dominant_mode,
                "narrative_count": len(analyses),
            },
            "sample_projections": sample_projections,
            "analysis_narrative": analysis_narrative,
            "critical_insights": critical_insights[:4],  # Limit to 4
        }

    def generate_strategic_insights(
        self,
        all_analyses: List[Analysis],
        category_comparisons: List[Dict[str, Any]]
    ) -> KeyFindings:
        """
        Generate key strategic findings including strengths and concerns.

        Args:
            all_analyses: All analyses across categories
            category_comparisons: Cross-category comparison data

        Returns:
            KeyFindings with strengths, concerns, and benchmark comparisons
        """
        coords = calculate_mean_coordinates(all_analyses)
        mode_dist = calculate_mode_distribution(all_analyses)
        grouped = group_analyses_by_category(all_analyses)

        # Format category summary for prompt
        category_summary = []
        for cat, cat_analyses in grouped.items():
            cat_coords = calculate_mean_coordinates(cat_analyses)
            cat_mode = get_dominant_mode(cat_analyses)
            category_summary.append(
                f"- {cat}: Agency={cat_coords['agency']}, "
                f"PJ={cat_coords['perceived_justice']}, "
                f"B={cat_coords['belonging']}, Mode={cat_mode}"
            )

        system_prompt = """You are an expert analyst synthesizing Cultural Soliton Observatory findings.
Identify key strengths, concerns, and benchmark comparisons.

Strengths should highlight effective narrative patterns.
Concerns should identify risks or areas needing attention.
Be specific and actionable."""

        user_prompt = f"""Analyze narrative patterns to identify key findings.

Overall Profile:
- Total narratives: {len(all_analyses)}
- Mean Agency: {coords['agency']}
- Mean Perceived Justice: {coords['perceived_justice']}
- Mean Belonging: {coords['belonging']}

Mode Distribution:
{chr(10).join(f"- {mode}: {pct*100:.1f}%" for mode, pct in mode_dist.items())}

By Category:
{chr(10).join(category_summary)}

Generate findings in this exact format:
STRENGTHS:
1. [Specific strength with data reference]
2. [Another strength]
3. [Another strength]

CONCERNS:
1. [Specific concern with data reference]
2. [Another concern]

BENCHMARKS:
- [How this compares to typical nonprofit/corporate communications]
- [Notable patterns relative to research benchmarks]"""

        response = self._call_claude(system_prompt, user_prompt)

        # Parse response
        strengths = []
        concerns = []
        benchmarks = []

        current_section = None
        for line in response.split("\n"):
            line = line.strip()
            if "STRENGTHS:" in line:
                current_section = "strengths"
            elif "CONCERNS:" in line:
                current_section = "concerns"
            elif "BENCHMARKS:" in line:
                current_section = "benchmarks"
            elif line and current_section:
                # Clean up numbered or bulleted items
                clean_line = line.lstrip("0123456789.-) ").strip()
                if clean_line:
                    if current_section == "strengths":
                        strengths.append(clean_line)
                    elif current_section == "concerns":
                        concerns.append(clean_line)
                    elif current_section == "benchmarks":
                        benchmarks.append({"comparison": clean_line})

        return {
            "strengths": strengths[:5],
            "concerns": concerns[:4],
            "benchmarks": benchmarks[:3],
        }

    def generate_recommendations(
        self,
        analyses: List[Analysis],
        patterns: Dict[str, Any],
        insights: KeyFindings
    ) -> List[str]:
        """
        Generate actionable recommendations based on analysis.

        Args:
            analyses: All narrative analyses
            patterns: Identified patterns from analysis
            insights: Key findings (strengths, concerns)

        Returns:
            List of actionable recommendation strings
        """
        coords = calculate_mean_coordinates(analyses)
        mode_dist = calculate_mode_distribution(analyses)

        system_prompt = """You are a strategic communications advisor using Cultural Soliton Observatory data.
Generate specific, actionable recommendations for improving narrative effectiveness.

Recommendations should:
1. Address specific weaknesses identified in the data
2. Build on existing strengths
3. Be concrete and implementable
4. Reference specific modes, coordinates, or patterns"""

        user_prompt = f"""Generate recommendations based on this analysis.

Current Profile:
- Agency: {coords['agency']}
- Perceived Justice: {coords['perceived_justice']}
- Belonging: {coords['belonging']}
- Top modes: {', '.join(list(mode_dist.keys())[:3])}

Strengths identified:
{chr(10).join(f"- {s}" for s in insights['strengths'][:3])}

Concerns identified:
{chr(10).join(f"- {c}" for c in insights['concerns'][:3])}

Generate 4-6 specific recommendations. Format each as:
MAINTAIN: [What to keep doing]
ENHANCE: [What to improve]
ADD: [What new elements to incorporate]
MONITOR: [What to track going forward]"""

        response = self._call_claude(system_prompt, user_prompt)

        # Parse recommendations
        recommendations = []
        for line in response.split("\n"):
            line = line.strip()
            if line and any(line.startswith(prefix) for prefix in ["MAINTAIN:", "ENHANCE:", "ADD:", "MONITOR:", "-", "*", "1", "2", "3", "4", "5", "6"]):
                # Clean up the line
                for prefix in ["MAINTAIN:", "ENHANCE:", "ADD:", "MONITOR:"]:
                    if line.startswith(prefix):
                        line = f"**{prefix[:-1]}**: {line[len(prefix):].strip()}"
                        break
                else:
                    line = line.lstrip("0123456789.-*) ").strip()

                if line:
                    recommendations.append(line)

        return recommendations[:8]  # Limit to 8 recommendations

    async def synthesize_full_report(
        self,
        site_data: Dict[str, Any]
    ) -> FullReport:
        """
        Orchestrate full report generation from site analysis data.

        This is the main entry point that coordinates all synthesis functions
        to produce a comprehensive report.

        Args:
            site_data: Complete site analysis data containing:
                - org_name: Organization name
                - analyses: List of narrative analyses
                - pages: Page-level analysis results (optional)
                - Additional metadata

        Returns:
            FullReport dictionary with all sections synthesized
        """
        org_name = site_data.get("org_name", site_data.get("site_url", "Organization"))

        # Extract analyses from various possible structures
        analyses: List[Analysis] = []

        # Check for direct analyses list
        if "analyses" in site_data:
            analyses = site_data["analyses"]

        # Check for page-level analyses
        elif "pages" in site_data:
            for page in site_data["pages"]:
                analysis: Analysis = {
                    "text": page.get("sample_text", ""),
                    "url": page.get("url", ""),
                    "category": self._infer_category(page.get("url", ""), page.get("title", "")),
                    "mode": {
                        "primary_mode": page.get("mode", "NEUTRAL"),
                        "confidence": page.get("confidence", 0.5),
                    },
                    "coordinates": page.get("coordinates", {}),
                    "force_field": page.get("force_field", {}),
                }
                analyses.append(analysis)

        if not analyses:
            logger.warning("No analyses found in site_data")
            return self._empty_report(org_name)

        logger.info(f"Synthesizing report for {org_name} with {len(analyses)} analyses")

        # Calculate overall metrics
        coords = calculate_mean_coordinates(analyses)
        mode_dist = calculate_mode_distribution(analyses)
        dominant_mode = get_dominant_mode(analyses)
        health_score = calculate_health_score(analyses, mode_dist)

        # Group by category
        grouped = group_analyses_by_category(analyses)

        # Generate executive summary
        logger.info("Generating executive summary...")
        executive_summary = self.generate_executive_summary(analyses, org_name)

        # Generate category analyses
        logger.info("Generating category analyses...")
        category_analyses = []
        for category, cat_analyses in grouped.items():
            cat_analysis = self.generate_category_analysis(category, cat_analyses)
            category_analyses.append(cat_analysis)

        # Generate cross-category comparisons
        cross_category_comparisons = self._generate_cross_category_comparisons(grouped)

        # Generate strategic insights
        logger.info("Generating strategic insights...")
        key_findings = self.generate_strategic_insights(analyses, cross_category_comparisons)

        # Generate recommendations
        logger.info("Generating recommendations...")
        patterns = {
            "mode_distribution": mode_dist,
            "coordinate_averages": coords,
            "category_count": len(grouped),
        }
        recommendations = self.generate_recommendations(analyses, patterns, key_findings)

        # Generate conclusion
        logger.info("Generating conclusion...")
        conclusion = self._generate_conclusion(org_name, health_score, key_findings)

        return {
            "executive_summary": executive_summary,
            "organization_background": {
                "name": org_name,
                "narratives_analyzed": len(analyses),
                "categories_covered": len(grouped),
            },
            "overall_profile": {
                "total_narratives": len(analyses),
                "mean_agency": coords["agency"],
                "mean_perceived_justice": coords["perceived_justice"],
                "mean_belonging": coords["belonging"],
                "dominant_mode": dominant_mode,
                "mode_distribution": mode_dist,
            },
            "category_analyses": category_analyses,
            "cross_category_comparisons": cross_category_comparisons,
            "key_findings": key_findings,
            "recommendations": recommendations,
            "conclusion": conclusion,
            "health_score": health_score,
        }

    def _infer_category(self, url: str, title: str) -> str:
        """Infer category from URL path and title."""
        url_lower = url.lower()
        title_lower = title.lower()

        category_patterns = {
            "mission": ["mission", "vision", "purpose", "values"],
            "about": ["about", "who-we-are", "our-story"],
            "impact_stories": ["impact", "stories", "success", "testimonial"],
            "donor_appeals": ["donate", "give", "support", "contribute"],
            "team": ["team", "staff", "leadership", "people"],
            "services": ["services", "programs", "what-we-do"],
            "blog": ["blog", "news", "updates", "articles"],
            "advocacy": ["advocacy", "action", "campaign"],
            "homepage": ["/$", "index"],
        }

        for category, patterns in category_patterns.items():
            for pattern in patterns:
                if pattern in url_lower or pattern in title_lower:
                    return category

        return "general"

    def _generate_cross_category_comparisons(
        self,
        grouped: Dict[str, List[Analysis]]
    ) -> List[Dict[str, Any]]:
        """Generate cross-category comparison data."""
        comparisons = []
        categories = list(grouped.keys())

        # Compare each pair of categories
        comparison_pairs = [
            ("problem_framing", "solution_framing"),
            ("mission", "impact_stories"),
            ("donor_appeals", "organizational_voice"),
        ]

        for cat1, cat2 in comparison_pairs:
            if cat1 in grouped and cat2 in grouped:
                coords1 = calculate_mean_coordinates(grouped[cat1])
                coords2 = calculate_mean_coordinates(grouped[cat2])

                comparisons.append({
                    "categories": [cat1, cat2],
                    "agency_gap": round(coords2["agency"] - coords1["agency"], 3),
                    "justice_gap": round(coords2["perceived_justice"] - coords1["perceived_justice"], 3),
                    "belonging_gap": round(coords2["belonging"] - coords1["belonging"], 3),
                    "interpretation": self._interpret_gap(coords1, coords2, cat1, cat2),
                })

        return comparisons

    def _interpret_gap(
        self,
        coords1: Coordinates,
        coords2: Coordinates,
        cat1: str,
        cat2: str
    ) -> str:
        """Generate interpretation of coordinate gap between categories."""
        justice_gap = coords2["perceived_justice"] - coords1["perceived_justice"]
        belonging_gap = coords2["belonging"] - coords1["belonging"]

        display1 = CATEGORY_DISPLAY_NAMES.get(cat1, cat1)
        display2 = CATEGORY_DISPLAY_NAMES.get(cat2, cat2)

        if abs(justice_gap) > 0.3:
            direction = "increases" if justice_gap > 0 else "decreases"
            return f"Perceived Justice {direction} significantly from {display1} to {display2}"
        elif abs(belonging_gap) > 0.3:
            direction = "increases" if belonging_gap > 0 else "decreases"
            return f"Belonging {direction} significantly from {display1} to {display2}"
        else:
            return f"Relatively consistent positioning between {display1} and {display2}"

    def _generate_conclusion(
        self,
        org_name: str,
        health_score: int,
        key_findings: KeyFindings
    ) -> str:
        """Generate concluding paragraph for the report."""
        system_prompt = """You are an expert analyst writing a concluding summary for a Cultural Soliton Observatory report.
Write a concise, impactful conclusion that summarizes the organization's narrative effectiveness."""

        user_prompt = f"""Write a 1-2 paragraph conclusion for {org_name}'s narrative analysis.

Health Score: {health_score}/100
Top Strengths: {', '.join(key_findings['strengths'][:2])}
Key Concerns: {', '.join(key_findings['concerns'][:2])}

The conclusion should:
1. Summarize overall narrative health
2. Highlight what sets this organization apart
3. End with forward-looking insight"""

        return self._call_claude(system_prompt, user_prompt, max_tokens=500)

    def _empty_report(self, org_name: str) -> FullReport:
        """Return an empty report structure when no data is available."""
        return {
            "executive_summary": f"No narrative data available for {org_name}.",
            "organization_background": {"name": org_name, "narratives_analyzed": 0, "categories_covered": 0},
            "overall_profile": {
                "total_narratives": 0,
                "mean_agency": 0,
                "mean_perceived_justice": 0,
                "mean_belonging": 0,
                "dominant_mode": "NEUTRAL",
                "mode_distribution": {},
            },
            "category_analyses": [],
            "cross_category_comparisons": [],
            "key_findings": {"strengths": [], "concerns": [], "benchmarks": []},
            "recommendations": [],
            "conclusion": "Insufficient data for analysis.",
            "health_score": 0,
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_synthesis_service(api_key: Optional[str] = None) -> SynthesisService:
    """
    Factory function to create a SynthesisService instance.

    Args:
        api_key: Optional API key (defaults to ANTHROPIC_API_KEY env var)

    Returns:
        Configured SynthesisService instance
    """
    return SynthesisService(api_key=api_key)


async def synthesize_report(
    site_data: Dict[str, Any],
    api_key: Optional[str] = None
) -> FullReport:
    """
    Convenience function to synthesize a full report.

    Args:
        site_data: Site analysis data
        api_key: Optional API key

    Returns:
        Complete synthesized report
    """
    service = create_synthesis_service(api_key)
    return await service.synthesize_full_report(site_data)
