"""
Comprehensive Narrative Analysis Report API
=============================================

This module generates comprehensive narrative analysis reports similar to
detailed case studies. It provides deep analysis of website content across
multiple page categories with synthesized narratives and recommendations.

Endpoint:
- POST /api/report/generate: Generate a comprehensive narrative analysis report

Features:
- Site-wide crawling and analysis (leverages api_site_analysis)
- Page categorization (homepage, about, pricing, careers, etc.)
- Per-category metric aggregation
- Cross-category comparison analysis
- Health scoring with detailed breakdowns
- Executive summary generation
- Actionable recommendations

Usage:
    from api_comprehensive_report import router as report_router
    app.include_router(report_router, prefix="/api/report", tags=["report"])
"""

import re
import asyncio
import logging
import uuid
import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from urllib.parse import urlparse

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import narrative auditor for detailed per-narrative feedback
from narrative_auditor import audit_narratives_batch, NarrativeAudit

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# PAGE CATEGORY DEFINITIONS
# ============================================================================

PAGE_CATEGORIES = {
    "homepage": {
        "patterns": [r"^/$", r"^/index", r"^/home$"],
        "label": "Homepage",
        "description": "Main landing page - first impression",
        "weight": 1.5  # Higher importance
    },
    "about": {
        "patterns": [r"/about", r"/our-story", r"/who-we-are", r"/company$", r"/mission"],
        "label": "About/Company",
        "description": "Company story and values",
        "weight": 1.3
    },
    "pricing": {
        "patterns": [r"/pricing", r"/plans", r"/packages", r"/subscription", r"/buy"],
        "label": "Pricing",
        "description": "Pricing and purchase pages - critical for conversion",
        "weight": 1.4
    },
    "product": {
        "patterns": [r"/products?(?:/|$)", r"/solutions?", r"/features?", r"/platform"],
        "label": "Product/Features",
        "description": "Product and feature descriptions",
        "weight": 1.2
    },
    "careers": {
        "patterns": [r"/careers?", r"/jobs?", r"/hiring", r"/join", r"/work-with-us"],
        "label": "Careers",
        "description": "Employment and culture pages",
        "weight": 1.1
    },
    "support": {
        "patterns": [r"/support", r"/help", r"/faq", r"/docs", r"/documentation", r"/contact"],
        "label": "Support/Help",
        "description": "Customer support and documentation",
        "weight": 1.0
    },
    "legal": {
        "patterns": [r"/terms", r"/privacy", r"/legal", r"/policy", r"/compliance"],
        "label": "Legal/Policy",
        "description": "Legal and policy pages",
        "weight": 0.8
    },
    "blog": {
        "patterns": [r"/blog", r"/news", r"/articles?", r"/insights?", r"/resources?"],
        "label": "Blog/Content",
        "description": "Content marketing and thought leadership",
        "weight": 0.9
    },
    "other": {
        "patterns": [],
        "label": "Other Pages",
        "description": "Miscellaneous pages",
        "weight": 0.7
    }
}


# ============================================================================
# MODE QUALITY AND HEALTH METRICS
# ============================================================================

MODE_HEALTH_SCORES = {
    # Positive modes - high health impact
    "HEROIC": 0.95,
    "COMMUNAL": 0.90,
    "TRANSCENDENT": 0.80,

    # Ambivalent modes - neutral impact
    "TRANSITIONAL": 0.55,
    "CONFLICTED": 0.45,
    "NEUTRAL": 0.50,

    # Shadow modes - negative impact
    "CYNICAL_ACHIEVER": 0.35,
    "VICTIM": 0.20,
    "PARANOID": 0.15,

    # Exit modes - concerning
    "SPIRITUAL_EXIT": 0.40,
    "SOCIAL_EXIT": 0.35,
    "PROTEST_EXIT": 0.30,

    # Default
    "UNKNOWN": 0.50
}

CATEGORY_HEALTH_WEIGHTS = {
    "POSITIVE": 1.0,
    "AMBIVALENT": 0.5,
    "SHADOW": 0.2,
    "EXIT": 0.3,
    "NOISE": 0.0
}


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ReportRequest(BaseModel):
    """Request for generating a comprehensive report."""
    url: str = Field(
        ...,
        description="Site URL to analyze",
        examples=["https://example.com"]
    )
    organization_name: Optional[str] = Field(
        None,
        description="Organization name (auto-detected if not provided)"
    )
    max_pages: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum number of pages to crawl"
    )
    include_synthesis: bool = Field(
        default=True,
        description="Use Claude for narrative synthesis (more insightful)"
    )
    include_raw_data: bool = Field(
        default=False,
        description="Include raw analysis data in response"
    )


class CoordinateStats(BaseModel):
    """Statistical summary for a coordinate dimension."""
    mean: float
    std: float
    min: float
    max: float
    median: float


class OverallProfile(BaseModel):
    """Overall narrative profile for the organization."""
    primary_mode: str
    primary_mode_percentage: float
    secondary_mode: Optional[str] = None
    secondary_mode_percentage: Optional[float] = None
    mode_category: str
    agency: CoordinateStats
    perceived_justice: CoordinateStats
    belonging: CoordinateStats
    narrative_coherence: float = Field(
        ...,
        description="How consistent the narrative is across pages (0-1)"
    )
    narrative_summary: str


class CategoryAnalysis(BaseModel):
    """Analysis results for a specific page category."""
    category_id: str
    category_label: str
    pages_count: int
    mean_agency: float
    mean_perceived_justice: float
    mean_belonging: float
    std_agency: float
    std_perceived_justice: float
    std_belonging: float
    dominant_mode: str
    mode_distribution: Dict[str, float]
    sample_pages: List[Dict[str, Any]]
    issues: List[str]
    strengths: List[str]
    narrative_summary: str


class CategoryComparison(BaseModel):
    """Comparison between two page categories."""
    category_a: str
    category_b: str
    agency_gap: float
    justice_gap: float
    belonging_gap: float
    euclidean_distance: float
    interpretation: str
    severity: str = Field(
        default="info",
        description="Severity: 'info', 'warning', 'critical'"
    )


class KeyFinding(BaseModel):
    """A key finding from the analysis."""
    title: str
    description: str
    severity: str
    affected_categories: List[str]
    metric_values: Optional[Dict[str, float]] = None


class KeyFindings(BaseModel):
    """Collection of key findings."""
    strengths: List[KeyFinding]
    opportunities: List[KeyFinding]
    risks: List[KeyFinding]


class Recommendation(BaseModel):
    """An actionable recommendation."""
    priority: int = Field(..., ge=1, le=5, description="1=highest priority")
    title: str
    description: str
    category: str  # narrative, content, consistency, voice
    affected_pages: List[str]
    expected_impact: str


class NarrativeAuditResult(BaseModel):
    """Detailed audit of a single narrative."""
    text: str
    source_url: str
    category: str
    mode: str
    coordinates: Dict[str, float]
    effectiveness_score: float
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    counter_framing: Optional[str] = None
    best_practice_example: Optional[str] = None


class NarrativeAuditSummary(BaseModel):
    """Summary of narrative audits."""
    average_effectiveness: float
    total_narratives: int
    high_performing: int
    needs_improvement: int
    top_issues: List[Dict[str, Any]]
    best_narratives: List[NarrativeAuditResult] = Field(
        default_factory=list,
        description="Top performing narratives with explanations"
    )
    needs_work: List[NarrativeAuditResult] = Field(
        default_factory=list,
        description="Narratives that need improvement with suggestions"
    )


class ReportResponse(BaseModel):
    """Complete comprehensive report response."""
    report_id: str
    organization_name: str
    site_url: str
    analysis_date: str
    narratives_analyzed: int
    categories_count: int
    crawl_time_seconds: float

    executive_summary: str
    overall_profile: OverallProfile
    mode_distribution: Dict[str, float]
    category_analyses: List[CategoryAnalysis]
    cross_category_comparisons: List[CategoryComparison]
    key_findings: KeyFindings
    recommendations: List[Recommendation]
    conclusion: str
    health_score: int = Field(..., ge=0, le=100)

    # NEW: Detailed narrative audit results
    narrative_audit: Optional[NarrativeAuditSummary] = Field(
        None,
        description="Detailed per-narrative audit with strengths, weaknesses, and suggestions"
    )

    raw_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Full analysis data if requested"
    )


# ============================================================================
# PAGE CATEGORIZATION
# ============================================================================

def categorize_page(url: str) -> str:
    """
    Categorize a page URL into one of the defined categories.

    Args:
        url: The page URL to categorize

    Returns:
        Category ID (e.g., 'homepage', 'pricing', 'about')
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.lower()

        # Check each category's patterns
        for category_id, category_info in PAGE_CATEGORIES.items():
            if category_id == "other":
                continue  # Skip default category

            for pattern in category_info["patterns"]:
                if re.search(pattern, path):
                    return category_id

        return "other"
    except Exception:
        return "other"


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_coordinate_stats(values: List[float]) -> CoordinateStats:
    """Calculate statistical summary for a coordinate dimension."""
    if not values:
        return CoordinateStats(
            mean=0.0, std=0.0, min=0.0, max=0.0, median=0.0
        )

    arr = np.array(values)
    return CoordinateStats(
        mean=round(float(np.mean(arr)), 4),
        std=round(float(np.std(arr)), 4),
        min=round(float(np.min(arr)), 4),
        max=round(float(np.max(arr)), 4),
        median=round(float(np.median(arr)), 4)
    )


def calculate_category_metrics(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics for a group of page analyses.

    Args:
        analyses: List of page analysis results

    Returns:
        Dictionary with aggregated metrics
    """
    if not analyses:
        return {
            "mean_agency": 0.0,
            "mean_perceived_justice": 0.0,
            "mean_belonging": 0.0,
            "std_agency": 0.0,
            "std_perceived_justice": 0.0,
            "std_belonging": 0.0,
            "dominant_mode": "NEUTRAL",
            "mode_distribution": {},
            "sample_count": 0
        }

    # Extract coordinate values
    agency_values = []
    justice_values = []
    belonging_values = []
    mode_counts = defaultdict(int)

    for analysis in analyses:
        coords = analysis.get("coordinates", {})
        agency_values.append(coords.get("agency", 0))
        justice_values.append(coords.get("perceived_justice", 0))
        belonging_values.append(coords.get("belonging", 0))

        mode = analysis.get("mode", "NEUTRAL")
        mode_counts[mode] += 1

    # Calculate statistics
    total = len(analyses)
    mode_distribution = {
        mode: round(count / total, 4)
        for mode, count in mode_counts.items()
    }

    dominant_mode = max(mode_counts, key=mode_counts.get) if mode_counts else "NEUTRAL"

    return {
        "mean_agency": round(float(np.mean(agency_values)), 4),
        "mean_perceived_justice": round(float(np.mean(justice_values)), 4),
        "mean_belonging": round(float(np.mean(belonging_values)), 4),
        "std_agency": round(float(np.std(agency_values)), 4),
        "std_perceived_justice": round(float(np.std(justice_values)), 4),
        "std_belonging": round(float(np.std(belonging_values)), 4),
        "dominant_mode": dominant_mode,
        "mode_distribution": mode_distribution,
        "sample_count": total
    }


def calculate_euclidean_distance(
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any]
) -> float:
    """Calculate Euclidean distance between two category profiles."""
    point_a = np.array([
        metrics_a.get("mean_agency", 0),
        metrics_a.get("mean_perceived_justice", 0),
        metrics_a.get("mean_belonging", 0)
    ])
    point_b = np.array([
        metrics_b.get("mean_agency", 0),
        metrics_b.get("mean_perceived_justice", 0),
        metrics_b.get("mean_belonging", 0)
    ])

    return round(float(np.linalg.norm(point_a - point_b)), 4)


def compare_categories(
    cat_a: Dict[str, Any],
    cat_b: Dict[str, Any]
) -> CategoryComparison:
    """
    Compare two page categories and generate insights.

    Args:
        cat_a: First category's analysis (includes metrics and metadata)
        cat_b: Second category's analysis

    Returns:
        CategoryComparison with gap analysis
    """
    metrics_a = cat_a.get("metrics", {})
    metrics_b = cat_b.get("metrics", {})

    agency_gap = round(
        metrics_b.get("mean_agency", 0) - metrics_a.get("mean_agency", 0),
        4
    )
    justice_gap = round(
        metrics_b.get("mean_perceived_justice", 0) -
        metrics_a.get("mean_perceived_justice", 0),
        4
    )
    belonging_gap = round(
        metrics_b.get("mean_belonging", 0) - metrics_a.get("mean_belonging", 0),
        4
    )

    distance = calculate_euclidean_distance(metrics_a, metrics_b)

    # Generate interpretation
    interpretation = generate_comparison_interpretation(
        cat_a["label"], cat_b["label"],
        agency_gap, justice_gap, belonging_gap, distance
    )

    # Determine severity
    severity = "info"
    if distance > 1.0:
        severity = "critical"
    elif distance > 0.5:
        severity = "warning"

    return CategoryComparison(
        category_a=cat_a["label"],
        category_b=cat_b["label"],
        agency_gap=agency_gap,
        justice_gap=justice_gap,
        belonging_gap=belonging_gap,
        euclidean_distance=distance,
        interpretation=interpretation,
        severity=severity
    )


def generate_comparison_interpretation(
    label_a: str, label_b: str,
    agency_gap: float, justice_gap: float, belonging_gap: float,
    distance: float
) -> str:
    """Generate a human-readable interpretation of category comparison."""

    if distance < 0.2:
        return f"{label_a} and {label_b} have highly consistent narrative positioning."

    insights = []

    if abs(agency_gap) > 0.3:
        if agency_gap > 0:
            insights.append(f"{label_b} projects stronger agency (+{agency_gap:.2f})")
        else:
            insights.append(f"{label_a} projects stronger agency ({agency_gap:.2f})")

    if abs(justice_gap) > 0.3:
        if justice_gap > 0:
            insights.append(f"{label_b} conveys higher fairness perception (+{justice_gap:.2f})")
        else:
            insights.append(f"{label_a} conveys higher fairness perception ({justice_gap:.2f})")

    if abs(belonging_gap) > 0.3:
        if belonging_gap > 0:
            insights.append(f"{label_b} creates stronger belonging (+{belonging_gap:.2f})")
        else:
            insights.append(f"{label_a} creates stronger belonging ({belonging_gap:.2f})")

    if not insights:
        return f"Minor variations between {label_a} and {label_b} (distance: {distance:.2f})."

    return " | ".join(insights) + f" (distance: {distance:.2f})"


# ============================================================================
# NARRATIVE SYNTHESIS
# ============================================================================

def generate_category_narrative(
    category_label: str,
    metrics: Dict[str, Any],
    sample_pages: List[Dict[str, Any]]
) -> str:
    """Generate a narrative summary for a category."""
    mode = metrics.get("dominant_mode", "NEUTRAL")
    agency = metrics.get("mean_agency", 0)
    justice = metrics.get("mean_perceived_justice", 0)
    belonging = metrics.get("mean_belonging", 0)
    count = metrics.get("sample_count", 0)

    # Build narrative based on dimensions
    narrative_parts = []

    # Mode description
    if mode in ["HEROIC", "COMMUNAL", "TRANSCENDENT"]:
        narrative_parts.append(f"The {category_label} content presents a predominantly positive narrative stance")
    elif mode in ["VICTIM", "CYNICAL_ACHIEVER", "PARANOID"]:
        narrative_parts.append(f"The {category_label} content shows concerning shadow-mode patterns")
    elif mode in ["TRANSITIONAL", "CONFLICTED"]:
        narrative_parts.append(f"The {category_label} content displays mixed or transitional messaging")
    else:
        narrative_parts.append(f"The {category_label} content maintains a neutral tone")

    # Agency assessment
    if agency > 0.5:
        narrative_parts.append("projecting strong reader empowerment and action potential")
    elif agency < -0.5:
        narrative_parts.append("which may leave readers feeling limited in their options")

    # Belonging assessment
    if belonging > 0.5:
        narrative_parts.append("while fostering a sense of community and inclusion")
    elif belonging < -0.5:
        narrative_parts.append("though the tone may feel somewhat exclusive or impersonal")

    # Justice assessment
    if justice < -0.5:
        narrative_parts.append("with language that may raise fairness concerns")

    return ", ".join(narrative_parts) + f" (analyzed {count} pages)."


def generate_category_issues(
    metrics: Dict[str, Any],
    pages: List[Dict[str, Any]]
) -> List[str]:
    """Identify issues in a category's content."""
    issues = []

    mode = metrics.get("dominant_mode", "NEUTRAL")
    agency = metrics.get("mean_agency", 0)
    justice = metrics.get("mean_perceived_justice", 0)
    belonging = metrics.get("mean_belonging", 0)
    mode_dist = metrics.get("mode_distribution", {})

    # Mode-based issues
    if mode in ["VICTIM", "CYNICAL_ACHIEVER", "PARANOID"]:
        issues.append(f"Shadow mode ({mode}) dominates - may create negative reader experience")

    # Dimension issues
    if agency < -0.3:
        issues.append("Low agency language may leave readers feeling powerless")

    if justice < -0.3:
        issues.append("Perceived justice issues detected - content may feel unfair to readers")

    if belonging < -0.3:
        issues.append("Weak belonging signals - content may feel exclusionary")

    # Consistency issues
    if len(mode_dist) > 3:
        issues.append("High mode fragmentation - inconsistent voice across pages")

    # Check individual page issues
    page_issues = 0
    for page in pages:
        if page.get("issues"):
            page_issues += len(page["issues"])

    if page_issues > len(pages):
        issues.append(f"Multiple page-level issues detected ({page_issues} total)")

    return issues


def generate_category_strengths(
    metrics: Dict[str, Any],
    pages: List[Dict[str, Any]]
) -> List[str]:
    """Identify strengths in a category's content."""
    strengths = []

    mode = metrics.get("dominant_mode", "NEUTRAL")
    agency = metrics.get("mean_agency", 0)
    justice = metrics.get("mean_perceived_justice", 0)
    belonging = metrics.get("mean_belonging", 0)
    std_agency = metrics.get("std_agency", 1)
    std_justice = metrics.get("std_perceived_justice", 1)
    std_belonging = metrics.get("std_belonging", 1)

    # Mode-based strengths
    if mode in ["HEROIC", "COMMUNAL", "TRANSCENDENT"]:
        strengths.append(f"Strong positive mode ({mode}) creates engaging narrative")

    # Dimension strengths
    if agency > 0.3:
        strengths.append("Empowering language gives readers sense of agency")

    if justice > 0.3:
        strengths.append("Strong fairness perception builds trust")

    if belonging > 0.3:
        strengths.append("Community-building language fosters connection")

    # Consistency strengths
    avg_std = (std_agency + std_justice + std_belonging) / 3
    if avg_std < 0.3:
        strengths.append("Highly consistent messaging across pages")

    return strengths


def generate_overall_narrative(
    overall_metrics: Dict[str, Any],
    category_analyses: List[Dict[str, Any]],
    organization_name: str
) -> str:
    """Generate the overall narrative profile summary."""
    mode = overall_metrics.get("dominant_mode", "NEUTRAL")
    agency = overall_metrics.get("mean_agency", 0)
    justice = overall_metrics.get("mean_perceived_justice", 0)
    belonging = overall_metrics.get("mean_belonging", 0)

    # Determine overall stance
    if mode in ["HEROIC", "COMMUNAL"]:
        stance = "empowering and positive"
    elif mode == "TRANSCENDENT":
        stance = "aspirational and values-driven"
    elif mode in ["VICTIM", "CYNICAL_ACHIEVER"]:
        stance = "concerning, with shadow-mode characteristics"
    elif mode == "TRANSITIONAL":
        stance = "in flux, showing mixed messaging"
    else:
        stance = "neutral and professionally measured"

    summary = f"{organization_name}'s website presents an overall narrative that is {stance}. "

    # Add dimension commentary
    dimension_notes = []
    if agency > 0.3:
        dimension_notes.append("strong agency messaging empowers visitors")
    elif agency < -0.3:
        dimension_notes.append("agency language could be strengthened")

    if belonging > 0.3:
        dimension_notes.append("community belonging is emphasized")
    elif belonging < -0.3:
        dimension_notes.append("belonging cues are limited")

    if justice > 0.3:
        dimension_notes.append("fairness perception is positive")
    elif justice < -0.3:
        dimension_notes.append("perceived fairness needs attention")

    if dimension_notes:
        summary += "Key observations: " + ", ".join(dimension_notes) + "."

    return summary


def calculate_narrative_coherence(category_analyses: List[Dict[str, Any]]) -> float:
    """
    Calculate how coherent the narrative is across categories.

    Returns:
        Float between 0-1, where 1 is perfectly coherent
    """
    if len(category_analyses) < 2:
        return 1.0

    # Calculate pairwise distances
    distances = []
    for i, cat_a in enumerate(category_analyses):
        for cat_b in category_analyses[i+1:]:
            dist = calculate_euclidean_distance(
                cat_a.get("metrics", {}),
                cat_b.get("metrics", {})
            )
            distances.append(dist)

    if not distances:
        return 1.0

    # Convert average distance to coherence score
    avg_distance = np.mean(distances)
    # Distance of 0 = coherence 1.0, distance of 2+ = coherence ~0
    coherence = max(0, 1 - (avg_distance / 2))

    return round(float(coherence), 4)


# ============================================================================
# KEY FINDINGS & RECOMMENDATIONS
# ============================================================================

def extract_key_findings(
    overall_metrics: Dict[str, Any],
    category_analyses: List[Dict[str, Any]],
    comparisons: List[CategoryComparison]
) -> KeyFindings:
    """Extract rich key findings from the analysis."""
    strengths = []
    opportunities = []
    risks = []

    # Analyze overall metrics for findings
    mode = overall_metrics.get("dominant_mode", "NEUTRAL")
    agency = overall_metrics.get("mean_agency", 0)
    justice = overall_metrics.get("mean_perceived_justice", 0)
    belonging = overall_metrics.get("mean_belonging", 0)
    mode_dist = overall_metrics.get("mode_distribution", {})

    # Mode distribution analysis
    positive_modes = ["HEROIC", "COMMUNAL", "TRANSCENDENT", "TRANSITIONAL"]
    positive_pct = sum(mode_dist.get(m, 0) for m in positive_modes) * 100
    if positive_pct >= 30:
        strengths.append(KeyFinding(
            title="Positive Mode Balance",
            description=f"Communications show {positive_pct:.0f}% positive modes ({', '.join(m for m in positive_modes if mode_dist.get(m, 0) > 0.05)}), creating an engaging reader experience.",
            severity="info",
            affected_categories=["all"],
            metric_values={"positive_percentage": positive_pct}
        ))

    # Find best performing category
    best_cat = None
    best_belonging = -999
    for cat in category_analyses:
        metrics = cat.get("metrics", {})
        if metrics.get("mean_belonging", 0) > best_belonging:
            best_belonging = metrics.get("mean_belonging", 0)
            best_cat = cat

    if best_cat and best_belonging > 0.3:
        cat_label = best_cat.get("label", "Unknown")
        strengths.append(KeyFinding(
            title=f"Strong Community Building in {cat_label}",
            description=f"{cat_label} content effectively creates sense of belonging (B={best_belonging:.2f}), fostering community connection.",
            severity="info",
            affected_categories=[cat_label],
            metric_values={"belonging": best_belonging}
        ))

    # Find highest justice category
    best_justice_cat = None
    best_justice = -999
    for cat in category_analyses:
        metrics = cat.get("metrics", {})
        if metrics.get("mean_perceived_justice", 0) > best_justice:
            best_justice = metrics.get("mean_perceived_justice", 0)
            best_justice_cat = cat

    if best_justice_cat and best_justice > 0.3:
        cat_label = best_justice_cat.get("label", "Unknown")
        strengths.append(KeyFinding(
            title=f"Fairness Framing in {cat_label}",
            description=f"{cat_label} effectively frames work as just and fair (PJ={best_justice:.2f}), building trust with readers.",
            severity="info",
            affected_categories=[cat_label],
            metric_values={"perceived_justice": best_justice}
        ))

    # Check for positive mode presence
    if mode_dist.get("TRANSCENDENT", 0) > 0.05:
        strengths.append(KeyFinding(
            title="Aspirational Messaging Present",
            description=f"TRANSCENDENT mode ({mode_dist.get('TRANSCENDENT', 0)*100:.0f}%) provides inspiring, meaning-focused content that resonates emotionally.",
            severity="info",
            affected_categories=["all"],
            metric_values={"transcendent_pct": mode_dist.get("TRANSCENDENT", 0) * 100}
        ))

    if mode_dist.get("COMMUNAL", 0) > 0.02:
        strengths.append(KeyFinding(
            title="Community Voice Present",
            description=f"COMMUNAL mode signals collective action and shared purpose, strengthening community identity.",
            severity="info",
            affected_categories=["all"],
            metric_values={"communal_pct": mode_dist.get("COMMUNAL", 0) * 100}
        ))

    # Overall dimension strengths (lower thresholds)
    if agency > 0.2:
        strengths.append(KeyFinding(
            title="Empowering Agency Language",
            description=f"Content provides readers with sense of capability and action (A={agency:.2f}), supporting engagement.",
            severity="info",
            affected_categories=["all"],
            metric_values={"agency": agency}
        ))

    if belonging > 0.2:
        strengths.append(KeyFinding(
            title="Connection and Belonging",
            description=f"Inclusive language creates community connection (B={belonging:.2f}), fostering loyalty.",
            severity="info",
            affected_categories=["all"],
            metric_values={"belonging": belonging}
        ))

    # Opportunities - look at specific categories
    for cat in category_analyses:
        metrics = cat.get("metrics", {})
        cat_mode = metrics.get("dominant_mode", "NEUTRAL")
        label = cat.get("label", "Unknown")
        cat_agency = metrics.get("mean_agency", 0)
        cat_justice = metrics.get("mean_perceived_justice", 0)
        cat_belonging = metrics.get("mean_belonging", 0)

        # Category-specific opportunities
        mode_dist_cat = metrics.get("mode_distribution", {})
        neutral_pct = mode_dist_cat.get("NEUTRAL", 0)

        if cat_mode == "NEUTRAL" and label in ["Mission Vision", "Impact Stories", "Donor Appeals"]:
            opportunities.append(KeyFinding(
                title=f"Strengthen Emotional Resonance in {label}",
                description=f"{label} content is predominantly NEUTRAL ({neutral_pct:.0%}) - consider strengthening toward more engaging modes like TRANSCENDENT or COMMUNAL.",
                severity="info",
                affected_categories=[label],
                metric_values={"neutral_pct": neutral_pct}
            ))

        if cat_justice < 0.1 and label in ["Problem Framing", "Impact Stories"]:
            opportunities.append(KeyFinding(
                title=f"Enhance Justice Framing in {label}",
                description=f"{label} has low perceived justice (PJ={cat_justice:.2f}). Reframe to emphasize fairness and what's right.",
                severity="info",
                affected_categories=[label],
                metric_values={"perceived_justice": cat_justice}
            ))

    # Cross-category opportunities
    for comp in comparisons[:5]:
        if comp.euclidean_distance > 0.5:
            opportunities.append(KeyFinding(
                title=f"Narrative Bridge: {comp.category_a} → {comp.category_b}",
                description=f"Significant gap ({comp.euclidean_distance:.2f}) between categories - consider creating smoother narrative transition.",
                severity="info",
                affected_categories=[comp.category_a, comp.category_b],
                metric_values={"distance": comp.euclidean_distance}
            ))

    # Risks - category-specific
    for cat in category_analyses:
        metrics = cat.get("metrics", {})
        cat_mode = metrics.get("dominant_mode", "NEUTRAL")
        label = cat.get("label", "Unknown")
        mode_dist_cat = metrics.get("mode_distribution", {})

        if cat_mode in ["VICTIM", "CYNICAL_ACHIEVER", "PARANOID"]:
            shadow_mode_pct = mode_dist_cat.get(cat_mode, 0)
            risks.append(KeyFinding(
                title=f"Shadow Mode in {label}",
                description=f"{label} exhibits {cat_mode} mode ({shadow_mode_pct:.0%}) which may negatively impact reader perception and engagement.",
                severity="critical",
                affected_categories=[label],
                metric_values={"shadow_mode_pct": shadow_mode_pct}
            ))

        # Check for mode fragmentation
        if len(mode_dist_cat) > 3:
            risks.append(KeyFinding(
                title=f"Voice Inconsistency in {label}",
                description=f"{label} shows {len(mode_dist_cat)} different modes - inconsistent voice may confuse readers.",
                severity="warning",
                affected_categories=[label],
                metric_values={"mode_count": len(mode_dist_cat)}
            ))

    # Overall risks
    if mode in ["VICTIM", "CYNICAL_ACHIEVER", "PARANOID"]:
        risks.append(KeyFinding(
            title="Overall Shadow Mode",
            description=f"The site's overall {mode} mode creates trust and engagement issues.",
            severity="critical",
            affected_categories=["all"],
            metric_values={"mode_quality": MODE_HEALTH_SCORES.get(mode, 0.5)}
        ))

    if agency < -0.2:
        risks.append(KeyFinding(
            title="Low Agency Language",
            description=f"Content lacks empowering language (A={agency:.2f}), which may reduce conversion and engagement.",
            severity="warning",
            affected_categories=["all"],
            metric_values={"agency": agency}
        ))

    if justice < -0.2:
        risks.append(KeyFinding(
            title="Perceived Fairness Concerns",
            description=f"Content may trigger fairness concerns (PJ={justice:.2f}), potentially damaging trust.",
            severity="warning",
            affected_categories=["all"],
            metric_values={"perceived_justice": justice}
        ))

    # Limit and return
    return KeyFindings(
        strengths=strengths[:6],
        opportunities=opportunities[:4],
        risks=risks[:4]
    )


def generate_recommendations(
    key_findings: KeyFindings,
    category_analyses: List[Dict[str, Any]],
    overall_metrics: Dict[str, Any]
) -> List[Recommendation]:
    """Generate actionable recommendations based on findings."""
    recommendations = []
    priority = 1

    mode = overall_metrics.get("dominant_mode", "NEUTRAL")
    agency = overall_metrics.get("mean_agency", 0)
    justice = overall_metrics.get("mean_perceived_justice", 0)
    belonging = overall_metrics.get("mean_belonging", 0)

    # Address critical risks first
    for risk in key_findings.risks:
        if risk.severity == "critical":
            if "Shadow Mode" in risk.title:
                recommendations.append(Recommendation(
                    priority=priority,
                    title=f"Address Shadow Mode in {', '.join(risk.affected_categories)}",
                    description="Rewrite content to eliminate victim or cynical language. Focus on empowerment, fairness, and community.",
                    category="narrative",
                    affected_pages=risk.affected_categories,
                    expected_impact="Significant improvement in reader perception and trust"
                ))
                priority += 1

    # Category-specific recommendations
    for cat in category_analyses:
        metrics = cat.get("metrics", {})
        label = cat.get("label", "Unknown")
        cat_mode = metrics.get("dominant_mode", "NEUTRAL")
        mode_dist = metrics.get("mode_distribution", {})
        cat_agency = metrics.get("mean_agency", 0)
        cat_justice = metrics.get("mean_perceived_justice", 0)
        cat_belonging = metrics.get("mean_belonging", 0)

        # Mode fragmentation
        if len(mode_dist) > 3:
            recommendations.append(Recommendation(
                priority=min(priority, 2),
                title=f"Unify Voice in {label}",
                description=f"Content shows {len(mode_dist)} different narrative modes. Audit and rewrite to achieve consistent {cat_mode if cat_mode in ['TRANSCENDENT', 'COMMUNAL', 'HEROIC'] else 'COMMUNAL'} voice throughout.",
                category="consistency",
                affected_pages=[label],
                expected_impact="Improved brand coherence and reader trust"
            ))
            priority += 1

        # Low justice in key categories
        if cat_justice < 0.2 and label in ["Impact Stories", "Problem Framing", "Donor Appeals"]:
            recommendations.append(Recommendation(
                priority=min(priority, 3),
                title=f"Strengthen Justice Framing in {label}",
                description=f"Current PJ={cat_justice:.2f} is low. Reframe content to emphasize fairness: 'every person deserves', 'creating equity', 'restoring dignity'.",
                category="content",
                affected_pages=[label],
                expected_impact="Enhanced emotional resonance and donor motivation"
            ))
            priority += 1

        # Neutral mode in emotional categories
        if cat_mode == "NEUTRAL" and label in ["Mission Vision", "Impact Stories"]:
            recommendations.append(Recommendation(
                priority=min(priority, 3),
                title=f"Elevate Emotional Tone in {label}",
                description=f"{label} content is predominantly NEUTRAL. Transform into TRANSCENDENT or COMMUNAL mode using aspirational language and community framing.",
                category="voice",
                affected_pages=[label],
                expected_impact="Stronger emotional engagement and memorability"
            ))
            priority += 1

    # Overall dimension recommendations
    if agency < 0.3:
        recommendations.append(Recommendation(
            priority=min(priority, 4),
            title="Boost Reader Empowerment",
            description="Transform passive statements to active: 'You can change lives', 'Your gift transforms', 'Together, we will'. Add clear action pathways.",
            category="voice",
            affected_pages=["all"],
            expected_impact="Improved conversion through reader empowerment"
        ))
        priority += 1

    if justice < 0.2:
        recommendations.append(Recommendation(
            priority=min(priority, 4),
            title="Strengthen Fairness Messaging",
            description="Frame work as restoring justice: emphasize what's right, fair treatment, and dignity for all. Use language like 'every child deserves' rather than 'we provide'.",
            category="content",
            affected_pages=["all"],
            expected_impact="Enhanced trust and moral resonance"
        ))
        priority += 1

    if belonging < 0.4:
        recommendations.append(Recommendation(
            priority=min(priority, 4),
            title="Build Community Connection",
            description="Use 'we' and 'together' language. Add testimonials, partner stories, and community acknowledgment. Position donors as partners, not just funders.",
            category="content",
            affected_pages=["homepage", "donor_appeals", "about"],
            expected_impact="Stronger emotional connection and loyalty"
        ))
        priority += 1

    # Opportunity-based recommendations
    for opp in key_findings.opportunities[:3]:
        if "Emotional Resonance" in opp.title:
            cat = opp.affected_categories[0] if opp.affected_categories else "content"
            recommendations.append(Recommendation(
                priority=min(priority, 5),
                title=f"Transform {cat} Narrative",
                description=f"Shift from factual to aspirational. Lead with vision, use sensory language, and connect to larger purpose. Example: 'Every day, you're creating futures' vs 'We serve children'.",
                category="voice",
                affected_pages=opp.affected_categories,
                expected_impact="Improved emotional engagement and recall"
            ))
            priority += 1
        elif "Justice" in opp.title:
            recommendations.append(Recommendation(
                priority=min(priority, 5),
                title="Reframe Through Justice Lens",
                description="Position work as restoring what's right rather than providing charity. Emphasize dignity, fairness, and what every person deserves.",
                category="content",
                affected_pages=opp.affected_categories,
                expected_impact="Enhanced moral resonance and motivation"
            ))
            priority += 1

    # Always add at least 3 recommendations if we don't have enough
    if len(recommendations) < 3:
        if mode == "NEUTRAL":
            recommendations.append(Recommendation(
                priority=priority,
                title="Develop Distinctive Voice",
                description="Current NEUTRAL tone may be too bland. Identify 2-3 key emotional themes and weave them consistently throughout content.",
                category="voice",
                affected_pages=["all"],
                expected_impact="Improved brand memorability and differentiation"
            ))
            priority += 1

        recommendations.append(Recommendation(
            priority=priority,
            title="Create Narrative Consistency Guide",
            description="Document preferred modes, tone, and key phrases for each content category. Train team to maintain consistent voice.",
            category="process",
            affected_pages=["all"],
            expected_impact="Sustained narrative quality over time"
        ))

    # Sort by priority and limit
    recommendations.sort(key=lambda x: x.priority)
    return recommendations[:8]


# ============================================================================
# EXECUTIVE SUMMARY & CONCLUSION
# ============================================================================

def generate_executive_summary(
    organization_name: str,
    narratives_analyzed: int,
    health_score: int,
    overall_profile: OverallProfile,
    key_findings: KeyFindings
) -> str:
    """Generate the executive summary paragraph."""

    # Health assessment
    if health_score >= 80:
        health_desc = "excellent"
    elif health_score >= 60:
        health_desc = "good"
    elif health_score >= 40:
        health_desc = "moderate"
    else:
        health_desc = "concerning"

    # Build summary
    summary = f"This comprehensive narrative analysis of {organization_name} examined {narratives_analyzed} pages "
    summary += f"across the website. The overall narrative health score is {health_score}/100, indicating {health_desc} performance. "

    # Primary mode context
    mode = overall_profile.primary_mode
    if mode in ["HEROIC", "COMMUNAL"]:
        summary += f"The site maintains a positive {mode} narrative stance that supports engagement and trust. "
    elif mode in ["VICTIM", "CYNICAL_ACHIEVER", "PARANOID"]:
        summary += f"The predominant {mode} mode represents a significant concern requiring immediate attention. "
    else:
        summary += f"The {mode} narrative mode suggests room for strengthening the overall message. "

    # Key highlights
    num_risks = len(key_findings.risks)
    num_strengths = len(key_findings.strengths)

    if num_risks > num_strengths:
        summary += f"The analysis identified {num_risks} risk areas that should be prioritized for improvement."
    elif num_strengths > num_risks:
        summary += f"The analysis found {num_strengths} notable strengths that differentiate the brand."
    else:
        summary += "The analysis found a balance of strengths and improvement opportunities."

    return summary


def generate_conclusion(
    organization_name: str,
    health_score: int,
    key_findings: KeyFindings,
    recommendations: List[Recommendation]
) -> str:
    """Generate the conclusion paragraph."""

    # Overall assessment
    if health_score >= 70:
        assessment = f"{organization_name} demonstrates a generally healthy narrative profile"
    elif health_score >= 50:
        assessment = f"{organization_name} shows a mixed narrative profile with clear opportunities for improvement"
    else:
        assessment = f"{organization_name}'s narrative profile requires significant attention"

    conclusion = f"{assessment}. "

    # Priority actions
    if recommendations:
        top_rec = recommendations[0]
        conclusion += f"The highest priority action is to {top_rec.title.lower()}, "
        conclusion += f"which is expected to result in {top_rec.expected_impact.lower()}. "

    # Forward-looking
    if key_findings.strengths:
        conclusion += f"Building on existing strengths like {key_findings.strengths[0].title.lower()}, "
        conclusion += "the organization can create a more compelling and consistent narrative experience."

    return conclusion


# ============================================================================
# HEALTH SCORE CALCULATION
# ============================================================================

def calculate_health_score(
    overall_metrics: Dict[str, Any],
    category_analyses: List[Dict[str, Any]],
    narrative_coherence: float
) -> int:
    """
    Calculate the overall narrative health score (0-100).

    Components:
    - Mode quality (35%): Based on primary mode health value
    - Dimension balance (30%): Average of normalized dimensions
    - Category consistency (20%): How consistent across categories
    - Narrative coherence (15%): How coherent the overall narrative
    """
    # Mode quality score (0-35)
    mode = overall_metrics.get("dominant_mode", "NEUTRAL")
    mode_quality = MODE_HEALTH_SCORES.get(mode, 0.5)
    mode_score = mode_quality * 35

    # Dimension balance score (0-30)
    agency = overall_metrics.get("mean_agency", 0)
    justice = overall_metrics.get("mean_perceived_justice", 0)
    belonging = overall_metrics.get("mean_belonging", 0)

    # Normalize dimensions from [-2, 2] to [0, 1]
    agency_norm = (agency + 2) / 4
    justice_norm = (justice + 2) / 4
    belonging_norm = (belonging + 2) / 4

    dimension_score = ((agency_norm + justice_norm + belonging_norm) / 3) * 30

    # Category consistency score (0-20)
    # Lower variance = higher consistency
    category_variances = []
    for cat in category_analyses:
        metrics = cat.get("metrics", {})
        variance = (
            metrics.get("std_agency", 0) ** 2 +
            metrics.get("std_perceived_justice", 0) ** 2 +
            metrics.get("std_belonging", 0) ** 2
        ) / 3
        category_variances.append(variance)

    avg_variance = np.mean(category_variances) if category_variances else 0
    # Variance of 0 = score 20, variance of 1+ = score ~0
    consistency_score = max(0, 20 * (1 - min(avg_variance, 1)))

    # Narrative coherence score (0-15)
    coherence_score = narrative_coherence * 15

    # Total
    total = mode_score + dimension_score + consistency_score + coherence_score

    return min(100, max(0, int(round(total))))


# ============================================================================
# MAIN ENDPOINT
# ============================================================================

@router.post("/generate", response_model=ReportResponse)
async def generate_comprehensive_report(request: ReportRequest):
    """
    Generate a comprehensive narrative analysis report for a website.

    This endpoint:
    1. Crawls the website (up to max_pages)
    2. Analyzes each page for narrative dimensions and mode
    3. Categorizes pages by type (homepage, about, pricing, etc.)
    4. Calculates per-category metrics and cross-category comparisons
    5. Generates key findings and recommendations
    6. Produces an executive summary and overall health score

    The report provides actionable insights for improving website narrative health.

    Example:
    ```json
    POST /api/report/generate
    {
        "url": "https://example.com",
        "organization_name": "Example Corp",
        "max_pages": 50,
        "include_synthesis": true
    }
    ```
    """
    import time
    start_time = time.time()

    # Generate unique report ID
    report_id = str(uuid.uuid4())[:8]
    analysis_date = datetime.now().isoformat()

    # Normalize URL
    url = request.url
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # Check that projection is available
    try:
        from main import current_projection
        if current_projection is None:
            raise HTTPException(
                status_code=400,
                detail="No projection trained. Train a projection first using /training/train"
            )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Backend not initialized: {e}")

    # Perform narrative-level analysis for comprehensive reports
    try:
        from api_site_analysis import perform_narrative_analysis, perform_site_analysis

        logger.info(f"Starting comprehensive report for {url} (max {request.max_pages} pages)")

        # Try narrative-level analysis first (produces better reports)
        narrative_result = await perform_narrative_analysis(
            url=url,
            max_pages=request.max_pages,
            min_narratives=30,
            max_narratives=100
        )

        use_narrative_analysis = "error" not in narrative_result and narrative_result.get("narratives")

        if use_narrative_analysis:
            logger.info(f"Using narrative-level analysis: {len(narrative_result['narratives'])} narratives")
        else:
            # Fall back to page-level analysis
            logger.info("Falling back to page-level analysis")
            site_result = await perform_site_analysis(
                url=url,
                max_pages=request.max_pages,
                include_forces=True
            )

            if "error" in site_result:
                raise HTTPException(status_code=400, detail=site_result["error"])

    except ImportError:
        raise HTTPException(status_code=500, detail="Site analysis module not available")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Site analysis timed out")

    # Handle narrative-level vs page-level analysis
    if use_narrative_analysis:
        # Build report from narrative-level analysis
        narratives = narrative_result["narratives"]
        statistics = narrative_result["statistics"]
        by_category = narrative_result["by_category"]

        # Determine organization name
        organization_name = request.organization_name
        if not organization_name:
            parsed = urlparse(url)
            domain_parts = parsed.netloc.replace("www.", "").split(".")
            organization_name = domain_parts[0].replace("-", " ").title() if domain_parts else "Organization"

        # Build category analyses from narrative data
        category_analyses_data = []
        for cat_name, cat_narratives in by_category.items():
            if not cat_narratives:
                continue

            # Calculate category metrics
            cat_agency = [n["coordinates"]["agency"] for n in cat_narratives]
            cat_justice = [n["coordinates"]["perceived_justice"] for n in cat_narratives]
            cat_belonging = [n["coordinates"]["belonging"] for n in cat_narratives]
            cat_modes = [n["mode"] for n in cat_narratives]

            mode_counts = {}
            for m in cat_modes:
                mode_counts[m] = mode_counts.get(m, 0) + 1
            dominant_mode = max(mode_counts, key=mode_counts.get) if mode_counts else "NEUTRAL"
            mode_dist = {m: c/len(cat_narratives) for m, c in mode_counts.items()}

            # Get sample projections (actual narrative quotes) - pick diverse modes
            sample_projections = []
            seen_modes = set()
            seen_titles = set()
            for n in cat_narratives:
                if len(sample_projections) >= 5:
                    break
                # Use longer text excerpts (150 chars) for meaningful quotes
                text = n["text"][:150] + "..." if len(n["text"]) > 150 else n["text"]
                # Skip duplicate titles
                if text in seen_titles:
                    continue
                # Prefer samples with different modes for diversity
                if n["mode"] in seen_modes and len(sample_projections) < 3:
                    continue
                seen_modes.add(n["mode"])
                seen_titles.add(text)
                sample_projections.append({
                    "url": n["source_url"],
                    "title": text,
                    "mode": n["mode"],
                    "coordinates": n["coordinates"]
                })

            # Fill remaining slots if we haven't hit 5
            for n in cat_narratives:
                if len(sample_projections) >= 5:
                    break
                text = n["text"][:150] + "..." if len(n["text"]) > 150 else n["text"]
                if text not in seen_titles:
                    seen_titles.add(text)
                    sample_projections.append({
                        "url": n["source_url"],
                        "title": text,
                        "mode": n["mode"],
                        "coordinates": n["coordinates"]
                    })

            # Format category label
            category_label = cat_name.replace("_", " ").title()

            # Generate richer category narrative
            mean_agency = round(float(np.mean(cat_agency)), 4)
            mean_justice = round(float(np.mean(cat_justice)), 4)
            mean_belonging = round(float(np.mean(cat_belonging)), 4)

            narrative_parts = [f"The {category_label} content"]
            if dominant_mode in ["HEROIC", "COMMUNAL", "TRANSCENDENT"]:
                narrative_parts.append(f"presents a predominantly positive {dominant_mode.replace('_', ' ').lower()} narrative stance")
            elif dominant_mode in ["VICTIM", "CYNICAL_ACHIEVER", "PARANOID"]:
                narrative_parts.append(f"shows concerning {dominant_mode.replace('_', ' ').lower()} patterns")
            elif dominant_mode in ["TRANSITIONAL", "CONFLICTED"]:
                narrative_parts.append(f"displays mixed {dominant_mode.replace('_', ' ').lower()} messaging")
            else:
                narrative_parts.append(f"maintains a {dominant_mode.replace('_', ' ').lower()} tone")

            if mean_agency > 0.3:
                narrative_parts.append("projecting strong reader empowerment")
            elif mean_agency < -0.3:
                narrative_parts.append("which may leave readers feeling limited in options")

            if mean_belonging > 0.3:
                narrative_parts.append("while fostering community connection")
            elif mean_belonging < -0.3:
                narrative_parts.append("though the tone may feel exclusionary")

            category_narrative = ", ".join(narrative_parts) + f" (analyzed {len(cat_narratives)} narratives)."

            # Generate issues and strengths
            issues = []
            strengths = []

            if dominant_mode in ["VICTIM", "CYNICAL_ACHIEVER", "PARANOID"]:
                issues.append(f"Shadow mode ({dominant_mode}) dominates - may create negative reader experience")
            if mean_agency < -0.3:
                issues.append("Low agency language may leave readers feeling powerless")
            if mean_belonging < -0.3:
                issues.append("Weak belonging signals - content may feel exclusionary")
            if len(mode_dist) > 3:
                issues.append("High mode fragmentation - inconsistent voice across narratives")

            if dominant_mode in ["HEROIC", "COMMUNAL", "TRANSCENDENT"]:
                strengths.append(f"Strong positive mode ({dominant_mode}) creates engaging narrative")
            if mean_agency > 0.3:
                strengths.append("Empowering language gives readers sense of agency")
            if mean_belonging > 0.3:
                strengths.append("Community-building language fosters connection")

            category_analyses_data.append({
                "category_id": cat_name,
                "label": category_label,
                "narratives": cat_narratives,
                "metrics": {
                    "mean_agency": mean_agency,
                    "std_agency": round(float(np.std(cat_agency)), 4),
                    "mean_perceived_justice": mean_justice,
                    "std_perceived_justice": round(float(np.std(cat_justice)), 4),
                    "mean_belonging": mean_belonging,
                    "std_belonging": round(float(np.std(cat_belonging)), 4),
                    "dominant_mode": dominant_mode,
                    "mode_distribution": mode_dist,
                    "sample_count": len(cat_narratives)
                },
                "sample_pages": sample_projections,
                "issues": issues,
                "strengths": strengths,
                "narrative": category_narrative
            })

        # Calculate overall metrics
        overall_metrics = {
            "mean_agency": statistics["mean_agency"],
            "std_agency": statistics["std_agency"],
            "mean_perceived_justice": statistics["mean_perceived_justice"],
            "std_perceived_justice": statistics["std_perceived_justice"],
            "mean_belonging": statistics["mean_belonging"],
            "std_belonging": statistics["std_belonging"],
            "dominant_mode": statistics["dominant_mode"],
            "mode_distribution": statistics["mode_distribution"],
            "sample_count": statistics["total_narratives"]
        }

        # Use narratives as "pages" for compatibility
        pages = narratives

    else:
        # Use page-level analysis (fallback)
        pages = site_result.get("pages", [])

    if not pages:
        raise HTTPException(
            status_code=400,
            detail=f"No content could be analyzed from {url}"
        )

    # Continue with page-level processing if not using narrative analysis
    if not use_narrative_analysis:
        # Determine organization name
        organization_name = request.organization_name
        if not organization_name:
            # Try to extract from site URL or first page title
            parsed = urlparse(url)
            domain_parts = parsed.netloc.replace("www.", "").split(".")
            organization_name = domain_parts[0].capitalize() if domain_parts else "Organization"

            # Try to get from homepage title
            for page in pages:
                if categorize_page(page.get("url", "")) == "homepage":
                    title = page.get("title", "")
                    if title and len(title) < 50:
                        # Remove common suffixes
                        for suffix in [" - Home", " | Home", " - Homepage", " | Homepage"]:
                            if title.endswith(suffix):
                                title = title[:-len(suffix)]
                        organization_name = title.strip() or organization_name
                    break

        # Categorize pages
        categorized_pages: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for page in pages:
            category = categorize_page(page.get("url", ""))
            categorized_pages[category].append(page)

        # Build category analyses
        category_analyses_data = []
        for category_id, category_pages in categorized_pages.items():
            if not category_pages:
                continue

            category_info = PAGE_CATEGORIES.get(category_id, PAGE_CATEGORIES["other"])
            metrics = calculate_category_metrics(category_pages)

            # Generate sample pages (top 3 by word count)
            sorted_pages = sorted(
                category_pages,
                key=lambda p: p.get("word_count", 0),
                reverse=True
            )[:3]
            sample_pages = [
                {
                    "url": p.get("url"),
                    "title": p.get("title"),
                    "mode": p.get("mode"),
                    "coordinates": p.get("coordinates")
                }
                for p in sorted_pages
            ]

            issues = generate_category_issues(metrics, category_pages)
            strengths = generate_category_strengths(metrics, category_pages)
            narrative = generate_category_narrative(
                category_info["label"], metrics, category_pages
            )

            category_analyses_data.append({
                "category_id": category_id,
                "label": category_info["label"],
                "pages": category_pages,
                "metrics": metrics,
                "sample_pages": sample_pages,
                "issues": issues,
                "strengths": strengths,
                "narrative": narrative
            })

        # Calculate overall metrics
        all_analyses = pages
        overall_metrics = calculate_category_metrics(all_analyses)

    # Calculate narrative coherence
    narrative_coherence = calculate_narrative_coherence(category_analyses_data)

    # Generate cross-category comparisons
    comparisons = []
    # Get all category IDs from analysis data
    available_categories = [c["category_id"] for c in category_analyses_data]

    # Compare categories to each other
    for i, cat_a_data in enumerate(category_analyses_data):
        for cat_b_data in category_analyses_data[i+1:]:
            comparison = compare_categories(cat_a_data, cat_b_data)
            comparisons.append(comparison)

    # Sort by distance (most significant first)
    comparisons.sort(key=lambda x: -x.euclidean_distance)

    # Calculate health score
    health_score = calculate_health_score(
        overall_metrics, category_analyses_data, narrative_coherence
    )

    # Build overall profile
    agency_values = [p.get("coordinates", {}).get("agency", 0) for p in pages]
    justice_values = [p.get("coordinates", {}).get("perceived_justice", 0) for p in pages]
    belonging_values = [p.get("coordinates", {}).get("belonging", 0) for p in pages]

    mode_dist = overall_metrics.get("mode_distribution", {})
    sorted_modes = sorted(mode_dist.items(), key=lambda x: -x[1])

    overall_narrative = generate_overall_narrative(
        overall_metrics, category_analyses_data, organization_name
    )

    overall_profile = OverallProfile(
        primary_mode=overall_metrics.get("dominant_mode", "NEUTRAL"),
        primary_mode_percentage=round(sorted_modes[0][1] * 100, 1) if sorted_modes else 0,
        secondary_mode=sorted_modes[1][0] if len(sorted_modes) > 1 else None,
        secondary_mode_percentage=round(sorted_modes[1][1] * 100, 1) if len(sorted_modes) > 1 else None,
        mode_category=MODE_PROFILES.get(
            overall_metrics.get("dominant_mode", "NEUTRAL"),
            {"category": "AMBIVALENT"}
        ).category if "analysis" in str(type(overall_metrics)) else "AMBIVALENT",
        agency=calculate_coordinate_stats(agency_values),
        perceived_justice=calculate_coordinate_stats(justice_values),
        belonging=calculate_coordinate_stats(belonging_values),
        narrative_coherence=narrative_coherence,
        narrative_summary=overall_narrative
    )

    # Fix mode_category calculation
    try:
        from analysis.mode_classifier import MODE_PROFILES
        overall_profile.mode_category = MODE_PROFILES.get(
            overall_metrics.get("dominant_mode", "NEUTRAL"),
            type("DefaultProfile", (), {"category": "AMBIVALENT"})()
        ).category
    except Exception:
        overall_profile.mode_category = "AMBIVALENT"

    # Build category analyses response objects
    category_analyses_response = []
    for cat_data in category_analyses_data:
        metrics = cat_data["metrics"]
        category_analyses_response.append(CategoryAnalysis(
            category_id=cat_data["category_id"],
            category_label=cat_data["label"],
            pages_count=metrics["sample_count"],
            mean_agency=metrics["mean_agency"],
            mean_perceived_justice=metrics["mean_perceived_justice"],
            mean_belonging=metrics["mean_belonging"],
            std_agency=metrics["std_agency"],
            std_perceived_justice=metrics["std_perceived_justice"],
            std_belonging=metrics["std_belonging"],
            dominant_mode=metrics["dominant_mode"],
            mode_distribution=metrics["mode_distribution"],
            sample_pages=cat_data["sample_pages"],
            issues=cat_data["issues"],
            strengths=cat_data["strengths"],
            narrative_summary=cat_data["narrative"]
        ))

    # Generate key findings
    key_findings = extract_key_findings(
        overall_metrics, category_analyses_data, comparisons
    )

    # Generate recommendations
    recommendations = generate_recommendations(
        key_findings, category_analyses_data, overall_metrics
    )

    # Generate executive summary
    executive_summary = generate_executive_summary(
        organization_name,
        len(pages),
        health_score,
        overall_profile,
        key_findings
    )

    # Generate conclusion
    conclusion = generate_conclusion(
        organization_name,
        health_score,
        key_findings,
        recommendations
    )

    # Calculate crawl time
    crawl_time = time.time() - start_time

    # Build raw data if requested
    raw_data = None
    if request.include_raw_data:
        if use_narrative_analysis:
            raw_data = {
                "analysis_type": "narrative",
                "narrative_result": narrative_result,
                "category_analyses": {
                    cat["category_id"]: {
                        "metrics": cat["metrics"],
                        "sample_pages": cat["sample_pages"]
                    }
                    for cat in category_analyses_data
                }
            }
        else:
            raw_data = {
                "analysis_type": "page",
                "site_analysis": site_result,
                "categorized_pages": {
                    cat_id: [
                        {
                            "url": p.get("url"),
                            "title": p.get("title"),
                            "mode": p.get("mode"),
                            "coordinates": p.get("coordinates"),
                            "issues": p.get("issues", [])
                        }
                        for p in cat_pages
                    ]
                    for cat_id, cat_pages in categorized_pages.items()
                },
                "category_metrics": {
                    cat["category_id"]: cat["metrics"]
                    for cat in category_analyses_data
                }
            }

    # Run narrative audit for detailed per-narrative feedback
    narrative_audit = None
    if use_narrative_analysis and narratives:
        try:
            audit_result = audit_narratives_batch(narratives)
            narrative_audit = NarrativeAuditSummary(
                average_effectiveness=audit_result["summary_stats"]["average_effectiveness"],
                total_narratives=audit_result["summary_stats"]["total_narratives"],
                high_performing=audit_result["summary_stats"]["high_performing"],
                needs_improvement=audit_result["summary_stats"]["needs_improvement"],
                top_issues=audit_result["summary_stats"]["top_issues"],
                best_narratives=[
                    NarrativeAuditResult(**n) for n in audit_result["best_narratives"]
                ],
                needs_work=[
                    NarrativeAuditResult(**n) for n in audit_result["needs_work"]
                ]
            )
            logger.info(
                f"Narrative audit complete: {audit_result['summary_stats']['high_performing']} high-performing, "
                f"{audit_result['summary_stats']['needs_improvement']} need work"
            )
        except Exception as e:
            logger.error(f"Narrative audit failed: {e}")
            narrative_audit = None

    logger.info(
        f"Comprehensive report generated for {organization_name}: "
        f"{len(pages)} pages, {len(category_analyses_response)} categories, "
        f"health score {health_score}/100, {crawl_time:.1f}s"
    )

    return ReportResponse(
        report_id=report_id,
        organization_name=organization_name,
        site_url=url,
        analysis_date=analysis_date,
        narratives_analyzed=len(pages),
        categories_count=len(category_analyses_response),
        crawl_time_seconds=round(crawl_time, 2),
        executive_summary=executive_summary,
        overall_profile=overall_profile,
        mode_distribution={k: round(v * 100, 1) for k, v in overall_metrics.get("mode_distribution", {}).items()},
        category_analyses=category_analyses_response,
        cross_category_comparisons=comparisons[:10],  # Top 10 comparisons
        key_findings=key_findings,
        recommendations=recommendations,
        conclusion=conclusion,
        health_score=health_score,
        narrative_audit=narrative_audit,
        raw_data=raw_data
    )


@router.get("/status")
async def report_api_status():
    """Get status of the comprehensive report API."""
    try:
        from main import current_projection
        has_projection = current_projection is not None
    except ImportError:
        has_projection = False

    return {
        "status": "running",
        "version": "1.0.0",
        "has_projection": has_projection,
        "page_categories": list(PAGE_CATEGORIES.keys()),
        "endpoints": [
            {
                "path": "/api/report/generate",
                "method": "POST",
                "description": "Generate comprehensive narrative analysis report"
            },
            {
                "path": "/api/report/status",
                "method": "GET",
                "description": "Check API status"
            }
        ],
        "capabilities": [
            "Site-wide crawling and analysis",
            "Page categorization",
            "Per-category metrics",
            "Cross-category comparison",
            "Key findings extraction",
            "Recommendation generation",
            "Health score calculation",
            "Executive summary generation"
        ]
    }


# Import MODE_PROFILES for health calculation
try:
    from analysis.mode_classifier import MODE_PROFILES
except ImportError:
    # Fallback if import fails
    MODE_PROFILES = {}
