"""
Site-Wide Analysis API
======================

This module provides comprehensive site-wide analysis capabilities, crawling
multiple pages on a domain and aggregating narrative analysis across the entire site.

Endpoint:
- POST /api/observer/analyze-site: Deep analysis of an entire website

Features:
- Crawls up to 10 pages on the same domain
- Discovers linked pages (homepage, about, pricing, contact, blog, etc.)
- Extracts text content from each page
- Analyzes each page through /v2/analyze with force field analysis
- Aggregates results for visualization
- Calculates site-wide health score

Usage:
    from api_site_analysis import router as site_analysis_router
    app.include_router(site_analysis_router, prefix="/api/observer", tags=["site-analysis"])
"""

import re
import asyncio
import logging
import time
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import text extraction utilities from api_observer
from api_observer import extract_title, extract_readable_text, count_words

logger = logging.getLogger(__name__)
router = APIRouter()

# Import narrative extractor for enhanced analysis
try:
    from narrative_extractor import extract_site_narratives, ExtractedNarrative
    NARRATIVE_EXTRACTOR_AVAILABLE = True
except ImportError:
    NARRATIVE_EXTRACTOR_AVAILABLE = False
    logger.warning("Narrative extractor not available - using page-level analysis")


# ============================================================================
# Constants
# ============================================================================

# Maximum pages to crawl per site
MAX_PAGES = 10

# Delay between requests to be respectful
REQUEST_DELAY = 0.5  # seconds

# Request timeout
REQUEST_TIMEOUT = 30.0  # seconds

# Priority page patterns (higher priority = crawled first)
PAGE_PRIORITY = {
    r'^/$': 100,  # Homepage
    r'/about': 90,
    r'/pricing': 85,
    r'/contact': 80,
    r'/team': 75,
    r'/careers': 70,
    r'/blog/?$': 65,  # Blog index
    r'/products?': 60,
    r'/services?': 55,
    r'/features?': 50,
    r'/faq': 45,
    r'/help': 40,
    r'/support': 35,
}

# Mode quality scores for health calculation
MODE_QUALITY = {
    "GROWTH_MINDSET": 1.0,
    "COMMUNITY": 0.9,
    "AUTHENTIC_STRUGGLE": 0.8,
    "PRAGMATIC": 0.7,
    "HERO": 0.6,
    "NEUTRAL": 0.5,
    "CONFLICTED": 0.4,
    "TRANSITIONAL": 0.3,
    "VICTIM": 0.2,
    "OUTSIDER": 0.2,
    "NIHILIST": 0.1,
    "NOISE": 0.0,
}

# Category quality scores
CATEGORY_QUALITY = {
    "POSITIVE": 1.0,
    "AMBIVALENT": 0.6,
    "SHADOW": 0.3,
    "EXIT": 0.2,
    "NOISE": 0.0,
}


# ============================================================================
# Request/Response Models
# ============================================================================

class SiteAnalysisRequest(BaseModel):
    """Request for site-wide analysis."""
    url: str = Field(
        ...,
        description="The starting URL to analyze (typically the homepage)",
        examples=["https://example.com"]
    )
    max_pages: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum number of pages to crawl"
    )
    include_forces: bool = Field(
        default=True,
        description="Include attractor/detractor force field analysis"
    )
    timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Timeout per page request in seconds"
    )


class PageAnalysis(BaseModel):
    """Analysis result for a single page."""
    url: str
    title: Optional[str]
    word_count: int
    mode: str
    confidence: float
    coordinates: Dict[str, float]
    force_field: Optional[Dict[str, Any]] = None
    issues: List[str] = []
    sample_text: str = ""


class SiteAnalysisResponse(BaseModel):
    """Response containing site-wide analysis."""
    site_url: str
    pages_analyzed: int
    crawl_time_seconds: float

    # Overall metrics
    health_score: int = Field(..., ge=0, le=100)
    overall_mode: str
    overall_coordinates: Dict[str, float]

    # Per-page results
    pages: List[Dict[str, Any]]

    # Aggregated insights
    mode_distribution: Dict[str, float]
    coordinate_averages: Dict[str, float]
    force_field_aggregate: Optional[Dict[str, Any]] = None

    # Issues and recommendations
    critical_issues: List[Dict[str, Any]]
    recommendations: List[str]

    # Visualization data
    scatter_data: List[Dict[str, Any]]
    trajectory_data: Optional[List[Dict[str, Any]]] = None


# ============================================================================
# URL Crawling Utilities
# ============================================================================

def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs are on the same domain."""
    try:
        parsed1 = urlparse(url1)
        parsed2 = urlparse(url2)

        # Handle www prefix
        domain1 = parsed1.netloc.lower().replace('www.', '')
        domain2 = parsed2.netloc.lower().replace('www.', '')

        return domain1 == domain2
    except Exception:
        return False


def normalize_url(url: str, base_url: str) -> Optional[str]:
    """Normalize a URL, making it absolute if relative."""
    try:
        # Handle relative URLs
        if url.startswith('/'):
            url = urljoin(base_url, url)
        elif not url.startswith(('http://', 'https://')):
            url = urljoin(base_url, url)

        # Parse and reconstruct to normalize
        parsed = urlparse(url)

        # Skip non-HTTP URLs
        if parsed.scheme not in ('http', 'https'):
            return None

        # Skip fragments and query strings for deduplication
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        # Remove trailing slash for consistency (except for root)
        if normalized.endswith('/') and len(parsed.path) > 1:
            normalized = normalized[:-1]

        return normalized
    except Exception:
        return None


def get_page_priority(url: str) -> int:
    """Get priority score for a URL based on page type."""
    parsed = urlparse(url)
    path = parsed.path.lower()

    for pattern, priority in PAGE_PRIORITY.items():
        if re.search(pattern, path):
            return priority

    # Lower priority for deep paths
    depth = path.count('/')
    return max(0, 30 - depth * 5)


def extract_links(html: str, base_url: str) -> List[str]:
    """Extract all links from HTML content."""
    links = []

    # Find all href attributes
    href_pattern = r'href=["\']([^"\']+)["\']'
    matches = re.findall(href_pattern, html, re.IGNORECASE)

    for href in matches:
        # Skip anchors, javascript, mailto, tel
        if href.startswith(('#', 'javascript:', 'mailto:', 'tel:', 'data:')):
            continue

        # Skip common non-content URLs
        if any(ext in href.lower() for ext in ['.pdf', '.jpg', '.png', '.gif', '.svg', '.css', '.js', '.xml', '.json']):
            continue

        normalized = normalize_url(href, base_url)
        if normalized and is_same_domain(normalized, base_url):
            links.append(normalized)

    return links


def should_skip_url(url: str) -> bool:
    """Check if a URL should be skipped (login pages, admin, etc.)."""
    skip_patterns = [
        r'/login',
        r'/signin',
        r'/signup',
        r'/register',
        r'/admin',
        r'/dashboard',
        r'/account',
        r'/cart',
        r'/checkout',
        r'/api/',
        r'/auth/',
        r'/oauth/',
        r'/cdn-cgi/',
        r'/wp-admin',
        r'/wp-json',
    ]

    path = urlparse(url).path.lower()
    return any(re.search(pattern, path) for pattern in skip_patterns)


# ============================================================================
# Page Analysis
# ============================================================================

async def fetch_page(
    client: httpx.AsyncClient,
    url: str,
    timeout: float = REQUEST_TIMEOUT
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Fetch a page and extract its content.

    Returns:
        Tuple of (html, title, text) or (None, None, None) on error
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; CulturalSolitonObservatory/1.0; +https://github.com/cultural-soliton-observatory)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
        }

        response = await client.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get('content-type', '')
        if 'text/html' not in content_type and 'application/xhtml' not in content_type:
            logger.debug(f"Skipping non-HTML content: {url}")
            return None, None, None

        html = response.text
        title = extract_title(html)
        text = extract_readable_text(html)

        return html, title, text

    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching {url}")
        return None, None, None
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP error {e.response.status_code} for {url}")
        return None, None, None
    except Exception as e:
        logger.warning(f"Error fetching {url}: {e}")
        return None, None, None


async def analyze_text_content(
    text: str,
    include_forces: bool = True
) -> Dict[str, Any]:
    """
    Analyze text content using the observatory's analysis endpoints.

    This calls the internal analysis functions directly rather than making HTTP requests.
    """
    # Import here to avoid circular imports
    from main import (
        model_manager, embedding_extractor, current_projection,
        ModelType
    )
    from analysis.mode_classifier import get_mode_classifier
    from analysis.force_field import analyze_force_field
    import numpy as np

    if current_projection is None:
        raise ValueError("No projection trained")

    # Default model
    model_id = "all-MiniLM-L6-v2"

    # Ensure model is loaded
    if not model_manager.is_loaded(model_id):
        model_manager.load_model(model_id, ModelType.SENTENCE_TRANSFORMER)

    # Get embedding
    result = embedding_extractor.extract(text, model_id)

    # Project to 3D
    coords = current_projection.project(result.embedding)
    coords_array = np.array([coords.agency, coords.fairness, coords.belonging])

    # Classify mode
    classifier = get_mode_classifier()
    mode_result = classifier.classify(coords_array)

    analysis = {
        "mode": mode_result["primary_mode"],
        "secondary_mode": mode_result.get("secondary_mode"),
        "confidence": mode_result["confidence"],
        "category": mode_result.get("category", "UNKNOWN"),
        "coordinates": {
            "agency": round(coords.agency, 4),
            "perceived_justice": round(coords.fairness, 4),
            "belonging": round(coords.belonging, 4),
        },
        "mode_probabilities": mode_result.get("mode_probabilities", {}),
    }

    # Add force field analysis if requested
    if include_forces:
        try:
            force_result = analyze_force_field(text, result.embedding)
            analysis["force_field"] = force_result
        except Exception as e:
            logger.warning(f"Force field analysis failed: {e}")
            analysis["force_field"] = None

    return analysis


def identify_issues(
    analysis: Dict[str, Any],
    word_count: int
) -> List[str]:
    """Identify issues with a page based on its analysis."""
    issues = []

    mode = analysis.get("mode", "NEUTRAL")
    confidence = analysis.get("confidence", 0)
    coords = analysis.get("coordinates", {})
    category = analysis.get("category", "")
    force_field = analysis.get("force_field", {})

    # Low word count
    if word_count < 50:
        issues.append("Very little content (under 50 words)")
    elif word_count < 100:
        issues.append("Low content (under 100 words)")

    # Problematic modes
    if mode == "VICTIM":
        issues.append("Victim language detected - may alienate readers")
    elif mode == "NIHILIST":
        issues.append("Nihilistic tone detected - may undermine trust")
    elif mode == "OUTSIDER":
        issues.append("Outsider positioning - may create us-vs-them dynamic")

    # Shadow or Exit categories
    if category == "SHADOW":
        issues.append(f"Shadow mode ({mode}) may create negative impressions")
    elif category == "EXIT":
        issues.append(f"Exit mode ({mode}) suggests disengagement patterns")

    # Low confidence indicates ambiguous messaging
    if confidence < 0.4:
        issues.append("Ambiguous narrative - unclear value proposition")

    # Coordinate issues
    if coords.get("agency", 0) < -0.5:
        issues.append("Low agency language - readers may feel powerless")
    if coords.get("belonging", 0) < -0.5:
        issues.append("Low belonging language - may feel exclusionary")
    if coords.get("perceived_justice", 0) < -0.5:
        issues.append("Low perceived justice - may feel unfair or hostile")

    # Force field issues
    if force_field:
        detractor_strength = force_field.get("detractor_strength", 0)
        if detractor_strength > 1.0:
            issues.append(f"High detractor language - narrative focused on escaping from {force_field.get('primary_detractor', 'unknown')}")

        if force_field.get("force_quadrant") == "PURE_ESCAPE":
            issues.append("Pure escape narrative - lacking positive vision")
        elif force_field.get("force_quadrant") == "STASIS":
            issues.append("Low energy narrative - may not engage readers")

    return issues


# ============================================================================
# Aggregation Utilities
# ============================================================================

def calculate_mode_distribution(analyses: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate the distribution of modes across analyzed pages."""
    mode_counts = defaultdict(int)
    total = len(analyses)

    if total == 0:
        return {}

    for analysis in analyses:
        mode = analysis.get("mode", "NEUTRAL")
        mode_counts[mode] += 1

    return {mode: round(count / total, 3) for mode, count in mode_counts.items()}


def calculate_coordinate_averages(analyses: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate average coordinates across all pages."""
    if not analyses:
        return {"agency": 0, "perceived_justice": 0, "belonging": 0}

    agency_sum = 0
    justice_sum = 0
    belonging_sum = 0
    count = 0

    for analysis in analyses:
        coords = analysis.get("coordinates", {})
        if coords:
            agency_sum += coords.get("agency", 0)
            justice_sum += coords.get("perceived_justice", 0)
            belonging_sum += coords.get("belonging", 0)
            count += 1

    if count == 0:
        return {"agency": 0, "perceived_justice": 0, "belonging": 0}

    return {
        "agency": round(agency_sum / count, 4),
        "perceived_justice": round(justice_sum / count, 4),
        "belonging": round(belonging_sum / count, 4),
    }


def aggregate_force_fields(analyses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Aggregate force field analysis across all pages."""
    force_analyses = [a.get("force_field") for a in analyses if a.get("force_field")]

    if not force_analyses:
        return None

    # Calculate averages
    attractor_sum = sum(f.get("attractor_strength", 0) for f in force_analyses)
    detractor_sum = sum(f.get("detractor_strength", 0) for f in force_analyses)
    count = len(force_analyses)

    avg_attractor = attractor_sum / count
    avg_detractor = detractor_sum / count

    # Count primary attractors/detractors
    attractor_counts = defaultdict(int)
    detractor_counts = defaultdict(int)

    for f in force_analyses:
        if f.get("primary_attractor"):
            attractor_counts[f["primary_attractor"]] += 1
        if f.get("primary_detractor"):
            detractor_counts[f["primary_detractor"]] += 1

    primary_attractor = max(attractor_counts, key=attractor_counts.get) if attractor_counts else None
    primary_detractor = max(detractor_counts, key=detractor_counts.get) if detractor_counts else None

    return {
        "attractor_strength": round(avg_attractor, 4),
        "detractor_strength": round(avg_detractor, 4),
        "primary_attractor": primary_attractor,
        "primary_detractor": primary_detractor,
        "attractor_distribution": dict(attractor_counts),
        "detractor_distribution": dict(detractor_counts),
    }


def calculate_health_score(
    mode_distribution: Dict[str, float],
    coordinate_averages: Dict[str, float],
    force_field_aggregate: Optional[Dict[str, Any]],
    issues_by_page: Dict[str, List[str]],
    page_count: int
) -> int:
    """
    Calculate overall site health score (0-100).

    Components:
    - Mode quality (40%): Higher for positive modes
    - Coordinate balance (30%): Higher for positive coordinates
    - Force field health (15%): Higher for strong attractors, low detractors
    - Consistency (15%): Lower when issues are frequent
    """
    # Mode quality score (0-40)
    mode_score = 0
    for mode, proportion in mode_distribution.items():
        quality = MODE_QUALITY.get(mode, 0.5)
        mode_score += proportion * quality * 40

    # Coordinate balance score (0-30)
    # Transform coordinates from [-2, 2] to [0, 1]
    agency = (coordinate_averages.get("agency", 0) + 2) / 4
    justice = (coordinate_averages.get("perceived_justice", 0) + 2) / 4
    belonging = (coordinate_averages.get("belonging", 0) + 2) / 4

    coord_score = ((agency + justice + belonging) / 3) * 30

    # Force field health score (0-15)
    force_score = 7.5  # Default to middle if no force analysis
    if force_field_aggregate:
        attractor = force_field_aggregate.get("attractor_strength", 0)
        detractor = force_field_aggregate.get("detractor_strength", 0)

        # High attractor + low detractor = healthy
        # Transform from [-2, 2] to [0, 1]
        attractor_norm = (attractor + 2) / 4
        detractor_norm = (detractor + 2) / 4

        # We want high attractor and low detractor
        force_health = (attractor_norm + (1 - detractor_norm)) / 2
        force_score = force_health * 15

    # Consistency score (0-15)
    # Lower when there are many issues
    total_issues = sum(len(issues) for issues in issues_by_page.values())
    avg_issues = total_issues / max(page_count, 1)

    # 0 issues = 15 points, 4+ issues per page = 0 points
    consistency_score = max(0, 15 - (avg_issues * 3.75))

    # Total score
    total = mode_score + coord_score + force_score + consistency_score

    return min(100, max(0, int(round(total))))


def generate_recommendations(
    mode_distribution: Dict[str, float],
    coordinate_averages: Dict[str, float],
    force_field_aggregate: Optional[Dict[str, Any]],
    issues_by_page: Dict[str, List[str]],
    critical_issues: List[Dict[str, Any]]
) -> List[str]:
    """Generate actionable recommendations based on analysis."""
    recommendations = []

    # Mode-based recommendations
    victim_proportion = mode_distribution.get("VICTIM", 0)
    nihilist_proportion = mode_distribution.get("NIHILIST", 0)
    outsider_proportion = mode_distribution.get("OUTSIDER", 0)

    if victim_proportion > 0.2:
        recommendations.append("Reduce victim language - reframe challenges as opportunities for growth")

    if nihilist_proportion > 0.1:
        recommendations.append("Add purpose-driven language - connect features to meaningful outcomes")

    if outsider_proportion > 0.2:
        recommendations.append("Soften us-vs-them framing - focus on shared values rather than opposition")

    # Coordinate-based recommendations
    coords = coordinate_averages

    if coords.get("agency", 0) < 0:
        recommendations.append("Add agency-focused language - use 'you can', 'choose', 'take control'")

    if coords.get("belonging", 0) < 0:
        recommendations.append("Include community testimonials and social proof for belonging")

    if coords.get("perceived_justice", 0) < 0:
        recommendations.append("Add fairness language - emphasize transparency and equal treatment")

    # Force field recommendations
    if force_field_aggregate:
        detractor = force_field_aggregate.get("detractor_strength", 0)
        attractor = force_field_aggregate.get("attractor_strength", 0)

        if detractor > attractor:
            recommendations.append("Balance escape narratives with positive aspirations")

        if force_field_aggregate.get("primary_detractor") == "INJUSTICE":
            recommendations.append("Address fairness concerns explicitly rather than implicitly")

        if attractor < 0.5 and detractor < 0.5:
            recommendations.append("Add more emotional engagement - current copy may feel flat")

    # Issue-based recommendations
    all_issues = []
    for issues in issues_by_page.values():
        all_issues.extend(issues)

    if all_issues.count("Ambiguous narrative - unclear value proposition") > 2:
        recommendations.append("Clarify value proposition on key pages - readers may be confused")

    if all_issues.count("Low agency language - readers may feel powerless") > 2:
        recommendations.append("Reframe limitations as choices throughout the site")

    # Deduplicate and limit
    seen = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recommendations.append(rec)

    return unique_recommendations[:8]  # Limit to 8 recommendations


# ============================================================================
# Main Endpoint
# ============================================================================

@router.post("/analyze-site", response_model=SiteAnalysisResponse)
async def analyze_site(request: SiteAnalysisRequest):
    """
    Perform comprehensive site-wide analysis.

    This endpoint:
    1. Crawls up to max_pages on the same domain
    2. Discovers and prioritizes key pages (homepage, about, pricing, etc.)
    3. Extracts text content from each page
    4. Analyzes each page for narrative mode and force field
    5. Aggregates results across all pages
    6. Calculates a site-wide health score
    7. Generates recommendations

    The response includes:
    - Overall site health score (0-100)
    - Per-page analysis with mode, coordinates, and issues
    - Aggregated mode distribution and coordinate averages
    - Force field aggregate (attractors/detractors)
    - Critical issues and actionable recommendations
    - Visualization data for scatter plots

    Example:
    ```json
    POST /api/observer/analyze-site
    {
        "url": "https://example.com",
        "max_pages": 10,
        "include_forces": true
    }
    ```
    """
    start_time = time.time()

    # Normalize starting URL
    url = request.url
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    base_url = url
    max_pages = min(request.max_pages, MAX_PAGES)

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

    # Crawl and analyze pages
    visited: Set[str] = set()
    to_visit: List[Tuple[int, str]] = [(get_page_priority(url), url)]  # (priority, url)
    page_analyses: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(http2=True) as client:
        while to_visit and len(page_analyses) < max_pages:
            # Sort by priority (highest first) and pop
            to_visit.sort(key=lambda x: -x[0])
            _, current_url = to_visit.pop(0)

            if current_url in visited:
                continue

            if should_skip_url(current_url):
                visited.add(current_url)
                continue

            visited.add(current_url)

            # Fetch page
            html, title, text = await fetch_page(client, current_url, request.timeout)

            if not html or not text:
                continue

            word_count = count_words(text)

            # Skip pages with very little content
            if word_count < 20:
                logger.debug(f"Skipping {current_url} - too little content ({word_count} words)")
                continue

            # Extract links for crawling
            new_links = extract_links(html, current_url)
            for link in new_links:
                if link not in visited:
                    to_visit.append((get_page_priority(link), link))

            # Analyze the page content
            try:
                # Truncate text if too long to avoid analysis issues
                analysis_text = text[:10000] if len(text) > 10000 else text
                analysis = await analyze_text_content(analysis_text, request.include_forces)

                # Identify issues
                issues = identify_issues(analysis, word_count)

                # Build page result
                page_result = {
                    "url": current_url,
                    "title": title or "Untitled",
                    "word_count": word_count,
                    "mode": analysis["mode"],
                    "confidence": round(analysis["confidence"], 4),
                    "coordinates": analysis["coordinates"],
                    "category": analysis.get("category", "UNKNOWN"),
                    "issues": issues,
                    "sample_text": text[:200] + "..." if len(text) > 200 else text,
                }

                if request.include_forces and analysis.get("force_field"):
                    page_result["force_field"] = analysis["force_field"]

                page_analyses.append(page_result)

            except Exception as e:
                logger.error(f"Analysis failed for {current_url}: {e}")
                continue

            # Delay between requests
            await asyncio.sleep(REQUEST_DELAY)

    # Check if we got any results
    if not page_analyses:
        raise HTTPException(
            status_code=400,
            detail=f"Could not analyze any pages from {url}. Check that the site is accessible."
        )

    # Calculate aggregates
    mode_distribution = calculate_mode_distribution(page_analyses)
    coordinate_averages = calculate_coordinate_averages(page_analyses)
    force_field_aggregate = aggregate_force_fields(page_analyses) if request.include_forces else None

    # Build issues map
    issues_by_page = {p["url"]: p["issues"] for p in page_analyses}

    # Identify critical issues
    critical_issues = []
    for page in page_analyses:
        for issue in page["issues"]:
            severity = "high" if any(word in issue.lower() for word in ["victim", "nihilist", "outsider", "alienate"]) else "medium"
            critical_issues.append({
                "page": page["title"],
                "url": page["url"],
                "issue": issue,
                "severity": severity,
            })

    # Sort by severity
    critical_issues.sort(key=lambda x: 0 if x["severity"] == "high" else 1)
    critical_issues = critical_issues[:10]  # Limit to 10 most critical

    # Calculate health score
    health_score = calculate_health_score(
        mode_distribution,
        coordinate_averages,
        force_field_aggregate,
        issues_by_page,
        len(page_analyses)
    )

    # Generate recommendations
    recommendations = generate_recommendations(
        mode_distribution,
        coordinate_averages,
        force_field_aggregate,
        issues_by_page,
        critical_issues
    )

    # Determine overall mode (most common)
    overall_mode = max(mode_distribution, key=mode_distribution.get) if mode_distribution else "NEUTRAL"

    # Build scatter data for visualization
    scatter_data = [
        {
            "page": p["title"],
            "url": p["url"],
            "x": p["coordinates"]["agency"],
            "y": p["coordinates"]["perceived_justice"],
            "z": p["coordinates"]["belonging"],
            "mode": p["mode"],
            "confidence": p["confidence"],
        }
        for p in page_analyses
    ]

    # Calculate crawl time
    crawl_time = time.time() - start_time

    return SiteAnalysisResponse(
        site_url=base_url,
        pages_analyzed=len(page_analyses),
        crawl_time_seconds=round(crawl_time, 2),
        health_score=health_score,
        overall_mode=overall_mode,
        overall_coordinates=coordinate_averages,
        pages=page_analyses,
        mode_distribution=mode_distribution,
        coordinate_averages=coordinate_averages,
        force_field_aggregate=force_field_aggregate,
        critical_issues=critical_issues,
        recommendations=recommendations,
        scatter_data=scatter_data,
        trajectory_data=None,  # Could be populated if pages have natural order
    )


@router.get("/analyze-site/status")
async def site_analysis_status():
    """Get status of the site analysis API."""
    # Check projection status
    try:
        from main import current_projection
        has_projection = current_projection is not None
    except ImportError:
        has_projection = False

    return {
        "status": "running",
        "version": "1.0.0",
        "has_projection": has_projection,
        "max_pages": MAX_PAGES,
        "request_delay_seconds": REQUEST_DELAY,
        "endpoints": [
            {
                "path": "/api/observer/analyze-site",
                "method": "POST",
                "description": "Comprehensive site-wide analysis"
            }
        ],
        "capabilities": [
            "Same-domain crawling",
            "Page prioritization",
            "Text extraction",
            "Mode classification",
            "Force field analysis",
            "Health scoring",
            "Recommendation generation"
        ]
    }


# ============================================================================
# Direct Function Call (for use from other modules)
# ============================================================================

async def perform_site_analysis(url: str, max_pages: int = 10, include_forces: bool = True) -> Dict[str, Any]:
    """
    Perform site analysis directly without going through the FastAPI endpoint.
    This can be called from other modules like api_observer_chat.

    Args:
        url: The URL to analyze
        max_pages: Maximum number of pages to crawl (default: 10)
        include_forces: Whether to include force field analysis (default: True)

    Returns:
        Dict with site analysis results (same structure as SiteAnalysisResponse)
    """
    try:
        request = SiteAnalysisRequest(url=url, max_pages=max_pages, include_forces=include_forces)
        response = await analyze_site(request)
        # Convert Pydantic model to dict
        return response.model_dump()
    except HTTPException as e:
        return {"error": e.detail}
    except Exception as e:
        logger.error(f"perform_site_analysis failed: {e}")
        return {"error": str(e)}


async def crawl_site_with_html(
    url: str,
    max_pages: int = 20
) -> List[Dict[str, Any]]:
    """
    Crawl a site and return pages with raw HTML for narrative extraction.

    Args:
        url: Starting URL
        max_pages: Maximum pages to crawl

    Returns:
        List of dicts with url, title, html, text for each page
    """
    # Normalize URL
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    pages_data = []
    visited: Set[str] = set()
    to_visit: List[Tuple[int, str]] = [(100, url)]

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        while to_visit and len(pages_data) < max_pages:
            # Sort by priority and get next URL
            to_visit.sort(key=lambda x: -x[0])
            _, current_url = to_visit.pop(0)

            # Skip if already visited or should skip
            normalized = normalize_url(current_url, base_url)
            if not normalized or normalized in visited:
                continue
            if should_skip_url(normalized):
                continue

            visited.add(normalized)

            try:
                html, title, text = await fetch_page(client, normalized)

                if not html or not text:
                    continue

                word_count = count_words(text)
                if word_count < 30:
                    continue

                # Store the page with HTML
                pages_data.append({
                    "url": normalized,
                    "title": title or "Untitled",
                    "html": html,
                    "text": text,
                    "word_count": word_count
                })

                # Extract links for crawling
                new_links = extract_links(html, normalized)
                for link in new_links:
                    if link not in visited:
                        to_visit.append((get_page_priority(link), link))

                await asyncio.sleep(REQUEST_DELAY)

            except Exception as e:
                logger.warning(f"Error fetching {normalized}: {e}")
                continue

    logger.info(f"Crawled {len(pages_data)} pages from {base_url}")
    return pages_data


async def perform_narrative_analysis(
    url: str,
    max_pages: int = 20,
    min_narratives: int = 30,
    max_narratives: int = 100
) -> Dict[str, Any]:
    """
    Perform narrative-level analysis of a website.

    This extracts individual narrative statements from pages and analyzes
    each one separately, producing case-study quality reports.

    Args:
        url: Starting URL
        max_pages: Maximum pages to crawl
        min_narratives: Target minimum narratives
        max_narratives: Maximum total narratives

    Returns:
        Dict with:
        - narratives: List of analyzed narratives
        - by_category: Narratives grouped by category
        - page_data: Original page data
        - statistics: Aggregate statistics
    """
    import time
    start_time = time.time()

    if not NARRATIVE_EXTRACTOR_AVAILABLE:
        return {"error": "Narrative extractor not available"}

    # Crawl site and get HTML
    pages_data = await crawl_site_with_html(url, max_pages)

    if not pages_data:
        return {"error": f"Could not crawl any pages from {url}"}

    # Extract narratives from pages
    all_narratives, by_category = extract_site_narratives(
        pages_data,
        min_narratives=min_narratives,
        max_narratives=max_narratives
    )

    if not all_narratives:
        return {"error": "No narratives could be extracted from the site"}

    # Analyze each narrative through the observatory
    analyzed_narratives = []

    for narrative in all_narratives:
        try:
            analysis = await analyze_text_content(narrative.text, include_forces=True)

            analyzed_narratives.append({
                "text": narrative.text,
                "source_url": narrative.source_url,
                "source_title": narrative.source_title,
                "category": narrative.category,
                "element_type": narrative.element_type,
                "word_count": narrative.word_count,
                "mode": analysis["mode"],
                "confidence": round(analysis["confidence"], 4),
                "coordinates": analysis["coordinates"],
                "force_field": analysis.get("force_field"),
                "mode_category": analysis.get("category", "UNKNOWN")
            })
        except Exception as e:
            logger.warning(f"Failed to analyze narrative: {e}")
            continue

    if not analyzed_narratives:
        return {"error": "No narratives could be analyzed"}

    # Group by category
    analyzed_by_category: Dict[str, List[Dict]] = {}
    for narr in analyzed_narratives:
        cat = narr["category"]
        if cat not in analyzed_by_category:
            analyzed_by_category[cat] = []
        analyzed_by_category[cat].append(narr)

    # Calculate statistics
    mode_counts: Dict[str, int] = {}
    agency_values = []
    justice_values = []
    belonging_values = []

    for narr in analyzed_narratives:
        mode = narr["mode"]
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        agency_values.append(narr["coordinates"]["agency"])
        justice_values.append(narr["coordinates"]["perceived_justice"])
        belonging_values.append(narr["coordinates"]["belonging"])

    total = len(analyzed_narratives)
    mode_distribution = {mode: count/total for mode, count in mode_counts.items()}

    # Find dominant mode
    dominant_mode = max(mode_counts, key=mode_counts.get) if mode_counts else "NEUTRAL"

    import numpy as np
    statistics = {
        "total_narratives": total,
        "pages_crawled": len(pages_data),
        "categories_found": len(analyzed_by_category),
        "dominant_mode": dominant_mode,
        "mode_distribution": mode_distribution,
        "mean_agency": round(float(np.mean(agency_values)), 4),
        "std_agency": round(float(np.std(agency_values)), 4),
        "mean_perceived_justice": round(float(np.mean(justice_values)), 4),
        "std_perceived_justice": round(float(np.std(justice_values)), 4),
        "mean_belonging": round(float(np.mean(belonging_values)), 4),
        "std_belonging": round(float(np.std(belonging_values)), 4),
        "crawl_time_seconds": round(time.time() - start_time, 2)
    }

    logger.info(
        f"Narrative analysis complete: {total} narratives from {len(pages_data)} pages "
        f"in {statistics['crawl_time_seconds']}s"
    )

    return {
        "narratives": analyzed_narratives,
        "by_category": analyzed_by_category,
        "page_data": [{"url": p["url"], "title": p["title"], "word_count": p["word_count"]} for p in pages_data],
        "statistics": statistics
    }
