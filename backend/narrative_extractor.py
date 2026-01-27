"""
Narrative Extractor
====================

Extracts individual narrative statements from web pages for granular analysis.
Instead of analyzing entire pages as single units, this module breaks down
content into analyzable narrative chunks.

A narrative is defined as:
- A meaningful statement that expresses a position, value, or call to action
- Minimum 10 words, maximum 500 words
- Can be: headings, paragraphs, quotes, CTAs, mission statements, testimonials

Categories:
- mission_vision: Mission statements, vision, purpose
- problem_framing: Crisis descriptions, statistics, challenges
- solution_framing: Approach, methodology, how we help
- impact_stories: Success stories, testimonials, transformations
- organizational_voice: Team, founder stories, achievements
- donor_appeals: Donation CTAs, giving language, sponsorship
- advocacy: Calls to action, volunteer, policy positions
- general: Other content
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class ExtractedNarrative:
    """A single extracted narrative statement."""
    text: str
    source_url: str
    source_title: str
    category: str
    element_type: str  # heading, paragraph, quote, cta, list_item
    position: int  # order on page
    word_count: int
    confidence: float = 0.0  # category confidence

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Category Detection
# =============================================================================

CATEGORY_PATTERNS = {
    "mission_vision": {
        "keywords": [
            "mission", "vision", "purpose", "believe", "committed to",
            "our goal", "we exist", "changing the world", "dedicated to",
            "transform", "revolutionize", "reimagine", "strive to",
            "core values", "what we stand for", "why we", "founded on"
        ],
        "url_patterns": ["/about", "/mission", "/vision", "/who-we-are", "/our-story"],
        "weight": 1.5
    },
    "problem_framing": {
        "keywords": [
            "million", "billion", "crisis", "problem", "challenge",
            "statistics", "sadly", "unfortunately", "struggle",
            "orphan", "homeless", "hungry", "vulnerable", "at risk",
            "die", "suffer", "lack", "without", "need", "crisis"
        ],
        "url_patterns": ["/crisis", "/problem", "/why", "/need"],
        "weight": 1.3
    },
    "solution_framing": {
        "keywords": [
            "solution", "approach", "how we", "our work", "our model",
            "we provide", "we offer", "through our", "by providing",
            "program", "initiative", "strategy", "method", "process"
        ],
        "url_patterns": ["/solution", "/approach", "/how", "/our-work", "/programs"],
        "weight": 1.3
    },
    "impact_stories": {
        "keywords": [
            "story", "meet", "journey", "testimonial", "changed",
            "transformed", "before", "after", "now", "thanks to",
            "because of", "life", "family", "child", "person",
            "was able to", "finally", "dream", "hope"
        ],
        "url_patterns": ["/stories", "/impact", "/testimonials", "/success"],
        "weight": 1.4
    },
    "organizational_voice": {
        "keywords": [
            "team", "founder", "staff", "history", "started",
            "launched", "established", "since", "years", "experience",
            "our team", "leadership", "board", "partnership"
        ],
        "url_patterns": ["/team", "/about-us", "/leadership", "/history"],
        "weight": 1.2
    },
    "donor_appeals": {
        "keywords": [
            "donate", "give", "support", "sponsor", "monthly",
            "contribution", "gift", "tax-deductible", "100%",
            "your donation", "help us", "join us", "make a difference",
            "change a life", "save", "rescue", "provide"
        ],
        "url_patterns": ["/donate", "/give", "/support", "/ways-to-give"],
        "weight": 1.6
    },
    "advocacy": {
        "keywords": [
            "advocate", "volunteer", "action", "policy", "join",
            "movement", "campaign", "sign", "petition", "voice",
            "stand with", "fight for", "speak up", "take action"
        ],
        "url_patterns": ["/advocate", "/volunteer", "/action", "/get-involved"],
        "weight": 1.3
    }
}


def categorize_narrative(
    text: str,
    url: str = "",
    title: str = ""
) -> Tuple[str, float]:
    """
    Categorize a narrative statement.

    Returns:
        Tuple of (category, confidence)
    """
    text_lower = text.lower()
    url_lower = url.lower()
    title_lower = title.lower()

    scores: Dict[str, float] = {}

    for category, patterns in CATEGORY_PATTERNS.items():
        score = 0.0

        # Keyword matching
        keywords = patterns["keywords"]
        for keyword in keywords:
            if keyword in text_lower:
                score += 1.0
            if keyword in title_lower:
                score += 0.5

        # URL pattern matching
        url_patterns = patterns["url_patterns"]
        for pattern in url_patterns:
            if pattern in url_lower:
                score += 2.0
                break

        # Apply category weight
        score *= patterns["weight"]
        scores[category] = score

    # Find best category
    if not scores or max(scores.values()) == 0:
        return "general", 0.5

    best_category = max(scores, key=lambda k: scores[k])
    best_score = scores[best_category]

    # Normalize confidence (0-1)
    confidence = min(1.0, best_score / 10.0)

    # If confidence is too low, fall back to general
    if confidence < 0.2:
        return "general", 0.5

    return best_category, confidence


# =============================================================================
# HTML Parsing
# =============================================================================

def extract_narratives_from_html(
    html: str,
    url: str,
    title: str = "",
    min_words: int = 8,
    max_words: int = 500
) -> List[ExtractedNarrative]:
    """
    Extract individual narrative statements from HTML.

    Args:
        html: Raw HTML content
        url: Source URL
        title: Page title
        min_words: Minimum words for a narrative
        max_words: Maximum words for a narrative

    Returns:
        List of ExtractedNarrative objects
    """
    narratives = []
    position = 0
    seen_texts = set()  # Deduplication

    # Clean HTML first
    html = _clean_html(html)

    # Extract from different element types
    extractors = [
        (_extract_headings, "heading"),
        (_extract_paragraphs, "paragraph"),
        (_extract_blockquotes, "quote"),
        (_extract_ctas, "cta"),
        (_extract_list_items, "list_item"),
    ]

    for extractor, element_type in extractors:
        texts = extractor(html)
        for text in texts:
            # Clean and validate
            text = _clean_text(text)
            word_count = len(text.split())

            # Skip if too short/long or duplicate
            if word_count < min_words or word_count > max_words:
                continue

            # Dedup using normalized text
            text_key = text.lower().strip()[:100]
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            # Categorize
            category, confidence = categorize_narrative(text, url, title)

            narrative = ExtractedNarrative(
                text=text,
                source_url=url,
                source_title=title,
                category=category,
                element_type=element_type,
                position=position,
                word_count=word_count,
                confidence=confidence
            )
            narratives.append(narrative)
            position += 1

    # Sort by position
    narratives.sort(key=lambda n: n.position)

    return narratives


def _clean_html(html: str) -> str:
    """Remove non-content elements."""
    # Remove script/style blocks
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<noscript[^>]*>.*?</noscript>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove nav/footer/header (often boilerplate)
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

    # Remove SVG/canvas
    html = re.sub(r'<svg[^>]*>.*?</svg>', '', html, flags=re.DOTALL | re.IGNORECASE)

    return html


def _clean_text(text: str) -> str:
    """Clean extracted text."""
    # Decode HTML entities
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&#39;', "'", text)
    text = re.sub(r'&mdash;', '—', text)
    text = re.sub(r'&ndash;', '–', text)

    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def _extract_headings(html: str) -> List[str]:
    """Extract h1-h3 headings."""
    texts = []
    pattern = r'<h[1-3][^>]*>(.*?)</h[1-3]>'
    matches = re.findall(pattern, html, flags=re.DOTALL | re.IGNORECASE)
    texts.extend(matches)
    return texts


def _extract_paragraphs(html: str) -> List[str]:
    """Extract paragraph content."""
    texts = []
    pattern = r'<p[^>]*>(.*?)</p>'
    matches = re.findall(pattern, html, flags=re.DOTALL | re.IGNORECASE)
    texts.extend(matches)
    return texts


def _extract_blockquotes(html: str) -> List[str]:
    """Extract blockquotes and testimonials."""
    texts = []
    pattern = r'<blockquote[^>]*>(.*?)</blockquote>'
    matches = re.findall(pattern, html, flags=re.DOTALL | re.IGNORECASE)
    texts.extend(matches)
    return texts


def _extract_ctas(html: str) -> List[str]:
    """Extract call-to-action buttons and links with meaningful text."""
    texts = []

    # Button text
    pattern = r'<button[^>]*>(.*?)</button>'
    matches = re.findall(pattern, html, flags=re.DOTALL | re.IGNORECASE)
    texts.extend(matches)

    # Links with class containing 'cta', 'btn', 'button', 'donate'
    pattern = r'<a[^>]*class="[^"]*(?:cta|btn|button|donate)[^"]*"[^>]*>(.*?)</a>'
    matches = re.findall(pattern, html, flags=re.DOTALL | re.IGNORECASE)
    texts.extend(matches)

    return texts


def _extract_list_items(html: str) -> List[str]:
    """Extract meaningful list items (often used for features/benefits)."""
    texts = []
    pattern = r'<li[^>]*>(.*?)</li>'
    matches = re.findall(pattern, html, flags=re.DOTALL | re.IGNORECASE)

    # Only include list items with substantial content
    for match in matches:
        clean = _clean_text(match)
        if len(clean.split()) >= 5:  # At least 5 words
            texts.append(match)

    return texts


# =============================================================================
# Batch Processing
# =============================================================================

def extract_narratives_from_pages(
    pages: List[Dict[str, str]],
    min_words: int = 8,
    max_words: int = 500,
    max_narratives_per_page: int = 20
) -> List[ExtractedNarrative]:
    """
    Extract narratives from multiple pages.

    Args:
        pages: List of {"url": str, "title": str, "html": str}
        min_words: Minimum words per narrative
        max_words: Maximum words per narrative
        max_narratives_per_page: Limit narratives per page

    Returns:
        List of all extracted narratives
    """
    all_narratives = []

    for page in pages:
        url = page.get("url", "")
        title = page.get("title", "")
        html = page.get("html", "")

        if not html:
            continue

        page_narratives = extract_narratives_from_html(
            html=html,
            url=url,
            title=title,
            min_words=min_words,
            max_words=max_words
        )

        # Limit per page to avoid one page dominating
        all_narratives.extend(page_narratives[:max_narratives_per_page])

    logger.info(f"Extracted {len(all_narratives)} narratives from {len(pages)} pages")

    return all_narratives


def group_narratives_by_category(
    narratives: List[ExtractedNarrative]
) -> Dict[str, List[ExtractedNarrative]]:
    """Group narratives by their category."""
    groups: Dict[str, List[ExtractedNarrative]] = {}

    for narrative in narratives:
        category = narrative.category
        if category not in groups:
            groups[category] = []
        groups[category].append(narrative)

    return groups


# =============================================================================
# Main API
# =============================================================================

def extract_site_narratives(
    pages_data: List[Dict[str, str]],
    min_narratives: int = 20,
    max_narratives: int = 100
) -> Tuple[List[ExtractedNarrative], Dict[str, List[ExtractedNarrative]]]:
    """
    Main entry point for extracting narratives from a site.

    Args:
        pages_data: List of page data with url, title, html
        min_narratives: Target minimum narratives
        max_narratives: Maximum total narratives

    Returns:
        Tuple of (all_narratives, grouped_by_category)
    """
    # Extract all narratives
    all_narratives = extract_narratives_from_pages(
        pages_data,
        max_narratives_per_page=30
    )

    # If we have too few, lower the word threshold
    if len(all_narratives) < min_narratives:
        logger.info(f"Only {len(all_narratives)} narratives, retrying with lower threshold")
        all_narratives = extract_narratives_from_pages(
            pages_data,
            min_words=5,
            max_narratives_per_page=40
        )

    # Limit total
    if len(all_narratives) > max_narratives:
        # Prioritize diversity - take top N from each category
        grouped = group_narratives_by_category(all_narratives)
        per_category = max_narratives // max(len(grouped), 1)

        limited = []
        for category, narrs in grouped.items():
            limited.extend(narrs[:per_category])

        all_narratives = limited[:max_narratives]

    # Group by category
    grouped = group_narratives_by_category(all_narratives)

    logger.info(
        f"Final: {len(all_narratives)} narratives across {len(grouped)} categories"
    )

    return all_narratives, grouped
