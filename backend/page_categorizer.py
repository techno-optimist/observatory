"""
Page Categorizer Module

Intelligently categorizes web pages into narrative categories based on
URL patterns, keyword matching, and content analysis.

Categories:
- mission_vision: Mission statements, vision, about pages
- problem_framing: Crisis descriptions, statistics about problems
- solution_framing: How the org solves problems, approach
- impact_stories: Success stories, testimonials, case studies
- organizational_voice: Team pages, founder stories, org achievements
- donor_appeals: Donation CTAs, giving pages, sponsorship
- advocacy: Calls to action, volunteer, policy positions
- general: Fallback when no clear category matches
"""

from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
import re


# Keyword dictionaries for each category
# Keywords are weighted by position in list (earlier = higher weight)
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "mission_vision": [
        "mission", "vision", "purpose", "why we exist", "about us",
        "who we are", "our story", "what we believe", "core values",
        "guiding principles", "founding", "heritage", "philosophy",
        "committed to", "dedicated to", "driven by"
    ],
    "problem_framing": [
        "crisis", "million", "statistics", "problem", "challenge", "orphan",
        "poverty", "hunger", "disease", "suffering", "vulnerable",
        "at-risk", "homeless", "displaced", "struggling", "urgent need",
        "devastating", "alarming", "tragic", "epidemic", "emergency",
        "percent", "annually", "worldwide", "globally"
    ],
    "solution_framing": [
        "approach", "solution", "how we", "our work", "model",
        "methodology", "strategy", "program", "initiative", "intervention",
        "framework", "process", "system", "method", "technique",
        "evidence-based", "proven", "effective", "sustainable", "holistic",
        "comprehensive", "innovative", "unique approach"
    ],
    "impact_stories": [
        "story", "meet", "journey", "testimonial", "changed", "transformed",
        "success story", "case study", "real life", "firsthand",
        "before and after", "life changed", "now she", "now he",
        "thanks to", "because of you", "your support", "made possible",
        "graduate", "survivor", "overcame", "thriving"
    ],
    "organizational_voice": [
        "team", "founder", "staff", "history", "started", "launched",
        "leadership", "board", "director", "ceo", "president",
        "established", "founded in", "years of experience", "our people",
        "meet the team", "our experts", "headquarters", "offices",
        "partners", "awards", "recognition", "achievement", "milestone"
    ],
    "donor_appeals": [
        "donate", "give", "support", "sponsor", "monthly", "contribution",
        "donation", "gift", "pledge", "recurring", "one-time",
        "tax-deductible", "make a difference", "change a life",
        "your gift", "give today", "donate now", "support us",
        "ways to give", "planned giving", "legacy", "matching gift",
        "fundraise", "campaign", "goal"
    ],
    "advocacy": [
        "advocate", "volunteer", "action", "policy", "join", "movement",
        "campaign", "petition", "sign up", "take action", "raise awareness",
        "speak out", "contact your", "write to", "call your",
        "legislation", "bill", "congress", "lawmakers", "rights",
        "justice", "equality", "change", "reform", "mobilize"
    ]
}

# URL patterns for each category
# Patterns are matched against the URL path (case-insensitive)
URL_PATTERNS: Dict[str, List[str]] = {
    "mission_vision": [
        "/about", "/mission", "/vision", "/who-we-are", "/our-story",
        "/about-us", "/what-we-do", "/our-mission", "/our-vision",
        "/values", "/beliefs", "/philosophy"
    ],
    "problem_framing": [
        "/the-problem", "/the-crisis", "/why-it-matters", "/the-need",
        "/statistics", "/facts", "/the-issue", "/challenges",
        "/global-crisis", "/urgency"
    ],
    "solution_framing": [
        "/our-approach", "/how-we-work", "/our-model", "/our-solution",
        "/methodology", "/programs", "/initiatives", "/what-we-do",
        "/our-work", "/strategy", "/theory-of-change"
    ],
    "impact_stories": [
        "/stories", "/impact", "/testimonials", "/blog", "/news",
        "/success-stories", "/case-studies", "/results", "/outcomes",
        "/meet", "/voices", "/updates", "/reports"
    ],
    "organizational_voice": [
        "/team", "/leadership", "/staff", "/board", "/founders",
        "/our-team", "/about-us/team", "/history", "/timeline",
        "/careers", "/jobs", "/partners", "/awards"
    ],
    "donor_appeals": [
        "/donate", "/give", "/support", "/ways-to-give", "/contribute",
        "/donation", "/giving", "/support-us", "/make-a-gift",
        "/sponsor", "/fundraise", "/campaign", "/monthly"
    ],
    "advocacy": [
        "/advocate", "/volunteer", "/action", "/take-action", "/get-involved",
        "/join", "/policy", "/campaigns", "/petition", "/advocacy",
        "/mobilize", "/events", "/signup"
    ]
}

# Scoring weights for different signal types
URL_PATTERN_WEIGHT: float = 3.0  # URL patterns are strong signals
KEYWORD_BASE_WEIGHT: float = 1.0  # Base weight for keyword matches
TITLE_MULTIPLIER: float = 2.0  # Keywords in title are weighted higher
MIN_CONFIDENCE_THRESHOLD: float = 2.0  # Minimum score to assign a category


def _normalize_text(text: str) -> str:
    """
    Normalize text for keyword matching.

    Args:
        text: Raw text content

    Returns:
        Lowercase text with extra whitespace removed
    """
    if not text:
        return ""
    # Convert to lowercase and normalize whitespace
    return re.sub(r'\s+', ' ', text.lower().strip())


def _extract_url_path(url: str) -> str:
    """
    Extract and normalize the path from a URL.

    Args:
        url: Full URL string

    Returns:
        Lowercase URL path
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        return parsed.path.lower()
    except Exception:
        return url.lower()


def _calculate_url_score(url: str, category: str) -> float:
    """
    Calculate URL pattern match score for a category.

    Args:
        url: The page URL
        category: Category to check against

    Returns:
        Score based on URL pattern matches
    """
    if not url or category not in URL_PATTERNS:
        return 0.0

    path = _extract_url_path(url)
    patterns = URL_PATTERNS[category]

    score = 0.0
    for pattern in patterns:
        if pattern.lower() in path:
            score += URL_PATTERN_WEIGHT
            # Exact path match gets bonus
            if path == pattern.lower() or path == f"{pattern.lower()}/":
                score += URL_PATTERN_WEIGHT * 0.5
            break  # Only count once per category

    return score


def _calculate_keyword_score(
    text: str,
    title: str,
    category: str
) -> Tuple[float, int]:
    """
    Calculate keyword match score for a category.

    Args:
        text: Page text content
        title: Page title
        category: Category to check against

    Returns:
        Tuple of (score, match_count)
    """
    if category not in CATEGORY_KEYWORDS:
        return 0.0, 0

    keywords = CATEGORY_KEYWORDS[category]
    normalized_text = _normalize_text(text)
    normalized_title = _normalize_text(title)

    score = 0.0
    match_count = 0

    for i, keyword in enumerate(keywords):
        keyword_lower = keyword.lower()

        # Check title (higher weight)
        if keyword_lower in normalized_title:
            # Keywords earlier in list get slightly higher weight
            position_weight = 1.0 + (0.1 * (len(keywords) - i) / len(keywords))
            score += KEYWORD_BASE_WEIGHT * TITLE_MULTIPLIER * position_weight
            match_count += 1

        # Check body text
        if keyword_lower in normalized_text:
            # Count occurrences (capped to avoid gaming)
            occurrences = min(normalized_text.count(keyword_lower), 5)
            position_weight = 1.0 + (0.1 * (len(keywords) - i) / len(keywords))
            score += KEYWORD_BASE_WEIGHT * occurrences * position_weight * 0.5
            match_count += 1

    return score, match_count


def _calculate_category_scores(
    url: str,
    title: str,
    text_content: str
) -> Dict[str, float]:
    """
    Calculate scores for all categories.

    Args:
        url: Page URL
        title: Page title
        text_content: Page text content

    Returns:
        Dictionary mapping category names to scores
    """
    scores: Dict[str, float] = {}

    for category in CATEGORY_KEYWORDS.keys():
        url_score = _calculate_url_score(url, category)
        keyword_score, _ = _calculate_keyword_score(text_content, title, category)
        scores[category] = url_score + keyword_score

    return scores


def categorize_page(
    url: str,
    title: str,
    text_content: str
) -> str:
    """
    Categorize a single web page into a narrative category.

    Uses a combination of URL pattern matching and keyword density analysis
    to determine the most appropriate category for the page content.

    Args:
        url: The page URL (used for pattern matching)
        title: The page title (keywords here are weighted higher)
        text_content: The main text content of the page

    Returns:
        Category string, one of:
        - mission_vision
        - problem_framing
        - solution_framing
        - impact_stories
        - organizational_voice
        - donor_appeals
        - advocacy
        - general (fallback)

    Examples:
        >>> categorize_page(
        ...     "https://example.org/donate",
        ...     "Give Today",
        ...     "Your donation helps children worldwide."
        ... )
        'donor_appeals'

        >>> categorize_page(
        ...     "https://example.org/about",
        ...     "Our Mission",
        ...     "We exist to serve communities in need."
        ... )
        'mission_vision'
    """
    scores = _calculate_category_scores(url, title, text_content)

    # Find the highest scoring category
    if not scores:
        return "general"

    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]

    # Check if score meets minimum threshold
    if best_score < MIN_CONFIDENCE_THRESHOLD:
        return "general"

    return best_category


def categorize_page_with_confidence(
    url: str,
    title: str,
    text_content: str
) -> Dict[str, any]:
    """
    Categorize a page and return detailed scoring information.

    Args:
        url: The page URL
        title: The page title
        text_content: The main text content

    Returns:
        Dictionary containing:
        - category: The assigned category
        - confidence: Confidence score (0-1)
        - scores: All category scores
        - top_matches: Top 3 categories by score
    """
    scores = _calculate_category_scores(url, title, text_content)

    if not scores:
        return {
            "category": "general",
            "confidence": 0.0,
            "scores": {},
            "top_matches": []
        }

    # Sort categories by score
    sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_category, best_score = sorted_categories[0]

    # Calculate confidence (normalize based on score range)
    max_possible_score = 20.0  # Rough estimate of maximum achievable score
    confidence = min(best_score / max_possible_score, 1.0)

    # Determine final category
    if best_score < MIN_CONFIDENCE_THRESHOLD:
        final_category = "general"
        confidence = 0.0
    else:
        final_category = best_category

    return {
        "category": final_category,
        "confidence": round(confidence, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "top_matches": [
            {"category": cat, "score": round(score, 3)}
            for cat, score in sorted_categories[:3]
        ]
    }


def categorize_pages(pages: List[Dict]) -> List[Dict]:
    """
    Batch categorize multiple web pages.

    Processes a list of page dictionaries and adds category information
    to each one.

    Args:
        pages: List of page dictionaries, each containing:
            - url: Page URL (required)
            - title: Page title (optional, defaults to "")
            - text: Page text content (optional, defaults to "")

    Returns:
        Same list with added "category" field for each page.
        Original page data is preserved.

    Examples:
        >>> pages = [
        ...     {"url": "https://example.org/donate", "title": "Give", "text": "..."},
        ...     {"url": "https://example.org/about", "title": "About Us", "text": "..."}
        ... ]
        >>> categorized = categorize_pages(pages)
        >>> [p["category"] for p in categorized]
        ['donor_appeals', 'mission_vision']
    """
    categorized_pages: List[Dict] = []

    for page in pages:
        # Extract page data with defaults
        url = page.get("url", "")
        title = page.get("title", "")
        text = page.get("text", "")

        # Categorize the page
        category = categorize_page(url, title, text)

        # Create new dict with category added
        categorized_page = {**page, "category": category}
        categorized_pages.append(categorized_page)

    return categorized_pages


def categorize_pages_with_confidence(pages: List[Dict]) -> List[Dict]:
    """
    Batch categorize pages with detailed confidence information.

    Args:
        pages: List of page dictionaries (same format as categorize_pages)

    Returns:
        List with added categorization details for each page:
        - category: Assigned category
        - category_confidence: Confidence score
        - category_scores: All category scores
    """
    categorized_pages: List[Dict] = []

    for page in pages:
        url = page.get("url", "")
        title = page.get("title", "")
        text = page.get("text", "")

        result = categorize_page_with_confidence(url, title, text)

        categorized_page = {
            **page,
            "category": result["category"],
            "category_confidence": result["confidence"],
            "category_scores": result["scores"]
        }
        categorized_pages.append(categorized_page)

    return categorized_pages


def get_category_summary(pages: List[Dict]) -> Dict[str, int]:
    """
    Get a summary count of pages by category.

    Args:
        pages: List of categorized page dictionaries

    Returns:
        Dictionary mapping category names to page counts
    """
    summary: Dict[str, int] = {}

    for page in pages:
        category = page.get("category", "general")
        summary[category] = summary.get(category, 0) + 1

    return summary


def get_all_categories() -> List[str]:
    """
    Get list of all available categories.

    Returns:
        List of category names including 'general' fallback
    """
    return list(CATEGORY_KEYWORDS.keys()) + ["general"]


def get_category_keywords(category: str) -> List[str]:
    """
    Get keywords for a specific category.

    Args:
        category: Category name

    Returns:
        List of keywords, or empty list if category not found
    """
    return CATEGORY_KEYWORDS.get(category, [])


def get_category_url_patterns(category: str) -> List[str]:
    """
    Get URL patterns for a specific category.

    Args:
        category: Category name

    Returns:
        List of URL patterns, or empty list if category not found
    """
    return URL_PATTERNS.get(category, [])


# Convenience exports
__all__ = [
    "categorize_page",
    "categorize_page_with_confidence",
    "categorize_pages",
    "categorize_pages_with_confidence",
    "get_category_summary",
    "get_all_categories",
    "get_category_keywords",
    "get_category_url_patterns",
    "CATEGORY_KEYWORDS",
    "URL_PATTERNS",
]
