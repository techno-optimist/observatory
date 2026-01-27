"""
Observer API - URL Scraping and Text Extraction

This module provides endpoints for:
- Fetching webpage content from URLs
- Extracting readable text (strips HTML tags, scripts, styles)
- Returning extracted text for analysis

Usage:
    from api_observer import router as observer_router
    app.include_router(observer_router, prefix="/api/observer", tags=["observer"])

Endpoints:
- POST /api/observer/fetch-url: Fetch and extract text from a URL
"""

import re
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, HttpUrl

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class FetchUrlRequest(BaseModel):
    """Request to fetch and extract text from a URL."""
    url: str = Field(
        ...,
        description="The URL to fetch and extract text from",
        examples=["https://example.com/article"]
    )
    max_length: Optional[int] = Field(
        default=None,
        description="Maximum length of extracted text (characters). None for no limit.",
        ge=100
    )
    include_title: bool = Field(
        default=True,
        description="Whether to extract the page title"
    )
    timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds",
        ge=5.0,
        le=120.0
    )


class FetchUrlResponse(BaseModel):
    """Response containing extracted text from a URL."""
    url: str = Field(..., description="The URL that was fetched")
    text: str = Field(..., description="Extracted readable text content")
    title: Optional[str] = Field(None, description="Page title if available")
    word_count: int = Field(..., description="Number of words in extracted text")
    char_count: int = Field(..., description="Number of characters in extracted text")
    success: bool = Field(default=True, description="Whether extraction was successful")


# ============================================================================
# Text Extraction Utilities
# ============================================================================

def extract_title(html: str) -> Optional[str]:
    """Extract the page title from HTML."""
    # Try to find <title> tag
    title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = title_match.group(1)
        # Clean up the title
        title = re.sub(r'<[^>]+>', '', title)  # Remove any nested tags
        title = re.sub(r'\s+', ' ', title).strip()
        return title if title else None

    # Fallback: try to find <h1> tag
    h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.IGNORECASE | re.DOTALL)
    if h1_match:
        h1 = h1_match.group(1)
        h1 = re.sub(r'<[^>]+>', '', h1)
        h1 = re.sub(r'\s+', ' ', h1).strip()
        return h1 if h1 else None

    return None


def extract_readable_text(html: str) -> str:
    """
    Extract readable text from HTML, stripping tags, scripts, and styles.

    This function:
    1. Removes <script>, <style>, <nav>, <footer>, <header>, <aside> blocks
    2. Removes HTML comments
    3. Extracts text from remaining content
    4. Normalizes whitespace
    5. Returns clean, readable text
    """
    # Remove script blocks
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove style blocks
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove noscript blocks
    html = re.sub(r'<noscript[^>]*>.*?</noscript>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove navigation blocks
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove footer blocks
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove header blocks (often contains nav elements)
    html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove aside blocks (sidebars)
    html = re.sub(r'<aside[^>]*>.*?</aside>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

    # Remove SVG and canvas elements
    html = re.sub(r'<svg[^>]*>.*?</svg>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<canvas[^>]*>.*?</canvas>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove form elements
    html = re.sub(r'<form[^>]*>.*?</form>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Convert block-level elements to newlines for better text structure
    block_elements = ['p', 'div', 'article', 'section', 'main', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                      'li', 'tr', 'blockquote', 'pre', 'br', 'hr']
    for elem in block_elements:
        html = re.sub(rf'<{elem}[^>]*>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(rf'</{elem}>', '\n', html, flags=re.IGNORECASE)

    # Remove all remaining HTML tags
    html = re.sub(r'<[^>]+>', ' ', html)

    # Decode common HTML entities
    html_entities = {
        '&nbsp;': ' ',
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&apos;': "'",
        '&#39;': "'",
        '&#x27;': "'",
        '&ldquo;': '"',
        '&rdquo;': '"',
        '&lsquo;': "'",
        '&rsquo;': "'",
        '&mdash;': '-',
        '&ndash;': '-',
        '&hellip;': '...',
        '&copy;': '(c)',
        '&reg;': '(R)',
        '&trade;': '(TM)',
    }
    for entity, replacement in html_entities.items():
        html = html.replace(entity, replacement)

    # Handle numeric entities
    html = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))) if int(m.group(1)) < 65536 else '', html)
    html = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)) if int(m.group(1), 16) < 65536 else '', html)

    # Normalize whitespace
    # Replace multiple spaces with single space
    html = re.sub(r'[ \t]+', ' ', html)
    # Replace multiple newlines with double newline (paragraph break)
    html = re.sub(r'\n\s*\n+', '\n\n', html)
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in html.split('\n')]
    # Remove empty lines at start/end
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    # Join and final cleanup
    text = '\n'.join(lines)
    text = text.strip()

    return text


def count_words(text: str) -> int:
    """Count the number of words in text."""
    # Split on whitespace and filter out empty strings
    words = [w for w in text.split() if w]
    return len(words)


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/fetch-url", response_model=FetchUrlResponse)
async def fetch_url(request: FetchUrlRequest):
    """
    Fetch a webpage and extract readable text content.

    This endpoint:
    1. Fetches the HTML content from the provided URL
    2. Strips scripts, styles, navigation, and other non-content elements
    3. Extracts readable text from the remaining HTML
    4. Returns the text along with metadata (title, word count)

    The extracted text is suitable for further analysis by the Observatory.

    Example:
    ```json
    POST /api/observer/fetch-url
    {
        "url": "https://example.com/article"
    }
    ```

    Response:
    ```json
    {
        "url": "https://example.com/article",
        "text": "This is the article content...",
        "title": "Example Article",
        "word_count": 150,
        "char_count": 850,
        "success": true
    }
    ```
    """
    import httpx

    url = request.url

    # Validate URL format
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        # Fetch the webpage
        async with httpx.AsyncClient(
            timeout=request.timeout,
            follow_redirects=True,
            http2=True
        ) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; CulturalSolitonObservatory/1.0; +https://github.com/cultural-soliton-observatory)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "DNT": "1",
                "Connection": "keep-alive",
            }

            response = await client.get(url, headers=headers)
            response.raise_for_status()

            # Get the HTML content
            html = response.text

            # Check if we got HTML
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                logger.warning(f"URL returned non-HTML content type: {content_type}")
                # Still try to process it, might be HTML without proper content-type

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=408,
            detail=f"Request timed out after {request.timeout} seconds"
        )
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to {url}: {str(e)}"
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"HTTP error {e.response.status_code}: {e.response.reason_phrase}"
        )
    except Exception as e:
        logger.error(f"Failed to fetch URL {url}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch URL: {str(e)}"
        )

    # Extract title if requested
    title = None
    if request.include_title:
        title = extract_title(html)

    # Extract readable text
    text = extract_readable_text(html)

    # Apply max length if specified
    if request.max_length and len(text) > request.max_length:
        # Try to cut at a sentence boundary
        truncated = text[:request.max_length]
        # Find last sentence-ending punctuation
        last_period = max(
            truncated.rfind('. '),
            truncated.rfind('! '),
            truncated.rfind('? '),
            truncated.rfind('.\n'),
            truncated.rfind('!\n'),
            truncated.rfind('?\n')
        )
        if last_period > request.max_length * 0.7:  # Only use if we keep at least 70%
            text = truncated[:last_period + 1]
        else:
            # Fall back to word boundary
            last_space = truncated.rfind(' ')
            if last_space > request.max_length * 0.9:
                text = truncated[:last_space] + '...'
            else:
                text = truncated + '...'

    # Calculate word count
    word_count = count_words(text)
    char_count = len(text)

    # Log success
    logger.info(f"Successfully extracted {word_count} words from {url}")

    return FetchUrlResponse(
        url=url,
        text=text,
        title=title,
        word_count=word_count,
        char_count=char_count,
        success=True
    )


@router.post("/fetch-url/batch")
async def fetch_urls_batch(
    urls: list[str],
    max_length: Optional[int] = None,
    timeout: float = 30.0
):
    """
    Fetch and extract text from multiple URLs.

    Returns results for each URL, including any errors encountered.
    """
    import asyncio

    async def fetch_one(url: str) -> dict:
        try:
            request = FetchUrlRequest(
                url=url,
                max_length=max_length,
                timeout=timeout
            )
            result = await fetch_url(request)
            return result.model_dump()
        except HTTPException as e:
            return {
                "url": url,
                "text": "",
                "title": None,
                "word_count": 0,
                "char_count": 0,
                "success": False,
                "error": e.detail
            }
        except Exception as e:
            return {
                "url": url,
                "text": "",
                "title": None,
                "word_count": 0,
                "char_count": 0,
                "success": False,
                "error": str(e)
            }

    # Fetch all URLs concurrently
    results = await asyncio.gather(*[fetch_one(url) for url in urls])

    # Calculate summary statistics
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]

    return {
        "results": results,
        "summary": {
            "total": len(urls),
            "successful": len(successful),
            "failed": len(failed),
            "total_words": sum(r.get("word_count", 0) for r in successful),
            "total_chars": sum(r.get("char_count", 0) for r in successful)
        }
    }


@router.get("/status")
async def observer_status():
    """Get status of the observer API."""
    return {
        "status": "running",
        "version": "1.0.0",
        "endpoints": [
            {
                "path": "/api/observer/fetch-url",
                "method": "POST",
                "description": "Fetch and extract text from a URL"
            },
            {
                "path": "/api/observer/fetch-url/batch",
                "method": "POST",
                "description": "Fetch and extract text from multiple URLs"
            }
        ],
        "capabilities": [
            "HTML text extraction",
            "Script/style removal",
            "Title extraction",
            "Word counting",
            "Batch URL processing"
        ]
    }
