"""
Browser Scraper using Playwright

Provides browser automation for scraping content from websites
that require JavaScript rendering, including social media platforms.
"""

import asyncio
import re
from typing import Optional
from dataclasses import dataclass
from playwright.async_api import async_playwright, Browser, Page


@dataclass
class ScrapedContent:
    """Result of a scraping operation."""
    url: str
    title: str
    text_content: str
    success: bool
    error: Optional[str] = None
    platform: Optional[str] = None


# Platform detection patterns
PLATFORM_PATTERNS = {
    'instagram': r'instagram\.com',
    'twitter': r'(twitter\.com|x\.com)',
    'linkedin': r'linkedin\.com',
    'facebook': r'facebook\.com',
    'tiktok': r'tiktok\.com',
    'youtube': r'youtube\.com',
}


def detect_platform(url: str) -> Optional[str]:
    """Detect which social media platform a URL belongs to."""
    for platform, pattern in PLATFORM_PATTERNS.items():
        if re.search(pattern, url, re.IGNORECASE):
            return platform
    return None


async def scrape_with_browser(
    url: str,
    wait_time: int = 3000,
    scroll: bool = True
) -> ScrapedContent:
    """
    Scrape a URL using a headless browser.

    Args:
        url: The URL to scrape
        wait_time: Time to wait for JS to render (ms)
        scroll: Whether to scroll to load more content

    Returns:
        ScrapedContent with the extracted text
    """
    platform = detect_platform(url)
    print(f"[BROWSER_SCRAPER] Starting scrape for URL: {url}")
    print(f"[BROWSER_SCRAPER] Detected platform: {platform}")

    try:
        async with async_playwright() as p:
            # Launch browser with stealth settings
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--no-sandbox',
                ]
            )

            # Create context with realistic settings
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                locale='en-US',
            )

            page = await context.new_page()

            # Navigate to URL
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=30000)
            except Exception as e:
                await browser.close()
                return ScrapedContent(
                    url=url,
                    title="",
                    text_content="",
                    success=False,
                    error=f"Failed to load page: {str(e)}",
                    platform=platform
                )

            # Wait for content to render
            await page.wait_for_timeout(wait_time)

            # Scroll to load dynamic content
            if scroll:
                for _ in range(3):
                    await page.evaluate('window.scrollBy(0, window.innerHeight)')
                    await page.wait_for_timeout(500)

            # Get page title
            title = await page.title()

            # Extract content based on platform
            text_content = await extract_content(page, platform)

            await browser.close()

            if not text_content.strip():
                print(f"[BROWSER_SCRAPER] No content extracted from {url}")
                return ScrapedContent(
                    url=url,
                    title=title,
                    text_content="",
                    success=False,
                    error="No content could be extracted. Page may require login.",
                    platform=platform
                )

            print(f"[BROWSER_SCRAPER] SUCCESS - Extracted {len(text_content)} chars from {url}")
            return ScrapedContent(
                url=url,
                title=title,
                text_content=text_content,
                success=True,
                platform=platform
            )

    except Exception as e:
        return ScrapedContent(
            url=url,
            title="",
            text_content="",
            success=False,
            error=str(e),
            platform=platform
        )


async def extract_content(page: Page, platform: Optional[str]) -> str:
    """Extract text content from page based on platform."""

    if platform == 'instagram':
        return await extract_instagram(page)
    elif platform == 'twitter':
        return await extract_twitter(page)
    elif platform == 'linkedin':
        return await extract_linkedin(page)
    elif platform == 'facebook':
        return await extract_facebook(page)
    elif platform == 'tiktok':
        return await extract_tiktok(page)
    else:
        return await extract_generic(page)


async def extract_instagram(page: Page) -> str:
    """Extract content from Instagram pages."""
    content_parts = []

    # Try to get bio
    try:
        bio = await page.query_selector('header section span')
        if bio:
            bio_text = await bio.inner_text()
            if bio_text:
                content_parts.append(f"Bio: {bio_text}")
    except:
        pass

    # Try to get post captions
    try:
        # Look for article elements (posts)
        articles = await page.query_selector_all('article')
        for article in articles[:10]:  # First 10 posts
            try:
                # Get caption text
                caption = await article.query_selector('span[class*="Caption"]')
                if caption:
                    text = await caption.inner_text()
                    if text:
                        content_parts.append(f"Post: {text[:500]}")
            except:
                pass
    except:
        pass

    # Fallback: get all visible text
    if not content_parts:
        try:
            body_text = await page.inner_text('body')
            # Clean up and limit
            lines = [l.strip() for l in body_text.split('\n') if l.strip() and len(l.strip()) > 20]
            content_parts = lines[:30]
        except:
            pass

    return '\n\n'.join(content_parts)


async def extract_twitter(page: Page) -> str:
    """Extract content from Twitter/X pages."""
    content_parts = []

    # Try to get tweets
    try:
        tweets = await page.query_selector_all('[data-testid="tweet"]')
        for tweet in tweets[:15]:  # First 15 tweets
            try:
                text_elem = await tweet.query_selector('[data-testid="tweetText"]')
                if text_elem:
                    text = await text_elem.inner_text()
                    if text:
                        content_parts.append(f"Tweet: {text}")
            except:
                pass
    except:
        pass

    # Try to get bio
    try:
        bio = await page.query_selector('[data-testid="UserDescription"]')
        if bio:
            bio_text = await bio.inner_text()
            if bio_text:
                content_parts.insert(0, f"Bio: {bio_text}")
    except:
        pass

    # Fallback
    if not content_parts:
        try:
            body_text = await page.inner_text('main')
            lines = [l.strip() for l in body_text.split('\n') if l.strip() and len(l.strip()) > 20]
            content_parts = lines[:30]
        except:
            pass

    return '\n\n'.join(content_parts)


async def extract_linkedin(page: Page) -> str:
    """Extract content from LinkedIn pages."""
    content_parts = []

    # Try to get about section
    try:
        about = await page.query_selector('[class*="about"]')
        if about:
            text = await about.inner_text()
            if text:
                content_parts.append(f"About: {text[:1000]}")
    except:
        pass

    # Try to get posts
    try:
        posts = await page.query_selector_all('[class*="feed-shared-update"]')
        for post in posts[:10]:
            try:
                text = await post.inner_text()
                if text:
                    content_parts.append(f"Post: {text[:500]}")
            except:
                pass
    except:
        pass

    # Fallback
    if not content_parts:
        try:
            body_text = await page.inner_text('main')
            lines = [l.strip() for l in body_text.split('\n') if l.strip() and len(l.strip()) > 20]
            content_parts = lines[:30]
        except:
            pass

    return '\n\n'.join(content_parts)


async def extract_facebook(page: Page) -> str:
    """Extract content from Facebook pages."""
    content_parts = []

    # Generic extraction for Facebook
    try:
        # Get main content area
        main = await page.query_selector('[role="main"]')
        if main:
            text = await main.inner_text()
            lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 20]
            content_parts = lines[:40]
    except:
        pass

    if not content_parts:
        try:
            body_text = await page.inner_text('body')
            lines = [l.strip() for l in body_text.split('\n') if l.strip() and len(l.strip()) > 20]
            content_parts = lines[:30]
        except:
            pass

    return '\n\n'.join(content_parts)


async def extract_tiktok(page: Page) -> str:
    """Extract content from TikTok pages."""
    content_parts = []

    # Try to get bio
    try:
        bio = await page.query_selector('[data-e2e="user-bio"]')
        if bio:
            text = await bio.inner_text()
            if text:
                content_parts.append(f"Bio: {text}")
    except:
        pass

    # Try to get video descriptions
    try:
        videos = await page.query_selector_all('[data-e2e="user-post-item-desc"]')
        for video in videos[:10]:
            try:
                text = await video.inner_text()
                if text:
                    content_parts.append(f"Video: {text}")
            except:
                pass
    except:
        pass

    # Fallback
    if not content_parts:
        try:
            body_text = await page.inner_text('main')
            lines = [l.strip() for l in body_text.split('\n') if l.strip() and len(l.strip()) > 15]
            content_parts = lines[:30]
        except:
            pass

    return '\n\n'.join(content_parts)


async def extract_generic(page: Page) -> str:
    """Generic content extraction for any website."""
    content_parts = []

    # Try main content areas
    selectors = ['main', 'article', '[role="main"]', '.content', '#content', '.main']

    for selector in selectors:
        try:
            elem = await page.query_selector(selector)
            if elem:
                text = await elem.inner_text()
                if text and len(text) > 100:
                    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 20]
                    return '\n\n'.join(lines[:50])
        except:
            pass

    # Fallback to body
    try:
        body_text = await page.inner_text('body')
        lines = [l.strip() for l in body_text.split('\n') if l.strip() and len(l.strip()) > 20]
        return '\n\n'.join(lines[:50])
    except:
        return ""


# Test function
async def test_scraper():
    """Test the scraper with a sample URL."""
    url = "https://www.example.com"
    print(f"Scraping {url}...")
    result = await scrape_with_browser(url)
    print(f"Success: {result.success}")
    print(f"Title: {result.title}")
    print(f"Content preview: {result.text_content[:500]}...")
    if result.error:
        print(f"Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(test_scraper())
