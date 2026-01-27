"""
Narrative Intelligence API Server

FastAPI backend for the Narrative Intelligence Dashboard.
Provides REST endpoints and WebSocket streaming for real-time analysis.
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator


# =============================================================================
# URL Validation Helper
# =============================================================================

def is_valid_url(url: str) -> bool:
    """
    Validate URL format and scheme.
    Returns True if URL is valid http/https URL, False otherwise.
    """
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url.strip())
        # Must have valid scheme and netloc
        if parsed.scheme not in ('http', 'https'):
            return False
        if not parsed.netloc:
            return False
        # Basic domain validation - must have at least one dot or be localhost
        if '.' not in parsed.netloc and 'localhost' not in parsed.netloc.lower():
            return False
        return True
    except Exception:
        return False

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from narrative_analysis import (
    fetch_narrative_source,
    build_narrative_profile,
    get_narrative_suggestions,
    ContentExtractor,
    ProfileBuilder,
    SuggestionEngine,
    SourceType,
)
from observatory_client import ObservatoryClient
from claude_orchestrator import analyze_with_claude, generate_content, get_content_types, AnalysisUpdate
from browser_scraper import scrape_with_browser, detect_platform


# =============================================================================
# Models
# =============================================================================

class AnalyzeRequest(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None
    source_type: Optional[str] = "auto"
    max_items: int = 100
    include_force_analysis: bool = True

    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        if v is not None and not is_valid_url(v):
            raise ValueError('Invalid URL format. Must be a valid http/https URL.')
        return v


class SuggestRequest(BaseModel):
    profile: dict
    intent: str = "understand"


class CampaignRequest(BaseModel):
    profile: dict
    theme: str = "community"  # community, legacy, urgency, justice


# =============================================================================
# App Setup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("🚀 Narrative Intelligence API starting...")
    yield
    print("👋 Narrative Intelligence API shutting down...")


app = FastAPI(
    title="Narrative Intelligence API",
    description="Analyze narratives from URLs and text using the Cultural Soliton Observatory",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for local development
# Note: allow_credentials=True requires specific origins (not "*")
# Using "*" without credentials for broader compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Cannot use True with allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REST Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Narrative Intelligence API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    # Check observatory backend
    try:
        async with ObservatoryClient() as client:
            status = await client.health_check()
            observatory_status = "connected"
    except Exception as e:
        observatory_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "observatory": observatory_status,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Analyze a URL or text and return a narrative profile.

    This is the synchronous version - for real-time progress,
    use the WebSocket endpoint /ws/analyze
    """
    if not request.url and not request.text:
        raise HTTPException(status_code=400, detail="Either 'url' or 'text' is required")

    try:
        if request.url:
            # Fetch content from URL
            source_result = await fetch_narrative_source(
                url=request.url,
                source_type=request.source_type if request.source_type != "auto" else None,
                max_items=request.max_items,
            )

            content_units = source_result["content_units"]
            source = source_result["source"]
            source_type = source_result["source_type"]
        else:
            # Use provided text directly
            content_units = [{"text": request.text}]
            source = "direct_input"
            source_type = "raw_text"

        # Build profile
        profile = await build_narrative_profile(
            content_units=content_units,
            source=source,
            source_type=source_type,
            include_force_analysis=request.include_force_analysis,
        )

        # Generate suggestions
        suggestions = get_narrative_suggestions(profile, intent="understand")

        return {
            "success": True,
            "profile": profile,
            "suggestions": suggestions,
            "content_units_count": len(content_units),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/suggest")
async def suggest(request: SuggestRequest):
    """Generate suggestions for a given profile."""
    try:
        suggestions = get_narrative_suggestions(
            profile_dict=request.profile,
            intent=request.intent,
        )
        return {
            "success": True,
            "suggestions": suggestions,
            "intent": request.intent,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/campaign")
async def generate_campaign(request: CampaignRequest):
    """Generate campaign copy based on profile analysis."""
    profile = request.profile
    theme = request.theme

    # Campaign templates based on profile and theme
    centroid = profile.get("centroid", {})
    dominant_mode = profile.get("dominant_mode", "NEUTRAL")
    attractors = profile.get("attractors", [])

    campaigns = {
        "community": {
            "headline": "Join thousands who believe every child deserves a family.",
            "subhead": "We are a movement of ordinary people doing extraordinary things together.",
            "body": "This isn't charity. This is a revolution. When we unite, change happens.",
            "cta": "Join the movement",
        },
        "legacy": {
            "headline": "Your love will outlive you.",
            "subhead": "Create impact that echoes through generations.",
            "body": "Long after we're gone, the lives we touched will raise families of their own.",
            "cta": "Plant your legacy",
        },
        "urgency": {
            "headline": "You decide their future.",
            "subhead": "Your choice today changes everything for one child.",
            "body": "Every day matters. Right now, you hold the power to transform a life.",
            "cta": "Take action now",
        },
        "justice": {
            "headline": "We're not just helping—we're fixing a broken system.",
            "subhead": "Research is clear. Family is better. We're making it happen.",
            "body": "For too long, good intentions funded the wrong solution. We're building something better.",
            "cta": "Be part of the solution",
        },
    }

    campaign = campaigns.get(theme, campaigns["community"])

    # Customize based on profile
    if centroid.get("agency", 0) > 0.3:
        campaign["cta"] = "Make your move"
    if centroid.get("belonging", 0) > 0.5:
        campaign["subhead"] = "Join our family of changemakers. " + campaign["subhead"]

    return {
        "success": True,
        "campaign": campaign,
        "theme": theme,
        "optimized_for": {
            "dominant_mode": dominant_mode,
            "top_attractor": attractors[0]["target"] if attractors else None,
        },
    }


# =============================================================================
# WebSocket for Real-time Streaming (Claude-Orchestrated)
# =============================================================================

@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """
    WebSocket endpoint for Claude-orchestrated narrative analysis.

    Claude actually decides which tools to use, interprets results,
    and generates genuine insights - not just executing a fixed pipeline.

    Send: {"url": "https://example.com"} or {"text": "...", "intent": "understand"}
    Receive: Real-time updates as Claude thinks, calls tools, and generates insights
    """
    await websocket.accept()

    try:
        while True:
            # Receive analysis request
            data = await websocket.receive_json()
            url = data.get("url")
            social_media_url = data.get("social_media_url")
            social_media_content = data.get("social_media_content")  # Pasted text content
            document_content = data.get("document_content")
            text = data.get("text")
            intent = data.get("intent", "understand")
            objective = data.get("objective", "leads")
            brand_voice = data.get("brand_voice", "professional")
            brand_personality = data.get("brand_personality", "")
            target_audience = data.get("target_audience", "")

            # Check if we have at least one data source
            has_any_input = url or social_media_url or social_media_content or document_content or text
            if not has_any_input:
                await websocket.send_json({
                    "type": "error",
                    "message": "At least one data source is required (URL, social media, documents, or text)",
                })
                continue

            # Validate URLs if provided
            if url and not is_valid_url(url):
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid website URL format. Must be a valid http/https URL.",
                })
                continue

            if social_media_url and not is_valid_url(social_media_url):
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid social media URL format. Must be a valid http/https URL.",
                })
                continue

            # Build combined input for analysis
            input_parts = []
            input_type = "multi_source"

            if url:
                input_parts.append(f"PRIMARY_URL: {url}")

            # Handle social media URL - use Playwright to scrape
            if social_media_url:
                platform = detect_platform(social_media_url)
                if platform:
                    await websocket.send_json({
                        "type": "thinking",
                        "step": "browser_scraping",
                        "message": f"Using browser automation to scrape {platform}...",
                        "progress": 15,
                    })

                    # Use Playwright to scrape the social media page
                    scraped = await scrape_with_browser(social_media_url)

                    if scraped.success and scraped.text_content:
                        input_parts.append(f"SOCIAL_MEDIA_CONTENT ({platform.upper()}):\n{scraped.text_content}")
                        await websocket.send_json({
                            "type": "tool_result",
                            "step": "browser_complete",
                            "message": f"Successfully extracted content from {platform}",
                            "progress": 25,
                        })
                    else:
                        input_parts.append(f"SOCIAL_MEDIA_URL: {social_media_url}\n(Browser scraping failed: {scraped.error})")
                        await websocket.send_json({
                            "type": "thinking",
                            "step": "browser_failed",
                            "message": f"Could not scrape {platform}: {scraped.error}. Will try standard fetch.",
                            "progress": 20,
                        })
                else:
                    # Not a recognized social platform, just pass the URL
                    input_parts.append(f"SOCIAL_MEDIA_URL: {social_media_url}")

            if social_media_content:
                input_parts.append(f"SOCIAL_MEDIA_CONTENT:\n{social_media_content}")
            if document_content:
                input_parts.append(f"DOCUMENTS:\n{document_content}")
            if text:
                input_parts.append(f"TEXT_CONTENT:\n{text}")

            input_text = "\n\n---\n\n".join(input_parts)

            try:
                # Stream Claude's analysis in real-time
                progress_counter = 0
                async for update in analyze_with_claude(input_text, input_type, intent, objective, brand_voice, brand_personality, target_audience):
                    progress_counter += 1

                    if update.type == "thinking":
                        await websocket.send_json({
                            "type": "thinking",
                            "step": "claude_thinking",
                            "message": update.content,
                            "progress": min(10 + progress_counter * 5, 30),
                        })

                    elif update.type == "tool_call":
                        await websocket.send_json({
                            "type": "tool_call",
                            "step": "using_tool",
                            "message": update.content,
                            "tool": update.data.get("tool") if update.data else None,
                            "tool_input": update.data.get("input") if update.data else None,
                            "progress": min(30 + progress_counter * 8, 75),
                        })

                    elif update.type == "tool_result":
                        await websocket.send_json({
                            "type": "tool_result",
                            "step": "tool_complete",
                            "message": update.content,
                            "tool": update.data.get("tool") if update.data else None,
                            "result_preview": update.data.get("result_preview") if update.data else None,
                            "progress": min(35 + progress_counter * 8, 80),
                        })

                    elif update.type == "complete":
                        # Parse the response to extract structured data if possible
                        await websocket.send_json({
                            "type": "complete",
                            "step": "done",
                            "message": "Analysis complete",
                            "progress": 100,
                            "claude_response": update.content,
                            "usage": update.data.get("usage") if update.data else None,
                        })

                    elif update.type == "error":
                        await websocket.send_json({
                            "type": "error",
                            "message": update.content,
                        })

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Analysis failed: {str(e)}",
                })

    except WebSocketDisconnect:
        print("WebSocket client disconnected")


# =============================================================================
# Content Generation WebSocket
# =============================================================================

@app.get("/content-types")
async def list_content_types():
    """List available content types for generation."""
    return {
        "content_types": get_content_types()
    }


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """
    WebSocket endpoint for generating specific content using analysis as context.

    Send: {"content_type": "twitter", "analysis_context": "...", "custom_instructions": "..."}
    Receive: Generated content
    """
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            content_type = data.get("content_type")
            analysis_context = data.get("analysis_context", "")
            custom_instructions = data.get("custom_instructions", "")
            variations = data.get("variations", False)

            if not content_type:
                await websocket.send_json({
                    "type": "error",
                    "message": "content_type is required"
                })
                continue

            if not analysis_context:
                await websocket.send_json({
                    "type": "error",
                    "message": "analysis_context is required"
                })
                continue

            try:
                async for update in generate_content(content_type, analysis_context, custom_instructions, variations):
                    if update.type == "thinking":
                        await websocket.send_json({
                            "type": "thinking",
                            "message": update.content
                        })
                    elif update.type == "complete":
                        await websocket.send_json({
                            "type": "complete",
                            "content": update.content,
                            "content_type": update.data.get("content_type") if update.data else None,
                            "usage": update.data.get("usage") if update.data else None
                        })
                    elif update.type == "error":
                        await websocket.send_json({
                            "type": "error",
                            "message": update.content
                        })

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Generation failed: {str(e)}"
                })

    except WebSocketDisconnect:
        print("WebSocket client disconnected")


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )
