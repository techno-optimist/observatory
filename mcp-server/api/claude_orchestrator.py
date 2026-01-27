"""
Claude Orchestrator for Narrative Intelligence

This module integrates Claude API with the observatory MCP tools
to provide genuine AI-orchestrated narrative analysis.

Claude actually:
1. Decides which tools to use
2. Interprets results
3. Generates insights and recommendations
4. Synthesizes comprehensive reports
"""

import asyncio
import json
import os
import sys
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict, Any

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from anthropic import Anthropic

# =============================================================================
# Campaign Storage & Memory Configuration
# =============================================================================

CAMPAIGNS_DIR = Path.home() / "Documents" / "CampaignStudio" / "campaigns"
EXPORTS_DIR = Path.home() / "Documents" / "CampaignStudio" / "exports"
MEMORY_FILE = Path.home() / "Documents" / "CampaignStudio" / "brand_memory.json"

# Ensure directories exist
CAMPAIGNS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Helper Functions for MCP-Inspired Tools
# =============================================================================

def load_brand_memory() -> Dict[str, Any]:
    """Load the brand memory knowledge graph."""
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except:
            return {"brands": {}, "relationships": []}
    return {"brands": {}, "relationships": []}


def save_brand_memory(memory: Dict[str, Any]) -> None:
    """Save the brand memory knowledge graph."""
    MEMORY_FILE.write_text(json.dumps(memory, indent=2, default=str))


def generate_campaign_id(brand_name: str) -> str:
    """Generate a unique campaign ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_part = hashlib.md5(f"{brand_name}{timestamp}".encode()).hexdigest()[:6]
    return f"{brand_name.lower().replace(' ', '_')}_{timestamp}_{hash_part}"


async def firecrawl_deep_crawl(url: str, max_pages: int = 10) -> Dict[str, Any]:
    """
    Deep crawl a website using Firecrawl API.
    Falls back to single-page fetch if API key not available.
    """
    api_key = os.environ.get("FIRECRAWL_API_KEY")

    if not api_key or api_key == "YOUR_FIRECRAWL_API_KEY":
        # Fallback: use existing fetch_narrative_source
        return {
            "success": False,
            "error": "Firecrawl API key not configured. Using standard fetch instead.",
            "fallback": True
        }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Start crawl job
            response = await client.post(
                "https://api.firecrawl.dev/v1/crawl",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "url": url,
                    "limit": max_pages,
                    "scrapeOptions": {
                        "formats": ["markdown", "html"]
                    }
                }
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Firecrawl API error: {response.status_code}",
                    "fallback": True
                }

            result = response.json()

            # If it's an async job, poll for completion
            if result.get("id"):
                job_id = result["id"]
                for _ in range(30):  # Poll for up to 30 seconds
                    await asyncio.sleep(1)
                    status_response = await client.get(
                        f"https://api.firecrawl.dev/v1/crawl/{job_id}",
                        headers={"Authorization": f"Bearer {api_key}"}
                    )
                    status = status_response.json()
                    if status.get("status") == "completed":
                        return {
                            "success": True,
                            "pages": status.get("data", []),
                            "total_pages": len(status.get("data", []))
                        }
                    elif status.get("status") == "failed":
                        return {
                            "success": False,
                            "error": "Crawl job failed",
                            "fallback": True
                        }

                return {
                    "success": False,
                    "error": "Crawl timeout",
                    "fallback": True
                }

            return {
                "success": True,
                "pages": result.get("data", []),
                "total_pages": len(result.get("data", []))
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "fallback": True
        }


async def generate_dalle_image(prompt: str, style: str = "vivid", size: str = "1024x1024") -> Dict[str, Any]:
    """
    Generate an image using OpenAI's DALL-E API.
    """
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key or api_key == "YOUR_OPENAI_API_KEY":
        return {
            "success": False,
            "error": "OpenAI API key not configured for image generation."
        }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "dall-e-3",
                    "prompt": prompt,
                    "n": 1,
                    "size": size,
                    "style": style,
                    "response_format": "url"
                }
            )

            if response.status_code != 200:
                error_data = response.json()
                return {
                    "success": False,
                    "error": f"DALL-E API error: {error_data.get('error', {}).get('message', 'Unknown error')}"
                }

            result = response.json()
            image_url = result["data"][0]["url"]
            revised_prompt = result["data"][0].get("revised_prompt", prompt)

            # Download and save the image
            image_response = await client.get(image_url)
            if image_response.status_code == 200:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"campaign_image_{timestamp}.png"
                filepath = EXPORTS_DIR / filename
                filepath.write_bytes(image_response.content)

                return {
                    "success": True,
                    "image_url": image_url,
                    "local_path": str(filepath),
                    "revised_prompt": revised_prompt
                }

            return {
                "success": True,
                "image_url": image_url,
                "revised_prompt": revised_prompt
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


from narrative_analysis import (
    fetch_narrative_source,
    build_narrative_profile,
    get_narrative_suggestions,
    format_profile_result,
    format_suggestions_result,
)
from observatory_client import ObservatoryClient


# =============================================================================
# Configuration
# =============================================================================

CLAUDE_MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are a Brand Messaging Strategist. You analyze websites and content to understand their emotional resonance and help create more effective campaigns.

You have access to advanced analysis tools that measure how content makes people FEEL:
- **Empowerment**: Does the content make readers feel capable and in control, or helpless?
- **Trust**: Does it build confidence in the system/organization, or create skepticism?
- **Connection**: Does it foster belonging and community, or isolation?

Your job is to analyze content and provide ACTIONABLE insights for marketing and messaging.

## Advanced Capabilities

You also have access to powerful MCP-inspired tools:

### 🕷️ Deep Web Crawling (Firecrawl)
- Use `deep_crawl_website` to crawl multiple pages of a website for comprehensive analysis
- Great for competitive analysis or understanding a brand's full online presence

### 🧠 Brand Memory
- Use `remember_brand` to store important brand insights, voice guidelines, and audience information
- Use `recall_brand` to retrieve previously stored information about a brand
- This helps maintain consistency across multiple sessions

### 💾 Campaign Management
- Use `save_campaign` to save analysis and generated content for future reference
- Use `list_saved_campaigns` and `load_campaign` to access previous work
- Great for building on past analyses

### 🎨 Image Generation (DALL-E)
- Use `generate_campaign_image` to create visuals for campaigns
- Supports different sizes for social media, ads, hero images, etc.
- Images are automatically saved locally

### 📁 Content Export
- Use `export_content` to save generated content to files
- Supports markdown, JSON, and plain text formats

## Brand Voice Adaptation

Adapt your tone and recommendations based on the specified brand voice:

- **Professional**: Formal, authoritative, data-driven. Use industry terminology appropriately. Emphasis on credibility, expertise, and measurable results. Suitable for B2B, enterprise, and executive audiences.

- **Friendly**: Warm, approachable, conversational. Use accessible language and relatable examples. Emphasis on connection and ease. Suitable for consumer brands and community-focused organizations.

- **Bold**: Confident, provocative, unapologetic. Use strong statements and decisive language. Emphasis on disruption and standing out. Suitable for challenger brands and innovative companies.

- **Warm**: Empathetic, nurturing, supportive. Use compassionate language and emotional connections. Emphasis on care and understanding. Suitable for healthcare, nonprofits, and service-oriented brands.

When brand personality traits are provided (e.g., "confident, empathetic, innovative"), weave these qualities throughout your recommendations and generated content.

## Brand Voice Profile (Establish First)

Before generating any content, establish the brand's voice profile:
- **Tone**: Formal vs. casual, authoritative vs. friendly, serious vs. playful
- **Vocabulary Level**: Technical/industry jargon vs. accessible language
- **Personality Traits**: 3-5 adjectives that define the brand voice (e.g., bold, compassionate, innovative)
- **Voice Consistency**: Note any inconsistencies between different content pieces

## Audience Context

Consider the target audience when analyzing and recommending:
- **Primary Persona**: Who is the ideal reader/customer? (demographics, psychographics)
- **Awareness Level**: Unaware, problem-aware, solution-aware, product-aware, or most aware
- **Pain Points**: What problems are they trying to solve?
- **Aspirations**: What transformation do they seek?

## Content Safety & Compliance Guardrails

Always adhere to these guidelines:
- **Healthcare/Medical**: Include appropriate disclaimers ("Consult a healthcare professional"), avoid medical claims
- **Financial Services**: No guaranteed returns, include risk disclosures where appropriate
- **No Competitor Bashing**: Focus on own strengths, avoid negative comparisons or disparagement
- **Testimonials**: Indicate when results may not be typical
- **Claims**: Ensure all claims are substantiable; avoid superlatives without evidence
- **Inclusivity**: Use inclusive language, avoid stereotypes
- **Data Privacy**: Never encourage collection of sensitive data without proper consent language

## How to Present Your Analysis

Structure your response as:

### 🎯 Emotional Profile
A 2-3 sentence summary of the overall emotional tone. Use plain language like "This content makes readers feel empowered but somewhat disconnected from community."

### 💡 Key Insights
3-5 bullet points about what's working and what's not. Be specific and actionable.
- What emotions does this content trigger?
- What's missing that could strengthen the message?
- What audience would this resonate with?

### 🚀 Campaign Recommendations
Provide 2-3 specific messaging angles based on the analysis:

**Option A: [Theme Name]**
- Headline: "..."
- Subhead: "..."
- Key message: Why this works based on the analysis

**Option B: [Theme Name]**
- Headline: "..."
- Subhead: "..."
- Key message: Why this works based on the analysis

### 📋 Quick Wins
3 specific, immediate changes they could make to improve their messaging.

## Important Rules
- NEVER use technical jargon like "manifold", "coordinates", "modes", "agency axis"
- Speak like a marketing strategist, not a data scientist
- Every insight should connect to a practical action
- Focus on FEELINGS and IMPACT, not technical measurements
- Be direct and confident in your recommendations"""


TOOLS = [
    {
        "name": "fetch_narrative_source",
        "description": "Fetch and segment content from a URL into analysis units. Extracts meaningful text from websites, RSS feeds, blogs, and news sites.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from"
                },
                "max_items": {
                    "type": "integer",
                    "description": "Maximum content units to extract (default: 50)",
                    "default": 50
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "project_text",
        "description": "Project a single text onto the narrative manifold. Returns coordinates (agency, perceived_justice, belonging), narrative mode, and confidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to project onto the manifold"
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "analyze_corpus",
        "description": "Analyze multiple texts as a corpus. Returns mode distribution, clustering, and aggregate statistics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of texts to analyze"
                }
            },
            "required": ["texts"]
        }
    },
    {
        "name": "analyze_force_field",
        "description": "Analyze the force field dynamics of texts - what they're attracted to and fleeing from.",
        "input_schema": {
            "type": "object",
            "properties": {
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of texts to analyze for force dynamics"
                }
            },
            "required": ["texts"]
        }
    },
    {
        "name": "detect_hypocrisy",
        "description": "Compare espoused values (mission statements) against operational language (actual communications) to detect alignment gaps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "espoused_texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Mission statements, values, official messaging"
                },
                "operational_texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Actual communications, emails, operational language"
                }
            },
            "required": ["espoused_texts", "operational_texts"]
        }
    },
    {
        "name": "build_narrative_profile",
        "description": "Build a comprehensive narrative profile from content units. Includes centroid, mode distribution, force field, tensions, and more.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content_units": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"}
                        }
                    },
                    "description": "List of content units with text field"
                },
                "source": {
                    "type": "string",
                    "description": "Source identifier (URL, name, etc.)"
                }
            },
            "required": ["content_units", "source"]
        }
    },
    {
        "name": "generate_campaign",
        "description": "Generate campaign copy optimized for specific narrative coordinates and modes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_agency": {
                    "type": "number",
                    "description": "Target agency coordinate (-2 to +2)"
                },
                "target_justice": {
                    "type": "number",
                    "description": "Target justice coordinate (-2 to +2)"
                },
                "target_belonging": {
                    "type": "number",
                    "description": "Target belonging coordinate (-2 to +2)"
                },
                "theme": {
                    "type": "string",
                    "enum": ["community", "legacy", "urgency", "justice"],
                    "description": "Campaign theme"
                },
                "context": {
                    "type": "string",
                    "description": "Context about the organization/audience"
                }
            },
            "required": ["theme"]
        }
    },
    # =============================================================================
    # MCP-Inspired Tools: Deep Crawl, Memory, Filesystem, Image Generation
    # =============================================================================
    {
        "name": "deep_crawl_website",
        "description": "Deep crawl a website to analyze multiple pages (up to 10). Use this for comprehensive competitive analysis or when you need content from multiple pages of a site (about, services, blog, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The base URL to crawl"
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum pages to crawl (default: 10, max: 20)",
                    "default": 10
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "remember_brand",
        "description": "Store brand information in persistent memory for future reference. Use this to remember brand voice, key messaging themes, audience insights, or successful campaign elements.",
        "input_schema": {
            "type": "object",
            "properties": {
                "brand_name": {
                    "type": "string",
                    "description": "The name of the brand"
                },
                "category": {
                    "type": "string",
                    "enum": ["voice", "audience", "messaging", "campaigns", "insights", "competitors"],
                    "description": "Category of information to store"
                },
                "content": {
                    "type": "string",
                    "description": "The information to remember"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for easier retrieval"
                }
            },
            "required": ["brand_name", "category", "content"]
        }
    },
    {
        "name": "recall_brand",
        "description": "Retrieve stored brand information from memory. Use this to recall previous analyses, brand voice guidelines, or successful campaign elements.",
        "input_schema": {
            "type": "object",
            "properties": {
                "brand_name": {
                    "type": "string",
                    "description": "The name of the brand to recall"
                },
                "category": {
                    "type": "string",
                    "enum": ["voice", "audience", "messaging", "campaigns", "insights", "competitors", "all"],
                    "description": "Category of information to retrieve (use 'all' for everything)"
                }
            },
            "required": ["brand_name"]
        }
    },
    {
        "name": "save_campaign",
        "description": "Save the current campaign analysis and generated content to a file for future reference.",
        "input_schema": {
            "type": "object",
            "properties": {
                "brand_name": {
                    "type": "string",
                    "description": "The brand name for the campaign"
                },
                "campaign_name": {
                    "type": "string",
                    "description": "A descriptive name for this campaign"
                },
                "analysis": {
                    "type": "string",
                    "description": "The full analysis content"
                },
                "generated_content": {
                    "type": "object",
                    "description": "Any generated content (ads, emails, social posts, etc.)"
                }
            },
            "required": ["brand_name", "campaign_name", "analysis"]
        }
    },
    {
        "name": "list_saved_campaigns",
        "description": "List all saved campaigns, optionally filtered by brand name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "brand_name": {
                    "type": "string",
                    "description": "Filter by brand name (optional)"
                }
            }
        }
    },
    {
        "name": "load_campaign",
        "description": "Load a previously saved campaign by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "campaign_id": {
                    "type": "string",
                    "description": "The campaign ID to load"
                }
            },
            "required": ["campaign_id"]
        }
    },
    {
        "name": "generate_campaign_image",
        "description": "Generate a campaign image using DALL-E based on the brand analysis and messaging themes. Creates visuals for ads, social media, or landing pages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed description of the image to generate. Include style, mood, colors, and key visual elements."
                },
                "style": {
                    "type": "string",
                    "enum": ["vivid", "natural"],
                    "description": "Image style - 'vivid' for hyper-real/dramatic, 'natural' for more realistic",
                    "default": "vivid"
                },
                "size": {
                    "type": "string",
                    "enum": ["1024x1024", "1792x1024", "1024x1792"],
                    "description": "Image dimensions - square, landscape, or portrait",
                    "default": "1024x1024"
                },
                "use_case": {
                    "type": "string",
                    "enum": ["social_media", "ad_banner", "hero_image", "blog_header", "email_header"],
                    "description": "The intended use case for the image"
                }
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "export_content",
        "description": "Export generated content to a file (markdown, JSON, or text format).",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to export"
                },
                "filename": {
                    "type": "string",
                    "description": "The filename (without extension)"
                },
                "format": {
                    "type": "string",
                    "enum": ["markdown", "json", "txt"],
                    "description": "Export format",
                    "default": "markdown"
                }
            },
            "required": ["content", "filename"]
        }
    }
]


# =============================================================================
# User-Friendly Progress Messages
# =============================================================================

TOOL_PROGRESS_MESSAGES = {
    "fetch_narrative_source": {
        "start": "Scanning website content...",
        "complete": "Website content collected"
    },
    "build_narrative_profile": {
        "start": "Understanding brand personality...",
        "complete": "Brand profile complete"
    },
    "project_text": {
        "start": "Analyzing emotional resonance...",
        "complete": "Emotional analysis complete"
    },
    "analyze_corpus": {
        "start": "Examining content patterns...",
        "complete": "Pattern analysis complete"
    },
    "analyze_force_field": {
        "start": "Mapping messaging dynamics...",
        "complete": "Dynamics mapped"
    },
    "detect_hypocrisy": {
        "start": "Checking message consistency...",
        "complete": "Consistency check complete"
    },
    "generate_campaign": {
        "start": "Crafting campaign concepts...",
        "complete": "Campaign concepts ready"
    },
    # MCP-Inspired Tools
    "deep_crawl_website": {
        "start": "Deep crawling website (multiple pages)...",
        "complete": "Website fully crawled"
    },
    "remember_brand": {
        "start": "Storing brand information in memory...",
        "complete": "Brand information saved to memory"
    },
    "recall_brand": {
        "start": "Retrieving brand information from memory...",
        "complete": "Brand information retrieved"
    },
    "save_campaign": {
        "start": "Saving campaign to file...",
        "complete": "Campaign saved successfully"
    },
    "list_saved_campaigns": {
        "start": "Loading saved campaigns...",
        "complete": "Campaign list retrieved"
    },
    "load_campaign": {
        "start": "Loading campaign from file...",
        "complete": "Campaign loaded successfully"
    },
    "generate_campaign_image": {
        "start": "Generating campaign image with DALL-E...",
        "complete": "Campaign image generated"
    },
    "export_content": {
        "start": "Exporting content to file...",
        "complete": "Content exported successfully"
    }
}


def get_tool_message(tool_name: str, phase: str) -> str:
    """Get user-friendly message for a tool operation."""
    if tool_name in TOOL_PROGRESS_MESSAGES:
        return TOOL_PROGRESS_MESSAGES[tool_name].get(phase, f"Processing {tool_name}...")
    return f"Running {tool_name}..." if phase == "start" else f"Finished {tool_name}"


# =============================================================================
# Tool Execution
# =============================================================================

async def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return results as string."""

    try:
        if name == "fetch_narrative_source":
            result = await fetch_narrative_source(
                url=arguments["url"],
                max_items=arguments.get("max_items", 50)
            )
            return json.dumps({
                "source": result["source"],
                "source_type": result["source_type"],
                "count": result["count"],
                "content_units": result["content_units"][:10],  # First 10 for context
                "note": f"Fetched {result['count']} total content units"
            }, indent=2)

        elif name == "project_text":
            async with ObservatoryClient() as client:
                result = await client.project_text(arguments["text"], use_v2=True)
                return json.dumps({
                    "text": arguments["text"][:100] + "..." if len(arguments["text"]) > 100 else arguments["text"],
                    "coordinates": {
                        "agency": result.agency,
                        "perceived_justice": result.perceived_justice,
                        "belonging": result.belonging
                    },
                    "mode": result.mode,
                    "confidence": result.confidence,
                    "soft_labels": dict(sorted(result.soft_labels.items(), key=lambda x: -x[1])[:5]) if result.soft_labels else None
                }, indent=2)

        elif name == "analyze_corpus":
            async with ObservatoryClient() as client:
                result = await client.analyze_corpus(arguments["texts"], detect_clusters=True)
                return json.dumps({
                    "total_texts": result.total_texts,
                    "mode_distribution": result.mode_distribution,
                    "clusters": [
                        {
                            "id": c.cluster_id,
                            "mode": c.mode,
                            "size": c.size,
                            "stability": c.stability,
                            "centroid": c.centroid
                        }
                        for c in (result.clusters or [])
                    ]
                }, indent=2)

        elif name == "analyze_force_field":
            async with ObservatoryClient() as client:
                result = await client.batch_analyze_forces(arguments["texts"])
                return json.dumps(result, indent=2)

        elif name == "detect_hypocrisy":
            async with ObservatoryClient() as client:
                espoused_results = await client.project_batch(arguments["espoused_texts"])
                operational_results = await client.project_batch(arguments["operational_texts"])

                def centroid(results):
                    n = len(results)
                    return {
                        "agency": sum(r.agency for r in results) / n,
                        "perceived_justice": sum(r.perceived_justice for r in results) / n,
                        "belonging": sum(r.belonging for r in results) / n
                    }

                esp = centroid(espoused_results)
                ops = centroid(operational_results)

                delta = (abs(esp["agency"] - ops["agency"]) +
                        abs(esp["perceived_justice"] - ops["perceived_justice"]) +
                        abs(esp["belonging"] - ops["belonging"])) / 3

                return json.dumps({
                    "espoused_centroid": esp,
                    "operational_centroid": ops,
                    "hypocrisy_gap": delta,
                    "interpretation": "LOW - Well aligned" if delta < 0.3 else
                                     "MODERATE - Some misalignment" if delta < 0.7 else
                                     "HIGH - Significant disconnect"
                }, indent=2)

        elif name == "build_narrative_profile":
            profile = await build_narrative_profile(
                content_units=arguments["content_units"],
                source=arguments["source"],
                source_type="website"
            )
            return json.dumps(profile, indent=2, default=str)

        elif name == "generate_campaign":
            # Campaign generation templates based on theme and targets
            theme = arguments.get("theme", "community")
            context = arguments.get("context", "")

            campaigns = {
                "community": {
                    "headline": "Join thousands who believe in making a difference.",
                    "subhead": "We are a movement of ordinary people doing extraordinary things together.",
                    "body": "This isn't just giving. This is belonging to something bigger than yourself.",
                    "cta": "Join the movement"
                },
                "legacy": {
                    "headline": "Your impact will outlive you.",
                    "subhead": "Create change that echoes through generations.",
                    "body": "What you do today shapes tomorrow. Leave a legacy of meaning.",
                    "cta": "Build your legacy"
                },
                "urgency": {
                    "headline": "The moment for action is now.",
                    "subhead": "Every day matters. Your choice today changes everything.",
                    "body": "Don't wait for someone else. You have the power to make this happen.",
                    "cta": "Act now"
                },
                "justice": {
                    "headline": "We're fixing what's broken.",
                    "subhead": "The old way failed. We're building something better.",
                    "body": "Join us in creating systems that actually work for everyone.",
                    "cta": "Be part of the solution"
                }
            }

            return json.dumps({
                "theme": theme,
                "campaign": campaigns.get(theme, campaigns["community"]),
                "note": "Customize based on profile analysis for optimal resonance"
            }, indent=2)

        # =================================================================
        # MCP-Inspired Tools Implementation
        # =================================================================

        elif name == "deep_crawl_website":
            url = arguments["url"]
            max_pages = min(arguments.get("max_pages", 10), 20)

            result = await firecrawl_deep_crawl(url, max_pages)

            if result.get("fallback"):
                # Fallback to standard fetch
                standard_result = await fetch_narrative_source(url=url, max_items=50)
                return json.dumps({
                    "note": "Used standard fetch (Firecrawl API not configured)",
                    "source": standard_result["source"],
                    "source_type": standard_result["source_type"],
                    "count": standard_result["count"],
                    "content_units": standard_result["content_units"][:10]
                }, indent=2)

            if result.get("success"):
                # Extract text from crawled pages
                pages_content = []
                for page in result.get("pages", [])[:max_pages]:
                    pages_content.append({
                        "url": page.get("url", ""),
                        "title": page.get("title", ""),
                        "content_preview": page.get("markdown", "")[:500] + "..." if page.get("markdown") else ""
                    })

                return json.dumps({
                    "success": True,
                    "total_pages_crawled": result.get("total_pages", 0),
                    "pages": pages_content
                }, indent=2)

            return json.dumps(result, indent=2)

        elif name == "remember_brand":
            memory = load_brand_memory()
            brand_name = arguments["brand_name"]
            category = arguments["category"]
            content = arguments["content"]
            tags = arguments.get("tags", [])

            if brand_name not in memory["brands"]:
                memory["brands"][brand_name] = {
                    "created_at": datetime.now().isoformat(),
                    "categories": {}
                }

            if category not in memory["brands"][brand_name]["categories"]:
                memory["brands"][brand_name]["categories"][category] = []

            memory["brands"][brand_name]["categories"][category].append({
                "content": content,
                "tags": tags,
                "added_at": datetime.now().isoformat()
            })

            save_brand_memory(memory)

            return json.dumps({
                "success": True,
                "brand": brand_name,
                "category": category,
                "message": f"Stored {len(content)} characters of {category} information for {brand_name}"
            }, indent=2)

        elif name == "recall_brand":
            memory = load_brand_memory()
            brand_name = arguments["brand_name"]
            category = arguments.get("category", "all")

            if brand_name not in memory["brands"]:
                return json.dumps({
                    "success": False,
                    "error": f"No information stored for brand: {brand_name}",
                    "available_brands": list(memory["brands"].keys())
                }, indent=2)

            brand_data = memory["brands"][brand_name]

            if category == "all":
                return json.dumps({
                    "success": True,
                    "brand": brand_name,
                    "created_at": brand_data.get("created_at"),
                    "categories": brand_data.get("categories", {})
                }, indent=2)
            else:
                cat_data = brand_data.get("categories", {}).get(category, [])
                return json.dumps({
                    "success": True,
                    "brand": brand_name,
                    "category": category,
                    "entries": cat_data
                }, indent=2)

        elif name == "save_campaign":
            brand_name = arguments["brand_name"]
            campaign_name = arguments["campaign_name"]
            analysis = arguments["analysis"]
            generated_content = arguments.get("generated_content", {})

            campaign_id = generate_campaign_id(brand_name)
            campaign_data = {
                "id": campaign_id,
                "brand_name": brand_name,
                "campaign_name": campaign_name,
                "created_at": datetime.now().isoformat(),
                "analysis": analysis,
                "generated_content": generated_content
            }

            filepath = CAMPAIGNS_DIR / f"{campaign_id}.json"
            filepath.write_text(json.dumps(campaign_data, indent=2, default=str))

            return json.dumps({
                "success": True,
                "campaign_id": campaign_id,
                "filepath": str(filepath),
                "message": f"Campaign '{campaign_name}' saved successfully"
            }, indent=2)

        elif name == "list_saved_campaigns":
            brand_filter = arguments.get("brand_name", "").lower()
            campaigns = []

            for filepath in CAMPAIGNS_DIR.glob("*.json"):
                try:
                    data = json.loads(filepath.read_text())
                    if not brand_filter or brand_filter in data.get("brand_name", "").lower():
                        campaigns.append({
                            "id": data.get("id"),
                            "brand_name": data.get("brand_name"),
                            "campaign_name": data.get("campaign_name"),
                            "created_at": data.get("created_at")
                        })
                except:
                    continue

            campaigns.sort(key=lambda x: x.get("created_at", ""), reverse=True)

            return json.dumps({
                "success": True,
                "total_campaigns": len(campaigns),
                "campaigns": campaigns[:20]  # Return most recent 20
            }, indent=2)

        elif name == "load_campaign":
            campaign_id = arguments["campaign_id"]
            filepath = CAMPAIGNS_DIR / f"{campaign_id}.json"

            if not filepath.exists():
                return json.dumps({
                    "success": False,
                    "error": f"Campaign not found: {campaign_id}"
                }, indent=2)

            campaign_data = json.loads(filepath.read_text())
            return json.dumps({
                "success": True,
                "campaign": campaign_data
            }, indent=2)

        elif name == "generate_campaign_image":
            prompt = arguments["prompt"]
            style = arguments.get("style", "vivid")
            size = arguments.get("size", "1024x1024")
            use_case = arguments.get("use_case", "social_media")

            # Enhance prompt based on use case
            use_case_hints = {
                "social_media": "vibrant, eye-catching, suitable for social media post",
                "ad_banner": "professional, clean layout with space for text overlay",
                "hero_image": "expansive, dramatic, high-impact hero section image",
                "blog_header": "editorial style, sophisticated, article header image",
                "email_header": "clean, simple, email-friendly with good contrast"
            }

            enhanced_prompt = f"{prompt}. Style: {use_case_hints.get(use_case, '')}. Professional marketing quality."

            result = await generate_dalle_image(enhanced_prompt, style, size)

            if result.get("success"):
                return json.dumps({
                    "success": True,
                    "image_url": result.get("image_url"),
                    "local_path": result.get("local_path"),
                    "revised_prompt": result.get("revised_prompt"),
                    "use_case": use_case,
                    "size": size
                }, indent=2)

            return json.dumps(result, indent=2)

        elif name == "export_content":
            content = arguments["content"]
            filename = arguments["filename"]
            format_type = arguments.get("format", "markdown")

            # Sanitize filename
            safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            extensions = {"markdown": ".md", "json": ".json", "txt": ".txt"}
            ext = extensions.get(format_type, ".txt")

            filepath = EXPORTS_DIR / f"{safe_filename}_{timestamp}{ext}"

            if format_type == "json":
                try:
                    # Try to parse as JSON first
                    parsed = json.loads(content)
                    filepath.write_text(json.dumps(parsed, indent=2))
                except:
                    filepath.write_text(json.dumps({"content": content}, indent=2))
            else:
                filepath.write_text(content)

            return json.dumps({
                "success": True,
                "filepath": str(filepath),
                "format": format_type,
                "size_bytes": filepath.stat().st_size
            }, indent=2)

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Claude Orchestrator
# =============================================================================

@dataclass
class AnalysisUpdate:
    """Real-time update from Claude's analysis."""
    type: str  # "thinking", "tool_call", "tool_result", "text", "complete", "error"
    content: str
    data: Optional[dict] = None


async def analyze_with_claude(
    input_text: str,
    input_type: str = "url",
    intent: str = "understand",
    objective: str = "leads",
    brand_voice: str = "professional",
    brand_personality: str = "",
    target_audience: str = ""
) -> AsyncGenerator[AnalysisUpdate, None]:
    """
    Run narrative analysis orchestrated by Claude.

    Yields real-time updates as Claude thinks, calls tools, and generates insights.
    """

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        yield AnalysisUpdate(
            type="error",
            content="ANTHROPIC_API_KEY not set. Please set your API key to enable Claude orchestration."
        )
        return

    client = Anthropic(api_key=api_key)

    # Build the user message based on intent
    intent_context = {
        "understand": "I want to understand their current messaging and emotional appeal.",
        "improve": "I want to improve their messaging to be more effective.",
        "compete": "I want to create messaging that outperforms theirs.",
        "campaign": "I need to create a new campaign based on what resonates with their audience.",
    }

    goal = intent_context.get(intent, intent_context["understand"])

    # Objective descriptions
    objective_context = {
        "awareness": "Build brand awareness and recognition. Focus on memorable, shareable messaging.",
        "engagement": "Drive audience engagement - comments, shares, interactions. Create conversation starters.",
        "leads": "Generate leads and capture contact information. Focus on compelling offers and clear CTAs.",
        "sales": "Increase direct sales and conversions. Focus on persuasive copy with urgency and value.",
        "loyalty": "Build customer loyalty and advocacy. Focus on emotional connection and community.",
        "authority": "Establish thought leadership and trust. Focus on expertise and credibility.",
    }

    # Build brand voice context
    brand_context = f"\n\n**Brand Voice Setting:** {brand_voice.capitalize()}"
    if brand_personality:
        brand_context += f"\n**Brand Personality Traits:** {brand_personality}"
    brand_context += f"\n**Primary Objective:** {objective_context.get(objective, objective_context['leads'])}"
    if target_audience:
        brand_context += f"\n**Target Audience:** {target_audience}"
    brand_context += "\n\nAdapt all recommendations and generated content to match this brand voice, personality, and primary objective."
    if target_audience:
        brand_context += f" Tailor messaging specifically for the target audience: {target_audience}."

    # Avatar profile section for the prompt
    avatar_section = """

## TARGET AUDIENCE PROFILES

Based on your analysis, identify 3 distinct audience segments. Use professional, descriptive titles (not alliterative nicknames).

For each profile, use this format:

### Profile 1: [Descriptive Title]
- **Summary:** One sentence describing this audience segment
- **Demographics:** Age, location, income, family status
- **Values:** What they care about, their beliefs
- **Pain Points:** Top 3 frustrations relevant to this brand
- **Goals:** What success looks like for them
- **Objections:** What would make them hesitate
- **Messaging Triggers:** What compels them to act
- **Channels:** Where to reach them (platforms, communities)

### Profile 2: [Descriptive Title]
[Same format]

### Profile 3: [Descriptive Title]
[Same format]

Make profiles SPECIFIC based on actual content signals, not generic stereotypes."""

    # Handle multi-source input
    if input_type == "multi_source":
        # Parse the multi-source input
        sources_description = []
        has_url = "PRIMARY_URL:" in input_text
        has_social = "SOCIAL_MEDIA_URL:" in input_text
        has_docs = "DOCUMENTS:" in input_text
        has_text = "TEXT_CONTENT:" in input_text

        has_social_content = "SOCIAL_MEDIA_CONTENT:" in input_text

        print(f"[DEBUG] Multi-source input received:")
        print(f"[DEBUG]   has_url: {has_url}")
        print(f"[DEBUG]   has_social_url: {has_social}")
        print(f"[DEBUG]   has_social_content: {has_social_content}")
        print(f"[DEBUG]   has_docs: {has_docs}")
        print(f"[DEBUG]   has_text: {has_text}")
        print(f"[DEBUG] Raw input_text:\n{input_text[:500]}...")

        if has_url:
            sources_description.append("website")
        if has_social:
            sources_description.append("social media URL")
        if has_social_content:
            sources_description.append("social media content (pasted)")
        if has_docs:
            sources_description.append("uploaded documents")
        if has_text:
            sources_description.append("provided text content")

        sources_str = ", ".join(sources_description)

        # Extract URLs from input for explicit instructions
        primary_url = None
        social_url = None
        for line in input_text.split('\n'):
            if line.startswith('PRIMARY_URL:'):
                primary_url = line.replace('PRIMARY_URL:', '').strip()
            elif line.startswith('SOCIAL_MEDIA_URL:'):
                social_url = line.replace('SOCIAL_MEDIA_URL:', '').strip()

        print(f"[DEBUG] Extracted URLs:")
        print(f"[DEBUG]   primary_url: {primary_url}")
        print(f"[DEBUG]   social_url: {social_url}")

        fetch_instructions = []
        if primary_url:
            fetch_instructions.append(f"1. **REQUIRED:** Use `fetch_narrative_source` tool with url=\"{primary_url}\" to get the website content")
        if social_url:
            fetch_instructions.append(f"{'2' if primary_url else '1'}. **REQUIRED:** Use `fetch_narrative_source` tool with url=\"{social_url}\" to get the social media content")

        fetch_steps = '\n'.join(fetch_instructions) if fetch_instructions else "Process the provided content"
        print(f"[DEBUG] Fetch instructions:\n{fetch_steps}")

        user_message = f"""Analyze the following data sources for comprehensive messaging and campaign insights:

{input_text}

{goal}{brand_context}

**DATA SOURCES PROVIDED:** {sources_str}

**CRITICAL: You MUST fetch content from ALL URLs provided. Do not skip any URL.**

{fetch_steps}

After fetching URLs, then:
- Analyze any SOCIAL_MEDIA_CONTENT provided (pasted social posts/content - already included above, no fetching needed)
- Process any DOCUMENTS content provided (already extracted text above)
- Synthesize ALL sources into a unified brand understanding
- Analyze the emotional resonance across all sources
- Identify consistencies and inconsistencies in messaging across channels
- Deeply understand WHO this content resonates with

**MULTI-SOURCE SYNTHESIS:**
- Look for consistent themes across sources
- Note any messaging gaps between channels (e.g., website vs. social)
- Identify the strongest messaging elements from each source
- Create a unified brand voice profile that accounts for all inputs

Then give me:

## DATA SOURCES ANALYZED
- List each source and key takeaways from each

## MESSAGING ANALYSIS
- Unified emotional profile (how does this brand make people feel across all touchpoints?)
- Cross-channel consistency assessment
- Messaging strengths and gaps
- What makes this brand compelling (or not)
{avatar_section}

## CAMPAIGN RECOMMENDATIONS
- Ready-to-use headlines and copy (matching the specified brand voice)
- Channel-specific recommendations (if multiple sources provided)
- Quick wins for immediate improvement

Be practical and specific. I need actionable outputs I can use right away."""

    elif input_type == "url":
        user_message = f"""Analyze this website for messaging and campaign insights: {input_text}

{goal}{brand_context}

Use your analysis tools to:
1. Fetch the content from the website
2. Analyze the emotional resonance of their messaging using the observatory
3. Identify what's working and what's missing
4. Deeply understand WHO this content resonates with

Then give me:

## MESSAGING ANALYSIS
- Emotional profile (how does this content make people feel?)
- Messaging strengths and gaps
- What makes this content compelling (or not)
{avatar_section}

## CAMPAIGN RECOMMENDATIONS
- Ready-to-use headlines and copy (matching the specified brand voice)
- Quick wins for immediate improvement

Be practical and specific. I need actionable outputs I can use right away."""
    else:
        user_message = f"""Analyze this text for messaging and campaign insights:

{input_text}

{goal}{brand_context}

Analyze the emotional resonance and give me:

## MESSAGING ANALYSIS
- Emotional profile (how does this make people feel?)
- Strengths and gaps in the messaging
{avatar_section}

## CAMPAIGN RECOMMENDATIONS
- Ready-to-use headlines and copy (matching the specified brand voice)
- Quick wins for immediate improvement

Be practical and specific. I need actionable outputs."""

    messages = [{"role": "user", "content": user_message}]

    yield AnalysisUpdate(
        type="thinking",
        content="Claude is analyzing your request..."
    )

    try:
        # Initial response
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        # Process response, handling tool use
        while response.stop_reason == "tool_use":
            # Extract tool uses
            tool_uses = [block for block in response.content if block.type == "tool_use"]

            # Execute each tool
            tool_results = []
            for tool_use in tool_uses:
                yield AnalysisUpdate(
                    type="tool_call",
                    content=get_tool_message(tool_use.name, "start"),
                    data={"tool": tool_use.name, "input": tool_use.input}
                )

                # Execute the tool
                result = await execute_tool(tool_use.name, tool_use.input)

                yield AnalysisUpdate(
                    type="tool_result",
                    content=get_tool_message(tool_use.name, "complete"),
                    data={"tool": tool_use.name, "result_preview": result[:500] + "..." if len(result) > 500 else result}
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result
                })

            # Continue the conversation with tool results
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            yield AnalysisUpdate(
                type="thinking",
                content="Claude is processing the results..."
            )

            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages
            )

        # Extract final text response
        text_blocks = [block.text for block in response.content if hasattr(block, 'text')]
        final_response = "\n".join(text_blocks)

        yield AnalysisUpdate(
            type="complete",
            content=final_response,
            data={"usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}}
        )

    except Exception as e:
        yield AnalysisUpdate(
            type="error",
            content=f"Analysis failed: {str(e)}"
        )


# =============================================================================
# Content Generation (using analysis as knowledge base)
# =============================================================================

CONTENT_TEMPLATES = {
    "twitter": {
        "name": "Twitter/X Posts",
        "icon": "𝕏",
        "prompt": """Based on the brand analysis, create engaging Twitter/X content.

Requirements:
- Create 5 standalone tweets (each under 280 characters)
- Create 1 thread (5-7 tweets) for deeper engagement
- Mix of styles: inspirational, educational, call-to-action, question/engagement
- Include relevant hashtag suggestions (2-3 per post, not excessive)
- Match the brand's emotional tone and voice profile

**Engagement Predictions** - For each post, rate:
- Expected engagement level: Low / Medium / High
- Best for: Awareness / Engagement / Conversion

**Posting Time Suggestions:**
- Recommend optimal posting windows (e.g., "Tue-Thu 9-11am EST for B2B")
- Note any time-sensitive hooks (trending topics, seasons, events)

Format each as:
**Standalone Post 1**
[Tweet text]
Hashtags: #hashtag1 #hashtag2
Engagement: [High] | Best for: [Awareness]

**Thread: [Topic]**
Tweet 1/5: [Hook tweet]
Tweet 2/5: [Context/problem]
...

**Posting Schedule:**
[Recommendations]"""
    },
    "linkedin": {
        "name": "LinkedIn Posts",
        "icon": "in",
        "prompt": """Based on the brand analysis, create 3 professional LinkedIn posts optimized for B2B engagement.

Requirements:
- Professional but human and engaging tone
- 150-300 words each (LinkedIn's algorithm favors this length)
- Include a strong hook in the first 2 lines (before "see more")
- Include a hook, value, and call-to-action
- Match the brand's messaging themes

**B2B Context:**
- Focus on business value, ROI, and professional insights
- Reference industry trends or challenges where relevant
- Position content for decision-makers and stakeholders

**Thought Leadership Angle:**
- For each post, identify the thought leadership angle:
  - Industry Insight / Contrarian Take / Personal Story / Data-Driven / How-To

**Document/Carousel Suggestions:**
- For each post, suggest if it could be enhanced with:
  - PDF carousel (key points as slides)
  - Infographic attachment
  - Video complement
  - Poll follow-up

Format each as:
**Post 1: [Thought Leadership Type]**
[Hook - first 2 lines that appear before "see more"]

[Full post body]

[CTA]

Document Suggestion: [Recommendation]
Best Posting Time: [Day/Time recommendation for B2B]"""
    },
    "instagram": {
        "name": "Instagram Captions",
        "icon": "📸",
        "prompt": """Based on the brand analysis, create 5 Instagram captions.

Requirements:
- Engaging, visual-friendly language
- Mix of lengths (short punchy + longer storytelling)
- Include emoji suggestions
- Hashtag recommendations (10-15 per post)
- Match the brand's emotional tone"""
    },
    "blog_outline": {
        "name": "Blog Post Outline",
        "icon": "📝",
        "prompt": """Based on the brand analysis, create a detailed blog post outline.

Requirements:
- Compelling headline options (3 variations)
- Clear structure with H2 and H3 subheadings
- Key points to cover in each section
- Suggested word count per section
- SEO keyword recommendations
- Call-to-action ideas"""
    },
    "email_sequence": {
        "name": "Email Sequence",
        "icon": "📧",
        "prompt": """Based on the brand analysis, create a strategic email sequence.

**Sequence Type Selection:**
First, identify the most appropriate sequence type based on the brand analysis:
- Welcome Sequence (new subscribers/customers)
- Nurture Sequence (lead warming)
- Win-Back Sequence (re-engagement)
- Launch Sequence (product/service announcement)
- Onboarding Sequence (new customer education)

**Create a 5-email sequence with:**

For each email, provide:
1. **Subject Line Options** (A/B test variants)
   - Option A: [Curiosity-driven]
   - Option B: [Benefit-driven]
   - Option C: [Urgency/FOMO - if appropriate]

2. **Preview Text** (40-90 characters, complements subject line)

3. **Email Body Copy**
   - Opening hook
   - Value/content section
   - Clear single CTA
   - P.S. line (optional secondary hook)

4. **Send Timing**
   - Days after trigger/previous email
   - Optimal send time (e.g., "Tuesday 10am local time")
   - Why this timing works

**Sequence Overview:**
- Email 1: [Purpose] - Send: [Timing]
- Email 2: [Purpose] - Send: [Timing]
- Email 3: [Purpose] - Send: [Timing]
- Email 4: [Purpose] - Send: [Timing]
- Email 5: [Purpose] - Send: [Timing]

**A/B Testing Recommendations:**
- Priority elements to test
- Expected impact of each test"""
    },
    "ad_copy": {
        "name": "Ad Copy",
        "icon": "📣",
        "prompt": """Based on the brand analysis, create ad copy variations with platform-specific optimization.

**Audience Targeting Notes:**
Before creating copy, define:
- Primary audience segment
- Awareness level (cold/warm/hot)
- Key pain points to address
- Desired action/conversion goal

**Facebook/Instagram Ads (3 variations):**
For each variation:
- Primary Text (125 chars visible, up to 500 total)
- Headline (40 chars max)
- Description (30 chars max)
- CTA Button recommendation
- Audience targeting notes (interests, behaviors, lookalikes)

**Google Search Ads (3 variations):**
For each variation:
- Headlines (3 x 30 chars each)
- Descriptions (2 x 90 chars each)
- Display path suggestions
- Keyword intent alignment

**Platform Compliance Flags:**
Check and note for each ad:
- [ ] No prohibited content (alcohol, gambling restrictions if applicable)
- [ ] No exaggerated claims or superlatives without proof
- [ ] No before/after implications without disclaimers
- [ ] Landing page alignment confirmed
- [ ] No discriminatory targeting implications
- [ ] Special category considerations (housing, employment, credit)

**Performance Predictions:**
For each ad variation, estimate:
- Expected CTR range: Low (<1%) / Medium (1-2%) / High (>2%)
- Funnel stage fit: Awareness / Consideration / Conversion
- Recommended daily budget tier: Test ($10-25) / Scale ($50-100) / Aggressive ($200+)"""
    },
    "landing_page": {
        "name": "Landing Page Copy",
        "icon": "🖥️",
        "prompt": """Based on the brand analysis, create conversion-optimized landing page copy.

**Conversion Goal Context:**
First, identify the primary conversion goal:
- Lead Generation (email capture)
- Free Trial Signup
- Demo Request
- Direct Purchase
- Webinar/Event Registration
- Content Download

**Above-the-Fold Priorities (Critical - users decide in 5 seconds):**
- Hero Headline (3 variations) - Clear value proposition, under 10 words
- Subheadline - Expands on headline, addresses "how" or "for whom"
- Primary CTA button text (3 variations)
- Trust indicator (one-liner: "Trusted by X companies" or "X customers served")
- Hero image/video direction suggestion

**Full Page Structure:**

1. **Hero Section** (above fold)
   - Headline options
   - Subheadline
   - CTA
   - Trust badge suggestion

2. **Problem/Agitation Section**
   - Pain point statements (3-4 bullets)
   - Emotional connection copy

3. **Solution/Benefits Section**
   - Value proposition bullets (5-6 points)
   - Feature-to-benefit translations

4. **Social Proof Section**
   - Testimonial placement suggestions
   - Key quote themes to gather
   - Logo bar recommendations

5. **How It Works** (if applicable)
   - 3-step process copy

6. **FAQ Section** (5 questions)
   - Objection-handling questions
   - SEO-friendly answers

7. **Final CTA Section**
   - Urgency/scarcity element (if authentic)
   - Risk reversal (guarantee, free trial)
   - Final CTA button text

**Mobile Considerations:**
- Thumb-friendly CTA placement notes
- Content prioritization for mobile scroll
- Recommended mobile-hidden elements
- Touch target sizing reminders

**Conversion Optimization Notes:**
- Single focused CTA vs. multiple options recommendation
- Form field reduction suggestions
- Exit intent popup copy (if appropriate)"""
    },
    "taglines": {
        "name": "Taglines & Slogans",
        "icon": "✨",
        "prompt": """Based on the brand analysis, create tagline and slogan options.

Requirements:
- 10 tagline variations (short, punchy)
- 5 longer slogans/value propositions
- Organize by emotional angle (empowerment, trust, community)
- Note which contexts each works best for"""
    },
    "video_script": {
        "name": "Video Script",
        "icon": "🎬",
        "prompt": """Based on the brand analysis, create a 60-90 second video script.

**Script Structure:**

**HOOK (0-3 seconds)** - Critical for retention
- Opening line that stops the scroll
- Visual cue: [Describe opening shot/visual]
- On-screen text suggestion

**PROBLEM/TENSION (3-15 seconds)**
- Identify the pain point
- Create emotional connection
- Visual cues: [B-roll suggestions]

**SOLUTION INTRODUCTION (15-30 seconds)**
- Introduce the brand/product as the answer
- Key differentiator statement
- Visual cues: [Product shots, demonstrations]

**BENEFITS/PROOF (30-50 seconds)**
- 2-3 key benefits with visual support
- Social proof moment (testimonial clip, stats)
- Visual cues: [Results, happy customers, data visualization]

**CALL-TO-ACTION (50-60 seconds)**
- Clear, single action to take
- Urgency element (if authentic)
- On-screen text: [CTA text overlay]
- End screen suggestions

**Script Format:**
```
[VISUAL]                    | [AUDIO/VO]
---------------------------|---------------------------
[B-roll of problem]        | "Ever feel like..."
[Cut to product]           | "That's why we created..."
```

**Production Notes:**
- Suggested music mood/tempo
- Pacing notes (fast cuts vs. lingering shots)
- Platform optimization (vertical for TikTok/Reels, horizontal for YouTube)
- Caption/subtitle recommendations (85% watch without sound)"""
    },
    "press_release": {
        "name": "Press Release",
        "icon": "📰",
        "prompt": """Based on the brand analysis, create a professional press release.

**Press Release Structure:**

**HEADLINE**
- Primary headline (compelling, newsworthy angle)
- Subheadline (additional context)

**DATELINE & LEAD PARAGRAPH**
- City, State — Date
- Lead paragraph answering: Who, What, When, Where, Why
- Most newsworthy information first (inverted pyramid)

**BODY PARAGRAPHS**
- Paragraph 2: Expand on the news, provide context
- Paragraph 3: Quote from company spokesperson
  - Name, Title
  - Quote that adds human element and vision
- Paragraph 4: Additional details, features, or benefits
- Paragraph 5: Quote from partner/customer (if applicable)
- Paragraph 6: Availability, pricing, or next steps

**BOILERPLATE (About Section)**
- Company description (50-100 words)
- Key facts: Founded, headquarters, mission
- Notable achievements or metrics

**MEDIA CONTACT**
- Contact Name
- Title
- Email
- Phone
- Website

**ADDITIONAL ELEMENTS:**
- Suggested multimedia assets (photos, videos, infographics)
- Relevant hashtags for social sharing
- Embargo information (if applicable)
- Editor's notes (background information not for publication)

**Distribution Notes:**
- Suggested wire services
- Target media outlets/journalists
- Optimal release timing"""
    },
    "case_study": {
        "name": "Case Study Outline",
        "icon": "📋",
        "prompt": """Based on the brand analysis, create a comprehensive case study framework.

**Case Study Structure:**

**1. TITLE & SNAPSHOT**
- Compelling title formula: "[Result] for [Client Type] with [Solution]"
- Quick stats box:
  - Industry: [Placeholder]
  - Company Size: [Placeholder]
  - Challenge: [One-liner]
  - Result: [Key metric]

**2. CLIENT BACKGROUND**
- Company overview (2-3 sentences)
- Industry context
- Relevant size/scale information
- Why they're a representative customer

**3. THE CHALLENGE**
- Primary problem statement
- Secondary challenges
- Business impact of the problem
- Previous solutions attempted (if any)
- Stakes: What would happen if unsolved?

**Questions to ask the client:**
- "What was the breaking point that made you seek a solution?"
- "How was this problem affecting your team/business daily?"
- "What had you tried before?"

**4. THE SOLUTION**
- How they discovered your product/service
- Implementation overview
- Key features/services utilized
- Timeline from start to results
- Support/partnership elements

**Questions to ask the client:**
- "What made you choose us over alternatives?"
- "How was the implementation experience?"
- "What surprised you about working with us?"

**5. THE RESULTS**
- Primary metric improvement: [Placeholder - e.g., "X% increase in..."]
- Secondary metrics: [Placeholder]
- Qualitative improvements
- ROI calculation (if applicable)
- Time to value

**Metrics to gather:**
- Before/after comparisons
- Percentage improvements
- Time saved
- Revenue impact
- Customer satisfaction changes

**6. KEY QUOTES TO GATHER**
- Challenge quote: Pain point in their words
- Solution quote: Why they chose you
- Results quote: Impact on their business
- Recommendation quote: Would they recommend?

**7. VISUAL SUGGESTIONS**
- Client logo (with permission)
- Before/after screenshots
- Data visualization of results
- Team/product photos
- Process diagram

**8. CALL-TO-ACTION**
- Related resources
- Demo/consultation offer
- Similar case studies

**Interview Guide:**
- Suggested interview questions for the client
- Information release/approval process reminder"""
    }
}


async def generate_content(
    content_type: str,
    analysis_context: str,
    custom_instructions: str = "",
    variations: bool = False
) -> AsyncGenerator[AnalysisUpdate, None]:
    """
    Generate specific content using the analysis as a knowledge base.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        yield AnalysisUpdate(
            type="error",
            content="ANTHROPIC_API_KEY not set."
        )
        return

    template = CONTENT_TEMPLATES.get(content_type)
    if not template:
        yield AnalysisUpdate(
            type="error",
            content=f"Unknown content type: {content_type}"
        )
        return

    client = Anthropic(api_key=api_key)

    system_prompt = """You are an expert content creator and copywriter. You create compelling, on-brand content based on brand analysis insights.

Your content should:
- Match the brand's emotional tone and messaging themes
- Be immediately usable (not generic templates)
- Include specific, actionable copy
- Be optimized for the target platform

Do not explain or add commentary - just deliver the requested content in a clean, organized format."""

    # Add variations instruction if requested
    variations_instruction = ""
    if variations:
        variations_instruction = """

**IMPORTANT: Generate 3 distinct variations with different angles:**

## Variation 1: Emotional Appeal
Focus on feelings, stories, and emotional connections. Use evocative language that tugs at heartstrings and creates empathy.

## Variation 2: Logic-Driven
Focus on facts, data, and rational benefits. Use evidence, statistics, and clear reasoning to make the case.

## Variation 3: Urgency-Based
Focus on time-sensitivity, scarcity, and immediate action. Create FOMO and emphasize why acting now matters.

Clearly label each variation and ensure they are distinctly different in approach while maintaining brand voice consistency."""

    user_message = f"""Here is the brand analysis to use as your knowledge base:

---
{analysis_context}
---

{template['prompt']}{variations_instruction}

{f"Additional instructions: {custom_instructions}" if custom_instructions else ""}"""

    yield AnalysisUpdate(
        type="thinking",
        content=f"Generating {template['name']}..."
    )

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )

        text_blocks = [block.text for block in response.content if hasattr(block, 'text')]
        final_response = "\n".join(text_blocks)

        yield AnalysisUpdate(
            type="complete",
            content=final_response,
            data={
                "content_type": content_type,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        )

    except Exception as e:
        yield AnalysisUpdate(
            type="error",
            content=f"Generation failed: {str(e)}"
        )


def get_content_types() -> list:
    """Return available content types for the UI."""
    return [
        {"id": k, "name": v["name"], "icon": v["icon"]}
        for k, v in CONTENT_TEMPLATES.items()
    ]


# =============================================================================
# Synchronous wrapper for testing
# =============================================================================

async def run_analysis(url_or_text: str, input_type: str = "url", intent: str = "understand") -> dict:
    """Run full analysis and return results."""
    results = {
        "updates": [],
        "final_response": None,
        "error": None
    }

    async for update in analyze_with_claude(url_or_text, input_type, intent):
        results["updates"].append({
            "type": update.type,
            "content": update.content,
            "data": update.data
        })

        if update.type == "complete":
            results["final_response"] = update.content
        elif update.type == "error":
            results["error"] = update.content

    return results


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import sys

    async def main():
        if len(sys.argv) < 2:
            print("Usage: python claude_orchestrator.py <url_or_text>")
            print("\nExample: python claude_orchestrator.py https://www.hopeeffect.com/")
            return

        input_text = sys.argv[1]
        input_type = "url" if input_text.startswith("http") else "text"

        print("=" * 60)
        print("CLAUDE-ORCHESTRATED NARRATIVE ANALYSIS")
        print("=" * 60)
        print(f"\nInput: {input_text[:100]}...")
        print(f"Type: {input_type}")
        print("\n" + "-" * 60 + "\n")

        async for update in analyze_with_claude(input_text, input_type):
            if update.type == "thinking":
                print(f"🤔 {update.content}")
            elif update.type == "tool_call":
                print(f"🔧 {update.content}")
                if update.data:
                    print(f"   Input: {json.dumps(update.data.get('input', {}))[:100]}...")
            elif update.type == "tool_result":
                print(f"📊 {update.content}")
            elif update.type == "complete":
                print("\n" + "=" * 60)
                print("ANALYSIS COMPLETE")
                print("=" * 60)
                print(update.content)
            elif update.type == "error":
                print(f"❌ ERROR: {update.content}")

    asyncio.run(main())
