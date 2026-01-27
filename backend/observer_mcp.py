"""
Observer Agent - MCP Server for Aperture Studio Orchestration

The Observer is an agent-first interface to the Cultural Soliton Observatory.
It provides:
- Lens selection and configuration workflows
- Conversational analysis setup
- Script generation for batch analysis
- Integration between Aperture Studio UI and Observatory backend

This is designed to be used with Claude Code or any MCP-compatible client.

Usage:
    # Add to Claude Code settings (~/.claude/settings.json):
    {
      "mcpServers": {
        "observer": {
          "command": "python3",
          "args": ["observer_mcp.py"],
          "cwd": "/path/to/cultural-soliton-observatory/backend"
        }
      }
    }

    # Or run directly:
    python observer_mcp.py
"""

import asyncio
import json
import os
from typing import Any, Optional, Dict, List
from datetime import datetime
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
    Prompt,
    PromptMessage,
    GetPromptResult,
)

# Configuration
BACKEND_URL = os.getenv("OBSERVATORY_BACKEND_URL", "http://127.0.0.1:8000")
APERTURE_URL = os.getenv("APERTURE_STUDIO_URL", "http://localhost:3001")

# Create server instance
server = Server("observer-agent")

# =============================================================================
# LENS DEFINITIONS - The core of Aperture Studio
# =============================================================================

KILLER_APP_LENSES = {
    "denial-messaging": {
        "id": "denial-messaging",
        "name": "Denial Messaging QA",
        "industry": "Fintech / Insurance",
        "description": "Detect shadow-zone language in claim denials before they trigger regulatory scrutiny or viral backlash",
        "primary_dimensions": ["perceived_justice", "agency", "belonging"],
        "watch_modes": ["INSTITUTIONAL_DECAY", "CYNICAL_BURNOUT", "VICTIM"],
        "alert_thresholds": {
            "perceived_justice": -0.8,
            "agency": -1.0
        },
        "questions": [
            "What type of denial messages do you want to analyze? (claim denials, policy rejections, coverage exclusions)",
            "What is your primary concern? (regulatory compliance, customer experience, brand protection)",
            "Do you have sample denial templates or historical denials to analyze?",
            "What threshold of 'shadow zone' language would trigger a review? (strict, moderate, lenient)"
        ],
        "kpis": [
            "Percentage of denials in shadow zone",
            "Average perceived_justice score",
            "Risk score distribution",
            "Templates flagged for revision"
        ]
    },
    "crisis-preflight": {
        "id": "crisis-preflight",
        "name": "Crisis Pre-Flight",
        "industry": "PR / Communications",
        "description": "Test crisis statements before release to predict narrative trajectory",
        "primary_dimensions": ["agency", "perceived_justice", "belonging"],
        "watch_modes": ["VICTIM", "TRANSITIONAL", "NEUTRAL"],
        "alert_thresholds": {
            "belonging": -0.5,
            "agency": 0.2
        },
        "questions": [
            "What type of crisis are you preparing for? (product issue, executive misconduct, data breach, layoffs)",
            "Who is the primary audience? (customers, employees, investors, regulators, media)",
            "What is your desired narrative outcome? (accountability, reassurance, action-focused)",
            "Do you have draft statements to test?"
        ],
        "kpis": [
            "Narrative mode prediction",
            "Stakeholder resonance scores",
            "Risk of backfire indicators",
            "Suggested tone adjustments"
        ]
    },
    "support-triage": {
        "id": "support-triage",
        "name": "Support Narrative Triage",
        "industry": "Customer Success",
        "description": "Prioritize support tickets by narrative distress signals, not just keywords",
        "primary_dimensions": ["belonging", "agency", "perceived_justice"],
        "watch_modes": ["QUIET_QUITTING", "NEUTRAL", "TRANSITIONAL"],
        "alert_thresholds": {
            "belonging": -1.0,
            "agency": -0.8
        },
        "questions": [
            "What support channels do you want to analyze? (tickets, chat, email, social)",
            "What is your escalation trigger? (churn risk, VIP customer, legal language)",
            "How do you currently prioritize tickets?",
            "What does 'high distress' look like for your customers?"
        ],
        "kpis": [
            "Distress score distribution",
            "Churn risk indicators",
            "Escalation recommendations",
            "Response urgency scores"
        ]
    }
}

STANDARD_LENSES = {
    "general-analysis": {
        "id": "general-analysis",
        "name": "General Narrative Analysis",
        "description": "Balanced analysis across all dimensions",
        "primary_dimensions": ["agency", "perceived_justice", "belonging"],
        "watch_modes": [],
        "alert_thresholds": {}
    },
    "engagement-health": {
        "id": "engagement-health",
        "name": "Employee Engagement Health",
        "description": "Measure organizational narrative health from employee communications",
        "primary_dimensions": ["belonging", "agency", "perceived_justice"],
        "watch_modes": ["QUIET_QUITTING", "CYNICAL_BURNOUT", "INSTITUTIONAL_DECAY"],
        "alert_thresholds": {
            "belonging": -0.5,
            "agency": -0.3
        }
    }
}

ALL_LENSES = {**KILLER_APP_LENSES, **STANDARD_LENSES}

# Session state for multi-turn conversations
SESSION_STATE: Dict[str, Any] = {}


# =============================================================================
# HTTP CLIENT
# =============================================================================

async def call_backend(method: str, endpoint: str, data: Optional[dict] = None) -> dict:
    """Call the Observatory backend."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        url = f"{BACKEND_URL}{endpoint}"
        if method == "GET":
            response = await client.get(url, params=data)
        elif method == "POST":
            response = await client.post(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        response.raise_for_status()
        return response.json()


# =============================================================================
# PROMPTS - Conversational workflows for lens configuration
# =============================================================================

@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    """List available conversational prompts."""
    return [
        Prompt(
            name="configure-lens",
            description="Interactive lens configuration wizard. Walk through selecting and customizing a lens for your specific use case.",
            arguments=[
                {
                    "name": "lens_type",
                    "description": "Optional: pre-select a lens (denial-messaging, crisis-preflight, support-triage)",
                    "required": False
                }
            ]
        ),
        Prompt(
            name="analyze-with-lens",
            description="Run analysis on texts using a configured lens. Returns scored results with alerts.",
            arguments=[
                {
                    "name": "lens_id",
                    "description": "The lens ID to use for analysis",
                    "required": True
                },
                {
                    "name": "texts",
                    "description": "JSON array of texts to analyze",
                    "required": True
                }
            ]
        ),
        Prompt(
            name="generate-analysis-script",
            description="Generate a reusable analysis script based on lens configuration",
            arguments=[
                {
                    "name": "lens_id",
                    "description": "The lens ID to generate script for",
                    "required": True
                },
                {
                    "name": "output_format",
                    "description": "Output format: python, curl, or typescript",
                    "required": False
                }
            ]
        ),
        Prompt(
            name="quick-check",
            description="Quickly check a single text for narrative signals",
            arguments=[
                {
                    "name": "text",
                    "description": "The text to analyze",
                    "required": True
                }
            ]
        )
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: Optional[dict] = None) -> GetPromptResult:
    """Get a prompt by name."""
    args = arguments or {}

    if name == "configure-lens":
        lens_type = args.get("lens_type")
        if lens_type and lens_type in ALL_LENSES:
            lens = ALL_LENSES[lens_type]
            return GetPromptResult(
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=f"""I want to configure the {lens['name']} lens for my analysis.

**Lens Details:**
- Industry: {lens.get('industry', 'General')}
- Description: {lens['description']}
- Primary Dimensions: {', '.join(lens['primary_dimensions'])}
- Watch Modes: {', '.join(lens.get('watch_modes', ['None']))}

Please help me configure this lens by asking the following questions one at a time:
{chr(10).join(f'- {q}' for q in lens.get('questions', ['What texts would you like to analyze?']))}

After gathering my answers, generate the analysis configuration."""
                        )
                    )
                ]
            )
        else:
            return GetPromptResult(
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text="""I want to set up narrative analysis for my use case. Please help me select and configure the right lens.

**Available Killer App Lenses:**

1. **Denial Messaging QA** (denial-messaging)
   - For: Fintech / Insurance
   - Detects shadow-zone language in claim denials

2. **Crisis Pre-Flight** (crisis-preflight)
   - For: PR / Communications
   - Tests crisis statements before release

3. **Support Triage** (support-triage)
   - For: Customer Success
   - Prioritizes tickets by narrative distress

**Standard Lenses:**
- General Analysis
- Employee Engagement Health

Ask me about my use case and recommend the best lens, then walk me through configuration."""
                        )
                    )
                ]
            )

    elif name == "analyze-with-lens":
        lens_id = args.get("lens_id", "general-analysis")
        texts_json = args.get("texts", "[]")
        lens = ALL_LENSES.get(lens_id, ALL_LENSES["general-analysis"])

        return GetPromptResult(
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""Run analysis using the **{lens['name']}** lens.

**Lens Configuration:**
- Primary Dimensions: {', '.join(lens['primary_dimensions'])}
- Watch Modes: {', '.join(lens.get('watch_modes', ['All']))}
- Alert Thresholds: {json.dumps(lens.get('alert_thresholds', {}))}

**Texts to Analyze:**
{texts_json}

Please:
1. Use the `observer_analyze_batch` tool to analyze these texts
2. Apply the lens thresholds to flag concerning results
3. Summarize findings with the lens KPIs: {', '.join(lens.get('kpis', ['Score distribution']))}
4. Provide actionable recommendations"""
                    )
                )
            ]
        )

    elif name == "generate-analysis-script":
        lens_id = args.get("lens_id", "general-analysis")
        output_format = args.get("output_format", "python")
        lens = ALL_LENSES.get(lens_id, ALL_LENSES["general-analysis"])

        return GetPromptResult(
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""Generate a reusable {output_format} script for the **{lens['name']}** lens.

**Lens Configuration:**
```json
{json.dumps(lens, indent=2)}
```

The script should:
1. Connect to the Observatory backend at {BACKEND_URL}
2. Accept a list of texts as input
3. Apply the lens configuration (dimensions, thresholds, watch modes)
4. Output results in a structured format with alerts
5. Be production-ready with error handling

Generate the complete script."""
                    )
                )
            ]
        )

    elif name == "quick-check":
        text = args.get("text", "")
        return GetPromptResult(
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""Quick narrative check on this text:

"{text}"

Use the `observer_analyze_text` tool to analyze it, then tell me:
1. The narrative mode detected
2. Key dimension scores (agency, perceived_justice, belonging)
3. Any concerning signals
4. Which lens would be most appropriate for deeper analysis"""
                    )
                )
            ]
        )

    return GetPromptResult(messages=[])


# =============================================================================
# TOOLS - Observer agent capabilities
# =============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List Observer agent tools."""
    return [
        # Lens Management
        Tool(
            name="observer_list_lenses",
            description="List all available analysis lenses with their configurations",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["all", "killer-apps", "standard"],
                        "default": "all"
                    }
                }
            }
        ),
        Tool(
            name="observer_get_lens",
            description="Get detailed configuration for a specific lens",
            inputSchema={
                "type": "object",
                "properties": {
                    "lens_id": {
                        "type": "string",
                        "description": "Lens ID (e.g., 'denial-messaging', 'crisis-preflight')"
                    }
                },
                "required": ["lens_id"]
            }
        ),
        Tool(
            name="observer_recommend_lens",
            description="Get lens recommendations based on use case description",
            inputSchema={
                "type": "object",
                "properties": {
                    "use_case": {
                        "type": "string",
                        "description": "Description of what you want to analyze and why"
                    },
                    "industry": {
                        "type": "string",
                        "description": "Your industry (optional)"
                    }
                },
                "required": ["use_case"]
            }
        ),

        # Analysis Tools
        Tool(
            name="observer_analyze_text",
            description="Analyze a single text and return full narrative analysis with coordinates, mode, and force field",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze"
                    },
                    "lens_id": {
                        "type": "string",
                        "description": "Optional lens to apply thresholds",
                        "default": "general-analysis"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="observer_analyze_batch",
            description="Analyze multiple texts with lens configuration applied",
            inputSchema={
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of texts to analyze"
                    },
                    "lens_id": {
                        "type": "string",
                        "description": "Lens ID to apply",
                        "default": "general-analysis"
                    },
                    "include_alerts": {
                        "type": "boolean",
                        "default": True
                    }
                },
                "required": ["texts"]
            }
        ),
        Tool(
            name="observer_compare_texts",
            description="Compare two groups of texts (e.g., before/after, group A/B)",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_a": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "texts": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "group_b": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "texts": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "lens_id": {
                        "type": "string",
                        "default": "general-analysis"
                    }
                },
                "required": ["group_a", "group_b"]
            }
        ),

        # Script Generation
        Tool(
            name="observer_generate_script",
            description="Generate a reusable analysis script for a lens configuration",
            inputSchema={
                "type": "object",
                "properties": {
                    "lens_id": {
                        "type": "string",
                        "description": "Lens to generate script for"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["python", "typescript", "curl"],
                        "default": "python"
                    },
                    "include_visualization": {
                        "type": "boolean",
                        "default": False
                    }
                },
                "required": ["lens_id"]
            }
        ),

        # Session Management
        Tool(
            name="observer_save_session",
            description="Save current analysis session for later retrieval",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_name": {"type": "string"},
                    "lens_id": {"type": "string"},
                    "results": {"type": "object"}
                },
                "required": ["session_name"]
            }
        ),

        # Backend Status
        Tool(
            name="observer_status",
            description="Check Observer agent and Observatory backend status",
            inputSchema={"type": "object", "properties": {}}
        ),

        # =================================================================
        # WEBSITE ANALYSIS & REPORT GENERATION
        # =================================================================
        Tool(
            name="observer_analyze_website",
            description="""Fetch a website's text content and analyze it with a specified lens.

This is the main entry point for website analysis. It will:
1. Fetch the webpage content
2. Extract meaningful text (paragraphs, headings, etc.)
3. Analyze each text segment through the Observatory
4. Return aggregated results with alerts

Use this for analyzing company websites, blog posts, press releases, etc.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to analyze (e.g., 'https://company.com/about')"
                    },
                    "lens_id": {
                        "type": "string",
                        "description": "Lens to apply (default: general-analysis)",
                        "default": "general-analysis"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Optional context about what you're looking for"
                    },
                    "max_segments": {
                        "type": "integer",
                        "description": "Max text segments to analyze (default: 20)",
                        "default": 20
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="observer_generate_report",
            description="""Generate a comprehensive narrative analysis report.

Creates a detailed report including:
- Executive summary
- Dimension analysis (agency, justice, belonging)
- Force field analysis (attractors vs detractors)
- Mode distribution
- Risk alerts and recommendations
- Visualizations (ASCII charts)

Use after running observer_analyze_website or observer_analyze_batch.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "analysis_results": {
                        "type": "object",
                        "description": "Results from a previous analysis"
                    },
                    "lens_id": {
                        "type": "string",
                        "description": "Lens that was used",
                        "default": "general-analysis"
                    },
                    "title": {
                        "type": "string",
                        "description": "Report title"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context for the report"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "json", "text"],
                        "default": "markdown"
                    }
                },
                "required": ["analysis_results"]
            }
        ),
        Tool(
            name="observer_full_audit",
            description="""Run a complete narrative audit on a website with full report.

This is the all-in-one tool that:
1. Fetches website content
2. Analyzes with specified lens
3. Generates comprehensive report
4. Provides actionable recommendations

Perfect for one-command website audits.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Website URL to audit"
                    },
                    "lens_id": {
                        "type": "string",
                        "description": "Lens to use (or 'auto' for recommendation)",
                        "default": "general-analysis"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "What are you looking for? (e.g., 'check for hostile language')"
                    },
                    "report_title": {
                        "type": "string",
                        "description": "Title for the report"
                    }
                },
                "required": ["url"]
            }
        ),

        # =================================================================
        # EMERGENT LANGUAGE RESEARCH TOOLS
        # =================================================================
        Tool(
            name="research_grammar_deletion",
            description="""Analyze which grammatical structures are coordination-necessary vs decorative.

Systematically removes linguistic features (articles, pronouns, tense markers, etc.)
and measures projection drift in the Observatory manifold.

Features that cause large drift when removed are NECESSARY for coordination.
Features with minimal impact are DECORATIVE cultural ornamentation.

Returns a ranked list of grammar features by coordination importance.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Drift threshold for 'necessary' classification (default: 0.3)",
                        "default": 0.3
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="research_legibility",
            description="""Compute legibility score for a message.

Measures how interpretable the message is to humans:
- 0.0 = completely opaque (emergent AI code)
- 1.0 = natural language equivalent

Components:
- Vocabulary coverage (overlap with common words)
- Syntactic regularity (standard grammar)
- Mode confidence (clear coordination pattern)
- Semantic density (information per token)

Also classifies communication regime: NATURAL, TECHNICAL, COMPRESSED, or OPAQUE.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze for legibility"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="research_legibility_stream",
            description="""Analyze a stream of messages for legibility patterns.

Monitors communication over time to detect:
- Phase transitions (when regime shifts)
- Drift from baseline legibility
- Trend direction (improving, degrading, stable)

Use for analyzing AI-to-AI communication logs or evolving protocols.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of messages to analyze"
                    },
                    "alert_threshold": {
                        "type": "number",
                        "description": "Trigger alert if legibility drops below this",
                        "default": 0.3
                    }
                },
                "required": ["messages"]
            }
        ),
        Tool(
            name="research_protocol_snapshot",
            description="""Analyze a communication protocol at a single point in time.

Returns:
- Vocabulary size and unique patterns
- Stability score (protocol consistency)
- Coordination centroid in manifold
- Mode distribution
- Estimated evolution stage

Use for snapshots during training or to compare different AI systems.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Sample messages from the protocol"
                    },
                    "checkpoint_id": {
                        "type": "string",
                        "description": "Identifier for this snapshot",
                        "default": "snapshot"
                    }
                },
                "required": ["messages"]
            }
        ),
        Tool(
            name="research_evolution_track",
            description="""Track evolution of a communication protocol across training.

Analyzes multiple checkpoints to detect:
- Stage transitions (RANDOM -> STABILIZING -> COMPOSITIONAL -> OSSIFIED)
- Compositionality emergence (when structured combination appears)
- Ossification point (when evolution stops)
- Trajectory length through coordination space

Provide messages grouped by checkpoint/training step.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "checkpoint_messages": {
                        "type": "object",
                        "description": "Dict mapping checkpoint_id to list of messages",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "required": ["checkpoint_messages"]
            }
        ),
        Tool(
            name="research_calibrate",
            description="""Compare human text against minimal coordination codes.

Finds the coordination core - meaning with the adjectives burned off.

Classifies linguistic features as:
- NECESSARY: Present in both, high drift when removed
- EFFICIENT: In minimal codes, improves performance
- DECORATIVE: Only in human language, no coordination function
- NOISE: Random variation

Use to identify what grammar humans use that AIs skip.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "human_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Natural human language texts"
                    },
                    "minimal_texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Minimal/stripped coordination codes"
                    }
                },
                "required": ["human_texts", "minimal_texts"]
            }
        ),
        Tool(
            name="research_coordination_core",
            description="""Extract the coordination core from a text.

Strips decorative features and returns only coordination-necessary language.
This is the minimal structure needed for two minds to achieve alignment.

Think of it as 'distilling meaning' - removing cultural ornamentation
to expose the bare coordination scaffold.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to extract coordination core from"
                    }
                },
                "required": ["text"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    """Handle Observer tool calls."""
    try:
        # --- Lens Management ---
        if name == "observer_list_lenses":
            category = arguments.get("category", "all")
            if category == "killer-apps":
                lenses = KILLER_APP_LENSES
            elif category == "standard":
                lenses = STANDARD_LENSES
            else:
                lenses = ALL_LENSES

            result = {
                "count": len(lenses),
                "lenses": [
                    {
                        "id": lid,
                        "name": l["name"],
                        "description": l["description"],
                        "industry": l.get("industry", "General"),
                        "primary_dimensions": l["primary_dimensions"]
                    }
                    for lid, l in lenses.items()
                ]
            }
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )

        elif name == "observer_get_lens":
            lens_id = arguments["lens_id"]
            if lens_id not in ALL_LENSES:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Unknown lens: {lens_id}. Use observer_list_lenses to see available options.")]
                )
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(ALL_LENSES[lens_id], indent=2))]
            )

        elif name == "observer_recommend_lens":
            use_case = arguments["use_case"].lower()
            industry = arguments.get("industry", "").lower()

            recommendations = []

            # Score each lens based on keyword matching
            for lid, lens in ALL_LENSES.items():
                score = 0
                lens_industry = lens.get("industry", "").lower()
                lens_desc = lens["description"].lower()

                # Industry match
                if industry and industry in lens_industry:
                    score += 3

                # Keyword matching
                keywords = {
                    "denial-messaging": ["denial", "claim", "reject", "insurance", "fintech", "compliance", "regulatory"],
                    "crisis-preflight": ["crisis", "statement", "pr", "communication", "backlash", "media"],
                    "support-triage": ["support", "ticket", "customer", "churn", "escalation", "help"],
                    "engagement-health": ["employee", "engagement", "morale", "culture", "internal"]
                }

                for kw in keywords.get(lid, []):
                    if kw in use_case:
                        score += 2

                if score > 0:
                    recommendations.append({
                        "lens_id": lid,
                        "name": lens["name"],
                        "score": score,
                        "reason": lens["description"]
                    })

            recommendations.sort(key=lambda x: x["score"], reverse=True)

            if not recommendations:
                recommendations = [{
                    "lens_id": "general-analysis",
                    "name": "General Narrative Analysis",
                    "score": 1,
                    "reason": "No specific match found - using general analysis"
                }]

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({
                    "use_case": arguments["use_case"],
                    "recommendations": recommendations[:3]
                }, indent=2))]
            )

        # --- Analysis Tools ---
        elif name == "observer_analyze_text":
            text = arguments["text"]
            lens_id = arguments.get("lens_id", "general-analysis")
            lens = ALL_LENSES.get(lens_id, ALL_LENSES["general-analysis"])

            # Call Observatory backend
            result = await call_backend("POST", "/api/v2/analyze", {
                "text": text,
                "include_forces": True
            })

            # Apply lens thresholds
            alerts = []
            if "coordinates" in result:
                coords = result["coordinates"]
                for dim, threshold in lens.get("alert_thresholds", {}).items():
                    if dim in coords and coords[dim] < threshold:
                        alerts.append({
                            "dimension": dim,
                            "value": coords[dim],
                            "threshold": threshold,
                            "severity": "high" if coords[dim] < threshold - 0.5 else "medium"
                        })

            # Check watch modes
            mode = result.get("mode", {}).get("primary_mode", "UNKNOWN")
            if mode in lens.get("watch_modes", []):
                alerts.append({
                    "type": "watch_mode",
                    "mode": mode,
                    "severity": "high"
                })

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({
                    "lens": lens_id,
                    "analysis": result,
                    "alerts": alerts,
                    "alert_count": len(alerts)
                }, indent=2))]
            )

        elif name == "observer_analyze_batch":
            texts = arguments["texts"]
            lens_id = arguments.get("lens_id", "general-analysis")
            lens = ALL_LENSES.get(lens_id, ALL_LENSES["general-analysis"])

            # Analyze all texts
            results = []
            all_alerts = []

            for i, text in enumerate(texts):
                try:
                    result = await call_backend("POST", "/api/v2/analyze", {
                        "text": text,
                        "include_forces": True
                    })

                    alerts = []
                    if "coordinates" in result:
                        coords = result["coordinates"]
                        for dim, threshold in lens.get("alert_thresholds", {}).items():
                            if dim in coords and coords[dim] < threshold:
                                alerts.append({
                                    "text_index": i,
                                    "dimension": dim,
                                    "value": coords[dim],
                                    "threshold": threshold
                                })

                    mode = result.get("mode", {}).get("primary_mode", "UNKNOWN")
                    if mode in lens.get("watch_modes", []):
                        alerts.append({
                            "text_index": i,
                            "type": "watch_mode",
                            "mode": mode
                        })

                    results.append({
                        "index": i,
                        "text_preview": text[:100] + "..." if len(text) > 100 else text,
                        "coordinates": result.get("coordinates"),
                        "mode": result.get("mode", {}).get("primary_mode"),
                        "alert_count": len(alerts)
                    })
                    all_alerts.extend(alerts)
                except Exception as e:
                    results.append({"index": i, "error": str(e)})

            # Compute summary statistics
            valid_results = [r for r in results if "coordinates" in r and r["coordinates"]]
            if valid_results:
                summary = {
                    "total_texts": len(texts),
                    "successful": len(valid_results),
                    "total_alerts": len(all_alerts),
                    "average_coordinates": {
                        dim: sum(r["coordinates"].get(dim, 0) for r in valid_results) / len(valid_results)
                        for dim in lens["primary_dimensions"]
                    },
                    "mode_distribution": {}
                }
                for r in valid_results:
                    mode = r.get("mode", "UNKNOWN")
                    summary["mode_distribution"][mode] = summary["mode_distribution"].get(mode, 0) + 1
            else:
                summary = {"total_texts": len(texts), "successful": 0, "total_alerts": 0}

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({
                    "lens": lens_id,
                    "summary": summary,
                    "results": results,
                    "alerts": all_alerts
                }, indent=2))]
            )

        elif name == "observer_compare_texts":
            group_a = arguments["group_a"]
            group_b = arguments["group_b"]
            lens_id = arguments.get("lens_id", "general-analysis")

            # Analyze both groups
            async def analyze_group(group):
                results = []
                for text in group["texts"]:
                    try:
                        r = await call_backend("POST", "/api/v2/analyze", {"text": text, "include_forces": True})
                        results.append(r)
                    except:
                        pass
                return results

            results_a = await analyze_group(group_a)
            results_b = await analyze_group(group_b)

            def compute_centroid(results):
                if not results:
                    return {}
                return {
                    "agency": sum(r.get("coordinates", {}).get("agency", 0) for r in results) / len(results),
                    "perceived_justice": sum(r.get("coordinates", {}).get("perceived_justice", 0) for r in results) / len(results),
                    "belonging": sum(r.get("coordinates", {}).get("belonging", 0) for r in results) / len(results)
                }

            centroid_a = compute_centroid(results_a)
            centroid_b = compute_centroid(results_b)

            delta = {
                dim: centroid_b.get(dim, 0) - centroid_a.get(dim, 0)
                for dim in ["agency", "perceived_justice", "belonging"]
            }

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({
                    "comparison": {
                        "group_a": {"name": group_a["name"], "n": len(results_a), "centroid": centroid_a},
                        "group_b": {"name": group_b["name"], "n": len(results_b), "centroid": centroid_b},
                        "delta": delta,
                        "interpretation": {
                            dim: "improved" if delta[dim] > 0.2 else "declined" if delta[dim] < -0.2 else "stable"
                            for dim in delta
                        }
                    }
                }, indent=2))]
            )

        # --- Script Generation ---
        elif name == "observer_generate_script":
            lens_id = arguments["lens_id"]
            language = arguments.get("language", "python")
            lens = ALL_LENSES.get(lens_id, ALL_LENSES["general-analysis"])

            if language == "python":
                script = f'''#!/usr/bin/env python3
"""
{lens["name"]} Analysis Script
Generated by Observer Agent
"""

import httpx
import json
from typing import List, Dict

BACKEND_URL = "{BACKEND_URL}"
LENS_CONFIG = {json.dumps(lens, indent=4)}

def analyze_texts(texts: List[str]) -> Dict:
    """Analyze texts with the {lens["name"]} lens."""
    results = []
    alerts = []

    with httpx.Client(timeout=120.0) as client:
        for i, text in enumerate(texts):
            response = client.post(
                f"{{BACKEND_URL}}/api/v2/analyze",
                json={{"text": text, "include_forces": True}}
            )
            result = response.json()

            # Check thresholds
            coords = result.get("coordinates", {{}})
            for dim, threshold in LENS_CONFIG.get("alert_thresholds", {{}}).items():
                if coords.get(dim, 0) < threshold:
                    alerts.append({{
                        "text_index": i,
                        "dimension": dim,
                        "value": coords.get(dim),
                        "threshold": threshold
                    }})

            # Check watch modes
            mode = result.get("mode", {{}}).get("primary_mode")
            if mode in LENS_CONFIG.get("watch_modes", []):
                alerts.append({{"text_index": i, "watch_mode": mode}})

            results.append({{
                "index": i,
                "coordinates": coords,
                "mode": mode
            }})

    return {{"results": results, "alerts": alerts, "total": len(texts)}}

if __name__ == "__main__":
    import sys

    # Example usage
    texts = [
        "Your claim has been denied due to pre-existing conditions.",
        "We are committed to supporting our valued customers."
    ]

    output = analyze_texts(texts)
    print(json.dumps(output, indent=2))
'''
            elif language == "typescript":
                script = f'''/**
 * {lens["name"]} Analysis Script
 * Generated by Observer Agent
 */

const BACKEND_URL = "{BACKEND_URL}";
const LENS_CONFIG = {json.dumps(lens, indent=2)};

interface AnalysisResult {{
  coordinates: Record<string, number>;
  mode: {{ primary_mode: string }};
}}

async function analyzeTexts(texts: string[]) {{
  const results = [];
  const alerts = [];

  for (let i = 0; i < texts.length; i++) {{
    const response = await fetch(`${{BACKEND_URL}}/api/v2/analyze`, {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ text: texts[i], include_forces: true }})
    }});

    const result: AnalysisResult = await response.json();
    const coords = result.coordinates || {{}};

    // Check thresholds
    for (const [dim, threshold] of Object.entries(LENS_CONFIG.alert_thresholds || {{}})) {{
      if ((coords[dim] || 0) < threshold) {{
        alerts.push({{ text_index: i, dimension: dim, value: coords[dim], threshold }});
      }}
    }}

    results.push({{ index: i, coordinates: coords, mode: result.mode?.primary_mode }});
  }}

  return {{ results, alerts, total: texts.length }};
}}

// Example usage
analyzeTexts([
  "Your claim has been denied due to pre-existing conditions.",
  "We are committed to supporting our valued customers."
]).then(console.log);
'''
            else:  # curl
                script = f'''#!/bin/bash
# {lens["name"]} Analysis Script
# Generated by Observer Agent

BACKEND_URL="{BACKEND_URL}"

analyze_text() {{
    curl -s -X POST "$BACKEND_URL/api/v2/analyze" \\
        -H "Content-Type: application/json" \\
        -d '{{"text": "'"$1"'", "include_forces": true}}'
}}

# Example usage
echo "Analyzing: Your claim has been denied..."
analyze_text "Your claim has been denied due to pre-existing conditions." | jq .

echo ""
echo "Analyzing: We are committed..."
analyze_text "We are committed to supporting our valued customers." | jq .
'''

            return CallToolResult(
                content=[TextContent(type="text", text=f"```{language}\n{script}\n```")]
            )

        # --- Status ---
        elif name == "observer_status":
            try:
                backend_status = await call_backend("GET", "/")
                backend_ok = True
            except:
                backend_status = {"error": "Cannot connect"}
                backend_ok = False

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({
                    "observer": {
                        "status": "running",
                        "version": "1.0.0",
                        "lenses_available": len(ALL_LENSES)
                    },
                    "backend": {
                        "status": "connected" if backend_ok else "disconnected",
                        "url": BACKEND_URL,
                        "details": backend_status
                    },
                    "aperture_studio": {
                        "url": APERTURE_URL
                    }
                }, indent=2))]
            )

        # --- Website Analysis ---
        elif name == "observer_analyze_website":
            url = arguments["url"]
            lens_id = arguments.get("lens_id", "general-analysis")
            prompt = arguments.get("prompt", "")
            max_segments = arguments.get("max_segments", 20)
            lens = ALL_LENSES.get(lens_id, ALL_LENSES["general-analysis"])

            # Fetch website content
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                try:
                    response = await client.get(url, headers={
                        "User-Agent": "Mozilla/5.0 (compatible; ObserverBot/1.0)"
                    })
                    response.raise_for_status()
                    html = response.text
                except Exception as e:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Failed to fetch {url}: {str(e)}")]
                    )

            # Extract text from HTML (simple extraction)
            import re
            # Remove scripts, styles, etc.
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)

            # Extract paragraphs and headings
            segments = []
            for match in re.finditer(r'<(p|h[1-6]|li|blockquote)[^>]*>(.*?)</\1>', html, re.DOTALL | re.IGNORECASE):
                text = re.sub(r'<[^>]+>', '', match.group(2))  # Remove tags
                text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
                if len(text) > 50:  # Only meaningful segments
                    segments.append(text)

            if not segments:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"No meaningful text found at {url}")]
                )

            # Limit segments
            segments = segments[:max_segments]

            # Analyze each segment
            results = []
            all_alerts = []

            for i, text in enumerate(segments):
                try:
                    result = await call_backend("POST", "/api/v2/analyze", {
                        "text": text,
                        "include_forces": True
                    })

                    alerts = []
                    if "coordinates" in result:
                        coords = result["coordinates"]
                        for dim, threshold in lens.get("alert_thresholds", {}).items():
                            if dim in coords and coords[dim] < threshold:
                                alerts.append({
                                    "segment_index": i,
                                    "dimension": dim,
                                    "value": coords[dim],
                                    "threshold": threshold,
                                    "severity": "high" if coords[dim] < threshold - 0.5 else "medium"
                                })

                    mode = result.get("mode", {}).get("primary_mode", "UNKNOWN")
                    if mode in lens.get("watch_modes", []):
                        alerts.append({
                            "segment_index": i,
                            "type": "watch_mode",
                            "mode": mode,
                            "severity": "high"
                        })

                    results.append({
                        "index": i,
                        "text_preview": text[:100] + "..." if len(text) > 100 else text,
                        "coordinates": result.get("coordinates"),
                        "mode": result.get("mode", {}).get("primary_mode"),
                        "force_field": result.get("force_field"),
                        "alert_count": len(alerts)
                    })
                    all_alerts.extend(alerts)
                except Exception as e:
                    results.append({"index": i, "error": str(e)})

            # Compute summary statistics
            valid_results = [r for r in results if "coordinates" in r and r["coordinates"]]
            if valid_results:
                summary = {
                    "url": url,
                    "lens_used": lens_id,
                    "total_segments": len(segments),
                    "successful": len(valid_results),
                    "total_alerts": len(all_alerts),
                    "high_severity_alerts": len([a for a in all_alerts if a.get("severity") == "high"]),
                    "average_coordinates": {
                        dim: sum(r["coordinates"].get(dim, 0) for r in valid_results) / len(valid_results)
                        for dim in lens["primary_dimensions"]
                    },
                    "mode_distribution": {},
                    "dominant_mode": None
                }
                for r in valid_results:
                    mode = r.get("mode", "UNKNOWN")
                    summary["mode_distribution"][mode] = summary["mode_distribution"].get(mode, 0) + 1
                if summary["mode_distribution"]:
                    summary["dominant_mode"] = max(summary["mode_distribution"], key=summary["mode_distribution"].get)
            else:
                summary = {"url": url, "total_segments": len(segments), "successful": 0, "total_alerts": 0}

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({
                    "website_analysis": {
                        "summary": summary,
                        "results": results,
                        "alerts": all_alerts,
                        "context": prompt
                    }
                }, indent=2))]
            )

        elif name == "observer_generate_report":
            analysis = arguments.get("analysis_results", {})
            lens_id = arguments.get("lens_id", "general-analysis")
            title = arguments.get("title", "Narrative Analysis Report")
            context = arguments.get("context", "")
            format_type = arguments.get("format", "markdown")
            lens = ALL_LENSES.get(lens_id, ALL_LENSES["general-analysis"])

            # Extract data from analysis
            summary = analysis.get("website_analysis", {}).get("summary", analysis.get("summary", {}))
            results = analysis.get("website_analysis", {}).get("results", analysis.get("results", []))
            alerts = analysis.get("website_analysis", {}).get("alerts", analysis.get("alerts", []))

            if format_type == "markdown":
                # Generate Markdown report
                report = f"""# {title}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Lens:** {lens['name']}
**Source:** {summary.get('url', 'N/A')}

---

## Executive Summary

Analyzed **{summary.get('total_segments', 0)}** text segments with **{summary.get('successful', 0)}** successful analyses.

### Key Findings

| Metric | Value |
|--------|-------|
| Total Alerts | {summary.get('total_alerts', 0)} |
| High Severity | {summary.get('high_severity_alerts', 0)} |
| Dominant Mode | {summary.get('dominant_mode', 'N/A')} |

"""
                # Add dimension analysis
                avg_coords = summary.get('average_coordinates', {})
                if avg_coords:
                    report += """### Dimension Analysis

| Dimension | Average Score | Assessment |
|-----------|--------------|------------|
"""
                    for dim, val in avg_coords.items():
                        assessment = "Positive" if val > 0.5 else "Negative" if val < -0.5 else "Neutral"
                        report += f"| {dim.replace('_', ' ').title()} | {val:.2f} | {assessment} |\n"

                # Add mode distribution
                mode_dist = summary.get('mode_distribution', {})
                if mode_dist:
                    report += """\n### Mode Distribution

| Mode | Count | Percentage |
|------|-------|------------|
"""
                    total = sum(mode_dist.values())
                    for mode, count in sorted(mode_dist.items(), key=lambda x: -x[1]):
                        pct = (count / total * 100) if total > 0 else 0
                        report += f"| {mode} | {count} | {pct:.1f}% |\n"

                # Add alerts section
                if alerts:
                    report += f"""\n---

## Alerts ({len(alerts)} total)

"""
                    high_alerts = [a for a in alerts if a.get('severity') == 'high']
                    if high_alerts:
                        report += "### High Severity Alerts\n\n"
                        for alert in high_alerts[:10]:
                            if alert.get('type') == 'watch_mode':
                                report += f"- **Segment {alert['segment_index']}**: Watch mode triggered - `{alert['mode']}`\n"
                            else:
                                report += f"- **Segment {alert['segment_index']}**: `{alert['dimension']}` = {alert['value']:.2f} (threshold: {alert['threshold']})\n"

                # Add recommendations
                report += f"""\n---

## Recommendations

Based on the **{lens['name']}** lens analysis:

"""
                if summary.get('high_severity_alerts', 0) > 0:
                    report += "1. **Urgent Review Required**: High severity alerts detected. Review flagged segments immediately.\n"
                if avg_coords.get('perceived_justice', 0) < -0.5:
                    report += "2. **Justice Perception Issue**: Low perceived justice scores indicate potential trust/fairness concerns.\n"
                if avg_coords.get('agency', 0) < -0.5:
                    report += "3. **Agency Concern**: Low agency scores suggest disempowering language patterns.\n"
                if avg_coords.get('belonging', 0) < -0.5:
                    report += "4. **Belonging Gap**: Low belonging scores indicate potential alienation in messaging.\n"
                if not any([summary.get('high_severity_alerts', 0) > 0,
                           avg_coords.get('perceived_justice', 0) < -0.5,
                           avg_coords.get('agency', 0) < -0.5,
                           avg_coords.get('belonging', 0) < -0.5]):
                    report += "1. **Generally Positive**: No major concerns detected. Continue monitoring.\n"

                report += f"""\n---

*Report generated by Observer Agent using the Cultural Soliton Observatory*
"""
                return CallToolResult(content=[TextContent(type="text", text=report)])

            else:  # JSON format
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps({
                        "report": {
                            "title": title,
                            "generated": datetime.now().isoformat(),
                            "lens": lens_id,
                            "summary": summary,
                            "alerts": alerts,
                            "context": context
                        }
                    }, indent=2))]
                )

        elif name == "observer_full_audit":
            url = arguments["url"]
            lens_id = arguments.get("lens_id", "general-analysis")
            prompt = arguments.get("prompt", "")
            report_title = arguments.get("report_title", f"Narrative Audit: {url}")

            # If lens is 'auto', recommend one based on prompt
            if lens_id == "auto" and prompt:
                keywords = {
                    "denial-messaging": ["denial", "claim", "reject", "insurance", "fintech"],
                    "crisis-preflight": ["crisis", "statement", "pr", "backlash"],
                    "support-triage": ["support", "ticket", "customer", "churn"],
                }
                prompt_lower = prompt.lower()
                best_lens = "general-analysis"
                best_score = 0
                for lid, kws in keywords.items():
                    score = sum(1 for kw in kws if kw in prompt_lower)
                    if score > best_score:
                        best_score = score
                        best_lens = lid
                lens_id = best_lens

            lens = ALL_LENSES.get(lens_id, ALL_LENSES["general-analysis"])

            # Step 1: Fetch and analyze website
            analyze_result = await call_tool("observer_analyze_website", {
                "url": url,
                "lens_id": lens_id,
                "prompt": prompt,
                "max_segments": 25
            })

            # Parse the analysis result
            try:
                analysis_text = analyze_result.content[0].text
                analysis_data = json.loads(analysis_text)
            except:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Analysis failed: {analyze_result.content[0].text}")]
                )

            # Step 2: Generate report
            report_result = await call_tool("observer_generate_report", {
                "analysis_results": analysis_data,
                "lens_id": lens_id,
                "title": report_title,
                "context": prompt,
                "format": "markdown"
            })

            return report_result

        # =================================================================
        # EMERGENT LANGUAGE RESEARCH TOOL HANDLERS
        # =================================================================

        elif name == "research_grammar_deletion":
            text = arguments["text"]
            threshold = arguments.get("threshold", 0.3)

            try:
                from research.grammar_deletion_test import GrammarDeletionAnalyzer

                analyzer = GrammarDeletionAnalyzer(drift_threshold=threshold)
                analysis = await analyzer.analyze_text(text)

                result = {
                    "original_text": analysis.original_text[:200] + "..." if len(analysis.original_text) > 200 else analysis.original_text,
                    "original_mode": analysis.original_mode,
                    "original_projection": analysis.original_projection,
                    "necessary_features": analysis.necessary_features,
                    "decorative_features": analysis.decorative_features,
                    "coordination_core": analysis.coordination_core,
                    "feature_ranking": [
                        {
                            "feature": d.feature_name,
                            "drift": d.projection_drift,
                            "classification": "necessary" if d.projection_drift > threshold else "decorative",
                            "mode_changed": d.mode_changed,
                            "axis_drifts": d.axis_drifts
                        }
                        for d in analysis.deletions
                    ]
                }

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result, indent=2))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Grammar deletion analysis failed: {str(e)}")]
                )

        elif name == "research_legibility":
            text = arguments["text"]

            try:
                from research.legibility_analyzer import compute_legibility

                result = await compute_legibility(text)

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result, indent=2))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Legibility analysis failed: {str(e)}")]
                )

        elif name == "research_legibility_stream":
            messages = arguments["messages"]
            alert_threshold = arguments.get("alert_threshold", 0.3)

            try:
                from research.legibility_analyzer import LegibilityAnalyzer

                analyzer = LegibilityAnalyzer(alert_threshold=alert_threshold)
                result = await analyzer.analyze_stream(messages, return_all=True)

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result, indent=2))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Legibility stream analysis failed: {str(e)}")]
                )

        elif name == "research_protocol_snapshot":
            messages = arguments["messages"]
            checkpoint_id = arguments.get("checkpoint_id", "snapshot")

            try:
                from research.evolution_tracker import EvolutionTracker
                from datetime import datetime

                tracker = EvolutionTracker()

                # Create timestamps for each message (spaced evenly)
                now = datetime.now()
                texts_over_time = [
                    (now, msg) for msg in messages
                ]

                if len(texts_over_time) >= 2:
                    analysis = await tracker.track_evolution(texts_over_time)
                    result = {
                        "checkpoint": checkpoint_id,
                        "messages_analyzed": len(messages),
                        "current_stage": analysis.current_stage.value,
                        "trajectory_length": len(analysis.trajectory),
                        "total_distance": analysis.total_distance_traveled,
                        "net_displacement": analysis.net_displacement,
                        "interpretation": analysis.interpretation
                    }
                else:
                    # Single message - just project it
                    projection = await tracker.project_text(messages[0])
                    result = {
                        "checkpoint": checkpoint_id,
                        "messages_analyzed": 1,
                        "projection": projection
                    }

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result, indent=2))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Protocol snapshot failed: {str(e)}")]
                )

        elif name == "research_evolution_track":
            checkpoint_messages = arguments["checkpoint_messages"]

            try:
                from research.evolution_tracker import EvolutionTracker
                from datetime import datetime, timedelta

                tracker = EvolutionTracker()

                # Convert checkpoint dict to timestamped list
                texts_over_time = []
                base_time = datetime.now()
                sorted_checkpoints = sorted(checkpoint_messages.keys())

                for i, cp_id in enumerate(sorted_checkpoints):
                    cp_time = base_time + timedelta(hours=i)
                    for msg in checkpoint_messages[cp_id]:
                        texts_over_time.append((cp_time, msg))

                if len(texts_over_time) < 2:
                    return CallToolResult(
                        content=[TextContent(type="text", text="Need at least 2 messages across checkpoints")]
                    )

                analysis = await tracker.track_evolution(texts_over_time)

                result = {
                    "total_checkpoints": len(sorted_checkpoints),
                    "total_messages": len(texts_over_time),
                    "current_stage": analysis.current_stage.value,
                    "trajectory_length": len(analysis.trajectory),
                    "total_distance": analysis.total_distance_traveled,
                    "net_displacement": analysis.net_displacement,
                    "compositionality": analysis.compositionality.__dict__ if hasattr(analysis.compositionality, '__dict__') else str(analysis.compositionality),
                    "stage_history": [
                        {"time": str(t[0]), "stage": t[1].value}
                        for t in analysis.stage_history
                    ],
                    "interpretation": analysis.interpretation
                }

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result, indent=2))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Evolution tracking failed: {str(e)}")]
                )

        elif name == "research_calibrate":
            human_texts = arguments["human_texts"]
            minimal_texts = arguments["minimal_texts"]

            try:
                from research.calibration_baseline import CalibrationBaseline

                baseline = CalibrationBaseline()
                baseline.load_human_corpus(human_texts)  # Sync method
                baseline.load_minimal_corpus(minimal_texts)  # Sync method
                calibration = await baseline.full_calibration()

                result = calibration.to_dict()

                await baseline.close()

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result, indent=2))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Calibration analysis failed: {str(e)}")]
                )

        elif name == "research_coordination_core":
            text = arguments["text"]

            try:
                from research.grammar_deletion_test import GrammarDeletionAnalyzer

                analyzer = GrammarDeletionAnalyzer(drift_threshold=0.3)
                analysis = await analyzer.analyze_text(text)

                result = {
                    "original_text": text,
                    "coordination_core": analysis.coordination_core,
                    "removed_features": analysis.decorative_features,
                    "preserved_features": analysis.necessary_features,
                    "compression_ratio": len(analysis.coordination_core) / len(text) if text else 1.0
                }

                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result, indent=2))]
                )
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Coordination core extraction failed: {str(e)}")]
                )

        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")]
            )

    except httpx.ConnectError:
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"ERROR: Cannot connect to Observatory backend at {BACKEND_URL}\n\n"
                     "Start it with: cd backend && python -m uvicorn main:app --port 8000"
            )]
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")]
        )


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run the Observer MCP server."""
    print("Starting Observer Agent MCP Server...", flush=True)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
