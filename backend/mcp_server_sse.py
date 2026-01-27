"""
Cultural Soliton Observatory - MCP Server (SSE Transport)

This version uses Server-Sent Events for remote deployment on Render, Railway, etc.

Run with:
    uvicorn mcp_server_sse:app --host 0.0.0.0 --port 3001

Environment variables:
    OBSERVATORY_BACKEND_URL: URL of the FastAPI backend (default: http://127.0.0.1:8000)
    OBSERVATORY_DEFAULT_MODEL: Default embedding model (default: all-MiniLM-L6-v2)
"""

import os
import json
from typing import Optional
import httpx
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent, CallToolResult

# Configuration
BACKEND_URL = os.getenv("OBSERVATORY_BACKEND_URL", "http://127.0.0.1:8000")
DEFAULT_MODEL = os.getenv("OBSERVATORY_DEFAULT_MODEL", "all-MiniLM-L6-v2")

# Create MCP server
server = Server("cultural-soliton-observatory")
sse = SseServerTransport("/messages/")


# --- HTTP Client ---

async def call_backend(method: str, endpoint: str, data: Optional[dict] = None) -> dict:
    """Call the FastAPI backend."""
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


# --- Tool Definitions (same as stdio version) ---

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available observatory tools."""
    return [
        Tool(
            name="observatory_status",
            description="Check if the observatory backend is running and get current status.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="project_text",
            description="""Project a text onto the 3D cultural manifold (Agency, Fairness, Belonging axes).
Returns coordinates, narrative mode, and confidence score.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to analyze"},
                    "model_id": {"type": "string", "default": DEFAULT_MODEL}
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="project_batch",
            description="Project multiple texts onto the manifold. Returns projections and detected clusters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "texts": {"type": "array", "items": {"type": "string"}},
                    "model_id": {"type": "string", "default": DEFAULT_MODEL},
                    "detect_clusters": {"type": "boolean", "default": True}
                },
                "required": ["texts"]
            }
        ),
        Tool(
            name="measure_hypocrisy",
            description="Measure gap between stated values and actual narrative positions in a corpus.",
            inputSchema={
                "type": "object",
                "properties": {
                    "espoused_values": {
                        "type": "object",
                        "properties": {
                            "agency": {"type": "number"},
                            "fairness": {"type": "number"},
                            "belonging": {"type": "number"}
                        },
                        "required": ["agency", "fairness", "belonging"]
                    },
                    "texts": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["espoused_values", "texts"]
            }
        ),
        Tool(
            name="add_training_example",
            description="Add a labeled training example to improve projection quality.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "agency": {"type": "number", "minimum": -2, "maximum": 2},
                    "fairness": {"type": "number", "minimum": -2, "maximum": 2},
                    "belonging": {"type": "number", "minimum": -2, "maximum": 2},
                    "source": {"type": "string", "default": "mcp-agent"}
                },
                "required": ["text", "agency", "fairness", "belonging"]
            }
        ),
        Tool(
            name="train_projection",
            description="Train/retrain the projection model. Methods: ridge, gp, neural, cav.",
            inputSchema={
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["ridge", "gp", "neural", "cav"], "default": "ridge"},
                    "model_id": {"type": "string", "default": DEFAULT_MODEL}
                },
                "required": []
            }
        ),
        Tool(
            name="list_models",
            description="List available and loaded embedding models.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    """Handle tool calls."""
    try:
        if name == "observatory_status":
            result = await call_backend("GET", "/")
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, indent=2))])

        elif name == "project_text":
            data = {"text": arguments["text"], "model_id": arguments.get("model_id", DEFAULT_MODEL), "layer": -1}
            result = await call_backend("POST", "/analyze", data)
            formatted = {
                "text": result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"],
                "coordinates": result["vector"],
                "narrative_mode": result["mode"],
                "confidence": round(result["confidence"], 3)
            }
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(formatted, indent=2))])

        elif name == "project_batch":
            data = {
                "texts": arguments["texts"],
                "model_id": arguments.get("model_id", DEFAULT_MODEL),
                "layer": -1,
                "detect_clusters": arguments.get("detect_clusters", True)
            }
            result = await call_backend("POST", "/corpus/analyze", data)
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, indent=2))])

        elif name == "measure_hypocrisy":
            data = {"texts": arguments["texts"], "model_id": DEFAULT_MODEL, "layer": -1, "detect_clusters": False}
            result = await call_backend("POST", "/corpus/analyze", data)
            projections = result.get("projections", [])
            if not projections:
                return CallToolResult(content=[TextContent(type="text", text="No texts projected")])

            mean_agency = sum(p["vector"]["agency"] for p in projections) / len(projections)
            mean_fairness = sum(p["vector"]["fairness"] for p in projections) / len(projections)
            mean_belonging = sum(p["vector"]["belonging"] for p in projections) / len(projections)
            espoused = arguments["espoused_values"]

            delta_agency = abs(espoused["agency"] - mean_agency)
            delta_fairness = abs(espoused["fairness"] - mean_fairness)
            delta_belonging = abs(espoused["belonging"] - mean_belonging)
            total_gap = (delta_agency + delta_fairness + delta_belonging) / 3

            return CallToolResult(content=[TextContent(type="text", text=json.dumps({
                "espoused_values": espoused,
                "inferred_values": {"agency": round(mean_agency, 3), "fairness": round(mean_fairness, 3), "belonging": round(mean_belonging, 3)},
                "hypocrisy_gap": {"agency": round(delta_agency, 3), "fairness": round(delta_fairness, 3), "belonging": round(delta_belonging, 3), "total": round(total_gap, 3)},
                "interpretation": "Low" if total_gap < 0.5 else "Moderate" if total_gap < 1.0 else "High"
            }, indent=2))])

        elif name == "add_training_example":
            data = {
                "text": arguments["text"],
                "agency": arguments["agency"],
                "fairness": arguments["fairness"],
                "belonging": arguments["belonging"],
                "source": arguments.get("source", "mcp-agent")
            }
            result = await call_backend("POST", "/training/examples", data)
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, indent=2))])

        elif name == "train_projection":
            data = {"model_id": arguments.get("model_id", DEFAULT_MODEL), "method": arguments.get("method", "ridge"), "layer": -1}
            result = await call_backend("POST", "/training/train", data)
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, indent=2))])

        elif name == "list_models":
            available = await call_backend("GET", "/models/available")
            loaded = await call_backend("GET", "/models/loaded")
            return CallToolResult(content=[TextContent(type="text", text=json.dumps({"available": available, "loaded": loaded}, indent=2))])

        else:
            return CallToolResult(content=[TextContent(type="text", text=f"Unknown tool: {name}")])

    except httpx.ConnectError:
        return CallToolResult(content=[TextContent(type="text", text=f"Cannot connect to backend at {BACKEND_URL}")])
    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=f"Error: {str(e)}")])


# --- Starlette Routes ---

async def handle_sse(request):
    """Handle SSE connection for MCP."""
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())


async def handle_messages(request):
    """Handle POST messages for MCP."""
    await sse.handle_post_message(request.scope, request.receive, request._send)


async def health(request):
    """Health check endpoint."""
    try:
        status = await call_backend("GET", "/")
        return JSONResponse({"mcp_server": "running", "backend": status})
    except Exception as e:
        return JSONResponse({"mcp_server": "running", "backend": "unavailable", "error": str(e)})


# --- Starlette App ---

app = Starlette(
    debug=True,
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/sse", handle_sse, methods=["GET"]),
        Route("/messages/", handle_messages, methods=["POST"]),
    ]
)
