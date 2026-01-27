#!/usr/bin/env python3
"""
Observatory MCP Server - SSE Transport

Server-Sent Events transport layer for the Observatory MCP server,
enabling remote access from Claude Desktop, Claude Code, and other MCP clients.

The Observatory provides tools for analyzing cultural narratives in embedding space,
detecting coordination patterns, and monitoring AI communication.

Run locally:
    python server_sse.py

Run with uvicorn:
    uvicorn server_sse:app --host 0.0.0.0 --port 8080

Production (Render/Railway/Fly.io):
    Automatically uses PORT environment variable
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from sse_starlette.sse import EventSourceResponse

from server import server, call_tool, list_tools, list_resources, read_resource

# --- Configuration ---
VERSION = "2.0.0"
SERVICE_NAME = "observatory-mcp"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(SERVICE_NAME)

# Session storage for SSE connections
sessions: dict[str, dict[str, Any]] = {}


async def handle_sse(request: Request) -> EventSourceResponse:
    """Handle SSE connection for MCP messages."""
    session_id = str(uuid.uuid4())

    async def event_generator():
        # Send session ID
        yield {
            "event": "session",
            "data": json.dumps({"session_id": session_id}),
        }

        # Initialize session
        sessions[session_id] = {
            "queue": asyncio.Queue(),
            "connected": True,
        }

        try:
            # Send server info
            yield {
                "event": "message",
                "data": json.dumps({
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                }),
            }

            # Wait for messages
            while sessions[session_id]["connected"]:
                try:
                    message = await asyncio.wait_for(
                        sessions[session_id]["queue"].get(),
                        timeout=30.0,
                    )
                    yield {
                        "event": "message",
                        "data": json.dumps(message),
                    }
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": ""}

        finally:
            sessions.pop(session_id, None)

    return EventSourceResponse(event_generator())


async def handle_message(request: Request) -> JSONResponse:
    """Handle incoming MCP messages."""
    session_id = request.query_params.get("session_id")
    if not session_id or session_id not in sessions:
        return JSONResponse(
            {"error": "Invalid session"},
            status_code=400,
        )

    body = await request.json()
    method = body.get("method")
    params = body.get("params", {})
    request_id = body.get("id")

    result = None
    error = None

    try:
        if method == "tools/list":
            tools = await list_tools()
            result = {"tools": [t.model_dump() for t in tools]}

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            call_result = await call_tool(tool_name, arguments)
            result = {
                "content": [c.model_dump() for c in call_result.content],
                "isError": call_result.isError,
            }

        elif method == "resources/list":
            resources = await list_resources()
            result = {"resources": [r.model_dump() for r in resources]}

        elif method == "resources/read":
            uri = params.get("uri")
            contents = await read_resource(uri)
            result = {"contents": [c.model_dump() for c in contents]}

        elif method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "observatory",
                    "version": VERSION,
                },
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"listChanged": False},
                },
            }

        else:
            error = {"code": -32601, "message": f"Method not found: {method}"}

    except Exception as e:
        error = {"code": -32603, "message": str(e)}

    response = {"jsonrpc": "2.0", "id": request_id}
    if error:
        response["error"] = error
    else:
        response["result"] = result

    return JSONResponse(response)


async def health(request: Request) -> JSONResponse:
    """Health check endpoint for load balancers and monitoring."""
    backend_url = os.environ.get("OBSERVATORY_BACKEND_URL", "http://127.0.0.1:8000")
    return JSONResponse({
        "status": "healthy",
        "service": SERVICE_NAME,
        "version": VERSION,
        "active_sessions": len(sessions),
        "backend_url": backend_url,
    })


async def ready(request: Request) -> JSONResponse:
    """
    Readiness check - verifies backend connectivity.

    Returns 503 if backend is unreachable, allowing load balancers
    to route traffic away during backend issues.
    """
    import httpx

    backend_url = os.environ.get("OBSERVATORY_BACKEND_URL", "http://127.0.0.1:8000")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{backend_url}/")
            if response.status_code == 200:
                return JSONResponse({
                    "status": "ready",
                    "backend": "connected",
                    "backend_url": backend_url,
                })
    except Exception as e:
        logger.warning(f"Backend health check failed: {e}")

    return JSONResponse(
        {"status": "not_ready", "backend": "unreachable", "backend_url": backend_url},
        status_code=503
    )


async def info(request: Request) -> JSONResponse:
    """
    Service information endpoint.

    Returns details about the Observatory MCP server for documentation
    and client configuration.
    """
    backend_url = os.environ.get("OBSERVATORY_BACKEND_URL", "http://127.0.0.1:8000")

    return JSONResponse({
        "name": "Observatory",
        "description": "MCP server for analyzing cultural narratives and coordination patterns in text",
        "version": VERSION,
        "service": SERVICE_NAME,
        "transport": "sse",
        "endpoints": {
            "sse": "/sse",
            "message": "/message",
            "health": "/health",
            "ready": "/ready",
        },
        "backend_url": backend_url,
        "mcp_config": {
            "example": {
                "mcpServers": {
                    "observatory": {
                        "url": "https://observatory-mcp.onrender.com/sse"
                    }
                }
            }
        },
        "documentation": "https://github.com/your-org/observatory",
    })


# Starlette app
app = Starlette(
    routes=[
        Route("/", info, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/ready", ready, methods=["GET"]),
        Route("/sse", handle_sse, methods=["GET"]),
        Route("/message", handle_message, methods=["POST"]),
    ],
)

# CORS configuration
# In production, consider restricting origins to known clients
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting Observatory MCP Server v{VERSION}")
    logger.info(f"Listening on {host}:{port}")
    logger.info(f"Backend URL: {os.environ.get('OBSERVATORY_BACKEND_URL', 'http://127.0.0.1:8000')}")

    uvicorn.run(app, host=host, port=port)
