#!/usr/bin/env node
/**
 * Observatory MCP Server (Node.js)
 *
 * SSE transport layer with MCP Apps support for the Cultural Soliton Observatory.
 * Proxies tool calls to the Python backend and serves interactive UI visualizations.
 */

import express, { Request, Response } from "express";
import cors from "cors";
import { randomUUID } from "crypto";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

// ES Module __dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const PORT = parseInt(process.env.PORT || "8080");
const BACKEND_URL = process.env.OBSERVATORY_BACKEND_URL || "http://localhost:8000";
const VERSION = "2.1.0";

// MCP Apps configuration
const APPS_DIR = path.join(__dirname, "..", "apps", "dist");
const UI_RESOURCES: Record<string, string> = {
  "manifold-viewer": "ui://observatory/manifold-viewer",
  "cohort-heatmap": "ui://observatory/cohort-heatmap",
  "trajectory-viewer": "ui://observatory/trajectory-viewer",
  "force-field": "ui://observatory/force-field",
  "mode-flow": "ui://observatory/mode-flow",
  "gap-analysis": "ui://observatory/gap-analysis",
};

const APP_DESCRIPTIONS: Record<string, string> = {
  "manifold-viewer": "Interactive 3D visualization of the cultural manifold",
  "cohort-heatmap": "Heatmap visualization for cohort analysis",
  "trajectory-viewer": "Timeline visualization for narrative trajectory tracking",
  "force-field": "Force field diagram for attractor/detractor analysis",
  "mode-flow": "Sankey diagram for narrative mode flow analysis",
  "gap-analysis": "Dashboard for comparing narrative groups",
};

// Session storage
const sessions: Map<string, { queue: any[]; connected: boolean }> = new Map();

// Helper: Load MCP App HTML
function loadAppHtml(appName: string): string | null {
  const appPath = path.join(APPS_DIR, appName, "index.html");
  if (fs.existsSync(appPath)) {
    return fs.readFileSync(appPath, "utf-8");
  }
  return null;
}

// Helper: Check if app exists
function appExists(appName: string): boolean {
  return fs.existsSync(path.join(APPS_DIR, appName, "index.html"));
}

// Helper: Build _meta for tools with UI
function buildMeta(appName: string): object | undefined {
  if (appExists(appName)) {
    return { ui: { resourceUri: UI_RESOURCES[appName] } };
  }
  return undefined;
}

// Tool definitions
function getTools() {
  return [
    {
      name: "project_text",
      description: `Project a single text onto the 3D cultural manifold.

Returns coordinates on three axes:
- Agency (-2 to +2): Self-determination vs fatalism
- Perceived Justice (-2 to +2): Fair treatment vs corruption/rigged systems
- Belonging (-2 to +2): Connected/embedded vs alienated

Also returns the detected narrative mode and confidence score.

🎨 Interactive UI: Results can be visualized in an interactive 3D manifold viewer.`,
      inputSchema: {
        type: "object",
        properties: {
          text: { type: "string", description: "The text to project onto the manifold" },
          model_id: { type: "string", description: "Embedding model to use (default: all-MiniLM-L6-v2)" },
          include_soft_labels: { type: "boolean", description: "Include probability distribution across all modes", default: false },
        },
        required: ["text"],
      },
      _meta: buildMeta("manifold-viewer"),
    },
    {
      name: "project_batch",
      description: `Project multiple texts onto the manifold in a single call.

More efficient than calling project_text repeatedly. Returns coordinates and modes for each text.

🎨 Interactive UI: Results can be visualized in an interactive 3D manifold viewer.`,
      inputSchema: {
        type: "object",
        properties: {
          texts: { type: "array", items: { type: "string" }, description: "List of texts to project" },
          detect_clusters: { type: "boolean", description: "Whether to detect narrative clusters", default: true },
        },
        required: ["texts"],
      },
      _meta: buildMeta("manifold-viewer"),
    },
    {
      name: "compare_narratives",
      description: `Compare two groups of narratives to detect gaps.

Analyzes differences between groups (e.g., leadership vs employees, stated values vs actions).
Returns gap analysis with statistical significance.

🎨 Interactive UI: Results visualized in gap analysis dashboard.`,
      inputSchema: {
        type: "object",
        properties: {
          group_a_texts: { type: "array", items: { type: "string" }, description: "First group of texts" },
          group_b_texts: { type: "array", items: { type: "string" }, description: "Second group of texts" },
          group_a_label: { type: "string", description: "Label for first group", default: "Group A" },
          group_b_label: { type: "string", description: "Label for second group", default: "Group B" },
        },
        required: ["group_a_texts", "group_b_texts"],
      },
      _meta: buildMeta("gap-analysis"),
    },
    {
      name: "track_trajectory",
      description: `Track narrative evolution over time.

Analyzes a sequence of texts to detect trajectory patterns and phase transitions.

🎨 Interactive UI: Results visualized in trajectory viewer.`,
      inputSchema: {
        type: "object",
        properties: {
          texts: { type: "array", items: { type: "string" }, description: "Texts in chronological order" },
          timestamps: { type: "array", items: { type: "string" }, description: "Optional timestamps for each text" },
        },
        required: ["texts"],
      },
      _meta: buildMeta("trajectory-viewer"),
    },
    {
      name: "analyze_cohorts",
      description: `Analyze multiple cohorts/groups simultaneously.

Compares narrative patterns across multiple groups with heatmap visualization.

🎨 Interactive UI: Results visualized in cohort heatmap.`,
      inputSchema: {
        type: "object",
        properties: {
          cohorts: {
            type: "object",
            additionalProperties: { type: "array", items: { type: "string" } },
            description: "Map of cohort names to their texts",
          },
        },
        required: ["cohorts"],
      },
      _meta: buildMeta("cohort-heatmap"),
    },
    {
      name: "analyze_mode_flow",
      description: `Analyze transitions between narrative modes over time.

Shows how narratives flow from one mode to another.

🎨 Interactive UI: Results visualized in Sankey flow diagram.`,
      inputSchema: {
        type: "object",
        properties: {
          text_sequences: {
            type: "array",
            items: { type: "array", items: { type: "string" } },
            description: "List of text sequences (each sequence is a list of texts over time)",
          },
        },
        required: ["text_sequences"],
      },
      _meta: buildMeta("mode-flow"),
    },
    {
      name: "analyze_force_field",
      description: `Analyze attractor and detractor forces in narrative space.

Identifies what's pulling narratives toward or away from certain modes.

🎨 Interactive UI: Results visualized in force field diagram.`,
      inputSchema: {
        type: "object",
        properties: {
          texts: { type: "array", items: { type: "string" }, description: "Texts to analyze" },
          target_mode: { type: "string", description: "Target mode to analyze forces relative to" },
        },
        required: ["texts"],
      },
      _meta: buildMeta("force-field"),
    },
    {
      name: "quick_analyze",
      description: `Quick one-liner analysis for rapid text assessment.

Returns essential metrics without full telescope overhead.`,
      inputSchema: {
        type: "object",
        properties: {
          text: { type: "string", description: "Text to analyze" },
        },
        required: ["text"],
      },
    },
    {
      name: "measure_cbr",
      description: `Measure Coordination Background Radiation (CBR) for text(s).

CBR is the universal coordination "glow" present in any communicative act.`,
      inputSchema: {
        type: "object",
        properties: {
          text: { type: "string", description: "Single text to analyze" },
          texts: { type: "array", items: { type: "string" }, description: "Multiple texts to analyze" },
        },
      },
    },
    {
      name: "analyze_ai_text",
      description: `Analyze AI-generated text for behavior patterns and coordination signals.

Detects AI behavior modes: CONFIDENT, UNCERTAIN, EVASIVE, HELPFUL, DEFENSIVE, OPAQUE.`,
      inputSchema: {
        type: "object",
        properties: {
          text: { type: "string", description: "AI-generated text to analyze" },
        },
        required: ["text"],
      },
    },
    {
      name: "detect_phase_transitions",
      description: `Detect phase transitions in a signal history.

Analyzes a sequence of signals to detect when significant shifts occurred.`,
      inputSchema: {
        type: "object",
        properties: {
          signal_history: { type: "array", items: { type: "string" }, description: "Sequence of texts to analyze" },
          window_size: { type: "number", description: "Window size for transition detection" },
        },
        required: ["signal_history"],
      },
    },
    {
      name: "observatory_status",
      description: `Check observatory backend status and loaded models.`,
      inputSchema: {
        type: "object",
        properties: {},
      },
    },
  ];
}

// Resource definitions
function getResources() {
  const resources: any[] = [];
  for (const [appName, uri] of Object.entries(UI_RESOURCES)) {
    if (appExists(appName)) {
      resources.push({
        uri,
        name: `Observatory ${appName.split("-").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ")}`,
        description: APP_DESCRIPTIONS[appName] || `Interactive UI for ${appName}`,
        mimeType: "text/html;charset=utf-8",
      });
    }
  }
  return resources;
}

// Proxy tool call to backend
async function callBackendTool(toolName: string, args: any): Promise<any> {
  // Map MCP tool names to backend endpoints
  const endpointMap: Record<string, { method: string; path: string; transform?: (args: any) => any }> = {
    project_text: { method: "POST", path: "/project" },
    project_batch: { method: "POST", path: "/project/batch", transform: (a) => ({ texts: a.texts, detect_clusters: a.detect_clusters ?? true }) },
    compare_narratives: { method: "POST", path: "/analysis/compare" },
    track_trajectory: { method: "POST", path: "/analysis/trajectory" },
    analyze_cohorts: { method: "POST", path: "/analysis/cohorts" },
    analyze_mode_flow: { method: "POST", path: "/analysis/mode-flow" },
    analyze_force_field: { method: "POST", path: "/analysis/force-field" },
    quick_analyze: { method: "POST", path: "/telescope/quick" },
    measure_cbr: { method: "POST", path: "/cbr/measure" },
    analyze_ai_text: { method: "POST", path: "/ai/analyze" },
    detect_phase_transitions: { method: "POST", path: "/analysis/phase-transitions" },
    observatory_status: { method: "GET", path: "/" },
  };

  const endpoint = endpointMap[toolName];
  if (!endpoint) {
    throw new Error(`Unknown tool: ${toolName}`);
  }

  const url = `${BACKEND_URL}${endpoint.path}`;
  const body = endpoint.transform ? endpoint.transform(args) : args;

  const response = await fetch(url, {
    method: endpoint.method,
    headers: { "Content-Type": "application/json" },
    body: endpoint.method === "POST" ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Backend error: ${response.status} - ${errorText}`);
  }

  return response.json();
}

// Create Express app
const app = express();
app.use(cors());
app.use(express.json());

// Health check
app.get("/health", (req: Request, res: Response) => {
  res.json({
    status: "healthy",
    service: "observatory-mcp",
    version: VERSION,
    active_sessions: sessions.size,
    backend_url: BACKEND_URL,
    apps_available: Object.keys(UI_RESOURCES).filter(appExists),
  });
});

// Root info
app.get("/", (req: Request, res: Response) => {
  res.json({
    name: "Observatory MCP Server",
    version: VERSION,
    description: "Cultural Soliton Observatory - MCP tools for narrative analysis",
    endpoints: {
      sse: "/sse",
      message: "/message",
      health: "/health",
    },
  });
});

// SSE endpoint
app.get("/sse", (req: Request, res: Response) => {
  const sessionId = randomUUID();

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  // Send session ID
  res.write(`event: session\ndata: ${JSON.stringify({ session_id: sessionId })}\n\n`);

  // Initialize session
  sessions.set(sessionId, { queue: [], connected: true });

  // Send initialized notification
  res.write(`event: message\ndata: ${JSON.stringify({
    jsonrpc: "2.0",
    method: "notifications/initialized",
    params: {},
  })}\n\n`);

  // Keep connection alive
  const heartbeat = setInterval(() => {
    if (sessions.get(sessionId)?.connected) {
      res.write(": heartbeat\n\n");
    }
  }, 30000);

  // Handle disconnect
  req.on("close", () => {
    clearInterval(heartbeat);
    sessions.delete(sessionId);
  });
});

// Message handler
app.post("/message", async (req: Request, res: Response) => {
  const sessionId = req.query.session_id as string;
  if (!sessionId || !sessions.has(sessionId)) {
    return res.status(400).json({ error: "Invalid session" });
  }

  const { method, params, id } = req.body;
  let result: any = null;
  let error: any = null;

  try {
    switch (method) {
      case "initialize":
        result = {
          protocolVersion: "2024-11-05",
          serverInfo: { name: "observatory", version: VERSION },
          capabilities: {
            tools: { listChanged: false },
            resources: { listChanged: false },
          },
        };
        break;

      case "tools/list":
        result = { tools: getTools() };
        break;

      case "tools/call":
        const toolResult = await callBackendTool(params.name, params.arguments || {});
        result = {
          content: [{ type: "text", text: JSON.stringify(toolResult, null, 2) }],
          isError: false,
        };
        break;

      case "resources/list":
        result = { resources: getResources() };
        break;

      case "resources/read":
        const uri = params.uri as string;
        const appName = Object.entries(UI_RESOURCES).find(([_, u]) => u === uri)?.[0];
        if (appName) {
          const html = loadAppHtml(appName);
          if (html) {
            result = {
              contents: [{
                uri,
                mimeType: "text/html;charset=utf-8",
                text: html,
              }],
            };
          } else {
            error = { code: -32602, message: `App '${appName}' not found` };
          }
        } else {
          error = { code: -32602, message: `Unknown resource URI: ${uri}` };
        }
        break;

      default:
        error = { code: -32601, message: `Method not found: ${method}` };
    }
  } catch (e: any) {
    error = { code: -32603, message: e.message };
  }

  const response: any = { jsonrpc: "2.0", id };
  if (error) {
    response.error = error;
  } else {
    response.result = result;
  }

  res.json(response);
});

// Start server
app.listen(PORT, () => {
  console.log(`Observatory MCP Server v${VERSION}`);
  console.log(`Listening on port ${PORT}`);
  console.log(`Backend URL: ${BACKEND_URL}`);
  console.log(`MCP Apps directory: ${APPS_DIR}`);
  console.log(`Available apps: ${Object.keys(UI_RESOURCES).filter(appExists).join(", ") || "none (run npm run build in apps/)"}`);
});
