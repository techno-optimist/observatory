# Cultural Soliton Observatory MCP Server

Expose the Cultural Soliton Observatory as MCP tools for any LLM to use.

## What This Does

Allows Claude (or any MCP-compatible LLM) to:

- **Project text** onto a 3D cultural manifold (Agency, Fairness, Belonging)
- **Analyze corpora** with clustering detection
- **Detect hypocrisy** between stated values and operational language
- **Compare projection methods** (Ridge, GP, Neural)
- **Add training examples** to calibrate the projection

## Quick Start

### 1. Start the Observatory Backend

```bash
cd /path/to/cultural-soliton-observatory
./launch-observatory.command
```

Or manually:
```bash
cd backend
source venv/bin/activate
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 2. Configure Claude Code

Add to your `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "observatory": {
      "command": "python",
      "args": ["/path/to/cultural-soliton-observatory/mcp-server/server.py"],
      "env": {
        "OBSERVATORY_BACKEND_URL": "http://127.0.0.1:8000"
      }
    }
  }
}
```

### 3. Use in Claude Code

```
You: Project this text onto the cultural manifold: "We believe in fair wages for all employees"

Claude: [Uses project_text tool]
```

## Deployment Options

### Option A: Local (Recommended for Development)

Just run `server.py` directly. Claude Code connects via stdio.

```bash
# Test with MCP inspector
npx @modelcontextprotocol/inspector python server.py
```

### Option B: Self-Hosted (Docker)

```bash
# Start full stack
docker-compose up -d

# MCP server available at http://localhost:8080
```

For Claude Desktop, configure SSE endpoint:
```json
{
  "mcpServers": {
    "observatory": {
      "transport": "sse",
      "url": "http://localhost:8080/sse"
    }
  }
}
```

### Option C: Render.com

1. Fork/push this repo
2. Create a Render Blueprint from `render.yaml`
3. Note: Full backend needs **Starter tier** ($7/mo) for PyTorch RAM

```yaml
# In render.yaml, update backend URL
- key: OBSERVATORY_BACKEND_URL
  value: https://your-backend.onrender.com
```

### Option D: Fly.io

```bash
# From mcp-server directory
fly launch --no-deploy
fly secrets set OBSERVATORY_BACKEND_URL=https://your-backend.fly.dev
fly deploy
```

## 🎨 Interactive Visualizations (MCP Apps)

The Observatory now includes interactive UI visualizations that render directly in Claude conversations. When you use tools with the 🎨 indicator, an interactive visualization appears inline:

| Visualization | Tool | Description |
|---------------|------|-------------|
| **3D Manifold Viewer** | `project_text`, `analyze_corpus` | Interactive 3D scatter plot with orbit controls |
| **Cohort Heatmap** | `analyze_cohorts` | Multi-cohort comparison with dimension filtering |
| **Trajectory Timeline** | `track_trajectory` | Animated timeline showing narrative evolution |
| **Force Field Diagram** | `analyze_force_field` | Attractor/detractor visualization with quadrants |
| **Mode Flow Sankey** | `analyze_mode_flow` | Flow diagram of mode transitions |
| **Gap Analysis Dashboard** | `compare_narratives` | Side-by-side group comparison with gap metrics |

These work automatically in:
- Claude Desktop (web)
- Claude Code
- Any MCP host with MCP Apps support

## Available Tools

| Tool | Description |
|------|-------------|
| `project_text` 🎨 | Project text → (agency, fairness, belonging) |
| `analyze_corpus` 🎨 | Batch analysis with clustering |
| `compare_projections` | Compare Ridge/GP/Neural projections |
| `detect_hypocrisy` | Measure gap between stated and operational values |
| `generate_probe` | Describe text that would land at target coords |
| `get_manifold_state` | Get observatory status |
| `add_training_example` | Add labeled example for calibration |
| `explain_modes` | Explain narrative mode system |
| `run_scenario` | Simulate narrative evolution |
| `compare_narratives` 🎨 | Compare two groups with gap analysis |
| `track_trajectory` 🎨 | Track narrative evolution over time |
| `analyze_cohorts` 🎨 | Multi-group analysis with ANOVA |
| `analyze_mode_flow` 🎨 | Analyze mode transitions and patterns |
| `analyze_force_field` 🎨 | Attractor/detractor force field analysis |

## Architecture

```
┌─────────────────────────┐
│  Claude / LLM Client    │
└───────────┬─────────────┘
            │ MCP (stdio or SSE)
            ▼
┌─────────────────────────┐
│  Observatory MCP Server │  ← This package
│  (server.py)            │
└───────────┬─────────────┘
            │ HTTP
            ▼
┌─────────────────────────┐
│  FastAPI Backend        │  ← Existing backend
│  (backend/main.py)      │
│  - Embedding models     │
│  - Projection training  │
│  - Clustering           │
└─────────────────────────┘
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OBSERVATORY_BACKEND_URL` | `http://127.0.0.1:8000` | FastAPI backend URL |
| `OBSERVATORY_DEFAULT_MODEL` | `all-MiniLM-L6-v2` | Default embedding model |
| `PORT` | `8080` | SSE server port |

## Example Session

```
Human: I want to analyze these corporate communications for hypocrisy.

Mission statement: "We put employees first and believe in work-life balance."

Recent emails:
- "Weekend work is expected during Q4 crunch"
- "PTO requests during busy periods will be denied"
- "Team players make sacrifices for the company"

Claude: I'll use the observatory to detect potential hypocrisy between
        stated values and operational language.

[Uses detect_hypocrisy tool]

Results show a high hypocrisy gap (Δw = 0.89):
- Espoused: High belonging (+1.4), moderate agency (+0.6)
- Operational: Low belonging (-0.3), negative agency (-0.8)

The mission statement emphasizes employee wellbeing (high belonging),
but operational communications frame sacrifice as mandatory (low agency)
and create distance between "team players" and others (low belonging).
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Type check
mypy server.py observatory_client.py
```

## Limitations

- Projection is trained on ~100 examples (consider domain-specific calibration)
- 384D → 3D loses 99%+ of variance (by design, for interpretability)
- Western liberal value framework encoded in axes
- Backend must be running for tools to work
