# Observatory

**A powerful MCP server for analyzing narratives, cultural positioning, and AI behavior patterns.**

Observatory projects text onto a 3D coordination manifold, revealing how narratives position themselves across Agency, Perceived Justice, and Belonging dimensions. Use it to understand organizational culture, detect narrative drift, monitor AI safety, and analyze communication patterns.

## Quick Start (Hosted)

Add Observatory to Claude Desktop or Claude Code:

```json
{
  "mcpServers": {
    "observatory": {
      "url": "https://observatory-mcp.onrender.com/sse"
    }
  }
}
```

Restart Claude and ask: *"What Observatory tools are available?"*

## What It Does

### The Cultural Manifold

Every text lands somewhere in a 3D space defined by:

| Axis | Range | Low (-2) | High (+2) |
|------|-------|----------|-----------|
| **Agency** | Self-determination | Fatalistic, powerless | Empowered, in control |
| **Perceived Justice** | System legitimacy | Corrupt, rigged | Fair, meritocratic |
| **Belonging** | Community connection | Alienated, outsider | Embedded, connected |

### Narrative Modes

Texts are classified into 12 modes across 4 categories:

| Category | Modes | Characteristics |
|----------|-------|-----------------|
| **Positive** | Growth Mindset, Civic Idealism, Faithful Zeal | High agency + high justice |
| **Shadow** | Cynical Burnout, Institutional Decay, Schismatic Doubt | High agency + low justice |
| **Exit** | Quiet Quitting, Grid Exit, Apostasy | Low agency, withdrawal |
| **Ambivalent** | Conflicted, Transitional, Neutral | Mixed signals |

## Available Tools (35+)

### Core Projection Tools

| Tool | Description |
|------|-------------|
| `project_text` | Project text onto the 3D manifold with mode classification |
| `analyze_corpus` | Batch analyze texts with clustering detection |
| `compare_projections` | Compare projection methods (ridge, GP, neural) for stability |
| `generate_probe` | Generate text predicted to land at specific coordinates |
| `get_manifold_state` | Get current observatory configuration and status |
| `add_training_example` | Contribute labeled examples to improve calibration |
| `explain_modes` | Explain the narrative mode classification system |

### Projection Mode Management

| Tool | Description |
|------|-------------|
| `list_projection_modes` | List available projections with accuracy/speed tradeoffs |
| `select_projection_mode` | Switch between MiniLM (fast), MPNet (accurate), or ensemble |
| `get_current_projection_mode` | Check which projection is active |
| `analyze_with_uncertainty` | Get 95% confidence intervals for coordinates |
| `get_soft_labels` | Get probability distribution across all 12 modes |

### Comparative Analysis

| Tool | Description |
|------|-------------|
| `compare_narratives` | Gap analysis between two groups (e.g., leadership vs employees) |
| `track_trajectory` | Track narrative evolution over timestamped texts |
| `detect_outlier` | Detect anomalous narratives using Mahalanobis distance |
| `analyze_cohorts` | Multi-group ANOVA analysis |
| `analyze_mode_flow` | Analyze mode transitions and identify patterns |
| `detect_hypocrisy` | Measure gap between stated values and actual language |

### Alert System

| Tool | Description |
|------|-------------|
| `create_alert` | Set up monitoring rules (gap threshold, mode shift, velocity) |
| `check_alerts` | Check texts against all configured alerts |

### Force Field Analysis

Extends the manifold with attractor/detractor dynamics:

| Tool | Description |
|------|-------------|
| `analyze_force_field` | Analyze what a narrative is drawn toward and fleeing from |
| `analyze_trajectory_forces` | Track how forces evolve across a sequence |
| `compare_force_fields` | Compare motivational structure between groups |
| `get_force_targets` | List all attractor targets and detractor sources |

**Attractor Targets**: AUTONOMY, COMMUNITY, JUSTICE, MEANING, SECURITY, RECOGNITION
**Detractor Sources**: OPPRESSION, ISOLATION, INJUSTICE, MEANINGLESSNESS, INSTABILITY, INVISIBILITY

### High-Level Narrative Analysis

| Tool | Description |
|------|-------------|
| `fetch_narrative_source` | Fetch and segment content from URLs (websites, RSS, blogs) |
| `build_narrative_profile` | Build comprehensive profile with mode, force field, tensions |
| `get_narrative_suggestions` | Generate actionable insights based on intent |

### AI Behavior Analysis

| Tool | Description |
|------|-------------|
| `analyze_ai_text` | Detect AI behavior modes (confident, uncertain, evasive, etc.) |
| `analyze_ai_conversation` | Track behavior dynamics across conversation turns |
| `monitor_ai_safety` | Real-time safety monitoring with alerts |
| `detect_ai_anomalies` | Detect anomalous AI behavior vs baseline |
| `compare_ai_human` | Compare systematic differences in AI vs human text |
| `fingerprint_ai` | Create behavioral signature of an AI model |

### Coordination & Safety

| Tool | Description |
|------|-------------|
| `measure_cbr` | Measure Coordination Background Radiation temperature |
| `check_ossification` | Check for protocol ossification risk in AI communication |
| `telescope_observe` | Full 18D hierarchical coordinate extraction |
| `detect_gaming` | Detect coordination gaming attempts (legibility, feature) |
| `analyze_opacity` | Detect opaque/obfuscated content |
| `detect_covert_channels` | Analyze for hidden communication channels |

### Advanced Research Tools (Optional)

These require additional dependencies:

| Tool | Requirements | Description |
|------|--------------|-------------|
| `extract_parsed` | spaCy | Dependency parsing for linguistically accurate extraction |
| `extract_semantic` | sentence-transformers | Semantic similarity-based extraction |
| `compare_extraction_methods` | spaCy | Compare regex vs parsed vs semantic methods |
| `validate_external` | - | Validate against psychological scales |
| `get_safety_report` | - | Comprehensive deployment safety assessment |

## Use Cases

### Organizational Culture Analysis
```
Compare leadership communications vs employee feedback to detect alignment gaps.
Track cultural evolution over time. Alert on drift toward shadow modes.
```

### AI Safety Monitoring
```
Monitor AI responses for evasion patterns, opacity, behavior drift.
Create behavioral fingerprints. Detect gaming attempts.
```

### Narrative Intelligence
```
Profile websites, blogs, or feeds. Understand what attracts and repels audiences.
Generate engagement strategies based on narrative positioning.
```

### Research
```
Validate against psychological scales. Compare extraction methods.
Run statistical analyses with proper uncertainty quantification.
```

## Local Development

### Prerequisites
- Python 3.10+
- Docker (optional)

### Using Docker Compose
```bash
# Start both services
docker compose up

# Services available at:
# - Backend: http://localhost:8000
# - MCP Server: http://localhost:8080
```

### Manual Setup
```bash
# Terminal 1: Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8000

# Terminal 2: MCP Server
cd mcp-server
pip install -r requirements.txt
python server_sse.py
```

### Local MCP Configuration
```json
{
  "mcpServers": {
    "observatory": {
      "url": "http://localhost:8080/sse"
    }
  }
}
```

## Self-Hosting on Render

1. Fork this repository
2. Go to [Render Dashboard](https://dashboard.render.com/) → New Blueprint
3. Connect your fork
4. Render auto-deploys from `render.yaml`:
   - `observatory-backend` (Starter plan, ~$7/mo)
   - `observatory-mcp` (Free tier)
5. Your MCP endpoint: `https://observatory-mcp.onrender.com/sse`

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Claude Client  │────▶│  MCP Server     │────▶│  Backend        │
│  (Desktop/Code) │ SSE │  (Lightweight)  │HTTP │  (FastAPI + ML) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              Free tier            Starter plan
                              ~50MB RAM            ~2GB RAM
```

- **MCP Server**: Stateless SSE proxy, minimal dependencies
- **Backend**: FastAPI + PyTorch + transformers, handles embedding and projection

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Backend health check |
| `GET /health` | MCP server health |
| `GET /ready` | MCP server readiness (checks backend) |
| `GET /sse` | MCP SSE endpoint for Claude |

## Contributing

1. Fork the repository
2. Make changes
3. Test locally with `docker compose up`
4. Submit a pull request

## License

MIT
