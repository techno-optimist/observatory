# Observatory Deployment Guide

Deploy the Observatory as a hosted MCP server that anyone can use with Claude Desktop, Claude Code, or other MCP-compatible clients.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Claude Client  │────▶│  MCP Server     │────▶│  Backend        │
│  (Desktop/Code) │ SSE │  (Lightweight)  │HTTP │  (FastAPI + ML) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              Free tier            Starter plan
                              ~50MB RAM            ~2GB RAM
```

- **MCP Server**: Lightweight SSE proxy, stateless, can run on free tier
- **Backend**: FastAPI + PyTorch, handles embedding and projection

## Quick Deploy to Render

### Option 1: Full Stack (Recommended)

1. Fork this repository to your GitHub account

2. Go to [Render Dashboard](https://dashboard.render.com/) and click "New Blueprint"

3. Connect your forked repository

4. Render will detect `render.yaml` and create both services:
   - `observatory-backend` (Starter plan, ~$7/mo)
   - `observatory-mcp` (Free tier)

5. Wait for deployment (backend takes ~5 min due to ML dependencies)

6. Your MCP server will be available at:
   ```
   https://observatory-mcp.onrender.com/sse
   ```

### Option 2: MCP Server Only

If you're hosting the backend elsewhere:

1. Navigate to `mcp-server/` directory
2. Deploy using the standalone `render.yaml`
3. Set `OBSERVATORY_BACKEND_URL` in the Render dashboard

## Client Configuration

### Claude Desktop

Add to `~/.config/claude/claude_desktop_config.json` (macOS/Linux) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "observatory": {
      "url": "https://observatory-mcp.onrender.com/sse"
    }
  }
}
```

### Claude Code

Add to `~/.config/claude-code/settings.json`:

```json
{
  "mcpServers": {
    "observatory": {
      "url": "https://observatory-mcp.onrender.com/sse"
    }
  }
}
```

### Verify Installation

Restart Claude and ask:
> "What Observatory tools are available?"

Claude should list all available tools like `project_text`, `analyze_ai_text`, etc.

## Local Development

### Using Docker Compose

```bash
# Start both services
docker compose up

# Start in background
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

Services will be available at:
- Backend: http://localhost:8000
- MCP Server: http://localhost:8080

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

## Environment Variables

### MCP Server

| Variable | Default | Description |
|----------|---------|-------------|
| `OBSERVATORY_BACKEND_URL` | `http://127.0.0.1:8000` | Backend API URL |
| `PORT` | `8080` | Server port |
| `OBSERVATORY_DEFAULT_MODEL` | `all-MiniLM-L6-v2` | Default embedding model |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |

### Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `CORS_ORIGINS` | localhost URLs | Allowed CORS origins |
| `ANTHROPIC_API_KEY` | - | For AI-powered features |
| `LOG_LEVEL` | `INFO` | Logging level |

## Scaling Considerations

### Backend Resources

| Plan | RAM | Best For |
|------|-----|----------|
| Starter ($7/mo) | 2GB | Light usage, single model |
| Standard ($25/mo) | 4GB | Multiple models, batch processing |
| Pro ($85/mo) | 8GB | Heavy traffic, all models loaded |

### Cold Start

The backend downloads ML models on first request (~30-60 seconds). To minimize cold start impact:

1. **Pre-warm on deploy**: Add a startup script that makes a test request
2. **Keep alive**: Use Render's "Always On" setting (paid plans)
3. **Health checks**: Configured to allow time for model loading

### Horizontal Scaling

- **MCP Server**: Stateless, can scale horizontally
- **Backend**: Stateful (models in memory), scale vertically

## Monitoring

### Health Endpoints

```bash
# MCP Server health
curl https://observatory-mcp.onrender.com/health

# MCP Server readiness (checks backend)
curl https://observatory-mcp.onrender.com/ready

# Backend health
curl https://observatory-backend.onrender.com/
```

### Logs

View logs in the Render dashboard or use:
```bash
render logs observatory-mcp
render logs observatory-backend
```

## Troubleshooting

### "Backend unreachable"

1. Check if backend is deployed and healthy
2. Verify `OBSERVATORY_BACKEND_URL` is set correctly
3. Check backend logs for errors

### "Model loading failed"

1. Ensure backend has sufficient RAM (2GB minimum)
2. Check if disk has space for model cache
3. Try a smaller model: `all-MiniLM-L6-v2`

### "CORS error"

1. Add your client's origin to `CORS_ORIGINS`
2. For development, use `ALLOWED_ORIGINS=*`

### Cold start timeout

1. Increase health check `start_period` in Dockerfile
2. Upgrade to a plan with more RAM
3. Consider pre-downloading models in Docker build

## Security

- Both services run as non-root users
- CORS is configurable for production
- No sensitive data in environment variables by default
- Health endpoints don't expose internal state

## Cost Optimization

1. **Free tier MCP server**: The MCP server has no ML dependencies
2. **Shared backend**: Multiple MCP servers can share one backend
3. **Sleep on inactivity**: Free tier sleeps after 15 min inactivity
4. **Model selection**: Smaller models = less RAM = cheaper plans

## Contributing

To contribute improvements:

1. Fork the repository
2. Make changes
3. Test locally with `docker compose up`
4. Submit a pull request
