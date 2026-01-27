# Observatory MCP Server
# Lightweight SSE transport layer (no ML dependencies)
#
# Build: docker build -t observatory-mcp .
# Run:   docker run -p 8080:8080 -e OBSERVATORY_BACKEND_URL=http://host.docker.internal:8000 observatory-mcp

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies (curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 observatory && \
    useradd --uid 1000 --gid observatory --shell /bin/bash --create-home observatory

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy server code
COPY --chown=observatory:observatory *.py ./

# Switch to non-root user
USER observatory

# Environment defaults
# OBSERVATORY_BACKEND_URL should be set at runtime to point to the backend
ENV OBSERVATORY_BACKEND_URL=http://host.docker.internal:8000 \
    OBSERVATORY_DEFAULT_MODEL=all-MiniLM-L6-v2 \
    PORT=8080 \
    HOST=0.0.0.0

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run SSE server
CMD ["python", "server_sse.py"]
