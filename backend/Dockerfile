# Observatory Backend
# Full ML stack with PyTorch and transformers
#
# Build: docker build -t observatory-backend .
# Run:   docker run -p 8000:8000 -v observatory-data:/app/data observatory-backend

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Dependencies Stage ---
FROM base AS dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch CPU-only (significantly smaller image)
# For GPU support, use: --index-url https://download.pytorch.org/whl/cu118
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install -r requirements.txt

# --- Production Stage ---
FROM base AS production

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN groupadd --gid 1000 observatory && \
    useradd --uid 1000 --gid observatory --shell /bin/bash --create-home observatory

# Create data directory with proper permissions
RUN mkdir -p /app/data/projections /app/data/annotations && \
    chown -R observatory:observatory /app

# Copy application code
COPY --chown=observatory:observatory . .

# Switch to non-root user
USER observatory

# Environment
ENV PORT=8000 \
    HOST=0.0.0.0

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Run server with production settings
# - workers: adjust based on CPU cores (2 * cores + 1)
# - timeout: longer for ML model loading
CMD ["sh", "-c", "uvicorn main:app --host ${HOST} --port ${PORT} --workers 1 --timeout-keep-alive 120"]
