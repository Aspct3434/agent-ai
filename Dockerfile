# syntax=docker/dockerfile:1
# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Build tools needed by binary wheels (cryptography, grpcio, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# requirements.txt is now clean UTF-8; filter out the Windows-only pywin32 wheel
RUN grep -iv "^pywin32" requirements.txt > requirements_linux.txt \
    && pip install --no-cache-dir --prefix=/install -r requirements_linux.txt \
    && pip install --no-cache-dir --prefix=/install mcp-server-sqlite

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="agent-ai" \
      org.opencontainers.image.description="Production-grade ReAct agent with MCP tool integration" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Copy the entire installed package tree from builder (no pip needed at runtime)
COPY --from=builder /install /usr/local

# Application source
COPY src/ ./src/

# Dedicated non-root user
RUN groupadd --gid 1001 agent \
    && useradd --uid 1001 --gid agent --no-create-home --shell /sbin/nologin agent \
    && mkdir -p /app/data /app/published_sites /app/skills /app/chroma_data \
    && chown -R agent:agent /app

USER agent

EXPOSE 8000

# Persist state across container restarts
VOLUME ["/app/data", "/app/published_sites", "/app/skills", "/app/chroma_data"]

# Liveness probe — gateway exposes GET /health → {"status": "ok"}
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    SQLITE_DB_PATH=/app/data/agent.db \
    CHECKPOINT_DB_PATH=/app/data/checkpoints.db \
    PUBLISHED_SITES_DIR=/app/published_sites \
    SKILLS_DIR=/app/skills \
    CHROMA_PATH=/app/chroma_data

CMD ["python", "-m", "uvicorn", "gateway:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
