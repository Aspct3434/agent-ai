# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN grep -iv "^pywin32" requirements.txt > requirements_linux.txt \
    && pip install --no-cache-dir --prefix=/install -r requirements_linux.txt \
    && pip install --no-cache-dir --prefix=/install mcp-server-sqlite

FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="agent-ai" \
      org.opencontainers.image.description="Production-grade ReAct agent with MCP tool integration" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

COPY --from=builder /install /usr/local

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PLAYWRIGHT_BROWSERS_PATH=/usr/local/ms-playwright

RUN playwright install-deps chromium \
    && playwright install chromium \
    && chmod -R a+rx /usr/local/ms-playwright

ENV NODE_VERSION=20.18.1

RUN ARCH="$(dpkg --print-architecture)" \
    && case "$ARCH" in \
         amd64) NODE_ARCH="x64" ;; \
         arm64) NODE_ARCH="arm64" ;; \
         *) echo "Unsupported architecture for Node install: $ARCH" >&2; exit 1 ;; \
       esac \
    && curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz" -o /tmp/node.tar.xz \
    && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 \
    && rm /tmp/node.tar.xz \
    && node --version \
    && npm --version \
    && npx --version

COPY src/ ./src/

RUN groupadd --gid 1001 agent \
    && useradd --uid 1001 --gid agent --create-home --shell /bin/bash agent \
    && mkdir -p /app/data /app/skills /app/chroma_data /app/workspace \
    && chown -R agent:agent /app

USER agent

EXPOSE 8000

VOLUME ["/app/data", "/app/skills", "/app/chroma_data", "/app/workspace"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    SQLITE_DB_PATH=/app/data/agent.db \
    CHECKPOINT_DB_PATH=/app/data/checkpoints.db \
    SKILLS_DIR=/app/skills \
    CHROMA_PATH=/app/chroma_data

CMD ["python", "-m", "uvicorn", "gateway:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
