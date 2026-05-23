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

# Runtime system packages:
#  ca-certificates — required by rustup/cargo TLS and any HTTPS curl from the agent;
#                    python:3.12-slim ships without this, so cargo silently fails TLS
#  curl            — needed by rustup installer and agent wget-style fetches
#  git             — cargo fetches git-source deps; also a common agent tool
#  build-essential — gcc/make to compile native Rust crates (e.g. ring, openssl-sys)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Pre-install Rust toolchain (system-wide) ───────────────────────────────
# Install rustup/cargo/rustc into /usr/local/{rustup,cargo} so every user
# (including the non-root `agent`) can compile Rust without a per-user download.
#
# Design choices:
#   --profile minimal   → toolchain only (no clippy/rustfmt/docs) — saves ~300 MB
#   --no-modify-path    → we manage PATH via ENV below; no shell-rc edits needed
#   chmod a+rwx         → agent user can write the registry/build cache to CARGO_HOME
#
# Result: `rustc --version` and `cargo build` work immediately as the agent user,
# and the zero-dependency built-in skill (build_std_rust_http_server) can build
# offline without touching crates.io at all.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH="/usr/local/cargo/bin:${PATH}"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --default-toolchain stable --profile minimal \
    && chmod -R a+rwx /usr/local/rustup /usr/local/cargo \
    && rustc --version \
    && cargo --version

# Application source (includes src/builtin_skills/ for zero-dep Rust skill)
COPY src/ ./src/

# Dedicated non-root user WITH a home directory so that rustup/cargo work
# without the agent having to override CARGO_HOME on every invocation.
# The shell is set to /bin/bash (not nologin) so execute_terminal_command works.
RUN groupadd --gid 1001 agent \
    && useradd --uid 1001 --gid agent --create-home --shell /bin/bash agent \
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
