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

# ── Pre-install Playwright Chromium browser (system-wide) ─────────────────
# We install the browser as root into a world-readable path so the non-root
# `agent` user can use it without a per-user download on every container start.
#
# PLAYWRIGHT_BROWSERS_PATH=/usr/local/ms-playwright — shared, not in ~ of any user
# playwright install-deps chromium   — installs the OS packages Chromium needs
# playwright install chromium        — downloads the browser binary itself
# chmod a+rx                         — agent user can read/execute the binary
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/local/ms-playwright

RUN playwright install-deps chromium \
    && playwright install chromium \
    && chmod -R a+rx /usr/local/ms-playwright

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

# ── Pre-install Node.js + npm + npx (system-wide) ──────────────────────────
# A request for a React / Vue / Vite / Next app needs Node + npm. The container
# runs as the non-root `agent` user, which cannot apt-get install anything, so —
# exactly as with the Rust toolchain above — we bake Node in at build time.
# The official binary tarball is extracted into /usr/local, putting node/npm/npx
# on PATH for every user with no per-user download and no NodeSource apt repo.
# Without this, "build a React site" hits a missing-runtime wall and the agent
# wastes its whole budget trying (and failing) to apt-get nodejs as non-root.
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
