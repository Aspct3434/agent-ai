# Agent AI

A production-grade ReAct agent with **task-contract execution**, MCP tool integration, skill distillation, hybrid memory, and a real-time streaming control panel.

## Architecture

```
src/
  agent.py          -- AgentEngine: ReAct loop, session management, checkpointing
  contract.py       -- Task-contract system: evidence tracking, completion gating
  planning.py       -- Plan management, executive summary, action ledger
  llm_utils.py      -- LLM retry, rate-limit handling, streaming utilities
  tools.py          -- ToolManager: MCP servers, terminal, file, process, port tools
  evaluator.py      -- Skill distiller: trajectory -> reusable MCP skill
  checkpointer.py   -- SQLite checkpoint store (WAL mode, aiosqlite)
  memory.py         -- HybridMemory: ChromaDB (semantic) + Neo4j (graph)
  gateway.py        -- FastAPI app, WebSocket stream, session-per-FIFO-lane
  eval_harness.py   -- Evaluation harness with token/cost metering

control-panel/      -- React + Tailwind chat UI with live token streaming
tests/              -- Pytest suite (scripted streaming model, contract tests)
```

## Key design decisions

**Task-contract system** â€” the model's first call is always `set_task_contract`, declaring whether the task needs execution evidence (files created, service served, DB mutated, process started) or just a text answer. The loop gates the final response on that evidence, so "I'll create the file now" with no actual tool call is impossible. During execute mode, the tool list is narrowed to evidence producers and `tool_choice=required` prevents prose escape.

**Session-per-FIFO-lane concurrency** â€” each session gets its own asyncio queue + worker task; different sessions run concurrently; same-session messages are strictly ordered.

**Skill distillation** â€” after a successful run, an LLM synthesizes the trajectory into a parameterized Python function saved to `skills/`. A quality gate requires â‰¥2 successful side-effect calls; the function is syntax-checked before writing.

**Prompt caching** â€” system directive + tool schemas carry `cache_control: ephemeral` breakpoints; fast/strong model tiers share the same provider so cache hits apply across escalations.

## One-click install

For a fresh machine with no project files yet, use the bootstrap installer:

```powershell
iwr -useb https://raw.githubusercontent.com/Aspct3434/agent-ai/master/scripts/bootstrap.ps1 | iex
```

```bash
curl -fsSL https://raw.githubusercontent.com/Aspct3434/agent-ai/master/scripts/bootstrap.sh | bash
```

To install from a fork or branch, pass the repo URL explicitly:

```powershell
& ([scriptblock]::Create((iwr -useb https://raw.githubusercontent.com/Aspct3434/agent-ai/master/scripts/bootstrap.ps1))) -RepoUrl "https://github.com/Aspct3434/agent-ai.git"
```

The bootstrap checks prerequisites, clones the repo to `%USERPROFILE%\agent-ai` on Windows or `~/agent-ai` on Bash, then runs the interactive installer.

From an existing local clone, run the second-stage installer directly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install.ps1
```

```bash
bash scripts/install.sh
```

The installer asks for your model provider, sandbox mode, and messaging app. It writes `an-api.env`, backs up an existing env file before changing managed keys, and starts the full app.

Sandbox choices:

- `Sandbox on` uses Docker Compose container isolation and starts the backend, Neo4j, and control panel.
- `Sandbox off` starts the backend and control panel directly on the host under `.run-venv` and `control-panel/node_modules`.

Smoke checks after install:

```bash
curl http://localhost:8000/health
```

Open `http://localhost:5173` for the chat UI and `http://localhost:8000/docs` for the API docs.

## Manual quick start

```bash
# Copy and fill in your Kimi / Moonshot API key
cp an-api.env.example an-api.env

# Run with Docker Compose (recommended)
docker compose up -d --build

# Local development can still route agent execution into Docker.
pip install -r requirements.txt
AGENT_SANDBOX=docker AGENT_SANDBOX_HOST_FALLBACK=false \
uvicorn gateway:app --app-dir src --reload
```

Open `http://localhost:8000` for the API. Docker Compose serves the chat UI at `http://localhost:5173`; local frontend development still uses `control-panel/` with `npm run dev`.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MOONSHOT_API_KEY` | required | Kimi / Moonshot API key used by LiteLLM |
| `MOONSHOT_API_BASE` | `https://api.moonshot.ai/v1` | Kimi / Moonshot API base URL |
| `AGENT_MODEL` | `moonshot/kimi-k2.5` | Main model |
| `FAST_AGENT_MODEL` | `AGENT_MODEL` | Routine iterations |
| `STRONG_AGENT_MODEL` | `AGENT_MODEL` | Escalated iterations / sub-agents |
| `AGENT_MAX_REACT_ITERATIONS` | `16` | Q&A iteration cap |
| `AGENT_ACTION_MAX_REACT_ITERATIONS` | `30` | Execution iteration cap |
| `AGENT_SANDBOX` | blank | Leave blank for Docker Compose container isolation; set `docker` only when running the backend directly on a host and you want nested Docker execution. |
| `AGENT_SANDBOX_HOST_FALLBACK` | `false` | Allow direct host execution if a requested sandbox fails. The installer sets this to `true` only for sandbox-off host mode. |
| `AGENT_USE_HYBRID_MEMORY` | `true` | Enable ChromaDB + Neo4j memory |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j (optional; falls back to Chroma-only) |

## Running tests

```bash
pytest                        # full suite
pytest tests/test_task_contract_loop.py -v   # contract system only
pytest --self-test src/eval_harness.py       # harness self-test (no API key needed)
```

## Windows development

For one-click installs on Windows, use the PowerShell bootstrap/installer and keep `Sandbox on` for Docker Compose container isolation. If you run `uvicorn` directly on Windows instead of Compose, set `AGENT_SANDBOX=docker` and `AGENT_SANDBOX_HOST_FALLBACK=false` so agent tasks execute inside Docker rather than the host shell.
