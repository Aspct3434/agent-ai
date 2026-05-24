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

## Quick start

```bash
# Copy and fill in your Kimi / Moonshot API key
cp an-api.env.example an-api.env

# Run with Docker (recommended: correct POSIX shell, isolated filesystem)
docker compose up

# Local development still routes agent execution into Docker.
pip install -r requirements.txt
AGENT_SANDBOX=docker AGENT_SANDBOX_HOST_FALLBACK=false \
uvicorn gateway:app --app-dir src --reload
```

Open `http://localhost:8000` for the API. Open `control-panel/` for the chat UI (`npm run dev`).

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
| `AGENT_USE_HYBRID_MEMORY` | `true` | Enable ChromaDB + Neo4j memory |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j (optional; falls back to Chroma-only) |

## Running tests

```bash
pytest                        # full suite
pytest tests/test_task_contract_loop.py -v   # contract system only
pytest --self-test src/eval_harness.py       # harness self-test (no API key needed)
```

## Windows development

For local development on Windows, keep `AGENT_SANDBOX=docker` and `AGENT_SANDBOX_HOST_FALLBACK=false` so agent tasks execute inside the Docker sandbox instead of the host shell. Docker is the recommended runtime on Windows.
