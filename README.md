<p align="center">
  <img src="docs/assets/distill-terminal-banner.svg" alt="Colored Distill terminal banner" width="780">
</p>

# Distill

A production-grade ReAct agent built for reliability. Features **task-contract execution** to prevent hallucination, **skill distillation** to evolve its own tools, **hybrid memory**, and a real-time streaming control panel.

### Distill TUI

Launch the interactive terminal interface with:

```bash
distill
# or
distill tui
```

![Distill TUI command demo](docs/assets/distill-tui.gif)

## Architecture

```mermaid
graph TD
    %% External Interfaces
    UI[Control Panel UI <br/>React + Tailwind]
    Adapters[Messaging Adapters <br/>Telegram, Discord, Slack, Email]
    TUI[Terminal UI]

    %% Gateway & Concurrency
    subgraph Gateway Layer
        API[FastAPI Gateway]
        WS[WebSocket Stream]
        Queue[Session FIFO Queue]
        
        API --- WS
        WS --> Queue
    end

    UI --> API
    Adapters --> API
    TUI --> WS

    %% Core Agent Engine
    subgraph Core Agent Engine
        Agent[Agent Engine <br/>ReAct Loop]
        Contract[Task Contract System <br/>Evidence Gating]
        Plan[Plan Management <br/>Action Ledger]
        Eval[Skill Distiller <br/>Evaluator]
        
        Agent <--> Contract
        Agent <--> Plan
        Agent --> Eval
    end

    Queue --> Agent

    %% State & Memory
    subgraph State & Memory
        Checkpoint[(SQLite Checkpointer <br/>State & History)]
        Chroma[(ChromaDB <br/>Semantic Memory)]
        Neo4j[(Neo4j <br/>Graph Memory)]
        
        Agent <--> Checkpoint
        Agent <--> Chroma
        Agent <--> Neo4j
    end

    %% External Services
    LLM((LLM Provider <br/>LiteLLM))
    Agent <--> LLM

    %% Tooling & Execution
    subgraph Tooling & Execution Sandbox
        Tools[Tool Manager]
        MCP[MCP Servers]
        Sandbox[Terminal Sandbox <br/>Docker / Host / Serverless]
        Skills[Evolved Skills <br/>Auto-Maker]
        
        Agent --> Tools
        Tools --> MCP
        Tools --> Sandbox
        Tools --> Skills
        Eval -.->|Synthesizes & Validates| Skills
    end
```

### Directory Structure

```text
src/
  agent.py          -- ReAct loop, session management, checkpointing
  contract.py       -- Task-contract system (evidence tracking)
  evaluator.py      -- Skill distiller: trajectory -> reusable MCP skill
  memory.py         -- HybridMemory: ChromaDB (semantic) + Neo4j (graph)
  gateway.py        -- FastAPI app, WebSocket stream, session-per-FIFO-lane
  tools.py          -- MCP servers, terminal, file, process, port tools
control-panel/      -- React + Tailwind chat UI with live token streaming
```

## 🌟 Key Features

* **Task-Contract Execution**: The agent must declare the required execution evidence (files created, services running) before starting a task. The final response is gated on this physical evidence, eliminating "I'll do it now" hallucinations.
* **Skill Distillation**: After a successful complex task, an LLM evaluates the trajectory and synthesizes a parameterized Python tool. New skills are versioned, validated, and automatically rolled back if their success rate drops.
* **Session-per-FIFO-Lane Concurrency**: Every user session gets a dedicated queue and worker task, allowing high concurrency with strict message ordering.
* **Hybrid Memory**: Combines SQLite full-text search, ChromaDB semantic embeddings, and Neo4j graph relationships to recall cross-session context.
* **Universal Sandboxing**: Run shell operations locally, in Docker, or via serverless platforms (Daytona, E2B, Modal).
* **Shareable Skills**: Export and import distilled skills via the open `SKILL.md` format.

## 🚀 Quick Start

The fastest way to run Distill locally is using Docker Compose:

```bash
# 1. Copy the environment template
cp an-api.env.example an-api.env

# 2. Add your LLM API Key to an-api.env (e.g., MOONSHOT_API_KEY or OPENAI_API_KEY)
# By default, Distill uses LiteLLM and supports OpenAI, Anthropic, Gemini, Ollama, etc.

# 3. Start the application
docker compose up -d --build
```

* **Control Panel (UI)**: `http://localhost:5173`
* **API / Docs**: `http://localhost:8000/docs`

### CLI Installation

You can also install Distill globally using `npm` to get the robust interactive CLI installer and TUI:

```powershell
npx @aspct3434/distill-agent install
```

After installation, use the CLI to manage the agent:
```bash
npm install -g @aspct3434/distill-agent
distill                           # Open the interactive terminal UI
distill start                     # Start the backend and control panel
distill logs                      # View running logs
distill update                    # Pull the latest changes

# npx works too if you prefer not to install globally:
npx @aspct3434/distill-agent start    # Start the backend and control panel
npx @aspct3434/distill-agent logs     # View running logs
npx @aspct3434/distill-agent update   # Pull the latest changes
```

## ⚙️ Configuration

Distill is highly configurable via environment variables. Key settings in `an-api.env`:

| Variable | Default | Description |
|---|---|---|
| `AGENT_MODEL` | `moonshot/kimi-k2.5` | The primary LLM to use (supports any LiteLLM provider string). |
| `AGENT_SANDBOX` | (blank) | Leave blank for Docker Compose. Set to `docker` to run Docker-in-Docker, or `http` for serverless. |
| `AGENT_REQUIRE_APPROVAL`| `off` | Set to `risky` or `all` to require human-in-the-loop approval before executing shell commands. |
| `AGENT_API_TOKEN` | (blank) | Secure the API/WebSockets with a Bearer token. |

## 🔌 Integrations & Adapters

Distill can operate directly in your favorite platforms. Adapters are disabled by default and activate when you provide a bot token in `an-api.env`:
* **Telegram**: Set `TELEGRAM_BOT_TOKEN`. Supports voice note transcriptions via Whisper.
* **Discord**: Set `DISCORD_BOT_TOKEN`.
* **Slack**: Set `SLACK_BOT_TOKEN` & `SLACK_APP_TOKEN` (Socket Mode).
* **Email**: Set `EMAIL_ADDRESS`, `EMAIL_PASSWORD`, `EMAIL_IMAP_HOST`, `EMAIL_SMTP_HOST`.

Each channel provides a live typing indicator, streams real-time tool execution logs, and isolates conversations.

## 🧪 Testing

Distill maintains a robust test suite covering contracts, planners, adapters, and the ReAct loop:

```bash
pytest                        # Run the full suite
pytest tests/test_task_contract_loop.py -v   # Test the anti-hallucination contract system
```
