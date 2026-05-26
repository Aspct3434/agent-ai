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

The premium installer is the npm onboarding CLI:

```powershell
npx @aspct3434/agent-ai install
```

It opens an OpenClaw-style terminal onboarding flow with an Agent AI banner, a security warning, QuickStart/manual setup modes, model-provider/API-key setup, sandbox choice, messaging-app setup, startup, and health-check guidance.

After install, manage the app with `npx`:

```powershell
npx @aspct3434/agent-ai doctor
npx @aspct3434/agent-ai start
npx @aspct3434/agent-ai stop
npx @aspct3434/agent-ai restart
npx @aspct3434/agent-ai update
npx @aspct3434/agent-ai logs
```

For shorter repeat commands, install the CLI globally:

```powershell
npm i -g @aspct3434/agent-ai
agent-ai doctor
```

The npm package is a thin installer. It clones or updates `https://github.com/Aspct3434/agent-ai.git`, writes `an-api.env`, and starts the selected runtime mode.

Before the npm package is published, you can test the same CLI from GitHub after pushing this repo:

```powershell
npx --yes github:Aspct3434/agent-ai install
```

If Node/npm is not available yet, use the raw bootstrap installer instead:

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

## Sign in with OpenAI (Codex OAuth)

Instead of pasting an `OPENAI_API_KEY`, you can sign in with your ChatGPT/OpenAI account. An OAuth 2.0 + PKCE flow against `auth.openai.com` (public Codex client) mints an API key from your account, stores it at `AGENT_AUTH_FILE` (`~/.agent-ai/codex_auth.json`), refreshes it, and injects it into `OPENAI_API_KEY` so LiteLLM uses it.

- **Dashboard:** Settings → Authentication → **Sign in with OpenAI**.
- **CLI:** `PYTHONPATH=src python -m auth login` (also `status` / `logout`).
- **Gateway:** `POST /api/auth/login` → open the returned URL; `GET /api/auth/status`; `POST /api/auth/logout`.

The browser redirects to `http://localhost:1455/auth/callback`, so this works for host/local runs. (Heads-up: OpenAI's hosted login couldn't be exercised end-to-end here — if the token-exchange shape has changed, only the request payloads in `src/auth/oauth.py` need adjusting; the surrounding PKCE flow is standard.)

## Local models (Ollama / vLLM / OpenRouter)

The agent uses [LiteLLM](https://docs.litellm.ai/docs/providers) internally, so any provider it supports works — just change `AGENT_MODEL` and supply the matching API key.

**Ollama (fully local, no API key)**

```bash
ollama pull llama3.2          # install Ollama first: https://ollama.com
cp ollama.env.example an-api.env
docker compose up
```

Model strings follow the `ollama/<name>` convention (`ollama/llama3.2`, `ollama/qwen2.5:7b`, etc.). Override `OLLAMA_API_BASE` if Ollama runs on another host.

**vLLM** — set `AGENT_MODEL=openai/<model>`, `OPENAI_API_KEY=dummy`, and `OPENAI_API_BASE=http://localhost:8000/v1`.

**OpenRouter** — set `AGENT_MODEL=openrouter/<org>/<model>` and `OPENROUTER_API_KEY=sk-or-…`. 400+ models, many with a free tier.

See `ollama.env.example` for the full provider reference (Anthropic, OpenAI, Gemini, Azure).

## Messaging adapters (Telegram / Discord / Slack)

The agent can receive and reply to messages on Telegram, Discord, and Slack alongside the browser control panel. Adapters are **disabled by default** and start only when their bot token(s) are present. On every channel the bot:

- shows a **live typing indicator** and streams **what it's executing** (a line per tool call) before the final answer;
- supports in-chat commands: **`/new`** or **`/reset`** (fresh conversation), **`/stop`** (interrupt the current task), **`/help`**;
- can **proactively message you**: ask it to "every morning at 8am summarise X" and the scheduled job delivers its result straight back to that chat;
- (Telegram) **transcribes voice notes** when `AGENT_TRANSCRIBE_MODEL` is set.

**Telegram**

1. Talk to [@BotFather](https://t.me/BotFather) → create a bot → copy the token.
2. Add to `an-api.env`:
   ```
   TELEGRAM_BOT_TOKEN=<token>
   TELEGRAM_ALLOWED_IDS=<comma-separated chat_ids>   # omit to allow everyone
   ```
3. Restart the server — the adapter polls automatically.

**Discord**

1. [discord.com/developers](https://discord.com/developers/applications) → New Application → Bot → copy the token.
2. Under **Bot → Privileged Gateway Intents**, enable **Message Content Intent**.
3. Invite the bot with the `bot` scope and `Send Messages` + `Read Message History` permissions.
4. Add to `an-api.env`:
   ```
   DISCORD_BOT_TOKEN=<token>
   DISCORD_ALLOWED_USER_IDS=<comma-separated user_ids>   # omit to allow everyone
   ```

**Slack**

1. [api.slack.com/apps](https://api.slack.com/apps) → Create New App → enable **Socket Mode**.
2. Add a **Bot Token** (`xoxb-`, scope `chat:write`) and an **App-Level Token** (`xapp-`, scope `connections:write`); subscribe to `message.channels` / `message.im` events.
3. Add to `an-api.env`:
   ```
   SLACK_BOT_TOKEN=xoxb-<token>
   SLACK_APP_TOKEN=xapp-<token>
   SLACK_ALLOWED_USERS=<comma-separated user_ids>   # omit to allow everyone
   ```

Each Telegram chat, Discord channel, and Slack channel gets its own isolated session, so conversation history is scoped correctly per chat.

**Voice notes (Telegram)** — set `AGENT_TRANSCRIBE_MODEL` (e.g. `whisper-1`, or `groq/whisper-large-v3`) and the bot transcribes voice/audio messages, echoes what it heard, and acts on it. Leave blank to disable.

## Rich terminal UI

A streaming terminal client that shows the agent's work live (a line per tool call) and renders the final answer as Markdown — start a gateway, then:

```bash
python src/tui.py                    # connects to ws://127.0.0.1:8000
python src/tui.py --url ws://host:9000/ws/stream
```

In-session commands: `/new` (or `/reset`), `/help`, `/quit`.

## Serverless sandbox backend

Besides `docker`, `ssh`, and host execution, set `AGENT_SANDBOX=http` to run commands on any serverless sandbox (Daytona, E2B, Modal, Vercel Sandbox) via a tiny exec shim. Point `AGENT_SANDBOX_EXEC_URL` at a service exposing one endpoint:

```
POST {url}/exec  {command, cwd, timeout, stdin?, background?, log_path?}
              →  {exit_code, stdout, stderr, pid?}
```

That ~20-line shim is the entire provider integration — no SDK is baked into the repo.

## Cross-session memory (recall)

Every user/assistant turn is indexed in a full-text store (SQLite FTS5, with a LIKE fallback). The agent can search its **own past conversations across all prior sessions** with the `recall_memory` tool ("what did we decide about X last week?"), and you can search them from the dashboard (Memory → *Recall past conversations*) or `GET /api/sessions/search?q=…`.

## Self-improving skills (evidence-gated evolution)

agent-ai doesn't just rewrite skills on use like other agents — it evolves them **safely and reversibly**:

- Every skill is **versioned**; each version's real success rate is tracked from actual invocations.
- An LLM synthesises an improvement after repeated uses *or* repeated failures — but a candidate is **promoted only if it passes a validation gate** (valid Python, keeps the `@skill` decorator, preserves the public function signature). The prior version is archived.
- If a promoted version **measurably regresses** (success rate drops past `SKILL_ROLLBACK_MARGIN` over `SKILL_ROLLBACK_MIN_SAMPLES` uses), it is **automatically rolled back** to the previous version.

Net effect: skills only ever get better — a bad "improvement" can't stick.

**Auto skill maker** — the agent can author its own reusable tools on the fly with the `create_skill` tool (validated for syntax + `@skill` + a defined function, then registered and immediately callable), in addition to the automatic post-task distillation of successful trajectories.

## Shareable skills (agentskills.io)

Distilled skills export to / import from the open [agentskills.io](https://agentskills.io) `SKILL.md` format (YAML frontmatter + a fenced `python` block):

```bash
curl localhost:8000/api/skills/<name>/export.md      # download a SKILL.md
curl -X POST localhost:8000/api/skills/import-md -d '{"text":"<SKILL.md>"}'
```

## Running tests

```bash
pytest                        # full suite
pytest tests/test_task_contract_loop.py -v   # contract system only
pytest --self-test src/eval_harness.py       # harness self-test (no API key needed)
```

## Windows development

For one-click installs on Windows, use the PowerShell bootstrap/installer and keep `Sandbox on` for Docker Compose container isolation. If you run `uvicorn` directly on Windows instead of Compose, set `AGENT_SANDBOX=docker` and `AGENT_SANDBOX_HOST_FALLBACK=false` so agent tasks execute inside Docker rather than the host shell.
