# ruff: noqa: E402, I001
import asyncio
import base64
import json
import logging
import os
import shutil
import sqlite3
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

# Load .env BEFORE any local imports so module-level os.getenv() calls in
# agent.py / tools.py see the values (e.g. AGENT_SANDBOX, AGENT_MODEL, etc.).
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / "an-api.env")

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from agent import AgentEngine, NormalizedMessage
from approvals import ApprovalGate
from auth.oauth import CodexOAuth, wait_for_callback
from checkpointer import StateCheckpointer, initialize_checkpoints_db
from evaluator import SkillDistiller, SkillRegistry
from evolution import EvolutionEngine
from memory import UserProfileStore
from persona import PersonaLoader
from scheduler import CronScheduler, _validate_schedule
from tools import ToolManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory log ring buffer (powers the dashboard Logs panel)
# ---------------------------------------------------------------------------

_LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


class _LogRingBuffer(logging.Handler):
    """Keep the most recent log records in memory for the /api/logs endpoint."""

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self._records: deque[dict[str, Any]] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._records.append({
                "time": record.created,
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
            })
        except Exception:
            pass

    def snapshot(self, level: str = "ALL", limit: int = 200) -> list[dict[str, Any]]:
        items = list(self._records)
        if level and level != "ALL":
            floor = _LOG_LEVELS.get(level.upper(), 0)
            items = [r for r in items if _LOG_LEVELS.get(r["level"], 0) >= floor]
        return items[-limit:]

    def clear(self) -> None:
        self._records.clear()


# Attach once at import so the buffer captures startup logs too.
_log_buffer = _LogRingBuffer()
_log_buffer.setLevel(logging.INFO)
_root_logger = logging.getLogger()
if _root_logger.level == 0 or _root_logger.level > logging.INFO:
    _root_logger.setLevel(logging.INFO)
_root_logger.addHandler(_log_buffer)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

_RATE_LIMIT_RPM = int(os.getenv("GATEWAY_RATE_LIMIT_RPM", "60"))
_RATE_LIMIT_WINDOW = 60  # seconds


class _SlidingWindowRateLimiter:
    """Per-key sliding-window rate limiter (requests per 60-second window).

    Safe for concurrent coroutines via an asyncio.Lock. Set
    ``GATEWAY_RATE_LIMIT_RPM=0`` to disable (unlimited).
    """

    def __init__(self, calls_per_minute: int) -> None:
        self._limit = calls_per_minute
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._last_sweep = 0.0

    async def check(self, key: str) -> None:
        """Raise HTTP 429 if *key* has exceeded its per-minute quota."""
        if self._limit == 0:
            return
        async with self._lock:
            now = time.monotonic()
            cutoff = now - _RATE_LIMIT_WINDOW
            self._sweep_expired(cutoff, now)
            bucket = [t for t in self._windows[key] if t > cutoff]
            if len(bucket) >= self._limit:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: max {self._limit} requests/min per session.",
                )
            bucket.append(now)
            self._windows[key] = bucket

    def _sweep_expired(self, cutoff: float, now: float) -> None:
        """Drop keys whose timestamps have all expired so the dict stays bounded.

        Without this, every distinct session/chat id would leave a permanent
        entry behind, leaking memory over the lifetime of the server. Runs at
        most once per window to keep per-request cost negligible.
        """
        if now - self._last_sweep < _RATE_LIMIT_WINDOW:
            return
        self._last_sweep = now
        stale = [
            key
            for key, hits in self._windows.items()
            if not any(t > cutoff for t in hits)
        ]
        for key in stale:
            del self._windows[key]


_rate_limiter = _SlidingWindowRateLimiter(_RATE_LIMIT_RPM)

_STOP = object()


class Message(BaseModel):
    session_id: str
    payload: dict[str, Any]
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class Result(BaseModel):
    message_id: str
    session_id: str
    output: Any = None
    error: str | None = None


class WebhookPayload(BaseModel):
    chat_id: str
    user_id: str
    text: str


class WebhookResponse(BaseModel):
    chat_id: str
    user_id: str
    reply: str | None = None
    error: str | None = None


class ReplayCheckpointRequest(BaseModel):
    correction: str | None = None


Handler = Callable[[Message], Awaitable[Any]]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _SessionLane:
    """
    One FIFO queue + one worker task per session.
    Messages submitted here are processed strictly in order.
    """

    def __init__(self, session_id: str, handler: Handler) -> None:
        self.session_id = session_id
        self._handler = handler
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"lane:{self.session_id}")

    async def submit(self, message: Message) -> Result:
        future: asyncio.Future[Result] = asyncio.get_running_loop().create_future()
        await self._queue.put((message, future))
        return await future

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is _STOP:
                    return
                message, future = item
                try:
                    output = await self._handler(message)
                    result = Result(
                        message_id=message.message_id,
                        session_id=message.session_id,
                        output=output,
                    )
                except Exception as exc:
                    result = Result(
                        message_id=message.message_id,
                        session_id=message.session_id,
                        error=str(exc),
                    )
                if not future.cancelled():
                    future.set_result(result)
            finally:
                self._queue.task_done()

    async def drain_and_stop(self) -> None:
        """Enqueue the stop sentinel and wait for the worker to finish."""
        await self._queue.put(_STOP)
        if self._task:
            await self._task


class SessionLaneManager:
    """
    Routes each message to its session's lane, creating one on first use.
    Different sessions run their lanes concurrently; same-session messages
    are serialised inside their lane's queue.
    """

    def __init__(self, handler: Handler) -> None:
        self._handler = handler
        self._lanes: dict[str, _SessionLane] = {}
        self._lock = asyncio.Lock()

    async def submit(self, message: Message) -> Result:
        lane = await self._get_or_create_lane(message.session_id)
        return await lane.submit(message)

    async def _get_or_create_lane(self, session_id: str) -> _SessionLane:
        async with self._lock:
            if session_id not in self._lanes:
                lane = _SessionLane(session_id, self._handler)
                lane.start()
                self._lanes[session_id] = lane
            return self._lanes[session_id]

    async def shutdown(self) -> None:
        async with self._lock:
            lanes = list(self._lanes.values())
            self._lanes.clear()
        await asyncio.gather(*(lane.drain_and_stop() for lane in lanes))

    @property
    def active_sessions(self) -> list[str]:
        return list(self._lanes.keys())


class Gateway:
    """Public entry point. Wraps SessionLaneManager with a simpler send() API."""

    def __init__(self, handler: Handler) -> None:
        self._manager = SessionLaneManager(handler)

    async def send(self, session_id: str, payload: dict[str, Any]) -> Result:
        message = Message(session_id=session_id, payload=payload)
        return await self._manager.submit(message)

    async def shutdown(self) -> None:
        await self._manager.shutdown()

    @property
    def active_sessions(self) -> list[str]:
        return self._manager.active_sessions


class _EmptyMemory:
    """Runtime fallback when no memory backend is needed for a simple webhook."""

    def retrieve_context(self, query: str, query_type: str) -> dict[str, Any]:
        return {"query_type": query_type, "results": []}

    def store_event(
        self, session_id: str, raw_text: str, entities: dict[str, Any]
    ) -> str:
        return ""


def _build_memory() -> Any:
    """Return a memory backend for the agent.

    Defaults to :class:`_EmptyMemory`. Set ``AGENT_USE_HYBRID_MEMORY=true`` to
    opt into the ChromaDB + Neo4j :class:`HybridMemory`; if its dependencies or
    backends are unavailable the call falls back to ``_EmptyMemory`` so the API
    still comes up.
    """
    if os.getenv("AGENT_USE_HYBRID_MEMORY", "true").lower() not in {"1", "true", "yes", "on"}:
        return _EmptyMemory()
    try:
        from memory import HybridMemory

        return HybridMemory(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "neo4j"),
            chroma_path=os.getenv("CHROMA_PATH", "./chroma_db"),
        )
    except Exception as exc:
        logger.warning("HybridMemory unavailable (%s); falling back to empty memory.", exc)
        return _EmptyMemory()


def _find_sqlite_server() -> str:
    if command := shutil.which("mcp-server-sqlite"):
        return command
    raise RuntimeError(
        "mcp-server-sqlite is not installed or is not available on PATH"
    )


def _ensure_sqlite_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE
            )
            """
        )
        conn.executemany(
            "INSERT OR IGNORE INTO users (id, name, email) VALUES (?, ?, ?)",
            [
                (1, "Alice Example", "alice@example.test"),
                (2, "Bob Sample", "bob@example.test"),
                (3, "Charlie Demo", "charlie@example.test"),
            ],
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = Path(os.getenv("SQLITE_DB_PATH", str(_PROJECT_ROOT / "test.db")))
    checkpoint_db_path = Path(
        os.getenv("CHECKPOINT_DB_PATH", str(_PROJECT_ROOT / "checkpoints.db"))
    )
    cron_db_path = Path(os.getenv("CRON_DB_PATH", str(_PROJECT_ROOT / "cron_jobs.db")))
    skills_dir = Path(os.getenv("SKILLS_DIR", str(_PROJECT_ROOT / "skills")))
    persona_dir = Path(os.getenv("PERSONA_DIR", str(_PROJECT_ROOT / "persona")))
    chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")

    model = os.getenv("AGENT_MODEL", "moonshot/kimi-k2.5")
    fast_model = os.getenv("FAST_AGENT_MODEL", model)
    strong_model = os.getenv("STRONG_AGENT_MODEL", model)

    _ensure_sqlite_db(db_path)
    await initialize_checkpoints_db(checkpoint_db_path)

    data_dir = Path(os.getenv("AGENT_DATA_DIR", str(_PROJECT_ROOT / "data")))
    checkpointer = StateCheckpointer(checkpoint_db_path)

    from tools import _SANDBOX_ENABLED, _SANDBOX_IMAGE
    if _SANDBOX_ENABLED:
        logger.info("Docker sandbox mode ENABLED (image=%s)", _SANDBOX_IMAGE)
    else:
        logger.info("Docker sandbox mode disabled — running commands on host")

    tools = ToolManager()
    # A missing/broken optional MCP server must never take down the whole
    # gateway — otherwise the UI just shows "disconnected". Each connect is
    # isolated so the API still comes up (in a degraded mode) on failure.
    try:
        await tools.connect_server(
            name="sqlite",
            command=_find_sqlite_server(),
            args=["--db-path", str(db_path)],
        )
    except Exception as exc:
        logger.warning(
            "SQLite MCP server unavailable (%s); continuing without it. "
            "Install it with: pip install mcp-server-sqlite",
            exc,
        )
    try:
        await tools.connect_filesystem_server(data_dir)
    except RuntimeError:
        pass
    try:
        await tools.connect_skills_server(skills_dir)
    except Exception as exc:
        logger.warning("Skills MCP server unavailable (%s); continuing without it.", exc)

    # -- Persona ----------------------------------------------------------
    persona = PersonaLoader(persona_dir if persona_dir.exists() else None)
    persona_content = persona.load()

    # -- User profile store -----------------------------------------------
    profile_store = UserProfileStore(profile_dir=chroma_path)

    # -- Skill registry ---------------------------------------------------
    skill_registry = SkillRegistry(
        skills_dir=skills_dir,
        model=fast_model,
        improve_after_uses=int(os.getenv("SKILL_IMPROVE_AFTER_USES", "5")),
    )

    # -- Proof-carrying evolution ----------------------------------------
    evolution_db_path = Path(os.getenv("EVOLUTION_DB_PATH", str(data_dir / "evolution.db")))
    evolution_engine = EvolutionEngine(
        ledger_path=evolution_db_path,
        skills_dir=skills_dir,
        skill_registry=skill_registry,
        model=fast_model,
    )

    # -- Distiller --------------------------------------------------------
    distiller = SkillDistiller(
        skills_dir=skills_dir,
        model=fast_model,
        evolution_engine=evolution_engine,
    )
    await distiller.start()

    # -- Memory -----------------------------------------------------------
    memory = _build_memory()

    # -- Cross-session memory (full-text recall across past chats) --------
    from session_store import SessionStore
    session_store = SessionStore(
        os.getenv("AGENT_SESSION_DB", str(data_dir / "sessions.db"))
    )

    # -- Command-approval gate (human-in-the-loop for risky commands) -----
    approval_gate = ApprovalGate()

    # -- Engine -----------------------------------------------------------
    engine = AgentEngine(
        memory=memory,
        tools=tools,
        model=model,
        fast_model=fast_model,
        strong_model=strong_model,
        distiller=distiller,
        checkpointer=checkpointer,
        profile_store=profile_store,
        skill_registry=skill_registry,
        persona_content=persona_content,
        session_store=session_store,
        approval_gate=approval_gate,
        evolution_engine=evolution_engine,
    )

    # -- Cron scheduler ---------------------------------------------------
    async def _scheduled_runner(session_id: str, prompt: str) -> str:
        return await engine.process_task(
            NormalizedMessage(session_id=session_id, role="user", content=prompt)
        )

    scheduler = CronScheduler(db_path=str(cron_db_path), runner=_scheduled_runner)
    await scheduler.start()
    engine._scheduler = scheduler  # inject after both are created

    async def handler(message: Message) -> str:
        return await engine.process_task(
            NormalizedMessage(
                session_id=message.session_id,
                role="user",
                content=message.payload["text"],
            )
        )

    # -- Codex OAuth (sign-in alternative to a pasted API key) ------------
    oauth = CodexOAuth()
    oauth.ensure_fresh()  # refresh + re-inject OPENAI_API_KEY if a token exists
    app.state.oauth = oauth

    gw = Gateway(handler)
    app.state.tools = tools
    app.state.distiller = distiller
    app.state.engine = engine
    app.state.checkpointer = checkpointer
    app.state.memory = memory
    app.state.persona = persona
    app.state.profile_store = profile_store
    app.state.skill_registry = skill_registry
    app.state.evolution_engine = evolution_engine
    app.state.scheduler = scheduler
    app.state.session_store = session_store
    app.state.approval_gate = approval_gate
    app.state.gateway = gw
    app.state.active_stream_tasks = {}

    # -- Messaging adapters (optional) ------------------------------------
    # Each adapter starts only when its bot-token env var is present so the
    # server comes up cleanly with no messaging credentials configured.

    from adapters.telegram import TelegramAdapter
    from adapters.discord_bot import DiscordAdapter
    from adapters.slack import SlackAdapter
    from adapters.email_adapter import EmailAdapter

    async def _adapter_stream(session_id: str, text: str):
        """Stream the agent's live progress (tool calls + final answer) so
        messaging adapters can surface what the agent is doing, OpenClaw-style.

        Same-session ordering is enforced inside each adapter (per-chat lock),
        mirroring how the WebSocket UI drives ``stream_task`` directly.
        """
        async for event in engine.stream_task(
            NormalizedMessage(session_id=session_id, role="user", content=text)
        ):
            yield event

    def _adapter_reset(session_id: str) -> bool:
        """Clear a chat's conversation history (the /new and /reset commands)."""
        return engine.reset_session(session_id)

    telegram_adapter: TelegramAdapter | None = None
    discord_adapter: DiscordAdapter | None = None
    slack_adapter: SlackAdapter | None = None
    email_adapter: EmailAdapter | None = None

    if tg_token := os.getenv("TELEGRAM_BOT_TOKEN"):
        telegram_adapter = TelegramAdapter(
            token=tg_token, stream_fn=_adapter_stream, reset_fn=_adapter_reset
        )
        await telegram_adapter.start()
        app.state.telegram_adapter = telegram_adapter

    if dc_token := os.getenv("DISCORD_BOT_TOKEN"):
        discord_adapter = DiscordAdapter(
            token=dc_token, stream_fn=_adapter_stream, reset_fn=_adapter_reset
        )
        await discord_adapter.start()
        app.state.discord_adapter = discord_adapter

    slack_bot = os.getenv("SLACK_BOT_TOKEN")
    slack_app = os.getenv("SLACK_APP_TOKEN")
    if slack_bot and slack_app:
        slack_adapter = SlackAdapter(
            bot_token=slack_bot,
            app_token=slack_app,
            stream_fn=_adapter_stream,
            reset_fn=_adapter_reset,
        )
        await slack_adapter.start()
        app.state.slack_adapter = slack_adapter

    email_addr = os.getenv("EMAIL_ADDRESS")
    email_pass = os.getenv("EMAIL_PASSWORD")
    email_imap = os.getenv("EMAIL_IMAP_HOST")
    email_smtp = os.getenv("EMAIL_SMTP_HOST")
    if email_addr and email_pass and email_imap and email_smtp:
        email_adapter = EmailAdapter(
            imap_host=email_imap,
            smtp_host=email_smtp,
            address=email_addr,
            password=email_pass,
            stream_fn=_adapter_stream,
            reset_fn=_adapter_reset,
            imap_port=int(os.getenv("EMAIL_IMAP_PORT", "993")),
            smtp_port=int(os.getenv("EMAIL_SMTP_PORT", "465")),
            poll_interval=float(os.getenv("EMAIL_POLL_INTERVAL", "20")),
        )
        await email_adapter.start()
        app.state.email_adapter = email_adapter

    # -- Scheduled-task delivery to messaging --------------------------------
    # Lets cron jobs push their result to a chat (deliver_to="tg:123" etc.),
    # so the agent can proactively message you — the always-on use case.
    async def _deliver(target: str, text: str) -> None:
        if not text:
            return
        if target.startswith("tg:") and telegram_adapter is not None:
            await telegram_adapter.deliver(int(target[3:]), text)
        elif target.startswith("discord:") and discord_adapter is not None:
            await discord_adapter.deliver(target[len("discord:") :], text)
        elif target.startswith("slack:") and slack_adapter is not None:
            await slack_adapter.deliver(target[len("slack:") :], text)
        elif target.startswith("email:") and email_adapter is not None:
            await email_adapter.deliver(target[len("email:") :], text)
        else:
            logger.warning("Cron delivery target %r has no active adapter", target)

    scheduler.set_delivery(_deliver)

    try:
        yield
    finally:
        # Adapters first — they hold references to the gateway.
        if telegram_adapter is not None:
            await telegram_adapter.shutdown()
        if discord_adapter is not None:
            await discord_adapter.shutdown()
        if slack_adapter is not None:
            await slack_adapter.shutdown()
        if email_adapter is not None:
            await email_adapter.shutdown()
        for task in list(app.state.active_stream_tasks.values()):
            task.cancel()
        await asyncio.gather(
            *list(app.state.active_stream_tasks.values()),
            return_exceptions=True,
        )
        await gw.shutdown()
        await scheduler.shutdown()
        await distiller.shutdown()
        await tools.close()
        if hasattr(memory, "close"):
            try:
                memory.close()
            except Exception as exc:
                logger.warning("Memory close failed: %s", exc)


app = FastAPI(lifespan=lifespan)


_PROXY_REQUEST_SKIP_HEADERS = {
    "host",
    "content-length",
    "connection",
    "accept-encoding",
}
_PROXY_RESPONSE_SKIP_HEADERS = {
    "content-encoding",
    "content-length",
    "connection",
    "transfer-encoding",
}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
async def status() -> dict[str, Any]:
    """At-a-glance state for the dashboard Overview page."""
    engine: AgentEngine = app.state.engine
    skills = app.state.skill_registry.list_skills()
    jobs = app.state.scheduler.list_jobs()
    evolution = app.state.evolution_engine.status()
    return {
        "model": engine._model,
        "fast_model": engine._fast_model,
        "strong_model": engine._strong_model,
        "sandbox": os.getenv("AGENT_SANDBOX", "") or "host",
        "channels": {
            "telegram": hasattr(app.state, "telegram_adapter"),
            "discord": hasattr(app.state, "discord_adapter"),
            "slack": hasattr(app.state, "slack_adapter"),
            "email": hasattr(app.state, "email_adapter"),
        },
        "skills": {
            "count": len(skills),
            "improved": sum(1 for s in skills if s.get("version", 1) > 1),
        },
        "cron": {
            "count": len(jobs),
            "enabled": sum(1 for j in jobs if j.get("enabled")),
        },
        "evolution": evolution["candidates"],
        "task_graph": engine.task_graph_status_summary(),
        "active_sessions": len(app.state.gateway.active_sessions),
    }


@app.get("/api/sessions/search")
async def search_sessions(q: str = "", limit: int = 20) -> list[dict[str, Any]]:
    """Full-text search across all past conversation turns."""
    return await asyncio.to_thread(
        app.state.session_store.search, q, max(1, min(limit, 100))
    )


@app.get("/api/sessions/recent")
async def recent_sessions(limit: int = 20) -> list[dict[str, Any]]:
    return await asyncio.to_thread(
        app.state.session_store.recent_sessions, max(1, min(limit, 100))
    )


@app.get("/api/tools")
async def list_tools() -> list[dict[str, Any]]:
    """The agent's live tool inventory (builtins + connected MCP servers)."""
    tools = await app.state.tools.list_all_tools()
    return [
        {
            "name": t.get("name", ""),
            "server": t.get("server", ""),
            "description": (t.get("description") or "")[:200],
        }
        for t in tools
    ]


@app.get("/api/logs")
async def get_logs(level: str = "ALL", limit: int = 200) -> list[dict[str, Any]]:
    """Recent gateway/agent log records (in-memory ring buffer)."""
    return _log_buffer.snapshot(level=level, limit=max(1, min(limit, 500)))


@app.delete("/api/logs")
async def clear_logs() -> dict[str, str]:
    _log_buffer.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Proof-carrying task graph API
# ---------------------------------------------------------------------------


@app.get("/api/task-graph/{session_id}")
async def get_task_graph(session_id: str) -> dict[str, Any]:
    snapshot = app.state.engine.task_graph_snapshot(session_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return snapshot


@app.post("/api/task-graph/{session_id}/verify")
async def verify_task_graph(session_id: str) -> dict[str, Any]:
    result = app.state.engine.verify_task_graph(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return result


# ---------------------------------------------------------------------------
# Proof-carrying evolution API
# ---------------------------------------------------------------------------


@app.get("/api/evolution/status")
async def evolution_status() -> dict[str, Any]:
    return app.state.evolution_engine.status()


@app.get("/api/evolution/candidates")
async def list_evolution_candidates(status: str | None = None) -> list[dict[str, Any]]:
    return app.state.evolution_engine.list_candidates(status=status)


@app.get("/api/evolution/candidates/{candidate_id}")
async def get_evolution_candidate(candidate_id: str) -> dict[str, Any]:
    candidate = app.state.evolution_engine.inspect_candidate(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id!r} not found")
    return candidate


class EvolutionRunRequest(BaseModel):
    limit: int = 5


@app.post("/api/evolution/run")
async def run_evolution(payload: EvolutionRunRequest | None = None) -> dict[str, Any]:
    limit = payload.limit if payload is not None else 5
    result = app.state.evolution_engine.run_cycle(limit=limit)
    try:
        await app.state.tools.connect_skills_server()
    except Exception as exc:
        logger.warning("Evolution run: skills reconnect failed: %s", exc)
    return result


@app.post("/api/evolution/candidates/{candidate_id}/rollback")
async def rollback_evolution_candidate(candidate_id: str) -> dict[str, Any]:
    result = app.state.evolution_engine.rollback(candidate_id)
    if result.get("error") == "candidate not found":
        raise HTTPException(status_code=404, detail=result["error"])
    try:
        await app.state.tools.connect_skills_server()
    except Exception as exc:
        logger.warning("Evolution rollback: skills reconnect failed: %s", exc)
    return result


# ---------------------------------------------------------------------------
# Command-approval API (human-in-the-loop)
# ---------------------------------------------------------------------------


@app.get("/api/approvals")
async def list_approvals() -> dict[str, Any]:
    gate = app.state.approval_gate
    return {"mode": gate.mode, "pending": gate.pending()}


class ApprovalDecision(BaseModel):
    approved: bool


@app.post("/api/approvals/{request_id}")
async def resolve_approval(request_id: str, payload: ApprovalDecision) -> dict[str, Any]:
    resolved = app.state.approval_gate.resolve(request_id, payload.approved)
    if not resolved:
        raise HTTPException(status_code=404, detail="Approval request not found or already resolved")
    return {"request_id": request_id, "approved": payload.approved}


# ---------------------------------------------------------------------------
# Codex OAuth API
# ---------------------------------------------------------------------------


@app.get("/api/auth/status")
async def auth_status() -> dict[str, Any]:
    return app.state.oauth.status()


@app.post("/api/auth/login")
async def auth_login() -> dict[str, Any]:
    """Start the OAuth flow: return the authorize URL and await the callback.

    The browser is opened client-side to the returned URL; the provider then
    redirects to the local callback (port 1455) which a background task awaits.
    """
    oauth: CodexOAuth = app.state.oauth
    url = oauth.authorize_url()

    async def _await_callback() -> None:
        try:
            result = await asyncio.to_thread(wait_for_callback, 300.0)
            if result.get("error"):
                logger.warning("OAuth callback error: %s", result["error"])
                return
            await asyncio.to_thread(oauth.complete, result["code"], result["state"])
        except Exception as exc:
            logger.warning("OAuth login flow failed: %s", exc)

    asyncio.create_task(_await_callback())  # noqa: RUF006
    return {"authorize_url": url}


@app.post("/api/auth/logout")
async def auth_logout() -> dict[str, str]:
    app.state.oauth.logout()
    return {"status": "signed_out"}


@app.post("/api/sessions/{session_id}/cancel")
async def cancel_session(session_id: str) -> dict[str, Any]:
    task = app.state.active_stream_tasks.get(session_id)
    if task is None or task.done():
        return {"cancelled": False, "session_id": session_id}
    task.cancel()
    return {"cancelled": True, "session_id": session_id}


@app.api_route(
    "/proxy/{port}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
@app.api_route(
    "/proxy/{port}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_local_http_service(
    port: int,
    request: Request,
    path: str = "",
) -> Response:
    """Proxy browser traffic to HTTP services running inside the agent container."""
    if port < 1 or port > 65535:
        raise HTTPException(status_code=400, detail="Port must be between 1 and 65535")

    target_url = httpx.URL(
        f"http://127.0.0.1:{port}/{path}",
        query=request.url.query.encode("utf-8"),
    )
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _PROXY_REQUEST_SKIP_HEADERS
    }

    tools = getattr(app.state, "tools", None)
    if tools is not None and getattr(tools, "sandbox_active", False):
        try:
            upstream_payload = await asyncio.to_thread(
                tools.proxy_local_http_service,
                port=port,
                path=path,
                query=request.url.query,
                method=request.method,
                headers=headers,
                body=await request.body(),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Sandbox service proxy failed for port {port}: {exc}",
            ) from exc

        body = base64.b64decode(str(upstream_payload.get("body_b64") or ""))
        response_headers = {
            key: value
            for key, value in (upstream_payload.get("headers") or {}).items()
            if key.lower() not in _PROXY_RESPONSE_SKIP_HEADERS
        }
        return Response(
            content=body,
            status_code=int(upstream_payload.get("status_code") or 502),
            headers=response_headers,
            media_type=response_headers.get("content-type"),
        )

    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=30.0) as client:
            upstream = await client.request(
                request.method,
                target_url,
                headers=headers,
                content=await request.body(),
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach local HTTP service on port {port}: {exc}",
        ) from exc

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in _PROXY_RESPONSE_SKIP_HEADERS
    }
    content = _rewrite_proxy_html(upstream.content, upstream.headers.get("content-type", ""), port)
    return Response(
        content=content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )


def _rewrite_proxy_html(content: bytes, content_type: str, port: int) -> bytes:
    if "text/html" not in content_type.lower():
        return content
    try:
        html = content.decode("utf-8")
    except UnicodeDecodeError:
        return content

    prefix = f"/proxy/{port}/"
    replacements = {
        'href="/': f'href="{prefix}',
        'src="/': f'src="{prefix}',
        'action="/': f'action="{prefix}',
        "href='/": f"href='{prefix}",
        "src='/": f"src='{prefix}",
        "action='/": f"action='{prefix}",
    }
    for old, new in replacements.items():
        html = html.replace(old, new)
    return html.encode("utf-8")


@app.post("/webhook", response_model=WebhookResponse)
async def webhook(payload: WebhookPayload) -> WebhookResponse:
    await _rate_limiter.check(payload.chat_id)
    result = await app.state.gateway.send(
        payload.chat_id,
        {
            "user_id": payload.user_id,
            "text": payload.text,
        },
    )
    if result.error is not None:
        raise HTTPException(status_code=500, detail=result.error)
    return WebhookResponse(
        chat_id=payload.chat_id,
        user_id=payload.user_id,
        reply=str(result.output),
    )


@app.get("/api/sessions/{session_id}/checkpoints")
async def list_session_checkpoints(session_id: str) -> list[dict[str, Any]]:
    return list(await app.state.checkpointer.list_checkpoints(session_id))


@app.post("/api/checkpoints/{checkpoint_id}/replay")
async def replay_checkpoint(
    checkpoint_id: str,
    payload: ReplayCheckpointRequest,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    final_output = ""

    try:
        async for event in app.state.engine.replay_from_checkpoint(
            checkpoint_id,
            user_correction=payload.correction,
        ):
            events.append(event)
            if event.get("type") == "text":
                final_output = str(event.get("content", ""))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "checkpoint_id": checkpoint_id,
        "events": events,
        "final_output": final_output,
    }


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    """Stream agent execution events over WebSocket.

    Client sends: {"session_id": "...", "text": "..."}
    Server sends a sequence of JSON payloads until the turn is complete:
      {"type": "status",    "message": "Thinking..."}
      {"type": "tool_call", "tool": "...", "params": {...}}
      {"type": "text",      "content": "..."}
    """
    await websocket.accept()
    current_task: asyncio.Task[None] | None = None

    async def stream_turn(msg: NormalizedMessage) -> None:
        try:
            async for event in app.state.engine.stream_task(msg):
                await websocket.send_text(json.dumps(event))
        except asyncio.CancelledError:
            with suppress(Exception):
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "final_answer",
                            "reason": "cancelled",
                            "content": "Stopped by user.",
                        }
                    )
                )
            raise

    def cleanup_stream_task(session_id: str, task: asyncio.Task[None]) -> None:
        if app.state.active_stream_tasks.get(session_id) is task:
            app.state.active_stream_tasks.pop(session_id, None)
        with suppress(asyncio.CancelledError, Exception):
            task.result()

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            session_id = data.get("session_id", "__anon__")
            if data.get("type") == "cancel":
                task = app.state.active_stream_tasks.get(session_id)
                if task is not None and not task.done():
                    task.cancel()
                continue

            if current_task is not None and not current_task.done():
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "detail": "A task is already running for this connection.",
                        }
                    )
                )
                continue

            try:
                await _rate_limiter.check(session_id)
            except HTTPException as exc:
                await websocket.send_text(json.dumps({"type": "error", "detail": exc.detail}))
                continue
            msg = NormalizedMessage(
                session_id=session_id,
                role="user",
                content=data["text"],
            )
            current_task = asyncio.create_task(stream_turn(msg))
            app.state.active_stream_tasks[session_id] = current_task
            current_task.add_done_callback(
                lambda task, sid=session_id: cleanup_stream_task(sid, task)
            )
    except WebSocketDisconnect:
        if current_task is not None and not current_task.done():
            current_task.cancel()
            with suppress(asyncio.CancelledError):
                await current_task


# ---------------------------------------------------------------------------
# Model config API
# ---------------------------------------------------------------------------


class ModelConfigRequest(BaseModel):
    model: str | None = None
    fast_model: str | None = None
    strong_model: str | None = None


@app.get("/api/config/model")
async def get_model_config() -> dict[str, Any]:
    engine: AgentEngine = app.state.engine
    return {
        "model": engine._model,
        "fast_model": engine._fast_model,
        "strong_model": engine._strong_model,
    }


@app.post("/api/config/model")
async def set_model_config(payload: ModelConfigRequest) -> dict[str, Any]:
    engine: AgentEngine = app.state.engine
    engine.update_models(
        model=payload.model,
        fast_model=payload.fast_model,
        strong_model=payload.strong_model,
    )
    return {
        "model": engine._model,
        "fast_model": engine._fast_model,
        "strong_model": engine._strong_model,
        "updated": True,
    }


# ---------------------------------------------------------------------------
# Persona API
# ---------------------------------------------------------------------------


@app.get("/api/persona")
async def get_persona() -> dict[str, Any]:
    return app.state.persona.describe()


class PersonaLoadRequest(BaseModel):
    persona_name: str | None = None


@app.post("/api/persona/load")
async def load_persona(payload: PersonaLoadRequest) -> dict[str, Any]:
    persona: PersonaLoader = app.state.persona
    content = persona.load(payload.persona_name)
    # Hot-reload: update the engine's system directive with new persona.
    engine: AgentEngine = app.state.engine
    from agent import SYSTEM_DIRECTIVE
    engine._system_directive = (
        f"{content}\n\n{SYSTEM_DIRECTIVE}" if content else SYSTEM_DIRECTIVE
    )
    return {**persona.describe(), "content_preview": content[:300]}


# ---------------------------------------------------------------------------
# User profile API
# ---------------------------------------------------------------------------


@app.get("/api/profile")
async def get_user_profile() -> dict[str, Any]:
    return app.state.profile_store.get()


@app.delete("/api/profile")
async def clear_user_profile() -> dict[str, str]:
    app.state.profile_store.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Skill registry API
# ---------------------------------------------------------------------------


@app.get("/api/skills")
async def list_skills() -> list[dict[str, Any]]:
    return app.state.skill_registry.list_skills()


@app.get("/api/skills/{skill_name}/export")
async def export_skill(skill_name: str) -> dict[str, Any]:
    result = app.state.skill_registry.export_skill(skill_name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Skill {skill_name!r} not found")
    return result


class SkillImportRequest(BaseModel):
    name: str
    code: str
    description: str = ""
    tags: list[str] = []


@app.post("/api/skills/import")
async def import_skill(payload: SkillImportRequest) -> dict[str, str]:
    try:
        path = app.state.skill_registry.import_skill(payload.model_dump())
        # Reconnect the skills MCP server so the new skill is immediately available.
        try:
            await app.state.tools.connect_skills_server()
        except Exception:
            pass
        return {"status": "imported", "path": path}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/skills/{skill_name}/export.md")
async def export_skill_md(skill_name: str) -> Response:
    """Export a skill as an agentskills.io-compatible SKILL.md document."""
    md = app.state.skill_registry.export_skill_md(skill_name)
    if md is None:
        raise HTTPException(status_code=404, detail=f"Skill {skill_name!r} not found")
    return Response(content=md, media_type="text/markdown")


class SkillImportMdRequest(BaseModel):
    text: str  # raw SKILL.md document


@app.post("/api/skills/import-md")
async def import_skill_md(payload: SkillImportMdRequest) -> dict[str, str]:
    """Import a skill from a raw agentskills.io SKILL.md document."""
    try:
        path = app.state.skill_registry.import_skill_md(payload.text)
        try:
            await app.state.tools.connect_skills_server()
        except Exception:
            pass
        return {"status": "imported", "path": path}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Cron scheduler API
# ---------------------------------------------------------------------------


class CronJobRequest(BaseModel):
    prompt: str
    schedule_type: str
    schedule_spec: str
    label: str = ""
    session_id: str = "cron"
    deliver_to: str = ""  # e.g. "tg:12345" / "discord:67890" / "slack:C123"


@app.get("/api/cron/jobs")
async def list_cron_jobs() -> list[dict[str, Any]]:
    return app.state.scheduler.list_jobs()


@app.post("/api/cron/jobs")
async def create_cron_job(payload: CronJobRequest) -> dict[str, Any]:
    try:
        _validate_schedule(payload.schedule_type, payload.schedule_spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job = await app.state.scheduler.add_job(
        schedule_type=payload.schedule_type,
        schedule_spec=payload.schedule_spec,
        prompt=payload.prompt,
        session_id=payload.session_id,
        label=payload.label,
        deliver_to=payload.deliver_to,
    )
    return job.to_dict()


@app.get("/api/cron/jobs/{job_id}")
async def get_cron_job(job_id: str) -> dict[str, Any]:
    job = app.state.scheduler.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return job


@app.delete("/api/cron/jobs/{job_id}")
async def delete_cron_job(job_id: str) -> dict[str, Any]:
    removed = await app.state.scheduler.remove_job(job_id)
    return {"removed": removed, "job_id": job_id}


class CronJobToggleRequest(BaseModel):
    enabled: bool


@app.patch("/api/cron/jobs/{job_id}")
async def toggle_cron_job(job_id: str, payload: CronJobToggleRequest) -> dict[str, Any]:
    ok = await app.state.scheduler.toggle_job(job_id, enabled=payload.enabled)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return {"job_id": job_id, "enabled": payload.enabled}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def _echo_handler(message: Message) -> str:
    delay: float = message.payload.get("delay", 0.1)
    await asyncio.sleep(delay)
    return f"done: {message.payload}"


async def main() -> None:
    gateway = Gateway(_echo_handler)

    # session-A messages must finish in order (1 → 2 → 3).
    # session-B runs in parallel with session-A.
    sends = [
        gateway.send("session-A", {"step": 1, "delay": 0.20}),
        gateway.send("session-A", {"step": 2, "delay": 0.10}),
        gateway.send("session-A", {"step": 3, "delay": 0.05}),
        gateway.send("session-B", {"step": 1, "delay": 0.15}),
        gateway.send("session-B", {"step": 2, "delay": 0.10}),
    ]

    results = await asyncio.gather(*sends)
    for r in results:
        print(r.model_dump_json(indent=2))

    await gateway.shutdown()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
