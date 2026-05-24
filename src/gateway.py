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
from collections import defaultdict
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
from checkpointer import StateCheckpointer, initialize_checkpoints_db
from evaluator import SkillDistiller, SkillRegistry
from memory import UserProfileStore
from persona import PersonaLoader
from scheduler import CronScheduler, _validate_schedule
from tools import ToolManager

logger = logging.getLogger(__name__)

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
    await tools.connect_server(
        name="sqlite",
        command=_find_sqlite_server(),
        args=["--db-path", str(db_path)],
    )
    try:
        await tools.connect_filesystem_server(data_dir)
    except RuntimeError:
        pass
    await tools.connect_skills_server(skills_dir)

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

    # -- Distiller --------------------------------------------------------
    distiller = SkillDistiller(skills_dir=skills_dir, model=fast_model)
    await distiller.start()

    # -- Memory -----------------------------------------------------------
    memory = _build_memory()

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

    gw = Gateway(handler)
    app.state.tools = tools
    app.state.distiller = distiller
    app.state.engine = engine
    app.state.checkpointer = checkpointer
    app.state.memory = memory
    app.state.persona = persona
    app.state.profile_store = profile_store
    app.state.skill_registry = skill_registry
    app.state.scheduler = scheduler
    app.state.gateway = gw
    app.state.active_stream_tasks = {}

    # -- Messaging adapters (optional) ------------------------------------
    # Each adapter starts only when its bot-token env var is present so the
    # server comes up cleanly with no messaging credentials configured.

    from adapters.telegram import TelegramAdapter
    from adapters.discord_bot import DiscordAdapter

    async def _adapter_send(session_id: str, text: str) -> str:
        result = await gw.send(session_id, {"text": text})
        if result.error:
            raise RuntimeError(result.error)
        return str(result.output)

    telegram_adapter: TelegramAdapter | None = None
    discord_adapter: DiscordAdapter | None = None

    if tg_token := os.getenv("TELEGRAM_BOT_TOKEN"):
        telegram_adapter = TelegramAdapter(token=tg_token, send_fn=_adapter_send)
        await telegram_adapter.start()
        app.state.telegram_adapter = telegram_adapter

    if dc_token := os.getenv("DISCORD_BOT_TOKEN"):
        discord_adapter = DiscordAdapter(token=dc_token, send_fn=_adapter_send)
        await discord_adapter.start()
        app.state.discord_adapter = discord_adapter

    try:
        yield
    finally:
        # Adapters first — they hold references to the gateway.
        if telegram_adapter is not None:
            await telegram_adapter.shutdown()
        if discord_adapter is not None:
            await discord_adapter.shutdown()
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


# ---------------------------------------------------------------------------
# Cron scheduler API
# ---------------------------------------------------------------------------


class CronJobRequest(BaseModel):
    prompt: str
    schedule_type: str
    schedule_spec: str
    label: str = ""
    session_id: str = "cron"


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
