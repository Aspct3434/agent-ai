import asyncio
import json
import logging
import os
import shutil
import sqlite3
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import AgentEngine, NormalizedMessage
from checkpointer import StateCheckpointer, initialize_checkpoints_db
from evaluator import SkillDistiller
from tools import ToolManager

logger = logging.getLogger(__name__)

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
PUBLISHED_SITES_DIR = Path(
    os.getenv("PUBLISHED_SITES_DIR", str(_PROJECT_ROOT / "published_sites"))
).expanduser()
PUBLISHED_SITES_DIR.mkdir(parents=True, exist_ok=True)


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
    db_path = Path(os.getenv("SQLITE_DB_PATH", "/app/test.db"))
    checkpoint_db_path = Path(os.getenv("CHECKPOINT_DB_PATH", "/app/checkpoints.db"))
    skills_dir = Path(os.getenv("SKILLS_DIR", "/app/skills"))
    model = os.getenv("AGENT_MODEL", "anthropic/claude-haiku-4-5-20251001")
    # Model tiers: routine work runs on FAST (defaults to AGENT_MODEL); the loop
    # escalates to STRONG on repeated failure and routes coder/auditor sub-agents
    # there. Set STRONG_AGENT_MODEL to a stronger Claude model to enable routing.
    fast_model = os.getenv("FAST_AGENT_MODEL", model)
    strong_model = os.getenv("STRONG_AGENT_MODEL", model)

    _ensure_sqlite_db(db_path)
    await initialize_checkpoints_db(checkpoint_db_path)

    data_dir = Path(os.getenv("AGENT_DATA_DIR", "/app/data"))
    checkpointer = StateCheckpointer(checkpoint_db_path)

    tools = ToolManager()
    await tools.connect_server(
        name="sqlite",
        command=_find_sqlite_server(),
        args=["--db-path", str(db_path)],
    )
    try:
        await tools.connect_filesystem_server(data_dir)
    except RuntimeError:
        # The production Python image does not include Node/npx by default.
        # Filesystem MCP is optional; keep the API online with SQLite + skills.
        pass
    await tools.connect_skills_server(skills_dir)

    distiller = SkillDistiller(skills_dir=skills_dir, model=fast_model)
    await distiller.start()
    memory = _build_memory()
    engine = AgentEngine(
        memory=memory,
        tools=tools,
        model=model,
        fast_model=fast_model,
        strong_model=strong_model,
        distiller=distiller,
        checkpointer=checkpointer,
    )

    async def handler(message: Message) -> str:
        return await engine.process_task(
            NormalizedMessage(
                session_id=message.session_id,
                role="user",
                content=message.payload["text"],
            )
        )

    app.state.tools = tools
    app.state.distiller = distiller
    app.state.engine = engine
    app.state.checkpointer = checkpointer
    app.state.memory = memory
    app.state.gateway = Gateway(handler)

    try:
        yield
    finally:
        await app.state.gateway.shutdown()
        await distiller.shutdown()
        await tools.close()
        if hasattr(memory, "close"):
            try:
                memory.close()
            except Exception as exc:
                logger.warning("Memory close failed: %s", exc)


app = FastAPI(lifespan=lifespan)
app.mount(
    "/sites",
    StaticFiles(directory=str(PUBLISHED_SITES_DIR), html=True),
    name="sites",
)


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
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg = NormalizedMessage(
                session_id=data["session_id"],
                role="user",
                content=data["text"],
            )
            async for event in app.state.engine.stream_task(msg):
                await websocket.send_text(json.dumps(event))
    except WebSocketDisconnect:
        pass


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
