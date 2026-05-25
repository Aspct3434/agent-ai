"""Slack adapter — Socket Mode, zero extra dependencies.

Connects to Slack via Socket Mode (a WebSocket, no public URL needed),
receives message events, and streams each agent turn back to the channel:
a concise line per tool the agent runs, then the final answer. (Slack has
no public bot "typing" indicator, so the per-tool progress lines stand in.)

Session scoping
---------------
Each Slack channel maps to ``slack:{channel}`` so conversation history is
scoped per channel/DM.

Configuration (environment variables)
--------------------------------------
``SLACK_BOT_TOKEN``
    Bot token (``xoxb-…``) with ``chat:write`` — required.
``SLACK_APP_TOKEN``
    App-level token (``xapp-…``) with ``connections:write`` — required for
    Socket Mode.
``SLACK_ALLOWED_USERS``
    Optional comma-separated Slack user-id allowlist. Unset = open to all.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from websockets.asyncio.client import connect as ws_connect

from adapters._progress import format_tool_call

logger = logging.getLogger(__name__)

_SLACK_API: str = "https://slack.com/api"
# Slack's text field accepts up to ~40k chars, but messages read better well
# under that; chunk conservatively.
_MSG_LIMIT: int = 3500

StreamFn = Callable[[str, str], AsyncIterator[dict[str, Any]]]
ResetFn = Callable[[str], bool]

_HELP = (
    "Commands:\n"
    "`/new` (or `/reset`) — start a fresh conversation\n"
    "`/stop` — interrupt the current task\n"
    "`/help` — show this message\n\n"
    "Otherwise just send a message and I'll work on it."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_text(text: str, limit: int = _MSG_LIMIT) -> list[str]:
    """Split *text* into parts of at most *limit* characters (newline-aware)."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return parts


def _parse_str_set(raw: str) -> frozenset[str] | None:
    """Parse comma-separated *raw* into a frozenset, or None if empty."""
    stripped = raw.strip()
    if not stripped:
        return None
    return frozenset(part.strip() for part in stripped.split(",") if part.strip())


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SlackAdapter:
    """Slack Socket Mode bot adapter — auto-reconnecting."""

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        stream_fn: StreamFn,
        reset_fn: ResetFn,
    ) -> None:
        self._bot_token = bot_token
        self._app_token = app_token
        self._stream_fn = stream_fn
        self._reset_fn = reset_fn
        self._bot_headers = {"Authorization": f"Bearer {bot_token}"}
        self._app_headers = {"Authorization": f"Bearer {app_token}"}
        self._allowed: frozenset[str] | None = _parse_str_set(
            os.getenv("SLACK_ALLOWED_USERS", "")
        )
        self._running: bool = False
        self._task: asyncio.Task[None] | None = None
        self._http: httpx.AsyncClient | None = None
        self._locks: dict[str, asyncio.Lock] = {}
        self._turn_tasks: dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._http = httpx.AsyncClient(timeout=30.0)
        self._running = True
        self._task = asyncio.create_task(self._run(), name="slack:socket")
        logger.info("Slack adapter started")

    async def shutdown(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        logger.info("Slack adapter stopped")

    # ------------------------------------------------------------------
    # Socket Mode connection
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        while self._running:
            try:
                url = await self._open_connection()
                await self._connect(url)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Slack Socket Mode error: %s — reconnecting in 10 s", exc)
                await asyncio.sleep(10)

    async def _open_connection(self) -> str:
        """Request a Socket Mode WebSocket URL using the app-level token."""
        assert self._http is not None
        resp = await self._http.post(
            f"{_SLACK_API}/apps.connections.open", headers=self._app_headers
        )
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"apps.connections.open failed: {data}")
        return str(data["url"])

    async def _connect(self, url: str) -> None:
        async with ws_connect(url) as ws:
            async for raw in ws:
                envelope: dict[str, Any] = json.loads(raw)
                etype = envelope.get("type")

                if etype == "hello":
                    logger.info("Slack Socket Mode connected")
                    continue
                if etype == "disconnect":
                    break  # Slack asked us to reconnect

                # Acknowledge every enveloped event immediately (Slack expects
                # an ack within 3 s or it redelivers).
                envelope_id = envelope.get("envelope_id")
                if envelope_id is not None:
                    await ws.send(json.dumps({"envelope_id": envelope_id}))

                if etype == "events_api":
                    event = (envelope.get("payload") or {}).get("event") or {}
                    asyncio.create_task(  # noqa: RUF006
                        self._handle_event(event), name="slack:event"
                    )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _handle_event(self, event: dict[str, Any]) -> None:
        if event.get("type") != "message":
            return
        # Ignore the bot's own messages, edits, joins, and other subtypes.
        if event.get("bot_id") or event.get("subtype"):
            return

        text: str = (event.get("text") or "").strip()
        if not text:
            return

        user_id: str = str(event.get("user") or "")
        if self._allowed is not None and user_id not in self._allowed:
            return

        channel: str = str(event.get("channel") or "")

        if text.startswith("/"):
            await self._handle_command(channel, text)
            return

        async with self._channel_lock(channel):
            task = asyncio.create_task(self._run_turn(channel, text), name="slack:turn")
            self._turn_tasks[channel] = task
            try:
                await task
            except asyncio.CancelledError:
                await self._post_message(channel, "🛑 Stopped.")
            finally:
                self._turn_tasks.pop(channel, None)

    async def _run_turn(self, channel: str, text: str) -> None:
        session_id = f"slack:{channel}"
        final = ""
        try:
            async for event in self._stream_fn(session_id, text):
                etype = event.get("type")
                if etype == "tool_call":
                    line = format_tool_call(
                        str(event.get("tool") or ""), event.get("params") or {}
                    )
                    if line:
                        await self._post_message(channel, line)
                elif etype in ("text", "final_answer"):
                    final = str(event.get("content") or final)
        except Exception as exc:
            final = f"⚠️ Error: {exc}"

        final = final.strip()
        if final:
            for chunk in _chunk_text(final):
                await self._post_message(channel, chunk)

    async def _handle_command(self, channel: str, text: str) -> None:
        cmd = text.split(maxsplit=1)[0].lower()
        if cmd in ("/new", "/reset"):
            self._reset_fn(f"slack:{channel}")
            await self._post_message(channel, "🧹 Started a new conversation.")
        elif cmd == "/stop":
            task = self._turn_tasks.get(channel)
            if task is not None and not task.done():
                task.cancel()
            else:
                await self._post_message(channel, "Nothing is running.")
        elif cmd == "/help":
            await self._post_message(channel, _HELP)
        else:
            await self._post_message(channel, f"Unknown command {cmd}. Type /help.")

    def _channel_lock(self, channel: str) -> asyncio.Lock:
        lock = self._locks.get(channel)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[channel] = lock
        return lock

    async def deliver(self, channel: str, text: str) -> None:
        """Post a (possibly long) message to a channel — used for scheduled delivery."""
        text = str(text).strip()
        if not text:
            return
        for chunk in _chunk_text(text):
            await self._post_message(channel, chunk)

    async def _post_message(self, channel: str, text: str) -> None:
        assert self._http is not None
        try:
            await self._http.post(
                f"{_SLACK_API}/chat.postMessage",
                headers=self._bot_headers,
                json={"channel": channel, "text": text},
            )
        except Exception as exc:
            logger.warning("Slack postMessage failed (channel=%s): %s", channel, exc)
