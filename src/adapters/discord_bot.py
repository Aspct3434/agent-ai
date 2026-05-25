"""Discord Gateway adapter — raw WebSocket, zero extra dependencies.

Connects to the Discord Gateway WebSocket, receives ``MESSAGE_CREATE``
events, and streams each agent turn back to the channel via the injected
*stream_fn*: a live typing indicator, a concise line per tool the agent
runs, then the final answer.

Session scoping
---------------
Each Discord ``channel_id`` maps to ``discord:{channel_id}`` so all
participants in a channel or DM share one conversation session.

Configuration (environment variables)
--------------------------------------
``DISCORD_BOT_TOKEN``
    Bot token from discord.com/developers — required to enable this adapter.
``DISCORD_ALLOWED_USER_IDS``
    Optional comma-separated list of Discord user ID strings.  When set,
    messages from any other user are ignored.  Unset means open to all.

Privileged intent note
----------------------
``MESSAGE_CONTENT`` (intent bit 15) must be enabled in the Discord Developer
Portal under **Bot → Privileged Gateway Intents**.  Without it the agent
can still read DMs, but guild (server) message content will be empty.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from typing import Any

import httpx
from websockets.asyncio.client import connect as ws_connect

from adapters._progress import format_tool_call

logger = logging.getLogger(__name__)

_DISCORD_API: str = "https://discord.com/api/v10"
_GW_QUERY: str = "?v=10&encoding=json"

# Gateway opcodes — https://discord.com/developers/docs/topics/opcodes-and-status-codes
_OP_DISPATCH: int = 0
_OP_HEARTBEAT: int = 1
_OP_IDENTIFY: int = 2
_OP_RECONNECT: int = 7
_OP_INVALID_SESSION: int = 9
_OP_HELLO: int = 10
_OP_HEARTBEAT_ACK: int = 11

# Intent bitmask.  MESSAGE_CONTENT (bit 15) is privileged — see module docstring.
_INTENT_GUILDS: int = 1 << 0
_INTENT_GUILD_MESSAGES: int = 1 << 9
_INTENT_DIRECT_MESSAGES: int = 1 << 12
_INTENT_MESSAGE_CONTENT: int = 1 << 15
_INTENTS: int = (
    _INTENT_GUILDS
    | _INTENT_GUILD_MESSAGES
    | _INTENT_DIRECT_MESSAGES
    | _INTENT_MESSAGE_CONTENT
)

# Discord's hard limit on outgoing message length (characters).
_MSG_LIMIT: int = 2000
# Discord clears a typing indicator after ~10 s, so refresh it sooner.
_TYPING_REFRESH_S: float = 8.0

# A streaming task runner: given (session_id, text) it yields the agent's
# event stream — {"type": "tool_call"|"status"|"text"|"final_answer", ...}.
StreamFn = Callable[[str, str], AsyncIterator[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_text(text: str, limit: int = _MSG_LIMIT) -> list[str]:
    """Split *text* into parts of at most *limit* characters.

    Prefers breaking at the last newline before the limit; falls back to a
    hard cut when no newline is available in the window.
    """
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
    """Parse a comma-separated string into a frozenset of stripped strings.

    Returns ``None`` (open access) when *raw* is empty or whitespace-only.
    """
    stripped = raw.strip()
    if not stripped:
        return None
    return frozenset(part.strip() for part in stripped.split(",") if part.strip())


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class DiscordAdapter:
    """Discord Gateway bot adapter — raw WebSocket, auto-reconnecting.

    Typical usage::

        adapter = DiscordAdapter(token=os.environ["DISCORD_BOT_TOKEN"],
                                 stream_fn=my_stream_fn)
        await adapter.start()
        # ... server runs ...
        await adapter.shutdown()
    """

    def __init__(self, token: str, stream_fn: StreamFn) -> None:
        self._token = token
        self._stream_fn = stream_fn
        self._auth_headers = {"Authorization": f"Bot {token}"}
        self._allowed: frozenset[str] | None = _parse_str_set(
            os.getenv("DISCORD_ALLOWED_USER_IDS", "")
        )
        # Gateway session state — reset on each fresh IDENTIFY.
        self._seq: int | None = None
        self._session_id: str | None = None
        self._hb_acked: bool = True
        self._running: bool = False
        self._task: asyncio.Task[None] | None = None
        self._http: httpx.AsyncClient | None = None
        # Per-channel lock: serialise turns so one channel's history isn't
        # mutated by two concurrent agent runs.
        self._locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the HTTP client and connect to the Discord Gateway."""
        self._http = httpx.AsyncClient(headers=self._auth_headers, timeout=15.0)
        self._running = True
        self._task = asyncio.create_task(self._run(), name="discord:gateway")
        logger.info("Discord adapter started")

    async def shutdown(self) -> None:
        """Disconnect and clean up resources."""
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
        logger.info("Discord adapter stopped")

    # ------------------------------------------------------------------
    # Reconnect loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Outer loop: obtain the Gateway URL and connect, reconnecting on error."""
        while self._running:
            try:
                url = await self._get_gateway_url()
                await self._connect(url)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Discord gateway error: %s — reconnecting in 10 s", exc)
                await asyncio.sleep(10)

    async def _get_gateway_url(self) -> str:
        assert self._http is not None
        resp = await self._http.get(f"{_DISCORD_API}/gateway")
        return str(resp.json()["url"]) + _GW_QUERY

    # ------------------------------------------------------------------
    # WebSocket connection
    # ------------------------------------------------------------------

    async def _connect(self, url: str) -> None:
        """Maintain one WebSocket connection until Discord requests a reconnect."""
        async with ws_connect(url) as ws:
            hb_task: asyncio.Task[None] | None = None
            try:
                async for raw in ws:
                    payload: dict[str, Any] = json.loads(raw)
                    op: int = int(payload.get("op", -1))
                    data: Any = payload.get("d")
                    seq: Any = payload.get("s")
                    event: str | None = payload.get("t")

                    if seq is not None:
                        self._seq = int(seq)

                    if op == _OP_HELLO:
                        interval = int((data or {}).get("heartbeat_interval", 41250)) / 1000
                        first_delay = interval * random.random()
                        hb_task = asyncio.create_task(
                            self._heartbeat_loop(ws, interval, first_delay),
                            name="discord:heartbeat",
                        )
                        await ws.send(self._identify_payload())

                    elif op == _OP_HEARTBEAT_ACK:
                        self._hb_acked = True

                    elif op == _OP_HEARTBEAT:
                        # Server-requested heartbeat.
                        await ws.send(json.dumps({"op": _OP_HEARTBEAT, "d": self._seq}))
                        self._hb_acked = False

                    elif op == _OP_DISPATCH:
                        if event == "READY":
                            info: dict[str, Any] = data or {}
                            self._session_id = str(info.get("session_id", ""))
                            user: dict[str, Any] = info.get("user") or {}
                            logger.info(
                                "Discord READY: logged in as %s#%s",
                                user.get("username", "?"),
                                user.get("discriminator", "0"),
                            )
                        elif event == "MESSAGE_CREATE":
                            asyncio.create_task(  # noqa: RUF006
                                self._handle_message(data or {}),
                                name="discord:message",
                            )

                    elif op in (_OP_RECONNECT, _OP_INVALID_SESSION):
                        if op == _OP_INVALID_SESSION:
                            # Drop resume state; Discord recommends 1-5 s delay.
                            self._session_id = None
                            self._seq = None
                            await asyncio.sleep(5)
                        break  # exits loop → _connect returns → _run reconnects

            finally:
                if hb_task is not None:
                    hb_task.cancel()

    def _identify_payload(self) -> str:
        return json.dumps(
            {
                "op": _OP_IDENTIFY,
                "d": {
                    "token": self._token,
                    "intents": _INTENTS,
                    "properties": {
                        "os": "linux",
                        "browser": "agent-ai",
                        "device": "agent-ai",
                    },
                },
            }
        )

    async def _heartbeat_loop(self, ws: Any, interval: float, first_delay: float) -> None:
        """Send periodic heartbeats; close the connection if the server stops ACKing."""
        await asyncio.sleep(first_delay)
        while True:
            if not self._hb_acked:
                # Previous heartbeat was never acknowledged — zombie connection.
                logger.warning("Discord heartbeat not ACKed; closing for reconnect")
                await ws.close()
                return
            self._hb_acked = False
            await ws.send(json.dumps({"op": _OP_HEARTBEAT, "d": self._seq}))
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_message(self, data: dict[str, Any]) -> None:
        author: dict[str, Any] = data.get("author") or {}

        # Ignore bots (including ourselves).
        if author.get("bot"):
            return

        content: str = (data.get("content") or "").strip()
        if not content:
            return

        user_id: str = str(author.get("id") or "")
        if self._allowed is not None and user_id not in self._allowed:
            return

        channel_id: str = str(data.get("channel_id") or "")

        # Serialise turns within a channel so its history is not mutated by two
        # concurrent agent runs.
        async with self._channel_lock(channel_id):
            await self._run_turn(channel_id, content)

    async def _run_turn(self, channel_id: str, content: str) -> None:
        """Stream the agent's work to the channel: a live typing indicator, a
        concise line per tool the agent runs, then the final answer.
        """
        session_id = f"discord:{channel_id}"
        # Show the typing indicator immediately, then keep it alive.
        await self._trigger_typing(channel_id)
        typing = asyncio.create_task(
            self._typing_loop(channel_id), name="discord:typing"
        )
        final = ""
        try:
            async for event in self._stream_fn(session_id, content):
                etype = event.get("type")
                if etype == "tool_call":
                    line = format_tool_call(
                        str(event.get("tool") or ""), event.get("params") or {}
                    )
                    if line:
                        await self._post_message(channel_id, line)
                elif etype in ("text", "final_answer"):
                    final = str(event.get("content") or final)
        except Exception as exc:
            final = f"⚠️ Error: {exc}"
        finally:
            typing.cancel()
            with suppress(asyncio.CancelledError):
                await typing

        final = final.strip()
        if final:
            for chunk in _chunk_text(final):
                await self._post_message(channel_id, chunk)

    def _channel_lock(self, channel_id: str) -> asyncio.Lock:
        lock = self._locks.get(channel_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[channel_id] = lock
        return lock

    async def _typing_loop(self, channel_id: str) -> None:
        """Refresh the channel typing indicator until cancelled.

        The first indicator is sent by the caller; this loop only re-sends it
        before Discord's ~10 s timeout clears it.
        """
        while True:
            await asyncio.sleep(_TYPING_REFRESH_S)
            await self._trigger_typing(channel_id)

    async def _trigger_typing(self, channel_id: str) -> None:
        assert self._http is not None
        try:
            await self._http.post(f"{_DISCORD_API}/channels/{channel_id}/typing")
        except Exception as exc:
            logger.debug("Discord typing indicator failed (channel=%s): %s", channel_id, exc)

    async def _post_message(self, channel_id: str, text: str) -> None:
        assert self._http is not None
        try:
            await self._http.post(
                f"{_DISCORD_API}/channels/{channel_id}/messages",
                json={"content": text},
            )
        except Exception as exc:
            logger.warning("Discord post failed (channel=%s): %s", channel_id, exc)
