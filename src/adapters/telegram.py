"""Telegram Bot API adapter — long-polling, zero extra dependencies.

One ``TelegramAdapter`` instance runs per process.  It long-polls
``getUpdates`` on the Telegram Bot API and streams each agent turn back to
the chat via the injected *stream_fn*: a live "typing…" status, a concise
line per tool the agent runs, then the final answer.

Session scoping
---------------
Each Telegram ``chat_id`` maps to the session key ``tg:{chat_id}`` so
conversation history is scoped per chat (group chats and DMs each get
their own session).

Configuration (environment variables)
--------------------------------------
``TELEGRAM_BOT_TOKEN``
    Bot token from @BotFather — required to enable this adapter.
``TELEGRAM_ALLOWED_IDS``
    Optional comma-separated list of integer ``chat_id`` / ``user_id``
    values.  When set, messages from any other ID are silently rejected
    with an "Access denied" reply.  Unset means open to all.
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from typing import Any

import httpx

from adapters._progress import format_tool_call

logger = logging.getLogger(__name__)

# Telegram long-poll window in seconds.  Max allowed by Telegram is 60.
_POLL_TIMEOUT_S: int = 30
# HTTP client timeout — slightly above the poll window so long-polls complete.
_HTTP_TIMEOUT_S: float = float(_POLL_TIMEOUT_S + 5)
# Telegram's hard limit on outgoing message length (characters).
_MSG_LIMIT: int = 4096
# Telegram clears a chat action after ~5 s, so refresh "typing…" faster than that.
_TYPING_REFRESH_S: float = 4.0

# A streaming task runner: given (session_id, text) it yields the agent's
# event stream — {"type": "tool_call"|"status"|"text"|"final_answer", ...}.
StreamFn = Callable[[str, str], AsyncIterator[dict[str, Any]]]
# Clears a session's history; returns True if a session existed.
ResetFn = Callable[[str], bool]

_WELCOME = "Hello! Send me a task and I'll get it done. Type /help for commands."
_HELP = (
    "Commands:\n"
    "/new (or /reset) — start a fresh conversation\n"
    "/stop — interrupt the current task\n"
    "/help — show this message\n\n"
    "Otherwise just send a message and I'll work on it."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_text(text: str, limit: int = _MSG_LIMIT) -> list[str]:
    """Split *text* into parts of at most *limit* characters.

    Prefers breaking at the last newline before the limit so paragraph
    boundaries are preserved.  Falls back to a hard cut when no newline
    exists in the window.
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


def _parse_int_set(raw: str) -> frozenset[int] | None:
    """Parse a comma-separated string of integers into a frozenset.

    Returns ``None`` (open access) when *raw* is empty or whitespace-only.
    Raises ``ValueError`` if any token is not a valid integer.
    """
    stripped = raw.strip()
    if not stripped:
        return None
    return frozenset(int(part.strip()) for part in stripped.split(",") if part.strip())


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class TelegramAdapter:
    """Long-poll the Telegram Bot API and forward messages to the Gateway.

    Typical usage::

        adapter = TelegramAdapter(token=os.environ["TELEGRAM_BOT_TOKEN"],
                                  stream_fn=my_stream_fn)
        await adapter.start()
        # ... server runs ...
        await adapter.shutdown()
    """

    def __init__(self, token: str, stream_fn: StreamFn, reset_fn: ResetFn) -> None:
        self._token = token
        self._base = f"https://api.telegram.org/bot{token}"
        self._stream_fn = stream_fn
        self._reset_fn = reset_fn
        self._allowed: frozenset[int] | None = _parse_int_set(
            os.getenv("TELEGRAM_ALLOWED_IDS", "")
        )
        self._offset: int = 0
        self._running: bool = False
        self._task: asyncio.Task[None] | None = None
        self._http: httpx.AsyncClient | None = None
        # Per-chat lock: serialise turns so one chat's history isn't mutated
        # by two concurrent agent runs.
        self._locks: dict[int, asyncio.Lock] = {}
        # In-flight turn per chat, so /stop can interrupt it out-of-band.
        self._turn_tasks: dict[int, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the HTTP client, validate the bot, and begin long-polling."""
        self._http = httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S)
        self._running = True
        await self._prepare()
        self._task = asyncio.create_task(self._poll_loop(), name="telegram:poll")
        logger.info("Telegram adapter started")

    async def _prepare(self) -> None:
        """Validate the token and clear any webhook that would block polling.

        Two silent-failure modes are surfaced/fixed here:
        - A bad or empty token makes every call fail; ``getMe`` turns that into
          one clear startup error instead of an endless warning loop.
        - A previously registered webhook makes ``getUpdates`` fail with HTTP
          409, so the bot would receive nothing; ``deleteWebhook`` restores
          long-polling. Pending updates are kept (``drop_pending_updates`` off).

        Failures here are non-fatal: the poll loop still starts and retries.
        """
        assert self._http is not None
        try:
            resp = await self._http.get(f"{self._base}/getMe", timeout=15.0)
            data = resp.json()
            if data.get("ok"):
                logger.info(
                    "Telegram bot authenticated as @%s",
                    data.get("result", {}).get("username", "?"),
                )
            else:
                logger.error("Telegram getMe failed — check TELEGRAM_BOT_TOKEN: %s", data)
        except Exception as exc:
            logger.warning("Telegram getMe check failed: %s", exc)
        await self._delete_webhook()

    async def _delete_webhook(self) -> None:
        """Remove any registered webhook so long-polling can receive updates."""
        assert self._http is not None
        try:
            await self._http.post(
                f"{self._base}/deleteWebhook",
                json={"drop_pending_updates": False},
                timeout=15.0,
            )
        except Exception as exc:
            logger.warning("Telegram deleteWebhook failed: %s", exc)

    async def shutdown(self) -> None:
        """Stop the polling loop and close the HTTP client gracefully."""
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
        logger.info("Telegram adapter stopped")

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        assert self._http is not None
        while self._running:
            try:
                updates = await self._get_updates()
                for update in updates:
                    self._offset = update["update_id"] + 1
                    asyncio.create_task(  # noqa: RUF006
                        self._handle_update(update),
                        name=f"telegram:update:{update['update_id']}",
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Telegram poll error: %s — retrying in 5 s", exc)
                await asyncio.sleep(5)

    async def _get_updates(self) -> list[dict[str, Any]]:
        """Long-poll /getUpdates; returns the result list (may be empty)."""
        assert self._http is not None
        resp = await self._http.post(
            f"{self._base}/getUpdates",
            json={
                "offset": self._offset,
                "timeout": _POLL_TIMEOUT_S,
                "allowed_updates": ["message"],
            },
            timeout=_HTTP_TIMEOUT_S,
        )
        data: dict[str, Any] = resp.json()
        if not data.get("ok"):
            # HTTP 409: a webhook was registered after startup — clear it so
            # polling resumes on the next iteration.
            if data.get("error_code") == 409:
                logger.warning("Telegram getUpdates conflict (webhook active); deleting it")
                await self._delete_webhook()
            else:
                logger.warning("Telegram getUpdates error: %s", data)
            return []
        return list(data.get("result") or [])

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_update(self, update: dict[str, Any]) -> None:
        message: dict[str, Any] = update.get("message") or {}
        if not message:
            return

        text: str = (message.get("text") or "").strip()
        chat_id: int = int(message["chat"]["id"])
        from_user: dict[str, Any] = message.get("from") or {}
        user_id: int = int(from_user.get("id") or 0)

        if not text:
            return

        # /start is a friendly greeting available to anyone who can reach the bot.
        if text == "/start":
            await self._send_message(chat_id, _WELCOME)
            return

        # Allowlist gate for everything that follows.
        if self._allowed is not None and user_id not in self._allowed:
            await self._send_message(chat_id, "Access denied.")
            return

        # Slash commands are handled out-of-band (so /stop can interrupt a turn).
        if text.startswith("/"):
            await self._handle_command(chat_id, text)
            return

        # Normal message → run a streamed turn that /stop can cancel.
        async with self._chat_lock(chat_id):
            task = asyncio.create_task(
                self._run_turn(chat_id, text), name=f"tg:turn:{chat_id}"
            )
            self._turn_tasks[chat_id] = task
            try:
                await task
            except asyncio.CancelledError:
                await self._send_message(chat_id, "🛑 Stopped.")
            finally:
                self._turn_tasks.pop(chat_id, None)

    async def _handle_command(self, chat_id: int, text: str) -> None:
        """Handle an in-chat slash command (/new, /reset, /stop, /help)."""
        cmd = text.split(maxsplit=1)[0].lower()
        if cmd in ("/new", "/reset"):
            self._reset_fn(f"tg:{chat_id}")
            await self._send_message(chat_id, "🧹 Started a new conversation.")
        elif cmd == "/stop":
            task = self._turn_tasks.get(chat_id)
            if task is not None and not task.done():
                task.cancel()
            else:
                await self._send_message(chat_id, "Nothing is running.")
        elif cmd == "/help":
            await self._send_message(chat_id, _HELP)
        else:
            await self._send_message(chat_id, f"Unknown command {cmd}. Type /help.")

    async def _run_turn(self, chat_id: int, text: str) -> None:
        """Stream the agent's work to the chat: a live 'typing…' status, a
        concise line per tool the agent runs, then the final answer.
        """
        session_id = f"tg:{chat_id}"
        # Show "typing…" in the chat immediately (instead of an "On it"
        # message), then keep it alive for the whole turn.
        await self._send_chat_action(chat_id, "typing")
        typing = asyncio.create_task(
            self._typing_loop(chat_id), name=f"tg:typing:{chat_id}"
        )
        final = ""
        try:
            async for event in self._stream_fn(session_id, text):
                etype = event.get("type")
                if etype == "tool_call":
                    line = format_tool_call(
                        str(event.get("tool") or ""), event.get("params") or {}
                    )
                    if line:
                        await self._send_message(chat_id, line)
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
                await self._send_message(chat_id, chunk)

    def _chat_lock(self, chat_id: int) -> asyncio.Lock:
        lock = self._locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[chat_id] = lock
        return lock

    async def _typing_loop(self, chat_id: int) -> None:
        """Refresh the 'typing…' chat action until cancelled.

        The first action is sent by the caller; this loop only re-sends it
        before Telegram's ~5 s timeout clears it.
        """
        while True:
            await asyncio.sleep(_TYPING_REFRESH_S)
            await self._send_chat_action(chat_id, "typing")

    async def _send_chat_action(self, chat_id: int, action: str) -> None:
        """Show a transient chat action (e.g. 'typing…') in the chat header."""
        assert self._http is not None
        try:
            await self._http.post(
                f"{self._base}/sendChatAction",
                json={"chat_id": chat_id, "action": action},
                timeout=15.0,
            )
        except Exception as exc:
            logger.debug("Telegram sendChatAction failed (chat=%s): %s", chat_id, exc)

    async def _send_message(self, chat_id: int, text: str) -> None:
        """POST a message to a Telegram chat; logs warnings on failure."""
        assert self._http is not None
        try:
            await self._http.post(
                f"{self._base}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=15.0,
            )
        except Exception as exc:
            logger.warning("Telegram sendMessage failed (chat=%s): %s", chat_id, exc)
