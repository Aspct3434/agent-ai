"""Telegram Bot API adapter — long-polling, zero extra dependencies.

One ``TelegramAdapter`` instance runs per process.  It long-polls
``getUpdates`` on the Telegram Bot API and routes each incoming message
to the Agent Gateway via the injected *send_fn*.

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
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Telegram long-poll window in seconds.  Max allowed by Telegram is 60.
_POLL_TIMEOUT_S: int = 30
# HTTP client timeout — slightly above the poll window so long-polls complete.
_HTTP_TIMEOUT_S: float = float(_POLL_TIMEOUT_S + 5)
# Telegram's hard limit on outgoing message length (characters).
_MSG_LIMIT: int = 4096

SendFn = Callable[[str, str], Awaitable[str]]


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
                                  send_fn=my_send_fn)
        await adapter.start()
        # ... server runs ...
        await adapter.shutdown()
    """

    def __init__(self, token: str, send_fn: SendFn) -> None:
        self._token = token
        self._base = f"https://api.telegram.org/bot{token}"
        self._send_fn = send_fn
        self._allowed: frozenset[int] | None = _parse_int_set(
            os.getenv("TELEGRAM_ALLOWED_IDS", "")
        )
        self._offset: int = 0
        self._running: bool = False
        self._task: asyncio.Task[None] | None = None
        self._http: httpx.AsyncClient | None = None

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

        # /start command — send a welcome message and stop processing.
        if text == "/start":
            await self._send_message(chat_id, "Hello! Send me a task and I'll get it done.")
            return

        # Ignore other bot commands and empty/whitespace messages.
        if text.startswith("/") or not text:
            return

        # Allowlist check.
        if self._allowed is not None and user_id not in self._allowed:
            await self._send_message(chat_id, "Access denied.")
            return

        await self._send_message(chat_id, "⏳ Working on it…")

        session_id = f"tg:{chat_id}"
        try:
            reply: str = await self._send_fn(session_id, text)
        except Exception as exc:
            reply = f"⚠️ Error: {exc}"

        reply_text = str(reply).strip()
        if reply_text:
            for chunk in _chunk_text(reply_text):
                await self._send_message(chat_id, chunk)

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
