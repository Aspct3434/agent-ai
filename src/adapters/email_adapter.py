"""Email channel adapter — poll IMAP for new mail, run the agent, reply via SMTP.

Email isn't a live medium, so this adapter collects the agent's final answer
(prefixed with a compact list of the actions it took) and sends it back as a
reply. Uses only the stdlib (imaplib/smtplib) run in worker threads.

Session scoping: each sender address maps to ``email:{address}`` so a person's
thread of mail shares one conversation.

Configuration (environment variables)
--------------------------------------
``EMAIL_ADDRESS`` / ``EMAIL_PASSWORD``   mailbox login (required to enable)
``EMAIL_IMAP_HOST`` / ``EMAIL_IMAP_PORT``   inbox (default port 993, SSL)
``EMAIL_SMTP_HOST`` / ``EMAIL_SMTP_PORT``   outbound (default port 465, SSL)
``EMAIL_ALLOWED_SENDERS``   optional comma-separated allowlist of From addresses
``EMAIL_POLL_INTERVAL``   seconds between inbox polls (default 20)
"""
from __future__ import annotations

import asyncio
import email
import imaplib
import logging
import os
import smtplib
from collections.abc import AsyncIterator, Callable
from email.message import EmailMessage
from email.utils import parseaddr
from typing import Any

from adapters._progress import format_tool_call

logger = logging.getLogger(__name__)

StreamFn = Callable[[str, str], AsyncIterator[dict[str, Any]]]
ResetFn = Callable[[str], bool]


def _parse_allowlist(raw: str) -> frozenset[str] | None:
    stripped = raw.strip()
    if not stripped:
        return None
    return frozenset(p.strip().lower() for p in stripped.split(",") if p.strip())


def _extract_body(msg: email.message.Message) -> str:
    """Return the plain-text body of an email message."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and "attachment" not in str(
                part.get("Content-Disposition", "")
            ):
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, "replace")
        return ""
    payload = msg.get_payload(decode=True)
    if isinstance(payload, bytes):
        return payload.decode(msg.get_content_charset() or "utf-8", "replace")
    return str(msg.get_payload() or "")


def _strip_quoted(body: str) -> str:
    """Drop quoted reply history so the agent sees just the new message."""
    lines: list[str] = []
    for line in body.splitlines():
        if line.startswith(">") or (line.strip().startswith("On ") and line.rstrip().endswith("wrote:")):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _build_reply(to_addr: str, from_addr: str, subject: str, body: str) -> EmailMessage:
    reply = EmailMessage()
    reply["From"] = from_addr
    reply["To"] = to_addr
    reply["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    reply.set_content(body)
    return reply


class EmailAdapter:
    def __init__(
        self,
        *,
        imap_host: str,
        smtp_host: str,
        address: str,
        password: str,
        stream_fn: StreamFn,
        reset_fn: ResetFn,
        imap_port: int = 993,
        smtp_port: int = 465,
        poll_interval: float = 20.0,
    ) -> None:
        self._imap_host = imap_host
        self._imap_port = imap_port
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._address = address
        self._password = password
        self._stream_fn = stream_fn
        self._reset_fn = reset_fn
        self._poll_interval = poll_interval
        self._allowed = _parse_allowlist(os.getenv("EMAIL_ALLOWED_SENDERS", ""))
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name="email:poll")
        logger.info("Email adapter started (%s)", self._address)

    async def shutdown(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Email adapter stopped")

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                messages = await asyncio.to_thread(self._fetch_unseen)
                for msg in messages:
                    await self._handle_email(msg)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Email poll error: %s — retrying", exc)
            await asyncio.sleep(self._poll_interval)

    def _fetch_unseen(self) -> list[dict[str, str]]:
        """Fetch + mark-seen unread mail. Runs in a worker thread (blocking)."""
        out: list[dict[str, str]] = []
        with imaplib.IMAP4_SSL(self._imap_host, self._imap_port) as imap:
            imap.login(self._address, self._password)
            imap.select("INBOX")
            _typ, data = imap.search(None, "UNSEEN")
            ids = (data[0].split() if data and data[0] else [])
            for num in ids:
                _t, fetched = imap.fetch(num, "(RFC822)")
                if not fetched or not isinstance(fetched[0], tuple):
                    continue
                msg = email.message_from_bytes(fetched[0][1])
                sender = parseaddr(msg.get("From", ""))[1]
                out.append({
                    "from": sender,
                    "subject": msg.get("Subject", "") or "(no subject)",
                    "body": _strip_quoted(_extract_body(msg)),
                })
        return out

    async def deliver(self, to_addr: str, text: str) -> None:
        """Proactively email someone (used by cron jobs: deliver_to=email:addr)."""
        if not to_addr or not text:
            return
        await asyncio.to_thread(self._send, to_addr, "Update from your agent", text)

    async def _handle_email(self, msg: dict[str, str]) -> None:
        sender = (msg.get("from") or "").lower()
        body = (msg.get("body") or "").strip()
        if not sender or not body:
            return
        if self._allowed is not None and sender not in self._allowed:
            logger.info("Email from %s ignored (not in allowlist)", sender)
            return

        session_id = f"email:{sender}"
        steps: list[str] = []
        final = ""
        try:
            async for event in self._stream_fn(session_id, body):
                etype = event.get("type")
                if etype == "tool_call":
                    line = format_tool_call(str(event.get("tool") or ""), event.get("params") or {})
                    if line:
                        steps.append(line)
                elif etype in ("text", "final_answer"):
                    final = str(event.get("content") or final)
        except Exception as exc:
            final = f"Sorry — something went wrong: {exc}"

        reply_body = final.strip() or "(no response)"
        if steps:
            reply_body += "\n\n—\nWhat I did:\n" + "\n".join(f"• {s}" for s in steps)
        await asyncio.to_thread(self._send, sender, msg.get("subject", ""), reply_body)

    def _send(self, to_addr: str, subject: str, body: str) -> None:
        try:
            reply = _build_reply(to_addr, self._address, subject, body)
            with smtplib.SMTP_SSL(self._smtp_host, self._smtp_port) as smtp:
                smtp.login(self._address, self._password)
                smtp.send_message(reply)
        except Exception as exc:
            logger.warning("Email send to %s failed: %s", to_addr, exc)
