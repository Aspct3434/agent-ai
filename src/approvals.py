"""Human-in-the-loop approval gate for risky agent commands.

Complements the hard block-list in ``tools.py``: instead of silently allowing
(or hard-refusing) a command, the agent can be made to *pause and ask* before
running it. Controlled by ``AGENT_REQUIRE_APPROVAL``:

  off    — never ask (default; behaviour unchanged)
  risky  — ask only for commands matching the risk patterns below
  all    — ask before every terminal/background command

A pending request is exposed via the gateway (``/api/approvals``) and resolved
by the user (dashboard Approve/Deny). The agent's tool execution awaits the
decision, with a timeout that defaults to deny.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Commands that warrant a confirmation in "risky" mode.
_RISKY_PATTERNS = [
    re.compile(p)
    for p in (
        r"\bsudo\b",
        r"\brm\s+-[a-z]*[rf]",
        r"\bmkfs\b",
        r"\bdd\b",
        r"\bshred\b",
        r"\bchmod\s+-?R?\s*777",
        r"\bchown\b",
        r"\bcurl\b.*\|\s*(sudo\s+)?(ba)?sh",
        r"\bwget\b.*\|\s*(sudo\s+)?(ba)?sh",
        r"\bgit\s+push\b.*--force",
        r"\bdocker\s+(rm|rmi|system\s+prune|volume\s+rm)",
        r"\b(apt|apt-get|yum|dnf|brew)\s+(remove|purge|uninstall)",
        r"\bnpm\s+publish\b",
        r"\bsystemctl\b",
        r"\bkillall\b",
        r">\s*/dev/sd",
        r":\(\)\s*\{",  # fork bomb
    )
]


class ApprovalGate:
    """Tracks pending command approvals and lets the agent await a decision."""

    def __init__(self, mode: str | None = None, timeout: float | None = None) -> None:
        self._mode = (mode or os.getenv("AGENT_REQUIRE_APPROVAL") or "off").lower()
        self._timeout = timeout if timeout is not None else float(
            os.getenv("AGENT_APPROVAL_TIMEOUT", "120")
        )
        self._pending: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, asyncio.Future[bool]] = {}

    @property
    def mode(self) -> str:
        return self._mode

    def requires_approval(self, command: str) -> bool:
        if self._mode == "all":
            return True
        if self._mode == "risky":
            return any(p.search(command or "") for p in _RISKY_PATTERNS)
        return False

    async def request(self, command: str, session_id: str = "") -> tuple[bool, str]:
        """Register a pending approval and block until resolved or timed out."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        rid = uuid.uuid4().hex[:12]
        self._pending[rid] = {
            "id": rid,
            "command": (command or "")[:500],
            "session_id": session_id,
            "created_at": time.time(),
        }
        self._futures[rid] = future
        try:
            approved = await asyncio.wait_for(future, timeout=self._timeout)
            return approved, "approved" if approved else "denied by user"
        except TimeoutError:
            return False, f"approval timed out after {int(self._timeout)}s"
        finally:
            self._pending.pop(rid, None)
            self._futures.pop(rid, None)

    def resolve(self, request_id: str, approved: bool) -> bool:
        """Resolve a pending request (called from the gateway). Same event loop."""
        future = self._futures.get(request_id)
        if future is None or future.done():
            return False
        future.set_result(approved)
        return True

    def pending(self) -> list[dict[str, Any]]:
        return sorted(self._pending.values(), key=lambda r: r["created_at"])
