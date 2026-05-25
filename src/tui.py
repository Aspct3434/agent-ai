"""Rich terminal UI for agent-ai.

A streaming terminal client that talks to the gateway's ``/ws/stream``
WebSocket. Unlike the bare ``cli_chat.py`` request/response loop, it shows the
agent's work live — a dim line per tool call (same OpenClaw-style feed as the
messaging adapters) — then renders the final answer as Markdown.

Run a gateway first (``uvicorn gateway:app --app-dir src``), then::

    python -m tui                       # connects to ws://127.0.0.1:8000
    python -m tui --url ws://host:9000/ws/stream

In-session commands: ``/new`` (or ``/reset``) starts a fresh conversation,
``/help`` lists commands, ``/quit`` (or ``/exit``) leaves.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from websockets.asyncio.client import connect as ws_connect

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters._progress import format_tool_call

_DEFAULT_URL = "ws://127.0.0.1:8000/ws/stream"
_BANNER = "[bold cyan]agent-ai[/bold cyan] — type a task, /help for commands, /quit to exit."
_HELP = (
    "[bold]Commands[/bold]\n"
    "  /new, /reset   start a fresh conversation\n"
    "  /help          show this message\n"
    "  /quit, /exit   leave"
)


def render_event(event: dict[str, Any]) -> str | None:
    """Map a stream event to a one-line progress string (or None to skip).

    The final answer (``text`` / ``final_answer``) and streamed ``token``s are
    handled separately by the turn loop, so they return None here.
    """
    etype = event.get("type")
    if etype == "tool_call":
        return format_tool_call(str(event.get("tool") or ""), event.get("params") or {})
    if etype == "status":
        msg = str(event.get("message") or "").strip()
        return msg if msg and msg != "Thinking..." else None
    return None


class AgentTUI:
    def __init__(self, ws_url: str, console: Console | None = None) -> None:
        self._ws_url = ws_url
        self._console = console or Console()
        self._session_id = self._fresh_session()

    @staticmethod
    def _fresh_session() -> str:
        return f"tui:{uuid.uuid4().hex[:8]}"

    def _handle_command(self, line: str) -> bool:
        """Handle a slash command. Returns False if the session should end."""
        cmd = line.split(maxsplit=1)[0].lower()
        if cmd in ("/quit", "/exit"):
            return False
        if cmd in ("/new", "/reset"):
            self._session_id = self._fresh_session()
            self._console.print("[dim]Started a new conversation.[/dim]")
        elif cmd == "/help":
            self._console.print(_HELP)
        else:
            self._console.print(f"[yellow]Unknown command {cmd}. Type /help.[/yellow]")
        return True

    async def run(self) -> None:
        try:
            async with ws_connect(self._ws_url) as ws:
                self._console.print(_BANNER)
                while True:
                    line = await asyncio.to_thread(self._read_line)
                    if line is None:  # EOF (Ctrl-D)
                        break
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("/"):
                        if not self._handle_command(line):
                            break
                        continue
                    await self._run_turn(ws, line)
        except (ConnectionError, OSError) as exc:
            self._console.print(f"[red]Could not reach the gateway at {self._ws_url}: {exc}[/red]")
        except KeyboardInterrupt:
            pass
        self._console.print("[dim]Bye.[/dim]")

    def _read_line(self) -> str | None:
        try:
            return input("\nyou > ")
        except EOFError:
            return None

    async def _run_turn(self, ws: Any, text: str) -> None:
        await ws.send(json.dumps({"session_id": self._session_id, "text": text}))
        tokens: list[str] = []
        try:
            async for raw in ws:
                event: dict[str, Any] = json.loads(raw)
                etype = event.get("type")
                if etype == "token":
                    tokens.append(str(event.get("content") or ""))
                    continue
                if etype in ("text", "final_answer"):
                    answer = str(event.get("content") or "".join(tokens)).strip()
                    if answer:
                        self._console.print(Panel(Markdown(answer), border_style="cyan"))
                    return
                line = render_event(event)
                if line:
                    self._console.print(f"[dim]▸ {line}[/dim]")
        except KeyboardInterrupt:
            with suppress(Exception):
                await ws.send(json.dumps({"type": "cancel", "session_id": self._session_id}))
            self._console.print("[yellow]Interrupted.[/yellow]")


def main() -> None:
    parser = argparse.ArgumentParser(description="agent-ai rich terminal UI")
    parser.add_argument(
        "--url",
        default=os.getenv("AGENT_TUI_WS_URL", _DEFAULT_URL),
        help=f"gateway WebSocket URL (default: {_DEFAULT_URL})",
    )
    args = parser.parse_args()
    try:
        asyncio.run(AgentTUI(args.url).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
