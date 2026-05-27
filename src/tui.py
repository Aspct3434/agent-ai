"""Interactive Rich terminal UI for Distill.

The TUI is a WebSocket client for the gateway's ``/ws/stream`` endpoint. It
keeps the implementation dependency-light (Rich + websockets, both already in
the project) while giving the CLI a proper control-room feel: immediate first
paint, live status, transcript, tool activity, detail toggles, themes, and
small in-session overlays.

Run a gateway first (``uvicorn gateway:app --app-dir src``), then::

    python src/tui.py
    python src/tui.py --url ws://host:9000/ws/stream

The npm wrapper also launches this surface with ``distill``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text
from websockets.asyncio.client import connect as ws_connect

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters._progress import format_tool_call

_DEFAULT_URL = "ws://127.0.0.1:8000/ws/stream"
_DETAIL_MODES = ("expanded", "collapsed", "hidden")
DetailMode = Literal["expanded", "collapsed", "hidden"]


@dataclass(frozen=True)
class Theme:
    name: str
    primary: str
    accent: str
    good: str
    warn: str
    danger: str
    muted: str
    panel: str


_THEMES: dict[str, Theme] = {
    "distill": Theme(
        name="distill",
        primary="bright_magenta",
        accent="cyan",
        good="green",
        warn="yellow",
        danger="red",
        muted="bright_black",
        panel="magenta",
    ),
    "mono": Theme(
        name="mono",
        primary="white",
        accent="bright_white",
        good="white",
        warn="bright_white",
        danger="white",
        muted="bright_black",
        panel="white",
    ),
    "ember": Theme(
        name="ember",
        primary="bright_red",
        accent="yellow",
        good="green",
        warn="yellow",
        danger="red",
        muted="bright_black",
        panel="red",
    ),
    "ocean": Theme(
        name="ocean",
        primary="bright_cyan",
        accent="blue",
        good="green",
        warn="yellow",
        danger="red",
        muted="bright_black",
        panel="cyan",
    ),
}


@dataclass
class TranscriptEntry:
    role: str
    content: str
    created_at: float = field(default_factory=time.time)


@dataclass
class ActivityEntry:
    kind: str
    text: str
    tool: str = ""
    is_error: bool = False
    created_at: float = field(default_factory=time.time)


def _clip(text: str, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def _elapsed(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "0s"
    seconds = int(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minute}m"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _short_session_id(session_id: str) -> str:
    return session_id.split(":", 1)[-1]


def _coerce_detail_mode(value: str | None, current: DetailMode = "expanded") -> DetailMode:
    raw = (value or "").strip().lower()
    if raw == "cycle":
        index = _DETAIL_MODES.index(current)
        return _DETAIL_MODES[(index + 1) % len(_DETAIL_MODES)]  # type: ignore[return-value]
    if raw in _DETAIL_MODES:
        return raw  # type: ignore[return-value]
    return current


def _workspace_label(cwd: Path | None = None) -> str:
    cwd = cwd or Path.cwd()
    branch = ""
    with suppress(Exception):
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
        branch = result.stdout.strip()
    suffix = f" ({branch})" if branch else ""
    home = str(Path.home())
    label = str(cwd)
    if label.startswith(home):
        label = "~" + label[len(home) :]
    return label + suffix


def render_event(event: dict[str, Any]) -> str | None:
    """Map a stream event to a concise progress string, or ``None`` to skip."""
    etype = event.get("type")
    if etype == "tool_call":
        return format_tool_call(str(event.get("tool") or ""), event.get("params") or {})
    if etype == "tool_result":
        content = _clip(str(event.get("content") or ""), 240)
        if not content:
            return None
        tool = str(event.get("tool") or "tool")
        status = "failed" if event.get("is_error") else "finished"
        return f"{tool} {status}: {content}"
    if etype == "status":
        msg = str(event.get("message") or "").strip()
        return msg if msg and msg != "Thinking..." else None
    if etype == "error":
        return str(event.get("detail") or event.get("message") or "Gateway error")
    return None


class AgentTUI:
    def __init__(
        self,
        ws_url: str,
        console: Console | None = None,
        *,
        theme: str = "distill",
        session_id: str | None = None,
    ) -> None:
        self._ws_url = ws_url
        self._console = console or Console()
        self._session_id = session_id or self._fresh_session()
        self._theme_name = theme if theme in _THEMES else "distill"
        self._status = "starting"
        self._connected = False
        self._detail_mode: DetailMode = "expanded"
        self._started_at = time.time()
        self._turn_started_at: float | None = None
        self._workspace = _workspace_label()
        self._transcript: list[TranscriptEntry] = []
        self._activity: list[ActivityEntry] = []
        self._streaming_answer = ""
        self._turn_count = 0
        self._tool_counts: Counter[str] = Counter()
        self._last_error = ""

    @staticmethod
    def _fresh_session() -> str:
        return f"tui:{uuid.uuid4().hex[:8]}"

    @property
    def _theme(self) -> Theme:
        return _THEMES[self._theme_name]

    def _push_activity(
        self,
        kind: str,
        text: str,
        *,
        tool: str = "",
        is_error: bool = False,
    ) -> None:
        if tool:
            self._tool_counts[tool] += 1
        self._activity.append(ActivityEntry(kind=kind, text=text, tool=tool, is_error=is_error))
        self._activity = self._activity[-80:]

    def _print_overlay(self, title: str, body: RenderableType, *, border: str | None = None) -> None:
        self._console.print(Panel(body, title=title, title_align="left", border_style=border or self._theme.panel))

    def _commands_table(self) -> Table:
        table = Table.grid(padding=(0, 2))
        table.add_column(style=f"bold {self._theme.accent}", no_wrap=True)
        table.add_column(style=self._theme.muted)
        table.add_row("/new, /reset", "start a fresh conversation")
        table.add_row("/clear", "clear the local transcript and activity feed")
        table.add_row("/details [cycle|expanded|collapsed|hidden]", "change tool/result visibility")
        table.add_row("/theme [distill|ocean|ember|mono]", "switch the TUI palette live")
        table.add_row("/usage", "show turn, tool, and elapsed-time counters")
        table.add_row("/status", "show connection, workspace, and session details")
        table.add_row("/sessions", "show the active local session card")
        table.add_row("/export [path]", "write a Markdown transcript")
        table.add_row("/help", "open this command overlay")
        table.add_row("/quit, /exit", "leave the TUI")
        return table

    def _handle_command(self, line: str) -> bool:
        """Handle a slash command. Returns False if the session should end."""
        parts = line.split(maxsplit=2)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit"):
            return False
        if cmd in ("/new", "/reset"):
            self._session_id = self._fresh_session()
            self._transcript.clear()
            self._activity.clear()
            self._streaming_answer = ""
            self._tool_counts.clear()
            self._turn_count = 0
            self._status = "ready"
            self._console.print(f"[{self._theme.good}]Started a new conversation.[/{self._theme.good}]")
        elif cmd == "/clear":
            self._transcript.clear()
            self._activity.clear()
            self._streaming_answer = ""
            self._console.print(f"[{self._theme.good}]Cleared the local transcript.[/{self._theme.good}]")
        elif cmd == "/help":
            self._print_overlay("Command Palette", self._commands_table())
        elif cmd == "/details":
            self._detail_mode = _coerce_detail_mode(arg or "cycle", self._detail_mode)
            self._console.print(f"[{self._theme.accent}]Details mode: {self._detail_mode}[/{self._theme.accent}]")
        elif cmd == "/theme":
            if not arg:
                self._console.print("Themes: " + ", ".join(sorted(_THEMES)))
            elif arg not in _THEMES:
                self._console.print(f"[{self._theme.warn}]Unknown theme {arg}. Choose: {', '.join(sorted(_THEMES))}[/{self._theme.warn}]")
            else:
                self._theme_name = arg
                self._console.print(f"[{self._theme.accent}]Theme: {arg}[/{self._theme.accent}]")
        elif cmd == "/usage":
            self._print_overlay("Usage", self._usage_table())
        elif cmd == "/status":
            self._print_overlay("Status", self._status_table())
        elif cmd in ("/sessions", "/switch"):
            self._print_overlay("Live Session", self._session_panel())
        elif cmd == "/export":
            export_path = Path(arg or f"distill-transcript-{_short_session_id(self._session_id)}.md")
            self._export_transcript(export_path)
        else:
            self._console.print(f"[{self._theme.warn}]Unknown command {cmd}. Type /help.[/{self._theme.warn}]")
        return True

    async def run(self) -> None:
        self._console.print(self._render_dashboard())
        try:
            self._status = "connecting"
            async with ws_connect(self._ws_url, open_timeout=5.0) as ws:
                self._connected = True
                self._status = "ready"
                self._push_activity("status", f"Connected to {self._ws_url}")
                self._console.print(self._render_dashboard())
                while True:
                    line = await asyncio.to_thread(self._read_line)
                    if line is None:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("/"):
                        if not self._handle_command(line):
                            break
                        continue
                    await self._run_turn(ws, line)
        except (ConnectionError, OSError, TimeoutError) as exc:
            self._connected = False
            self._status = "offline"
            self._last_error = str(exc)
            self._print_connection_error(exc)
        except KeyboardInterrupt:
            pass
        finally:
            self._status = "closed"
            self._console.print(f"[{self._theme.muted}]Bye.[/{self._theme.muted}]")

    def _read_line(self) -> str | None:
        try:
            prompt = Text("\nYou ", style=f"bold {self._theme.good}")
            prompt.append("> ", style=f"bold {self._theme.accent}")
            self._console.print(prompt, end="")
            return input()
        except EOFError:
            return None

    async def _run_turn(self, ws: Any, text: str) -> None:
        await ws.send(json.dumps({"session_id": self._session_id, "text": text}))
        self._transcript.append(TranscriptEntry("you", text))
        self._turn_count += 1
        self._turn_started_at = time.time()
        self._streaming_answer = ""
        self._status = "thinking"

        status = Status(
            f"[bold {self._theme.primary}]Thinking...[/bold {self._theme.primary}]",
            console=self._console,
            spinner="dots",
        )
        status_running = False

        def start_status() -> None:
            nonlocal status_running
            if not status_running:
                status.start()
                status_running = True

        def stop_status() -> None:
            nonlocal status_running
            if status_running:
                status.stop()
                status_running = False

        live: Live | None = None

        def start_live() -> None:
            nonlocal live
            if live is None:
                live = Live(
                    self._render_dashboard(),
                    console=self._console,
                    refresh_per_second=12,
                    transient=False,
                )
                live.start()

        def update_live() -> None:
            if live is not None:
                live.update(self._render_dashboard())

        start_status()
        try:
            async for raw in ws:
                event: dict[str, Any] = json.loads(raw)
                etype = event.get("type")

                if etype == "token":
                    stop_status()
                    self._status = "streaming"
                    self._streaming_answer += str(event.get("content") or "")
                    start_live()
                    update_live()
                    continue

                if etype in ("text", "final_answer"):
                    stop_status()
                    answer = str(event.get("content") or self._streaming_answer).strip()
                    if answer:
                        self._transcript.append(TranscriptEntry("distill", answer))
                    self._streaming_answer = ""
                    self._status = "ready"
                    if live is None:
                        self._console.print(self._render_dashboard())
                    else:
                        update_live()
                    return

                if etype == "tool_call":
                    self._status = "running"
                elif etype == "tool_result":
                    self._status = "checking" if not event.get("is_error") else "repairing"
                elif etype == "error":
                    self._status = "error"
                    self._last_error = str(event.get("detail") or event.get("message") or "Gateway error")

                line = render_event(event)
                if line:
                    self._push_activity(
                        etype or "event",
                        line,
                        tool=str(event.get("tool") or ""),
                        is_error=bool(event.get("is_error") or etype == "error"),
                    )
                    if status_running:
                        status.update(f"[bold {self._theme.accent}]Working:[/bold {self._theme.accent}] {line}")
                    start_live()
                    update_live()

        except KeyboardInterrupt:
            self._status = "interrupted"
            with suppress(Exception):
                await ws.send(json.dumps({"type": "cancel", "session_id": self._session_id}))
            self._push_activity("status", "Interrupted by user", is_error=True)
            self._console.print(f"[{self._theme.warn}]Interrupted.[/{self._theme.warn}]")
        finally:
            stop_status()
            if live is not None:
                live.stop()
            self._turn_started_at = None
            if self._status not in {"offline", "closed", "error"}:
                self._status = "ready"

    def _render_dashboard(self) -> RenderableType:
        width = self._console.width or 100
        if width < 88:
            return Group(
                self._header_panel(compact=True),
                self._transcript_panel(max_items=4),
                self._activity_panel(max_items=5),
                self._footer_panel(),
            )

        table = Table.grid(expand=True)
        table.add_column(ratio=3)
        table.add_column(ratio=2)
        table.add_row(self._transcript_panel(max_items=6), self._side_panel())
        return Group(self._header_panel(), table, self._footer_panel())

    def _header_panel(self, *, compact: bool = False) -> Panel:
        theme = self._theme
        title = Text("DISTILL", style=f"bold {theme.primary}")
        title.append(" terminal", style=theme.muted)
        subtitle = Text("Evidence-gated agent workspace", style=theme.muted)
        meta = Text()
        meta.append(f"session {_short_session_id(self._session_id)}", style=theme.accent)
        meta.append("  |  ", style=theme.muted)
        meta.append(self._status, style=self._status_style())
        meta.append("  |  ", style=theme.muted)
        meta.append(_elapsed(time.time() - self._started_at), style=theme.muted)
        if not compact:
            meta.append("\n")
            meta.append(_clip(self._workspace, 110), style=theme.muted)
        return Panel(
            Group(Align.center(title), Align.center(subtitle), Align.center(meta)),
            border_style=theme.panel,
            padding=(0, 1),
        )

    def _transcript_panel(self, *, max_items: int) -> Panel:
        theme = self._theme
        items: list[RenderableType] = []
        visible = self._transcript[-max_items:]
        if not visible and not self._streaming_answer:
            intro = Text()
            intro.append("Ready for a task.\n", style=f"bold {theme.good}")
            intro.append("Type naturally, or use /help for commands. ", style=theme.muted)
            intro.append("Tool work streams on the right.", style=theme.muted)
            items.append(intro)
        for entry in visible:
            role_style = f"bold {theme.good}" if entry.role == "you" else f"bold {theme.primary}"
            role = "You" if entry.role == "you" else "Distill"
            items.append(
                Panel(
                    Markdown(_clip(entry.content, 3200)),
                    title=role,
                    title_align="left",
                    border_style=role_style,
                    padding=(0, 1),
                )
            )
        if self._streaming_answer:
            items.append(
                Panel(
                    Markdown(_clip(self._streaming_answer, 3200)),
                    title="Distill (streaming)",
                    title_align="left",
                    border_style=theme.primary,
                    padding=(0, 1),
                )
            )
        return Panel(
            Group(*items),
            title="Conversation",
            title_align="left",
            border_style=theme.panel,
        )

    def _side_panel(self) -> Panel:
        return Panel(
            Group(
                self._status_table(),
                self._activity_panel(max_items=9),
                self._quick_commands_panel(),
            ),
            title="Control Room",
            title_align="left",
            border_style=self._theme.panel,
        )

    def _status_table(self) -> Table:
        theme = self._theme
        table = Table.grid(padding=(0, 1))
        table.add_column(style=theme.muted, no_wrap=True)
        table.add_column(style=theme.accent)
        table.add_row("connection", "online" if self._connected else "offline")
        table.add_row("gateway", _clip(self._ws_url, 40))
        table.add_row("session", self._session_id)
        table.add_row("status", self._status)
        table.add_row("details", self._detail_mode)
        table.add_row("turns", str(self._turn_count))
        table.add_row("tools", str(sum(self._tool_counts.values())))
        if self._turn_started_at:
            table.add_row("turn time", _elapsed(time.time() - self._turn_started_at))
        table.add_row("uptime", _elapsed(time.time() - self._started_at))
        if self._last_error:
            table.add_row("last error", _clip(self._last_error, 40))
        return table

    def _usage_table(self) -> Table:
        theme = self._theme
        table = Table(title=None, box=None, expand=False, show_header=False, padding=(0, 2))
        table.add_column(style=theme.muted)
        table.add_column(style=theme.accent)
        table.add_row("session", self._session_id)
        table.add_row("turns", str(self._turn_count))
        table.add_row("transcript entries", str(len(self._transcript)))
        table.add_row("visible tool events", str(len(self._activity)))
        table.add_row("tool calls/results", str(sum(self._tool_counts.values())))
        table.add_row("uptime", _elapsed(time.time() - self._started_at))
        if self._tool_counts:
            table.add_row("top tools", ", ".join(f"{name} x{count}" for name, count in self._tool_counts.most_common(4)))
        return table

    def _activity_panel(self, *, max_items: int) -> Panel:
        theme = self._theme
        if self._detail_mode == "hidden":
            body: RenderableType = Text("Tool activity hidden. Use /details to show it.", style=theme.muted)
        elif self._detail_mode == "collapsed":
            calls = sum(1 for item in self._activity if item.kind == "tool_call")
            results = sum(1 for item in self._activity if item.kind == "tool_result")
            errors = sum(1 for item in self._activity if item.is_error)
            body = Text(f"{calls} calls, {results} results, {errors} errors", style=theme.muted)
        else:
            rows = Table.grid(padding=(0, 1))
            rows.add_column(style=theme.muted, no_wrap=True)
            rows.add_column()
            entries = self._activity[-max_items:]
            if not entries:
                rows.add_row("--", Text("No tool activity yet.", style=theme.muted))
            for entry in entries:
                age = _elapsed(time.time() - entry.created_at)
                style = theme.danger if entry.is_error else theme.accent if entry.kind == "tool_call" else theme.muted
                rows.add_row(age, Text(_clip(entry.text, 120), style=style))
            body = rows
        return Panel(body, title="Activity", title_align="left", border_style=theme.panel)

    def _quick_commands_panel(self) -> Panel:
        theme = self._theme
        body = Text()
        body.append("/help", style=f"bold {theme.accent}")
        body.append(" commands  ", style=theme.muted)
        body.append("/details", style=f"bold {theme.accent}")
        body.append(" view  ", style=theme.muted)
        body.append("/theme", style=f"bold {theme.accent}")
        body.append(" skin", style=theme.muted)
        return Panel(body, title="Shortcuts", title_align="left", border_style=theme.panel)

    def _session_panel(self) -> RenderableType:
        theme = self._theme
        table = Table.grid(padding=(0, 2))
        table.add_column(style=theme.muted)
        table.add_column(style=theme.accent)
        table.add_row("active session", self._session_id)
        table.add_row("workspace", self._workspace)
        table.add_row("turns", str(self._turn_count))
        table.add_row("started", _elapsed(time.time() - self._started_at) + " ago")
        table.add_row("gateway", self._ws_url)
        return table

    def _footer_panel(self) -> Panel:
        theme = self._theme
        text = Text()
        text.append("ready" if self._status == "ready" else self._status, style=self._status_style())
        text.append("  |  ", style=theme.muted)
        text.append("Enter sends  /help commands  Ctrl+C interrupts  /quit exits", style=theme.muted)
        return Panel(text, border_style=theme.panel, padding=(0, 1))

    def _status_style(self) -> str:
        theme = self._theme
        if self._status in {"ready", "streaming"}:
            return theme.good
        if self._status in {"running", "thinking", "checking", "connecting", "starting"}:
            return theme.accent
        if self._status in {"interrupted", "repairing"}:
            return theme.warn
        if self._status in {"offline", "error"}:
            return theme.danger
        return theme.muted

    def _export_transcript(self, export_path: Path) -> None:
        export_path = export_path.expanduser()
        lines = ["# Distill Transcript", "", f"Session: `{self._session_id}`", ""]
        for entry in self._transcript:
            role = "User" if entry.role == "you" else "Distill"
            lines.extend([f"## {role}", "", entry.content.strip(), ""])
        export_path.write_text("\n".join(lines), encoding="utf-8")
        self._console.print(f"[{self._theme.good}]Exported transcript to {export_path}[/{self._theme.good}]")

    def _print_connection_error(self, exc: BaseException) -> None:
        theme = self._theme
        body = Text()
        body.append(f"Could not reach the gateway at {self._ws_url}\n\n", style=theme.danger)
        body.append(str(exc), style=theme.muted)
        body.append("\n\nStart Distill first, then reopen the TUI:\n", style=theme.muted)
        body.append("  distill start\n", style=theme.accent)
        body.append("  distill\n\n", style=theme.accent)
        body.append("Or connect to another gateway:\n", style=theme.muted)
        body.append("  distill --url ws://host:8000/ws/stream", style=theme.accent)
        self._print_overlay("Gateway Offline", body, border=theme.danger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill interactive terminal UI")
    parser.add_argument(
        "--url",
        default=os.getenv("AGENT_TUI_WS_URL", _DEFAULT_URL),
        help=f"gateway WebSocket URL (default: {_DEFAULT_URL})",
    )
    parser.add_argument(
        "--theme",
        default=os.getenv("DISTILL_TUI_THEME", "distill"),
        choices=sorted(_THEMES),
        help="TUI color theme",
    )
    parser.add_argument(
        "--session",
        default=os.getenv("DISTILL_TUI_SESSION", ""),
        help="resume/use a specific session id",
    )
    args = parser.parse_args()
    try:
        asyncio.run(AgentTUI(args.url, theme=args.theme, session_id=args.session or None).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
