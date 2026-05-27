"""Unit tests for the rich TUI's pure helpers and command handling.

The WebSocket loop is integration-tested manually against a live gateway;
here we cover the deterministic pieces: event rendering and slash commands.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tui import AgentTUI, render_event


class TestRenderEvent:
    def test_tool_call_rendered(self) -> None:
        out = render_event(
            {"type": "tool_call", "tool": "execute_terminal_command", "params": {"command": "ls"}}
        )
        assert out is not None
        assert "ls" in out

    def test_internal_tool_hidden(self) -> None:
        assert render_event({"type": "tool_call", "tool": "set_task_contract", "params": {}}) is None

    def test_status_rendered(self) -> None:
        assert render_event({"type": "status", "message": "Escalating…"}) == "Escalating…"

    def test_tool_result_rendered(self) -> None:
        out = render_event(
            {
                "type": "tool_result",
                "tool": "execute_terminal_command",
                "is_error": False,
                "content": "exit_code=0",
            }
        )
        assert out is not None
        assert "execute_terminal_command finished" in out

    def test_thinking_status_hidden(self) -> None:
        assert render_event({"type": "status", "message": "Thinking..."}) is None

    def test_token_and_text_skipped(self) -> None:
        # Handled separately by the turn loop.
        assert render_event({"type": "token", "content": "x"}) is None
        assert render_event({"type": "text", "content": "done"}) is None


def _tui() -> tuple[AgentTUI, io.StringIO]:
    buf = io.StringIO()
    return AgentTUI("ws://test/ws", console=Console(file=buf, force_terminal=False)), buf


class TestCommands:
    def test_quit_ends_session(self) -> None:
        tui, _ = _tui()
        assert tui._handle_command("/quit") is False
        assert tui._handle_command("/exit") is False

    def test_new_starts_fresh_session(self) -> None:
        tui, buf = _tui()
        old = tui._session_id
        assert tui._handle_command("/new") is True
        assert tui._session_id != old
        assert "new conversation" in buf.getvalue().lower()

    def test_reset_alias(self) -> None:
        tui, _ = _tui()
        old = tui._session_id
        assert tui._handle_command("/reset") is True
        assert tui._session_id != old

    def test_help(self) -> None:
        tui, buf = _tui()
        assert tui._handle_command("/help") is True
        assert "/quit" in buf.getvalue()

    def test_details_cycle(self) -> None:
        tui, _ = _tui()
        assert tui._detail_mode == "expanded"
        assert tui._handle_command("/details") is True
        assert tui._detail_mode == "collapsed"

    def test_theme_command(self) -> None:
        tui, _ = _tui()
        assert tui._handle_command("/theme ocean") is True
        assert tui._theme_name == "ocean"

    def test_unknown_command(self) -> None:
        tui, buf = _tui()
        assert tui._handle_command("/bogus") is True
        assert "Unknown command" in buf.getvalue()

    def test_fresh_session_format(self) -> None:
        tui, _ = _tui()
        assert tui._session_id.startswith("tui:")
