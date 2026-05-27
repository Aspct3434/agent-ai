"""Unit tests for the shared chat-adapter progress formatter."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapters._progress import format_tool_call


class TestSilentTools:
    def test_set_task_contract_hidden(self) -> None:
        assert format_tool_call("set_task_contract", {"mode": "execute"}) is None

    def test_update_plan_hidden(self) -> None:
        assert format_tool_call("update_plan", {"steps": []}) is None

    def test_read_only_tools_hidden(self) -> None:
        for tool in ("read_file", "list_directory", "read_query", "get_system_environment"):
            assert format_tool_call(tool, {}) is None


class TestActionTools:
    def test_terminal_command_shows_command(self) -> None:
        out = format_tool_call("execute_terminal_command", {"command": "java -version"})
        assert out is not None
        assert "java -version" in out

    def test_terminal_command_collapses_newlines(self) -> None:
        out = format_tool_call("execute_terminal_command", {"command": "a\nb\nc"})
        assert out is not None
        assert "\n" not in out
        assert "a b c" in out

    def test_long_command_truncated(self) -> None:
        out = format_tool_call("execute_terminal_command", {"command": "x" * 5000})
        assert out is not None
        assert len(out) < 400
        assert out.endswith("…")

    def test_background_service(self) -> None:
        out = format_tool_call("execute_background_service", {"command": "python app.py"})
        assert out is not None and "python app.py" in out

    def test_background_probe_shows_blocked_not_starting(self) -> None:
        out = format_tool_call(
            "execute_background_service",
            {"command": "ss -tlnp | grep 3000 || lsof -i :3000"},
        )
        assert out is not None
        assert "[blocked] Background probe" in out
        assert "Starting service" not in out

    def test_write_file_shows_path(self) -> None:
        assert "server.properties" in str(
            format_tool_call("write_text_file", {"path": "server.properties"})
        )

    def test_web_search_shows_query(self) -> None:
        assert "minecraft jar" in str(
            format_tool_call("web_search", {"query": "minecraft jar"})
        )

    def test_wait_for_port(self) -> None:
        assert "25565" in str(format_tool_call("wait_for_port", {"port": 25565}))

    def test_browser_tool(self) -> None:
        out = format_tool_call("browser_navigate", {"url": "https://example.com"})
        assert out is not None
        assert "example.com" in out

    def test_unknown_tool_generic_fallback(self) -> None:
        out = format_tool_call("some_new_tool", {})
        assert out == "🔧 some_new_tool"
