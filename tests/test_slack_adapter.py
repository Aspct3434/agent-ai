"""Unit tests for the Slack Socket Mode adapter.

All tests run without a live Slack connection — the stream_fn and HTTP
client are fully mocked.
"""
from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapters.slack import SlackAdapter, _chunk_text, _parse_str_set

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_chunk_short_unchanged(self) -> None:
        assert _chunk_text("hi", 3500) == ["hi"]

    def test_chunk_hard_split(self) -> None:
        text = "z" * 8000
        chunks = _chunk_text(text, 3500)
        assert len(chunks) == 3
        assert all(len(c) <= 3500 for c in chunks)
        assert "".join(chunks) == text

    def test_parse_str_set_empty(self) -> None:
        assert _parse_str_set("  ") is None

    def test_parse_str_set_values(self) -> None:
        assert _parse_str_set("U1, U2") == frozenset({"U1", "U2"})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _stream_of(*events: dict[str, Any]):
    async def _fn(_session_id: str, _text: str) -> AsyncIterator[dict[str, Any]]:
        for event in events:
            yield event

    return _fn


@pytest.fixture
def echo_stream_fn():
    async def _fn(_session_id: str, text: str) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "text", "content": f"reply:{text}"}

    return _fn


@pytest.fixture
def mock_http():
    http = AsyncMock()
    http.post = AsyncMock(return_value=MagicMock())
    http.aclose = AsyncMock()
    return http


@pytest.fixture
def reset_fn():
    return MagicMock(return_value=True)


@pytest.fixture
def adapter(echo_stream_fn, mock_http, reset_fn):
    a = SlackAdapter(
        bot_token="xoxb-test",
        app_token="xapp-test",
        stream_fn=echo_stream_fn,
        reset_fn=reset_fn,
    )
    a._http = mock_http
    a._allowed = None
    return a


def _posts(mock_http: AsyncMock) -> list[str]:
    return [
        c.kwargs["json"]["text"]
        for c in mock_http.post.call_args_list
        if c.args and c.args[0].endswith("/chat.postMessage")
    ]


def _msg(text: str, channel: str = "C1", user: str = "U1", **extra) -> dict:
    return {"type": "message", "text": text, "channel": channel, "user": user, **extra}


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


class TestSlackGating:
    @pytest.mark.asyncio
    async def test_non_message_event_ignored(self, adapter, mock_http) -> None:
        await adapter._handle_event({"type": "reaction_added"})
        mock_http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_bot_message_ignored(self, adapter, mock_http) -> None:
        await adapter._handle_event(_msg("hi", bot_id="B123"))
        mock_http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_subtype_ignored(self, adapter, mock_http) -> None:
        await adapter._handle_event(_msg("edited", subtype="message_changed"))
        mock_http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_text_ignored(self, adapter, mock_http) -> None:
        await adapter._handle_event(_msg("   "))
        mock_http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_user_ignored(self, adapter, mock_http) -> None:
        adapter._allowed = frozenset({"U_ALLOWED"})
        await adapter._handle_event(_msg("hi", user="U_OTHER"))
        mock_http.post.assert_not_called()


# ---------------------------------------------------------------------------
# Streaming + commands
# ---------------------------------------------------------------------------


class TestSlackStreaming:
    @pytest.mark.asyncio
    async def test_final_answer_posted(self, adapter, mock_http) -> None:
        await adapter._handle_event(_msg("hello"))
        assert any("reply:hello" in m for m in _posts(mock_http))

    @pytest.mark.asyncio
    async def test_session_scoped_to_channel(self, adapter) -> None:
        captured: list[str] = []

        async def _cap(sid: str, _text: str) -> AsyncIterator[dict[str, Any]]:
            captured.append(sid)
            yield {"type": "text", "content": "ok"}

        adapter._stream_fn = _cap
        await adapter._handle_event(_msg("hi", channel="C9"))
        assert captured == ["slack:C9"]

    @pytest.mark.asyncio
    async def test_tool_calls_shown(self, mock_http) -> None:
        stream = _stream_of(
            {"type": "tool_call", "tool": "set_task_contract", "params": {}},  # silent
            {"type": "tool_call", "tool": "web_search", "params": {"query": "rust book"}},
            {"type": "text", "content": "done"},
        )
        a = SlackAdapter("xoxb", "xapp", stream, lambda _s: True)
        a._http = mock_http
        a._allowed = None
        await a._handle_event(_msg("go"))
        posts = _posts(mock_http)
        assert any("rust book" in m for m in posts)
        assert any("done" in m for m in posts)
        assert not any("set_task_contract" in m for m in posts)

    @pytest.mark.asyncio
    async def test_new_command_resets(self, adapter, mock_http, reset_fn) -> None:
        await adapter._handle_event(_msg("/new", channel="C5"))
        reset_fn.assert_called_once_with("slack:C5")
        assert any("new conversation" in m.lower() for m in _posts(mock_http))

    @pytest.mark.asyncio
    async def test_help_command(self, adapter, mock_http) -> None:
        await adapter._handle_event(_msg("/help"))
        assert any("/stop" in m for m in _posts(mock_http))

    @pytest.mark.asyncio
    async def test_stop_with_nothing_running(self, adapter, mock_http) -> None:
        await adapter._handle_event(_msg("/stop"))
        assert any("Nothing is running" in m for m in _posts(mock_http))

    @pytest.mark.asyncio
    async def test_stop_cancels_running_turn(self, mock_http) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def _blocking(_sid: str, _text: str) -> AsyncIterator[dict[str, Any]]:
            yield {"type": "tool_call", "tool": "web_search", "params": {"query": "x"}}
            started.set()
            await release.wait()
            yield {"type": "text", "content": "never"}

        a = SlackAdapter("xoxb", "xapp", _blocking, lambda _s: True)
        a._http = mock_http
        a._allowed = None
        turn = asyncio.create_task(a._handle_event(_msg("long", channel="C1")))
        await asyncio.wait_for(started.wait(), timeout=2)
        await a._handle_command("C1", "/stop")
        await asyncio.wait_for(turn, timeout=2)
        assert any("Stopped" in m for m in _posts(mock_http))
        assert not any("never" in m for m in _posts(mock_http))
