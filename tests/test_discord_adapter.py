"""Unit tests for the Discord Gateway adapter.

All tests run without a live Discord connection — the stream_fn and HTTP
client are fully mocked. The adapter streams progress (a typing indicator,
a line per tool call, then the final answer) back to the channel.
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

from adapters.discord_bot import DiscordAdapter, _chunk_text, _parse_str_set

# ---------------------------------------------------------------------------
# Pure helper: _chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_short_message_unchanged(self) -> None:
        assert _chunk_text("hello", 2000) == ["hello"]

    def test_exact_limit_unchanged(self) -> None:
        text = "d" * 2000
        assert _chunk_text(text, 2000) == [text]

    def test_splits_at_newline_within_limit(self) -> None:
        text = "a" * 1500 + "\n" + "b" * 1500
        chunks = _chunk_text(text, 2000)
        assert len(chunks) == 2
        assert all(len(c) <= 2000 for c in chunks)

    def test_hard_split_when_no_newline(self) -> None:
        text = "z" * 3000
        chunks = _chunk_text(text, 2000)
        assert len(chunks) == 2
        assert all(len(c) <= 2000 for c in chunks)
        assert "".join(chunks) == text

    def test_three_chunks(self) -> None:
        text = "x" * 5000
        chunks = _chunk_text(text, 2000)
        assert len(chunks) == 3
        assert all(len(c) <= 2000 for c in chunks)


# ---------------------------------------------------------------------------
# Pure helper: _parse_str_set
# ---------------------------------------------------------------------------


class TestParseStrSet:
    def test_empty_returns_none(self) -> None:
        assert _parse_str_set("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert _parse_str_set("   ") is None

    def test_single_id(self) -> None:
        assert _parse_str_set("abc123") == frozenset({"abc123"})

    def test_multiple_ids(self) -> None:
        assert _parse_str_set("a, b, c") == frozenset({"a", "b", "c"})

    def test_trailing_comma_ignored(self) -> None:
        assert _parse_str_set("x, y,") == frozenset({"x", "y"})


# ---------------------------------------------------------------------------
# Streaming helpers + fixtures
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
    a = DiscordAdapter(token="Bot-test-token", stream_fn=echo_stream_fn, reset_fn=reset_fn)
    a._http = mock_http
    a._allowed = None
    return a


def _message_contents(mock_http: AsyncMock) -> list[str]:
    """Contents of all posted channel messages (the /messages endpoint)."""
    return [
        c.kwargs["json"]["content"]
        for c in mock_http.post.call_args_list
        if c.args and c.args[0].endswith("/messages")
    ]


def _typing_calls(mock_http: AsyncMock) -> list:
    return [c for c in mock_http.post.call_args_list if c.args and c.args[0].endswith("/typing")]


def _make_msg(
    content: str,
    channel_id: str = "c1",
    user_id: str = "u1",
    is_bot: bool = False,
) -> dict:
    return {
        "author": {"id": user_id, "bot": is_bot},
        "content": content,
        "channel_id": channel_id,
    }


# ---------------------------------------------------------------------------
# Gating (runs before any streaming)
# ---------------------------------------------------------------------------


class TestDiscordGating:
    @pytest.mark.asyncio
    async def test_bot_messages_ignored(self, adapter, mock_http) -> None:
        await adapter._handle_message(_make_msg("I'm a bot", is_bot=True))
        mock_http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_content_ignored(self, adapter, mock_http) -> None:
        await adapter._handle_message(_make_msg(""))
        mock_http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_content_ignored(self, adapter, mock_http) -> None:
        await adapter._handle_message(_make_msg("   "))
        mock_http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_user_ignored(self, adapter, mock_http) -> None:
        adapter._allowed = frozenset({"allowed_user"})
        await adapter._handle_message(_make_msg("hi", user_id="stranger"))
        mock_http.post.assert_not_called()


# ---------------------------------------------------------------------------
# Streaming behaviour
# ---------------------------------------------------------------------------


class TestDiscordStreaming:
    @pytest.mark.asyncio
    async def test_typing_indicator_sent(self, adapter, mock_http) -> None:
        await adapter._handle_message(_make_msg("hello"))
        assert len(_typing_calls(mock_http)) >= 1

    @pytest.mark.asyncio
    async def test_final_answer_sent(self, adapter, mock_http) -> None:
        await adapter._handle_message(_make_msg("hello"))
        assert any("reply:hello" in m for m in _message_contents(mock_http))

    @pytest.mark.asyncio
    async def test_session_id_scoped_to_channel(self, adapter) -> None:
        captured: list[str] = []

        async def _capturing(sid: str, _text: str) -> AsyncIterator[dict[str, Any]]:
            captured.append(sid)
            yield {"type": "text", "content": "ok"}

        adapter._stream_fn = _capturing
        await adapter._handle_message(_make_msg("test", channel_id="chan-xyz"))
        assert captured == ["discord:chan-xyz"]

    @pytest.mark.asyncio
    async def test_tool_calls_shown_as_progress(self, mock_http) -> None:
        stream = _stream_of(
            {"type": "tool_call", "tool": "update_plan", "params": {}},  # silent
            {"type": "tool_call", "tool": "web_search", "params": {"query": "minecraft jar"}},
            {"type": "text", "content": "done"},
        )
        a = DiscordAdapter(token="t", stream_fn=stream, reset_fn=lambda _s: True)
        a._http = mock_http
        a._allowed = None
        await a._handle_message(_make_msg("find it"))
        contents = _message_contents(mock_http)
        assert any("minecraft jar" in m for m in contents)  # tool surfaced
        assert any("done" in m for m in contents)  # final answer
        assert not any("update_plan" in m for m in contents)  # internal hidden

    @pytest.mark.asyncio
    async def test_long_final_answer_chunked(self, mock_http) -> None:
        stream = _stream_of({"type": "text", "content": "A" * 5000})
        a = DiscordAdapter(token="t", stream_fn=stream, reset_fn=lambda _s: True)
        a._http = mock_http
        a._allowed = None
        await a._handle_message(_make_msg("go"))
        contents = _message_contents(mock_http)
        assert len(contents) == 3
        assert all(len(m) <= 2000 for m in contents)

    @pytest.mark.asyncio
    async def test_stream_error_reported_to_channel(self, mock_http) -> None:
        async def _failing(_sid: str, _text: str) -> AsyncIterator[dict[str, Any]]:
            raise RuntimeError("Model offline")
            yield {}  # unreachable; marks this as an async generator

        a = DiscordAdapter(token="t", stream_fn=_failing, reset_fn=lambda _s: True)
        a._http = mock_http
        a._allowed = None
        await a._handle_message(_make_msg("crash me"))
        assert any("Error" in m for m in _message_contents(mock_http))


# ---------------------------------------------------------------------------
# In-chat slash commands
# ---------------------------------------------------------------------------


class TestDiscordCommands:
    @pytest.mark.asyncio
    async def test_new_command_resets_session(self, adapter, mock_http, reset_fn) -> None:
        await adapter._handle_message(_make_msg("/new", channel_id="c9"))
        reset_fn.assert_called_once_with("discord:c9")
        assert any("new conversation" in m.lower() for m in _message_contents(mock_http))

    @pytest.mark.asyncio
    async def test_reset_command_alias(self, adapter, mock_http, reset_fn) -> None:
        await adapter._handle_message(_make_msg("/reset", channel_id="c9"))
        reset_fn.assert_called_once_with("discord:c9")

    @pytest.mark.asyncio
    async def test_help_command_lists_commands(self, adapter, mock_http) -> None:
        await adapter._handle_message(_make_msg("/help"))
        assert any("/stop" in m for m in _message_contents(mock_http))

    @pytest.mark.asyncio
    async def test_unknown_command_reported(self, adapter, mock_http) -> None:
        await adapter._handle_message(_make_msg("/frobnicate"))
        assert any("Unknown command" in m for m in _message_contents(mock_http))

    @pytest.mark.asyncio
    async def test_stop_with_nothing_running(self, adapter, mock_http) -> None:
        await adapter._handle_message(_make_msg("/stop"))
        assert any("Nothing is running" in m for m in _message_contents(mock_http))

    @pytest.mark.asyncio
    async def test_stop_cancels_running_turn(self, mock_http) -> None:
        started = asyncio.Event()
        release = asyncio.Event()  # never set — the turn blocks here

        async def _blocking(_sid: str, _text: str) -> AsyncIterator[dict[str, Any]]:
            yield {"type": "tool_call", "tool": "web_search", "params": {"query": "x"}}
            started.set()
            await release.wait()
            yield {"type": "text", "content": "never reached"}

        a = DiscordAdapter(token="t", stream_fn=_blocking, reset_fn=lambda _s: True)
        a._http = mock_http
        a._allowed = None

        turn = asyncio.create_task(a._handle_message(_make_msg("long task", channel_id="c1")))
        await asyncio.wait_for(started.wait(), timeout=2)
        await a._handle_command("c1", "/stop")
        await asyncio.wait_for(turn, timeout=2)

        contents = _message_contents(mock_http)
        assert any("Stopped" in m for m in contents)
        assert not any("never reached" in m for m in contents)
