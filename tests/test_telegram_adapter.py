"""Unit tests for the Telegram Bot API adapter.

All tests run without a real bot token — HTTP is fully mocked. The adapter
consumes a streaming task runner (stream_fn) and surfaces a live "typing…"
action, a line per tool call, and the final answer.
"""
from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapters.telegram import TelegramAdapter, _chunk_text, _parse_int_set

# ---------------------------------------------------------------------------
# Pure helper: _chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_short_message_unchanged(self) -> None:
        assert _chunk_text("hello", 100) == ["hello"]

    def test_exact_limit_unchanged(self) -> None:
        text = "x" * 100
        assert _chunk_text(text, 100) == [text]

    def test_splits_at_last_newline_before_limit(self) -> None:
        chunks = _chunk_text("line1\nline2", 8)
        assert len(chunks) == 2
        assert chunks[0] == "line1"
        assert chunks[1] == "line2"

    def test_hard_split_when_no_newline(self) -> None:
        text = "abcdefghij"
        chunks = _chunk_text(text, 4)
        assert all(len(c) <= 4 for c in chunks)
        assert "".join(chunks) == text

    def test_empty_string_returns_single_empty(self) -> None:
        assert _chunk_text("", 100) == [""]

    def test_multi_chunk_loses_only_boundary_newlines(self) -> None:
        text = ("abc\n" * 500).rstrip()
        chunks = _chunk_text(text, 100)
        assert all(len(c) <= 100 for c in chunks)
        assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


# ---------------------------------------------------------------------------
# Pure helper: _parse_int_set
# ---------------------------------------------------------------------------


class TestParseIntSet:
    def test_empty_returns_none(self) -> None:
        assert _parse_int_set("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert _parse_int_set("   ") is None

    def test_single_id(self) -> None:
        assert _parse_int_set("12345") == frozenset({12345})

    def test_multiple_ids_with_spaces(self) -> None:
        assert _parse_int_set("1, 2, 3") == frozenset({1, 2, 3})

    def test_trailing_comma_ignored(self) -> None:
        assert _parse_int_set("10, 20,") == frozenset({10, 20})

    def test_invalid_token_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_int_set("abc")


# ---------------------------------------------------------------------------
# Streaming helpers + fixtures
# ---------------------------------------------------------------------------


def _stream_of(*events: dict[str, Any]):
    """Build a stream_fn that yields the given events then stops."""

    async def _fn(_session_id: str, _text: str) -> AsyncIterator[dict[str, Any]]:
        for event in events:
            yield event

    return _fn


@pytest.fixture
def echo_stream_fn():
    """stream_fn yielding a single final-answer event «reply to: <text>»."""

    async def _fn(_session_id: str, text: str) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "text", "content": f"reply to: {text}"}

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
    a = TelegramAdapter(token="test-token", stream_fn=echo_stream_fn, reset_fn=reset_fn)
    a._http = mock_http
    a._allowed = None  # open by default
    return a


def _make_update(update_id: int, chat_id: int, user_id: int, text: str) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "chat": {"id": chat_id},
            "from": {"id": user_id},
            "text": text,
        },
    }


def _sent_texts(mock_http: AsyncMock) -> list[str]:
    """All text strings sent via sendMessage."""
    return [
        c.kwargs["json"]["text"]
        for c in mock_http.post.call_args_list
        if c.args and "sendMessage" in c.args[0]
    ]


def _sent_messages(mock_http: AsyncMock) -> list[dict]:
    """All sendMessage JSON payloads (text + parse_mode)."""
    return [
        c.kwargs["json"]
        for c in mock_http.post.call_args_list
        if c.args and "sendMessage" in c.args[0]
    ]


def _chat_actions(mock_http: AsyncMock) -> list:
    """All sendChatAction calls (typing indicators)."""
    return [
        c for c in mock_http.post.call_args_list if c.args and "sendChatAction" in c.args[0]
    ]


# ---------------------------------------------------------------------------
# Gating (runs before any streaming)
# ---------------------------------------------------------------------------


class TestTelegramGating:
    @pytest.mark.asyncio
    async def test_start_command_sends_welcome(self, adapter, mock_http) -> None:
        await adapter._handle_update(_make_update(1, 42, 99, "/start"))
        assert any("Hello" in t for t in _sent_texts(mock_http))

    @pytest.mark.asyncio
    async def test_slash_command_does_not_start_a_turn(self, adapter, mock_http) -> None:
        # A command replies but must not kick off a streamed agent turn
        # (no typing indicator).
        await adapter._handle_update(_make_update(2, 42, 99, "/unknown"))
        assert _chat_actions(mock_http) == []

    @pytest.mark.asyncio
    async def test_whitespace_only_message_ignored(self, adapter, mock_http) -> None:
        await adapter._handle_update(_make_update(3, 42, 99, "   "))
        mock_http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_message_ignored(self, adapter, mock_http) -> None:
        await adapter._handle_update(_make_update(4, 42, 99, ""))
        mock_http.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_user_receives_access_denied(self, adapter, mock_http) -> None:
        adapter._allowed = frozenset({111})
        await adapter._handle_update(_make_update(5, 42, 999, "do something"))
        texts = _sent_texts(mock_http)
        assert len(texts) == 1
        assert "denied" in texts[0].lower()
        # No typing action for a rejected message.
        assert _chat_actions(mock_http) == []


# ---------------------------------------------------------------------------
# Streaming behaviour
# ---------------------------------------------------------------------------


class TestTelegramStreaming:
    @pytest.mark.asyncio
    async def test_typing_action_sent_immediately(self, adapter, mock_http) -> None:
        await adapter._handle_update(_make_update(6, 42, 1, "hello"))
        actions = _chat_actions(mock_http)
        assert len(actions) >= 1
        assert actions[0].kwargs["json"]["action"] == "typing"

    @pytest.mark.asyncio
    async def test_final_answer_sent(self, adapter, mock_http) -> None:
        await adapter._handle_update(_make_update(7, 42, 1, "hello"))
        assert any("reply to: hello" in t for t in _sent_texts(mock_http))

    @pytest.mark.asyncio
    async def test_tool_calls_shown_as_progress(self, mock_http) -> None:
        """Action-producing tool calls are shown; internal ones are hidden."""
        stream = _stream_of(
            {"type": "status", "message": "Thinking..."},
            {"type": "tool_call", "tool": "set_task_contract", "params": {}},  # silent
            {
                "type": "tool_call",
                "tool": "execute_terminal_command",
                "params": {"command": "java -version"},
            },
            {"type": "text", "content": "all done"},
        )
        a = TelegramAdapter(token="t", stream_fn=stream, reset_fn=lambda _s: True)
        a._http = mock_http
        a._allowed = None
        await a._handle_update(_make_update(8, 42, 1, "install java"))
        texts = _sent_texts(mock_http)
        assert any("java -version" in t for t in texts)  # tool call surfaced
        assert any("all done" in t for t in texts)  # final answer
        assert not any("set_task_contract" in t for t in texts)  # internal tool hidden

    @pytest.mark.asyncio
    async def test_long_final_answer_chunked(self, mock_http) -> None:
        stream = _stream_of({"type": "text", "content": "x" * 9000})
        a = TelegramAdapter(token="t", stream_fn=stream, reset_fn=lambda _s: True)
        a._http = mock_http
        a._allowed = None
        await a._handle_update(_make_update(9, 1, 1, "go"))
        texts = _sent_texts(mock_http)
        assert len(texts) == 3
        assert all(len(t) <= 4096 for t in texts)

    @pytest.mark.asyncio
    async def test_stream_error_reported_to_user(self, mock_http) -> None:
        async def _failing(_sid: str, _text: str) -> AsyncIterator[dict[str, Any]]:
            raise RuntimeError("LLM unavailable")
            yield {}  # unreachable; marks this as an async generator

        a = TelegramAdapter(token="t", stream_fn=_failing, reset_fn=lambda _s: True)
        a._http = mock_http
        a._allowed = None
        await a._handle_update(_make_update(10, 42, 1, "crash"))
        assert any("Error" in t for t in _sent_texts(mock_http))


class TestTelegramRichFormatting:
    @pytest.mark.asyncio
    async def test_final_answer_sent_as_html_with_bold(self, mock_http) -> None:
        stream = _stream_of({"type": "text", "content": "## Done\n\n**all good**"})
        a = TelegramAdapter(token="t", stream_fn=stream, reset_fn=lambda _s: True)
        a._http = mock_http
        a._allowed = None
        await a._handle_update(_make_update(40, 42, 1, "go"))

        msgs = [m for m in _sent_messages(mock_http) if m["text"]]
        answer = msgs[-1]
        assert answer["parse_mode"] == "HTML"
        assert "<b>Done</b>" in answer["text"]
        assert "<b>all good</b>" in answer["text"]
        assert "**" not in answer["text"]
        assert "##" not in answer["text"]

    @pytest.mark.asyncio
    async def test_tool_progress_lines_stay_plain(self, mock_http) -> None:
        stream = _stream_of(
            {
                "type": "tool_call",
                "tool": "execute_terminal_command",
                "params": {"command": "ls -la"},
            },
            {"type": "text", "content": "done"},
        )
        a = TelegramAdapter(token="t", stream_fn=stream, reset_fn=lambda _s: True)
        a._http = mock_http
        a._allowed = None
        await a._handle_update(_make_update(41, 42, 1, "go"))

        tool_msg = next(m for m in _sent_messages(mock_http) if "ls -la" in m["text"])
        assert "parse_mode" not in tool_msg  # progress lines are sent as plain text

    @pytest.mark.asyncio
    async def test_html_rejection_falls_back_to_plain(self) -> None:
        # First sendMessage (HTML) is rejected; adapter must retry as plain text.
        responses = [
            MagicMock(json=MagicMock(return_value={"ok": False, "description": "can't parse entities"})),
            MagicMock(json=MagicMock(return_value={"ok": True})),
        ]
        http = AsyncMock()
        http.post = AsyncMock(side_effect=responses)
        a = TelegramAdapter(token="t", stream_fn=_stream_of(), reset_fn=lambda _s: True)
        a._http = http
        a._allowed = None

        await a._send_rich(42, "**bold**")

        sent = [c.kwargs["json"] for c in http.post.call_args_list]
        assert sent[0]["parse_mode"] == "HTML"
        assert sent[0]["text"] == "<b>bold</b>"
        assert "parse_mode" not in sent[1]  # plain-text retry
        assert sent[1]["text"] == "bold"  # tags stripped


# ---------------------------------------------------------------------------
# Webhook handling (token validation + 409 recovery)
# ---------------------------------------------------------------------------


class TestTelegramWebhookHandling:
    @pytest.mark.asyncio
    async def test_prepare_deletes_webhook(self, adapter, mock_http) -> None:
        mock_http.get = AsyncMock(
            return_value=MagicMock(
                json=MagicMock(return_value={"ok": True, "result": {"username": "bot"}})
            )
        )
        await adapter._prepare()
        delete_calls = [
            c for c in mock_http.post.call_args_list if "deleteWebhook" in c.args[0]
        ]
        assert len(delete_calls) == 1
        assert delete_calls[0].kwargs["json"]["drop_pending_updates"] is False

    @pytest.mark.asyncio
    async def test_get_updates_409_triggers_webhook_deletion(self, adapter, mock_http) -> None:
        mock_http.post = AsyncMock(
            return_value=MagicMock(
                json=MagicMock(
                    return_value={"ok": False, "error_code": 409, "description": "Conflict"}
                )
            )
        )
        result = await adapter._get_updates()
        assert result == []
        assert any("deleteWebhook" in c.args[0] for c in mock_http.post.call_args_list)

    @pytest.mark.asyncio
    async def test_get_updates_parses_ok_result(self, adapter, mock_http) -> None:
        mock_http.post = AsyncMock(
            return_value=MagicMock(
                json=MagicMock(return_value={"ok": True, "result": [{"update_id": 5}]})
            )
        )
        result = await adapter._get_updates()
        assert result == [{"update_id": 5}]


class TestTelegramLifecycle:
    @pytest.mark.asyncio
    async def test_shutdown_cancels_task_and_closes_client(self, echo_stream_fn) -> None:
        http_mock = AsyncMock()
        http_mock.get = AsyncMock(
            return_value=MagicMock(
                json=MagicMock(return_value={"ok": True, "result": {"username": "bot"}})
            )
        )
        http_mock.post = AsyncMock(
            return_value=MagicMock(json=MagicMock(return_value={"ok": True, "result": []}))
        )
        http_mock.aclose = AsyncMock()

        with patch("httpx.AsyncClient", return_value=http_mock):
            a = TelegramAdapter(token="t", stream_fn=echo_stream_fn, reset_fn=lambda _s: True)
            await a.start()
            assert a._task is not None and not a._task.done()
            await a.shutdown()

        assert not a._running
        http_mock.aclose.assert_called_once()


# ---------------------------------------------------------------------------
# In-chat slash commands
# ---------------------------------------------------------------------------


class TestTelegramCommands:
    @pytest.mark.asyncio
    async def test_new_command_resets_session(self, adapter, mock_http, reset_fn) -> None:
        await adapter._handle_update(_make_update(20, 42, 1, "/new"))
        reset_fn.assert_called_once_with("tg:42")
        assert any("new conversation" in t.lower() for t in _sent_texts(mock_http))

    @pytest.mark.asyncio
    async def test_reset_command_alias(self, adapter, mock_http, reset_fn) -> None:
        await adapter._handle_update(_make_update(21, 42, 1, "/reset"))
        reset_fn.assert_called_once_with("tg:42")

    @pytest.mark.asyncio
    async def test_help_command_lists_commands(self, adapter, mock_http) -> None:
        await adapter._handle_update(_make_update(22, 42, 1, "/help"))
        assert any("/stop" in t for t in _sent_texts(mock_http))

    @pytest.mark.asyncio
    async def test_unknown_command_reported(self, adapter, mock_http) -> None:
        await adapter._handle_update(_make_update(23, 42, 1, "/frobnicate"))
        assert any("Unknown command" in t for t in _sent_texts(mock_http))

    @pytest.mark.asyncio
    async def test_stop_with_nothing_running(self, adapter, mock_http) -> None:
        await adapter._handle_update(_make_update(24, 42, 1, "/stop"))
        assert any("Nothing is running" in t for t in _sent_texts(mock_http))

    @pytest.mark.asyncio
    async def test_stop_cancels_running_turn(self, mock_http) -> None:
        started = asyncio.Event()
        release = asyncio.Event()  # never set — the turn blocks here

        async def _blocking(_sid: str, _text: str) -> AsyncIterator[dict[str, Any]]:
            yield {
                "type": "tool_call",
                "tool": "execute_terminal_command",
                "params": {"command": "sleep 999"},
            }
            started.set()
            await release.wait()
            yield {"type": "text", "content": "never reached"}

        a = TelegramAdapter(token="t", stream_fn=_blocking, reset_fn=lambda _s: True)
        a._http = mock_http
        a._allowed = None

        turn = asyncio.create_task(a._handle_update(_make_update(25, 42, 1, "long task")))
        await asyncio.wait_for(started.wait(), timeout=2)  # turn is mid-stream
        await a._handle_command(42, "/stop")  # cancel out-of-band
        await asyncio.wait_for(turn, timeout=2)

        assert any("Stopped" in t for t in _sent_texts(mock_http))
        assert not any("never reached" in t for t in _sent_texts(mock_http))


# ---------------------------------------------------------------------------
# Voice transcription
# ---------------------------------------------------------------------------


def _make_voice_update(update_id: int, chat_id: int, user_id: int, file_id: str = "vf") -> dict:
    return {
        "update_id": update_id,
        "message": {
            "chat": {"id": chat_id},
            "from": {"id": user_id},
            "voice": {"file_id": file_id, "duration": 3},
        },
    }


class TestTelegramVoice:
    @pytest.mark.asyncio
    async def test_voice_disabled_without_model(self, adapter, mock_http) -> None:
        adapter._transcribe_model = ""
        await adapter._handle_update(_make_voice_update(30, 42, 1))
        assert any("aren't enabled" in t for t in _sent_texts(mock_http))
        assert _chat_actions(mock_http) == []  # no turn started

    @pytest.mark.asyncio
    async def test_voice_transcribed_and_processed(self, adapter, mock_http) -> None:
        adapter._transcribe_model = "whisper-1"
        adapter._transcribe = AsyncMock(return_value="hello world")
        await adapter._handle_update(_make_voice_update(31, 42, 1))
        texts = _sent_texts(mock_http)
        assert any('"hello world"' in t for t in texts)  # echoed transcript
        assert any("reply to: hello world" in t for t in texts)  # turn ran on it

    @pytest.mark.asyncio
    async def test_voice_empty_transcript_reported(self, adapter, mock_http) -> None:
        adapter._transcribe_model = "whisper-1"
        adapter._transcribe = AsyncMock(return_value="")
        await adapter._handle_update(_make_voice_update(32, 42, 1))
        assert any("Couldn't transcribe" in t for t in _sent_texts(mock_http))
        assert _chat_actions(mock_http) == []

    @pytest.mark.asyncio
    async def test_unauthorized_voice_silently_ignored(self, adapter, mock_http) -> None:
        adapter._allowed = frozenset({111})
        adapter._transcribe_model = "whisper-1"
        await adapter._handle_update(_make_voice_update(33, 42, 999))
        # No transcription attempted, no reply (don't nag on stranger voice spam).
        mock_http.post.assert_not_called()
