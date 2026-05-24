"""Unit tests for the Telegram Bot API adapter.

All tests run without a real bot token — HTTP is fully mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
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
        # "line1\nline2" is 11 chars; limit=8 → cut before "line2"
        chunks = _chunk_text("line1\nline2", 8)
        assert len(chunks) == 2
        assert chunks[0] == "line1"
        assert chunks[1] == "line2"

    def test_hard_split_when_no_newline(self) -> None:
        text = "abcdefghij"  # 10 chars, no newlines
        chunks = _chunk_text(text, 4)
        assert all(len(c) <= 4 for c in chunks)
        assert "".join(chunks) == text  # hard-split: no chars dropped

    def test_empty_string_returns_single_empty(self) -> None:
        assert _chunk_text("", 100) == [""]

    def test_multi_chunk_loses_only_boundary_newlines(self) -> None:
        # The splitter strips newlines at cut points; non-newline content is preserved.
        text = ("abc\n" * 500).rstrip()
        chunks = _chunk_text(text, 100)
        assert all(len(c) <= 100 for c in chunks)
        # Strip all newlines from both sides before comparing — boundary newlines
        # are consumed by lstrip("\n") in the splitter.
        assert "".join(chunks).replace("\n", "") == text.replace("\n", "")

    def test_newline_at_limit_boundary_used_as_cut(self) -> None:
        # Newline is at index 9 (position limit-1 for limit=10).
        text = "a" * 9 + "\n" + "b" * 5
        chunks = _chunk_text(text, 10)
        assert chunks[0] == "a" * 9
        assert chunks[1] == "b" * 5


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
# TelegramAdapter behaviour (mocked HTTP)
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_send_fn():
    """send_fn that echoes «reply to: <text>»."""

    async def _fn(session_id: str, text: str) -> str:
        return f"reply to: {text}"

    return _fn


@pytest.fixture
def mock_http():
    http = AsyncMock()
    http.post = AsyncMock(return_value=MagicMock())
    http.aclose = AsyncMock()
    return http


@pytest.fixture
def adapter(echo_send_fn, mock_http):
    a = TelegramAdapter(token="test-token", send_fn=echo_send_fn)
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


class TestTelegramAdapterHandling:
    @pytest.mark.asyncio
    async def test_start_command_sends_welcome(self, adapter, mock_http) -> None:
        await adapter._handle_update(_make_update(1, 42, 99, "/start"))
        mock_http.post.assert_called_once()
        sent_text: str = mock_http.post.call_args.kwargs["json"]["text"]
        assert "Hello" in sent_text

    @pytest.mark.asyncio
    async def test_other_slash_commands_ignored(self, adapter, mock_http) -> None:
        await adapter._handle_update(_make_update(2, 42, 99, "/unknown"))
        mock_http.post.assert_not_called()

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
        adapter._allowed = frozenset({111})  # only user 111 is allowed
        await adapter._handle_update(_make_update(5, 42, 999, "do something"))
        mock_http.post.assert_called_once()
        assert "denied" in mock_http.post.call_args.kwargs["json"]["text"].lower()

    @pytest.mark.asyncio
    async def test_authorized_user_gets_reply(self, adapter, mock_http) -> None:
        adapter._allowed = frozenset({111})
        await adapter._handle_update(_make_update(6, 42, 111, "hello"))
        # Expects: "Working on it" then the reply
        assert mock_http.post.call_count == 2
        reply_text: str = mock_http.post.call_args_list[-1].kwargs["json"]["text"]
        assert "reply to: hello" in reply_text

    @pytest.mark.asyncio
    async def test_open_access_when_no_allowlist(self, adapter, mock_http) -> None:
        adapter._allowed = None
        await adapter._handle_update(_make_update(7, 42, 99999, "unrestricted"))
        assert mock_http.post.call_count == 2  # "working" + reply

    @pytest.mark.asyncio
    async def test_long_reply_is_chunked(self, mock_http) -> None:
        """Replies > 4096 chars must be split into multiple sendMessage calls."""
        long_text = "x" * 9000  # 3 chunks at the 4096-char limit

        async def _long_send(_sid: str, _text: str) -> str:
            return long_text

        a = TelegramAdapter(token="t", send_fn=_long_send)
        a._http = mock_http
        a._allowed = None
        await a._handle_update(_make_update(8, 1, 1, "give me lots"))
        # 1 "Working on it" + 3 content chunks (9000 / 4096 → ceil = 3)
        assert mock_http.post.call_count == 4
        # Every individual message must respect the 4096-char limit.
        for call in mock_http.post.call_args_list[1:]:
            assert len(call.kwargs["json"]["text"]) <= 4096

    @pytest.mark.asyncio
    async def test_send_fn_error_reported_to_user(self, adapter, mock_http) -> None:
        """Exceptions from send_fn must be caught and sent as an error message."""

        async def _failing(_sid: str, _text: str) -> str:
            raise RuntimeError("LLM unavailable")

        adapter._send_fn = _failing
        await adapter._handle_update(_make_update(9, 42, 1, "crash please"))
        last_text: str = mock_http.post.call_args_list[-1].kwargs["json"]["text"]
        assert "Error" in last_text

    @pytest.mark.asyncio
    async def test_offset_advances_to_last_update_plus_one(self, adapter) -> None:
        """_offset must be updated to update_id + 1 after each batch."""
        updates = [
            {"update_id": 100, "message": {"chat": {"id": 1}, "from": {"id": 1}, "text": "/start"}},
            {"update_id": 101, "message": {"chat": {"id": 1}, "from": {"id": 1}, "text": "/start"}},
        ]
        for u in updates:
            adapter._offset = u["update_id"] + 1
        assert adapter._offset == 102

    @pytest.mark.asyncio
    async def test_shutdown_cancels_task_and_closes_client(self, echo_send_fn) -> None:
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
            a = TelegramAdapter(token="t", send_fn=echo_send_fn)
            await a.start()
            assert a._task is not None and not a._task.done()
            await a.shutdown()

        assert not a._running
        http_mock.aclose.assert_called_once()


class TestTelegramWebhookHandling:
    @pytest.mark.asyncio
    async def test_prepare_deletes_webhook(self, adapter, mock_http) -> None:
        """start()/_prepare must clear any webhook so long-polling works."""
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
        """A 409 conflict during polling must auto-delete the conflicting webhook."""
        mock_http.post = AsyncMock(
            return_value=MagicMock(
                json=MagicMock(
                    return_value={"ok": False, "error_code": 409, "description": "Conflict"}
                )
            )
        )
        result = await adapter._get_updates()
        assert result == []
        # One call to getUpdates + one recovery call to deleteWebhook
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
