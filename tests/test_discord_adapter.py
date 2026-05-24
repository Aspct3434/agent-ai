"""Unit tests for the Discord Gateway adapter.

All tests run without a live Discord connection — the send_fn and HTTP
client are fully mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
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
        # 1500 + "\n" + 1500 = 3001 chars; limit=2000 → first chunk ≤ 2000
        text = "a" * 1500 + "\n" + "b" * 1500
        chunks = _chunk_text(text, 2000)
        assert len(chunks) == 2
        assert all(len(c) <= 2000 for c in chunks)

    def test_hard_split_when_no_newline(self) -> None:
        text = "z" * 3000  # no newlines → exact hard split
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
# DiscordAdapter._handle_message (mocked send_fn + HTTP)
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_send_fn():
    async def _fn(session_id: str, text: str) -> str:
        return f"reply:{text}"

    return _fn


@pytest.fixture
def mock_http():
    http = AsyncMock()
    http.post = AsyncMock(return_value=MagicMock())
    http.aclose = AsyncMock()
    return http


@pytest.fixture
def adapter(echo_send_fn, mock_http):
    a = DiscordAdapter(token="Bot-test-token", send_fn=echo_send_fn)
    a._http = mock_http
    a._allowed = None
    return a


def _msg_posts(mock_http: AsyncMock) -> list:
    """Return only the HTTP post calls that carry a JSON body (message posts, not typing)."""
    return [c for c in mock_http.post.call_args_list if c.kwargs.get("json") is not None]


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


class TestDiscordMessageHandling:
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

    @pytest.mark.asyncio
    async def test_authorized_user_receives_reply(self, adapter, mock_http) -> None:
        adapter._allowed = frozenset({"user42"})
        await adapter._handle_message(_make_msg("hello", user_id="user42"))
        posts = _msg_posts(mock_http)
        assert len(posts) == 1
        assert "reply:hello" in posts[0].kwargs["json"]["content"]

    @pytest.mark.asyncio
    async def test_no_allowlist_allows_all_users(self, adapter, mock_http) -> None:
        adapter._allowed = None
        await adapter._handle_message(_make_msg("open access"))
        posts = _msg_posts(mock_http)
        assert len(posts) == 1

    @pytest.mark.asyncio
    async def test_session_id_scoped_to_channel(self, adapter) -> None:
        """The session_id forwarded to send_fn must be discord:{channel_id}."""
        captured: list[str] = []

        async def _capturing(_sid: str, _text: str) -> str:
            captured.append(_sid)
            return "ok"

        adapter._send_fn = _capturing
        await adapter._handle_message(_make_msg("test", channel_id="chan-xyz"))
        assert captured == ["discord:chan-xyz"]

    @pytest.mark.asyncio
    async def test_long_reply_chunked_into_multiple_posts(self, adapter, mock_http) -> None:
        """Replies > 2000 chars must be split across multiple message posts."""

        async def _long_reply(_sid: str, _text: str) -> str:
            return "A" * 5000  # → 3 chunks: 2000 + 2000 + 1000

        adapter._send_fn = _long_reply
        await adapter._handle_message(_make_msg("go"))
        posts = _msg_posts(mock_http)
        assert len(posts) == 3
        assert all(len(p.kwargs["json"]["content"]) <= 2000 for p in posts)

    @pytest.mark.asyncio
    async def test_send_fn_error_reported_to_channel(self, adapter, mock_http) -> None:
        """Exceptions from send_fn must be caught and sent as an error message."""

        async def _failing(_sid: str, _text: str) -> str:
            raise RuntimeError("Model offline")

        adapter._send_fn = _failing
        await adapter._handle_message(_make_msg("crash me"))
        posts = _msg_posts(mock_http)
        assert len(posts) == 1
        assert "Error" in posts[0].kwargs["json"]["content"]

    @pytest.mark.asyncio
    async def test_typing_indicator_sent_before_reply(self, adapter, mock_http) -> None:
        """The typing-indicator POST (no json body) must precede the message POST."""
        await adapter._handle_message(_make_msg("hi"))
        all_calls = mock_http.post.call_args_list
        # First call = typing (no json kwarg), second = message (has json)
        assert all_calls[0].kwargs.get("json") is None
        assert all_calls[1].kwargs.get("json") is not None

    @pytest.mark.asyncio
    async def test_typing_failure_does_not_abort_reply(self, mock_http) -> None:
        """If the typing-indicator POST fails, the reply is still sent."""
        call_count = 0

        async def _flaky_post(*args: object, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("typing endpoint down")
            return MagicMock()

        mock_http.post.side_effect = _flaky_post

        async def _send(_sid: str, _text: str) -> str:
            return "ok"

        a = DiscordAdapter(token="t", send_fn=_send)
        a._http = mock_http
        a._allowed = None
        await a._handle_message(_make_msg("hello"))
        # Despite the typing failure, the message post still happened.
        assert call_count == 2
