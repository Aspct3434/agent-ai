"""Unit tests for the Email (IMAP poll + SMTP reply) channel adapter.

All tests run without a live mailbox — imaplib/smtplib and the stream_fn are
fully mocked, and the blocking IMAP/SMTP work (normally pushed to worker
threads via ``asyncio.to_thread``) runs inline against fakes.
"""
from __future__ import annotations

import email
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapters.email_adapter import (
    EmailAdapter,
    _build_reply,
    _extract_body,
    _parse_allowlist,
    _strip_quoted,
)

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestParseAllowlist:
    def test_empty_is_none(self) -> None:
        assert _parse_allowlist("   ") is None

    def test_lowercases_and_splits(self) -> None:
        assert _parse_allowlist("A@x.com, B@Y.com") == frozenset(
            {"a@x.com", "b@y.com"}
        )

    def test_drops_blanks(self) -> None:
        assert _parse_allowlist("a@x.com, ,") == frozenset({"a@x.com"})


class TestExtractBody:
    def test_plain_text(self) -> None:
        msg = email.message_from_string(
            "Subject: hi\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nhello world"
        )
        assert _extract_body(msg).strip() == "hello world"

    def test_multipart_prefers_text_plain(self) -> None:
        raw = (
            "Subject: hi\r\n"
            'Content-Type: multipart/alternative; boundary="B"\r\n\r\n'
            "--B\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n"
            "plain body\r\n"
            "--B\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
            "<p>html body</p>\r\n"
            "--B--\r\n"
        )
        msg = email.message_from_string(raw)
        assert "plain body" in _extract_body(msg)
        assert "html body" not in _extract_body(msg)


class TestStripQuoted:
    def test_drops_gt_quote(self) -> None:
        body = "my reply\n> old line\n> another"
        assert _strip_quoted(body) == "my reply"

    def test_drops_on_wrote_attribution(self) -> None:
        body = "thanks!\nOn Mon, X <x@y.com> wrote:\nold stuff"
        assert _strip_quoted(body) == "thanks!"

    def test_keeps_plain_body(self) -> None:
        assert _strip_quoted("just one line") == "just one line"


class TestBuildReply:
    def test_adds_re_prefix(self) -> None:
        reply = _build_reply("to@x.com", "me@x.com", "Question", "body")
        assert reply["Subject"] == "Re: Question"
        assert reply["To"] == "to@x.com"
        assert reply["From"] == "me@x.com"
        assert reply.get_content().strip() == "body"

    def test_does_not_double_prefix(self) -> None:
        reply = _build_reply("to@x.com", "me@x.com", "Re: Question", "body")
        assert reply["Subject"] == "Re: Question"


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
        yield {"type": "final_answer", "content": f"reply:{text}"}

    return _fn


def _make_adapter(stream_fn, *, allowed=None) -> EmailAdapter:
    a = EmailAdapter(
        imap_host="imap.x.com",
        smtp_host="smtp.x.com",
        address="me@x.com",
        password="pw",
        stream_fn=stream_fn,
        reset_fn=MagicMock(return_value=True),
    )
    a._allowed = allowed
    return a


# ---------------------------------------------------------------------------
# _handle_email: gating + session scoping + reply assembly
# ---------------------------------------------------------------------------


class TestHandleEmail:
    @pytest.mark.asyncio
    async def test_replies_with_final_answer(self, echo_stream_fn) -> None:
        adapter = _make_adapter(echo_stream_fn)
        sent: list[tuple[str, str, str]] = []
        adapter._send = lambda to, subj, body: sent.append((to, subj, body))
        await adapter._handle_email(
            {"from": "Bob@X.com", "subject": "Hi", "body": "hello"}
        )
        assert len(sent) == 1
        to, _subj, body = sent[0]
        assert to == "bob@x.com"
        assert "reply:hello" in body

    @pytest.mark.asyncio
    async def test_session_scoped_to_sender(self) -> None:
        captured: list[str] = []

        async def _cap(sid: str, _text: str) -> AsyncIterator[dict[str, Any]]:
            captured.append(sid)
            yield {"type": "final_answer", "content": "ok"}

        adapter = _make_adapter(_cap)
        adapter._send = lambda *a: None
        await adapter._handle_email(
            {"from": "Person@Host.com", "subject": "s", "body": "hi"}
        )
        assert captured == ["email:person@host.com"]

    @pytest.mark.asyncio
    async def test_allowlist_blocks_stranger(self, echo_stream_fn) -> None:
        adapter = _make_adapter(echo_stream_fn, allowed=frozenset({"friend@x.com"}))
        sent: list[Any] = []
        adapter._send = lambda *a: sent.append(a)
        await adapter._handle_email(
            {"from": "stranger@x.com", "subject": "s", "body": "hi"}
        )
        assert sent == []

    @pytest.mark.asyncio
    async def test_allowlist_permits_member(self, echo_stream_fn) -> None:
        adapter = _make_adapter(echo_stream_fn, allowed=frozenset({"friend@x.com"}))
        sent: list[Any] = []
        adapter._send = lambda *a: sent.append(a)
        await adapter._handle_email(
            {"from": "Friend@x.com", "subject": "s", "body": "hi"}
        )
        assert len(sent) == 1

    @pytest.mark.asyncio
    async def test_empty_body_ignored(self, echo_stream_fn) -> None:
        adapter = _make_adapter(echo_stream_fn)
        sent: list[Any] = []
        adapter._send = lambda *a: sent.append(a)
        await adapter._handle_email({"from": "bob@x.com", "subject": "s", "body": "  "})
        assert sent == []

    @pytest.mark.asyncio
    async def test_tool_steps_appended(self) -> None:
        stream = _stream_of(
            {"type": "tool_call", "tool": "set_task_contract", "params": {}},  # silent
            {"type": "tool_call", "tool": "web_search", "params": {"query": "rust book"}},
            {"type": "final_answer", "content": "done"},
        )
        adapter = _make_adapter(stream)
        sent: list[tuple[str, str, str]] = []
        adapter._send = lambda to, subj, body: sent.append((to, subj, body))
        await adapter._handle_email({"from": "bob@x.com", "subject": "s", "body": "go"})
        _to, _subj, body = sent[0]
        assert "done" in body
        assert "rust book" in body
        assert "set_task_contract" not in body

    @pytest.mark.asyncio
    async def test_stream_error_is_reported(self) -> None:
        async def _boom(_sid: str, _text: str) -> AsyncIterator[dict[str, Any]]:
            raise RuntimeError("kaboom")
            yield  # pragma: no cover

        adapter = _make_adapter(_boom)
        sent: list[tuple[str, str, str]] = []
        adapter._send = lambda to, subj, body: sent.append((to, subj, body))
        await adapter._handle_email({"from": "bob@x.com", "subject": "s", "body": "go"})
        _to, _subj, body = sent[0]
        assert "went wrong" in body.lower()


# ---------------------------------------------------------------------------
# _fetch_unseen: IMAP parsing against a fake imaplib
# ---------------------------------------------------------------------------


class _FakeIMAP:
    def __init__(self, raw_messages: list[bytes]) -> None:
        self._raw = raw_messages

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, *a):
        return ("OK", [])

    def select(self, *a):
        return ("OK", [])

    def search(self, _charset, _criteria):
        ids = b" ".join(str(i).encode() for i in range(1, len(self._raw) + 1))
        return ("OK", [ids])

    def fetch(self, num, _spec):
        idx = int(num) - 1
        return ("OK", [(b"1 (RFC822)", self._raw[idx])])


class TestFetchUnseen:
    def test_parses_unseen_messages(self, monkeypatch, echo_stream_fn) -> None:
        raw = (
            b"From: Alice <alice@x.com>\r\n"
            b"Subject: Test\r\n"
            b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
            b"the body\r\n> quoted\r\n"
        )
        adapter = _make_adapter(echo_stream_fn)
        monkeypatch.setattr(
            "adapters.email_adapter.imaplib.IMAP4_SSL",
            lambda *a, **k: _FakeIMAP([raw]),
        )
        out = adapter._fetch_unseen()
        assert len(out) == 1
        assert out[0]["from"] == "alice@x.com"
        assert out[0]["subject"] == "Test"
        assert out[0]["body"] == "the body"
