"""Tests for cross-session full-text memory (SessionStore)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from session_store import SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "sessions.db")


def test_add_and_search_finds_turn(store):
    store.add_turn("tg:1", "user", "Let's set up a Minecraft server on port 25565")
    store.add_turn("tg:1", "assistant", "Installed Java and started the server")
    hits = store.search("minecraft")
    assert any("Minecraft" in (h["snippet"] or h["content"]) for h in hits)
    assert hits[0]["session_id"] == "tg:1"


def test_search_across_sessions(store):
    store.add_turn("s1", "user", "remember my favorite color is teal")
    store.add_turn("s2", "user", "what database should I use")
    hits = store.search("teal")
    assert len(hits) == 1
    assert hits[0]["session_id"] == "s1"


def test_session_filter(store):
    store.add_turn("a", "user", "deploy the api gateway")
    store.add_turn("b", "user", "deploy the api gateway")
    assert len(store.search("deploy", session_id="a")) == 1


def test_empty_query_returns_nothing(store):
    store.add_turn("s", "user", "hello world")
    assert store.search("   ") == []


def test_blank_content_ignored(store):
    store.add_turn("s", "user", "   ")
    assert store.search("anything") == []


def test_special_chars_do_not_crash(store):
    store.add_turn("s", "user", "use rm -rf / carefully")
    # FTS5 special characters in the query must not raise.
    assert isinstance(store.search('rm -rf / "quoted" AND OR *'), list)


def test_recent_sessions(store):
    store.add_turn("s1", "user", "one")
    store.add_turn("s1", "assistant", "two")
    store.add_turn("s2", "user", "three")
    recent = store.recent_sessions()
    ids = {r["session_id"] for r in recent}
    assert ids == {"s1", "s2"}
    s1 = next(r for r in recent if r["session_id"] == "s1")
    assert s1["turns"] == 2


def test_limit_respected(store):
    for i in range(10):
        store.add_turn("s", "user", f"alpha message number {i}")
    assert len(store.search("alpha", limit=3)) == 3
