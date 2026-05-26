"""Cross-session conversation memory with full-text search.

Every user/assistant turn is appended to a SQLite store so the agent can
recall what was discussed in *past* sessions — the "agent that grows with you"
capability. Uses FTS5 when the runtime's sqlite is compiled with it, and falls
back to a plain table + LIKE search otherwise.

Synchronous on purpose (one short-lived connection per call, thread-safe);
callers wrap it in ``asyncio.to_thread`` so it never blocks the event loop.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_CONTENT = 8000


class SessionStore:
    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        self._fts = self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _init_schema(self) -> bool:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connect() as c:
                c.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS turns USING fts5("
                    "session_id UNINDEXED, role UNINDEXED, ts UNINDEXED, content)"
                )
            return True
        except sqlite3.OperationalError as exc:
            logger.warning("SQLite FTS5 unavailable (%s); using LIKE search.", exc)
            with self._connect() as c:
                c.execute(
                    "CREATE TABLE IF NOT EXISTS turns "
                    "(session_id TEXT, role TEXT, ts REAL, content TEXT)"
                )
                c.execute("CREATE INDEX IF NOT EXISTS idx_turns_ts ON turns(ts)")
            return False

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        text = (content or "").strip()
        if not text:
            return
        try:
            with self._connect() as c:
                c.execute(
                    "INSERT INTO turns (session_id, role, ts, content) VALUES (?, ?, ?, ?)",
                    (session_id, role, time.time(), text[:_MAX_CONTENT]),
                )
        except sqlite3.Error as exc:
            logger.debug("session add_turn failed: %s", exc)

    @staticmethod
    def _fts_query(query: str) -> str:
        # Quote each token so user text can't break FTS5 query syntax.
        tokens = [t for t in re.split(r"\s+", query) if t]
        return " OR ".join(f'"{t}"' for t in tokens) or '""'

    def search(
        self, query: str, limit: int = 8, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        try:
            with self._connect() as c:
                c.row_factory = sqlite3.Row
                if self._fts:
                    sql = (
                        "SELECT session_id, role, ts, content, "
                        "snippet(turns, 3, '«', '»', '…', 14) AS snippet "
                        "FROM turns WHERE turns MATCH ?"
                    )
                    params: list[Any] = [self._fts_query(q)]
                    if session_id:
                        sql += " AND session_id = ?"
                        params.append(session_id)
                    sql += " ORDER BY rank LIMIT ?"
                    params.append(limit)
                else:
                    sql = (
                        "SELECT session_id, role, ts, content, content AS snippet "
                        "FROM turns WHERE content LIKE ?"
                    )
                    params = [f"%{q}%"]
                    if session_id:
                        sql += " AND session_id = ?"
                        params.append(session_id)
                    sql += " ORDER BY ts DESC LIMIT ?"
                    params.append(limit)
                return [dict(r) for r in c.execute(sql, params).fetchall()]
        except sqlite3.Error as exc:
            logger.warning("session search failed: %s", exc)
            return []

    def recent_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        try:
            with self._connect() as c:
                c.row_factory = sqlite3.Row
                rows = c.execute(
                    "SELECT session_id, COUNT(*) AS turns, MAX(ts) AS last_ts "
                    "FROM turns GROUP BY session_id ORDER BY last_ts DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error:
            return []

    def close(self) -> None:
        """No persistent connection to close (one per call)."""
