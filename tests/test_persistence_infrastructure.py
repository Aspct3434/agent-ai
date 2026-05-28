from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gateway
from checkpointer import initialize_checkpoints_db
from scheduler import CronScheduler
from session_store import SessionStore


def _migration_names(db_path: Path) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {
            str(row[0])
            for row in conn.execute(
                "SELECT namespace FROM schema_migrations"
            ).fetchall()
        }


def test_session_store_records_schema_migration_and_preserves_data(tmp_path: Path):
    db = tmp_path / "sessions.db"
    store = SessionStore(db)
    store.add_turn("s1", "user", "persistent migration marker")

    namespaces = _migration_names(db)
    assert namespaces & {"session_store_fts", "session_store_plain"}

    reloaded = SessionStore(db)
    assert reloaded.search("persistent")


@pytest.mark.asyncio
async def test_checkpointer_records_schema_migration(tmp_path: Path):
    db = tmp_path / "checkpoints.db"
    await initialize_checkpoints_db(db)
    assert "checkpointer" in _migration_names(db)


@pytest.mark.asyncio
async def test_scheduler_records_schema_migration(tmp_path: Path):
    async def runner(_session_id: str, _prompt: str) -> str:
        return "ok"

    db = tmp_path / "cron.db"
    scheduler = CronScheduler(str(db), runner)
    await scheduler.start()
    await scheduler.shutdown()
    assert "scheduler" in _migration_names(db)


def test_gateway_logs_are_persisted_across_handler_reopen(tmp_path: Path):
    db = tmp_path / "gateway_logs.db"
    record = logging.LogRecord(
        name="test.gateway",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="durable log line",
        args=(),
        exc_info=None,
    )

    store = gateway._SQLiteLogStore(db, capacity=10)
    store.emit(record)
    store.close()

    reopened = gateway._SQLiteLogStore(db, capacity=10)
    try:
        logs = reopened.snapshot(level="ERROR", limit=10)
    finally:
        reopened.close()

    assert logs[-1]["message"] == "durable log line"
    assert "gateway_logs" in _migration_names(db)
