from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

DEFAULT_DB_PATH = Path("checkpoints.db")

# Keep at most this many checkpoints per session; older ones are pruned on each
# save so the table does not grow without bound over a long-running server.
_RETAIN_PER_SESSION = max(1, int(os.getenv("CHECKPOINT_RETAIN_PER_SESSION", "20")))


CREATE_STATE_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS state_snapshots (
    checkpoint_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    state_payload JSON NOT NULL CHECK (json_valid(state_payload)),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


class StateCheckpointer:
    """Persist and retrieve agent message-array snapshots via aiosqlite.

    Usage::

        checkpointer = StateCheckpointer("checkpoints.db")
        checkpoint_id = await checkpointer.save_checkpoint(
            session_id="abc", step_number=3, state_payload={"messages": [...]}
        )
        payload = await checkpointer.load_checkpoint(checkpoint_id)
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)

    async def save_checkpoint(
        self,
        session_id: str,
        step_number: int,
        state_payload: dict[str, Any],
    ) -> str:
        """Serialize *state_payload* and persist it; return the new checkpoint_id."""
        checkpoint_id = str(uuid.uuid4())
        serialized = json.dumps(state_payload, ensure_ascii=False)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO state_snapshots
                    (checkpoint_id, session_id, step_number, state_payload)
                VALUES (?, ?, ?, ?)
                """,
                (checkpoint_id, session_id, step_number, serialized),
            )
            # Bound the table: keep only the most recent N checkpoints per session.
            await db.execute(
                """
                DELETE FROM state_snapshots
                WHERE session_id = ?
                  AND checkpoint_id NOT IN (
                      SELECT checkpoint_id FROM state_snapshots
                      WHERE session_id = ?
                      ORDER BY step_number DESC, created_at DESC
                      LIMIT ?
                  )
                """,
                (session_id, session_id, _RETAIN_PER_SESSION),
            )
            await db.commit()

        return checkpoint_id

    async def load_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        """Return the deserialized payload for *checkpoint_id*.

        Raises ``KeyError`` if no matching checkpoint exists.
        """
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT state_payload FROM state_snapshots WHERE checkpoint_id = ?",
                (checkpoint_id,),
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            raise KeyError(f"Checkpoint not found: {checkpoint_id!r}")

        return dict(json.loads(row["state_payload"]))

    async def list_checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        """Return all saved checkpoint rows for *session_id* in step order."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT checkpoint_id, session_id, step_number, state_payload, created_at
                FROM state_snapshots
                WHERE session_id = ?
                ORDER BY step_number ASC, created_at ASC
                """,
                (session_id,),
            ) as cursor:
                rows = await cursor.fetchall()

        return [
            {
                "checkpoint_id": row["checkpoint_id"],
                "session_id": row["session_id"],
                "step_number": row["step_number"],
                "state_payload": json.loads(row["state_payload"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]


async def initialize_checkpoints_db(db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    """Create the async SQLite checkpoint database and required tables."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(path) as db:
        # WAL improves write throughput and read/write concurrency for the
        # frequent small writes this table receives. It is a persistent DB
        # property, so setting it once at init is sufficient.
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute(CREATE_STATE_SNAPSHOTS_SQL)
        await db.commit()

    return path


async def main() -> None:
    db_path = os.getenv("CHECKPOINT_DB_PATH", str(DEFAULT_DB_PATH))
    path = await initialize_checkpoints_db(db_path)
    print(f"Initialized checkpoint database: {path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
