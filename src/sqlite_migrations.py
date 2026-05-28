from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SQLiteMigration:
    version: int
    name: str
    statements: tuple[str, ...]


_CREATE_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    namespace TEXT NOT NULL,
    version INTEGER NOT NULL,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (namespace, version)
)
"""


def apply_sqlite_migrations(
    conn: Any,
    namespace: str,
    migrations: list[SQLiteMigration] | tuple[SQLiteMigration, ...],
) -> None:
    """Apply pending SQLite migrations on a synchronous sqlite3 connection."""
    conn.execute(_CREATE_MIGRATIONS_TABLE)
    applied = {
        int(row[0])
        for row in conn.execute(
            "SELECT version FROM schema_migrations WHERE namespace = ?",
            (namespace,),
        ).fetchall()
    }
    for migration in sorted(migrations, key=lambda item: item.version):
        if migration.version in applied:
            continue
        for statement in migration.statements:
            conn.execute(statement)
        conn.execute(
            """
            INSERT INTO schema_migrations (namespace, version, name)
            VALUES (?, ?, ?)
            """,
            (namespace, migration.version, migration.name),
        )


async def apply_async_sqlite_migrations(
    db: Any,
    namespace: str,
    migrations: list[SQLiteMigration] | tuple[SQLiteMigration, ...],
) -> None:
    """Apply pending SQLite migrations on an aiosqlite connection."""
    await db.execute(_CREATE_MIGRATIONS_TABLE)
    async with db.execute(
        "SELECT version FROM schema_migrations WHERE namespace = ?",
        (namespace,),
    ) as cursor:
        rows = await cursor.fetchall()
    applied = {int(row[0]) for row in rows}
    for migration in sorted(migrations, key=lambda item: item.version):
        if migration.version in applied:
            continue
        for statement in migration.statements:
            await db.execute(statement)
        await db.execute(
            """
            INSERT INTO schema_migrations (namespace, version, name)
            VALUES (?, ?, ?)
            """,
            (namespace, migration.version, migration.name),
        )
