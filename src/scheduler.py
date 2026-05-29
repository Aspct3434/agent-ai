"""Async task scheduler: recurring and one-shot agent jobs.

Supports three schedule types:
    interval  — every N seconds  (spec: "300" = every 5 min)
    cron      — standard 5-field cron expression  (spec: "*/5 * * * *")
    once      — run at a specific UTC datetime  (spec: ISO 8601 string)

Jobs persist to SQLite and are restored on startup with corrected next-run
times. The scheduler polls at 10-second granularity — fine for all interval
and cron use cases (minimum practical granularity is 1 minute for cron).
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite

from sqlite_migrations import SQLiteMigration, apply_async_sqlite_migrations

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Minimal cron expression parser (5-field: min hour dom month dow)
# Supports: *, N, */N, N-M, N,M
# ------------------------------------------------------------------


def _matches_field(value: int, spec: str) -> bool:
    for part in spec.split(","):
        part = part.strip()
        if part == "*":
            return True
        if part.startswith("*/"):
            # A step term in a comma list must not short-circuit the whole
            # field: "*/30,45" has to keep checking "45" when value % 30 != 0.
            step = int(part[2:])
            if step > 0 and value % step == 0:
                return True
        elif "-" in part:
            a, b = part.split("-", 1)
            if int(a) <= value <= int(b):
                return True
        elif int(part) == value:
            return True
    return False


def _cron_next_run(expr: str, after: datetime) -> datetime:
    """Return the next UTC datetime matching a 5-field cron expression."""
    parts = expr.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Cron expression must have 5 fields, got: {expr!r}")
    min_f, hour_f, dom_f, mon_f, dow_f = parts
    dt = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(366 * 24 * 60):
        # Cron day-of-week is 0=Sunday..6=Saturday (with 7 also = Sunday),
        # whereas Python's weekday() is 0=Monday..6=Sunday.
        cron_dow = (dt.weekday() + 1) % 7
        if (
            _matches_field(dt.minute, min_f)
            and _matches_field(dt.hour, hour_f)
            and _matches_field(dt.day, dom_f)
            and _matches_field(dt.month, mon_f)
            and (
                _matches_field(cron_dow, dow_f)
                or (cron_dow == 0 and _matches_field(7, dow_f))
            )
        ):
            return dt
        dt += timedelta(minutes=1)
    raise ValueError(f"Could not find next run for cron {expr!r} within 1 year")


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------


@dataclass
class ScheduledJob:
    job_id: str
    schedule_type: str          # "interval" | "cron" | "once"
    schedule_spec: str          # seconds | cron expr | ISO datetime
    prompt: str
    session_id: str
    label: str = ""
    # Optional messaging target for the result, e.g. "tg:12345",
    # "discord:67890", "slack:C123". Empty = no delivery.
    deliver_to: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    next_run: datetime | None = None
    last_run: datetime | None = None
    run_count: int = 0
    last_result: str = ""
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "schedule_type": self.schedule_type,
            "schedule_spec": self.schedule_spec,
            "prompt": self.prompt,
            "session_id": self.session_id,
            "label": self.label,
            "deliver_to": self.deliver_to,
            "created_at": self.created_at.isoformat(),
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
            "last_result": self.last_result,
            "enabled": self.enabled,
        }

    def compute_next_run(self, after: datetime | None = None) -> datetime | None:
        now = after or datetime.now(UTC)
        if self.schedule_type == "interval":
            return now + timedelta(seconds=float(self.schedule_spec))
        if self.schedule_type == "cron":
            raw = _cron_next_run(self.schedule_spec, now)
            return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
        if self.schedule_type == "once":
            dt = datetime.fromisoformat(self.schedule_spec)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt if dt > now else None
        return None


# Callable the scheduler uses to run an agent task: (session_id, prompt) -> result
AgentRunner = Callable[[str, str], Coroutine[Any, Any, str]]
# Callable that delivers a job's result to a messaging target: (target, text) -> None
DeliveryFn = Callable[[str, str], Coroutine[Any, Any, None]]


class CronScheduler:
    """Asyncio-native recurring job scheduler backed by SQLite."""

    _MIGRATIONS = (
        SQLiteMigration(
            version=1,
            name="create_scheduled_jobs",
            statements=(
                """
                CREATE TABLE IF NOT EXISTS scheduled_jobs (
                    job_id  TEXT PRIMARY KEY,
                    data    TEXT NOT NULL
                )
                """,
            ),
        ),
    )

    def __init__(self, db_path: str, runner: AgentRunner) -> None:
        self._db_path = db_path
        self._runner = runner
        self._deliver: DeliveryFn | None = None
        self._jobs: dict[str, ScheduledJob] = {}
        self._task: asyncio.Task[None] | None = None
        self._running_tasks: set[asyncio.Task[None]] = set()
        self._started = False

    def set_delivery(self, deliver: DeliveryFn) -> None:
        """Register the callback used to deliver job results to messaging targets."""
        self._deliver = deliver

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await apply_async_sqlite_migrations(
                db, "scheduler", self._MIGRATIONS
            )
            await db.commit()
        await self._load_jobs()
        self._task = asyncio.create_task(self._loop(), name="cron-scheduler")
        self._started = True
        logger.info("CronScheduler started with %d jobs", len(self._jobs))

    async def shutdown(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        for task in list(self._running_tasks):
            task.cancel()
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)
        self._started = False

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    async def add_job(
        self,
        *,
        schedule_type: str,
        schedule_spec: str,
        prompt: str,
        session_id: str,
        label: str = "",
        deliver_to: str = "",
        job_id: str | None = None,
    ) -> ScheduledJob:
        """Create, persist, and return a new scheduled job."""
        _validate_schedule(schedule_type, schedule_spec)
        job = ScheduledJob(
            job_id=job_id or str(uuid.uuid4()),
            schedule_type=schedule_type,
            schedule_spec=schedule_spec,
            prompt=prompt,
            session_id=session_id,
            label=label or prompt[:60],
            deliver_to=deliver_to,
        )
        job.next_run = job.compute_next_run()
        self._jobs[job.job_id] = job
        await self._persist(job)
        logger.info(
            "Scheduled job created: %s [%s %s] next=%s",
            job.job_id,
            schedule_type,
            schedule_spec,
            job.next_run,
        )
        return job

    async def remove_job(self, job_id: str) -> bool:
        if job_id not in self._jobs:
            return False
        del self._jobs[job_id]
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM scheduled_jobs WHERE job_id = ?", (job_id,))
            await db.commit()
        return True

    async def toggle_job(self, job_id: str, *, enabled: bool) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        job.enabled = enabled
        if enabled and job.next_run is None:
            job.next_run = job.compute_next_run()
        await self._persist(job)
        return True

    def list_jobs(self) -> list[dict[str, Any]]:
        return [j.to_dict() for j in self._jobs.values()]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(10)
                now = datetime.now(UTC)
                due = [
                    j for j in list(self._jobs.values())
                    if j.enabled and j.next_run and j.next_run <= now
                ]
                for job in due:
                    task = asyncio.create_task(
                        self._run_job(job),
                        name=f"cron:{job.job_id}",
                    )
                    self._running_tasks.add(task)
                    task.add_done_callback(self._running_tasks.discard)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Cron scheduler loop error")

    async def _run_job(self, job: ScheduledJob) -> None:
        logger.info("Running job %s: %r", job.job_id, job.prompt[:80])
        now = datetime.now(UTC)
        job.last_run = now
        job.run_count += 1
        next_run = job.compute_next_run(now)
        if next_run is None:
            job.enabled = False
            logger.info("One-shot job %s completed and disabled", job.job_id)
        else:
            job.next_run = next_run
        await self._persist(job)

        try:
            result = await self._runner(job.session_id, job.prompt)
            job.last_result = (result or "")[:500]
            # Deliver the full result to the configured messaging target.
            if job.deliver_to and self._deliver and result:
                try:
                    await self._deliver(job.deliver_to, result)
                except Exception:
                    logger.exception("Cron delivery to %s failed", job.deliver_to)
        except Exception as exc:
            job.last_result = f"ERROR: {exc}"
            logger.exception("Scheduled job %s failed", job.job_id)
        finally:
            await self._persist(job)

    async def _persist(self, job: ScheduledJob) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO scheduled_jobs (job_id, data) VALUES (?, ?)",
                (job.job_id, json.dumps(job.to_dict())),
            )
            await db.commit()

    async def _load_jobs(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute("SELECT data FROM scheduled_jobs")
            rows = await cur.fetchall()
        now = datetime.now(UTC)
        for (data,) in rows:
            d = json.loads(data)
            job = ScheduledJob(
                job_id=d["job_id"],
                schedule_type=d["schedule_type"],
                schedule_spec=d["schedule_spec"],
                prompt=d["prompt"],
                session_id=d["session_id"],
                label=d.get("label", ""),
                deliver_to=d.get("deliver_to", ""),
                created_at=datetime.fromisoformat(d["created_at"]),
                next_run=datetime.fromisoformat(d["next_run"]) if d.get("next_run") else None,
                last_run=datetime.fromisoformat(d["last_run"]) if d.get("last_run") else None,
                run_count=d.get("run_count", 0),
                last_result=d.get("last_result", ""),
                enabled=d.get("enabled", True),
            )
            # Recompute overdue next_run on restore so we don't replay every
            # interval/cron tick the server slept through. A one-shot job whose
            # fire time elapsed during downtime has compute_next_run() == None;
            # without the fallback below it would be silently dropped (left
            # enabled with next_run=None, which the loop never selects). Run it
            # immediately instead so missed reminders survive a restart.
            if job.enabled and (job.next_run is None or job.next_run < now):
                recomputed = job.compute_next_run()
                if recomputed is None and job.schedule_type == "once":
                    recomputed = now
                job.next_run = recomputed
            self._jobs[job.job_id] = job


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


def _validate_schedule(schedule_type: str, spec: str) -> None:
    if schedule_type == "interval":
        secs = float(spec)
        if secs < 10:
            raise ValueError("Interval must be at least 10 seconds")
    elif schedule_type == "cron":
        _cron_next_run(spec, datetime.now(UTC))  # dry-run parse
    elif schedule_type == "once":
        datetime.fromisoformat(spec)
    else:
        raise ValueError(f"Unknown schedule_type {schedule_type!r}; use interval/cron/once")
