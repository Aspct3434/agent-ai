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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Coroutine

import aiosqlite

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
            step = int(part[2:])
            return value % step == 0
        if "-" in part:
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
        if (
            _matches_field(dt.minute, min_f)
            and _matches_field(dt.hour, hour_f)
            and _matches_field(dt.day, dom_f)
            and _matches_field(dt.month, mon_f)
            and _matches_field(dt.weekday(), dow_f)
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
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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
            "created_at": self.created_at.isoformat(),
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
            "last_result": self.last_result,
            "enabled": self.enabled,
        }

    def compute_next_run(self, after: datetime | None = None) -> datetime | None:
        now = after or datetime.now(timezone.utc)
        if self.schedule_type == "interval":
            return now + timedelta(seconds=float(self.schedule_spec))
        if self.schedule_type == "cron":
            raw = _cron_next_run(self.schedule_spec, now)
            return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        if self.schedule_type == "once":
            dt = datetime.fromisoformat(self.schedule_spec)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt if dt > now else None
        return None


# Callable the scheduler uses to run an agent task: (session_id, prompt) -> result
AgentRunner = Callable[[str, str], Coroutine[Any, Any, str]]


class CronScheduler:
    """Asyncio-native recurring job scheduler backed by SQLite."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS scheduled_jobs (
        job_id  TEXT PRIMARY KEY,
        data    TEXT NOT NULL
    )
    """

    def __init__(self, db_path: str, runner: AgentRunner) -> None:
        self._db_path = db_path
        self._runner = runner
        self._jobs: dict[str, ScheduledJob] = {}
        self._task: asyncio.Task[None] | None = None
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(self._SCHEMA)
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
                now = datetime.now(timezone.utc)
                due = [
                    j for j in list(self._jobs.values())
                    if j.enabled and j.next_run and j.next_run <= now
                ]
                for job in due:
                    asyncio.create_task(
                        self._run_job(job),
                        name=f"cron:{job.job_id}",
                    )
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Cron scheduler loop error")

    async def _run_job(self, job: ScheduledJob) -> None:
        logger.info("Running job %s: %r", job.job_id, job.prompt[:80])
        now = datetime.now(timezone.utc)
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
        now = datetime.now(timezone.utc)
        for (data,) in rows:
            d = json.loads(data)
            job = ScheduledJob(
                job_id=d["job_id"],
                schedule_type=d["schedule_type"],
                schedule_spec=d["schedule_spec"],
                prompt=d["prompt"],
                session_id=d["session_id"],
                label=d.get("label", ""),
                created_at=datetime.fromisoformat(d["created_at"]),
                next_run=datetime.fromisoformat(d["next_run"]) if d.get("next_run") else None,
                last_run=datetime.fromisoformat(d["last_run"]) if d.get("last_run") else None,
                run_count=d.get("run_count", 0),
                last_result=d.get("last_result", ""),
                enabled=d.get("enabled", True),
            )
            # Recompute overdue next_run on restore
            if job.enabled and (job.next_run is None or job.next_run < now):
                job.next_run = job.compute_next_run()
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
        _cron_next_run(spec, datetime.now(timezone.utc))  # dry-run parse
    elif schedule_type == "once":
        datetime.fromisoformat(spec)
    else:
        raise ValueError(f"Unknown schedule_type {schedule_type!r}; use interval/cron/once")
