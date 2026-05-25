"""Tests for cron-job result delivery to messaging targets."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scheduler import CronScheduler, ScheduledJob


@pytest.mark.asyncio
async def test_run_job_delivers_result_to_target(tmp_path) -> None:
    async def _runner(_session_id: str, prompt: str) -> str:
        return f"result for {prompt}"

    delivered: list[tuple[str, str]] = []

    async def _deliver(target: str, text: str) -> None:
        delivered.append((target, text))

    sched = CronScheduler(db_path=str(tmp_path / "cron.db"), runner=_runner)
    sched.set_delivery(_deliver)
    await sched.start()
    try:
        job = ScheduledJob(
            job_id="j1",
            schedule_type="interval",
            schedule_spec="3600",
            prompt="daily report",
            session_id="tg:42",
            deliver_to="tg:42",
        )
        await sched._run_job(job)
        assert delivered == [("tg:42", "result for daily report")]
    finally:
        await sched.shutdown()


@pytest.mark.asyncio
async def test_no_delivery_without_target(tmp_path) -> None:
    async def _runner(_s: str, _p: str) -> str:
        return "output"

    delivered: list[tuple[str, str]] = []

    async def _deliver(target: str, text: str) -> None:
        delivered.append((target, text))

    sched = CronScheduler(db_path=str(tmp_path / "cron.db"), runner=_runner)
    sched.set_delivery(_deliver)
    await sched.start()
    try:
        job = ScheduledJob(
            job_id="j2",
            schedule_type="interval",
            schedule_spec="3600",
            prompt="p",
            session_id="cron",
            deliver_to="",
        )
        await sched._run_job(job)
        assert delivered == []
    finally:
        await sched.shutdown()


@pytest.mark.asyncio
async def test_deliver_to_persists_across_reload(tmp_path) -> None:
    db = str(tmp_path / "cron.db")

    async def _runner(_s: str, _p: str) -> str:
        return "x"

    sched = CronScheduler(db_path=db, runner=_runner)
    await sched.start()
    try:
        job = await sched.add_job(
            schedule_type="interval",
            schedule_spec="3600",
            prompt="p",
            session_id="discord:9",
            deliver_to="discord:9",
        )
    finally:
        await sched.shutdown()

    # A fresh scheduler restores the job with deliver_to intact.
    reloaded = CronScheduler(db_path=db, runner=_runner)
    await reloaded.start()
    try:
        assert reloaded._jobs[job.job_id].deliver_to == "discord:9"
    finally:
        await reloaded.shutdown()


@pytest.mark.asyncio
async def test_delivery_failure_does_not_break_job(tmp_path) -> None:
    async def _runner(_s: str, _p: str) -> str:
        return "ok"

    async def _bad_deliver(_target: str, _text: str) -> None:
        raise ConnectionError("messaging down")

    sched = CronScheduler(db_path=str(tmp_path / "cron.db"), runner=_runner)
    sched.set_delivery(_bad_deliver)
    await sched.start()
    try:
        job = ScheduledJob(
            job_id="j3",
            schedule_type="interval",
            schedule_spec="3600",
            prompt="p",
            session_id="tg:1",
            deliver_to="tg:1",
        )
        await sched._run_job(job)  # must not raise
        assert job.last_result == "ok"
    finally:
        await sched.shutdown()
