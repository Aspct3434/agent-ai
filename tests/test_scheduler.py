"""Unit tests for the cron parser in src/scheduler.py.

These cover the pure scheduling math (field matching, next-run computation,
validation) without needing the SQLite-backed CronScheduler or a live runner.
"""
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scheduler import (
    CronScheduler,
    ScheduledJob,
    _cron_next_run,
    _matches_field,
    _validate_schedule,
)


class TestMatchesField:
    def test_wildcard(self):
        assert _matches_field(5, "*")

    def test_exact(self):
        assert _matches_field(5, "5")
        assert not _matches_field(6, "5")

    def test_range(self):
        assert _matches_field(5, "1-10")
        assert not _matches_field(11, "1-10")

    def test_step(self):
        assert _matches_field(30, "*/30")
        assert not _matches_field(15, "*/30")

    def test_step_in_comma_list_does_not_short_circuit(self):
        # Regression: "*/30,45" used to return False for 45 because the step
        # term short-circuited the whole field before "45" was checked.
        assert _matches_field(45, "*/30,45")
        assert _matches_field(30, "*/30,45")
        assert not _matches_field(15, "*/30,45")

    def test_comma_list_of_exacts(self):
        assert _matches_field(0, "0,30")
        assert _matches_field(30, "0,30")
        assert not _matches_field(15, "0,30")

    def test_zero_step_does_not_crash(self):
        # "*/0" must not raise ZeroDivisionError.
        assert not _matches_field(5, "*/0")


class TestCronNextRun:
    # 2026-05-20 is a Wednesday.
    BASE = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)

    def test_day_of_week_sunday_is_zero(self):
        # Cron DOW 0 = Sunday (NOT Python's weekday 0 = Monday).
        nxt = _cron_next_run("0 0 * * 0", self.BASE)
        assert nxt.strftime("%A") == "Sunday"

    def test_day_of_week_seven_is_also_sunday(self):
        nxt = _cron_next_run("0 0 * * 7", self.BASE)
        assert nxt.strftime("%A") == "Sunday"

    def test_day_of_week_monday(self):
        nxt = _cron_next_run("0 0 * * 1", self.BASE)
        assert nxt.strftime("%A") == "Monday"

    def test_day_of_week_saturday(self):
        nxt = _cron_next_run("0 0 * * 6", self.BASE)
        assert nxt.strftime("%A") == "Saturday"

    def test_every_5_minutes(self):
        nxt = _cron_next_run("*/5 * * * *", datetime(2026, 5, 20, 12, 2, tzinfo=UTC))
        assert nxt == datetime(2026, 5, 20, 12, 5, tzinfo=UTC)

    def test_result_is_timezone_aware(self):
        assert _cron_next_run("* * * * *", self.BASE).tzinfo is not None

    def test_invalid_field_count(self):
        with pytest.raises(ValueError):
            _cron_next_run("* * *", self.BASE)


class TestComputeNextRun:
    def test_interval(self):
        job = ScheduledJob("id", "interval", "300", "p", "s")
        after = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
        assert job.compute_next_run(after) == datetime(2026, 5, 20, 12, 5, tzinfo=UTC)

    def test_once_in_future(self):
        job = ScheduledJob("id", "once", "2026-05-20T13:00:00+00:00", "p", "s")
        after = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
        assert job.compute_next_run(after) == datetime(2026, 5, 20, 13, 0, tzinfo=UTC)

    def test_once_in_past_returns_none(self):
        job = ScheduledJob("id", "once", "2020-01-01T00:00:00+00:00", "p", "s")
        after = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
        assert job.compute_next_run(after) is None

    def test_once_naive_treated_as_utc(self):
        job = ScheduledJob("id", "once", "2026-05-20T13:00:00", "p", "s")
        after = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
        nxt = job.compute_next_run(after)
        assert nxt is not None and nxt.tzinfo is not None


class TestRestoreOverdueJobs:
    """_load_jobs must not silently drop one-shot jobs missed during downtime."""

    async def _runner(self, session_id: str, prompt: str) -> str:
        return "ok"

    def _make_db(self, tmp_path) -> str:
        return str(tmp_path / "scheduler.db")

    def test_missed_one_shot_runs_on_restart(self, tmp_path):
        async def scenario() -> None:
            db = self._make_db(tmp_path)
            sched = CronScheduler(db, self._runner)
            await sched.start()
            await sched.shutdown()  # creates the table, stops the loop

            # A one-shot whose fire time elapsed while the server was down.
            job = ScheduledJob(
                "missed-1",
                "once",
                "2020-01-01T00:00:00+00:00",
                "ping",
                "s",
                next_run=datetime(2020, 1, 1, tzinfo=UTC),
                enabled=True,
            )
            await sched._persist(job)

            fresh = CronScheduler(db, self._runner)
            await fresh._load_jobs()
            restored = fresh._jobs["missed-1"]

            # Regression: previously next_run became None and the loop's
            # `j.next_run and ...` due-filter never selected it -> dropped.
            assert restored.enabled is True
            assert restored.next_run is not None
            assert restored.next_run <= datetime.now(UTC)

        asyncio.run(scenario())

    def test_overdue_interval_skips_missed_ticks(self, tmp_path):
        async def scenario() -> None:
            db = self._make_db(tmp_path)
            sched = CronScheduler(db, self._runner)
            await sched.start()
            await sched.shutdown()

            job = ScheduledJob(
                "interval-1",
                "interval",
                "300",
                "tick",
                "s",
                next_run=datetime(2020, 1, 1, tzinfo=UTC),
                enabled=True,
            )
            await sched._persist(job)

            fresh = CronScheduler(db, self._runner)
            await fresh._load_jobs()
            restored = fresh._jobs["interval-1"]

            # Overdue interval is rescheduled forward, not replayed from 2020.
            assert restored.next_run is not None
            assert restored.next_run > datetime.now(UTC)

        asyncio.run(scenario())


class TestValidateSchedule:
    def test_interval_minimum(self):
        with pytest.raises(ValueError):
            _validate_schedule("interval", "5")
        _validate_schedule("interval", "10")  # no raise

    def test_unknown_type(self):
        with pytest.raises(ValueError):
            _validate_schedule("hourly", "1")

    def test_cron_dry_run(self):
        _validate_schedule("cron", "*/5 * * * *")  # no raise
        with pytest.raises(ValueError):
            _validate_schedule("cron", "bad expr")
