"""Unit tests for Docker sandbox self-healing.

A sandbox container running with ``--rm`` is removed when Docker restarts,
the host sleeps, or it is stopped manually. The cached container id then
points at nothing and every ``docker exec`` fails with "No such container".
These tests verify the sandbox detects that and transparently rebuilds the
container instead of staying permanently broken. No real Docker is used —
``subprocess.run`` is mocked throughout.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools import _DockerSandbox, _is_missing_container_error


def _completed(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _sandbox() -> _DockerSandbox:
    return _DockerSandbox(image="python:3.12-slim", host_workdir="/tmp")


class TestMissingContainerDetection:
    def test_no_such_container(self) -> None:
        assert _is_missing_container_error(
            "Error response from daemon: No such container: 7ab0b9a1"
        )

    def test_not_running(self) -> None:
        assert _is_missing_container_error("Container abc is not running")

    def test_case_insensitive(self) -> None:
        assert _is_missing_container_error("NO SUCH CONTAINER")

    def test_unrelated_error_is_not_missing(self) -> None:
        assert not _is_missing_container_error("bash: foo: command not found")

    def test_empty_is_not_missing(self) -> None:
        assert not _is_missing_container_error("")


class TestIsRunning:
    def test_false_when_no_container_id(self) -> None:
        assert _sandbox()._is_running() is False

    def test_true_when_inspect_reports_running(self) -> None:
        sb = _sandbox()
        sb._container_id = "abc123"
        with patch("subprocess.run", return_value=_completed(0, "true\n")):
            assert sb._is_running() is True

    def test_false_when_inspect_reports_stopped(self) -> None:
        sb = _sandbox()
        sb._container_id = "abc123"
        with patch("subprocess.run", return_value=_completed(0, "false\n")):
            assert sb._is_running() is False

    def test_false_when_inspect_fails(self) -> None:
        sb = _sandbox()
        sb._container_id = "deadbeef"
        with patch("subprocess.run", return_value=_completed(1, "", "No such object")):
            assert sb._is_running() is False

    def test_false_when_docker_missing(self) -> None:
        sb = _sandbox()
        sb._container_id = "abc123"
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert sb._is_running() is False


class TestEnsureRunning:
    def test_recreates_when_container_dead(self) -> None:
        sb = _sandbox()
        sb._container_id = "deadbeef"
        with (
            patch.object(sb, "_is_running", return_value=False),
            patch.object(sb, "start") as start,
        ):
            sb._ensure_running()
        start.assert_called_once()
        # Stale id is cleared before the rebuild.
        assert sb._container_id is None

    def test_noop_when_already_running(self) -> None:
        sb = _sandbox()
        sb._container_id = "abc123"
        with (
            patch.object(sb, "_is_running", return_value=True),
            patch.object(sb, "start") as start,
        ):
            sb._ensure_running()
        start.assert_not_called()
        assert sb._container_id == "abc123"


class TestExecAsyncSelfHeal:
    @pytest.mark.asyncio
    async def test_retries_after_missing_container(self) -> None:
        """A 'No such container' error triggers one rebuild + retry."""
        sb = _sandbox()
        sb._container_id = "abc123"
        results = [
            _completed(1, "", "Error response from daemon: No such container: abc123"),
            _completed(0, "hello\n", ""),
        ]

        def _fake_ensure() -> None:
            sb._container_id = "abc123"  # simulate successful rebuild

        with (
            patch.object(sb, "_ensure_running", side_effect=_fake_ensure) as ensure,
            patch("subprocess.run", side_effect=results) as run,
        ):
            code, out, _err = await sb.exec_async("echo hello", "/workspace", 30)

        assert code == 0
        assert out == "hello\n"
        assert ensure.call_count == 2  # proactive + reactive
        assert run.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self) -> None:
        sb = _sandbox()
        sb._container_id = "abc123"
        with (
            patch.object(sb, "_ensure_running") as ensure,
            patch("subprocess.run", return_value=_completed(0, "ok\n", "")) as run,
        ):
            code, _out, _ = await sb.exec_async("echo ok", "/workspace", 30)

        assert code == 0
        assert ensure.call_count == 1  # proactive only
        assert run.call_count == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_normal_command_failure(self) -> None:
        """A real command failure (not a missing container) must not retry."""
        sb = _sandbox()
        sb._container_id = "abc123"
        with (
            patch.object(sb, "_ensure_running") as ensure,
            patch("subprocess.run", return_value=_completed(127, "", "command not found")) as run,
        ):
            code, _, err = await sb.exec_async("nope", "/workspace", 30)

        assert code == 127
        assert "command not found" in err
        assert ensure.call_count == 1
        assert run.call_count == 1
