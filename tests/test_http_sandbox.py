"""Unit tests for the pluggable HTTP-exec serverless sandbox backend.

No live service is needed — httpx.post is mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools import _HttpExecSandbox


def _resp(payload: dict, status: int = 200) -> MagicMock:
    m = MagicMock()
    m.json.return_value = payload
    m.raise_for_status.side_effect = (
        None
        if status < 400
        else httpx.HTTPStatusError("err", request=MagicMock(), response=MagicMock())
    )
    return m


def _sandbox() -> _HttpExecSandbox:
    return _HttpExecSandbox(url="http://sandbox.test", token="secret", workdir="/workspace")


class TestConstruction:
    def test_requires_url(self) -> None:
        with pytest.raises(RuntimeError):
            _HttpExecSandbox(url="", token="", workdir="/workspace")

    def test_auth_header_when_token(self) -> None:
        assert _sandbox()._headers() == {"Authorization": "Bearer secret"}

    def test_no_auth_header_without_token(self) -> None:
        assert _HttpExecSandbox(url="http://x")._headers() == {}


class TestExecAsync:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        sb = _sandbox()
        with patch("httpx.post", return_value=_resp({"exit_code": 0, "stdout": "hi", "stderr": ""})):
            code, out, err = await sb.exec_async("echo hi", "/workspace", 30)
        assert (code, out, err) == (0, "hi", "")

    @pytest.mark.asyncio
    async def test_nonzero_exit_passed_through(self) -> None:
        sb = _sandbox()
        with patch("httpx.post", return_value=_resp({"exit_code": 127, "stdout": "", "stderr": "nope"})):
            code, _out, err = await sb.exec_async("bogus", "/workspace", 30)
        assert code == 127
        assert "nope" in err

    @pytest.mark.asyncio
    async def test_network_error_becomes_error_tuple(self) -> None:
        sb = _sandbox()
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            code, _out, err = await sb.exec_async("echo hi", "/workspace", 30)
        assert code == 1
        assert "http sandbox error" in err

    @pytest.mark.asyncio
    async def test_timeout_raises_timeouterror(self) -> None:
        sb = _sandbox()
        with patch("httpx.post", side_effect=httpx.TimeoutException("slow")):
            with pytest.raises(TimeoutError):
                await sb.exec_async("sleep 99", "/workspace", 1)

    @pytest.mark.asyncio
    async def test_sends_expected_payload(self) -> None:
        sb = _sandbox()
        with patch("httpx.post", return_value=_resp({"exit_code": 0, "stdout": "", "stderr": ""})) as post:
            await sb.exec_async("ls -la", "/work", 12)
        body = post.call_args.kwargs["json"]
        assert body == {"command": "ls -la", "cwd": "/work", "timeout": 12}
        assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer secret"}


class TestExecBackground:
    def test_returns_pid(self) -> None:
        sb = _sandbox()
        with patch("httpx.post", return_value=_resp({"pid": 4321})):
            assert sb.exec_background("server", "/workspace", "/tmp/log") == 4321

    def test_failure_returns_none(self) -> None:
        sb = _sandbox()
        with patch("httpx.post", side_effect=httpx.ConnectError("x")):
            assert sb.exec_background("server", "/workspace", "/tmp/log") is None


class TestRunPython:
    def test_parses_json_stdout(self) -> None:
        sb = _sandbox()
        with patch("httpx.post", return_value=_resp({"exit_code": 0, "stdout": '{"ok": true}'})):
            assert sb._run_python("print('x')", {"a": 1}, 30) == {"ok": True}

    def test_nonzero_raises(self) -> None:
        sb = _sandbox()
        with patch("httpx.post", return_value=_resp({"exit_code": 1, "stdout": "", "stderr": "boom"})):
            with pytest.raises(RuntimeError):
                sb._run_python("x", {}, 30)


class TestEnvironmentProbe:
    def test_probe_returns_remote_json(self) -> None:
        sb = _sandbox()
        probe_out = '{"os": "Linux", "sandbox": {"mode": "http"}}'
        with patch("httpx.post", return_value=_resp({"exit_code": 0, "stdout": probe_out})):
            assert '"mode": "http"' in sb.collect_environment_json()

    def test_probe_failure_falls_back(self) -> None:
        sb = _sandbox()
        with patch("httpx.post", side_effect=httpx.ConnectError("x")):
            out = sb.collect_environment_json()
        assert "http" in out  # static fallback still identifies the backend
