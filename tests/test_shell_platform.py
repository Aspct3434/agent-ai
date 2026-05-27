"""Tests for platform-aware shell detection and command safety gate."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tools as tools_module
from tools import (
    _SANDBOX_PACKAGE_CAPS,
    ToolManager,
    _detect_posix_shell,
    _DockerSandbox,
    _is_dangerous_command,
    _normalize_sudo_noninteractive,
    _sudo_password_input_reason,
    _wrong_environment_command_reason,
)

# ---------------------------------------------------------------------------
# Shell detection
# ---------------------------------------------------------------------------

class TestDetectPosixShell:
    def test_posix_platform_returns_none(self):
        """On Linux/macOS, no override is needed; create_subprocess_shell is fine."""
        with patch("tools.platform.system", return_value="Linux"):
            result = _detect_posix_shell()
        assert result is None

    def test_macos_returns_none(self):
        with patch("tools.platform.system", return_value="Darwin"):
            result = _detect_posix_shell()
        assert result is None

    def test_windows_with_bash_returns_list(self):
        """Windows + bash on PATH -> return [bash_path, '-c']."""
        with (
            patch("tools.platform.system", return_value="Windows"),
            patch("tools.shutil.which", return_value="/usr/bin/bash"),
        ):
            result = _detect_posix_shell()
        assert result == ["/usr/bin/bash", "-c"]

    def test_windows_without_bash_returns_none(self):
        """Windows + no bash -> return None (caller must handle cmd.exe limitation)."""
        with (
            patch("tools.platform.system", return_value="Windows"),
            patch("tools.shutil.which", return_value=None),
        ):
            result = _detect_posix_shell()
        assert result is None

    def test_posix_shell_module_var_type(self):
        """_POSIX_SHELL is either None or a non-empty list of strings."""
        val = tools_module._POSIX_SHELL
        assert val is None or (isinstance(val, list) and len(val) >= 1)


# ---------------------------------------------------------------------------
# Security / command blocking
# ---------------------------------------------------------------------------

class TestIsDangerousCommand:
    # Commands that MUST be blocked
    BLOCKED = [
        "rm -rf /",
        "rm -rf //",
        "rm  -rf  /",       # extra spaces collapsed
        "rm -fr /",
        "rm -fr /*",
        "rm --recursive --force /",
        "rm --force --recursive //",
        "dd if=/dev/zero of=/dev/sda",
        "dd if=/dev/urandom of=/dev/nvme0n1",
        ":(){ :|:& };:",
        "> /etc/passwd",
        "> /etc/shadow",
        "> /etc/sudoers",
        "chmod 777 /etc",
        "chmod 777 /bin",
        "mkfs.ext4 /dev/sda1",
        "shred -n3 /dev/sdb",
    ]

    # Commands that MUST NOT be blocked (legitimate)
    ALLOWED = [
        "rm -rf /tmp/my-build",
        "rm -rf /home/user/project",
        "echo hello",
        "mkdir -p /tmp/foo",
        "cat /etc/hostname",
        "chmod 755 /tmp/mydir",
        "chmod 777 /tmp/world-writable",
        "dd if=/dev/urandom of=/tmp/random.bin bs=1M count=1",
        "python3 -m pytest",
        "git commit -m 'fix: something'",
    ]

    def test_blocked_commands(self):
        for cmd in self.BLOCKED:
            assert _is_dangerous_command(cmd), f"Should have been blocked: {cmd!r}"

    def test_allowed_commands(self):
        for cmd in self.ALLOWED:
            assert not _is_dangerous_command(cmd), f"Should NOT be blocked: {cmd!r}"


class TestWrongEnvironmentCommandReason:
    def test_blocks_windows_shell_inside_docker_sandbox(self):
        reason = _wrong_environment_command_reason(
            'cmd /c "java -version"',
            sandbox_active=True,
        )
        assert reason is not None
        assert "Windows shell" in reason

    def test_blocks_powershell_inside_docker_sandbox(self):
        reason = _wrong_environment_command_reason(
            'powershell -Command "java -version"',
            sandbox_active=True,
        )
        assert reason is not None
        assert "Windows shell" in reason

    def test_blocks_windows_path_inside_docker_sandbox(self):
        reason = _wrong_environment_command_reason(
            r"mkdir C:\Users\karim\example-server",
            sandbox_active=True,
        )
        assert reason is not None
        assert "Windows absolute path" in reason

    def test_allows_posix_command_inside_docker_sandbox(self):
        assert _wrong_environment_command_reason(
            "mkdir -p /workspace/server && java -version",
            sandbox_active=True,
        ) is None


class TestSudoSupport:
    def test_normalizes_sudo_to_noninteractive(self):
        assert _normalize_sudo_noninteractive("sudo apt-get update") == (
            "sudo -n apt-get update"
        )
        assert _normalize_sudo_noninteractive("sudo -n apt-get update") == (
            "sudo -n apt-get update"
        )
        assert _normalize_sudo_noninteractive("cd /tmp && sudo systemctl status nginx") == (
            "cd /tmp && sudo -n systemctl status nginx"
        )

    def test_blocks_password_fed_sudo(self):
        reason = _sudo_password_input_reason("echo secret | sudo -S apt-get update")
        assert reason is not None
        assert "non-interactive sudo" in reason
        assert _sudo_password_input_reason("sudo -Sk apt-get update") is not None

    def test_plain_sudo_allowed_to_fail_fast(self):
        assert _sudo_password_input_reason("sudo apt-get update") is None


# ---------------------------------------------------------------------------
# Background log path portability
# ---------------------------------------------------------------------------

def test_background_log_path_is_portable():
    """_BACKGROUND_LOG_PATH must not hardcode /tmp (breaks on Windows)."""
    import tempfile
    log = tools_module._BACKGROUND_LOG_PATH
    assert not log.startswith("/tmp"), (
        f"_BACKGROUND_LOG_PATH should use tempfile.gettempdir(), got {log!r}"
    )
    assert tempfile.gettempdir().lower() in log.lower() or Path(log).parent.exists(), (
        f"Expected log inside tempdir, got {log!r}"
    )


def test_docker_sandbox_keeps_package_install_capabilities():
    """apt/dpkg need these root capabilities inside the constrained sandbox."""
    sandbox = _DockerSandbox(image="python:3.12-slim", host_workdir="C:/workspace")
    command = sandbox._docker_run_command()
    cap_pairs = [command[i : i + 2] for i in range(len(command) - 1)]

    assert "--cap-drop" in command
    assert ["--cap-drop", "ALL"] in cap_pairs
    for cap in _SANDBOX_PACKAGE_CAPS:
        assert ["--cap-add", cap] in cap_pairs
    assert ["--cap-add", "NET_BIND_SERVICE"] in cap_pairs


def test_terminal_command_blocked_when_sandbox_required_but_unavailable():
    manager = ToolManager.__new__(ToolManager)
    manager._host_execution_disabled_reason = "Docker unavailable"
    manager.current_cwd = "C:/workspace"

    result = asyncio.run(manager.execute_terminal_command("echo should-not-run"))

    assert result["exit_code"] == -1
    assert result["scope"] == "sandbox_unavailable"
    assert "Host execution is blocked" in result["stdout"]
    assert "should-not-run" in result["stdout"]


def test_background_service_blocked_when_sandbox_required_but_unavailable():
    manager = ToolManager.__new__(ToolManager)
    manager._host_execution_disabled_reason = "Docker unavailable"
    manager.current_cwd = "C:/workspace"
    manager._sandbox = None

    result = manager.execute_background_service("python -m http.server")

    assert result["status"] == "error"
    assert result["scope"] == "sandbox_unavailable"
    assert "Host execution is blocked" in result["message"]


def _sandbox_file_manager(tmp_path: Path) -> ToolManager:
    class FakeSandbox:
        def write_text_file(self, path: str, content: str):
            target = tmp_path / path.removeprefix("/workspace/").replace("/", "\\")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {"path": path, "exists": True, "size_bytes": target.stat().st_size}

    manager = ToolManager.__new__(ToolManager)
    manager._sandbox = FakeSandbox()
    manager._host_workdir = str(tmp_path)
    manager.current_cwd = "/workspace"
    manager.public_base_url = "http://localhost:8000"
    return manager


def test_write_text_file_reports_sandbox_path(tmp_path):
    manager = _sandbox_file_manager(tmp_path)

    result = json.loads(
        manager.write_text_file(
            "generated_sites/site/index.html",
            "<!doctype html><html><head><style>body{color:black}</style></head>"
            "<body><h1>Site</h1><p>Content</p></body></html>",
        )
    )

    assert result["scope"] == "docker_sandbox"
    assert result["path"] == "/workspace/generated_sites/site/index.html"
    assert "C:" not in result["path"]
    assert (tmp_path / "generated_sites" / "site" / "index.html").is_file()
