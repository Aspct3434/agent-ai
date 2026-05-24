"""Tests for platform-aware shell detection and command safety gate."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tools as tools_module
from tools import (
    _SANDBOX_PACKAGE_CAPS,
    _detect_posix_shell,
    _DockerSandbox,
    _is_dangerous_command,
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
