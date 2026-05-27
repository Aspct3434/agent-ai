from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "bin" / "agent-ai.js"


def _node() -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    return node


def _run_cli(args: list[str], *, input_text: str = "") -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    return subprocess.run(
        [_node(), str(CLI), *args],
        cwd=PROJECT_ROOT,
        env=env,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def test_npm_cli_quickstart_dry_run(tmp_path: Path) -> None:
    result = _run_cli(
        [
            "install",
            "--dry-run",
            "--yes",
            "--no-start",
            "--install-dir",
            str(tmp_path / "agent-ai"),
            "--provider",
            "openai",
            "--sandbox",
            "on",
            "--messaging",
            "none",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert "AGENT AI" in result.stdout
    assert "Security warning - please read." in result.stdout
    assert "git clone --branch master https://github.com/Aspct3434/agent-ai.git" in result.stdout
    assert 'AGENT_MODEL="gpt-4o"' in result.stdout
    assert 'AGENT_SANDBOX_HOST_FALLBACK="false"' in result.stdout


def test_npm_cli_manual_dry_run_messaging_and_sandbox(tmp_path: Path) -> None:
    result = _run_cli(
        [
            "install",
            "--dry-run",
            "--yes",
            "--mode",
            "manual",
            "--no-start",
            "--install-dir",
            str(tmp_path / "agent-ai"),
            "--provider",
            "vllm",
            "--sandbox",
            "off",
            "--messaging",
            "both",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert 'OPENAI_API_BASE="http://localhost:8001/v1"' in result.stdout
    assert 'AGENT_SANDBOX_HOST_FALLBACK="true"' in result.stdout
    assert "TELEGRAM_BOT_TOKEN=" in result.stdout
    assert "DISCORD_BOT_TOKEN=" in result.stdout


def test_npm_cli_security_acknowledgement_blocks_install(tmp_path: Path) -> None:
    result = _run_cli(
        ["install", "--dry-run", "--no-start", "--install-dir", str(tmp_path / "agent-ai")],
        input_text="n\n",
    )

    assert result.returncode != 0
    assert "Installation cancelled." in result.stderr


def test_root_package_exposes_agent_ai_bin() -> None:
    package = (PROJECT_ROOT / "package.json").read_text(encoding="utf-8")

    assert '"name": "@aspct3434/agent-ai"' in package
    assert '"agent-ai": "bin/agent-ai.js"' in package
