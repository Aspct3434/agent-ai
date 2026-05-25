from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASH_INSTALLER = PROJECT_ROOT / "scripts" / "install.sh"
BASH_BOOTSTRAP = PROJECT_ROOT / "scripts" / "bootstrap.sh"
POWERSHELL_INSTALLER = PROJECT_ROOT / "scripts" / "install.ps1"
POWERSHELL_BOOTSTRAP = PROJECT_ROOT / "scripts" / "bootstrap.ps1"


def _run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=merged,
        text=True,
        capture_output=True,
        check=True,
    )


def _bash() -> str:
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not available")
    probe = subprocess.run([bash, "--version"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip("bash is present but not usable")
    return bash


def _powershell() -> str:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        pytest.skip("PowerShell is not available")
    return shell


def _ps_args(shell: str) -> list[str]:
    args = [shell, "-NoLogo", "-NoProfile", "-NonInteractive"]
    if os.name == "nt":
        args.extend(["-ExecutionPolicy", "Bypass"])
    return args


def test_bash_scripts_parse() -> None:
    bash = _bash()
    _run([bash, "-n", str(BASH_INSTALLER)])
    _run([bash, "-n", str(BASH_BOOTSTRAP)])


def test_pywin32_requirement_is_windows_only() -> None:
    text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert 'pywin32==311; sys_platform == "win32"' in text
    assert "\npywin32==311\n" not in text


def test_powershell_scripts_parse() -> None:
    shell = _powershell()
    for path in (POWERSHELL_INSTALLER, POWERSHELL_BOOTSTRAP):
        command = f"$null = [scriptblock]::Create([System.IO.File]::ReadAllText('{path.as_posix()}'))"
        _run([*_ps_args(shell), "-Command", command])


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("kimi", 'AGENT_MODEL="moonshot/kimi-k2.6"'),
        ("ollama", 'AGENT_MODEL="ollama/llama3.2"'),
        ("openrouter", 'OPENROUTER_API_KEY='),
        ("openai", 'AGENT_MODEL="gpt-4o"'),
        ("anthropic", 'ANTHROPIC_API_KEY='),
        ("gemini", 'GEMINI_API_KEY='),
        ("vllm", 'OPENAI_API_BASE="http://localhost:8001/v1"'),
    ],
)
def test_powershell_installer_dry_run_provider_choices(provider: str, expected: str, tmp_path: Path) -> None:
    result = _run(
        [
            *_ps_args(_powershell()),
            "-File",
            str(POWERSHELL_INSTALLER),
            "-DryRun",
            "-NoStart",
            "-Provider",
            provider,
            "-Sandbox",
            "on",
            "-Messaging",
            "none",
            "-EnvFile",
            str(tmp_path / "an-api.env"),
        ]
    )

    assert expected in result.stdout
    assert "AGENT_SANDBOX=" in result.stdout
    assert 'AGENT_SANDBOX_HOST_FALLBACK="false"' in result.stdout


def test_powershell_bootstrap_dry_run_forwards_options(tmp_path: Path) -> None:
    result = _run(
        [
            *_ps_args(_powershell()),
            "-File",
            str(POWERSHELL_BOOTSTRAP),
            "-DryRun",
            "-NoStart",
            "-RepoUrl",
            "https://github.com/example/agent-ai.git",
            "-InstallDir",
            str(tmp_path / "agent-ai"),
            "-Provider",
            "openrouter",
            "-Sandbox",
            "off",
            "-Messaging",
            "both",
        ]
    )

    assert "git clone --branch master https://github.com/example/agent-ai.git" in result.stdout
    assert "scripts\\install.ps1" in result.stdout
    assert "-Provider openrouter" in result.stdout
    assert "-Sandbox off" in result.stdout
    assert "-Messaging both" in result.stdout
    assert "-NoStart" in result.stdout


def test_docker_compose_config_accepts_temp_env_file(tmp_path: Path) -> None:
    docker = shutil.which("docker")
    if not docker:
        pytest.skip("docker is not available")
    probe = subprocess.run([docker, "compose", "version"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip("docker compose is not available")

    env_file = tmp_path / "an-api.env"
    env_file.write_text(
        "AGENT_MODEL=moonshot/kimi-k2.6\nFAST_AGENT_MODEL=moonshot/kimi-k2.6\nSTRONG_AGENT_MODEL=moonshot/kimi-k2.6\n",
        encoding="utf-8",
    )
    result = _run([docker, "compose", "config"], env={"AGENT_ENV_FILE": str(env_file)})

    assert "control_panel:" in result.stdout
    assert "5173" in result.stdout


def test_readme_empty_pc_commands_use_bootstrap() -> None:
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "npx @aspct3434/agent-ai install" in text
    assert "npx --yes github:Aspct3434/agent-ai install" in text
    assert "npx @aspct3434/agent-ai doctor" in text
    assert "npm i -g @aspct3434/agent-ai" in text
    assert "scripts/bootstrap.ps1" in text
    assert "scripts/bootstrap.sh" in text
    assert "https://raw.githubusercontent.com/Aspct3434/agent-ai/master/scripts/bootstrap.ps1" in text
    assert "https://raw.githubusercontent.com/Aspct3434/agent-ai/master/scripts/bootstrap.sh" in text
    assert "scripts/install.ps1 | iex" not in text
    assert "scripts/install.sh | bash" not in text
