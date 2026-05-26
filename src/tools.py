from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import platform
import posixpath
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult, PaginatedRequestParams

logger = logging.getLogger(__name__)

# Project root is one level above this file (src/../)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------
# Built-in (non-MCP) tool schemas
# Appended to list_all_tools() so the LLM sees them alongside MCP tools.
# server="__builtin__" distinguishes them from real MCP sessions;
# callers that need to execute them must route on that sentinel.
# ------------------------------------------------------------------

# Slow-but-finite commands (apt-get install, large downloads, builds) routinely
# take longer than a few seconds. A short timeout here was the root of a failure
# cascade: installs were killed mid-download, the agent then (wrongly) backgrounded
# them and burned its whole iteration budget polling the log. Give finite commands
# real time to finish synchronously; only truly non-terminating processes belong in
# execute_background_service. Tunable via AGENT_TERMINAL_TIMEOUT_SECONDS.
_TERMINAL_TIMEOUT_SECONDS = max(5, int(os.getenv("AGENT_TERMINAL_TIMEOUT_SECONDS", "300")))

# Combined stdout/stderr of every background service is appended here.
# Uses the OS temp dir so it works on Windows (no /tmp) and Linux/macOS.
_BACKGROUND_LOG_PATH = str(Path(tempfile.gettempdir()) / "background_task.log")


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

_PLACEHOLDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\badd\s+(?:more\s+)?(?:css|styles?|javascript|js|content|html|text|features?)\s+here\b", re.I),
    re.compile(r"\b(?:todo|fixme)\s*:\s*(?:add|implement|write|fill)\b", re.I),
    re.compile(r"\bplaceholder\s+(?:text|content|copy|image|section)\b", re.I),
    re.compile(r"\blorem\s+ipsum\b", re.I),
)

_INTERACTIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\baddEventListener\s*\(", re.I),
    re.compile(r"\bon(?:click|input|change|submit|mousemove|keydown)\s*=", re.I),
    re.compile(r"\b(querySelector|getElementById)\s*\(", re.I),
    re.compile(r"\b(setInterval|setTimeout|requestAnimationFrame)\s*\(", re.I),
    re.compile(r"\b(canvas|input|button|select|textarea|details)\b", re.I),
)


def _artifact_quality_for_text(content: str, path: str | None = None) -> dict[str, Any]:
    placeholder_matches: list[str] = []
    for pattern in _PLACEHOLDER_PATTERNS:
        placeholder_matches.extend(match.group(0) for match in pattern.finditer(content))

    style_rule_count = len(re.findall(r"\{[^{}]*:[^{};]+;?[^{}]*\}", content))
    interactive_signal_count = sum(
        len(pattern.findall(content)) for pattern in _INTERACTIVE_PATTERNS
    )
    visible_text = re.sub(r"<script\b[^>]*>.*?</script>", " ", content, flags=re.I | re.S)
    visible_text = re.sub(r"<style\b[^>]*>.*?</style>", " ", visible_text, flags=re.I | re.S)
    visible_text = re.sub(r"<[^>]+>", " ", visible_text)
    content_word_count = len(re.findall(r"[A-Za-z0-9]+", visible_text))

    suffix = Path(path or "").suffix.lower()
    is_web_artifact = suffix in {".html", ".css", ".js"} or "<html" in content.lower()
    return {
        "artifact_quality_ok": not placeholder_matches,
        "is_web_artifact": is_web_artifact,
        "placeholder_detected": bool(placeholder_matches),
        "placeholder_matches": placeholder_matches[:8],
        "style_rule_count": style_rule_count,
        "interactive_signal_count": interactive_signal_count,
        "content_word_count": content_word_count,
    }


def _artifact_quality_for_site(source: Path) -> dict[str, Any]:
    chunks: list[str] = []
    for path in sorted(source.rglob("*")):
        if path.suffix.lower() not in {".html", ".css", ".js"} or not path.is_file():
            continue
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    quality = _artifact_quality_for_text("\n".join(chunks), str(source / "index.html"))
    quality["is_web_artifact"] = True
    return quality


def _detect_posix_shell() -> list[str] | None:
    """Return a POSIX shell argv prefix for Windows hosts, or None on POSIX systems.

    On Linux/macOS, ``asyncio.create_subprocess_shell`` already uses ``/bin/sh``.
    On Windows it uses ``cmd.exe``, which breaks POSIX commands (``mkdir -p``,
    ``cat >``, path separators, etc.).  We look for ``bash`` (Git Bash or WSL)
    and return ``["bash", "-c"]`` so callers can use
    ``create_subprocess_exec(*_POSIX_SHELL, command, ...)`` instead.
    """
    if platform.system() != "Windows":
        return None
    bash = shutil.which("bash")
    if bash:
        logger.info("Windows: using POSIX shell %s for terminal commands", bash)
        return [bash, "-c"]
    logger.warning(
        "Windows: no bash found on PATH -- POSIX commands (mkdir -p, cat, etc.) "
        "will fail under cmd.exe. Install Git for Windows or WSL to get bash."
    )
    return None


# Resolved once at import time; None means the OS already provides a POSIX shell.
_POSIX_SHELL: list[str] | None = _detect_posix_shell()

_PROBED_RUNTIMES = (
    # Interpreters / runtimes
    "java", "python", "python3", "node", "ruby", "go",
    # Build toolchain — important: agent needs to know these BEFORE attempting compiles
    "rustc", "cargo", "gcc", "g++", "make", "cmake",
    # Package / download utilities
    "curl", "wget", "git",
    # OS package managers — reported so the agent installs with one that actually
    # exists for THIS host instead of guessing (e.g. apt-get on a Windows box).
    "apt-get", "yum", "dnf", "pacman", "apk", "brew",
    "choco", "scoop", "winget",
    # Container / orchestration
    "docker", "docker-compose",
    # Package managers / scaffolders — npx is how React/Vite/Next apps are created
    "npm", "npx", "pip", "pip3",
)

# ---------------------------------------------------------------------------
# Security: command safety gate
#
# Docker + a non-root user is the real security boundary for production.
# This regex-based gate is a defence-in-depth layer that blocks the most
# obviously catastrophic shell commands regardless of argument spacing or
# flag ordering, so an accidental (or adversarial) prompt can't wipe the
# filesystem in a local-dev session.
#
# Design: each pattern is anchored to the specific dangerous *effect*
# (destroying the root fs, forking unboundedly, overwriting /etc auth files)
# rather than a single literal string, so simple bypasses like extra spaces,
# doubled slashes, or reordered flags are still caught.
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    # rm targeting the filesystem root (/ or /* or /.) -- NOT /tmp/..., /var/...
    # Matches: rm -rf /, rm -fr /*, rm --recursive --force /, etc.
    # The path must be exactly '/' optionally followed by '*' or '.' to avoid
    # blocking legitimate subdirectory removals like rm -rf /tmp/work.
    re.compile(
        r"\brm\s+(-[a-z]*rf[a-z]*|-[a-z]*fr[a-z]*|--recursive\s+--force|--force\s+--recursive)"
        r"\s+['\"]?/['\"/\*\.]*['\"]?\s*$",
        re.IGNORECASE,
    ),
    # dd writing to block devices (sda/hdb style) or NVMe (nvme0n1 style)
    re.compile(r"\bdd\b.{0,60}\bof=/dev/(sd|hd|xvd|vd)[a-z]", re.IGNORECASE),
    re.compile(r"\bdd\b.{0,60}\bof=/dev/nvme\d", re.IGNORECASE),
    re.compile(r"\bdd\b.{0,60}\bof=/dev/zero\b", re.IGNORECASE),
    # Fork bomb
    re.compile(r":\(\)\s*\{.*:\|:", re.DOTALL),
    # Overwriting /etc auth files
    re.compile(r">\s*/etc/(passwd|shadow|sudoers)", re.IGNORECASE),
    # Wiping whole partition / boot sector
    re.compile(r"\bmkfs\b.{0,40}/dev/(sd|hd|xvd|vd)[a-z]", re.IGNORECASE),
    re.compile(r"\bmkfs\b.{0,40}/dev/nvme\d", re.IGNORECASE),
    re.compile(r"\bshred\b.{0,40}/dev/(sd|hd|xvd|vd)[a-z]", re.IGNORECASE),
    re.compile(r"\bshred\b.{0,40}/dev/nvme\d", re.IGNORECASE),
    # chmod 777 on system directories
    re.compile(r"\bchmod\s+777\s+/(etc|bin|sbin|lib|usr)\b", re.IGNORECASE),
)

# Matches the target of any `cd` call in a shell command string.
# Captures double-quoted, single-quoted, or bare (unquoted) paths.
# The last match wins when a command chains multiple cd calls.
_CD_RE = re.compile(
    r'(?:^|[;&|])\s*cd\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))',
    re.MULTILINE,
)

# Matches a command that *begins* with a cd -- used to intercept and pre-apply
# the directory change before any subprocess is spawned.
# Optionally consumes a trailing `&&` separator so the remainder of the
# command can be extracted cleanly.
_LEADING_CD_RE = re.compile(
    r'^\s*cd\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))\s*(?:&&\s*)?'
)

_WINDOWS_SHELL_INVOKE_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:cmd(?:\.exe)?|powershell(?:\.exe)?|pwsh(?:\.exe)?)\b",
    re.IGNORECASE,
)
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    r"(?:^|[\s\"'])"
    r"(?:[A-Za-z]:[\\/]|%USERPROFILE%|%HOMEDRIVE%|%APPDATA%|%LOCALAPPDATA%)",
    re.IGNORECASE,
)
_WINDOWS_WHERE_RE = re.compile(r"(?:^|[;&|]\s*)where(?:\.exe)?\s+\S+", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Docker sandbox mode
#
# When AGENT_SANDBOX=docker, all terminal and background commands are routed
# through an isolated Docker container instead of running on the Windows/host
# shell. This gives the agent a consistent Linux environment with safe package
# install access (apt-get etc.) without touching the host OS.
#
# Configuration env vars:
#   AGENT_SANDBOX          — set to "docker" to enable
#   AGENT_SANDBOX_IMAGE    — container image (default: python:3.12-slim)
#   AGENT_SANDBOX_MEMORY   — memory limit (default: 1g)
#   AGENT_SANDBOX_CPUS     — CPU limit (default: 1.0)
#   AGENT_SANDBOX_PIDS     — PID limit (default: 256)
#   AGENT_SANDBOX_HOST_FALLBACK
#                           — set true to allow direct host execution if Docker
#                             fails; default false blocks terminal/process tools
# ---------------------------------------------------------------------------

_SANDBOX_MODE = os.getenv("AGENT_SANDBOX", "").lower()
_SANDBOX_ENABLED = _SANDBOX_MODE == "docker"
_SSH_SANDBOX_ENABLED = _SANDBOX_MODE == "ssh"
_HTTP_SANDBOX_ENABLED = _SANDBOX_MODE == "http"

# SSH sandbox configuration
_SSH_HOST = os.getenv("AGENT_SSH_HOST", "")
_SSH_USER = os.getenv("AGENT_SSH_USER", "")
_SSH_KEY_PATH = os.getenv("AGENT_SSH_KEY_PATH", "")
_SSH_WORKDIR = os.getenv("AGENT_SSH_WORKDIR", "/tmp/agent-workspace")
_SSH_PORT = int(os.getenv("AGENT_SSH_PORT", "22"))

# HTTP-exec sandbox configuration (pluggable serverless backend).
# Point AGENT_SANDBOX_EXEC_URL at a small shim in front of any serverless
# sandbox (Daytona, E2B, Modal, Vercel Sandbox, …); see _HttpExecSandbox.
_HTTP_EXEC_URL = os.getenv("AGENT_SANDBOX_EXEC_URL", "").rstrip("/")
_HTTP_EXEC_TOKEN = os.getenv("AGENT_SANDBOX_EXEC_TOKEN", "")
_HTTP_EXEC_WORKDIR = os.getenv("AGENT_SANDBOX_EXEC_WORKDIR", "/workspace")

# Vision and image generation model config
_VISION_MODEL = os.getenv("AGENT_VISION_MODEL", "")
_IMAGE_GEN_MODEL = os.getenv("AGENT_IMAGE_GEN_MODEL", "dall-e-3")
_SANDBOX_IMAGE = os.getenv("AGENT_SANDBOX_IMAGE", "python:3.12-slim")
_SANDBOX_WORKDIR = "/workspace"
_SANDBOX_MEMORY = os.getenv("AGENT_SANDBOX_MEMORY", "1g")
_SANDBOX_CPUS = os.getenv("AGENT_SANDBOX_CPUS", "1.0")
_SANDBOX_PIDS = os.getenv("AGENT_SANDBOX_PIDS", "256")
_SANDBOX_FALLBACK_TO_HOST = _env_flag("AGENT_SANDBOX_HOST_FALLBACK", "false")
_SANDBOX_PACKAGE_CAPS = (
    # apt/dpkg drop privileges to the _apt user and adjust package-owned files.
    # Keep the sandbox otherwise locked down, but allow normal package installs.
    "CHOWN",
    "DAC_OVERRIDE",
    "FOWNER",
    "SETGID",
    "SETUID",
)

# Baseline packages installed automatically inside every new sandbox container.
# Slim Docker images (python:*-slim, debian-slim, etc.) ship with almost nothing
# — no curl, wget, git, or even unzip — so any task that downloads a file, clones
# a repo, or extracts an archive would fail immediately without these.
# Env-tunable: set AGENT_SANDBOX_BASELINE_PACKAGES to a comma-separated list.
_SANDBOX_BASELINE_PACKAGES: list[str] = [
    p.strip()
    for p in os.getenv(
        "AGENT_SANDBOX_BASELINE_PACKAGES",
        "curl,wget,git,unzip,procps",
    ).split(",")
    if p.strip()
]

# Path the agent should read to inspect the background-service log, expressed
# from the agent's OWN vantage point: inside the container under sandbox mode,
# on the host otherwise. Tool *descriptions* and prompts must reference this --
# never the host-only ``_BACKGROUND_LOG_PATH`` -- so the advice is correct in
# whichever execution mode is active. Actual host file I/O still uses
# ``_BACKGROUND_LOG_PATH`` directly.
_AGENT_BACKGROUND_LOG_PATH = (
    "/tmp/background_task.log" if _SANDBOX_ENABLED else _BACKGROUND_LOG_PATH
)


class _ScriptSandbox:
    """Shared behaviour for sandboxes that run Linux Python helpers.

    Both the Docker and SSH backends execute the agent's commands on a Linux
    host, so the file-write, evidence-collection, and HTTP-proxy helpers are
    identical Python scripts fed a JSON payload on stdin. Only the *transport*
    differs (``docker exec`` vs ``ssh``), so subclasses implement just
    :meth:`_run_python`. Keeping the scripts here prevents the two backends
    from drifting apart and guarantees every backend supports the full tool
    surface (writing files, collecting evidence, proxying HTTP).
    """

    _WRITE_SCRIPT = r"""
import json, os, sys

req = json.load(sys.stdin)
path = req["path"]
parent = os.path.dirname(path)
if parent:
    os.makedirs(parent, exist_ok=True)
with open(path, "w", encoding="utf-8") as handle:
    handle.write(req.get("content", ""))
st = os.stat(path)
print(json.dumps({
    "path": path,
    "exists": os.path.isfile(path),
    "size_bytes": st.st_size,
}))
"""

    _EVIDENCE_SCRIPT = r"""
import json, os, socket, sys, time

req = json.load(sys.stdin)

def preview(path, max_bytes=1200):
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
    except OSError as exc:
        return "[preview error] " + str(exc)
    if b"\0" in data:
        return f"[binary file preview omitted; sampled {len(data)} bytes]"
    printable = sum(1 for b in data if b in b"\n\r\t" or 32 <= b <= 126)
    if data and printable / len(data) < 0.75:
        return f"[binary file preview omitted; sampled {len(data)} bytes]"
    return data.decode("utf-8", "replace")

def inspect_path(raw, cwd):
    path = raw if os.path.isabs(raw) else os.path.join(cwd, raw)
    path = os.path.normpath(path)
    info = {"input": raw, "path": path, "exists": os.path.exists(path)}
    if not info["exists"]:
        return info
    try:
        st = os.stat(path)
        info.update({
            "is_file": os.path.isfile(path),
            "is_dir": os.path.isdir(path),
            "size_bytes": st.st_size,
            "modified_at_unix": st.st_mtime,
            "modified_age_seconds": round(time.time() - st.st_mtime, 3),
        })
        if info["is_dir"]:
            children = []
            for name in sorted(os.listdir(path), key=str.lower)[:50]:
                child = os.path.join(path, name)
                children.append({
                    "name": name,
                    "is_file": os.path.isfile(child),
                    "is_dir": os.path.isdir(child),
                    "size_bytes": os.path.getsize(child) if os.path.exists(child) else None,
                })
            info["children"] = children
        elif info["is_file"]:
            info["preview"] = preview(path)
    except OSError as exc:
        info["error"] = str(exc)
    return info

def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""

def inspect_pid(pid):
    info = {"pid": pid, "running": os.path.exists(f"/proc/{pid}")}
    if info["running"]:
        line = cmdline(pid)
        if line:
            info["cmdline"] = line
    return info

def inspect_process_name(name):
    needle = name.lower()
    matches = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        line = cmdline(pid)
        try:
            with open(f"/proc/{pid}/comm", "r", encoding="utf-8", errors="replace") as f:
                comm = f.read().strip()
        except OSError:
            comm = ""
        haystack = f"{comm} {line}".lower()
        if needle in haystack:
            matches.append({"pid": int(pid), "comm": comm, "cmdline": line})
            if len(matches) >= 20:
                break
    return {"name": name, "matches": matches, "count": len(matches)}

def inspect_port(port):
    info = {"port": port, "host": "127.0.0.1", "connectable": False}
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        info["connectable"] = sock.connect_ex(("127.0.0.1", int(port))) == 0
    except OSError as exc:
        info["error"] = str(exc)
    finally:
        sock.close()
    return info

def tail_file(path, max_bytes=4000):
    info = {"path": path, "exists": os.path.exists(path)}
    if not info["exists"]:
        return info
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            info["tail"] = f.read().decode("utf-8", "replace")
            info["size_bytes"] = size
    except OSError as exc:
        info["error"] = str(exc)
    return info

cwd = req.get("cwd") or "/workspace"
out = {
    "scope": req.get("scope") or "docker_sandbox",
    "current_working_directory": cwd,
    "paths": [inspect_path(p, cwd) for p in req.get("paths") or []],
    "pids": [inspect_pid(int(p)) for p in req.get("pids") or []],
    "process_names": [inspect_process_name(p) for p in req.get("process_names") or []],
    "ports": [inspect_port(int(p)) for p in req.get("ports") or []],
}
if req.get("include_background_log", True):
    out["background_log"] = tail_file(req.get("background_log_path") or "/tmp/background_task.log")
print(json.dumps(out))
"""

    _HTTP_SCRIPT = r"""
import base64, json, sys, urllib.error, urllib.request

req = json.load(sys.stdin)
url = f"http://127.0.0.1:{int(req['port'])}/{req.get('path','').lstrip('/')}"
if req.get("query"):
    url += "?" + req["query"]
body = base64.b64decode(req.get("body_b64") or "")
data = body if body else None
request = urllib.request.Request(
    url,
    data=data,
    method=req.get("method") or "GET",
    headers=req.get("headers") or {},
)
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
        print(json.dumps({
            "status_code": response.status,
            "headers": dict(response.headers.items()),
            "body_b64": base64.b64encode(payload).decode("ascii"),
        }))
except urllib.error.HTTPError as exc:
    payload = exc.read()
    print(json.dumps({
        "status_code": exc.code,
        "headers": dict(exc.headers.items()),
        "body_b64": base64.b64encode(payload).decode("ascii"),
    }))
"""

    # Backend-specific transport. Runs *script* on the Linux host with *payload*
    # serialised to JSON on stdin and returns the parsed JSON the script prints.
    def _run_python(self, script: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        raise NotImplementedError

    # Subclasses override to label evidence with their own sandbox mode.
    _evidence_scope: str = "docker_sandbox"

    def write_text_file(self, path: str, content: str) -> dict[str, Any]:
        """Write a UTF-8 text file inside the sandbox workspace."""
        return self._run_python(
            self._WRITE_SCRIPT, {"path": path, "content": content}, timeout=30
        )

    def collect_evidence_json(
        self,
        *,
        paths: list[str] | None = None,
        pids: list[int] | None = None,
        process_names: list[str] | None = None,
        ports: list[int] | None = None,
        include_background_log: bool = True,
        cwd: str = _SANDBOX_WORKDIR,
    ) -> dict[str, Any]:
        """Inspect paths, processes, ports, and logs from inside the sandbox."""
        return self._run_python(
            self._EVIDENCE_SCRIPT,
            {
                "cwd": cwd,
                "scope": self._evidence_scope,
                "paths": paths or [],
                "pids": pids or [],
                "process_names": process_names or [],
                "ports": ports or [],
                "include_background_log": include_background_log,
            },
            timeout=10,
        )

    def fetch_http_json(
        self,
        *,
        port: int,
        path: str,
        query: str,
        method: str,
        headers: dict[str, str],
        body_b64: str,
    ) -> dict[str, Any]:
        """Fetch an HTTP URL from inside the sandbox for the backend proxy."""
        return self._run_python(
            self._HTTP_SCRIPT,
            {
                "port": int(port),
                "path": path,
                "query": query,
                "method": method,
                "headers": headers,
                "body_b64": body_b64,
            },
            timeout=35,
        )


def _is_missing_container_error(stderr: str) -> bool:
    """True if *stderr* indicates the target container is gone or stopped."""
    text = (stderr or "").lower()
    return "no such container" in text or "is not running" in text


class _DockerSandbox(_ScriptSandbox):
    """A long-lived Docker container used as an isolated command execution sandbox.

    One instance is created per ``ToolManager`` when ``AGENT_SANDBOX=docker``.
    The container runs ``sleep infinity`` so it stays alive between commands;
    each command is executed via ``docker exec``. The agentic workspace lives
    inside the container at ``/workspace`` rather than bind-mounting the host
    project directory, so general agent tasks cannot mutate the repo or host OS.
    """

    def __init__(self, image: str, host_workdir: str) -> None:
        self._image = image
        self._host_workdir = host_workdir
        self._container_id: str | None = None
        self._name = f"agent-sandbox-{os.getpid()}"
        # Guards container (re)creation so concurrent commands rebuild it once.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Pull image (if needed) and start the sandbox container."""
        # Force-remove any leftover container with the same name (running or
        # stopped) so `docker run --name` cannot fail with a name collision.
        subprocess.run(
            ["docker", "rm", "-f", self._name],
            capture_output=True, timeout=15,
        )

        result = subprocess.run(
            self._docker_run_command(),
            capture_output=True, text=True, timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Docker sandbox failed to start (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )

        self._container_id = result.stdout.strip()
        # Ensure the workspace dir exists inside the container.
        subprocess.run(
            ["docker", "exec", self._container_id, "mkdir", "-p", _SANDBOX_WORKDIR],
            capture_output=True, timeout=10,
        )
        # Install baseline packages (curl, wget, git, …) that slim images lack.
        self._install_baseline_packages()
        logger.info(
            "Docker sandbox started: container=%s image=%s container_workdir=%s",
            self._container_id[:12], self._image, _SANDBOX_WORKDIR,
        )

    def _install_baseline_packages(self) -> None:
        """Install essential packages that slim Docker images do not include.

        Without these, nearly every real task fails on the first download,
        clone, or archive extraction. The list is env-tunable via
        ``AGENT_SANDBOX_BASELINE_PACKAGES``.
        """
        if not _SANDBOX_BASELINE_PACKAGES:
            return
        pkgs = " ".join(_SANDBOX_BASELINE_PACKAGES)
        logger.info("Installing sandbox baseline packages: %s …", pkgs)
        result = subprocess.run(
            [
                "docker", "exec", self._container_id, "bash", "-c",
                f"apt-get update -qq && apt-get install -y -qq {pkgs} 2>&1 | tail -5",
            ],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            logger.warning(
                "Sandbox baseline package install failed (exit %d): %s",
                result.returncode, (result.stderr or result.stdout)[:300],
            )
        else:
            logger.info("Sandbox baseline packages installed successfully.")

    def _docker_run_command(self) -> list[str]:
        """Return the docker run command used to create the sandbox."""
        command = [
            "docker", "run", "-d", "--rm",
            "--name", self._name,
            "--memory", _SANDBOX_MEMORY,
            "--cpus", _SANDBOX_CPUS,
            "--pids-limit", _SANDBOX_PIDS,
            "--cap-drop", "ALL",
        ]
        for cap in _SANDBOX_PACKAGE_CAPS:
            command.extend(["--cap-add", cap])
        command.extend(
            [
                "--cap-add", "NET_BIND_SERVICE",
                "--security-opt", "no-new-privileges",
                "-w", _SANDBOX_WORKDIR,
                self._image,
                "sleep", "infinity",
            ]
        )
        return command

    def stop(self) -> None:
        """Stop and remove the sandbox container."""
        if self._container_id:
            subprocess.run(
                ["docker", "stop", "-t", "3", self._container_id],
                capture_output=True, timeout=15,
            )
            self._container_id = None
            logger.info("Docker sandbox stopped.")

    # ------------------------------------------------------------------
    # Self-healing
    # ------------------------------------------------------------------

    def _is_running(self) -> bool:
        """Return True only if the cached container exists and is running."""
        if not self._container_id:
            return False
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self._container_id],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0 and result.stdout.strip() == "true"

    def _ensure_running(self) -> None:
        """Recreate the container if it has died, been removed, or never started.

        Containers run with ``--rm``, so a Docker daemon restart, host sleep, or
        manual ``docker stop`` removes the container entirely; the cached id then
        points at nothing and every ``docker exec`` fails with "No such
        container". This makes the sandbox self-healing — it transparently
        rebuilds on the next command instead of staying broken until the whole
        process restarts.
        """
        if self._is_running():
            return
        with self._lock:
            # Re-check inside the lock: another thread may have just rebuilt it.
            if self._is_running():
                return
            if self._container_id:
                logger.warning(
                    "Docker sandbox container %s is gone; recreating it.",
                    self._container_id[:12],
                )
                self._container_id = None
            self.start()

    @property
    def container_id(self) -> str:
        if not self._container_id:
            raise RuntimeError("Docker sandbox is not running.")
        return self._container_id

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    async def exec_async(
        self, command: str, cwd: str, timeout: float
    ) -> tuple[int, str, str]:
        """Run *command* inside the container at *cwd* and await its output."""
        def _docker_exec() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [
                    "docker",
                    "exec",
                    "-w",
                    cwd,
                    self.container_id,
                    "bash",
                    "-c",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        def _run() -> subprocess.CompletedProcess[str]:
            self._ensure_running()
            result = _docker_exec()
            # Reactive self-heal: the container can die in the narrow window
            # between the liveness check and the exec. Rebuild once and retry.
            if result.returncode != 0 and _is_missing_container_error(result.stderr):
                self._container_id = None
                self._ensure_running()
                result = _docker_exec()
            return result

        try:
            result = await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError from exc
        return result.returncode, result.stdout, result.stderr

    def exec_background(self, command: str, cwd: str, log_path: str) -> int | None:
        """Launch *command* detached inside the container; output appends to *log_path*."""
        self._ensure_running()
        wrapped = (
            f"cd {shlex.quote(cwd)} && "
            f"nohup bash -lc {shlex.quote(command)} "
            f">> {shlex.quote(log_path)} 2>&1 < /dev/null & echo $!"
        )
        try:
            result = subprocess.run(
                ["docker", "exec", self.container_id, "bash", "-c", wrapped],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.warning("sandbox exec_background failed: %s", result.stderr)
                return None
            return int(result.stdout.strip().splitlines()[-1])
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            logger.warning("sandbox exec_background failed: %s", exc)
            return None

    _evidence_scope = "docker_sandbox"

    def _run_python(
        self, script: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        """Run *script* inside the container with *payload* JSON on stdin."""
        self._ensure_running()
        result = subprocess.run(
            ["docker", "exec", "-i", self.container_id, "python3", "-c", script],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "sandbox script failed")
        return json.loads(result.stdout)

    # ------------------------------------------------------------------
    # Environment introspection
    # ------------------------------------------------------------------

    def collect_environment_json(self) -> str:
        """Return a JSON snapshot of the *container's* environment."""
        self._ensure_running()
        probe_script = (
            "python3 -c \""
            "import json, shutil, subprocess, os, platform; "
            "runtimes = {n: shutil.which(n) is not None for n in ["
            "'java','python','python3','node','ruby','go',"
            "'rustc','cargo','gcc','g++','make','cmake',"
            "'curl','wget','git',"
            "'apt-get','yum','dnf','pacman','apk','brew','choco','scoop','winget',"
            "'docker','docker-compose','npm','npx','pip','pip3']}; "
            "import getpass; "
            "try: user = getpass.getuser()\nexcept: user = os.environ.get('USER','unknown'); "
            "is_root = os.geteuid() == 0; "
            "sudo_av = shutil.which('sudo') is not None; "
            "import shutil as _s; u = _s.disk_usage('/'); "
            "disk = {'total_gb': round(u.total/1073741824,2),'used_gb': round(u.used/1073741824,2),'free_gb': round(u.free/1073741824,2)}; "
            "print(json.dumps({'os':'Linux','machine':platform.machine(),"
            "'disk_cwd':disk,'runtimes':runtimes,"
            "'shell':{'shell':'bash','posix':True},"
            "'user':{'username':user,'is_root':is_root,'sudo_available':sudo_av},"
            "'sandbox':{'mode':'docker','image':os.environ.get('SANDBOX_IMAGE','unknown'),"
            "'container_workdir':'/workspace'}}))"
            "\""
        )
        try:
            result = subprocess.run(
                ["docker", "exec",
                 "-e", f"SANDBOX_IMAGE={self._image}",
                 self.container_id, "bash", "-c", probe_script],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as exc:
            logger.warning("sandbox environment probe failed: %s", exc)

        # Fallback: minimal JSON so the agent knows it's in Linux Docker
        return json.dumps({
            "os": "Linux",
            "sandbox": {"mode": "docker", "image": self._image,
                        "container_workdir": _SANDBOX_WORKDIR},
            "shell": {"shell": "bash", "posix": True},
            "note": "environment probe unavailable; apt-get is available for installs",
        }, indent=2)


# ---------------------------------------------------------------------------
# SSH sandbox
# ---------------------------------------------------------------------------


class _SSHSandbox(_ScriptSandbox):
    """Route agent commands to a remote host over SSH.

    Authentication uses a key file (AGENT_SSH_KEY_PATH) or falls back to the
    SSH agent / host ~/.ssh/config. The remote working directory is created
    automatically on first use.
    """

    def __init__(
        self,
        host: str,
        user: str,
        key_path: str = "",
        workdir: str = "/tmp/agent-workspace",
        port: int = 22,
    ) -> None:
        self._host = host
        self._user = user
        self._key_path = key_path
        self._workdir = workdir
        self._port = port

    def _ssh_argv(self) -> list[str]:
        argv = ["ssh", "-o", "StrictHostKeyChecking=no", "-p", str(self._port)]
        if self._key_path:
            argv += ["-i", self._key_path]
        return [*argv, f"{self._user}@{self._host}"]

    def start(self) -> None:
        """Verify connectivity and create the remote workspace directory."""
        result = subprocess.run(
            [*self._ssh_argv(), f"mkdir -p {shlex.quote(self._workdir)}"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SSH sandbox connection failed: {result.stderr.strip()}"
            )
        logger.info("SSH sandbox ready: %s@%s:%s", self._user, self._host, self._workdir)

    def stop(self) -> None:
        pass  # stateless; no persistent process to tear down

    async def exec_async(
        self, command: str, cwd: str, timeout: float
    ) -> tuple[int, str, str]:
        full_cmd = f"cd {shlex.quote(cwd)} && {command}"
        argv = [*self._ssh_argv(), full_cmd]

        def _run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout
            )

        try:
            result = await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError from exc
        return result.returncode, result.stdout, result.stderr

    def exec_background(self, command: str, cwd: str, log_path: str) -> int | None:
        wrapped = (
            f"cd {shlex.quote(cwd)} && "
            f"nohup bash -c {shlex.quote(command)} "
            f">> {shlex.quote(log_path)} 2>&1 < /dev/null & echo $!"
        )
        try:
            result = subprocess.run(
                [*self._ssh_argv(), wrapped],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return None
            return int(result.stdout.strip().splitlines()[-1])
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            logger.warning("SSH exec_background failed: %s", exc)
            return None

    _evidence_scope = "ssh_sandbox"

    def _run_python(
        self, script: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        """Run *script* on the remote host with *payload* JSON piped on stdin."""
        remote_cmd = f"python3 -c {shlex.quote(script)}"
        try:
            result = subprocess.run(
                [*self._ssh_argv(), remote_cmd],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError from exc
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ssh sandbox script failed")
        return json.loads(result.stdout)

    def collect_environment_json(self) -> str:
        script = (
            "python3 -c \""
            "import json,platform,shutil,os; "
            "rts=['python3','node','npm','git','curl','apt-get']; "
            "found={r:shutil.which(r) is not None for r in rts}; "
            "print(json.dumps({'os':'Linux','shell':{'shell':'bash','posix':True},"
            "'sandbox':{'mode':'ssh','host':os.environ.get('SSH_CONNECTION','?')},"
            "'runtimes':found}))\""
        )
        try:
            result = subprocess.run(
                [*self._ssh_argv(), script],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as exc:
            logger.warning("SSH environment probe failed: %s", exc)
        return json.dumps({
            "os": "Linux", "sandbox": {"mode": "ssh", "host": self._host},
            "shell": {"shell": "bash", "posix": True},
        }, indent=2)


class _HttpExecSandbox(_ScriptSandbox):
    """Pluggable serverless backend: run commands via a remote HTTP exec shim.

    Set ``AGENT_SANDBOX=http`` and ``AGENT_SANDBOX_EXEC_URL`` to a service that
    fronts any serverless sandbox (Daytona, E2B, Modal, Vercel Sandbox, …). The
    contract is one endpoint::

        POST {url}/exec
          {"command": str, "cwd": str, "timeout": float,
           "stdin": str?, "background": bool?, "log_path": str?}
        → {"exit_code": int, "stdout": str, "stderr": str, "pid": int?}

    Keeping the backend protocol this small means a provider integration is a
    ~20-line shim, and the agent gets serverless hibernate-on-idle execution
    without any provider SDK baked into this repo.
    """

    def __init__(self, url: str, token: str = "", workdir: str = "/workspace") -> None:
        if not url:
            raise RuntimeError("AGENT_SANDBOX_EXEC_URL is required for the http sandbox")
        self._url = url
        self._token = token
        self._workdir = workdir

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def _post(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        resp = httpx.post(
            f"{self._url}/exec", json=payload, headers=self._headers(), timeout=timeout
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data

    def start(self) -> None:
        # Verify reachability and ensure the workspace exists.
        self._post(
            {"command": f"mkdir -p {shlex.quote(self._workdir)}", "cwd": "/", "timeout": 15},
            timeout=20,
        )
        logger.info("HTTP-exec sandbox ready at %s", self._url)

    def stop(self) -> None:
        pass  # the remote service owns lifecycle/hibernation

    async def exec_async(
        self, command: str, cwd: str, timeout: float
    ) -> tuple[int, str, str]:
        def _run() -> tuple[int, str, str]:
            try:
                data = self._post(
                    {"command": command, "cwd": cwd, "timeout": timeout}, timeout + 5
                )
            except httpx.TimeoutException:
                raise
            except Exception as exc:
                return 1, "", f"http sandbox error: {exc}"
            return (
                int(data.get("exit_code", 0)),
                str(data.get("stdout", "")),
                str(data.get("stderr", "")),
            )

        try:
            return await asyncio.to_thread(_run)
        except httpx.TimeoutException as exc:
            raise TimeoutError from exc

    def exec_background(self, command: str, cwd: str, log_path: str) -> int | None:
        try:
            data = self._post(
                {
                    "command": command,
                    "cwd": cwd,
                    "timeout": 15,
                    "background": True,
                    "log_path": log_path,
                },
                timeout=20,
            )
            pid = data.get("pid")
            return int(pid) if pid is not None else None
        except Exception as exc:
            logger.warning("http sandbox exec_background failed: %s", exc)
            return None

    _evidence_scope = "http_sandbox"

    def _run_python(
        self, script: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        try:
            data = self._post(
                {
                    "command": f"python3 -c {shlex.quote(script)}",
                    "cwd": self._workdir,
                    "timeout": timeout,
                    "stdin": json.dumps(payload),
                },
                timeout + 5,
            )
        except httpx.TimeoutException as exc:
            raise TimeoutError from exc
        if int(data.get("exit_code", 0)) != 0:
            raise RuntimeError(data.get("stderr") or "http sandbox script failed")
        return json.loads(data.get("stdout") or "{}")

    def collect_environment_json(self) -> str:
        probe = (
            "import json,shutil; "
            "rts=['python3','node','npm','git','curl','apt-get']; "
            "print(json.dumps({'os':'Linux','shell':{'shell':'bash','posix':True},"
            "'sandbox':{'mode':'http'},"
            "'runtimes':{r:shutil.which(r) is not None for r in rts}}))"
        )
        try:
            data = self._post(
                {"command": f"python3 -c {shlex.quote(probe)}", "cwd": self._workdir, "timeout": 15},
                timeout=20,
            )
            out = str(data.get("stdout", "")).strip()
            if int(data.get("exit_code", 0)) == 0 and out:
                return out
        except Exception as exc:
            logger.warning("http sandbox environment probe failed: %s", exc)
        return json.dumps(
            {
                "os": "Linux",
                "sandbox": {"mode": "http", "url": self._url},
                "shell": {"shell": "bash", "posix": True},
            },
            indent=2,
        )


GET_SYSTEM_ENVIRONMENT_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "get_system_environment",
    "description": (
        "Return a JSON snapshot of the active execution environment. Includes: OS type, "
        "disk space, which runtimes are on PATH (rustc, cargo, gcc, make, curl, wget, "
        "git, node, npm, npx, python, docker, and more), the active shell, and critically "
        "the running user identity with is_root and sudo_available flags. "
        "ALWAYS call this first before attempting any package install (apt-get, "
        "rustup, npm, pip) so you know whether system-level installs will succeed "
        "or whether you need a per-user install path."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

GET_FILESYSTEM_PROCESS_EVIDENCE_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "get_filesystem_process_evidence",
    "description": (
        "Return structured evidence about active-environment filesystem paths, process IDs, "
        "process names, localhost ports, and the background-service log. Use this "
        "after creating files, folders, servers, or background processes to prove "
        "the requested artifacts or service exist before giving a final answer. "
        "IMPORTANT: do NOT call this in a loop to wait for a slow service to start. "
        "If a port is not yet open, use wait_for_port instead — it blocks internally "
        "and returns only when the port opens (or the timeout expires), so you do "
        "not burn API calls polling."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filesystem paths to inspect. Relative paths resolve from the current tool working directory.",
            },
            "pids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Process IDs to check for liveness.",
            },
            "process_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Process name fragments to search for in the process table.",
            },
            "ports": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Local TCP ports to probe on 127.0.0.1.",
            },
            "include_background_log": {
                "type": "boolean",
                "description": f"Include the tail of {_AGENT_BACKGROUND_LOG_PATH}. Default true.",
            },
        },
        "additionalProperties": False,
    },
}

WAIT_FOR_PORT_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "wait_for_port",
    "description": (
        "Block until a local TCP port is reachable or the timeout expires, then return "
        "the result. Use this INSTEAD of calling get_filesystem_process_evidence in a "
        "loop after starting a slow service (e.g. a JVM server, a Rust binary, or any "
        "service that takes tens of seconds to bind). A single wait_for_port call "
        "handles the entire wait internally without burning API calls — it polls the "
        "port every few seconds until it opens or the timeout is reached. "
        "Returns {open, port, elapsed_seconds} on success or "
        "{open: false, port, timeout: true, elapsed_seconds, hint} on timeout."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["port"],
        "properties": {
            "port": {
                "type": "integer",
                "description": "Local TCP port to wait for on 127.0.0.1.",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum seconds to wait. Default 120. Capped at 300.",
            },
            "interval": {
                "type": "integer",
                "description": "Polling interval in seconds. Default 5. Min 1, max 30.",
            },
        },
        "additionalProperties": False,
    },
}

WRITE_TEXT_FILE_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "write_text_file",
    "description": (
        "Create or overwrite a UTF-8 text file in the active workspace. Parent "
        "directories are created automatically. Use this for concrete artifacts "
        "such as HTML, CSS, JavaScript, Markdown, JSON, config files, or docs; "
        "then verify the artifact as required by the task contract."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["path", "content"],
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to create or overwrite. Relative paths resolve from the current tool working directory.",
            },
            "content": {
                "type": "string",
                "description": "Complete UTF-8 text content to write to the file.",
            },
        },
        "additionalProperties": False,
    },
}

SET_TASK_CONTRACT_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "set_task_contract",
    "description": (
        "Declare how the current user task must be completed before doing any "
        "work. Use mode='answer' for pure Q&A and mode='execute' for tasks that "
        "must change files, services, databases, or other active-environment state. The engine "
        "uses this contract to decide whether a final text answer is acceptable. "
        "Optionally specify toolset to narrow the tools available for this task: "
        "'research' (web + read-only), 'coding' (files + terminal + web), "
        "'web' (browser + fetch + files), 'data' (SQLite + terminal), "
        "'ops' (shell + files), 'all' (everything, default)."
    ),
    "inputSchema": {
        "type": "object",
        "required": [
            "mode",
            "summary",
            "success_criteria",
            "evidence_requirements",
        ],
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["answer", "execute"],
                "description": "Whether the task is answered in text or requires tool-side execution evidence.",
            },
            "summary": {
                "type": "string",
                "description": "One concise sentence describing the current user task.",
            },
            "success_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete conditions that must be true before the task is complete.",
            },
            "evidence_requirements": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "filesystem_artifact",
                        "running_http_service",
                        "running_tcp_service",
                        "database_mutation",
                        "command_output",
                        "none",
                    ],
                },
                "description": (
                    "Structured proof the engine should require before accepting a "
                    "final answer. Use 'none' only for answer-mode tasks. Use "
                    "'running_http_service' for HTTP servers that should be exposed "
                    "through the browser proxy, and 'running_tcp_service' for "
                    "non-HTTP servers proven by process/port evidence."
                ),
            },
            "toolset": {
                "type": "string",
                "enum": ["all", "research", "coding", "web", "data", "ops"],
                "description": (
                    "Optional: narrow available tools to this category. "
                    "research=web+read-only, coding=files+terminal+web, "
                    "web=browser+fetch+files, data=SQLite+terminal, ops=shell+files. "
                    "Default 'all'."
                ),
            },
        },
        "additionalProperties": False,
    },
}

EXPOSE_LOCAL_HTTP_SERVICE_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "expose_local_http_service",
    "description": (
        "Return a browser-reachable URL for an HTTP service already running inside "
        "the agent container. The backend proxies /proxy/<port>/... through the "
        "existing localhost:8000 Docker mapping, so no manual docker-compose port "
        "editing is needed. Use after starting dev servers, APIs, dashboards, local "
        "UIs, or any other HTTP service with execute_background_service."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["port"],
        "properties": {
            "port": {
                "type": "integer",
                "minimum": 1,
                "maximum": 65535,
                "description": "The internal container TCP port the HTTP service is listening on.",
            },
            "path": {
                "type": "string",
                "description": "Optional path to append after /proxy/<port>/. Defaults to /.",
            },
            "name": {
                "type": "string",
                "description": "Optional human-readable service name for the returned metadata.",
            },
        },
        "additionalProperties": False,
    },
}

EXECUTE_TERMINAL_COMMAND_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "execute_terminal_command",
    "description": (
        "Execute a shell command in the active execution environment and return its "
        "combined stdout and stderr output. Commands run inside that environment's "
        "default shell, which is the Docker sandbox shell when sandbox mode is active, "
        "and this call WAITS for them to finish. "
        f"Use this for slow-but-finite work too -- package installs (apt-get/pip), "
        f"downloads, builds, and tests -- it allows up to {_TERMINAL_TIMEOUT_SECONDS} "
        "seconds, so let them run to completion here instead of backgrounding them. "
        "Only commands that never terminate on their own (servers, daemons, watchers) "
        "belong in execute_background_service. A command that exceeds the timeout is "
        "killed and returns an error."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["command"],
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute, exactly as you would type it in a terminal.",
            },
        },
        "additionalProperties": False,
    },
}

EXECUTE_BACKGROUND_SERVICE_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "execute_background_service",
    "description": (
        "Launch a process that NEVER terminates on its own -- a server, daemon, or "
        "watcher -- in the background WITHOUT waiting. The process is detached, its "
        f"combined stdout/stderr is appended to {_AGENT_BACKGROUND_LOG_PATH}, and the new "
        "PID is returned immediately. "
        "Use this ONLY for non-terminating processes. Do NOT use it for finite work "
        "such as package installs (apt-get/pip), downloads, or builds -- those finish "
        "on their own, so run them with execute_terminal_command and let it wait; "
        "backgrounding them just forces you to poll this log and can deadlock on "
        "resource locks (e.g. apt/dpkg). Never launch the same install or service "
        "more than once. After starting a service, check "
        f"'cat {_AGENT_BACKGROUND_LOG_PATH}' at most once or twice -- do not poll it in a loop."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["command"],
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command that starts the background process.",
            },
        },
        "additionalProperties": False,
    },
}

EXPAND_TOOL_OUTPUT_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "expand_tool_output",
    "description": (
        "Retrieve the full output of an earlier tool call that was condensed to a "
        "head/tail preview in this conversation. Pass the handle shown in the "
        "'[... hidden ...]' truncation notice (it is that call's tool_call_id). Page "
        "through long output with start_line and max_lines. Prefer this over re-running "
        "a command when you only need to see more of a result you already produced."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["handle"],
        "properties": {
            "handle": {
                "type": "string",
                "description": "The tool_call_id handle printed in the truncation notice.",
            },
            "start_line": {
                "type": "integer",
                "description": "0-based line to start from. Default 0.",
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to return. Default 200.",
            },
        },
        "additionalProperties": False,
    },
}

UPDATE_PLAN_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "update_plan",
    "description": (
        "Create or replace your working plan for the current task: an ordered "
        "checklist of concrete steps, each with a status. After set_task_contract, "
        "call this on any multi-step execute-mode task, then call it again to update "
        "statuses as you progress. "
        "Mark a step 'in_progress' when you start it and 'done' or 'failed' when it "
        "finishes. Always pass the FULL list of steps; it replaces the previous plan. "
        "The plan is shown back to you every turn, so it is how you avoid repeating "
        "finished steps and how you track what still remains. Do not give a final "
        "answer until every step is 'done' or explicitly 'failed'."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["steps"],
        "properties": {
            "steps": {
                "type": "array",
                "description": "The full ordered list of steps for the task.",
                "items": {
                    "type": "object",
                    "required": ["title", "status"],
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Short, concrete description of the step.",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "done", "failed"],
                            "description": "Current status of this step.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    },
}

DELEGATE_TASK_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "delegate_task",
    "description": (
        "Delegate one or more self-contained tasks to specialised sub-agents.\n\n"
        "Single-task form (backward compat):\n"
        "  {agent_type, task_description, context_payload}\n\n"
        "Parallel batch form (run multiple sub-agents concurrently):\n"
        "  {tasks: [{agent_type, task_description, context_payload},...], mode: 'parallel'}\n\n"
        "Agent types:\n"
        "  'researcher' — gather/synthesise information (no side effects)\n"
        "  'coder'      — write or refactor code\n"
        "  'auditor'    — security/correctness review\n"
        "  'planner'    — break a goal into ordered, verifiable steps\n\n"
        "Sub-agents do not share conversation history. Pass relevant context in "
        "context_payload. For filesystem/process operations, use "
        "execute_terminal_command directly instead."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "agent_type": {
                "type": "string",
                "enum": ["researcher", "coder", "auditor", "planner"],
                "description": "Sub-agent type for single-task form.",
            },
            "task_description": {
                "type": "string",
                "description": "Task instruction for single-task form.",
            },
            "context_payload": {
                "type": "object",
                "description": "Background facts for single-task form.",
                "additionalProperties": True,
            },
            "tasks": {
                "type": "array",
                "description": "List of tasks for batch form.",
                "items": {
                    "type": "object",
                    "required": ["agent_type", "task_description"],
                    "properties": {
                        "agent_type": {"type": "string"},
                        "task_description": {"type": "string"},
                        "context_payload": {"type": "object", "additionalProperties": True},
                    },
                },
            },
            "mode": {
                "type": "string",
                "enum": ["sequential", "parallel"],
                "description": "Execution mode for batch form. Default: sequential.",
            },
        },
        "additionalProperties": False,
    },
}


ANALYZE_IMAGE_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "analyze_image",
    "description": (
        "Analyze an image using a vision-capable LLM. Pass a public URL or a "
        "file:// path to an image, and ask a question about its contents. "
        "Returns the model's answer as plain text. Requires AGENT_VISION_MODEL "
        "to be set to a vision-capable model (e.g. gpt-4o-mini)."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["image_url", "question"],
        "properties": {
            "image_url": {
                "type": "string",
                "description": "Public URL or base64 data URI of the image to analyze.",
            },
            "question": {
                "type": "string",
                "description": "What to ask about the image.",
            },
        },
        "additionalProperties": False,
    },
}

GENERATE_IMAGE_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "generate_image",
    "description": (
        "Generate an image from a text prompt using DALL-E or a compatible API. "
        "Requires OPENAI_API_KEY and AGENT_IMAGE_GEN_MODEL (default: dall-e-3). "
        "Returns a JSON object with the image URL and optional local save path."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Detailed description of the image to generate.",
            },
            "size": {
                "type": "string",
                "enum": ["1024x1024", "1792x1024", "1024x1792"],
                "description": "Image dimensions. Default: 1024x1024.",
            },
            "save_path": {
                "type": "string",
                "description": (
                    "Optional workspace path to download and save the image "
                    "(e.g. /workspace/output.png). Omit to return only the URL."
                ),
            },
        },
        "additionalProperties": False,
    },
}

LIST_SCHEDULED_TASKS_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "list_scheduled_tasks",
    "description": (
        "List all jobs registered with the agent's internal cron scheduler. "
        "Returns job IDs, schedule type and spec, labels, next-run times, "
        "enabled status, and run counts. Use this to verify a schedule_task "
        "call succeeded, inspect what recurring tasks are active, or find a "
        "job_id before disabling one via the /api/cron/jobs API."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

LIST_SKILLS_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "list_skills",
    "description": (
        "List all skills in the agent's skill registry with their description, "
        "tags, usage count, success rate, and self-improvement history. "
        "Use this to see which skills exist, how often they have been called, "
        "and whether they qualify for LLM self-improvement."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

CREATE_SKILL_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "create_skill",
    "description": (
        "Author a NEW reusable skill on demand and register it immediately so "
        "future turns can call it. Use this when you've worked out a reusable "
        "procedure and want to save it as a callable tool (the auto skill maker), "
        "instead of waiting for post-task distillation.\n"
        "The code MUST: start with `from _skill import skill`, decorate the "
        "function with `@skill`, define a single well-named function with typed "
        "parameters and a docstring, and return a JSON-serialisable result. Keep "
        "it self-contained (imports inside the function) and side-effect-aware.\n"
        "Example code:\n"
        "from _skill import skill\n\n"
        "@skill\n"
        "def count_lines(path: str) -> int:\n"
        "    \"Return the number of lines in a file.\"\n"
        "    with open(path) as f:\n"
        "        return sum(1 for _ in f)\n"
    ),
    "inputSchema": {
        "type": "object",
        "required": ["name", "code"],
        "properties": {
            "name": {"type": "string", "description": "Short snake_case skill name."},
            "code": {"type": "string", "description": "Full @skill-decorated Python source."},
            "description": {"type": "string", "description": "One-line summary of the skill."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for discovery.",
            },
        },
        "additionalProperties": False,
    },
}

RECALL_MEMORY_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "recall_memory",
    "description": (
        "Search your own past conversations (across ALL prior sessions) by "
        "keyword and get the matching messages back. Use this to remember what "
        "the user told you before, decisions you made, or context from earlier "
        "chats — e.g. 'what did we decide about the database schema?'. "
        "Returns matching turns with a snippet, role, session, and timestamp."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "Keywords to search for."},
            "limit": {"type": "integer", "description": "Max results (default 8)."},
        },
        "additionalProperties": False,
    },
}

SCHEDULE_TASK_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "schedule_task",
    "description": (
        "Schedule a task to run automatically on a recurring schedule or at a "
        "specific future time. The task is dispatched to the agent and runs "
        "unattended. Returns the job ID. Use list_scheduled_tasks to verify.\n"
        "IMPORTANT: use this (not execute_background_service) for any work that "
        "must repeat on a time-based schedule — heartbeats, periodic reports, "
        "cleanup jobs. execute_background_service is for persistent daemons "
        "(servers/watchers) that run continuously, not for scheduled work.\n"
        "schedule_type options:\n"
        "  'interval' — run every N seconds (spec: '300' = every 5 min)\n"
        "  'cron'     — 5-field cron expression (spec: '0 9 * * 1' = every Monday 9am)\n"
        "  'once'     — run once at an ISO datetime (spec: '2026-06-01T12:00:00Z')\n"
    ),
    "inputSchema": {
        "type": "object",
        "required": ["prompt", "schedule_type", "schedule_spec"],
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The task instruction to run on schedule.",
            },
            "schedule_type": {
                "type": "string",
                "enum": ["interval", "cron", "once"],
                "description": "How the schedule repeats.",
            },
            "schedule_spec": {
                "type": "string",
                "description": (
                    "The schedule specification: seconds for 'interval', "
                    "5-field cron string for 'cron', ISO datetime for 'once'."
                ),
            },
            "label": {
                "type": "string",
                "description": "Optional human-readable name for the job.",
            },
        },
        "additionalProperties": False,
    },
}


class ToolManager:
    """Manages connections to one or more MCP servers over STDIO.

    Each server is launched as a subprocess and kept alive inside an
    ``AsyncExitStack``.  Use as an async context manager to guarantee cleanup::

        async with ToolManager() as tm:
            await tm.connect_server("fs", "npx", ["-y", "@modelcontextprotocol/server-filesystem", "."])
            tools = await tm.list_all_tools()
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ClientSession] = {}
        self._stacks: dict[str, AsyncExitStack] = {}
        # Snapshot of list_all_tools(), rebuilt lazily. The tool set only changes
        # when a server connects or disconnects, so caching it avoids a
        # list_tools round-trip to every MCP server on every agent turn.
        self._tools_cache: list[dict[str, Any]] | None = None
        default_cwd = Path(os.getenv("AGENT_WORKDIR", str(_PROJECT_ROOT))).expanduser()
        if not default_cwd.exists():
            default_cwd = _PROJECT_ROOT
        host_workdir = str(default_cwd.resolve())

        # Sandbox — started eagerly so the environment is ready before the first
        # terminal command and get_system_environment reflects the actual runtime.
        self._sandbox: _DockerSandbox | _SSHSandbox | None = None
        self._sandbox_startup_failed: bool = False
        self._host_execution_disabled_reason: str | None = None
        if _SANDBOX_ENABLED:
            try:
                sandbox = _DockerSandbox(image=_SANDBOX_IMAGE, host_workdir=host_workdir)
                sandbox.start()
                self._sandbox = sandbox
                self.current_cwd: str = _SANDBOX_WORKDIR
                self._env_snapshot: str = sandbox.collect_environment_json()
                logger.info("Sandbox mode: docker (image=%s)", _SANDBOX_IMAGE)
            except Exception as exc:
                self._sandbox = None
                self._sandbox_startup_failed = True
                self.current_cwd = host_workdir
                self._env_snapshot = _collect_system_environment()
                if not _SANDBOX_FALLBACK_TO_HOST:
                    logger.error("Docker sandbox failed to start; host execution is blocked: %s", exc)
                    self._host_execution_disabled_reason = (
                        "Docker sandbox failed to start and "
                        "AGENT_SANDBOX_HOST_FALLBACK is not enabled. Terminal, "
                        "background-service, process, and port tools are blocked "
                        "so the agent cannot accidentally mutate the host OS. "
                        f"Root cause: {exc}"
                    )
                else:
                    logger.error("Docker sandbox failed — falling back to host execution: %s", exc)
        elif _SSH_SANDBOX_ENABLED:
            try:
                ssh = _SSHSandbox(
                    host=_SSH_HOST,
                    user=_SSH_USER,
                    key_path=_SSH_KEY_PATH,
                    workdir=_SSH_WORKDIR,
                    port=_SSH_PORT,
                )
                ssh.start()
                self._sandbox = ssh
                self.current_cwd: str = _SSH_WORKDIR
                self._env_snapshot: str = ssh.collect_environment_json()
                logger.info("Sandbox mode: ssh (%s@%s)", _SSH_USER, _SSH_HOST)
            except Exception as exc:
                self._sandbox = None
                self._sandbox_startup_failed = True
                self.current_cwd = host_workdir
                self._env_snapshot = _collect_system_environment()
                logger.error("SSH sandbox failed to start — falling back to host: %s", exc)
        elif _HTTP_SANDBOX_ENABLED:
            try:
                http_sandbox = _HttpExecSandbox(
                    url=_HTTP_EXEC_URL,
                    token=_HTTP_EXEC_TOKEN,
                    workdir=_HTTP_EXEC_WORKDIR,
                )
                http_sandbox.start()
                self._sandbox = http_sandbox
                self.current_cwd = _HTTP_EXEC_WORKDIR
                self._env_snapshot = http_sandbox.collect_environment_json()
                logger.info("Sandbox mode: http (%s)", _HTTP_EXEC_URL)
            except Exception as exc:
                self._sandbox = None
                self._sandbox_startup_failed = True
                self.current_cwd = host_workdir
                self._env_snapshot = _collect_system_environment()
                logger.error("HTTP sandbox failed to start — falling back to host: %s", exc)
        else:
            self.current_cwd = host_workdir
            self._env_snapshot = _collect_system_environment()

        self.public_base_url = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
        self._host_workdir: str = host_workdir
        # Lazy browser session — started on first browser_* tool call.
        self._browser_session: _BrowserSession | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def connect_server(
        self,
        name: str,
        command: str,
        args: list[str],
    ) -> None:
        """Spin up an MCP server subprocess and initialise a session with it.

        If *name* is already registered the old server is shut down first.
        The connection stays alive until :meth:`disconnect_server`, :meth:`close`,
        or the async context manager exits.
        """
        if name in self._stacks:
            await self._disconnect(name)

        stack = AsyncExitStack()
        params = StdioServerParameters(command=command, args=args)

        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(params)
        )
        session: ClientSession = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        self._sessions[name] = session
        self._stacks[name] = stack
        self._tools_cache = None  # tool set changed; force a rebuild next call

    async def list_all_tools(self) -> list[dict[str, Any]]:
        """Return the JSON schemas of every tool across all connected servers.

        Follows pagination cursors so the full tool list is always returned
        regardless of server-side page size.

        Each entry has the shape::

            {
                "server":       "<name passed to connect_server>",
                "name":         "<tool name>",
                "description":  "<human-readable description>",   # omitted if absent
                "inputSchema":  { ... },   # JSON Schema for the tool's parameters
                "outputSchema": { ... },   # omitted if absent
            }

        The assembled list is cached and reused until a server connects or
        disconnects, so repeated calls (one per agent turn) don't re-query every
        MCP server.
        """
        if self._tools_cache is not None:
            return self._tools_cache

        results: list[dict[str, Any]] = []

        for server_name, session in self._sessions.items():
            cursor: str | None = None
            while True:
                page = await session.list_tools(
                    params=PaginatedRequestParams(cursor=cursor) if cursor else None
                )
                for tool in page.tools:
                    entry: dict[str, Any] = {
                        "server": server_name,
                        "name": tool.name,
                        "inputSchema": tool.inputSchema,
                    }
                    if tool.description is not None:
                        entry["description"] = tool.description
                    if tool.outputSchema is not None:
                        entry["outputSchema"] = tool.outputSchema
                    results.append(entry)

                cursor = page.nextCursor
                if not cursor:
                    break

        results.append(GET_SYSTEM_ENVIRONMENT_TOOL)
        results.append(GET_FILESYSTEM_PROCESS_EVIDENCE_TOOL)
        results.append(WAIT_FOR_PORT_TOOL)
        results.append(WEB_FETCH_TOOL)
        results.append(WEB_SEARCH_TOOL)
        # Browser tools only appear when Playwright is installed.
        if _PLAYWRIGHT_AVAILABLE:
            results.extend(_BROWSER_TOOL_SCHEMAS)
        results.append(WRITE_TEXT_FILE_TOOL)
        results.append(SET_TASK_CONTRACT_TOOL)
        results.append(EXPOSE_LOCAL_HTTP_SERVICE_TOOL)
        results.append(EXECUTE_TERMINAL_COMMAND_TOOL)
        results.append(EXECUTE_BACKGROUND_SERVICE_TOOL)
        results.append(EXPAND_TOOL_OUTPUT_TOOL)
        results.append(UPDATE_PLAN_TOOL)
        results.append(DELEGATE_TASK_TOOL)
        # Vision / image-gen tools only appear when a vision model is configured.
        if _VISION_MODEL:
            results.append(ANALYZE_IMAGE_TOOL)
        results.append(GENERATE_IMAGE_TOOL)
        # Schedule tool + introspection always available.
        results.append(SCHEDULE_TASK_TOOL)
        results.append(LIST_SCHEDULED_TASKS_TOOL)
        results.append(LIST_SKILLS_TOOL)
        results.append(CREATE_SKILL_TOOL)
        results.append(RECALL_MEMORY_TOOL)
        self._tools_cache = results
        return results

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        """Invoke *tool_name* on the named server with *arguments*.

        Raises ``KeyError`` if *server_name* is not a connected server.
        """
        session = self._sessions.get(server_name)
        if session is None:
            raise KeyError(f"No connected server named {server_name!r}")
        return await session.call_tool(tool_name, arguments)

    async def connect_filesystem_server(
        self,
        data_dir: Path | str | None = None,
    ) -> None:
        """Connect to the ``@modelcontextprotocol/server-filesystem`` MCP server.

        The server is launched via ``npx`` and granted access to *data_dir*
        only -- it cannot read or write outside that path.

        *data_dir* defaults to ``<project_root>/data``.  The directory is created if it
        does not already exist.  The server is registered under the name
        ``"filesystem"``; a previous ``"filesystem"`` connection is
        disconnected before the new one is opened.

        Raises ``RuntimeError`` if ``npx`` is not available on PATH.
        """
        path = Path(data_dir).resolve() if data_dir else _PROJECT_ROOT / "data"
        path.mkdir(parents=True, exist_ok=True)

        npx = shutil.which("npx")
        if npx is None:
            raise RuntimeError(
                "npx is not installed or is not available on PATH; "
                "install Node.js to use the filesystem MCP server"
            )

        await self.connect_server(
            name="filesystem",
            command=npx,
            args=["-y", "@modelcontextprotocol/server-filesystem", str(path)],
        )

    async def connect_skills_server(
        self,
        skills_dir: Path | str | None = None,
    ) -> None:
        """Connect to the local Python MCP server that serves the ``skills/`` directory.

        On every call the subprocess is launched fresh, so any ``.py`` file the
        Evaluator wrote to *skills_dir* since the last run is automatically
        picked up -- no manual registration required.

        *skills_dir* defaults to ``<project_root>/skills/``.  If the directory
        or either bootstrap file (``server.py``, ``_skill.py``) does not exist
        they are created so the server is always runnable on first call.

        The server is registered under the name ``"skills"``; a previous
        ``"skills"`` connection is disconnected before the new one is opened.
        """
        path = Path(skills_dir).resolve() if skills_dir else _PROJECT_ROOT / "skills"
        _ensure_skills_dir(path)
        await self.connect_server(
            name="skills",
            command=sys.executable,
            args=[str(path / "server.py")],
        )

    def get_system_environment(self) -> str:
        """Return the cached JSON environment snapshot collected at init time."""
        return self._env_snapshot

    @property
    def sandbox_active(self) -> bool:
        return self._sandbox is not None

    @property
    def sandbox_startup_failed(self) -> bool:
        """True when AGENT_SANDBOX=docker was set but Docker could not be reached.

        Exposed so the session-seeding environment message can warn the agent (and
        the operator) that commands run on the host, not inside a container.
        """
        return self._sandbox_startup_failed

    @property
    def host_execution_disabled_reason(self) -> str | None:
        return self._host_execution_disabled_reason

    def _host_execution_blocked_result(self, command: str | None = None) -> dict[str, Any]:
        reason = self._host_execution_disabled_reason or "Host execution is disabled."
        command_line = f"\nCommand was not executed: {command}" if command else ""
        return {
            "exit_code": -1,
            "stdout": (
                "SYSTEM ALERT: Host execution is blocked because the requested Docker "
                "sandbox is unavailable. Do not retry shell variants on this host. "
                "Start Docker Desktop and restart the backend, or explicitly set "
                "AGENT_SANDBOX_HOST_FALLBACK=true if direct host execution is intended."
                f"{command_line}"
            ),
            "stderr": reason,
            "current_working_directory": self.current_cwd,
            "scope": "sandbox_unavailable",
        }

    def _wrong_environment_blocked_result(self, command: str, reason: str) -> dict[str, Any]:
        scope = "docker_sandbox" if self._sandbox is not None else "host"
        return {
            "exit_code": -1,
            "stdout": (
                "SYSTEM ALERT: Command blocked before execution because it targets "
                "the wrong operating environment. The command was not run anywhere. "
                f"{reason} Read get_system_environment and issue a command that fits "
                "the active execution environment; do not retry with shell wrappers."
                f"\nCommand was not executed: {command}"
            ),
            "stderr": reason,
            "current_working_directory": self.current_cwd,
            "scope": scope,
        }

    async def web_fetch(
        self,
        url: str,
        max_chars: int = 8_000,
        timeout: float = 15.0,
    ) -> str:
        """Fetch *url* and return readable text + metadata as JSON.

        HTML is stripped to plain text via :func:`_html_to_text`.  JSON and
        plain-text responses are returned as-is.  The result is always valid
        JSON with keys: url, status_code, content_type, text, word_count,
        truncated, total_chars.

        Requires ``httpx`` (already in requirements).
        """
        import httpx as _httpx

        max_chars = min(max(100, int(max_chars)), 40_000)
        timeout = min(max(1.0, float(timeout)), 60.0)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64; compatible; agent-ai/1.0)"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,text/plain;q=0.8,*/*;q=0.5"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            async with _httpx.AsyncClient(
                follow_redirects=True,
                timeout=_httpx.Timeout(timeout),
            ) as client:
                response = await client.get(url, headers=headers)
        except _httpx.TimeoutException:
            return json.dumps({
                "error": f"Request timed out after {timeout}s",
                "url": url,
            }, indent=2)
        except _httpx.RequestError as exc:
            return json.dumps({"error": f"Request failed: {exc}", "url": url}, indent=2)

        content_type = response.headers.get("content-type", "").lower()
        raw = response.text

        if "html" in content_type or raw.lstrip().lower().startswith("<!"):
            text = _html_to_text(raw)
        else:
            text = raw

        total_chars = len(text)
        return json.dumps(
            {
                "url": str(response.url),
                "status_code": response.status_code,
                "content_type": content_type,
                "text": text[:max_chars],
                "word_count": len(text.split()),
                "truncated": total_chars > max_chars,
                "total_chars": total_chars,
            },
            indent=2,
        )

    async def web_search(self, query: str, max_results: int = 8) -> str:
        """Search the web and return result titles, URLs, and snippets as JSON.

        Backend priority:
        1. ``duckduckgo_search`` library (optional install — best quality).
        2. DuckDuckGo HTML Lite scrape via ``httpx`` (no extra dependencies).
        3. Graceful error JSON with a hint to use ``web_fetch`` directly.

        Returns JSON with keys: query, backend, results[].
        Each result has: title, url, snippet.
        """
        import urllib.parse

        max_results = min(max(1, int(max_results)), 20)

        # ── Backend 1: duckduckgo_search library ────────────────────────────
        try:
            from duckduckgo_search import DDGS  # type: ignore[import-untyped]

            raw = await asyncio.to_thread(
                lambda: list(DDGS().text(query, max_results=max_results))
            )
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in raw
            ]
            return json.dumps(
                {"query": query, "backend": "duckduckgo", "results": results},
                indent=2,
            )
        except ImportError:
            pass  # library not installed — fall through to HTML scrape
        except Exception as exc:
            logger.warning("duckduckgo_search backend failed: %s", exc)

        # ── Backend 2: DuckDuckGo HTML Lite via httpx ────────────────────────
        try:
            import httpx as _httpx

            encoded = urllib.parse.quote_plus(query)
            ddg_url = f"https://html.duckduckgo.com/html/?q={encoded}"
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; agent-ai/1.0)",
                "Accept-Language": "en-US,en;q=0.9",
            }
            async with _httpx.AsyncClient(
                follow_redirects=True, timeout=_httpx.Timeout(15.0)
            ) as client:
                resp = await client.get(ddg_url, headers=headers)
            results = _parse_ddg_html(resp.text, max_results)
            return json.dumps(
                {"query": query, "backend": "duckduckgo_html", "results": results},
                indent=2,
            )
        except Exception as exc:
            logger.warning("DuckDuckGo HTML fallback also failed: %s", exc)

        # ── Backend 3: graceful degradation ──────────────────────────────────
        hint = (
            "https://html.duckduckgo.com/html/?q="
            + urllib.parse.quote_plus(query)
        )
        return json.dumps(
            {
                "query": query,
                "backend": "unavailable",
                "error": (
                    "All web search backends failed. "
                    f"Use web_fetch(url='{hint}') as an alternative."
                ),
                "results": [],
            },
            indent=2,
        )

    def get_filesystem_process_evidence(
        self,
        *,
        paths: list[str] | None = None,
        pids: list[int] | None = None,
        process_names: list[str] | None = None,
        ports: list[int] | None = None,
        include_background_log: bool = True,
    ) -> str:
        """Return JSON evidence for files/folders, processes, ports, and logs."""
        if self._host_execution_disabled_reason is not None:
            return json.dumps(
                {
                    "scope": "sandbox_unavailable",
                    "error": self._host_execution_disabled_reason,
                    "current_working_directory": self.current_cwd,
                    "paths": [],
                    "pids": [],
                    "process_names": [],
                    "ports": [],
                    "background_log": {"exists": False, "tail": ""},
                },
                indent=2,
            )
        if self._sandbox is not None:
            return json.dumps(
                self._sandbox.collect_evidence_json(
                    paths=paths,
                    pids=pids,
                    process_names=process_names,
                    ports=ports,
                    include_background_log=include_background_log,
                    cwd=self.current_cwd,
                ),
                indent=2,
            )

        payload: dict[str, Any] = {
            "current_working_directory": self.current_cwd,
            "paths": [
                _inspect_path(path, self.current_cwd)
                for path in (paths or [])
            ],
            "pids": [_inspect_pid(pid) for pid in (pids or [])],
            "process_names": [
                _inspect_process_name(name) for name in (process_names or [])
            ],
            "ports": [_inspect_port(port) for port in (ports or [])],
        }
        if include_background_log:
            payload["background_log"] = _tail_file(_BACKGROUND_LOG_PATH)
        return json.dumps(payload, indent=2)

    def write_text_file(self, path: str, content: str) -> str:
        """Create or overwrite a UTF-8 text file and return structured evidence."""
        if self._sandbox is not None:
            display_path = self._resolve_sandbox_file_path(path)
            result = self._sandbox.write_text_file(display_path, content)
            payload = {
                "written": True,
                "path": display_path,
                "exists": bool(result.get("exists")),
                "size_bytes": int(result.get("size_bytes") or 0),
                "current_working_directory": self.current_cwd,
                "scope": "docker_sandbox",
                "artifact_quality": _artifact_quality_for_text(content, display_path),
            }
            return json.dumps(payload, indent=2)

        target = self._resolve_file_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        payload = {
            "written": True,
            "path": str(target),
            "exists": target.is_file(),
            "size_bytes": target.stat().st_size if target.exists() else 0,
            "current_working_directory": self.current_cwd,
            "scope": "host",
            "artifact_quality": _artifact_quality_for_text(content, str(target)),
        }
        return json.dumps(payload, indent=2)

    def expose_local_http_service(
        self,
        port: int,
        path: str = "",
        name: str | None = None,
    ) -> str:
        """Return a public backend-proxy URL for an internal HTTP service."""
        if self._host_execution_disabled_reason is not None:
            raise ConnectionError(self._host_execution_disabled_reason)
        service_port = int(port)
        if service_port < 1 or service_port > 65535:
            raise ValueError(f"Port out of range: {service_port}")

        if self._sandbox is not None:
            evidence = self._sandbox.collect_evidence_json(
                ports=[service_port],
                include_background_log=False,
                cwd=self.current_cwd,
            )
            port_info = (evidence.get("ports") or [{}])[0]
            connectable = bool(port_info.get("connectable"))
            error = port_info.get("error")
            scope = "docker_sandbox"
        else:
            connectable = False
            error: str | None = None
            scope = "host"
            try:
                with socket.create_connection(("127.0.0.1", service_port), timeout=1.0):
                    connectable = True
            except OSError as exc:
                error = str(exc)

        if not connectable:
            raise ConnectionError(
                f"No HTTP service is reachable on 127.0.0.1:{service_port}: {error}"
            )

        clean_path = path.strip().lstrip("/")
        suffix = f"/{clean_path}" if clean_path else "/"
        payload = {
            "exposed": True,
            "name": name or f"local-http-{service_port}",
            "port": service_port,
            "path": suffix,
            "url": f"{self.public_base_url}/proxy/{service_port}{suffix}",
            "connectable": connectable,
            "scope": scope,
        }
        return json.dumps(payload, indent=2)

    def proxy_local_http_service(
        self,
        *,
        port: int,
        path: str,
        query: str,
        method: str,
        headers: dict[str, str],
        body: bytes,
    ) -> dict[str, Any]:
        """Fetch a proxied HTTP service from the same place tools run."""
        if self._sandbox is None:
            raise RuntimeError("proxy_local_http_service is only for sandbox mode")
        return self._sandbox.fetch_http_json(
            port=port,
            path=path,
            query=query,
            method=method,
            headers=headers,
            body_b64=base64.b64encode(body).decode("ascii"),
        )

    async def wait_for_port(
        self,
        port: int,
        timeout: int = 120,
        interval: int = 5,
    ) -> str:
        """Block asynchronously until *port* is reachable or *timeout* expires.

        Polls every *interval* seconds using a non-blocking async sleep so the
        event loop stays responsive.  Returns a JSON string with:
          ``{open, port, elapsed_seconds}``          on success, or
          ``{open: false, port, timeout, elapsed_seconds, hint}``  on timeout.
        """
        if self._host_execution_disabled_reason is not None:
            return json.dumps(
                {
                    "open": False,
                    "port": int(port),
                    "error": self._host_execution_disabled_reason,
                    "scope": "sandbox_unavailable",
                }
            )
        timeout = min(max(int(timeout), 1), 300)
        interval = min(max(int(interval), 1), 30)
        start = asyncio.get_event_loop().time()
        deadline = start + timeout

        while True:
            try:
                with socket.create_connection(("127.0.0.1", int(port)), timeout=1.0):
                    elapsed = round(asyncio.get_event_loop().time() - start, 1)
                    return json.dumps({"open": True, "port": port, "elapsed_seconds": elapsed})
            except OSError:
                pass

            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(interval, remaining))

        elapsed = round(asyncio.get_event_loop().time() - start, 1)
        return json.dumps({
            "open": False,
            "port": port,
            "timeout": True,
            "elapsed_seconds": elapsed,
            "hint": (
                f"Port {port} did not open within {timeout}s. "
                f"Read the background log (cat {_AGENT_BACKGROUND_LOG_PATH}) "
                "to diagnose the failure before retrying."
            ),
        })

    async def execute_terminal_command(self, command: str) -> dict[str, Any]:
        """Run *command* inside ``self.current_cwd`` and return a structured result dict.

        If the command begins with ``cd <path>``, ``self.current_cwd`` is updated
        immediately and the remainder of the command (after an optional ``&&``) is
        executed in the new directory via the ``cwd=`` parameter on the subprocess --
        no shell-level ``cd &&`` wrapping is used.  A bare ``cd <path>`` with nothing
        following it returns instantly without spawning a process.

        Any additional ``cd`` calls embedded later in the command (e.g.
        ``mkdir /foo && cd /foo``) are tracked via ``_resolve_cd_target`` after
        a successful exit so that ``self.current_cwd`` always reflects where the
        shell would have landed.

        Keys in the returned dict:
          exit_code                 -- integer return code (âˆ'1 on timeout/unknown)
          stdout                    -- decoded stdout (SYSTEM ALERT prepended when
                                      exit_code > 0)
          stderr                    -- decoded stderr
          current_working_directory -- ``self.current_cwd`` after all cd tracking
        """
        if self._host_execution_disabled_reason is not None:
            return self._host_execution_blocked_result(command)

        # Safety gate: block unconditionally dangerous command patterns.
        if _is_dangerous_command(command):
            return {
                "exit_code": -1,
                "stdout": (
                    "SYSTEM ALERT: Command blocked for safety reasons. "
                    "The requested operation matches a blocked pattern and was not executed."
                ),
                "stderr": "",
                "current_working_directory": self.current_cwd,
            }

        mismatch_reason = _wrong_environment_command_reason(
            command,
            sandbox_active=self._sandbox is not None,
        )
        if mismatch_reason:
            return self._wrong_environment_blocked_result(command, mismatch_reason)

        # â"€â"€ Step 1: intercept a leading `cd <path>` â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
        cd_match = _LEADING_CD_RE.match(command)
        if cd_match:
            raw = next(g for g in cd_match.groups() if g is not None)
            if self._sandbox is not None:
                resolved = _resolve_sandbox_cd(f"cd {raw}", self.current_cwd)
            else:
                resolved = _resolve_cd_target(f"cd {raw}", self.current_cwd)
            if resolved is not None:
                self.current_cwd = resolved

            rest = command[cd_match.end():].strip()
            if not rest:
                # Pure `cd` with no trailing command -- no subprocess needed.
                return {
                    "exit_code": 0,
                    "stdout": f"Changed directory to {self.current_cwd}",
                    "stderr": "",
                    "current_working_directory": self.current_cwd,
                }
            command = rest  # run only what follows the cd

        # â"€â"€ Step 2: run the (possibly trimmed) command in current_cwd â"€â"€â"€â"€â"€â"€â"€â"€
        cwd_snapshot = self.current_cwd

        if self._sandbox is not None:
            # ── Sandbox path: run inside the Docker container ──────────────
            try:
                exit_code, stdout, stderr = await self._sandbox.exec_async(
                    command, cwd_snapshot, _TERMINAL_TIMEOUT_SECONDS
                )
            except TimeoutError:
                return {
                    "exit_code": -1,
                    "stdout": (
                        f"SYSTEM ALERT: Command execution failed with error code -1. "
                        f"You must troubleshoot this failure before executing subsequent "
                        f"commands.\nCommand exceeded the {_TERMINAL_TIMEOUT_SECONDS}s "
                        f"timeout and was killed inside the Docker sandbox. Only move a "
                        f"command to execute_background_service if it NEVER terminates "
                        f"(a server/daemon). A finite slow command should be re-run or "
                        f"split into smaller steps."
                    ),
                    "stderr": "",
                    "current_working_directory": cwd_snapshot,
                }
        else:
            # ── Host path: run directly on the OS ──────────────────────────
            try:
                # On Windows use bash (Git Bash / WSL) when available so POSIX
                # commands (mkdir -p, cat, etc.) work correctly.  On POSIX systems
                # create_subprocess_shell already uses /bin/sh.
                if _POSIX_SHELL is not None:
                    proc = await asyncio.create_subprocess_exec(
                        *_POSIX_SHELL, command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=cwd_snapshot,
                    )
                else:
                    proc = await asyncio.create_subprocess_shell(
                        command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=cwd_snapshot,
                    )
            except (FileNotFoundError, NotADirectoryError) as exc:
                return {
                    "exit_code": -1,
                    "stdout": (
                        f"SYSTEM ALERT: Command execution failed with error code -1. "
                        f"You must troubleshoot this failure before executing "
                        f"subsequent commands.\n"
                        f"Working directory does not exist: {cwd_snapshot} ({exc})"
                    ),
                    "stderr": "",
                    "current_working_directory": cwd_snapshot,
                }

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=_TERMINAL_TIMEOUT_SECONDS
                )
            except asyncio.CancelledError:
                proc.kill()
                try:
                    await proc.wait()
                except ProcessLookupError:
                    pass
                raise
            except TimeoutError:
                # wait_for only cancels the communicate() await -- the spawned shell
                # keeps running and would otherwise leak and hold the event loop.
                # Physically kill it, then reap so it does not become a zombie.
                proc.kill()
                try:
                    await proc.wait()
                except ProcessLookupError:
                    pass
                return {
                    "exit_code": -1,
                    "stdout": (
                        f"SYSTEM ALERT: Command execution failed with error code -1. "
                        f"You must troubleshoot this failure before executing subsequent "
                        f"commands.\nCommand exceeded the {_TERMINAL_TIMEOUT_SECONDS}s "
                        f"timeout and was killed. Only move a command to "
                        f"execute_background_service if it is a process that NEVER "
                        f"terminates (a server/daemon/watcher). A finite command that is "
                        f"merely slow (install, download, build) should NOT be "
                        f"backgrounded -- re-run it here, or split it into smaller steps; "
                        f"do not poll a background log waiting for it."
                    ),
                    "stderr": "",
                    "current_working_directory": cwd_snapshot,
                }

            stdout = stdout_bytes.decode(errors="replace")
            stderr = stderr_bytes.decode(errors="replace")
            exit_code = proc.returncode if proc.returncode is not None else -1

        if exit_code > 0:
            stdout = (
                f"SYSTEM ALERT: Command execution failed with error code "
                f"{exit_code}. You must troubleshoot this failure before "
                f"executing subsequent commands.\n"
            ) + stdout

        result = {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "current_working_directory": cwd_snapshot,
            "scope": "docker_sandbox" if self._sandbox is not None else "host",
        }

        # â"€â"€ Step 3: track any embedded cd calls (e.g. mkdir /x && cd /x) â"€â"€â"€â"€
        if result["exit_code"] == 0:
            if self._sandbox is not None:
                new_cwd = _resolve_sandbox_cd(command, cwd_snapshot)
            else:
                new_cwd = _resolve_cd_target(command, cwd_snapshot)
            if new_cwd is not None:
                self.current_cwd = new_cwd

        result["current_working_directory"] = self.current_cwd
        return result

    def execute_background_service(self, command: str) -> dict[str, Any]:
        """Launch *command* as a detached background process and return its PID.

        Unlike :meth:`execute_terminal_command`, this never awaits the process --
        it returns the moment the process is spawned.  It is the correct tool for
        servers and other long-running / non-terminating processes that would
        otherwise block the ReAct loop and deadlock the frontend.

        In sandbox mode the process runs inside the Docker container and output
        is appended to ``/tmp/background_task.log`` inside the container.
        On the host the log path is ``_BACKGROUND_LOG_PATH``.

        Returns a dict with keys ``pid`` (int, or ``None`` on failure),
        ``status`` (``"launched"`` or ``"error"``), ``message``, and ``log_file``.
        """
        if self._host_execution_disabled_reason is not None:
            blocked = self._host_execution_blocked_result(command)
            return {
                "pid": None,
                "status": "error",
                "message": blocked["stdout"],
                "error": blocked["stderr"],
                "log_file": _AGENT_BACKGROUND_LOG_PATH,
                "scope": "sandbox_unavailable",
            }

        mismatch_reason = _wrong_environment_command_reason(
            command,
            sandbox_active=self._sandbox is not None,
        )
        if mismatch_reason:
            blocked = self._wrong_environment_blocked_result(command, mismatch_reason)
            return {
                "pid": None,
                "status": "error",
                "message": blocked["stdout"],
                "error": blocked["stderr"],
                "log_file": _AGENT_BACKGROUND_LOG_PATH,
                "scope": blocked["scope"],
            }

        if self._sandbox is not None:
            # ── Sandbox path: launch inside Docker ────────────────────────
            sandbox_log = "/tmp/background_task.log"
            pid = self._sandbox.exec_background(command, self.current_cwd, sandbox_log)
            if pid is None:
                return {
                    "pid": None,
                    "status": "error",
                    "message": "Failed to launch background service inside Docker sandbox.",
                    "log_file": sandbox_log,
                }
            return {
                "pid": pid,
                "status": "launched",
                "message": (
                    f"Background service launched inside Docker sandbox. "
                    f"Output is appended to {sandbox_log} inside the container; "
                    f"run `cat {sandbox_log}` to inspect it."
                ),
                "log_file": sandbox_log,
            }

        # ── Host path: launch directly on the OS ──────────────────────────
        try:
            log_handle = open(_BACKGROUND_LOG_PATH, "ab")
        except OSError as exc:
            return {
                "pid": None,
                "status": "error",
                "message": f"Could not open log file {_BACKGROUND_LOG_PATH}: {exc}",
                "log_file": _BACKGROUND_LOG_PATH,
            }

        try:
            # Use bash on Windows so POSIX commands work (same logic as
            # execute_terminal_command).  shell=True on Windows invokes cmd.exe.
            if _POSIX_SHELL is not None:
                proc = subprocess.Popen(
                    [*_POSIX_SHELL, command],
                    shell=False,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    cwd=self.current_cwd,
                    start_new_session=True,
                )
            else:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    cwd=self.current_cwd,
                    start_new_session=True,
                )
        except (OSError, ValueError) as exc:
            return {
                "pid": None,
                "status": "error",
                "message": f"Failed to launch background service: {exc}",
                "log_file": _BACKGROUND_LOG_PATH,
            }
        finally:
            # The child inherits its own copy of the fd; close the parent's copy
            # so we don't leak a file handle for the lifetime of this manager.
            log_handle.close()

        return {
            "pid": proc.pid,
            "status": "launched",
            "message": (
                f"Background service started with PID {proc.pid}. Output is being "
                f"appended to {_BACKGROUND_LOG_PATH}; cat that file to check on it."
            ),
            "log_file": _BACKGROUND_LOG_PATH,
        }

    async def disconnect_server(self, name: str) -> None:
        """Shut down a single server and remove it from the registry."""
        await self._disconnect(name)

    async def close(self) -> None:
        """Shut down all connected servers, the browser session, and the sandbox."""
        for name in list(self._stacks):
            await self._disconnect(name)
        if self._browser_session is not None:
            await self._browser_session.close()
            self._browser_session = None
        if self._sandbox is not None:
            self._sandbox.stop()
            self._sandbox = None

    # ------------------------------------------------------------------
    # Browser convenience methods (delegates to lazy _BrowserSession)
    # ------------------------------------------------------------------

    def _get_browser(self) -> _BrowserSession:
        """Return the shared browser session, creating it if needed."""
        if self._browser_session is None:
            self._browser_session = _BrowserSession()
        return self._browser_session

    # ------------------------------------------------------------------
    # Vision and image generation
    # ------------------------------------------------------------------

    async def analyze_image(self, image_url: str, question: str) -> str:
        """Analyze an image using a vision-capable LLM model."""
        if not _VISION_MODEL:
            return json.dumps({"error": "AGENT_VISION_MODEL is not configured."})
        try:
            import litellm as _litellm
            content: list[dict[str, Any]] = [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
            response = await _litellm.acompletion(
                model=_VISION_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=1024,
            )
            answer = response.choices[0].message.content or ""
            return json.dumps({"image_url": image_url, "analysis": answer}, indent=2)
        except Exception as exc:
            logger.warning("analyze_image failed: %s", exc)
            return json.dumps({"error": str(exc), "image_url": image_url})

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        save_path: str | None = None,
    ) -> str:
        """Generate an image from *prompt* using DALL-E or a compatible API."""
        try:
            import httpx as _httpx
            import litellm as _litellm
            response = await _litellm.aimage_generation(
                model=_IMAGE_GEN_MODEL,
                prompt=prompt,
                size=size,
                n=1,
            )
            image_url = response.data[0].url or ""
            result: dict[str, Any] = {"prompt": prompt, "image_url": image_url, "size": size}

            if save_path and image_url:
                # Normalize path for sandbox mode
                if self._sandbox is not None:
                    dest = self._resolve_sandbox_file_path(save_path)
                else:
                    dest = str(self._resolve_file_path(save_path))
                try:
                    async with _httpx.AsyncClient(timeout=60.0) as client:
                        img_response = await client.get(image_url)
                    img_bytes = img_response.content
                    if self._sandbox is not None and hasattr(self._sandbox, "write_text_file"):
                        # For Docker: upload via the sandbox write method
                        import base64 as _b64
                        self._sandbox.write_text_file(
                            dest, _b64.b64encode(img_bytes).decode()
                        )
                    else:
                        Path(dest).parent.mkdir(parents=True, exist_ok=True)
                        Path(dest).write_bytes(img_bytes)
                    result["saved_to"] = dest
                except Exception as save_exc:
                    result["save_error"] = str(save_exc)
            return json.dumps(result, indent=2)
        except Exception as exc:
            logger.warning("generate_image failed: %s", exc)
            return json.dumps({"error": str(exc), "prompt": prompt})

    # ------------------------------------------------------------------
    # Browser convenience methods (delegates to lazy _BrowserSession)
    # ------------------------------------------------------------------

    async def browser_navigate(self, url: str, timeout: float = 30.0) -> str:
        """Navigate to *url* and return rendered text as JSON."""
        result = await self._get_browser().navigate(url, timeout=timeout)
        return json.dumps(result, indent=2)

    async def browser_get_text(self, max_chars: int = 8_000) -> str:
        """Return the current page's rendered text as JSON."""
        result = await self._get_browser().get_text(max_chars=max_chars)
        return json.dumps(result, indent=2)

    async def browser_screenshot(self, path: str | None = None) -> str:
        """Take a screenshot and return the save path as JSON."""
        result = await self._get_browser().screenshot(path=path)
        return json.dumps(result, indent=2)

    async def browser_click(self, selector: str, timeout: float = 5.0) -> str:
        """Click *selector* and return confirmation as JSON."""
        result = await self._get_browser().click(selector, timeout=timeout)
        return json.dumps(result, indent=2)

    async def browser_fill(
        self, selector: str, value: str, timeout: float = 5.0
    ) -> str:
        """Fill *selector* with *value* and return confirmation as JSON."""
        result = await self._get_browser().fill(selector, value, timeout=timeout)
        return json.dumps(result, indent=2)

    async def browser_evaluate(self, expression: str) -> str:
        """Evaluate a JS *expression* in the page and return result as JSON."""
        result = await self._get_browser().evaluate(expression)
        return json.dumps(result, indent=2)

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> ToolManager:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _disconnect(self, name: str) -> None:
        stack = self._stacks.pop(name, None)
        self._sessions.pop(name, None)
        self._tools_cache = None  # tool set changed; force a rebuild next call
        if stack:
            await stack.aclose()

    def _resolve_file_path(self, raw_path: str) -> Path:
        """Resolve a host path for host-mode file tools."""
        if self._sandbox is not None:
            raise RuntimeError("Host path resolution is disabled in Docker sandbox mode")
        return _resolve_tool_path(raw_path, self.current_cwd)

    def _resolve_sandbox_file_path(self, raw_path: str) -> str:
        """Resolve a tool path from the container's point of view."""
        path = (raw_path or self.current_cwd).replace("\\", "/")
        if not posixpath.isabs(path):
            path = posixpath.join(self.current_cwd or _SANDBOX_WORKDIR, path)
        return posixpath.normpath(path)


# ---------------------------------------------------------------------------
# System environment helpers
# ---------------------------------------------------------------------------

def _is_dangerous_command(command: str) -> bool:
    """Return True when *command* matches a known-destructive regex pattern.

    Collapses whitespace before matching so extra spaces can't bypass a rule.
    Each pattern targets the dangerous *effect* (wipe filesystem, fork bomb,
    corrupt auth files) rather than a single literal string, catching common
    spelling variants (flag reordering, doubled slashes, quoting).
    """
    # Collapse internal whitespace to defeat space-padding bypasses.
    normalized = re.sub(r"\s+", " ", command.strip())
    return any(pat.search(normalized) is not None for pat in _DANGEROUS_PATTERNS)


def _wrong_environment_command_reason(command: str, *, sandbox_active: bool) -> str | None:
    """Return a reason when *command* targets an OS other than the active runner."""
    normalized = re.sub(r"\s+", " ", command.strip())
    if not normalized:
        return None

    if sandbox_active:
        if _WINDOWS_SHELL_INVOKE_RE.search(normalized):
            return (
                "Command targets a Windows shell (cmd/PowerShell), but the active "
                "execution environment is the Linux Docker sandbox."
            )
        if _WINDOWS_ABSOLUTE_PATH_RE.search(normalized):
            return (
                "Command contains a Windows absolute path, but the active execution "
                "environment is the Linux Docker sandbox mounted at /workspace."
            )
        if _WINDOWS_WHERE_RE.search(normalized):
            return (
                "Command uses Windows 'where' lookup syntax, but the active execution "
                "environment is Linux. Use command -v or get_system_environment."
            )
    return None


def _resolve_cd_target(command: str, current_cwd: str) -> str | None:
    """Return the resolved absolute path the last ``cd`` in *command* targets.

    Returns ``None`` when no ``cd`` is found, when the target is ``-`` (which
    requires shell history the process doesn't have), or when path resolution
    fails.  Tilde expansion is handled for ``~`` and ``~/â€¦`` forms.
    """
    # Bare `cd` with no argument navigates to the home directory.
    bare_cd = re.search(r'(?:^|[;&|])\s*cd\s*(?:[;&|]|$)', command, re.MULTILINE)

    matches = _CD_RE.findall(command)
    if not matches and not bare_cd:
        return None
    if not matches:
        return str(Path.home())

    # Each match is a 3-tuple (double-quoted, single-quoted, unquoted); take
    # the last match since chained cd calls leave the shell in the final dir.
    raw = next(g for g in reversed(matches[-1]) if g)

    if raw == '-':
        return None  # cd - requires shell history; skip tracking
    if raw == '~':
        return str(Path.home())
    if raw.startswith('~/'):
        return str(Path.home() / raw[2:])

    candidate = Path(raw) if Path(raw).is_absolute() else Path(current_cwd) / raw
    try:
        # resolve() normalises .., symlinks, etc.; suppress OSError on missing paths
        return str(candidate.resolve())
    except OSError:
        return str(candidate)


def _resolve_sandbox_cd(command: str, current_cwd: str) -> str | None:
    """POSIX-aware cd tracker for sandbox mode.

    On a Windows host, ``Path("/workspace")`` resolves to ``C:\\workspace``,
    breaking container path tracking.  This function does pure string-based
    POSIX path resolution so that ``/workspace/mydir`` stays as-is.
    """
    bare_cd = re.search(r'(?:^|[;&|])\s*cd\s*(?:[;&|]|$)', command, re.MULTILINE)
    matches = _CD_RE.findall(command)
    if not matches and not bare_cd:
        return None
    if not matches:
        return "/root"  # cd with no args goes to ~ which is /root in most containers

    raw = next(g for g in reversed(matches[-1]) if g)
    if raw == '-':
        return None
    if raw in ('~', '/root', '$HOME'):
        return "/root"
    if raw.startswith('~/'):
        return "/root/" + raw[2:]

    if raw.startswith('/'):
        # Absolute POSIX path — normalise .. without using os.path
        parts = []
        for segment in raw.split('/'):
            if segment in ('', '.'):
                continue
            if segment == '..':
                if parts:
                    parts.pop()
            else:
                parts.append(segment)
        return '/' + '/'.join(parts)

    # Relative path — join against current_cwd
    base = current_cwd.rstrip('/')
    combined = base + '/' + raw
    parts = []
    for segment in combined.split('/'):
        if segment in ('', '.'):
            continue
        if segment == '..':
            if parts:
                parts.pop()
        else:
            parts.append(segment)
    return '/' + '/'.join(parts)


def _collect_system_environment() -> str:
    """Collect a JSON snapshot of the host environment synchronously.

    Called once at ``ToolManager.__init__`` time so there is no async
    overhead on the first ``get_system_environment`` call.
    """
    system = platform.system()  # 'Linux', 'Darwin', 'Windows', ...
    os_label = {"Darwin": "macOS", "Linux": "Linux", "Windows": "Windows"}.get(
        system, system
    )

    disk: dict[str, Any]
    try:
        usage = shutil.disk_usage(os.getcwd())
        disk = {
            "total_gb": round(usage.total / 1_073_741_824, 2),
            "used_gb": round(usage.used / 1_073_741_824, 2),
            "free_gb": round(usage.free / 1_073_741_824, 2),
        }
    except OSError:
        disk = {"error": "unavailable"}

    runtimes = {name: shutil.which(name) is not None for name in _PROBED_RUNTIMES}

    # Report the shell used by execute_terminal_command so the model knows
    # which syntax is safe. On Windows without bash, POSIX commands will fail.
    if _POSIX_SHELL is not None:
        shell_info = {"shell": _POSIX_SHELL[0], "shell_args": _POSIX_SHELL[1:], "posix": True}
    elif system == "Windows":
        shell_info = {"shell": "cmd.exe", "posix": False,
                      "warning": "No bash on PATH; POSIX commands unavailable"}
    else:
        shell_info = {"shell": "/bin/sh", "posix": True}

    # User identity and privilege level — critical for the agent to know BEFORE
    # attempting package installs (apt-get/yum/brew all require root or sudo).
    import getpass as _getpass
    try:
        username = _getpass.getuser()
    except Exception:
        username = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
    is_root = (os.geteuid() == 0) if hasattr(os, "geteuid") else (username in {"root", "Administrator"})
    sudo_available = shutil.which("sudo") is not None
    home_dir = str(Path.home())
    home_exists = Path(home_dir).is_dir()

    user_info: dict[str, Any] = {
        "username": username,
        "is_root": is_root,
        "sudo_available": sudo_available,
        "home_dir": home_dir,
        "home_exists": home_exists,
    }
    if not is_root and not sudo_available:
        user_info["warning"] = (
            "Running as non-root without sudo, so system package managers (apt-get, "
            "yum, brew) will fail -- do NOT attempt them. The common toolchains are "
            "already pre-installed system-wide: check the 'runtimes' block above for "
            "node/npm/npx, python/pip, and rustc/cargo before assuming anything is "
            "missing. For extra Python packages use 'pip install --user'; for extra "
            "npm packages prefer a local project install (npm install in the project)."
        )

    return json.dumps(
        {
            "os": os_label,
            "os_version": platform.version(),
            "machine": platform.machine(),
            "python_executable": sys.executable,
            "disk_cwd": disk,
            "runtimes": runtimes,
            "shell": shell_info,
            "user": user_info,
        },
        indent=2,
    )


def _resolve_tool_path(raw_path: str, current_cwd: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(current_cwd) / path
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _inspect_path(raw_path: str, current_cwd: str) -> dict[str, Any]:
    resolved = _resolve_tool_path(raw_path, current_cwd)

    info: dict[str, Any] = {
        "input": raw_path,
        "path": str(resolved),
        "exists": resolved.exists(),
    }
    if not info["exists"]:
        return info

    try:
        stat = resolved.stat()
        info.update(
            {
                "is_file": resolved.is_file(),
                "is_dir": resolved.is_dir(),
                "size_bytes": stat.st_size,
                "modified_at_unix": stat.st_mtime,
                "modified_age_seconds": round(time.time() - stat.st_mtime, 3),
            }
        )
        if resolved.is_dir():
            children = sorted(resolved.iterdir(), key=lambda p: p.name.lower())[:50]
            info["children"] = [
                {
                    "name": child.name,
                    "is_file": child.is_file(),
                    "is_dir": child.is_dir(),
                    "size_bytes": child.stat().st_size if child.exists() else None,
                }
                for child in children
            ]
        elif resolved.is_file():
            info["preview"] = _preview_file(resolved)
    except OSError as exc:
        info["error"] = str(exc)
    return info


def _preview_file(path: Path, max_bytes: int = 1200) -> str:
    try:
        data = path.read_bytes()[:max_bytes]
    except OSError as exc:
        return f"[preview error] {exc}"
    if b"\0" in data:
        return f"[binary file preview omitted; sampled {len(data)} bytes]"
    printable = sum(1 for byte in data if byte in b"\n\r\t" or 32 <= byte <= 126)
    if data and printable / len(data) < 0.75:
        return f"[binary file preview omitted; sampled {len(data)} bytes]"
    return data.decode("utf-8", errors="replace")


def _inspect_pid(pid: int) -> dict[str, Any]:
    info: dict[str, Any] = {"pid": pid, "running": False}
    try:
        os.kill(pid, 0)
        info["running"] = True
    except ProcessLookupError:
        return info
    except PermissionError:
        info["running"] = True
        info["permission_limited"] = True
    except OSError as exc:
        info["error"] = str(exc)

    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    if proc_cmdline.exists():
        try:
            info["cmdline"] = proc_cmdline.read_text(errors="replace").replace("\x00", " ").strip()
        except OSError:
            pass
    return info


def _inspect_process_name(name: str) -> dict[str, Any]:
    needle = name.lower()
    matches: list[dict[str, Any]] = []

    if platform.system() == "Windows":
        cmd = ["tasklist"]
    else:
        cmd = ["ps", "-eo", "pid=,comm=,args="]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"name": name, "matches": [], "error": str(exc)}

    for line in proc.stdout.splitlines():
        if needle not in line.lower():
            continue
        matches.append({"line": line.strip()})
        if len(matches) >= 20:
            break

    return {"name": name, "matches": matches, "count": len(matches)}


def _inspect_port(port: int) -> dict[str, Any]:
    info: dict[str, Any] = {"port": port, "host": "127.0.0.1", "connectable": False}
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1.0):
            info["connectable"] = True
    except OSError as exc:
        info["error"] = str(exc)
    return info


def _tail_file(raw_path: str, max_bytes: int = 4000) -> dict[str, Any]:
    path = Path(raw_path)
    info: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    try:
        data = path.read_bytes()[-max_bytes:]
        info["tail"] = data.decode("utf-8", errors="replace")
        info["size_bytes"] = path.stat().st_size
    except OSError as exc:
        info["error"] = str(exc)
    return info


# ---------------------------------------------------------------------------
# Web helpers (used by web_fetch and web_search)
# ---------------------------------------------------------------------------

def _html_to_text(html: str) -> str:
    """Convert HTML to readable plain text — no external dependency required.

    Strips non-content blocks (script/style/head), replaces block-level
    elements with newlines, removes all remaining tags, decodes common HTML
    entities, and collapses whitespace.
    """
    # Remove non-content blocks wholesale
    html = re.sub(
        r"<(script|style|noscript|svg|head)[^>]*>.*?</\1>",
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Block-level elements become line breaks so paragraph structure survives
    html = re.sub(
        r"</?(br|p|div|h[1-6]|li|tr|td|th|blockquote|pre|article|"
        r"section|header|footer|nav|main|aside)[^>]*>",
        "\n",
        html,
        flags=re.IGNORECASE,
    )
    # Strip all remaining HTML tags
    html = re.sub(r"<[^>]+>", "", html)
    # Decode named entities
    for entity, char in [
        ("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
        ("&quot;", '"'), ("&#39;", "'"), ("&apos;", "'"),
    ]:
        html = html.replace(entity, char)
    # Decode numeric entities (hex: &#x27; and decimal: &#39;)
    html = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), html)
    html = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), html)
    # Tidy whitespace
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n[ \t]+", "\n", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


def _decode_entities(text: str) -> str:
    """Decode common HTML entities in a short string (title / snippet)."""
    for entity, char in [
        ("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
        ("&quot;", '"'), ("&#39;", "'"), ("&apos;", "'"),
    ]:
        text = text.replace(entity, char)
    # Hex entities: &#x27; → '
    text = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), text)
    # Decimal entities: &#39; → '
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    return text


def _parse_ddg_html(html: str, max_results: int = 8) -> list[dict[str, str]]:
    """Extract result title/url/snippet from a DuckDuckGo HTML Lite response."""
    results: list[dict[str, str]] = []
    # DuckDuckGo Lite marks result links with class="result__a"
    title_pat = re.compile(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )
    snippet_pat = re.compile(
        r'class="result__snippet[^"]*"[^>]*>(.*?)</(?:a|span)>',
        re.DOTALL | re.IGNORECASE,
    )
    titles = title_pat.findall(html)
    snippets = [
        _decode_entities(re.sub(r"<[^>]+>", "", s)).strip()
        for s in snippet_pat.findall(html)
    ]
    for i, (url, title) in enumerate(titles[:max_results]):
        results.append({
            "title": _decode_entities(re.sub(r"<[^>]+>", "", title)).strip(),
            "url": url,
            "snippet": snippets[i] if i < len(snippets) else "",
        })
    return results


# ---------------------------------------------------------------------------
# Web tool schemas
# ---------------------------------------------------------------------------

WEB_FETCH_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "web_fetch",
    "description": (
        "Fetch any public URL and return its readable text plus metadata. "
        "HTML is automatically converted to clean text (scripts, styles, and "
        "markup stripped). JSON and plain text are returned as-is. "
        "Use this to read documentation, API responses, GitHub files, news "
        "articles, search-result pages, or any web resource. "
        "The text is truncated at max_chars (default 8 000); raise it for "
        "long technical documents. Returns: url, status_code, content_type, "
        "text, word_count, truncated, total_chars."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["url"],
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch (http:// or https://).",
            },
            "max_chars": {
                "type": "integer",
                "description": "Max characters of body text to return (100-40000). Default 8000.",
            },
            "timeout": {
                "type": "number",
                "description": "Request timeout in seconds (1-60). Default 15.",
            },
        },
        "additionalProperties": False,
    },
}

WEB_SEARCH_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "web_search",
    "description": (
        "Search the web and return result titles, URLs, and snippets. "
        "Use this to find information, look up documentation, research a topic, "
        "or locate solutions to errors — without writing a custom skill. "
        "Follow up with web_fetch on the most relevant URLs to read full content. "
        "Typical pattern: web_search → pick top URLs → web_fetch each one → synthesise. "
        "Returns: query, backend, results[] with title/url/snippet."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string.",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (1-20). Default 8.",
            },
        },
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Browser automation (Playwright)
# ---------------------------------------------------------------------------

# Whether playwright is importable — checked once at module load so the
# browser tool schemas are conditionally included in list_all_tools().
_PLAYWRIGHT_AVAILABLE: bool
try:
    import playwright  # noqa: F401
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False


class _BrowserSession:
    """Lazy Playwright browser session shared across all browser tool calls.

    A single Browser + BrowserContext + Page is created on the first
    ``browser_navigate`` call and reused until :meth:`close` is called
    (triggered by ``ToolManager.close()``).  All page operations are
    sequential — there is no multi-tab support.
    """

    def __init__(self) -> None:
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    async def _ensure_started(self) -> None:
        if self._page is not None:
            return
        from playwright.async_api import async_playwright  # type: ignore[import-untyped]

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        context = await self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        self._page = await context.new_page()

    async def navigate(self, url: str, timeout: float = 30.0) -> dict[str, Any]:
        await self._ensure_started()
        try:
            await self._page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=int(timeout * 1000),
            )
        except Exception as exc:
            return {"error": str(exc), "url": url}
        return await self._page_snapshot()

    async def get_text(self, max_chars: int = 8_000) -> dict[str, Any]:
        await self._ensure_started()
        snap = await self._page_snapshot(max_chars=max_chars)
        return snap

    async def screenshot(self, path: str | None = None) -> dict[str, Any]:
        await self._ensure_started()
        if path is None:
            path = str(Path(tempfile.gettempdir()) / "agent_browser_screenshot.png")
        try:
            await self._page.screenshot(path=path, full_page=False)
            return {
                "saved_to": path,
                "url": self._page.url,
                "title": await self._page.title(),
            }
        except Exception as exc:
            return {"error": str(exc)}

    async def click(self, selector: str, timeout: float = 5.0) -> dict[str, Any]:
        await self._ensure_started()
        try:
            await self._page.click(selector, timeout=int(timeout * 1000))
            await self._page.wait_for_load_state("domcontentloaded")
            return {"clicked": selector, "url": self._page.url}
        except Exception as exc:
            return {"error": str(exc), "selector": selector}

    async def fill(
        self, selector: str, value: str, timeout: float = 5.0
    ) -> dict[str, Any]:
        await self._ensure_started()
        try:
            await self._page.fill(selector, value, timeout=int(timeout * 1000))
            return {"filled": True, "selector": selector}
        except Exception as exc:
            return {"error": str(exc), "selector": selector}

    async def evaluate(self, expression: str) -> dict[str, Any]:
        await self._ensure_started()
        try:
            result = await self._page.evaluate(expression)
            return {"result": result, "url": self._page.url}
        except Exception as exc:
            return {"error": str(exc)}

    async def close(self) -> None:
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
        self._page = None
        self._browser = None
        self._playwright = None

    # ------------------------------------------------------------------
    async def _page_snapshot(self, max_chars: int = 8_000) -> dict[str, Any]:
        """Extract readable text from the current page via JavaScript."""
        try:
            raw: str = await self._page.evaluate(
                """() => {
                    const clone = document.documentElement.cloneNode(true);
                    for (const el of clone.querySelectorAll(
                            'script,style,noscript,svg,head')) {
                        el.remove();
                    }
                    return (clone.innerText || clone.textContent || '').trim();
                }"""
            )
        except Exception as exc:
            raw = f"[JS evaluation error: {exc}]"

        text = re.sub(r"\n{3,}", "\n\n", raw).strip()
        total = len(text)
        return {
            "url": self._page.url,
            "title": await self._page.title(),
            "text": text[:max_chars],
            "word_count": len(text.split()),
            "truncated": total > max_chars,
            "total_chars": total,
        }


# ---------------------------------------------------------------------------
# Browser tool schemas
# ---------------------------------------------------------------------------

BROWSER_NAVIGATE_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "browser_navigate",
    "description": (
        "Open a URL in a real headless Chromium browser and return the "
        "rendered page text and title. Unlike web_fetch, this executes "
        "JavaScript so single-page apps, login-gated pages, and "
        "dynamically loaded content are fully rendered. "
        "Use browser_navigate for JS-heavy sites; use web_fetch for "
        "static pages and APIs (faster). "
        "Returns: url, title, text (rendered), word_count, truncated."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["url"],
        "properties": {
            "url": {"type": "string", "description": "URL to navigate to."},
            "timeout": {
                "type": "number",
                "description": "Page load timeout in seconds (1-120). Default 30.",
            },
        },
        "additionalProperties": False,
    },
}

BROWSER_GET_TEXT_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "browser_get_text",
    "description": (
        "Return the rendered text of the browser's current page without "
        "reloading. Call after browser_navigate, browser_click, or "
        "browser_fill to read the updated content."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "max_chars": {
                "type": "integer",
                "description": "Max characters to return (100-40000). Default 8000.",
            },
        },
        "additionalProperties": False,
    },
}

BROWSER_SCREENSHOT_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "browser_screenshot",
    "description": (
        "Take a screenshot of the browser's current page and save it to a "
        "file. Returns the file path and current URL. Useful for verifying "
        "visual state or capturing evidence."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to save the PNG. Defaults to a temp file.",
            },
        },
        "additionalProperties": False,
    },
}

BROWSER_CLICK_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "browser_click",
    "description": (
        "Click an element on the current browser page using a CSS selector. "
        "Waits for the page to settle after the click. "
        "Use browser_get_text afterwards to read updated content."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["selector"],
        "properties": {
            "selector": {
                "type": "string",
                "description": "CSS selector of the element to click.",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds. Default 5.",
            },
        },
        "additionalProperties": False,
    },
}

BROWSER_FILL_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "browser_fill",
    "description": (
        "Fill an input or textarea with a value using a CSS selector. "
        "Clears the current value first, then types the new one. "
        "Use browser_click to submit the form afterwards."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["selector", "value"],
        "properties": {
            "selector": {
                "type": "string",
                "description": "CSS selector of the input element.",
            },
            "value": {
                "type": "string",
                "description": "Text to fill into the element.",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds. Default 5.",
            },
        },
        "additionalProperties": False,
    },
}

BROWSER_EVALUATE_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "browser_evaluate",
    "description": (
        "Execute a JavaScript expression in the browser page context and "
        "return the result. Use to extract data from DOM elements, trigger "
        "JS events, read page variables, or call page APIs. "
        "Example: \"document.querySelector('#price').textContent\""
    ),
    "inputSchema": {
        "type": "object",
        "required": ["expression"],
        "properties": {
            "expression": {
                "type": "string",
                "description": "JavaScript expression to evaluate in the page context.",
            },
        },
        "additionalProperties": False,
    },
}

# Collected for conditional inclusion in list_all_tools()
_BROWSER_TOOL_SCHEMAS: list[dict[str, Any]] = [
    BROWSER_NAVIGATE_TOOL,
    BROWSER_GET_TEXT_TOOL,
    BROWSER_SCREENSHOT_TOOL,
    BROWSER_CLICK_TOOL,
    BROWSER_FILL_TOOL,
    BROWSER_EVALUATE_TOOL,
]


# ---------------------------------------------------------------------------
# Skills-server bootstrap helpers
# ---------------------------------------------------------------------------

# Built-in skills live alongside the source tree so they're baked into the
# Docker image.  On every start-up they are copied (overwritten) into the
# writable skills volume so the agent MCP server can serve them.
_BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent / "builtin_skills"


def _ensure_skills_dir(skills_dir: Path) -> None:
    """Create *skills_dir*, write bootstrap files, and deploy built-in skills.

    Bootstrap files (_skill.py, server.py) are written only if absent so that
    manual edits are not overwritten.  Built-in skills (from src/builtin_skills/)
    are always written so that image updates propagate on restart.
    """
    skills_dir.mkdir(parents=True, exist_ok=True)
    _write_if_absent(skills_dir / "_skill.py", _SKILL_DECORATOR_SRC)
    _write_if_absent(skills_dir / "server.py", _SKILLS_SERVER_SRC)

    # Deploy built-in skills when their content changes. Avoid rewriting identical
    # files: uvicorn --reload watches the repo root in local dev, and touching
    # skills/*.py during startup causes a reload loop that drops WebSockets.
    if _BUILTIN_SKILLS_DIR.is_dir():
        for src_file in sorted(_BUILTIN_SKILLS_DIR.glob("*.py")):
            dest = skills_dir / src_file.name
            try:
                _write_if_changed(
                    dest,
                    src_file.read_text(encoding="utf-8"),
                )
                logger.debug("Deployed built-in skill: %s -> %s", src_file.name, dest)
            except OSError as exc:
                logger.warning("Could not deploy built-in skill %s: %s", src_file.name, exc)


def _write_if_absent(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def _write_if_changed(path: Path, content: str) -> None:
    try:
        if path.exists() and path.read_text(encoding="utf-8") == content:
            return
    except OSError:
        # Fall through and try to repair the file by writing the desired content.
        pass
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Bootstrap file sources
# Written to skills/ on first call to connect_skills_server.
# ---------------------------------------------------------------------------

_SKILL_DECORATOR_SRC = """\
from __future__ import annotations


def skill(fn=None, *, name=None, description=None):
    '''Mark a function as an MCP skill so the skills server auto-discovers it.

    Use as a plain decorator or with keyword arguments::

        from _skill import skill

        @skill
        def greet(name: str) -> str:
            "Return a personalised greeting."
            return f"Hello, {name}!"

        @skill(name="shout", description="Return an uppercased string.")
        def shout(text: str) -> str:
            return text.upper()
    '''
    def _decorate(f):
        f._is_skill = True
        f._skill_name = name or f.__name__
        f._skill_description = description or f.__doc__ or ""
        return f

    return _decorate(fn) if fn is not None else _decorate
"""

_SKILLS_SERVER_SRC = """\
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import anyio
from mcp.server.fastmcp import FastMCP

SKILLS_DIR = Path(__file__).parent
# Allow skills to import the _skill decorator via `from _skill import skill`
sys.path.insert(0, str(SKILLS_DIR))

logger = logging.getLogger(__name__)
mcp = FastMCP("skills")


def _load_skills() -> None:
    for skill_path in sorted(SKILLS_DIR.glob("*.py")):
        if skill_path.name.startswith("_") or skill_path.name == "server.py":
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"_skills.{skill_path.stem}", skill_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:
            logger.exception("Failed to load skill module: %s", skill_path.name)
            continue

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if callable(obj) and getattr(obj, "_is_skill", False):
                mcp.add_tool(
                    obj,
                    name=getattr(obj, "_skill_name", attr_name),
                    description=getattr(obj, "_skill_description", None),
                )
                logger.info("Registered skill: %s (from %s)", attr_name, skill_path.name)


_load_skills()

if __name__ == "__main__":
    anyio.run(mcp.run_stdio_async)
"""
