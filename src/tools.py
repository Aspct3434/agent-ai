from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

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
# ---------------------------------------------------------------------------

_SANDBOX_ENABLED = os.getenv("AGENT_SANDBOX", "").lower() == "docker"
_SANDBOX_IMAGE = os.getenv("AGENT_SANDBOX_IMAGE", "python:3.12-slim")
_SANDBOX_WORKDIR = "/workspace"
_SANDBOX_MEMORY = os.getenv("AGENT_SANDBOX_MEMORY", "1g")
_SANDBOX_CPUS = os.getenv("AGENT_SANDBOX_CPUS", "1.0")
_SANDBOX_PIDS = os.getenv("AGENT_SANDBOX_PIDS", "256")
_SANDBOX_PACKAGE_CAPS = (
    # apt/dpkg drop privileges to the _apt user and adjust package-owned files.
    # Keep the sandbox otherwise locked down, but allow normal package installs.
    "CHOWN",
    "DAC_OVERRIDE",
    "FOWNER",
    "SETGID",
    "SETUID",
)

# Path the agent should read to inspect the background-service log, expressed
# from the agent's OWN vantage point: inside the container under sandbox mode,
# on the host otherwise. Tool *descriptions* and prompts must reference this --
# never the host-only ``_BACKGROUND_LOG_PATH`` -- so the advice is correct in
# whichever execution mode is active. Actual host file I/O still uses
# ``_BACKGROUND_LOG_PATH`` directly.
_AGENT_BACKGROUND_LOG_PATH = (
    "/tmp/background_task.log" if _SANDBOX_ENABLED else _BACKGROUND_LOG_PATH
)


class _DockerSandbox:
    """A long-lived Docker container used as an isolated command execution sandbox.

    One instance is created per ``ToolManager`` when ``AGENT_SANDBOX=docker``.
    The container runs ``sleep infinity`` so it stays alive between commands;
    each command is executed via ``docker exec``.  The host workspace directory
    is bind-mounted to ``/workspace`` inside the container so file-tool writes
    (which happen on the host) are immediately visible to terminal commands.
    """

    def __init__(self, image: str, host_workdir: str) -> None:
        self._image = image
        self._host_workdir = host_workdir
        self._container_id: str | None = None
        self._name = f"agent-sandbox-{os.getpid()}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Pull image (if needed) and start the sandbox container."""
        # Stop any leftover container from a previous run with the same name.
        subprocess.run(
            ["docker", "stop", "-t", "2", self._name],
            capture_output=True, timeout=10,
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
        logger.info(
            "Docker sandbox started: container=%s image=%s workdir=%s",
            self._container_id[:12], self._image, self._host_workdir,
        )

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
                # Bind-mount the host workspace so file-tool writes are visible.
                "-v", f"{self._host_workdir}:{_SANDBOX_WORKDIR}",
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
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "-w", cwd, self.container_id,
            "bash", "-c", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.CancelledError:
            proc.kill()
            try:
                await proc.wait()
            except ProcessLookupError:
                pass
            raise
        except TimeoutError:
            proc.kill()
            try:
                await proc.wait()
            except ProcessLookupError:
                pass
            raise
        rc = proc.returncode if proc.returncode is not None else -1
        return rc, stdout_bytes.decode(errors="replace"), stderr_bytes.decode(errors="replace")

    def exec_background(self, command: str, cwd: str, log_path: str) -> int | None:
        """Launch *command* detached inside the container; output appends to *log_path*."""
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
        script = r"""
import json, os, socket, sys, time

req = json.load(sys.stdin)

def preview(path, max_bytes=1200):
    try:
        with open(path, "rb") as f:
            return f.read(max_bytes).decode("utf-8", "replace")
    except OSError as exc:
        return "[preview error] " + str(exc)

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
    "scope": "docker_sandbox",
    "current_working_directory": cwd,
    "paths": [inspect_path(p, cwd) for p in req.get("paths") or []],
    "pids": [inspect_pid(int(p)) for p in req.get("pids") or []],
    "process_names": [inspect_process_name(p) for p in req.get("process_names") or []],
    "ports": [inspect_port(int(p)) for p in req.get("ports") or []],
}
if req.get("include_background_log", True):
    out["background_log"] = tail_file("/tmp/background_task.log")
print(json.dumps(out))
"""
        payload = json.dumps(
            {
                "cwd": cwd,
                "paths": paths or [],
                "pids": pids or [],
                "process_names": process_names or [],
                "ports": ports or [],
                "include_background_log": include_background_log,
            }
        )
        result = subprocess.run(
            ["docker", "exec", "-i", self.container_id, "python3", "-c", script],
            input=payload,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "sandbox evidence failed")
        return json.loads(result.stdout)

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
        script = r"""
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
        payload = json.dumps(
            {
                "port": int(port),
                "path": path,
                "query": query,
                "method": method,
                "headers": headers,
                "body_b64": body_b64,
            }
        )
        result = subprocess.run(
            ["docker", "exec", "-i", self.container_id, "python3", "-c", script],
            input=payload,
            capture_output=True,
            text=True,
            timeout=35,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "sandbox HTTP fetch failed")
        return json.loads(result.stdout)

    # ------------------------------------------------------------------
    # Environment introspection
    # ------------------------------------------------------------------

    def collect_environment_json(self) -> str:
        """Return a JSON snapshot of the *container's* environment."""
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


GET_SYSTEM_ENVIRONMENT_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "get_system_environment",
    "description": (
        "Return a JSON snapshot of the host system environment. Includes: OS type, "
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
        "Return structured evidence about host filesystem paths, process IDs, "
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
        "Create or overwrite a UTF-8 text file on the host filesystem. Parent "
        "directories are created automatically. Use this for concrete artifacts "
        "such as HTML, CSS, JavaScript, Markdown, JSON, config files, or docs; "
        "then verify or publish the artifact as required by the task contract."
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
        "must change files, services, databases, or other host state. The engine "
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
                "description": "Whether the task is answered in text or requires host-side execution.",
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
                        "published_static_site_url",
                        "running_http_service",
                        "database_mutation",
                        "command_output",
                        "none",
                    ],
                },
                "description": (
                    "Structured proof the engine should require before accepting a "
                    "final answer. Use 'none' only for answer-mode tasks."
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

PUBLISH_STATIC_SITE_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "publish_static_site",
    "description": (
        "Publish a completed static website directory through the already-exposed "
        "agent backend at /sites/<slug>/. The source directory must contain an "
        "index.html file. Use this instead of starting python -m http.server on "
        "arbitrary container ports when the user asks to host, publish, serve, or "
        "get a browser link for a static site."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "source_path": {
                "type": "string",
                "description": (
                    "Directory containing the static website files. Relative paths "
                    "resolve from the current tool working directory. Defaults to "
                    "the current tool working directory."
                ),
            },
            "slug": {
                "type": "string",
                "description": (
                    "Optional URL-safe site slug. Defaults to the source directory name."
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
        "Execute a shell command on the host system and return its combined stdout "
        "and stderr output. Commands run inside the OS default shell (sh on POSIX, "
        "cmd on Windows) and this call WAITS for them to finish. "
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
        "Delegate a self-contained unit of work to a specialised sub-agent. "
        "Use 'researcher' to gather or synthesise information, 'coder' to write "
        "or modify source code, and 'auditor' to review, critique, or verify "
        "correctness. The sub-agent receives only task_description and "
        "context_payload -- it has no access to the current conversation history. "
        "Do not use this tool as the primary way to create files, install packages, "
        "or start services on the host; for physical environment changes, call "
        "execute_terminal_command or execute_background_service directly and verify "
        "the result afterwards."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["agent_type", "task_description", "context_payload"],
        "properties": {
            "agent_type": {
                "type": "string",
                "enum": ["researcher", "coder", "auditor"],
                "description": (
                    "Which specialised sub-agent to invoke: "
                    "'researcher' for information gathering, "
                    "'coder' for code generation or editing, "
                    "'auditor' for review and verification."
                ),
            },
            "task_description": {
                "type": "string",
                "description": "The exact, self-contained instruction for the sub-agent.",
            },
            "context_payload": {
                "type": "object",
                "description": (
                    "A flat or nested dictionary of background facts the sub-agent "
                    "needs to complete the task (e.g. relevant schema snippets, "
                    "prior tool outputs, file paths)."
                ),
                "additionalProperties": True,
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

        # Docker sandbox — started eagerly so the container is ready before the
        # first terminal command arrives and so get_system_environment reflects
        # the container's Linux environment from the very first session message.
        self._sandbox: _DockerSandbox | None = None
        self._sandbox_startup_failed: bool = False  # set True when AGENT_SANDBOX=docker but Docker unreachable
        if _SANDBOX_ENABLED:
            try:
                sandbox = _DockerSandbox(image=_SANDBOX_IMAGE, host_workdir=host_workdir)
                sandbox.start()
                self._sandbox = sandbox
                # Inside the container all commands run under /workspace.
                self.current_cwd: str = _SANDBOX_WORKDIR
                self._env_snapshot: str = sandbox.collect_environment_json()
                logger.info("Sandbox mode: docker (image=%s)", _SANDBOX_IMAGE)
            except Exception as exc:
                logger.error(
                    "Docker sandbox failed to start — falling back to host execution: %s", exc
                )
                self._sandbox = None
                self._sandbox_startup_failed = True
                self.current_cwd = host_workdir
                self._env_snapshot = _collect_system_environment()
        else:
            self.current_cwd = host_workdir
            self._env_snapshot = _collect_system_environment()

        self.published_sites_dir = Path(
            os.getenv("PUBLISHED_SITES_DIR", str(_PROJECT_ROOT / "published_sites"))
        ).expanduser()
        self.public_base_url = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
        # When sandbox is active, _host_workdir is the real host path mounted
        # into the container as /workspace. File tools use it to remap container
        # paths back to host paths so write_text_file("/workspace/x") resolves
        # correctly on the host.
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
        results.append(PUBLISH_STATIC_SITE_TOOL)
        results.append(EXPOSE_LOCAL_HTTP_SERVICE_TOOL)
        results.append(EXECUTE_TERMINAL_COMMAND_TOOL)
        results.append(EXECUTE_BACKGROUND_SERVICE_TOOL)
        results.append(EXPAND_TOOL_OUTPUT_TOOL)
        results.append(UPDATE_PLAN_TOOL)
        results.append(DELEGATE_TASK_TOOL)
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
        target = self._resolve_file_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        payload = {
            "written": True,
            "path": str(target),
            "exists": target.is_file(),
            "size_bytes": target.stat().st_size if target.exists() else 0,
            "current_working_directory": self.current_cwd,
            "artifact_quality": _artifact_quality_for_text(content, str(target)),
        }
        return json.dumps(payload, indent=2)

    def publish_static_site(
        self,
        source_path: str | None = None,
        slug: str | None = None,
    ) -> str:
        """Copy a static site into the backend-served published-sites directory.

        The FastAPI app serves ``published_sites_dir`` at ``/sites``. This gives
        browser-reachable URLs through the existing Docker port mapping instead
        of starting throwaway HTTP servers on ports that are internal to the
        container.
        """
        source = self._resolve_file_path(source_path or self.current_cwd)
        if not source.exists():
            raise FileNotFoundError(f"Static site source does not exist: {source}")
        if not source.is_dir():
            raise NotADirectoryError(f"Static site source is not a directory: {source}")

        index_path = source / "index.html"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"Static site source must contain index.html: {index_path}"
            )

        site_slug = _slugify_site_slug(slug or source.name)
        self.published_sites_dir.mkdir(parents=True, exist_ok=True)
        published_root = self.published_sites_dir.resolve()
        destination = (published_root / site_slug).resolve()
        if destination != published_root and published_root not in destination.parents:
            raise ValueError(f"Refusing to publish outside {published_root}: {destination}")

        if destination.exists():
            shutil.rmtree(destination)

        shutil.copytree(
            source,
            destination,
            symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )

        copied_files = [
            str(path.relative_to(destination)).replace("\\", "/")
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        ]
        payload = {
            "published": True,
            "source_path": str(source),
            "published_path": str(destination),
            "url": f"{self.public_base_url}/sites/{site_slug}/",
            "index_exists": (destination / "index.html").is_file(),
            "files": copied_files[:50],
            "artifact_quality": _artifact_quality_for_site(destination),
        }
        return json.dumps(payload, indent=2)

    def expose_local_http_service(
        self,
        port: int,
        path: str = "",
        name: str | None = None,
    ) -> str:
        """Return a public backend-proxy URL for an internal HTTP service."""
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
        """Resolve a path for file tools (write_text_file, publish_static_site, etc.).

        In sandbox mode the agent's cwd is ``/workspace`` (a container path).
        Absolute paths starting with ``/workspace`` are remapped to the actual
        host workdir so that host-side file writes land in the right place.
        In normal (non-sandbox) mode this is identical to
        ``_resolve_tool_path(raw_path, self.current_cwd)``.
        """
        if self._sandbox is not None:
            # Remap /workspace/* → host_workdir/*
            p = raw_path.replace("\\", "/")
            if p == "/workspace" or p.startswith("/workspace/"):
                suffix = p[len("/workspace"):]
                raw_path = self._host_workdir + suffix.replace("/", os.sep)
            elif not Path(raw_path).is_absolute():
                # Relative paths resolve against the host workdir.
                return _resolve_tool_path(raw_path, self._host_workdir)
        return _resolve_tool_path(raw_path, self.current_cwd)


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


def _slugify_site_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-._")
    return slug or "site"


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
