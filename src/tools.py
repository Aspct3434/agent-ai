from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import logging

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

_PROBED_RUNTIMES = ("java", "python", "python3", "node", "curl", "git", "docker")

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
    # dd writing to block devices
    re.compile(r"\bdd\b.{0,60}\bof=/dev/(sd|hd|nvme|xvd|vd)[a-z]", re.IGNORECASE),
    re.compile(r"\bdd\b.{0,60}\bof=/dev/zero\b", re.IGNORECASE),
    # Fork bomb
    re.compile(r":\(\)\s*\{.*:\|:", re.DOTALL),
    # Overwriting /etc auth files
    re.compile(r">\s*/etc/(passwd|shadow|sudoers)", re.IGNORECASE),
    # Wiping whole partition / boot sector
    re.compile(r"\bmkfs\b.{0,40}/dev/(sd|hd|nvme|xvd|vd)[a-z]", re.IGNORECASE),
    re.compile(r"\bshred\b.{0,40}/dev/(sd|hd|nvme|xvd|vd)[a-z]", re.IGNORECASE),
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

GET_SYSTEM_ENVIRONMENT_TOOL: dict[str, Any] = {
    "server": "__builtin__",
    "name": "get_system_environment",
    "description": (
        "Return a JSON snapshot of the host system environment: OS type, available "
        "disk space on the working directory's filesystem, and which core runtimes "
        "(java, python, python3, node, curl, git, docker) are present on PATH. "
        "Call this whenever you need to adapt behaviour to the host platform or "
        "verify that a required runtime exists before invoking it."
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
        "the requested artifacts or service exist before giving a final answer."
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
                "description": f"Include the tail of {_BACKGROUND_LOG_PATH}. Default true.",
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
        "uses this contract to decide whether a final text answer is acceptable."
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
        f"combined stdout/stderr is appended to {_BACKGROUND_LOG_PATH}, and the new "
        "PID is returned immediately. "
        "Use this ONLY for non-terminating processes. Do NOT use it for finite work "
        "such as package installs (apt-get/pip), downloads, or builds -- those finish "
        "on their own, so run them with execute_terminal_command and let it wait; "
        "backgrounding them just forces you to poll this log and can deadlock on "
        "resource locks (e.g. apt/dpkg). Never launch the same install or service "
        "more than once. After starting a service, check "
        f"'cat {_BACKGROUND_LOG_PATH}' at most once or twice -- do not poll it in a loop."
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
        self._env_snapshot: str = _collect_system_environment()
        default_cwd = Path(os.getenv("AGENT_WORKDIR", "/app")).expanduser()
        if not default_cwd.exists():
            default_cwd = _PROJECT_ROOT
        self.current_cwd: str = str(default_cwd.resolve())
        self.published_sites_dir = Path(
            os.getenv("PUBLISHED_SITES_DIR", str(_PROJECT_ROOT / "published_sites"))
        ).expanduser()
        self.public_base_url = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")

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

        *data_dir* defaults to ``/app/data``.  The directory is created if it
        does not already exist.  The server is registered under the name
        ``"filesystem"``; a previous ``"filesystem"`` connection is
        disconnected before the new one is opened.

        Raises ``RuntimeError`` if ``npx`` is not available on PATH.
        """
        path = Path(data_dir).resolve() if data_dir else Path("/app/data")
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
        target = _resolve_tool_path(path, self.current_cwd)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        payload = {
            "written": True,
            "path": str(target),
            "exists": target.is_file(),
            "size_bytes": target.stat().st_size if target.exists() else 0,
            "current_working_directory": self.current_cwd,
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
        source = _resolve_tool_path(source_path or self.current_cwd, self.current_cwd)
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

        connectable = False
        error: str | None = None
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
        }
        return json.dumps(payload, indent=2)

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
        except asyncio.TimeoutError:
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
        exit_code: int = proc.returncode if proc.returncode is not None else -1

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

        Combined stdout and stderr are appended to ``_BACKGROUND_LOG_PATH`` so the
        agent can inspect them later by ``cat``-ing that file.  The process runs in
        ``self.current_cwd`` and is started in its own session so it survives
        independently of this server.

        Returns a dict with keys ``pid`` (int, or ``None`` on failure),
        ``status`` (``"launched"`` or ``"error"``), ``message``, and ``log_file``.
        """
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
        """Shut down all connected servers."""
        for name in list(self._stacks):
            await self._disconnect(name)

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


def _collect_system_environment() -> str:
    """Collect a JSON snapshot of the host environment synchronously.

    Called once at ``ToolManager.__init__`` time so there is no async
    overhead on the first ``get_system_environment`` call.
    """
    system = platform.system()  # 'Linux', 'Darwin', 'Windows', ...
    os_label = {"Darwin": "macOS", "Linux": "Linux", "Windows": "Windows"}.get(
        system, system
    )

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

    return json.dumps(
        {
            "os": os_label,
            "os_version": platform.version(),
            "machine": platform.machine(),
            "python_executable": sys.executable,
            "disk_cwd": disk,
            "runtimes": runtimes,
            "shell": shell_info,
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
# Skills-server bootstrap helpers
# ---------------------------------------------------------------------------

def _ensure_skills_dir(skills_dir: Path) -> None:
    """Create *skills_dir* and write the two bootstrap files if absent."""
    skills_dir.mkdir(parents=True, exist_ok=True)
    _write_if_absent(skills_dir / "_skill.py", _SKILL_DECORATOR_SRC)
    _write_if_absent(skills_dir / "server.py", _SKILLS_SERVER_SRC)


def _write_if_absent(path: Path, content: str) -> None:
    if not path.exists():
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
