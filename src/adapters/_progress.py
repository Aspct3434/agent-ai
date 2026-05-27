"""Shared helpers for rendering live agent progress in chat adapters.

Both the Telegram and Discord adapters consume the engine's ``stream_task``
event stream and surface a concise, OpenClaw-style "here's what I'm doing"
feed.  This module turns a raw ``tool_call`` event into a short human line
(or ``None`` for internal/introspection tools that shouldn't be shown).
"""
from __future__ import annotations

from typing import Any

from contract import background_service_misuse_message

# Internal / read-only introspection tools that produce no user-visible step.
# Keeping these out of the feed stops it from being drowned in noise.
_SILENT_TOOLS: frozenset[str] = frozenset(
    {
        "set_task_contract",
        "update_plan",
        "set_task_graph",
        "inspect_task_graph",
        "update_task_node",
        "repair_task_graph",
        "verify_task_graph",
        "get_system_environment",
        "get_filesystem_process_evidence",
        "expand_tool_output",
        "list_allowed_directories",
        "read_file",
        "read_multiple_files",
        "list_directory",
        "directory_tree",
        "search_files",
        "get_file_info",
        "list_tables",
        "describe_table",
        "read_query",
        "list_skills",
    }
)


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())  # collapse newlines/whitespace to one line
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_tool_call(tool: str, params: dict[str, Any]) -> str | None:
    """Return a concise one-line progress message for a *tool* call.

    Returns ``None`` for internal/introspection tools (see ``_SILENT_TOOLS``)
    so the progress feed only shows meaningful, action-producing steps.
    """
    if tool in _SILENT_TOOLS:
        return None
    params = params or {}

    if tool == "execute_terminal_command":
        return f"🔧 Running: {_truncate(str(params.get('command', '')), 300)}"
    if tool == "execute_background_service":
        command = str(params.get("command", ""))
        if background_service_misuse_message(command):
            return f"[blocked] Background probe: {_truncate(command, 260)}"
        return f"🚀 Starting service: {_truncate(str(params.get('command', '')), 300)}"
    if tool in ("write_text_file", "write_file"):
        return f"📝 Writing {params.get('path') or params.get('file_path') or '(file)'}"
    if tool == "edit_file":
        return f"✏️ Editing {params.get('path') or params.get('file_path') or '(file)'}"
    if tool in ("create_directory", "move_file", "copy_file", "delete_file"):
        return f"📁 {tool.replace('_', ' ').title()}"
    if tool == "web_search":
        return f"🔍 Searching: {_truncate(str(params.get('query', '')), 160)}"
    if tool == "web_fetch":
        return f"🌐 Fetching {_truncate(str(params.get('url', '')), 200)}"
    if tool == "wait_for_port":
        return f"⏳ Waiting for port {params.get('port', '?')}"
    if tool == "expose_local_http_service":
        return f"🌍 Exposing port {params.get('port', '?')}"
    if tool in ("create_table", "write_query"):
        return "🗄️ Updating database"
    if tool == "delegate_task":
        return "🤝 Delegating a sub-task"
    if tool.startswith("browser_"):
        action = tool[len("browser_") :]
        target = params.get("url") or params.get("selector") or ""
        suffix = f": {_truncate(str(target), 120)}" if target else ""
        return f"🖥️ Browser {action}{suffix}"

    # Generic fallback for any other (non-silent) tool.
    return f"🔧 {tool}"
