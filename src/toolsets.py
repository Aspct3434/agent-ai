"""Tool categorisation and toolset routing.

A *toolset* is a named subset of the full tool list.  Activating one narrows
what the LLM sees to the tools relevant for the current task class, keeping the
context window tight and reducing the chance of the model picking the wrong
primitive.

Usage
-----
The agent selects a toolset by including the optional ``toolset`` field in the
``set_task_contract`` call::

    set_task_contract(mode="execute", toolset="coding", ...)

The contract system stores the value; :func:`filter_tools_by_toolset` is then
called each iteration before the LLM sees the tool schemas.

Toolsets
--------
``all``       Everything (default when no toolset is declared).
``research``  Web search/fetch + read-only tools.  No shell, no file writes.
``coding``    Files + terminal + web (for doc lookups / error search).
``web``       Browser + web_fetch + web_search + file writes.
``data``      SQLite + file writes + terminal.
``ops``       Shell + file writes + delegation.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Canonical tool groups
# ---------------------------------------------------------------------------

_SAFE_TOOLS: frozenset[str] = frozenset(
    {
        "get_system_environment",
        "get_filesystem_process_evidence",
        "expand_tool_output",
        "set_task_contract",
        "update_plan",
        # MCP filesystem (read-only)
        "list_directory",
        "directory_tree",
        "read_file",
        "read_multiple_files",
        "get_file_info",
        "search_files",
        "list_allowed_directories",
        # MCP SQLite (read-only)
        "list_tables",
        "describe_table",
        "read_query",
    }
)

_WEB_TOOLS: frozenset[str] = frozenset(
    {
        "web_search",
        "web_fetch",
        "browser_navigate",
        "browser_get_text",
        "browser_screenshot",
        "browser_click",
        "browser_fill",
        "browser_evaluate",
    }
)

_FILE_WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "write_text_file",
        "expose_local_http_service",
        # MCP filesystem (write)
        "write_file",
        "edit_file",
        "create_directory",
        "move_file",
        "copy_file",
        "delete_file",
    }
)

_EXEC_TOOLS: frozenset[str] = frozenset(
    {"execute_terminal_command", "execute_background_service"}
)

_DATA_WRITE_TOOLS: frozenset[str] = frozenset({"create_table", "write_query"})

_DELEGATION_TOOLS: frozenset[str] = frozenset({"delegate_task"})

# Builtin capabilities that must reach the model regardless of the active
# toolset (control, introspection, scheduling, skill authoring, image gen).
# Force-included by filter_tools_by_toolset so a narrow toolset never hides them.
_ALWAYS_TOOLS: frozenset[str] = frozenset(
    {
        "set_task_contract",
        "update_plan",
        "list_skills",
        "create_skill",
        "recall_memory",
        "schedule_task",
        "list_scheduled_tasks",
        "analyze_image",
        "generate_image",
    }
)

# ---------------------------------------------------------------------------
# Named toolsets
# ---------------------------------------------------------------------------

#: All valid toolset names.
TOOLSET_NAMES: frozenset[str] = frozenset(
    {"all", "research", "coding", "web", "data", "ops"}
)

TOOLSETS: dict[str, frozenset[str]] = {
    # Default: the full tool list.
    "all": (
        _SAFE_TOOLS
        | _WEB_TOOLS
        | _FILE_WRITE_TOOLS
        | _EXEC_TOOLS
        | _DATA_WRITE_TOOLS
        | _DELEGATION_TOOLS
    ),
    # Read-only research — no file mutations, no shell execution.
    # Good for: "What is X?", "Summarise this URL", "Find libraries for Y".
    "research": _SAFE_TOOLS | _WEB_TOOLS,
    # Software development — files, terminal, web for doc/error lookups.
    # Good for: "Build a REST API", "Fix this bug", "Write tests for X".
    "coding": (
        _SAFE_TOOLS
        | _WEB_TOOLS
        | _FILE_WRITE_TOOLS
        | _EXEC_TOOLS
        | _DELEGATION_TOOLS
    ),
    # Browser-heavy automation — scraping, form filling, screenshots.
    # Good for: "Log into X and extract Y", "Screenshot this page", "Fill this form".
    "web": _SAFE_TOOLS | _WEB_TOOLS | _FILE_WRITE_TOOLS,
    # Data work — SQLite mutations, CSV processing, terminal analytics.
    # Good for: "Load this CSV into SQLite", "Run this query", "Aggregate X".
    "data": _SAFE_TOOLS | _DATA_WRITE_TOOLS | _FILE_WRITE_TOOLS | _EXEC_TOOLS,
    # Ops / DevOps — shell + files + delegation, no browser.
    # Good for: "Deploy X", "Set up a cron job", "Configure Y service".
    "ops": _SAFE_TOOLS | _EXEC_TOOLS | _FILE_WRITE_TOOLS | _DELEGATION_TOOLS,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filter_tools_by_toolset(
    all_tools: list[dict[str, Any]],
    toolset_name: str,
) -> list[dict[str, Any]]:
    """Return only the tool *schemas* that belong to *toolset_name*.

    Unknown names fall back silently to ``"all"``.
    The control/introspection builtins in ``_ALWAYS_TOOLS`` (set_task_contract,
    update_plan, list_skills, create_skill, schedule_task, …) are always
    included so a narrow toolset never hides them from the model.
    """
    allowed = TOOLSETS.get(toolset_name, TOOLSETS["all"]) | _ALWAYS_TOOLS
    return [t for t in all_tools if t.get("name") in allowed]
