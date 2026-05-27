from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Literal

import litellm
from mcp.types import CallToolResult
from pydantic import BaseModel

from checkpointer import StateCheckpointer
from contract import (
    GENERIC_CAP_SKIP_TOOLS,
    HOST_EXECUTION_TOOLS,
    MAX_CONSECUTIVE_TERMINAL_COMMANDS,
    MAX_IDENTICAL_COMMAND_RUNS,
    MAX_IDENTICAL_TOOL_CALLS,
    TASK_CONTRACT_TOOL,
    attempted_tool_names,
    blocked_action_tool_message,
    build_incomplete_contract_cap_message,
    build_task_contract_instruction,
    can_stream_text_before_final,
    consecutive_terminal_cap_message,
    consecutive_terminal_run_length,
    contract_completion_status,
    duplicate_command_message,
    filter_tool_schemas,
    host_command_runs_since_state_change,
    identical_tool_call_runs_since_state_change,
    last_host_command,
    latest_task_contract,
    normalize_command,
    repeated_command_message,
    repeated_tool_call_message,
    run_set_task_contract,
    should_block_tool_for_action_task,
    terminal_failure_recovery_message,
    terminal_failure_since_diagnostic,
    tool_names_for_contract_status,
)
from evaluator import (
    ExecutionStep,
    ExecutionTrajectory,
    SkillDistiller,
    extract_and_update_user_profile,
)
from llm_utils import (
    _acompletion_stream_with_retry,
    _is_async_iterable,
    _is_rate_limit_error,
    _make_final_answer,
    _prepare_llm_request_messages,
    _rate_limit_user_message,
    _sanitize_messages_for_llm,
)
from planning import (
    _build_contract_continuation_instruction,
    _build_contract_execution_instruction,
    _build_executive_summary,
    _classify_tool_result,
    _count_done_plan_steps,
    _count_successful_side_effects,
    _prune_message_window,
    _run_update_plan,
    _store_iteration_cap_memory,
)
from task_graph import (
    GRAPH_CONTROL_TOOLS,
    INSPECT_TASK_GRAPH_TOOL_NAME,
    REPAIR_TASK_GRAPH_TOOL_NAME,
    SET_TASK_GRAPH_TOOL_NAME,
    UPDATE_TASK_NODE_TOOL_NAME,
    VERIFY_TASK_GRAPH_TOOL_NAME,
    TaskGraphEngine,
)
from tools import ToolManager
from toolsets import filter_tools_by_toolset

if TYPE_CHECKING:
    from evaluator import SkillRegistry
    from evolution import EvolutionEngine
    from memory import HybridMemory, UserProfileStore
    from scheduler import CronScheduler

logger = logging.getLogger(__name__)


def _llm_first_choice(response: Any) -> Any:
    """Return ``response.choices[0]`` without mypy union-attr noise.

    litellm's return type is ``ModelResponse | TextCompletionResponse | None``.
    In agent code we always guard for None before calling this; the
    ``TextCompletionResponse`` variant is only returned for legacy /completions
    endpoints which we never call. This wrapper isolates the unsafe cast so
    type: ignore comments don't scatter across business logic.
    """
    return response.choices[0]  # type: ignore[union-attr]


def _successful_tool_result_names(steps: list[ExecutionStep]) -> set[str]:
    return {
        str(step.metadata.get("tool_name"))
        for step in steps
        if step.kind == "tool_result"
        and step.metadata.get("tool_name")
        and not step.metadata.get("is_error")
    }


def _contract_request_text(contract: dict[str, Any] | None, original_prompt: str) -> str:
    if not contract:
        return original_prompt
    pieces = [
        original_prompt,
        str(contract.get("summary") or ""),
        *[str(item) for item in contract.get("success_criteria") or []],
    ]
    return " ".join(piece for piece in pieces if piece).lower()


def _load_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _short_evidence_value(value: str, max_chars: int = 700) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _evidence_items_from_payload(
    content: str,
    *,
    tool_name: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    payload = _load_json_object(content)
    if not payload:
        return []

    items: list[tuple[str, str]] = []
    url = payload.get("url")
    if payload.get("exposed") and payload.get("connectable"):
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            items.append(("Service URL", url))
        port = payload.get("port")
        if port:
            items.append(("Port", str(port)))

    if payload.get("written") and payload.get("exists"):
        path = payload.get("path")
        if isinstance(path, str) and path:
            items.append(("File", path))

    if tool_name == "execute_terminal_command" and payload.get("exit_code") == 0:
        command = (arguments or {}).get("command")
        stdout = payload.get("stdout")
        if isinstance(command, str) and command:
            items.append(("Command", command))
        if isinstance(stdout, str) and stdout.strip():
            items.append(("Command output", _short_evidence_value(stdout)))

    if payload.get("status") == "launched" and payload.get("pid"):
        items.append(("Process", f"PID {payload['pid']}"))
        log_file = payload.get("log_file")
        if isinstance(log_file, str) and log_file:
            items.append(("Log", log_file))

    paths = payload.get("paths")
    if isinstance(paths, list):
        for path_info in paths:
            if not isinstance(path_info, dict) or not path_info.get("exists"):
                continue
            path = path_info.get("path")
            if isinstance(path, str) and path:
                items.append(("Path", path))

    return items


def _successful_evidence_items(
    steps: list[ExecutionStep],
    messages: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []

    for step in steps:
        if step.kind != "tool_result" or step.metadata.get("is_error"):
            continue
        for item in _evidence_items_from_payload(
            step.content,
            tool_name=str(step.metadata.get("tool_name") or ""),
            arguments=step.metadata.get("arguments") or {},
        ):
            if item not in items:
                items.append(item)

    # Follow-up turns start with a fresh ExecutionStep list, but durable session
    # history still contains prior tool outputs. Recover concrete evidence from
    # those tool messages so "give me the URL/path/output" can be answered literally.
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for item in _evidence_items_from_payload(content):
            if item not in items:
                items.append(item)

    return items


def _should_include_evidence_summary(
    contract: dict[str, Any] | None,
    original_prompt: str,
) -> bool:
    evidence = set((contract or {}).get("evidence_requirements") or [])
    if evidence - {"none"}:
        return True

    text = original_prompt.lower()
    return any(
        token in text
        for token in (
            "url",
            "link",
            "served",
            "site",
            "website",
            "open",
            "view",
            "file",
            "path",
            "where",
            "output",
            "result",
            "service",
            "port",
        )
    )


def _ensure_evidence_in_final_response(
    final_response: str,
    *,
    contract: dict[str, Any] | None,
    original_prompt: str,
    steps: list[ExecutionStep],
    messages: list[dict[str, Any]],
) -> str:
    if not _should_include_evidence_summary(contract, original_prompt):
        return final_response

    missing_items = [
        (label, value)
        for label, value in _successful_evidence_items(steps, messages)
        if value not in final_response
    ]
    if not missing_items:
        return final_response

    prefix = final_response.rstrip()
    evidence_block = "Evidence:\n" + "\n".join(
        f"- {label}: {value}" for label, value in missing_items
    )
    return f"{prefix}\n\n{evidence_block}" if prefix else evidence_block


def _uses_moonshot_provider(model: str) -> bool:
    return model.startswith("moonshot/") or model.startswith("kimi-")


def _moonshot_required_tool_instruction(tool_schemas: list[dict[str, Any]]) -> str:
    names = [
        str(tool.get("function", {}).get("name"))
        for tool in tool_schemas
        if tool.get("function", {}).get("name")
    ]
    if len(names) == 1:
        return (
            f"Call the `{names[0]}` tool now. Return a tool call only; do not "
            "answer in prose, do not explain, and do not say the task is in progress."
        )
    return (
        "Call exactly one of these tools now: "
        f"{', '.join(f'`{name}`' for name in names)}. Return a tool call only; "
        "do not answer in prose, do not explain, and do not say the task is in progress."
    )


def _preferred_recovery_tool_name(
    contract: dict[str, Any] | None,
    status: dict[str, Any],
    steps: list[ExecutionStep],
    original_prompt: str,
    available_tool_names: set[str],
) -> str | None:
    """Choose one tool to force after the model emits prose before completion."""
    if contract is None or contract.get("mode") != "execute":
        return None

    missing = set(status.get("missing") or [])
    if "task_graph" in missing and SET_TASK_GRAPH_TOOL_NAME in available_tool_names:
        return SET_TASK_GRAPH_TOOL_NAME
    if "task_graph_open" in missing:
        if UPDATE_TASK_NODE_TOOL_NAME in available_tool_names:
            return UPDATE_TASK_NODE_TOOL_NAME
        if REPAIR_TASK_GRAPH_TOOL_NAME in available_tool_names:
            return REPAIR_TASK_GRAPH_TOOL_NAME
    if "task_graph_proof" in missing:
        if VERIFY_TASK_GRAPH_TOOL_NAME in available_tool_names:
            return VERIFY_TASK_GRAPH_TOOL_NAME
        if UPDATE_TASK_NODE_TOOL_NAME in available_tool_names:
            return UPDATE_TASK_NODE_TOOL_NAME
    if "plan" in missing and "update_plan" in available_tool_names:
        return "update_plan"

    evidence_missing = [
        item
        for item in missing
        if item not in {"plan", "plan_open_steps", "task_graph", "task_graph_open", "task_graph_proof"}
    ]
    if not evidence_missing:
        if "plan_open_steps" in missing and "update_plan" in available_tool_names:
            return "update_plan"
        return None

    succeeded = _successful_tool_result_names(steps)
    text = _contract_request_text(contract, original_prompt)

    for requirement in evidence_missing:
        if requirement == "artifact_quality":
            if "write_text_file" in available_tool_names:
                return "write_text_file"

        if requirement == "running_http_service":
            if (
                "execute_background_service" not in succeeded
                and "execute_background_service" in available_tool_names
            ):
                return "execute_background_service"
            if "expose_local_http_service" in available_tool_names:
                return "expose_local_http_service"

        if requirement == "running_tcp_service":
            if (
                "execute_background_service" not in succeeded
                and "execute_background_service" in available_tool_names
            ):
                return "execute_background_service"
            if "wait_for_port" in available_tool_names:
                return "wait_for_port"
            if "get_filesystem_process_evidence" in available_tool_names:
                return "get_filesystem_process_evidence"

        if requirement == "database_mutation":
            for name in ("write_query", "create_table"):
                if name in available_tool_names:
                    return name

        if requirement == "command_output" and "execute_terminal_command" in available_tool_names:
            return "execute_terminal_command"

        if requirement == "filesystem_artifact":
            if any(name in succeeded for name in HOST_EXECUTION_TOOLS):
                if "get_filesystem_process_evidence" in available_tool_names:
                    return "get_filesystem_process_evidence"
            if any(token in text for token in ("install", "build", "download", "run", "compile")):
                if "execute_terminal_command" in available_tool_names:
                    return "execute_terminal_command"
            for name in ("write_text_file", "execute_terminal_command", "get_filesystem_process_evidence"):
                if name in available_tool_names:
                    return name

    return None


def _recent_tool_failures(steps: list[ExecutionStep], limit: int = 3) -> list[ExecutionStep]:
    failures: list[ExecutionStep] = []
    for step in reversed(steps):
        if step.kind != "tool_result":
            continue
        if step.metadata.get("is_error"):
            failures.append(step)
            if len(failures) >= limit:
                break
        elif failures:
            break
    return list(reversed(failures))


def _build_self_repair_instruction(steps: list[ExecutionStep]) -> str:
    failures = _recent_tool_failures(steps)
    lines = []
    for step in failures:
        tool_name = str(step.metadata.get("tool_name") or "unknown_tool")
        _, text = _classify_tool_result(tool_name, step.content)
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if first_line:
            lines.append(f"- {tool_name}: {first_line[:220]}")
        else:
            lines.append(f"- {tool_name}: failed with no output")
    failure_block = "\n".join(lines) or "- unknown failure"
    return (
        "SELF-REPAIR MODE: one or more tool calls failed.\n"
        f"Recent failures:\n{failure_block}\n"
        "Before running another mutating command, diagnose the failure and choose a "
        "different strategy. Use get_system_environment to confirm the execution "
        "environment, get_filesystem_process_evidence to inspect actual files, "
        "processes, and ports, expand_tool_output for truncated logs, and web_search "
        "or web_fetch for unfamiliar error messages. Do not keep rephrasing the same "
        "shell operation with different wrappers. If the error proves the requested "
        "execution environment is unavailable, report that blocker clearly and stop "
        "instead of mutating the wrong host."
    )


def _json_tool_result_has_error(content: str, *, false_open_is_error: bool = False) -> bool:
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    if data.get("error"):
        return True
    if false_open_is_error and data.get("open") is False:
        return True
    return False


def _filesystem_process_evidence_has_negative_findings(content: str) -> bool:
    """True when an evidence probe disproves the thing it was asked to verify."""
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    if data.get("error"):
        return True

    paths = data.get("paths")
    if isinstance(paths, list) and any(
        isinstance(path, dict) and path.get("exists") is False for path in paths
    ):
        return True

    pids = data.get("pids")
    if isinstance(pids, list) and any(
        isinstance(pid, dict) and pid.get("running") is False for pid in pids
    ):
        return True

    process_names = data.get("process_names")
    process_entries = [
        process for process in process_names or [] if isinstance(process, dict)
    ] if isinstance(process_names, list) else []
    if process_entries and not any(
        int(process.get("count") or 0) > 0 for process in process_entries
    ):
        return True

    ports = data.get("ports")
    port_entries = [port for port in ports or [] if isinstance(port, dict)] if isinstance(ports, list) else []
    if port_entries and not any(port.get("connectable") is True for port in port_entries):
        return True

    return False


MAX_REACT_ITERATIONS = int(os.getenv("AGENT_MAX_REACT_ITERATIONS", "16"))

# Action tasks that change the host (install a runtime, download artifacts, write
# config, start and verify a service) legitimately need more steps than a Q&A
# turn -- each shell command is one iteration. They get a larger budget so the
# loop does not stop one step short of done and force the user to type
# "continue". Simple/informational tasks keep the smaller, cheaper cap.
_ACTION_MAX_REACT_ITERATIONS = max(
    MAX_REACT_ITERATIONS, int(os.getenv("AGENT_ACTION_MAX_REACT_ITERATIONS", "30"))
)

# Background auto-continuation: when a batch of iterations hits its cap but the
# task is still making progress and isn't finished, automatically run another
# batch instead of stopping and asking the user to type "continue". Bounded by
# _MAX_AUTO_CONTINUE_BATCHES so a non-progressing task can't run forever; a batch
# that makes no progress is never auto-continued.
_AUTO_CONTINUE_ENABLED = os.getenv("AGENT_AUTO_CONTINUE", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_MAX_AUTO_CONTINUE_BATCHES = max(1, int(os.getenv("AGENT_MAX_AUTO_CONTINUE_BATCHES", "3")))

# Escalate the main loop from the fast model to the strong model after this many
# consecutive iterations whose every tool call errored (only when the tiers differ).
_ESCALATE_AFTER_CONSECUTIVE_ERROR_ITERS = 2

_LLM_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "32768"))

# Larger budget for the final synthesised answer (e.g. the progress summary on a
# cap hit), which is prose rather than tool-call arguments and can need more room.
_LLM_FINAL_MAX_TOKENS = int(os.getenv("AGENT_FINAL_MAX_TOKENS", "8192"))

# Persist a checkpoint on the first iteration and then every Nth, instead of on
# every iteration, to cut redundant full-history writes.
_CHECKPOINT_EVERY = max(1, int(os.getenv("AGENT_CHECKPOINT_EVERY", "4")))

# The old first-turn bootstrap duplicated the tool list into chat history while
# the same tool schemas were also sent in the tools block. Leave it available for
# local debugging, but keep it off by default to reduce input TPM pressure.
_BOOTSTRAP_SESSIONS = os.getenv("AGENT_BOOTSTRAP_SESSION", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Cap on the number of per-session conversation histories kept in memory. The
# least-recently-used session is evicted past this so a long-lived server does
# not grow without bound (an evicted session can still be resumed from a
# checkpoint when a StateCheckpointer is configured).
_MAX_SESSIONS = int(os.getenv("AGENT_MAX_SESSIONS", "256"))

# Tools known to be read-only/side-effect-free, so multiple of them issued in one
# turn can run concurrently. Conservative allowlist: the read-only builtins plus
# the standard read tools of the sqlite and filesystem MCP servers in this stack.
# Anything not listed (terminal/background commands, MCP writes, delegation) runs
# serially in its original order so cwd mutations and write ordering stay correct.
_PARALLEL_SAFE_TOOLS: frozenset[str] = frozenset(
    {
        "get_system_environment",
        "get_filesystem_process_evidence",
        # Web tools are read-only network calls â€" safe to run concurrently
        "web_fetch",
        "web_search",
        "expand_tool_output",
        "set_task_contract",
        "update_plan",
        "inspect_task_graph",
        "verify_task_graph",
        # Scheduler / skill registry introspection (read-only)
        "list_scheduled_tasks",
        "list_skills",
        # MCP filesystem server (read-only ops)
        "list_tables",
        "describe_table",
        "read_query",
        "list_directory",
        "directory_tree",
        "read_file",
        "read_multiple_files",
        "get_file_info",
        "search_files",
        "list_allowed_directories",
    }
)

# Tool-result compaction. A tool output at or below this size is shown to the
# model in full; anything larger is condensed to a head+tail preview plus a
# handle the model can pass to expand_tool_output to page through the rest on
# demand. Nothing is ever silently destroyed -- the full text stays in the
# durable history and is retrievable -- so per-iteration tokens stay bounded
# without blinding the agent to verification output (ls / cat / ps / the
# background log) or the PID returned by execute_background_service.
_TOOL_OUTPUT_INLINE_MAX_CHARS = 3000
_TOOL_OUTPUT_PREVIEW_HEAD_LINES = 20
_TOOL_OUTPUT_PREVIEW_TAIL_LINES = 20
# Hard cap on a single expand_tool_output window so paging can't re-bloat context.
_EXPAND_MAX_LINES = 500

_SELF_REPAIR_TOOLS: frozenset[str] = frozenset(
    {
        "get_system_environment",
        "get_filesystem_process_evidence",
        "web_search",
        "web_fetch",
        "expand_tool_output",
        "wait_for_port",
    }
)

SYSTEM_DIRECTIVE = (
    "You are an autonomous engineering framework, not a conversational chatbot. "
    "NEVER output generic greetings, and NEVER ask the user what they want to do. "
    "If the user says hello or gives a vague input, do not reply with pleasantries. "
    "Instead, proactively use your tools: inspect the database schema, check the "
    "ChromaDB memory state, or look at the skills directory, and report a highly "
    "technical summary of the system state. Act immediately. Execute silently. "
    # â”€â”€ Task contract â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "For every new user task, your FIRST tool call is set_task_contract. Use it to "
    "declare whether the task is a pure text answer or requires real host-side "
    "execution evidence. For any execute-mode task that needs more than one step, "
    "call update_plan next with a short, ordered checklist of the concrete steps. "
    "Then work through them in order, calling update_plan again to mark each step "
    "'in_progress' when you start it and 'done' or 'failed' when it finishes. Keep "
    "the plan in sync with what has actually happened, use it to avoid repeating "
    "steps that are already done, and do not give a final answer until every step "
    "is 'done' or 'failed'. "
    # â”€â”€ Web tools â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "You have two web tools that make you general-purpose without custom skills: "
    "web_search and web_fetch. Use them freely. "
    "web_search: find information, look up docs, research a topic, locate solutions "
    "to errors. Returns titles + URLs + snippets. "
    "web_fetch: read any URL â€” documentation, GitHub files, API endpoints, search "
    "results, news, technical specs. Returns clean readable text. "
    "Standard research pattern: web_search to find candidate URLs, then web_fetch "
    "the most relevant ones to read the full content, then synthesise your answer. "
    "NEVER write a custom skill just to do web research â€” use these tools directly. "
    "When you encounter an error (build failure, missing dependency, unfamiliar API), "
    "read the tool result, identify the root cause, and web_search the exact error "
    "message before assuming you need a custom workaround. A failed tool call is "
    "not a reason to ask the user to debug for you: use your environment, evidence, "
    "web, and log-inspection tools to repair your own approach. "
    # â”€â”€ Terminal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "You have root access to a terminal shell tool. If the user requests an "
    "installation, setup, or file-system operation, DO NOT explain how the user "
    "can do it manually. Immediately use execute_terminal_command or "
    "execute_background_service to perform the operation, configure the "
    "active execution environment, and verify that the process is running there. "
    "After setting the task contract, if the task requires creating, making, "
    "building, scaffolding, writing, editing, serving, or configuring artifacts, "
    "continue with tool calls that perform the work. Do not say 'I'll create it' "
    "or describe a plan as a final answer unless you have already made the files "
    "or executed the requested operation and verified the required evidence. "
    "When using execute_terminal_command, you must NEVER assume a file was "
    "downloaded or an installation succeeded based on the command printout. "
    "Every time you create a file, download a package, or start a server, you "
    "MUST execute a follow-up verification check (e.g., ls -la, cat, or ps aux) "
    "in the very next iteration to verify the physical state inside the active "
    "execution environment before declaring a task finished. "
    "If you are asked to START or RUN a server or continuous process (one that never "
    "terminates on its own), you MUST use the execute_background_service tool; using "
    "the standard terminal for such a process will deadlock. "
    "But execute_background_service is ONLY for non-terminating processes. Finite work "
    "that simply takes a while -- installing packages (apt-get/pip), downloading files, "
    "building, running tests -- MUST go through execute_terminal_command, which waits "
    "for the command to finish; do not background it. Backgrounding a finite command "
    "forces you to poll the background-service log and can deadlock on resource locks "
    "(e.g. apt/dpkg). After you start a real background service, call wait_for_port "
    "to wait for it to bind -- that tool blocks internally and returns only once the "
    "port opens (or the timeout expires), so you never burn multiple API calls polling. "
    "Read the background-service log at most once -- the start call returns its exact "
    "path -- to confirm startup or diagnose a failure; NEVER poll any read-only tool in "
    "a loop to wait for state to change (an identical call returns no new information "
    "and just wastes your rate-limit budget -- wait once with a blocking primitive "
    "instead); and never launch the same service command again while one is already "
    "running. "
    "NEVER append '&' to a command in execute_terminal_command to background it: that "
    "tool is synchronous and returns only when the command exits, so a trailing '&' "
    "keeps nothing running and will mislead you -- use execute_background_service for "
    "anything that must keep running. "
    "Before installing or downloading something, first check whether it is already "
    "present (e.g. get_system_environment, or a quick 'which'/'test -f' probe), and "
    "never repeat a step already listed under Completed_Actions in the executive "
    "summary; build on finished work and respect ordering (do not start a service "
    "before its prerequisites are installed and in place). "
    # â”€â”€ Check tooling once; never assume, never repeat-probe â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "Tooling availability is environment-specific, so it is NEVER safe to assume a "
    "runtime exists. Call get_system_environment ONCE and read its 'runtimes' block "
    "to learn what is actually on PATH (node, npm, npx, python, rustc, cargo, ...). "
    "That single check is authoritative: do NOT probe the same tool over and over "
    "with 'node --version', 'node -v', 'where npm', 'npm --version', powershell "
    "variants, etc. -- check once, then decide. If a runtime is present, use it and "
    "do not try to reinstall it (e.g. as a non-root user 'apt-get install nodejs' "
    "will only fail). If a runtime is genuinely absent and you cannot install it, do "
    "NOT keep retrying the same command -- switch to an approach that does not need "
    "it (for a website, that means plain HTML/CSS/JS, which needs no toolchain). "
    # â”€â”€ Honor the requested technology (no silent substitution) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "When the user names a specific framework, language, library, or tool (for "
    "example 'using React', 'in Rust', 'with Next.js', 'a Flask API'), you MUST "
    "deliver exactly that. NEVER silently substitute a different stack -- e.g. do "
    "not replace a requested interactive React app with a server-rendered Flask "
    "page, or swap one language for another -- just because a runtime looks missing. "
    "Verify with get_system_environment first to see whether the runtime is "
    "available, and if it is, use it directly. If a genuinely required runtime is "
    "truly unavailable and cannot be installed, say so explicitly in your answer "
    "rather than quietly shipping a different deliverable and claiming the task is "
    "done. "
    # â”€â”€ Content generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "When asked to produce content (a website, a document, sample data, copy), "
    "generate complete, realistic content yourself -- do NOT ask the user what to "
    "include or leave placeholder text unless they explicitly request a skeleton. "
    # â”€â”€ Choosing a web stack: simplest that works (DEFAULT to vanilla HTML) â”€â”€â”€
    "Choosing how to build a website: pick the SIMPLEST approach that satisfies the "
    "request. If the user asks for 'a website', 'a web page', 'an interactive "
    "website', a landing page, or an info/marketing page WITHOUT naming a framework, "
    "build a single self-contained static site -- one index.html with embedded "
    "<style> CSS and vanilla <script> JavaScript. Vanilla JS is fully interactive "
    "(buttons, tabs, toggles, accordions, quizzes, counters, sliders, canvas "
    "animations, charts) and needs NO build step, NO Node.js, and NO npm, so it works "
    "in every environment and servees immediately. Write it with write_text_file, "
    "then serve it from the Docker workspace with a simple HTTP server, wait for "
    "the port, expose that port, then verify -- no scaffold, no install. Reach for React, Vue, Svelte, Vite, or Next -- or any npm-based "
    "build -- ONLY when the user EXPLICITLY names that framework; in that case first "
    "confirm node/npm exist via get_system_environment, then follow the build recipe "
    "below. Do NOT default to a heavy toolchain the user did not ask for, and do not "
    "scaffold a React/Vite project for a request that never mentioned React. "
    # â”€â”€ Website quality bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "CRITICAL QUALITY RULE for every website you create: a bare skeleton with "
    "unstyled HTML tags is NEVER acceptable. Every website deliverable MUST have: "
    "(1) a complete CSS design with a colour palette, typography (font-family, "
    "font-size, line-height), spacing (margins, padding), and responsive layout; "
    "(2) realistic, substantive content -- real headings, paragraphs, descriptions, "
    "feature lists -- not placeholder text or one-liners; (3) visual polish: "
    "background colours/gradients, border-radius, box-shadow, hover/transition "
    "effects, and a cohesive visual hierarchy; (4) at minimum 100+ words of visible "
    "content. Think of the output as a page you would show a client, not a code "
    "snippet. Generate ALL content yourself â€” do not leave blank sections or TODOs. "
    "If the file is getting large, that is FINE â€” write the complete content in a "
    "single write_text_file call. The tool supports large content payloads. "
    "If the user asks to host, serve, serve, or get a browser URL for a static "
    "website, start a normal HTTP server in the Docker workspace with "
    "execute_background_service (for example python -m http.server from the site "
    "directory), wait_for_port, then expose_local_http_service. The public URL "
    "must use the backend's generic /proxy/<port>/ path. A website deliverable "
    "is not complete when files merely exist: after creating or editing the site, "
    "serve it, verify it, and include the working localhost proxy URL in the final "
    "answer. The user must not need to ask separately for hosting or port setup. "
    # â”€â”€ React / Vite / SPA build-and-serve recipe (only when requested) â”€â”€â”€â”€â”€
    "When the user EXPLICITLY asked for React/Vue/Svelte/Vite/Next (otherwise use "
    "the vanilla single-file approach above): such a single-page app is NOT a static "
    "site until it is BUILT, and it needs node/npm. Follow this exact sequence: "
    "(0) confirm node and npm are present in the get_system_environment 'runtimes' "
    "block; if they are absent and cannot be installed, tell the user the framework "
    "runtime is unavailable rather than looping on failed npm calls. "
    "(1) scaffold the project -- e.g. 'npm create vite@latest <dir> -- --template "
    "react'; execute_terminal_command has NO interactive terminal, so prefix "
    "scaffold/create commands with 'CI=1' (and append a non-interactive flag where "
    "one exists) so they never block waiting for a 'Ok to proceed? (y)' prompt and "
    "hang until timeout; if scaffolding still will not run non-interactively, write "
    "the project files directly with write_text_file instead. "
    "(2) run 'npm install' in the project directory and let it finish. "
    "(3) run 'npm run build' -- this produces the production bundle in the 'dist/' "
    "folder (Vite) or 'build/' folder (CRA). "
    "(4) serve the built output directory (the 'dist/' or 'build/' folder that "
    "contains the compiled index.html) with execute_background_service, then "
    "wait_for_port and expose_local_http_service -- NEVER serve the project root, "
    "which has only source files and no compiled index.html. "
    "(5) verify the returned proxy URL. "
    "If the user instead wants a live dev server, run 'npm run dev' with "
    "execute_background_service and expose it with expose_local_http_service. "
    "Do not declare the task done until the built app is served or served and "
    "verified -- and do not waste turns re-writing the same source file you already "
    "wrote; check Completed_Actions first. "
    "If you start any HTTP app, API, dashboard, notebook, frontend dev server, or "
    "browser UI on an internal container port, call expose_local_http_service after "
    "the service is listening and give the returned /proxy URL. Do not ask the user "
    "to manually open ports or edit Docker Compose for normal HTTP access. "
    # ── Scheduled tasks vs background services ────────────────────────────
    "For time-based recurring work (heartbeats, periodic reports, cleanup), use "
    "schedule_task — it registers a persistent cron/interval/once job with the "
    "agent scheduler, survives restarts, and is visible via list_scheduled_tasks. "
    "execute_background_service is only for non-terminating daemons and servers "
    "that must run continuously inside the current container session. Never use "
    "execute_background_service for work that should repeat on a schedule. "
    "After calling schedule_task, always call list_scheduled_tasks to confirm the "
    "job was registered and show the user its job_id and next-run time. "
    # â”€â”€ Output format â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "Final answers must use clear GitHub-flavored Markdown. Use bold section labels "
    "and concise bullets when helpful. Include one to three relevant emoji characters "
    "in user-facing status summaries, but keep the tone professional and do not "
    "overload the answer with emojis."
)

# Per-type directives injected into sub-agents created by delegate_task.
_SUB_AGENT_DIRECTIVES: dict[str, str] = {
    "researcher": (
        "You are a specialist research sub-agent. Your sole purpose is to gather, "
        "synthesise, and return structured factual information relevant to the task "
        "you have been given. Do NOT write code, make edits, or take side-effecting "
        "actions. Retrieve data, cross-reference sources, and produce a precise, "
        "citation-rich summary. Terminate as soon as you have a complete answer."
    ),
    "coder": (
        "You are a specialist coding sub-agent. Your sole purpose is to produce, "
        "modify, or refactor source code exactly as instructed. Write clean, minimal, "
        "idiomatic code with no unnecessary abstractions. Do NOT add explanatory prose "
        "beyond inline comments that explain non-obvious invariants. Return only the "
        "final implementation -- no preamble, no caveats, no trailing summaries."
    ),
    "auditor": (
        "You are a specialist security and correctness auditor. Ruthlessly scan every "
        "line of the provided code for syntax errors, logic bugs, race conditions, "
        "injection vulnerabilities (SQL, command, XSS), insecure defaults, missing "
        "input validation, and any other OWASP Top 10 or CWE-ranked weakness. "
        "Report every finding with: file, line number, severity (CRITICAL/HIGH/MEDIUM/"
        "LOW), a one-line description, and a concrete remediation. Miss nothing. "
        "Do not soften findings. If the code is clean, say so explicitly."
    ),
    "planner": (
        "You are a specialist planning sub-agent. Break the given goal into a concrete, "
        "ordered sequence of verifiable steps. Each step must be atomic (one action), "
        "testable (a clear done/not-done condition), and assigned to the right agent "
        "type (researcher/coder/auditor). Output a JSON array of steps: "
        '[{"step": 1, "agent": "coder", "action": "...", "done_when": "..."}]. '
        "Do not implement anything. Plan only."
    ),
}


class NormalizedMessage(BaseModel):
    session_id: str
    role: Literal["user", "assistant", "system"] = "user"
    content: str


class AgentEngine:
    """ReAct agent backed by HybridMemory (context retrieval) and ToolManager (MCP tool execution).

    Process flow for each call to ``process_task``:
    1. Pull semantic context from ChromaDB for the incoming message.
    2. Snapshot the tool list from all connected MCP servers.
    3. Enter a ReAct loop (max ``MAX_REACT_ITERATIONS`` turns):
       a. Call the LLM with the current message history and available tools.
       b. If the LLM responds with ``finish_reason == "tool_calls"``, execute
          every requested tool via the appropriate MCP server, record an
          ``ExecutionStep`` for each call and its result, and append to history.
       c. If the LLM responds with plain text, break the loop.
    4. If the cap is reached without a text response, do one final tool-free
       completion to force a synthesised answer.
    5. Package the full message history and tool log into an
       ``ExecutionTrajectory`` and fire-and-forget it to ``SkillDistiller``
       so background evaluation never delays the caller.
    """

    def __init__(
        self,
        memory: HybridMemory,
        tools: ToolManager,
        model: str = "gpt-4o-mini",
        distiller: SkillDistiller | None = None,
        checkpointer: StateCheckpointer | None = None,
        system_directive: str | None = None,
        fast_model: str | None = None,
        strong_model: str | None = None,
        require_task_contract: bool = True,
        profile_store: UserProfileStore | None = None,
        scheduler: CronScheduler | None = None,
        skill_registry: SkillRegistry | None = None,
        persona_content: str = "",
        session_store: Any = None,
        approval_gate: Any = None,
        evolution_engine: EvolutionEngine | None = None,
    ) -> None:
        self._memory = memory
        self._tools = tools
        self._model = model
        self._fast_model = fast_model or model
        self._strong_model = strong_model or model
        self._caching_enabled = _model_supports_caching(
            self._fast_model
        ) and _model_supports_caching(self._strong_model)
        self._distiller = distiller
        self._checkpointer = checkpointer
        base_directive = system_directive if system_directive is not None else SYSTEM_DIRECTIVE
        # Prepend persona content (SOUL.md + AGENTS.md + TOOLS.md) when available.
        self._system_directive = (
            f"{persona_content}\n\n{base_directive}" if persona_content else base_directive
        )
        self._require_task_contract = require_task_contract
        self._profile_store = profile_store
        self._scheduler = scheduler
        self._skill_registry = skill_registry
        self._session_store = session_store
        self._approval_gate = approval_gate
        self._evolution_engine = evolution_engine
        self._task_graph = TaskGraphEngine()
        self._histories: dict[str, list[dict[str, Any]]] = {}
        self._session_steps: dict[str, list[ExecutionStep]] = {}

    def update_models(
        self,
        model: str | None = None,
        fast_model: str | None = None,
        strong_model: str | None = None,
    ) -> None:
        """Hot-swap model tiers without restarting the engine."""
        if model:
            self._model = model
        if fast_model:
            self._fast_model = fast_model
        if strong_model:
            self._strong_model = strong_model
        self._caching_enabled = _model_supports_caching(
            self._fast_model
        ) and _model_supports_caching(self._strong_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_task(self, message: NormalizedMessage) -> str:
        final_text = ""
        async for event in self.stream_task(message):
            if event["type"] in {"text", "final_answer"}:
                final_text = event["content"]
        return final_text

    async def stream_task(
        self, message: NormalizedMessage
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Yield a stream of JSON-serialisable payloads as the agent works.

        Payload types:
          {"type": "status",    "message": str}
          {"type": "tool_call", "tool": str, "params": dict}
          {"type": "text",      "content": str}
        """
        yield {"type": "status", "message": "Thinking..."}

        all_tools = await self._tools.list_all_tools()
        tool_index: dict[str, str] = {t["name"]: t["server"] for t in all_tools}

        messages = self._touch_history(message.session_id)
        if messages is None:
            # First turn for this session: seed the durable system prompts and the
            # one-time bootstrap, then keep the list so every later turn inherits it.
            context = await self._fetch_context(message.content)
            messages = [
                self._directive_system_message(),
                {"role": "system", "content": _build_system_prompt(context)},
                self._build_host_environment_message(),
            ]
            # Inject user profile if available — gives the agent personalization context.
            user_ctx = self._build_user_profile_message()
            if user_ctx:
                messages.append(user_ctx)
            if _BOOTSTRAP_SESSIONS:
                messages.extend(await self._bootstrap_session(tool_index, all_tools))
            self._histories[message.session_id] = messages
            self._evict_histories()

        # Carry the whole prior conversation forward by appending to the stored
        # list; the ReAct loop mutates this same list in place, so the assistant
        # and tool turns it produces persist into the next turn automatically.
        messages.append({"role": message.role, "content": message.content})
        self._session_steps[message.session_id] = []
        self._record_turn(message.session_id, "user", message.content)

        final_text = ""
        async for event in self._stream_react_loop(
            message.session_id, message.content, messages, all_tools
        ):
            if event.get("type") in ("text", "final_answer"):
                final_text = str(event.get("content") or final_text)
            yield event
        if final_text:
            self._record_turn(message.session_id, "assistant", final_text)

    async def replay_from_checkpoint(
        self,
        checkpoint_id: str,
        user_correction: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Resume the ReAct loop from a previously saved checkpoint.

        Loads the exact messages array persisted at *checkpoint_id*.  If
        *user_correction* is supplied it is appended as a ``system`` message
        before the loop resumes, steering the model away from whatever mistake
        it made in the original run.

        Raises ``RuntimeError`` if no ``StateCheckpointer`` is configured, and
        propagates ``KeyError`` from the checkpointer if the ID is unknown.
        """
        if self._checkpointer is None:
            raise RuntimeError(
                "replay_from_checkpoint requires a StateCheckpointer; "
                "pass one to AgentEngine.__init__ via the checkpointer= argument."
            )

        payload = await self._checkpointer.load_checkpoint(checkpoint_id)
        session_id: str = payload.get("session_id", checkpoint_id)
        messages: list[dict[str, Any]] = list(payload["messages"])

        if user_correction is not None:
            messages.append({"role": "system", "content": user_correction})

        all_tools = await self._tools.list_all_tools()

        # Recover the original user prompt for distiller metadata
        original_prompt = next(
            (m["content"] for m in messages if m.get("role") == "user"),
            "",
        )

        logger.debug(
            "Replaying checkpoint %s for session %s (%d messages, correction=%s)",
            checkpoint_id,
            session_id,
            len(messages),
            user_correction is not None,
        )
        yield {"type": "status", "message": f"Replaying from checkpoint {checkpoint_id}..."}

        async for event in self._stream_react_loop(
            session_id, original_prompt, messages, all_tools
        ):
            yield event

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _stream_react_loop(
        self,
        session_id: str,
        original_prompt: str,
        messages: list[dict[str, Any]],
        all_tools: list[dict[str, Any]],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Global crash-catching wrapper around the core ReAct loop.

        Guarantees a terminal ``final_answer`` event is always emitted -- even on
        an otherwise-unhandled backend failure -- so the WebSocket receives a
        completion status and the frontend unlocks its input field rather than
        hanging on a half-finished stream.
        """
        try:
            async for event in self._drive_react_loop(
                session_id, original_prompt, messages, all_tools
            ):
                yield event
        except Exception as exc:
            if _is_rate_limit_error(exc):
                logger.warning(
                    "Provider rate limit reached for session %s: %s",
                    session_id,
                    exc,
                )
                yield _make_final_answer("rate_limited", _rate_limit_user_message())
                return
            logger.exception(
                "CRITICAL: unhandled failure in ReAct loop for session %s", session_id
            )
            yield _make_final_answer(
                "critical_failure",
                f"CRITICAL SYSTEM FAILURE: {type(exc).__name__}: {exc}",
            )

    async def _drive_react_loop(
        self,
        session_id: str,
        original_prompt: str,
        messages: list[dict[str, Any]],
        all_tools: list[dict[str, Any]],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Auto-continuing driver around :meth:`_run_react_batch`.

        Runs ReAct batches back-to-back while the task is still making progress,
        so a long task finishes unattended instead of stopping at the iteration
        cap and asking the user to type "continue". A batch that ends terminally
        (final answer, exception, rate limit, or a non-action summary) ends the
        whole run. A batch that hits the cap while an action task is still
        *incomplete* signals back here: if it made progress and the
        auto-continue budget remains, another batch is launched; otherwise the
        terminal "paused" message is surfaced.
        """
        # One cumulative step log across all batches so completion evidence and
        # duplicate-command detection survive auto-continuation.
        steps = self._session_steps.setdefault(session_id, [])
        batch = 0
        while True:
            batch += 1
            incomplete: dict[str, Any] | None = None
            async for event in self._run_react_batch(
                session_id, original_prompt, messages, all_tools, steps
            ):
                if isinstance(event, dict) and event.get("type") == "_batch_incomplete":
                    incomplete = event
                    break
                yield event

            if incomplete is None:
                return  # batch ended terminally

            if (
                _AUTO_CONTINUE_ENABLED
                and incomplete.get("progressing")
                and batch < _MAX_AUTO_CONTINUE_BATCHES
            ):
                logger.info(
                    "Auto-continuing session %s (batch %d/%d): task progressing but incomplete",
                    session_id,
                    batch,
                    _MAX_AUTO_CONTINUE_BATCHES,
                )
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "You have NOT finished the task yet and you are still making "
                            "progress. Continue automatically: do the next not-done step "
                            "from your plan / Completed_Actions now. Do not stop to ask the "
                            "user to continue."
                        ),
                    }
                )
                yield {"type": "status", "message": "Still working autonomously..."}
                continue

            # Auto-continue exhausted or no progress: surface the paused message.
            final_response = incomplete["final_response"]
            messages.append({"role": "assistant", "content": final_response})
            yield _make_final_answer("iteration_limit", final_response)
            await _store_iteration_cap_memory(
                self._memory,
                session_id,
                original_prompt,
                incomplete.get("tools_attempted", []),
                final_response,
                MAX_REACT_ITERATIONS,
            )
            return

    def _completion_status_with_task_graph(
        self,
        contract: dict[str, Any] | None,
        messages: list[dict[str, Any]],
        steps: list[ExecutionStep],
        *,
        contract_required: bool,
    ) -> dict[str, Any]:
        status = contract_completion_status(
            contract,
            messages,
            steps,
            contract_required=contract_required,
        )
        if contract is None or contract.get("mode") != "execute":
            return status

        graph_status = self._task_graph.completion_status(messages, steps)
        status["task_graph"] = graph_status
        if graph_status.get("source") != "explicit":
            return status

        # An explicit task graph replaces the old requirement for a loose
        # update_plan checklist. Contract evidence still gates completion.
        missing = [
            item
            for item in status.get("missing", [])
            if item not in {"plan", "plan_open_steps"}
        ]
        if not graph_status.get("complete"):
            for item in graph_status.get("missing", []):
                if item not in missing:
                    missing.append(item)
        return {
            **status,
            "complete": not missing,
            "missing": missing,
            "plan_open": not bool(graph_status.get("complete")),
        }

    def _tool_names_for_status_with_task_graph(
        self,
        contract: dict[str, Any],
        status: dict[str, Any],
        messages: list[dict[str, Any]],
        steps: list[ExecutionStep],
    ) -> set[str]:
        graph_status = status.get("task_graph") or {}
        if graph_status.get("source") == "explicit" and not graph_status.get("complete"):
            return self._task_graph.allowed_tools_for_next(messages, steps)

        allowed = tool_names_for_contract_status(contract, status)
        missing = set(status.get("missing") or [])
        if "plan" in missing:
            allowed.update({SET_TASK_GRAPH_TOOL_NAME, INSPECT_TASK_GRAPH_TOOL_NAME})
        if "task_graph" in missing:
            allowed.update(GRAPH_CONTROL_TOOLS)
        return allowed

    def _task_graph_done_count(self, messages: list[dict[str, Any]]) -> int:
        graph = self._task_graph.latest_graph(messages)
        if graph is None:
            return 0
        return sum(1 for node in graph["nodes"] if node.get("status") == "done")

    async def _run_react_batch(
        self,
        session_id: str,
        original_prompt: str,
        messages: list[dict[str, Any]],
        all_tools: list[dict[str, Any]],
        steps: list[ExecutionStep],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """One bounded batch of the core ReAct loop (shared by stream + replay).

        Mutates *messages* and *steps* in-place as tool calls and results
        accumulate, so checkpoints capture full conversation state and
        completion-evidence / duplicate-command detection persist across
        auto-continued batches. When an action task is still incomplete at the
        cap, yields a single internal ``_batch_incomplete`` event (consumed by
        :meth:`_drive_react_loop`, never sent to the client) instead of a
        terminal message, so the driver can decide whether to auto-continue.
        """
        # Snapshots taken at batch entry so "did THIS batch make progress?" is
        # measurable even though *steps* accumulates across batches.
        batch_start_done_steps = _count_done_plan_steps(messages)
        batch_start_done_graph_nodes = self._task_graph_done_count(messages)
        batch_start_successes = _count_successful_side_effects(steps)
        tool_schemas = _to_litellm_tools(all_tools)
        # Cache the (large, stable) tool block. Anthropic caches every tool up to
        # and including the one carrying cache_control, so marking the last covers all.
        if tool_schemas and self._caching_enabled:
            tool_schemas[-1]["cache_control"] = {"type": "ephemeral"}
        tool_index: dict[str, str] = {t["name"]: t["server"] for t in all_tools}
        contract_tool_available = any(t["name"] == TASK_CONTRACT_TOOL for t in all_tools)
        contract_required = self._require_task_contract and contract_tool_available

        completion_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": _LLM_MAX_TOKENS,
        }
        if tool_schemas:
            completion_kwargs["tools"] = tool_schemas

        final_response = ""
        hit_cap = False
        # Routine work runs on the fast tier; escalate to strong on repeated failure.
        active_model = self._fast_model
        consecutive_error_iters = 0
        forced_recovery_tool: str | None = None
        contract_text_rejections = 0
        available_tool_names = {tool["name"] for tool in all_tools}
        # Contract negotiation costs one model turn up front. Use the larger cap
        # whenever contracts are enabled so execute-mode tasks do not stop one
        # step short, while answer-mode tasks still terminate early.
        iteration_cap = (
            _ACTION_MAX_REACT_ITERATIONS if contract_required else MAX_REACT_ITERATIONS
        )

        for iteration in range(iteration_cap):
            try:
                if iteration > 0:
                    yield {"type": "status", "message": "Thinking..."}

                contract = latest_task_contract(messages)
                must_set_contract = contract_required and contract is None
                completion_status = self._completion_status_with_task_graph(
                    contract,
                    messages,
                    steps,
                    contract_required=contract_required,
                )
                needs_execution = (
                    contract is not None
                    and contract.get("mode") == "execute"
                    and not completion_status["complete"]
                )
                completion_kwargs["model"] = active_model
                request_messages = self._compact_context_window(messages)

                # Apply toolset filtering: narrow the tool list to only what is
                # relevant for the declared task category.  Falls back to "all"
                # when no toolset is declared or when the contract forces a specific
                # sub-set (contract filtering takes priority in that branch).
                active_toolset = (contract or {}).get("toolset", "all")
                toolset_filtered = _to_litellm_tools(
                    filter_tools_by_toolset(all_tools, active_toolset)
                )
                if toolset_filtered and self._caching_enabled:
                    toolset_filtered[-1]["cache_control"] = {"type": "ephemeral"}
                request_tool_schemas = toolset_filtered
                if must_set_contract:
                    request_messages = [
                        *request_messages,
                        {"role": "system", "content": build_task_contract_instruction()},
                    ]
                    request_tool_schemas = filter_tool_schemas(
                        tool_schemas, {TASK_CONTRACT_TOOL}
                    )
                elif needs_execution:
                    assert contract is not None  # needs_execution implies contract is set
                    allowed_tool_names = self._tool_names_for_status_with_task_graph(
                        contract, completion_status, messages, steps
                    )
                    if forced_recovery_tool in allowed_tool_names:
                        allowed_tool_names = {str(forced_recovery_tool)}
                        recovery_note = (
                            "\nRECOVERY MODE: The previous assistant turn emitted "
                            "plain text while the task contract was still incomplete. "
                            f"The only available tool now is {forced_recovery_tool}; "
                            "call it with the arguments needed to produce or verify "
                            "the missing evidence. Do not answer in prose."
                        )
                    else:
                        forced_recovery_tool = None
                        recovery_note = ""
                        if _recent_tool_failures(steps):
                            allowed_tool_names.update(
                                name
                                for name in _SELF_REPAIR_TOOLS
                                if name in available_tool_names
                            )
                            recovery_note = "\n" + _build_self_repair_instruction(steps)
                    request_messages = [
                        *request_messages,
                        {
                            "role": "system",
                            "content": _build_contract_execution_instruction(
                                contract, completion_status, messages, steps
                            )
                            + "\n\n"
                            + self._task_graph.build_instruction(messages, steps)
                            + recovery_note,
                        },
                    ]
                    request_tool_schemas = filter_tool_schemas(
                        tool_schemas,
                        allowed_tool_names,
                    )
                request_messages = _prepare_llm_request_messages(
                    request_messages, original_prompt
                )
                completion_kwargs["messages"] = request_messages
                if request_tool_schemas:
                    completion_kwargs["tools"] = request_tool_schemas
                else:
                    completion_kwargs.pop("tools", None)
                requires_tool_call = bool(
                    (must_set_contract or needs_execution) and request_tool_schemas
                )
                # Moonshot/Kimi does not natively support OpenAI's
                # tool_choice="required". LiteLLM approximates it with a generic
                # final user message; make that provider shim explicit and exact
                # so contract-gated tasks do not drift through slow prose retries.
                if requires_tool_call and _uses_moonshot_provider(active_model):
                    request_messages = [
                        *request_messages,
                        {
                            "role": "user",
                            "content": _moonshot_required_tool_instruction(
                                request_tool_schemas
                            ),
                        },
                    ]
                    completion_kwargs["messages"] = request_messages
                    completion_kwargs.pop("tool_choice", None)
                elif requires_tool_call:
                    completion_kwargs["tool_choice"] = "required"
                else:
                    completion_kwargs.pop("tool_choice", None)

                # Disable parallel tool calls during contract-enforced iterations.
                # When the contract gate narrows tools to a single option (e.g.
                # only update_plan is allowed), the model sometimes emits two calls
                # to the same tool in one turn â€” the second call overwrites the
                # first (collapsing a 3-step plan to 1 step) before any result is
                # seen. Forcing sequential calls (one per turn) eliminates this.
                if must_set_contract or needs_execution:
                    completion_kwargs["parallel_tool_calls"] = False
                else:
                    completion_kwargs.pop("parallel_tool_calls", None)

                # Stream answer-mode text immediately. For execute-mode text,
                # buffer until evidence checks prove the final answer is valid;
                # rejected prose never reaches the UI.
                chunks: list[Any] = []
                buffered_tokens: list[str] = []
                emitted_tokens = False
                is_text_response: bool | None = None
                completion_result = await _acompletion_stream_with_retry(
                    **completion_kwargs
                )
                if _is_async_iterable(completion_result):
                    async for chunk in completion_result:
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta:
                                if getattr(delta, "tool_calls", None) and is_text_response is None:
                                    is_text_response = False
                                elif delta.content and is_text_response is None:
                                    is_text_response = True
                                if is_text_response and delta.content:
                                    if can_stream_text_before_final(contract, messages, steps):
                                        emitted_tokens = True
                                        yield {"type": "token", "content": delta.content}
                                    else:
                                        buffered_tokens.append(delta.content)
                        chunks.append(chunk)
                    response = litellm.stream_chunk_builder(
                        chunks, messages=request_messages
                    )
                else:
                    response = completion_result
                choice = _llm_first_choice(response)
                assistant_msg = choice.message

                if choice.finish_reason != "tool_calls" or not assistant_msg.tool_calls:
                    final_response = assistant_msg.content or ""
                    final_status = self._completion_status_with_task_graph(
                        latest_task_contract(messages),
                        messages,
                        steps,
                        contract_required=contract_required,
                    )
                    if not final_status["complete"]:
                        contract_text_rejections += 1
                        forced_recovery_tool = _preferred_recovery_tool_name(
                            latest_task_contract(messages),
                            final_status,
                            steps,
                            original_prompt,
                            available_tool_names,
                        )

                        if (
                            active_model != self._strong_model
                            and contract_text_rejections >= 1
                        ):
                            active_model = self._strong_model
                            logger.info(
                                "Escalating to strong model %s for session %s after rejected contract text",
                                active_model,
                                session_id,
                            )
                            yield {
                                "type": "status",
                                "message": f"Escalating to {active_model}...",
                            }

                        instruction = _build_contract_continuation_instruction(
                            latest_task_contract(messages),
                            final_status,
                            final_response,
                            messages,
                            steps,
                        )
                        if forced_recovery_tool:
                            instruction += (
                                "\n\nRECOVERY MODE: The next iteration will expose "
                                f"only `{forced_recovery_tool}` because the task "
                                "needs concrete evidence, not another progress note. "
                                "Call that tool with appropriate arguments."
                            )
                        if final_response.strip():
                            messages.append({"role": "assistant", "content": final_response})
                        messages.append({"role": "system", "content": instruction})
                        logger.info(
                            "Rejected final text for session %s "
                            "(contract_mode=%s, missing=%s, open_plan=%s, attempted_tools=%s): %r",
                            session_id,
                            (
                                (latest_task_contract(messages) or {}).get("mode")
                                if latest_task_contract(messages)
                                else "missing"
                            ),
                            final_status["missing"],
                            final_status["plan_open"],
                            attempted_tool_names(steps),
                            final_response[:120],
                        )
                        yield {
                            "type": "status",
                            "message": "Final answer withheld until required evidence is verified...",
                        }
                        continue
                    if not final_response.strip():
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "The previous assistant turn was empty. Provide a "
                                    "non-empty final answer now, or continue with tool "
                                    "calls if more work is still required."
                                ),
                            }
                        )
                        logger.info(
                            "Rejected empty final text for session %s after contract completion",
                            session_id,
                        )
                        continue
                    final_response = _ensure_evidence_in_final_response(
                        final_response,
                        contract=latest_task_contract(messages),
                        original_prompt=original_prompt,
                        steps=steps,
                        messages=messages,
                    )
                    if buffered_tokens and not emitted_tokens:
                        yield {"type": "token", "content": final_response}
                    # Record the reply so the next turn in this session sees it.
                    messages.append({"role": "assistant", "content": final_response})
                    yield {"type": "text", "content": final_response}
                    break

                tool_calls = assistant_msg.tool_calls
                steps.append(
                    ExecutionStep(
                        kind="llm_tool_decision",
                        content=f"Iteration {iteration + 1}: LLM requested {len(tool_calls)} tool call(s)",
                        metadata={
                            "iteration": iteration + 1,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "name": tc.function.name,
                                    "arguments": json.loads(tc.function.arguments),
                                }
                                for tc in tool_calls
                            ],
                        },
                    )
                )

                messages.append(_serialise_assistant_msg(assistant_msg))

                # Announce every call up front so the UI ordering is preserved.
                calls: list[tuple[str, str, dict[str, Any]]] = []
                for tc in tool_calls:
                    args = json.loads(tc.function.arguments)
                    yield {"type": "tool_call", "tool": tc.function.name, "params": args}
                    calls.append((str(tc.id), str(tc.function.name), args))

                iter_error_count, tool_result_events = await self._dispatch_tool_calls(
                    calls=calls,
                    messages=messages,
                    steps=steps,
                    tool_index=tool_index,
                    session_id=session_id,
                )
                for result_event in tool_result_events:
                    yield result_event
                if iter_error_count:
                    messages.append(
                        {
                            "role": "system",
                            "content": _build_self_repair_instruction(steps),
                        }
                    )
                forced_recovery_tool = None
                contract_text_rejections = 0

                # Escalate to the strong model when the fast one keeps failing.
                if tool_calls and iter_error_count == len(tool_calls):
                    consecutive_error_iters += 1
                else:
                    consecutive_error_iters = 0
                if (
                    active_model != self._strong_model
                    and consecutive_error_iters >= _ESCALATE_AFTER_CONSECUTIVE_ERROR_ITERS
                ):
                    active_model = self._strong_model
                    consecutive_error_iters = 0
                    logger.info(
                        "Escalating to strong model %s for session %s after repeated errors",
                        active_model, session_id,
                    )
                    yield {"type": "status", "message": f"Escalating to {active_model}â€¦"}

                # Checkpoint on the first iteration and then every Nth, rather than
                # every iteration, to cut redundant full-history writes. Replay
                # resumes from the most recent checkpoint (at most _CHECKPOINT_EVERY-1
                # iterations behind).
                if self._checkpointer is not None and (
                    iteration == 0 or (iteration + 1) % _CHECKPOINT_EVERY == 0
                ):
                    try:
                        await self._checkpointer.save_checkpoint(
                            session_id=session_id,
                            step_number=iteration,
                            state_payload={"session_id": session_id, "messages": messages},
                        )
                    except Exception as exc:
                        logger.warning("Checkpoint save failed at iteration %d: %s", iteration, exc)

                logger.debug(
                    "ReAct iteration %d/%d complete", iteration + 1, iteration_cap
                )

            except Exception as exc:
                if _is_rate_limit_error(exc):
                    logger.warning(
                        "Provider rate limit reached at iteration %d for session %s: %s",
                        iteration + 1,
                        session_id,
                        exc,
                    )
                    yield _make_final_answer(
                        "rate_limited",
                        _rate_limit_user_message(),
                    )
                    return
                hit_cap = True
                logger.exception(
                    "Unhandled exception at iteration %d for session %s: %s",
                    iteration + 1, session_id, exc,
                )
                tools_attempted = sorted({
                    step.metadata["tool_name"]
                    for step in steps
                    if step.kind == "tool_result"
                })
                yield _make_final_answer(
                    "exception",
                    f"Task failed at iteration {iteration + 1} due to an unhandled error.\n"
                    f"{type(exc).__name__}: {exc}\n"
                    f"Tools attempted so far: {', '.join(tools_attempted) or 'none'}",
                )
                return
        else:
            hit_cap = True
            logger.warning(
                "ReAct loop hit %d-iteration cap for session %s; requesting progress summary",
                iteration_cap,
                session_id,
            )
            # Transient instruction -- kept out of the persisted history so it does
            # not leak into the next turn; only the summary reply is recorded.
            cap_status = self._completion_status_with_task_graph(
                latest_task_contract(messages),
                messages,
                steps,
                contract_required=contract_required,
            )
            if not cap_status["complete"]:
                # Don't terminate here: hand control back to the auto-continuing
                # driver with whether this batch made progress. The driver either
                # launches another batch or surfaces this paused message.
                final_response = build_incomplete_contract_cap_message(
                    original_prompt, cap_status, steps
                )
                tools_attempted = sorted({
                    step.metadata["tool_name"]
                    for step in steps
                    if step.kind == "tool_result"
                })
                progressing = (
                    _count_successful_side_effects(steps) > batch_start_successes
                    or _count_done_plan_steps(messages) > batch_start_done_steps
                    or self._task_graph_done_count(messages) > batch_start_done_graph_nodes
                )
                yield {
                    "type": "_batch_incomplete",
                    "progressing": progressing,
                    "final_response": final_response,
                    "tools_attempted": tools_attempted,
                }
                return

            cap_instruction = {
                "role": "system",
                "content": (
                    f"SYSTEM: Hard iteration limit ({iteration_cap}) reached while "
                    f"working on: {original_prompt[:200]}. "
                    "Respond with exactly this structure -- do not deviate: "
                    "'Task paused. I hit the iteration limit while trying to <restate the "
                    "objective in one sentence>. Here is my progress: <bullet list of "
                    "completed steps, failed steps, and what still remains to be done>.'"
                ),
            }
            yield {"type": "status", "message": "Summarising progress..."}
            # tools= must still be sent: the history contains tool_use/tool_result
            # blocks and Anthropic rejects the request without it. tool_choice="none"
            # forces the model to answer in text rather than call another tool, which
            # keeps the original intent (a structured progress summary) intact.
            summary_kwargs: dict[str, Any] = {
                "model": active_model,
                "messages": _prepare_llm_request_messages(
                    self._compact_context_window([*messages, cap_instruction]),
                    original_prompt,
                ),
                "max_tokens": _LLM_FINAL_MAX_TOKENS,
            }
            if tool_schemas:
                summary_kwargs["tools"] = tool_schemas
                summary_kwargs["tool_choice"] = "none"
            summary_chunks: list[Any] = []
            summary_result = await _acompletion_stream_with_retry(**summary_kwargs)
            if _is_async_iterable(summary_result):
                async for chunk in summary_result:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            yield {"type": "token", "content": delta.content}
                    summary_chunks.append(chunk)
                final = litellm.stream_chunk_builder(
                    summary_chunks, messages=summary_kwargs["messages"]
                )
            else:
                final = summary_result
            final_response = _llm_first_choice(final).message.content or ""
            messages.append({"role": "assistant", "content": final_response})
            yield _make_final_answer("iteration_limit", final_response)

            # Awaited (not fire-and-forget) so the record is durable on cap-hit paths.
            tools_attempted = sorted({
                step.metadata["tool_name"]
                for step in steps
                if step.kind == "tool_result"
            })
            raw_text = (
                f"Task: {original_prompt}\n"
                f"Tools attempted: {', '.join(tools_attempted) or 'none'}\n"
                f"Outcome: hit {iteration_cap}-iteration cap\n"
                f"Final response: {final_response}"
            )
            entities = {
                "nodes": [{"label": "Tool", "name": name} for name in tools_attempted],
                "relationships": [],
            }
            if hasattr(self._memory, "store_event"):
                await asyncio.to_thread(
                    self._memory.store_event,
                    session_id,
                    raw_text,
                    entities,
                )

        if not hit_cap and hasattr(self._memory, "store_event"):
            tools_used = sorted({
                step.metadata["tool_name"]
                for step in steps
                if step.kind == "tool_result" and not step.metadata.get("is_error")
            })
            raw_text = (
                f"Task: {original_prompt}\n"
                f"Tools used: {', '.join(tools_used) or 'none'}\n"
                f"Outcome: completed successfully\n"
                f"Final response: {final_response}"
            )
            entities = {
                "nodes": [{"label": "Tool", "name": name} for name in tools_used],
                "relationships": [],
            }
            await asyncio.to_thread(
                self._memory.store_event,
                session_id,
                raw_text,
                entities,
            )

        final_contract_status = self._completion_status_with_task_graph(
            latest_task_contract(messages),
            messages,
            steps,
            contract_required=contract_required,
        )
        if self._distiller is not None and not (
            hit_cap and not final_contract_status["complete"]
        ):
            trajectory = ExecutionTrajectory(
                prompt=original_prompt,
                steps=steps,
                final_output=final_response,
                metadata={
                    "session_id": session_id,
                    "model": self._model,
                    "iterations": len([s for s in steps if s.kind == "llm_tool_decision"]),
                    "hit_iteration_cap": hit_cap,
                    "available_tools": [t["name"] for t in all_tools],
                    "task_graph": self._task_graph.inspect(messages, steps),
                },
            )
            if self._evolution_engine is not None:
                try:
                    trace_id = self._evolution_engine.record_trajectory(trajectory)
                    trajectory.metadata["evolution_trace_id"] = trace_id
                except Exception as exc:
                    logger.debug("evolution trace recording failed: %s", exc)
            asyncio.create_task(  # noqa: RUF006
                self._distiller.submit(trajectory),
                name=f"distill:{session_id}",
            )

        # Fire-and-forget: update user profile from this conversation.
        if self._profile_store is not None and not hit_cap and final_response:
            convo = f"User: {original_prompt}\nAssistant: {final_response}"
            asyncio.create_task(  # noqa: RUF006
                extract_and_update_user_profile(
                    self._profile_store, convo, self._fast_model
                ),
                name=f"profile:{session_id}",
            )

    async def _dispatch_tool_calls(
        self,
        *,
        calls: list[tuple[str, str, dict[str, Any]]],
        messages: list[dict[str, Any]],
        steps: list[ExecutionStep],
        tool_index: dict[str, str],
        session_id: str = "unknown",
    ) -> tuple[int, list[dict[str, Any]]]:
        """Execute one iteration's tool calls and record results.

        Read-only tools run concurrently via ``asyncio.gather``; side-effecting
        tools (terminal, background, MCP writes) run serially in call order so
        filesystem mutations and write ordering stay deterministic.

        Returns the number of calls that returned an error in this batch plus
        UI-safe tool-result events.
        """
        results: dict[str, tuple[str, bool, str | None]] = {}
        parallel = [c for c in calls if c[1] in _PARALLEL_SAFE_TOOLS]
        serial = [c for c in calls if c[1] not in _PARALLEL_SAFE_TOOLS]

        if parallel:
            # Pre-screen parallel (read-only) tools with the generic anti-spin
            # guard: any tool re-issued with identical arguments and no state
            # change in between cannot return new information, so it is blocked
            # instead of burning an API call. Covers port polling, file/process
            # re-checks, repeated web_fetch/web_search, get_system_environment, etc.
            parallel_to_run: list[tuple[str, str, dict[str, Any]]] = []
            for tc_id, name, args in parallel:
                graph_block = self._task_graph_tool_block(name, messages, steps)
                if graph_block is not None:
                    results[tc_id] = (graph_block, True, "__builtin__")
                    continue
                if (
                    name not in GENERIC_CAP_SKIP_TOOLS
                    and identical_tool_call_runs_since_state_change(steps, name, args)
                    >= MAX_IDENTICAL_TOOL_CALLS
                ):
                    results[tc_id] = (
                        repeated_tool_call_message(name, args),
                        True,
                        "__builtin__",
                    )
                    continue
                parallel_to_run.append((tc_id, name, args))

            gathered = await asyncio.gather(
                *(
                    self._execute_single_tool(name, args, messages, steps, tool_index, session_id)
                    for (_, name, args) in parallel_to_run
                )
            )
            for (tc_id, _, _), res in zip(parallel_to_run, gathered, strict=False):
                results[tc_id] = res

        # Track the most recent host command so an immediately-repeated,
        # identical one is short-circuited -- the real anti-spin guard
        # (e.g. the model issuing `mkdir -p x` twice).
        last_host_cmd = last_host_command(steps)
        # Count identical-command repeats accumulated *within this iteration* so a
        # command emitted several times in one turn also trips the cap; the
        # cross-iteration count is recovered from `steps` via the helper.
        local_repeat_counts: dict[str, int] = {}
        # Consecutive terminal-command run counter: cross-turn baseline + intra-turn
        # accumulator. Catches "varied spin" where commands differ but the agent is
        # still running terminal-only calls with no evidence/file-write in between.
        local_terminal_run_count = consecutive_terminal_run_length(steps)
        for tc_id, name, args in serial:
            cmd = normalize_command(args.get("command")) if name in HOST_EXECUTION_TOOLS else ""
            repeat_runs = (
                host_command_runs_since_state_change(steps, cmd)
                + local_repeat_counts.get(cmd, 0)
                if cmd
                else 0
            )
            graph_block = self._task_graph_tool_block(name, messages, steps)
            if graph_block is not None:
                results[tc_id] = (graph_block, True, "__builtin__")
            elif (
                not self._task_graph_explicitly_allows(name, messages, steps)
                and should_block_tool_for_action_task(
                    latest_task_contract(messages), messages, steps, name
                )
            ):
                results[tc_id] = (blocked_action_tool_message(name), True, "__builtin__")
            elif (
                name == "execute_terminal_command"
                and terminal_failure_since_diagnostic(steps)
            ):
                results[tc_id] = (
                    terminal_failure_recovery_message(args.get("command")),
                    True,
                    "__builtin__",
                )
            elif (
                name == "execute_terminal_command"
                and local_terminal_run_count >= MAX_CONSECUTIVE_TERMINAL_COMMANDS
            ):
                # Varied-spin guard: too many terminal commands in a row with no
                # other tool type (evidence check, file write, etc.) in between.
                # Unlike the identical-command cap, this triggers even when every
                # command is different -- the problem is the *lack of inspection*,
                # not the repetition of one specific command.
                results[tc_id] = (
                    consecutive_terminal_cap_message(local_terminal_run_count),
                    True,
                    "__builtin__",
                )
            elif cmd and repeat_runs >= MAX_IDENTICAL_COMMAND_RUNS:
                # Identical command re-run too many times with no state change in
                # between -- a spin. Hard-block it (is_error=True) so the loop also
                # escalates the model tier instead of burning the whole budget.
                local_repeat_counts[cmd] = local_repeat_counts.get(cmd, 0) + 1
                results[tc_id] = (repeated_command_message(name, cmd), True, "__builtin__")
            elif cmd and cmd == last_host_cmd:
                results[tc_id] = (duplicate_command_message(name), False, "__builtin__")
            elif (
                not cmd
                and name not in GENERIC_CAP_SKIP_TOOLS
                and identical_tool_call_runs_since_state_change(steps, name, args)
                >= MAX_IDENTICAL_TOOL_CALLS
            ):
                # Generic anti-spin guard for non-host serial tools (e.g. MCP
                # reads): identical call, no state change, no new information.
                results[tc_id] = (repeated_tool_call_message(name, args), True, "__builtin__")
            else:
                results[tc_id] = await self._execute_single_tool(
                    name, args, messages, steps, tool_index, session_id
                )
                if cmd:
                    last_host_cmd = cmd
                    local_repeat_counts[cmd] = local_repeat_counts.get(cmd, 0) + 1
                # A successful execute_terminal_command extends the run; reset only
                # when a non-terminal tool runs (handled by consecutive_terminal_run_length).
                if name == "execute_terminal_command":
                    local_terminal_run_count += 1

        # Record steps and tool messages in the original call order.
        iter_error_count = 0
        tool_result_events: list[dict[str, Any]] = []
        for tc_id, tool_name, arguments in calls:
            content, is_error, server = results[tc_id]
            if is_error:
                iter_error_count += 1
            steps.append(
                ExecutionStep(
                    kind="tool_result",
                    content=content,
                    metadata={
                        "tool_name": tool_name,
                        "tool_call_id": tc_id,
                        "server": server,
                        "arguments": arguments,
                        "is_error": is_error,
                    },
                )
            )
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": content})
            tool_result_events.append(
                _make_tool_result_event(
                    tool_name=tool_name,
                    tool_call_id=tc_id,
                    content=content,
                    is_error=is_error,
                )
            )

        return iter_error_count, tool_result_events

    async def _execute_single_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        messages: list[dict[str, Any]],
        steps: list[ExecutionStep],
        tool_index: dict[str, str],
        session_id: str = "unknown",
    ) -> tuple[str, bool, str | None]:
        """Execute one tool call and return ``(content, is_error, server)``.

        Pure with respect to *messages* -- it never appends; the caller records the
        result. This lets read-only calls be gathered concurrently.
        """
        if tool_name == "delegate_task":
            try:
                return await self._execute_delegate_task(arguments), False, "__builtin__"
            except Exception as exc:
                logger.warning("delegate_task raised: %s", exc)
                return f"[delegate_task error] {exc}", True, "__builtin__"
        if tool_name == "execute_terminal_command":
            denial = await self._check_approval(arguments.get("command", ""), session_id)
            if denial is not None:
                return json.dumps({"status": "denied", "reason": denial}), True, "__builtin__"
            try:
                result = await self._tools.execute_terminal_command(arguments["command"])
                return json.dumps(result), int(result.get("exit_code", -1)) != 0, "__builtin__"
            except Exception as exc:
                logger.warning(
                    "execute_terminal_command raised: %s: %r",
                    type(exc).__name__,
                    exc,
                )
                return (
                    f"[execute_terminal_command error] {type(exc).__name__}: {exc}",
                    True,
                    "__builtin__",
                )
        if tool_name == "execute_background_service":
            denial = await self._check_approval(arguments.get("command", ""), session_id)
            if denial is not None:
                return json.dumps({"status": "denied", "reason": denial}), True, "__builtin__"
            try:
                result = self._tools.execute_background_service(arguments["command"])
                return json.dumps(result), result.get("status") == "error", "__builtin__"
            except Exception as exc:
                logger.warning("execute_background_service raised: %s", exc)
                return f"[execute_background_service error] {exc}", True, "__builtin__"
        if tool_name == "get_system_environment":
            return self._tools.get_system_environment(), False, "__builtin__"
        if tool_name == "web_fetch":
            try:
                content = await self._tools.web_fetch(**arguments)
                return content, _json_tool_result_has_error(content), "__builtin__"
            except Exception as exc:
                logger.warning("web_fetch raised: %s", exc)
                return f"[web_fetch error] {exc}", True, "__builtin__"
        if tool_name == "web_search":
            try:
                return await self._tools.web_search(**arguments), False, "__builtin__"
            except Exception as exc:
                logger.warning("web_search raised: %s", exc)
                return f"[web_search error] {exc}", True, "__builtin__"
        # Browser tools â€” stateful, run serially (not in _PARALLEL_SAFE_TOOLS)
        if tool_name == "browser_navigate":
            try:
                return await self._tools.browser_navigate(**arguments), False, "__builtin__"
            except Exception as exc:
                logger.warning("browser_navigate raised: %s", exc)
                return f"[browser_navigate error] {exc}", True, "__builtin__"
        if tool_name == "browser_get_text":
            try:
                return await self._tools.browser_get_text(**arguments), False, "__builtin__"
            except Exception as exc:
                logger.warning("browser_get_text raised: %s", exc)
                return f"[browser_get_text error] {exc}", True, "__builtin__"
        if tool_name == "browser_screenshot":
            try:
                return await self._tools.browser_screenshot(**arguments), False, "__builtin__"
            except Exception as exc:
                logger.warning("browser_screenshot raised: %s", exc)
                return f"[browser_screenshot error] {exc}", True, "__builtin__"
        if tool_name == "browser_click":
            try:
                return await self._tools.browser_click(**arguments), False, "__builtin__"
            except Exception as exc:
                logger.warning("browser_click raised: %s", exc)
                return f"[browser_click error] {exc}", True, "__builtin__"
        if tool_name == "browser_fill":
            try:
                return await self._tools.browser_fill(**arguments), False, "__builtin__"
            except Exception as exc:
                logger.warning("browser_fill raised: %s", exc)
                return f"[browser_fill error] {exc}", True, "__builtin__"
        if tool_name == "browser_evaluate":
            try:
                return await self._tools.browser_evaluate(**arguments), False, "__builtin__"
            except Exception as exc:
                logger.warning("browser_evaluate raised: %s", exc)
                return f"[browser_evaluate error] {exc}", True, "__builtin__"
        if tool_name == "get_filesystem_process_evidence":
            try:
                content = self._tools.get_filesystem_process_evidence(**arguments)
                return (
                    content,
                    _filesystem_process_evidence_has_negative_findings(content),
                    "__builtin__",
                )
            except Exception as exc:
                logger.warning("get_filesystem_process_evidence raised: %s", exc)
                return f"[get_filesystem_process_evidence error] {exc}", True, "__builtin__"
        if tool_name == "wait_for_port":
            try:
                content = await self._tools.wait_for_port(**arguments)
                return (
                    content,
                    _json_tool_result_has_error(content, false_open_is_error=True),
                    "__builtin__",
                )
            except Exception as exc:
                logger.warning("wait_for_port raised: %s", exc)
                return f"[wait_for_port error] {exc}", True, "__builtin__"
        if tool_name == "write_text_file":
            try:
                return (
                    self._tools.write_text_file(**arguments),
                    False,
                    "__builtin__",
                )
            except Exception as exc:
                logger.warning("write_text_file raised: %s", exc)
                return f"[write_text_file error] {exc}", True, "__builtin__"
        if tool_name == "expose_local_http_service":
            try:
                return (
                    self._tools.expose_local_http_service(**arguments),
                    False,
                    "__builtin__",
                )
            except Exception as exc:
                logger.warning("expose_local_http_service raised: %s", exc)
                return f"[expose_local_http_service error] {exc}", True, "__builtin__"
        if tool_name == "expand_tool_output":
            content, is_error = _run_expand_tool_output(messages, arguments)
            return content, is_error, "__builtin__"
        if tool_name == "update_plan":
            content, is_error = _run_update_plan(arguments)
            return content, is_error, "__builtin__"
        if tool_name == SET_TASK_GRAPH_TOOL_NAME:
            content, is_error = self._task_graph.run_set_task_graph(arguments)
            return content, is_error, "__builtin__"
        if tool_name == INSPECT_TASK_GRAPH_TOOL_NAME:
            return json.dumps(self._task_graph.inspect(messages, steps), indent=2), False, "__builtin__"
        if tool_name == UPDATE_TASK_NODE_TOOL_NAME:
            content, is_error = self._task_graph.run_update_task_node(arguments, messages, steps)
            return content, is_error, "__builtin__"
        if tool_name == REPAIR_TASK_GRAPH_TOOL_NAME:
            content, is_error = self._task_graph.run_repair_task_graph(arguments, messages, steps)
            return content, is_error, "__builtin__"
        if tool_name == VERIFY_TASK_GRAPH_TOOL_NAME:
            return json.dumps(self._task_graph.verify(messages, steps), indent=2), False, "__builtin__"
        if tool_name == TASK_CONTRACT_TOOL:
            content, is_error = run_set_task_contract(arguments)
            return content, is_error, "__builtin__"

        if tool_name == "analyze_image":
            try:
                return await self._tools.analyze_image(**arguments), False, "__builtin__"
            except Exception as exc:
                return f"[analyze_image error] {exc}", True, "__builtin__"
        if tool_name == "generate_image":
            try:
                return await self._tools.generate_image(**arguments), False, "__builtin__"
            except Exception as exc:
                return f"[generate_image error] {exc}", True, "__builtin__"
        if tool_name == "schedule_task":
            try:
                output = await self._schedule_task(arguments, session_id)
                return output, False, "__builtin__"
            except Exception as exc:
                return f"[schedule_task error] {exc}", True, "__builtin__"
        if tool_name == "list_scheduled_tasks":
            try:
                return self._list_scheduled_tasks(), False, "__builtin__"
            except Exception as exc:
                return f"[list_scheduled_tasks error] {exc}", True, "__builtin__"
        if tool_name == "list_skills":
            try:
                return self._list_skills(), False, "__builtin__"
            except Exception as exc:
                return f"[list_skills error] {exc}", True, "__builtin__"
        if tool_name == "create_skill":
            try:
                return await self._create_skill(arguments), False, "__builtin__"
            except Exception as exc:
                return f"[create_skill error] {exc}", True, "__builtin__"
        if tool_name == "recall_memory":
            try:
                return await self._recall_memory(arguments), False, "__builtin__"
            except Exception as exc:
                return f"[recall_memory error] {exc}", True, "__builtin__"
        if tool_name == "list_evolution_candidates":
            try:
                return self._list_evolution_candidates(arguments), False, "__builtin__"
            except Exception as exc:
                return f"[list_evolution_candidates error] {exc}", True, "__builtin__"
        if tool_name == "inspect_evolution_candidate":
            try:
                return self._inspect_evolution_candidate(arguments), False, "__builtin__"
            except Exception as exc:
                return f"[inspect_evolution_candidate error] {exc}", True, "__builtin__"
        if tool_name == "run_evolution_cycle":
            try:
                output = self._run_evolution_cycle(arguments)
                try:
                    await self._tools.connect_skills_server()
                except Exception as exc:
                    logger.warning("run_evolution_cycle: skills reconnect failed: %s", exc)
                return output, False, "__builtin__"
            except Exception as exc:
                return f"[run_evolution_cycle error] {exc}", True, "__builtin__"
        if tool_name == "rollback_evolution_candidate":
            try:
                output = self._rollback_evolution_candidate(arguments)
                try:
                    await self._tools.connect_skills_server()
                except Exception as exc:
                    logger.warning("rollback_evolution_candidate: skills reconnect failed: %s", exc)
                return output, False, "__builtin__"
            except Exception as exc:
                return f"[rollback_evolution_candidate error] {exc}", True, "__builtin__"

        server = tool_index.get(tool_name)
        if server is None:
            return (
                f"[error] tool {tool_name!r} not found on any connected server",
                True,
                None,
            )
        mcp_result = await self._tools.call_tool(server, tool_name, arguments)
        text = _extract_tool_text(mcp_result)
        # Self-improvement loop: record real skill outcomes and evolve the skill
        # (evidence-gated improvement, or rollback of a regressed version).
        if server == "skills":
            self._observe_skill_use(
                tool_name, success=not mcp_result.isError, arguments=arguments, outcome=text
            )
        return text, mcp_result.isError, server

    def _build_user_profile_message(self) -> dict[str, Any] | None:
        """Return a system message with the user profile summary, or None if empty."""
        if self._profile_store is None:
            return None
        ctx = self._profile_store.as_context_string()
        if not ctx:
            return None
        return {
            "role": "system",
            "content": f"User context: {ctx}",
        }

    async def _schedule_task(
        self,
        arguments: dict[str, Any],
        session_id: str,
    ) -> str:
        if self._scheduler is None:
            return json.dumps({"error": "Scheduler is not configured on this engine."})
        # When scheduled from a messaging chat, deliver the result back to that
        # chat by default so recurring tasks proactively message the user.
        deliver_to = arguments.get("deliver_to", "")
        if not deliver_to and session_id.split(":", 1)[0] in ("tg", "discord", "slack"):
            deliver_to = session_id
        job = await self._scheduler.add_job(
            schedule_type=arguments["schedule_type"],
            schedule_spec=arguments["schedule_spec"],
            prompt=arguments["prompt"],
            session_id=session_id,
            label=arguments.get("label", ""),
            deliver_to=deliver_to,
        )
        return json.dumps({
            "job_id": job.job_id,
            "schedule_type": job.schedule_type,
            "schedule_spec": job.schedule_spec,
            "next_run": job.next_run.isoformat() if job.next_run else None,
            "label": job.label,
            "deliver_to": job.deliver_to,
        }, indent=2)

    def _list_scheduled_tasks(self) -> str:
        if self._scheduler is None:
            return json.dumps({"jobs": [], "note": "Scheduler not configured on this engine."})
        return json.dumps({"jobs": self._scheduler.list_jobs()}, indent=2)

    def _list_skills(self) -> str:
        if self._skill_registry is None:
            return json.dumps({"skills": [], "note": "Skill registry not configured on this engine."})
        return json.dumps({"skills": self._skill_registry.list_skills()}, indent=2)

    async def _check_approval(self, command: str, session_id: str) -> str | None:
        """Return a denial reason if a command needs and fails human approval.

        Returns ``None`` when approval is off, not required, or granted.
        """
        gate = self._approval_gate
        if gate is None or not gate.requires_approval(command):
            return None
        approved, reason = await gate.request(command, session_id)
        return None if approved else reason

    def _record_turn(self, session_id: str, role: str, content: str) -> None:
        """Append a turn to the cross-session memory store (fire-and-forget)."""
        if self._session_store is None or not content:
            return
        asyncio.create_task(  # noqa: RUF006
            asyncio.to_thread(self._session_store.add_turn, session_id, role, content)
        )

    async def _recall_memory(self, arguments: dict[str, Any]) -> str:
        """Full-text search across past conversations (the recall_memory tool)."""
        if self._session_store is None:
            return json.dumps({"results": [], "note": "Session memory is not configured."})
        query = str(arguments.get("query", ""))
        limit = int(arguments.get("limit", 8) or 8)
        results = await asyncio.to_thread(self._session_store.search, query, limit)
        return json.dumps({"query": query, "results": results}, indent=2)

    def _list_evolution_candidates(self, arguments: dict[str, Any]) -> str:
        if self._evolution_engine is None:
            return json.dumps({"candidates": [], "note": "Evolution engine not configured."})
        status = arguments.get("status")
        candidates = self._evolution_engine.list_candidates(str(status) if status else None)
        return json.dumps({"candidates": candidates}, indent=2)

    def _inspect_evolution_candidate(self, arguments: dict[str, Any]) -> str:
        if self._evolution_engine is None:
            return json.dumps({"error": "Evolution engine not configured."})
        candidate = self._evolution_engine.inspect_candidate(str(arguments["candidate_id"]))
        if candidate is None:
            return json.dumps({"error": "candidate not found"})
        return json.dumps(candidate, indent=2)

    def _run_evolution_cycle(self, arguments: dict[str, Any]) -> str:
        if self._evolution_engine is None:
            return json.dumps({"error": "Evolution engine not configured."})
        limit = int(arguments.get("limit", 5) or 5)
        return json.dumps(self._evolution_engine.run_cycle(limit=limit), indent=2)

    def _rollback_evolution_candidate(self, arguments: dict[str, Any]) -> str:
        if self._evolution_engine is None:
            return json.dumps({"error": "Evolution engine not configured."})
        return json.dumps(
            self._evolution_engine.rollback(str(arguments["candidate_id"])),
            indent=2,
        )

    def _task_graph_tool_block(
        self,
        tool_name: str,
        messages: list[dict[str, Any]],
        steps: list[ExecutionStep],
    ) -> str | None:
        contract = latest_task_contract(messages)
        if contract is None or contract.get("mode") != "execute":
            return None
        graph_status = self._task_graph.completion_status(messages, steps)
        if graph_status.get("source") != "explicit" or graph_status.get("complete"):
            return None
        allowed = self._task_graph.allowed_tools_for_next(messages, steps)
        if tool_name in allowed or tool_name == TASK_CONTRACT_TOOL:
            return None
        active = graph_status.get("active_node") or {}
        return (
            f"[task_graph blocked] Tool {tool_name!r} is not allowed for the current "
            f"task graph node {active.get('id', 'unknown')!r}. Work only the active "
            "ready node, use its allowed_tools, then call update_task_node with "
            "real evidence_refs."
        )

    def _task_graph_explicitly_allows(
        self,
        tool_name: str,
        messages: list[dict[str, Any]],
        steps: list[ExecutionStep],
    ) -> bool:
        contract = latest_task_contract(messages)
        if contract is None or contract.get("mode") != "execute":
            return False
        graph_status = self._task_graph.completion_status(messages, steps)
        return bool(
            graph_status.get("source") == "explicit"
            and not graph_status.get("complete")
            and tool_name in self._task_graph.allowed_tools_for_next(messages, steps)
        )

    async def _create_skill(self, arguments: dict[str, Any]) -> str:
        """Author a new skill on demand (the auto skill maker) and load it."""
        if self._skill_registry is None:
            return json.dumps({"error": "Skill registry is not configured on this engine."})
        if self._evolution_engine is not None:
            candidate = self._evolution_engine.stage_skill_candidate(
                name=arguments["name"],
                code=arguments["code"],
                reason="auto_skill_maker",
            )
            return json.dumps(
                {
                    "status": "staged",
                    "candidate_id": candidate["candidate_id"],
                    "message": (
                        "Skill candidate staged. It becomes callable only after "
                        "run_evolution_cycle promotes it with a proof bundle."
                    ),
                },
                indent=2,
            )
        path = self._skill_registry.create_skill(
            name=arguments["name"],
            code=arguments["code"],
            description=arguments.get("description", ""),
            tags=arguments.get("tags", []),
        )
        # Reconnect the skills MCP server so the new skill is immediately callable.
        try:
            await self._tools.connect_skills_server()
        except Exception as exc:
            logger.warning("create_skill: skills server reconnect failed: %s", exc)
        return json.dumps({"status": "created", "path": path}, indent=2)

    def _observe_skill_use(
        self, skill_name: str, *, success: bool, arguments: dict[str, Any], outcome: str
    ) -> None:
        """Record a skill invocation and kick off evidence-gated evolution."""
        if self._evolution_engine is not None:
            asyncio.create_task(  # noqa: RUF006
                self._evolution_engine.observe_skill_use(
                    skill_name,
                    success=success,
                    arguments=arguments,
                    outcome=outcome,
                )
            )
            return
        reg = self._skill_registry
        if reg is None:
            return
        try:
            reg.record_use(skill_name, success=success)
            reg.record_use_example(
                skill_name, task=json.dumps(arguments)[:200], outcome=outcome[:200]
            )
        except Exception as exc:
            logger.debug("record skill use failed for %s: %s", skill_name, exc)
            return
        # Fire-and-forget so the skill evolution never blocks the agent turn.
        asyncio.create_task(self._evolve_skill(skill_name))  # noqa: RUF006

    async def _evolve_skill(self, skill_name: str) -> None:
        if self._skill_registry is None:
            return
        try:
            action = await self._skill_registry.maybe_evolve(skill_name)
            if action != "unchanged":
                logger.info("Skill %s evolution: %s", skill_name, action)
        except Exception as exc:
            logger.debug("skill evolution failed for %s: %s", skill_name, exc)

    def _directive_system_message(self) -> dict[str, Any]:
        """Build the durable system-directive message.

        On Anthropic models the text is wrapped in a content block with a
        ``cache_control`` breakpoint so this large, session-stable prefix is read
        from cache on every iteration after the first instead of being re-billed.
        """
        directive = self._effective_system_directive()
        if self._caching_enabled:
            return {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": directive,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        return {"role": "system", "content": directive}

    def _effective_system_directive(self) -> str:
        directive = self._system_directive
        engine = self._evolution_engine
        if engine is None:
            return directive
        policy_sections: list[str] = []
        prompt_policy = engine.active_prompt_policy()
        if prompt_policy:
            policy_sections.append(
                "Active proof-promoted prompt policy (additive, cannot override "
                f"core task-contract/sandbox/approval rules):\n{prompt_policy}"
            )
        toolset_policy = engine.active_toolset_policy()
        if toolset_policy:
            policy_sections.append(
                "Active proof-promoted toolset-selection policy (additive, cannot "
                f"hide required evidence tools):\n{toolset_policy}"
            )
        if not policy_sections:
            return directive
        return f"{directive}\n\n" + "\n\n".join(policy_sections)

    def _build_host_environment_message(self) -> dict[str, Any]:
        """Inject an authoritative host-environment summary once per session.

        Surfacing the OS, shell, privilege level, and which package managers /
        runtimes are actually on PATH -- up front, without the model having to
        call get_system_environment -- stops it from guessing the platform wrong
        (e.g. running apt-get on a Windows host) and from spinning on
        trial-and-error which/where probes.

        Also flags sandbox startup failures so the agent knows it is on the
        host even when AGENT_SANDBOX=docker was requested.
        """
        raw = ""
        getter = getattr(self._tools, "get_system_environment", None)
        if callable(getter):
            try:
                raw = getter()
            except Exception:  # never let env probing break session seeding
                raw = ""
        sandbox_failed = getattr(self._tools, "sandbox_startup_failed", False)
        host_execution_disabled_reason = getattr(
            self._tools, "host_execution_disabled_reason", None
        )
        return {
            "role": "system",
            "content": _summarize_host_environment(
                raw,
                sandbox_failed=sandbox_failed,
                host_execution_disabled_reason=host_execution_disabled_reason,
            ),
        }

    def _compact_context_window(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return a copy of *messages* with bloated tool results condensed.

        Every ``tool`` role message is run through :func:`_summarize_tool_output`:
        outputs at or below ``_TOOL_OUTPUT_INLINE_MAX_CHARS`` are shown in full;
        larger ones become a head+tail preview carrying the ``expand_tool_output``
        handle, so nothing is silently lost -- the full text stays in the durable
        history and can be paged back in on demand.  ``expand_tool_output`` results
        are passed through untouched (the model already chose that window).

        Non-tool messages (system, user, assistant) are passed through unchanged.
        The original *messages* list is never mutated; callers should pass the
        returned list to the LLM and continue appending to the original.
        """
        # Build a lookup from tool_call_id to tool_name using every assistant
        # message that carries tool_calls.
        call_id_to_name: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    try:
                        call_id_to_name[tc["id"]] = tc["function"]["name"]
                    except (KeyError, TypeError):
                        pass

        compacted: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") != "tool":
                compacted.append(msg)
                continue

            handle = msg.get("tool_call_id", "")
            tool_name = call_id_to_name.get(handle, "unknown_tool")
            raw_content: str = msg.get("content", "")

            # expand_tool_output results are a deliberate, already-bounded slice the
            # model explicitly asked for -- pass them through untouched.
            if tool_name == "expand_tool_output":
                compacted.append(msg)
                continue

            is_error, text = _classify_tool_result(tool_name, raw_content)
            body = _summarize_tool_output(text, handle, tool_name)

            if is_error:
                content = body
            else:
                content = (
                    f"Tool {tool_name} output:\n{body}"
                    if body.strip()
                    else f"Tool {tool_name} executed successfully (no output)."
                )
            compacted.append({**msg, "content": content})

        executive_summary = _build_executive_summary(messages)
        logger.debug(
            "Context compacted: %d messages, %d tool messages condensed",
            len(messages),
            sum(1 for m in messages if m.get("role") == "tool"),
        )
        # Appended, not prepended: the stable system directive + tool block must
        # stay at the front of the request to remain a cacheable prefix. The
        # volatile summary sits at the end (after the cache breakpoint), where it
        # also carries useful recency weight close to generation.
        compacted = _prune_message_window(compacted)
        if self._caching_enabled:
            # Roll a cache breakpoint onto the most recent tool result so the
            # whole conversation prefix up to it is written once and read back
            # cheaply next iteration, instead of the transcript being re-billed
            # in full every loop. Done after pruning so the marked message is
            # one that actually ships, and before the volatile executive summary
            # so the cached prefix stays stable across iterations.
            _mark_last_tool_result_cache_breakpoint(compacted)
        return _sanitize_messages_for_llm([
            *compacted,
            {"role": "system", "content": executive_summary},
        ])

    async def _execute_delegate_task(self, arguments: dict[str, Any]) -> str:
        """Spin up one or more isolated sub-AgentEngines and run them.

        Supports two calling forms:
        1. Single task (backward compat):
           {agent_type, task_description, context_payload}
        2. Parallel batch:
           {tasks: [{agent_type, task_description, context_payload},...], mode: "parallel"}
        """
        # -- Parallel / batch form ----------------------------------------
        if "tasks" in arguments:
            tasks_list: list[dict[str, Any]] = arguments["tasks"]
            mode = arguments.get("mode", "sequential")
            if mode == "parallel":
                results = await asyncio.gather(
                    *(self._run_single_delegate(t) for t in tasks_list),
                    return_exceptions=True,
                )
                return json.dumps(
                    [
                        {"task": t.get("task_description", "")[:80],
                         "agent_type": t.get("agent_type"),
                         "result": r if isinstance(r, str) else f"[error] {r}"}
                        for t, r in zip(tasks_list, results, strict=False)
                    ],
                    indent=2,
                )
            else:
                parts: list[str] = []
                for task_args in tasks_list:
                    result = await self._run_single_delegate(task_args)
                    parts.append(f"[{task_args.get('agent_type')}] {result}")
                return "\n\n".join(parts)

        # -- Single task form (backward compat) ---------------------------
        return await self._run_single_delegate(arguments)

    async def _run_single_delegate(self, arguments: dict[str, Any]) -> str:
        agent_type: str = arguments.get("agent_type", "researcher")
        task_description: str = arguments.get("task_description", "")
        context_payload: dict[str, Any] = arguments.get("context_payload") or {}

        if agent_type not in _SUB_AGENT_DIRECTIVES:
            agent_type = "researcher"
        directive = _SUB_AGENT_DIRECTIVES[agent_type]

        prompt = task_description
        if context_payload:
            prompt = (
                f"Context:\n{json.dumps(context_payload, indent=2)}\n\n"
                f"Task:\n{task_description}"
            )

        sub_model = {
            "researcher": self._fast_model,
            "coder": self._strong_model,
            "auditor": self._strong_model,
            "planner": self._fast_model,
        }.get(agent_type, self._fast_model)

        sub_session_id = f"sub_{agent_type}_{uuid.uuid4().hex[:8]}"
        sub_agent = AgentEngine(
            memory=self._memory,
            tools=self._tools,
            model=sub_model,
            fast_model=self._fast_model,
            strong_model=self._strong_model,
            system_directive=directive,
            require_task_contract=False,
        )

        sub_message = NormalizedMessage(
            session_id=sub_session_id,
            role="user",
            content=prompt,
        )
        logger.debug(
            "Delegating to %r sub-agent (session %s, prompt_len=%d)",
            agent_type, sub_session_id, len(prompt),
        )
        return await sub_agent.process_task(sub_message)

    def reset_session(self, session_id: str) -> bool:
        """Drop a session's conversation history so the next turn starts fresh.

        Returns ``True`` if a session existed and was cleared. Used by the
        messaging adapters' ``/new`` / ``/reset`` commands.
        """
        existed = self._histories.pop(session_id, None) is not None
        self._session_steps.pop(session_id, None)
        return existed

    def task_graph_snapshot(self, session_id: str) -> dict[str, Any] | None:
        messages = self._histories.get(session_id)
        if messages is None:
            return None
        return self._task_graph.inspect(messages, self._session_steps.get(session_id, []))

    def verify_task_graph(self, session_id: str) -> dict[str, Any] | None:
        messages = self._histories.get(session_id)
        if messages is None:
            return None
        return self._task_graph.verify(messages, self._session_steps.get(session_id, []))

    def task_graph_status_summary(self) -> dict[str, Any]:
        active = 0
        blocked = 0
        open_nodes = 0
        for session_id, messages in self._histories.items():
            snapshot = self._task_graph.inspect(
                messages, self._session_steps.get(session_id, [])
            )
            if not snapshot.get("has_graph"):
                continue
            active += 1
            blocked += len(snapshot.get("blocked_nodes", []))
            summary = snapshot.get("summary") or {}
            open_nodes += int(summary.get("open") or 0)
        return {"active": active, "blocked": blocked, "open_nodes": open_nodes}

    def _touch_history(self, session_id: str) -> list[dict[str, Any]] | None:
        """Return the session's history, marking it most-recently-used (LRU)."""
        messages = self._histories.pop(session_id, None)
        if messages is not None:
            self._histories[session_id] = messages
        return messages

    def _evict_histories(self) -> None:
        """Drop least-recently-used sessions once the cap is exceeded."""
        while len(self._histories) > _MAX_SESSIONS:
            oldest = next(iter(self._histories))
            self._histories.pop(oldest, None)
            self._session_steps.pop(oldest, None)

    async def _fetch_context(self, query: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(
                self._memory.retrieve_context, query, "semantic"
            )
        except Exception:
            # Collection may be empty on first use; treat as no context
            logger.debug("Context retrieval returned no results for query: %r", query)
            return {"query_type": "semantic", "results": []}

    async def _bootstrap_session(
        self,
        tool_index: dict[str, str],
        all_tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return synthetic tool-call + result messages that pre-load system state.

        Injected once per session, before the first user message, so the model
        enters the ReAct loop already aware of available tables and tools.
        """
        CALL_ID_TABLES = "bootstrap_list_tables"
        CALL_ID_TOOLS = "bootstrap_list_all_tools"

        # Single synthetic assistant turn that "requests" both probes
        assistant_bootstrap: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": CALL_ID_TABLES,
                    "type": "function",
                    "function": {"name": "list_tables", "arguments": "{}"},
                },
                {
                    "id": CALL_ID_TOOLS,
                    "type": "function",
                    "function": {"name": "list_all_tools", "arguments": "{}"},
                },
            ],
        }

        # Execute list_tables via MCP
        server = tool_index.get("list_tables")
        if server is not None:
            try:
                mcp_result = await self._tools.call_tool(server, "list_tables", {})
                tables_content = _extract_tool_text(mcp_result)
            except Exception as exc:
                tables_content = f"[bootstrap error] list_tables failed: {exc}"
                logger.warning("Session bootstrap: list_tables raised %s", exc)
        else:
            tables_content = "[bootstrap] list_tables not available on any connected server"

        # Format list_all_tools() result from the already-fetched snapshot
        tools_content = json.dumps(
            [
                {
                    "name": t["name"],
                    "server": t.get("server", ""),
                }
                for t in all_tools
            ],
            indent=2,
        ) or "[]"

        logger.debug(
            "Session bootstrap injected: list_tables=%d chars, list_all_tools=%d tools",
            len(tables_content),
            len(all_tools),
        )

        return [
            assistant_bootstrap,
            {"role": "tool", "tool_call_id": CALL_ID_TABLES, "content": tables_content},
            {"role": "tool", "tool_call_id": CALL_ID_TOOLS, "content": tools_content},
        ]


class TypeSafeAgentEngine(AgentEngine):
    """Typed public alias for the tool-aware AgentEngine implementation."""


def _summarize_tool_output(text: str, handle: str, tool_name: str) -> str:
    """Show *text* in full when small, else a head+tail preview plus an expand handle.

    Large outputs are never silently dropped: the preview names how much is
    hidden and how to retrieve it with ``expand_tool_output``, and the full text
    remains in the durable history.  The preview itself is char-bounded so a few
    enormous lines can't blow the budget.
    """
    if len(text) <= _TOOL_OUTPUT_INLINE_MAX_CHARS:
        return text

    half = _TOOL_OUTPUT_INLINE_MAX_CHARS // 2
    lines = text.splitlines()
    head = "\n".join(lines[:_TOOL_OUTPUT_PREVIEW_HEAD_LINES])[:half]
    tail = "\n".join(lines[-_TOOL_OUTPUT_PREVIEW_TAIL_LINES:])[-half:]
    hidden = max(
        0, len(lines) - _TOOL_OUTPUT_PREVIEW_HEAD_LINES - _TOOL_OUTPUT_PREVIEW_TAIL_LINES
    )
    return (
        f"{head}\n"
        f"[... {hidden} line(s) / {len(text)} chars hidden. Retrieve more with "
        f'expand_tool_output(handle="{handle}", start_line=...) ...]\n'
        f"{tail}"
    )


def _make_tool_result_event(
    *,
    tool_name: str,
    tool_call_id: str,
    content: str,
    is_error: bool,
) -> dict[str, Any]:
    """Build a compact UI event for a tool result without hiding failures."""
    _, text = _classify_tool_result(tool_name, content)
    display = _summarize_tool_output(text, tool_call_id, tool_name)
    if not display.strip():
        display = "Tool executed successfully (no output)."
    metadata: dict[str, Any] = {"tool_call_id": tool_call_id}
    if tool_name in {
        "execute_terminal_command",
        "execute_background_service",
        "wait_for_port",
        "get_filesystem_process_evidence",
    }:
        try:
            parsed = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        if isinstance(parsed, dict):
            for key in ("exit_code", "status", "port", "open", "scope"):
                if key in parsed:
                    metadata[key] = parsed[key]
    return {
        "type": "tool_result",
        "tool": tool_name,
        "is_error": is_error,
        "content": display,
        "metadata": metadata,
    }


def _lookup_tool_output(
    messages: list[dict[str, Any]], handle: str
) -> tuple[str, str] | None:
    """Return ``(tool_name, display_text)`` for the tool message with *handle*.

    Reads from the durable history (the single source of truth -- no separate
    store), unwrapping the result the same way compaction does so previews and
    expansions stay consistent.  Returns ``None`` if no such tool result exists.
    """
    raw: str | None = None
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id") == handle:
            raw = msg.get("content", "")
            break
    if raw is None:
        return None

    tool_name = "unknown_tool"
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                try:
                    if tc["id"] == handle:
                        tool_name = tc["function"]["name"]
                except (KeyError, TypeError):
                    pass

    _, text = _classify_tool_result(tool_name, raw)
    return tool_name, text


def _run_expand_tool_output(
    messages: list[dict[str, Any]], arguments: dict[str, Any]
) -> tuple[str, bool]:
    """Execute the ``expand_tool_output`` builtin. Returns ``(content, is_error)``."""
    handle = str(arguments.get("handle", "")).strip()
    if not handle:
        return "[expand_tool_output error] a 'handle' argument is required.", True

    found = _lookup_tool_output(messages, handle)
    if found is None:
        return (
            f"[expand_tool_output error] no stored output for handle {handle!r}. "
            "Use the exact handle printed in the truncation notice.",
            True,
        )

    tool_name, text = found
    lines = text.splitlines()
    total = len(lines)

    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    start = max(0, _as_int(arguments.get("start_line", 0), 0))
    max_lines = max(1, min(_as_int(arguments.get("max_lines", 200), 200), _EXPAND_MAX_LINES))

    chunk = lines[start : start + max_lines]
    end = start + len(chunk)
    more = f" More remains; call again with start_line={end}." if end < total else ""
    return (
        f"[expand_tool_output] {tool_name} handle={handle}, lines {start}-{end} "
        f"of {total}.{more}\n" + "\n".join(chunk),
        False,
    )


_OS_PACKAGE_MANAGERS: tuple[str, ...] = (
    "apt-get", "apt", "yum", "dnf", "pacman", "apk", "brew",
    "choco", "scoop", "winget",
)


def _summarize_host_environment(
    raw: str,
    *,
    sandbox_failed: bool = False,
    host_execution_disabled_reason: str | None = None,
) -> str:
    """Render the get_system_environment JSON into a concise, directive summary.

    ``sandbox_failed=True`` adds a prominent warning when AGENT_SANDBOX=docker
    was configured but Docker could not be reached. When host fallback is
    disabled, the agent is told that host-changing tools are blocked instead of
    silently running on the bare host.
    """
    try:
        data = json.loads(raw) if raw else {}
    except (TypeError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict) or not data:
        return (
            "=== EXECUTION ENVIRONMENT ===\n"
            "Unavailable. Call get_system_environment BEFORE any install or setup "
            "command so you target the right OS, shell, and package manager. Never "
            "guess the platform.\n"
            "========================"
        )

    os_label = str(data.get("os") or "unknown")
    shell_raw = data.get("shell")
    shell = shell_raw if isinstance(shell_raw, dict) else {}
    shell_name = str(shell.get("shell") or "unknown")
    posix = bool(shell.get("posix"))
    user_raw = data.get("user")
    user = user_raw if isinstance(user_raw, dict) else {}
    is_root = bool(user.get("is_root"))
    sudo = bool(user.get("sudo_available"))
    runtimes_raw = data.get("runtimes")
    runtimes = runtimes_raw if isinstance(runtimes_raw, dict) else {}
    available = sorted(name for name, present in runtimes.items() if present)
    sandbox_raw = data.get("sandbox")
    sandbox = sandbox_raw if isinstance(sandbox_raw, dict) else {}

    pkg_present = [name for name in _OS_PACKAGE_MANAGERS if name in available]
    runtime_present = [name for name in available if name not in _OS_PACKAGE_MANAGERS]

    sandbox_line = ""
    if sandbox:
        mode = sandbox.get("mode", "unknown")
        image = sandbox.get("image", "")
        workdir = sandbox.get("container_workdir", "/workspace")
        sandbox_line = (
            f"Sandbox: {mode} (image={image}; workdir={workdir}) â€” "
            f"commands run inside an isolated Linux container; apt-get IS available.\n"
        )
    elif host_execution_disabled_reason:
        sandbox_line = (
            "SANDBOX: FAILED TO START; HOST EXECUTION IS BLOCKED. Terminal, "
            "background-service, process, and port tools will return a blocking "
            "error until Docker Desktop is started and the backend is restarted. "
            "Do NOT try Windows/cmd/PowerShell variants to work around this; the "
            "correct repair is to restore the sandbox or explicitly enable "
            "AGENT_SANDBOX_HOST_FALLBACK=true.\n"
            f"Sandbox blocker: {host_execution_disabled_reason}\n"
        )
    elif sandbox_failed:
        sandbox_line = (
            "SANDBOX: FAILED TO START (Docker Desktop is not running or unreachable). "
            "All commands execute DIRECTLY on the host OS below â€” treat it as a plain "
            "host session; do NOT assume a Linux container environment.\n"
        )

    return (
        "=== EXECUTION ENVIRONMENT (authoritative - do not guess the platform) ===\n"
        f"{sandbox_line}"
        f"OS: {os_label}    Shell: {shell_name} ({'POSIX' if posix else 'non-POSIX'})\n"
        f"Privileges: {'root' if is_root else 'non-root'}; "
        f"sudo {'available' if sudo else 'unavailable'}\n"
        f"Package managers on PATH: {', '.join(pkg_present) or 'none detected'}\n"
        f"Runtimes on PATH: {', '.join(runtime_present) or 'none detected'}\n"
        "Run only commands that fit THIS OS and shell. Do NOT run apt-get/yum/brew "
        "on Windows, and do NOT run choco/scoop/winget on Linux. Install only with a "
        "package manager listed above; if none is available but the runtime you need "
        "is already on PATH, just use it directly. These facts are stated here, so do "
        "NOT re-probe them with repeated which/where/--version commands.\n"
        "===================================================================="
    )


def _build_system_prompt(context: dict[str, Any]) -> str:
    results = context.get("results", [])
    if not results:
        return "You are a helpful AI assistant with access to tools."

    memory_lines = "\n".join(
        f"- {r['text']}" for r in results if r.get("text")
    )
    return (
        "You are a helpful AI assistant with access to tools.\n\n"
        "Relevant context from memory:\n"
        f"{memory_lines}"
    )


def _model_supports_caching(model: str) -> bool:
    """Whether *model* accepts Anthropic ``cache_control`` breakpoints.

    Caching markers are Anthropic-specific; sending them to other providers
    (e.g. the gpt-4o-mini default) can raise, so gate on the model string.
    """
    m = model.lower()
    return "claude" in m or "anthropic" in m


def _mark_last_tool_result_cache_breakpoint(messages: list[dict[str, Any]]) -> None:
    """Attach an Anthropic ``cache_control`` breakpoint to the most recent tool
    result, in place.

    Without this, only the system directive and the tool schema block are
    cached, so the entire (growing) conversation transcript is re-billed as
    fresh input on every ReAct iteration. litellm reads message-level
    ``cache_control`` for ``tool``-role messages and applies it to the emitted
    Anthropic ``tool_result`` block (see litellm prompt-template factory), so
    marking the latest tool result turns everything up to and including it into
    a cacheable prefix.

    The marked result is byte-stable across iterations -- its compacted preview
    is deterministic -- so the next iteration's request matches it as a cache
    read (~90% cheaper) rather than a rewrite. Only one result is marked, so the
    total breakpoints (directive + tool schema + this) stay within Anthropic's
    limit of 4. The element is replaced with a shallow copy so the caller's
    durable history is never mutated.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if (
            msg.get("role") == "tool"
            and isinstance(msg.get("content"), str)
            and msg["content"].strip()
        ):
            messages[i] = {**msg, "cache_control": {"type": "ephemeral"}}
            return


def _to_litellm_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for tool in tools:
        description = str(tool.get("description", "")).strip()
        if len(description) > 700:
            description = description[:700].rstrip() + "..."
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": description,
                    "parameters": tool["inputSchema"],
                },
            }
        )
    return schemas


def _serialise_assistant_msg(msg: Any) -> dict[str, Any]:
    """Convert a litellm assistant message with tool calls into a plain dict."""
    tool_calls = [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        }
        for tc in (msg.tool_calls or [])
    ]
    content = msg.content
    if tool_calls and (content is None or not str(content).strip()):
        content = None
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }


def _extract_tool_text(result: CallToolResult) -> str:
    """Flatten MCP content blocks into a single string for the tool message."""
    parts: list[str] = []
    for block in result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif hasattr(block, "mimeType"):
            parts.append(f"[{block.mimeType} data]")
        else:
            parts.append(str(block))
    text = "\n".join(parts) or "(empty result)"
    return f"[tool error] {text}" if result.isError else text
