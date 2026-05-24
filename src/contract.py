"""Task-contract system: validation, evidence checking, and status gating.

The contract is the agent's binding commitment declared on the first turn of
every task via ``set_task_contract``.  In execute mode, ``_contract_completion_status``
gates the final answer on structured evidence so "I'll create the file now"
with no actual tool call is impossible.

Dependency order: this module imports only from ``evaluator`` (for
``ExecutionStep``), so it can be safely imported by ``planning`` and ``agent``
without creating circular dependencies.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from evaluator import ExecutionStep
from toolsets import TOOLSET_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TASK_CONTRACT_TOOL = "set_task_contract"

_TASK_CONTRACT_EVIDENCE: frozenset[str] = frozenset(
    {
        "filesystem_artifact",
        "published_static_site_url",
        "running_http_service",
        "database_mutation",
        "command_output",
        "none",
    }
)

_HOST_EXECUTION_TOOLS: frozenset[str] = frozenset(
    {"execute_terminal_command", "execute_background_service"}
)

_CONTRACT_CONTROL_TOOLS: frozenset[str] = frozenset(
    {"update_plan", "expand_tool_output"}
)

_CONTRACT_EVIDENCE_TOOLS: dict[str, frozenset[str]] = {
    "filesystem_artifact": frozenset(
        {
            "write_text_file",
            "execute_terminal_command",
            "get_filesystem_process_evidence",
            "publish_static_site",
        }
    ),
    "published_static_site_url": frozenset(
        {
            "write_text_file",
            "execute_terminal_command",
            "get_filesystem_process_evidence",
            "publish_static_site",
        }
    ),
    "running_http_service": frozenset(
        {
            "execute_background_service",
            "execute_terminal_command",
            "get_filesystem_process_evidence",
            "expose_local_http_service",
        }
    ),
    "database_mutation": frozenset(
        {"create_table", "write_query", "read_query", "describe_table", "list_tables"}
    ),
    "command_output": frozenset(
        {"execute_terminal_command", "execute_background_service"}
    ),
    "artifact_quality": frozenset(
        {"write_text_file", "publish_static_site", "browser_navigate", "browser_screenshot"}
    ),
}

# ---------------------------------------------------------------------------
# Shared message-history helpers
# ---------------------------------------------------------------------------

def _is_continuation_signal(text: str) -> bool:
    normalized = text.strip().lower().strip(".!?")
    return normalized in {
        "yes",
        "y",
        "ok",
        "okay",
        "go",
        "continue",
        "do it",
        "proceed",
        "start",
        "run it",
        "carry on",
    }


def _current_task_start_index(messages: list[dict[str, Any]]) -> int:
    user_indices = [
        idx for idx, msg in enumerate(messages) if msg.get("role") == "user"
    ]
    if not user_indices:
        return 0

    latest_idx = user_indices[-1]
    latest_text = str(messages[latest_idx].get("content") or "")
    if _is_continuation_signal(latest_text):
        for idx in reversed(user_indices[:-1]):
            text = str(messages[idx].get("content") or "")
            if text and not _is_continuation_signal(text):
                return idx
    return latest_idx


def _current_task_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return messages[_current_task_start_index(messages):]


# ---------------------------------------------------------------------------
# Minimal plan lookups (needed by _contract_completion_status; kept here to
# avoid a circular import: planning.py imports contract, not vice-versa)
# ---------------------------------------------------------------------------

# Canonical plan statuses and tolerant synonyms. The model frequently emits
# near-miss spellings ("waiting", "complete", "in progress"); coercing them to
# the canonical set keeps a slightly-wrong update_plan call from silently
# no-opping and trapping the agent with a plan whose steps never close.
_VALID_PLAN_STATUSES: frozenset[str] = frozenset(
    {"pending", "in_progress", "done", "failed"}
)
_PLAN_STATUS_SYNONYMS: dict[str, str] = {
    "todo": "pending",
    "not_started": "pending",
    "notstarted": "pending",
    "waiting": "pending",
    "queued": "pending",
    "open": "pending",
    "new": "pending",
    "active": "in_progress",
    "doing": "in_progress",
    "started": "in_progress",
    "running": "in_progress",
    "wip": "in_progress",
    "inprogress": "in_progress",
    "complete": "done",
    "completed": "done",
    "finished": "done",
    "success": "done",
    "succeeded": "done",
    "ok": "done",
    "error": "failed",
    "errored": "failed",
    "cancelled": "failed",
    "canceled": "failed",
    "skipped": "failed",
    "blocked": "failed",
    "abandoned": "failed",
}


def _normalise_plan_status(value: Any) -> str:
    """Map a model-supplied status string onto the canonical four-value set."""
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if token in _VALID_PLAN_STATUSES:
        return token
    return _PLAN_STATUS_SYNONYMS.get(token, "pending")


def _coerce_plan_steps(value: Any) -> list[dict[str, Any]] | None:
    """Normalise an update_plan steps array into clean ``{title, status}`` dicts.

    Tolerant of the common malformed shapes the model produces: a JSON string
    instead of a list, ``step``/``name``/``description`` instead of ``title``,
    and non-canonical status spellings (coerced via ``_normalise_plan_status``).
    Returns ``None`` only when the value cannot be read as a list at all.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list):
        return None
    steps: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        title = (
            item.get("title")
            or item.get("step")
            or item.get("name")
            or item.get("description")
        )
        if title is None or not str(title).strip():
            continue
        steps.append(
            {
                "title": str(title),
                "status": _normalise_plan_status(item.get("status")),
            }
        )
    return steps


def _plan_steps_from_args(args: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Resolve the steps array from update_plan arguments.

    Accepts the canonical ``steps`` key and the common ``plan`` alias the model
    sometimes emits (often as a JSON string), so a near-miss call still updates
    the plan instead of silently erroring.
    """
    raw = args.get("steps")
    if raw is None:
        raw = args.get("plan")
    return _coerce_plan_steps(raw)


def _latest_plan(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Return the steps of the most recent ``update_plan`` call, or ``None``."""
    for msg in reversed(_current_task_messages(messages)):
        if msg.get("role") != "assistant":
            continue
        for tc in reversed(msg.get("tool_calls") or []):
            try:
                if tc["function"]["name"] != "update_plan":
                    continue
                args = json.loads(tc["function"].get("arguments") or "{}")
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            coerced = _plan_steps_from_args(args)
            if coerced:
                return coerced
    return None


_PLAN_OPEN_STATUSES: frozenset[str] = frozenset({"pending", "in_progress"})


def _plan_has_open_steps(messages: list[dict[str, Any]]) -> bool:
    """True when the current plan exists and still has pending/in_progress steps."""
    plan = _latest_plan(messages)
    if not plan:
        return False
    return any(
        isinstance(step, dict) and step.get("status") in _PLAN_OPEN_STATUSES
        for step in plan
    )


# ---------------------------------------------------------------------------
# Tool schema filtering
# ---------------------------------------------------------------------------

def _filter_tool_schemas(
    tool_schemas: list[dict[str, Any]], names: set[str]
) -> list[dict[str, Any]]:
    return [
        schema
        for schema in tool_schemas
        if schema.get("function", {}).get("name") in names
    ]


def _tool_names_for_contract_status(
    contract: dict[str, Any], status: dict[str, Any]
) -> set[str]:
    missing = set(status.get("missing") or [])
    if "plan" in missing:
        return {"update_plan"}

    evidence_missing = [
        item for item in missing if item not in {"plan", "plan_open_steps"}
    ]
    if evidence_missing:
        names: set[str] = set()
        for requirement in evidence_missing:
            names.update(_CONTRACT_EVIDENCE_TOOLS.get(requirement, frozenset()))
        return names or set(_CONTRACT_CONTROL_TOOLS)

    if "plan_open_steps" in missing:
        return {"update_plan"}

    names = set(_CONTRACT_CONTROL_TOOLS)
    for requirement in contract.get("evidence_requirements", []):
        names.update(_CONTRACT_EVIDENCE_TOOLS.get(requirement, frozenset()))
    return names


# ---------------------------------------------------------------------------
# Contract validation and retrieval
# ---------------------------------------------------------------------------

def _run_set_task_contract(arguments: dict[str, Any]) -> tuple[str, bool]:
    contract, error = _normalise_task_contract(arguments)
    if error is not None:
        return f"[set_task_contract error] {error}", True
    return json.dumps({"contract_set": True, **contract}, indent=2), False


def _split_criteria_string(text: str) -> list[str]:
    """Recover a list of success criteria from a single string.

    Handles the two shapes the model commonly emits when it ignores the array
    schema: ``<item>...</item>`` wrapped entries, and newline/semicolon
    separated lines.
    """
    items = re.findall(r"<item>(.*?)</item>", text, re.IGNORECASE | re.DOTALL)
    if not items:
        items = re.split(r"[\n;]+", text)
    return [stripped for stripped in (item.strip() for item in items) if stripped]


def _normalise_task_contract(
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    mode = arguments.get("mode")
    if mode not in {"answer", "execute"}:
        return {}, "mode must be 'answer' or 'execute'"

    summary = " ".join(str(arguments.get("summary", "")).split())
    if not summary:
        return {}, "summary must be a non-empty string"

    raw_criteria = arguments.get("success_criteria")
    if isinstance(raw_criteria, str):
        # The model sometimes sends a single string (often wrapped in <item>
        # tags or newline-separated) instead of a JSON array. Recover the items
        # rather than rejecting the whole contract.
        raw_criteria = _split_criteria_string(raw_criteria)
    if not isinstance(raw_criteria, list):
        return {}, "success_criteria must be an array of strings"
    success_criteria = [
        " ".join(str(item).split()) for item in raw_criteria if str(item).strip()
    ]
    if not success_criteria:
        return {}, "success_criteria must contain at least one item"

    raw_evidence = arguments.get("evidence_requirements")
    if not isinstance(raw_evidence, list):
        return {}, "evidence_requirements must be an array"
    evidence = [str(item) for item in raw_evidence]
    unknown = sorted(set(evidence) - _TASK_CONTRACT_EVIDENCE)
    if unknown:
        return {}, f"unknown evidence requirement(s): {', '.join(unknown)}"

    if mode == "answer":
        evidence = ["none"]
    else:
        evidence = [item for item in evidence if item != "none"]
        if not evidence:
            return {}, "execute mode requires at least one non-'none' evidence requirement"

    # Optional toolset selection — narrows available tools for subsequent iterations.
    toolset = str(arguments.get("toolset", "all"))
    if toolset not in TOOLSET_NAMES:
        return {}, f"unknown toolset {toolset!r}; valid values: {sorted(TOOLSET_NAMES)}"

    return (
        {
            "mode": mode,
            "summary": summary,
            "success_criteria": success_criteria,
            "evidence_requirements": list(dict.fromkeys(evidence)),
            "toolset": toolset,
        },
        None,
    )


def _latest_task_contract(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for msg in reversed(_current_task_messages(messages)):
        if msg.get("role") != "assistant":
            continue
        for tc in reversed(msg.get("tool_calls") or []):
            try:
                if tc["function"]["name"] != _TASK_CONTRACT_TOOL:
                    continue
                args = json.loads(tc["function"].get("arguments") or "{}")
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            contract, error = _normalise_task_contract(args)
            if error is None:
                return contract
    return None


# ---------------------------------------------------------------------------
# Contract completion status
# ---------------------------------------------------------------------------

def _contract_completion_status(
    contract: dict[str, Any] | None,
    messages: list[dict[str, Any]],
    steps: list[ExecutionStep],
    *,
    contract_required: bool,
) -> dict[str, Any]:
    if contract is None:
        if contract_required:
            return {
                "complete": False,
                "missing": ["task_contract"],
                "plan_open": False,
            }
        return {"complete": True, "missing": [], "plan_open": False}

    if contract.get("mode") == "answer":
        return {"complete": True, "missing": [], "plan_open": False}

    missing: list[str] = []
    plan = _latest_plan(messages)
    plan_open = plan is None or _plan_has_open_steps(messages)
    if plan is None:
        missing.append("plan")
    elif plan_open:
        missing.append("plan_open_steps")

    for requirement in contract.get("evidence_requirements", []):
        if requirement == "none":
            continue
        if not _evidence_requirement_satisfied(requirement, steps):
            missing.append(requirement)

    if _artifact_quality_missing(contract, steps):
        missing.append("artifact_quality")

    return {
        "complete": not missing,
        "missing": missing,
        "plan_open": plan_open,
    }


def _can_stream_text_before_final(
    contract: dict[str, Any] | None,
    messages: list[dict[str, Any]],
    steps: list[ExecutionStep],
) -> bool:
    if contract is None:
        return False
    if contract.get("mode") == "answer":
        return True
    return bool(
        _contract_completion_status(
            contract, messages, steps, contract_required=False
        )["complete"]
    )


# ---------------------------------------------------------------------------
# Evidence satisfaction checks
# ---------------------------------------------------------------------------

def _evidence_requirement_satisfied(
    requirement: str, steps: list[ExecutionStep]
) -> bool:
    for step in steps:
        if step.kind != "tool_result" or step.metadata.get("is_error"):
            continue
        tool_name = step.metadata.get("tool_name")

        if requirement == "published_static_site_url":
            if tool_name == "publish_static_site" and _publish_static_site_evidence_is_positive(step.content):
                return True
        elif requirement == "running_http_service":
            if tool_name == "expose_local_http_service" and _expose_local_http_service_evidence_is_positive(step.content):
                return True
        elif requirement == "filesystem_artifact":
            if tool_name == "get_filesystem_process_evidence" and _filesystem_process_evidence_is_positive(step.content):
                return True
            if tool_name == "publish_static_site" and _publish_static_site_evidence_is_positive(step.content):
                return True
            if tool_name == "write_text_file" and _write_text_file_evidence_is_positive(step.content):
                return True
        elif requirement == "database_mutation":
            if tool_name in {"create_table", "write_query"}:
                return True
        elif requirement == "command_output":
            if _successful_command_output_evidence(tool_name, step.content):
                return True
    return False


def _successful_command_output_evidence(tool_name: str | None, content: str) -> bool:
    if tool_name == "execute_background_service":
        try:
            data = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return False
        return data.get("status") not in {"error", "failed"}
    if tool_name != "execute_terminal_command":
        return False
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False
    return data.get("exit_code") == 0 and bool(
        str(data.get("stdout") or data.get("stderr") or "").strip()
    )


def _publish_static_site_evidence_is_positive(content: str) -> bool:
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False
    return bool(
        data.get("published")
        and data.get("index_exists")
        and data.get("url")
        and _artifact_quality_payload_is_acceptable(data)
    )


def _write_text_file_evidence_is_positive(content: str) -> bool:
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False
    return bool(
        data.get("written")
        and data.get("exists")
        and data.get("size_bytes", 0) > 0
        and _artifact_quality_payload_is_acceptable(data)
    )


def _artifact_quality_payload_is_acceptable(data: dict[str, Any]) -> bool:
    quality = data.get("artifact_quality")
    if not isinstance(quality, dict):
        return True
    return not bool(quality.get("placeholder_detected"))


def _contract_artifact_text(contract: dict[str, Any]) -> str:
    pieces = [
        str(contract.get("summary") or ""),
        *[str(item) for item in contract.get("success_criteria") or []],
    ]
    return " ".join(pieces).lower()


def _contract_requests_web_artifact(contract: dict[str, Any]) -> bool:
    text = _contract_artifact_text(contract)
    return any(
        token in text
        for token in ("website", "web site", "webpage", "web page", "html", "landing page", "static site")
    )


def _latest_artifact_quality(steps: list[ExecutionStep]) -> dict[str, Any] | None:
    for step in reversed(steps):
        if step.kind != "tool_result" or step.metadata.get("is_error"):
            continue
        if step.metadata.get("tool_name") not in {"write_text_file", "publish_static_site"}:
            continue
        try:
            data = json.loads(step.content)
        except (TypeError, json.JSONDecodeError):
            continue
        quality = data.get("artifact_quality")
        if isinstance(quality, dict):
            return quality
    return None


def _artifact_quality_missing(contract: dict[str, Any], steps: list[ExecutionStep]) -> bool:
    if not _contract_requests_web_artifact(contract):
        return False

    quality = _latest_artifact_quality(steps)
    if not quality:
        return False
    if quality.get("placeholder_detected"):
        return True

    text = _contract_artifact_text(contract)
    interactive_requested = any(
        token in text
        for token in ("interactive", "interaction", "javascript", "js", "quiz", "slider", "calculator", "button")
    )
    highly_interactive_requested = "highly interactive" in text
    interactive_count = int(quality.get("interactive_signal_count") or 0)
    style_count = int(quality.get("style_rule_count") or 0)
    word_count = int(quality.get("content_word_count") or 0)

    if highly_interactive_requested and interactive_count < 3:
        return True
    if interactive_requested and interactive_count < 1:
        return True
    if style_count < 2 and any(token in text for token in ("styled", "style", "beautiful", "highly interactive")):
        return True
    if word_count < 50 and any(token in text for token in ("about", "importance", "explain", "content")):
        return True

    # Baseline quality gate: ANY website request must have meaningful CSS and
    # non-trivial content. A bare <h1>Hello</h1> with no styles is never an
    # acceptable deliverable — the user expects a presentable page.
    if style_count < 5:
        return True
    if word_count < 30:
        return True

    return False


def _expose_local_http_service_evidence_is_positive(content: str) -> bool:
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False
    return bool(data.get("exposed") and data.get("connectable") and data.get("url"))


def _filesystem_process_evidence_is_positive(content: str) -> bool:
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False

    for path in data.get("paths", []):
        if path.get("exists"):
            return True
    for pid in data.get("pids", []):
        if pid.get("running"):
            return True
    for process in data.get("process_names", []):
        if process.get("count", 0) > 0:
            return True
    for port in data.get("ports", []):
        if port.get("connectable"):
            return True
    background_log = data.get("background_log") or {}
    return bool(background_log.get("exists") and background_log.get("tail"))


# ---------------------------------------------------------------------------
# Command deduplication helpers
# ---------------------------------------------------------------------------

def _normalize_command(command: Any) -> str:
    """Collapse whitespace so trivially-different spellings compare equal."""
    if not command:
        return ""
    return " ".join(str(command).split())


def _last_host_command(steps: list[ExecutionStep]) -> str:
    """Return the most recent host command (terminal/background) run so far."""
    for step in reversed(steps):
        if step.kind != "tool_result":
            continue
        if step.metadata.get("tool_name") not in _HOST_EXECUTION_TOOLS:
            continue
        arguments = step.metadata.get("arguments") or {}
        return _normalize_command(arguments.get("command"))
    return ""


def _duplicate_command_message(tool_name: str) -> str:
    return (
        f"[skipped] This is identical to the {tool_name} command you just ran; it "
        "was NOT executed again. Its previous result still applies -- do not repeat "
        "it. Move on to the next step, or change the command if you intended "
        "something different."
    )


# Tools whose successful result means host state actually changed -- so an
# identical command run afterwards may legitimately produce a different result
# (e.g. `cargo build` after editing a source file, or `curl` after starting a
# server). Deliberately EXCLUDES the host-execution tools themselves so a trivial
# command like `echo`/`whoami` can never "unblock" its own repetition.
_STATE_CHANGING_TOOLS: frozenset[str] = frozenset(
    {
        "write_text_file",
        "publish_static_site",
        "expose_local_http_service",
        "create_table",
        "write_query",
        # MCP filesystem writes
        "write_file",
        "edit_file",
        "create_directory",
        "move_file",
        "copy_file",
        "delete_file",
    }
)

# Anti-thrash cap: how many times one identical host command may run since the
# last host-state change before further repeats are short-circuited. Re-running
# the same command with nothing changed in between cannot produce a new result;
# past this many attempts the model must change approach instead of spinning.
# A state change resets the count, so a legitimate edit-then-rebuild or
# start-then-poll loop is never blocked.
_MAX_IDENTICAL_COMMAND_RUNS = max(
    1, int(os.getenv("AGENT_MAX_IDENTICAL_COMMAND_RUNS", "3"))
)


def _host_command_runs_since_state_change(
    steps: list[ExecutionStep], command: Any
) -> int:
    """Count prior runs of *command* since the most recent state-changing success.

    Only host-execution results count, and any successful ``_STATE_CHANGING_TOOLS``
    call resets the window, so commands repeated with no intervening host-state
    change are the only ones that accumulate toward the cap.
    """
    target = _normalize_command(command)
    if not target:
        return 0

    start = 0
    for index, step in enumerate(steps):
        if (
            step.kind == "tool_result"
            and not step.metadata.get("is_error")
            and step.metadata.get("tool_name") in _STATE_CHANGING_TOOLS
        ):
            start = index + 1

    count = 0
    for step in steps[start:]:
        if step.kind != "tool_result":
            continue
        if step.metadata.get("tool_name") not in _HOST_EXECUTION_TOOLS:
            continue
        arguments = step.metadata.get("arguments") or {}
        if _normalize_command(arguments.get("command")) == target:
            count += 1
    return count


def _repeated_command_message(tool_name: str, command: Any) -> str:
    return (
        f"[skipped: repeated command] You have already run this exact {tool_name} "
        "command several times in this task with nothing changed in between, so it "
        "was NOT executed again:\n"
        f"  {_normalize_command(command)[:200]}\n"
        "Repeating an identical command cannot produce a new result. If it FAILED "
        "before, it is wrong for THIS host -- re-read the HOST ENVIRONMENT summary "
        "from the start of the session (OS, shell, available package managers) and "
        "switch to a command that fits this platform, or change your approach. If it "
        "already SUCCEEDED, stop re-running it and move on to the next step. Do not "
        "issue this command again."
    )


# ---------------------------------------------------------------------------
# Action-task tool blocking
# ---------------------------------------------------------------------------

def _should_block_tool_for_action_task(
    contract: dict[str, Any] | None,
    messages: list[dict[str, Any]],
    steps: list[ExecutionStep],
    tool_name: str,
) -> bool:
    return (
        tool_name == "delegate_task"
        and contract is not None
        and contract.get("mode") == "execute"
        and not _contract_completion_status(
            contract, messages, steps, contract_required=False
        )["complete"]
    )


def _blocked_action_tool_message(tool_name: str) -> str:
    return (
        f"[{tool_name} blocked] This is a host-changing task and the requested "
        "artifacts have not been verified yet. Use execute_terminal_command or "
        "execute_background_service directly, then verify the result."
    )


def _attempted_tool_names(steps: list[ExecutionStep]) -> list[str]:
    return sorted({
        str(step.metadata.get("tool_name"))
        for step in steps
        if step.kind == "tool_result" and step.metadata.get("tool_name")
    })


# ---------------------------------------------------------------------------
# Instruction builders (contract-only; plan-aware builders live in planning.py)
# ---------------------------------------------------------------------------

def _build_task_contract_instruction() -> str:
    return (
        "Before doing any other work, call set_task_contract for the current user "
        "task. Choose mode='answer' for pure text answers. Choose mode='execute' "
        "when success requires changing or verifying host state. For execute mode, "
        "choose the evidence requirement(s) that would prove completion: "
        "filesystem_artifact, published_static_site_url, running_http_service, "
        "database_mutation, or command_output. Use 'none' only with answer mode."
    )


def _build_incomplete_contract_cap_message(
    original_prompt: str,
    status: dict[str, Any],
    steps: list[ExecutionStep],
) -> str:
    return (
        "**Task paused before completion**\n\n"
        f"I could not verify that this request was completed: `{original_prompt}`.\n\n"
        f"**Missing evidence:** {', '.join(status.get('missing', [])) or 'none'}\n"
        f"**Tools attempted:** {', '.join(_attempted_tool_names(steps)) or 'none'}\n\n"
        "Send `continue` and I will resume from the current contract."
    )
