"""Task-contract system: validation, evidence checking, and status gating.

The contract is the agent's binding commitment declared on the first turn of
every task via ``set_task_contract``.  In execute mode, ``contract_completion_status``
gates the final answer on structured evidence so "I'll create the file now"
with no actual tool call is impossible.

Dependency order: this module imports only from ``evaluator`` (for
``ExecutionStep``), so it can be safely imported by ``planning`` and ``agent``
without creating circular dependencies.

Public API: the module-level names *without* a leading underscore (e.g.
``run_set_task_contract``, ``contract_completion_status``, ``filter_tool_schemas``,
``normalize_command``) are the intentional cross-module interface consumed by
``agent`` and ``planning``. Names that keep a leading underscore are private
helpers internal to this module.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from urllib.parse import urlparse

from evaluator import ExecutionStep
from toolsets import TOOLSET_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_CONTRACT_TOOL = "set_task_contract"

_TASK_CONTRACT_EVIDENCE: frozenset[str] = frozenset(
    {
        "filesystem_artifact",
        "running_http_service",
        "running_tcp_service",
        "database_mutation",
        "command_output",
        "none",
    }
)

HOST_EXECUTION_TOOLS: frozenset[str] = frozenset(
    {"execute_terminal_command", "execute_background_service"}
)

_CONTRACT_CONTROL_TOOLS: frozenset[str] = frozenset(
    {"update_plan", "expand_tool_output"}
)

_RECOVERY_DIAGNOSTIC_TOOLS: frozenset[str] = frozenset(
    {
        "get_system_environment",
        "get_filesystem_process_evidence",
        "web_search",
        "web_fetch",
        "expand_tool_output",
        "wait_for_port",
    }
)

_CONTRACT_EVIDENCE_TOOLS: dict[str, frozenset[str]] = {
    "filesystem_artifact": frozenset(
        {
            "write_text_file",
            "execute_terminal_command",
            "get_filesystem_process_evidence",
        }
    ),
    "running_http_service": frozenset(
        {
            "write_text_file",
            "execute_background_service",
            "execute_terminal_command",
            "get_filesystem_process_evidence",
            "expose_local_http_service",
        }
    ),
    "running_tcp_service": frozenset(
        {
            "execute_background_service",
            "execute_terminal_command",
            "get_filesystem_process_evidence",
            "wait_for_port",
        }
    ),
    "database_mutation": frozenset(
        {"create_table", "write_query", "read_query", "describe_table", "list_tables"}
    ),
    "command_output": frozenset(
        {"execute_terminal_command", "execute_background_service"}
    ),
    "artifact_quality": frozenset(
        {"write_text_file", "browser_navigate", "browser_screenshot"}
    ),
}


def evidence_producing_tools(requirement: str) -> frozenset[str]:
    """Tools whose successful output can satisfy a proof/evidence requirement.

    A node that requires evidence it is structurally forbidden from producing
    can never pass — so callers use this to keep a node's allowed tools and its
    proof requirements consistent.
    """
    return _CONTRACT_EVIDENCE_TOOLS.get(requirement, frozenset())

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
# Minimal plan lookups (needed by contract_completion_status; kept here to
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

def filter_tool_schemas(
    tool_schemas: list[dict[str, Any]], names: set[str]
) -> list[dict[str, Any]]:
    return [
        schema
        for schema in tool_schemas
        if schema.get("function", {}).get("name") in names
    ]


def tool_names_for_contract_status(
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

def run_set_task_contract(arguments: dict[str, Any]) -> tuple[str, bool]:
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


def _sanitize_contract_environment_wording(text: str) -> str:
    """Keep model-authored task contracts from re-scoping sandbox work to the host."""
    replacements = (
        (r"\bon the host system\b", "in the active execution environment"),
        (r"\bon the host\b", "in the active execution environment"),
        (r"\bhost system\b", "active execution environment"),
        (r"\bhost OS\b", "active execution environment"),
        (r"\bhost filesystem\b", "active workspace"),
        (r"\bhost machine\b", "active execution environment"),
    )
    cleaned = text
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


def _normalise_task_contract(
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    mode = arguments.get("mode")
    if mode not in {"answer", "execute"}:
        return {}, "mode must be 'answer' or 'execute'"

    summary = _sanitize_contract_environment_wording(
        " ".join(str(arguments.get("summary", "")).split())
    )
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
        _sanitize_contract_environment_wording(" ".join(str(item).split()))
        for item in raw_criteria
        if str(item).strip()
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
        request_text = " ".join([summary, *success_criteria]).lower()
        http_hints = ("http", "web", "browser", "url", "html", "api", "endpoint", "site")
        if "running_http_service" in evidence and not any(
            hint in request_text for hint in http_hints
        ):
            # Generic services are proven by process/port evidence, not by an
            # HTTP proxy. If the task doesn't mention HTTP/web semantics, correct
            # the over-broad HTTP evidence choice to TCP evidence.
            evidence = [
                "running_tcp_service" if item == "running_http_service" else item
                for item in evidence
            ]
        artifact_hints = (
            "artifact",
            "file",
            "folder",
            "directory",
            "path",
            "download",
            "downloaded",
            "jar",
            "archive",
            "config",
            "configuration",
            "created",
            "written",
            "saved",
            "exists",
        )
        if "filesystem_artifact" not in evidence and any(
            hint in request_text for hint in artifact_hints
        ):
            evidence.append("filesystem_artifact")

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


def latest_task_contract(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for msg in reversed(_current_task_messages(messages)):
        if msg.get("role") != "assistant":
            continue
        for tc in reversed(msg.get("tool_calls") or []):
            try:
                if tc["function"]["name"] != TASK_CONTRACT_TOOL:
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

def contract_completion_status(
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


def can_stream_text_before_final(
    contract: dict[str, Any] | None,
    messages: list[dict[str, Any]],
    steps: list[ExecutionStep],
) -> bool:
    if contract is None:
        return False
    if contract.get("mode") == "answer":
        return True
    return bool(
        contract_completion_status(
            contract, messages, steps, contract_required=False
        )["complete"]
    )


# ---------------------------------------------------------------------------
# Evidence satisfaction checks
# ---------------------------------------------------------------------------

def _evidence_requirement_satisfied(
    requirement: str, steps: list[ExecutionStep]
) -> bool:
    return evidence_requirement_satisfied_by_steps(requirement, steps)


def evidence_requirement_satisfied_by_steps(
    requirement: str,
    steps: list[ExecutionStep],
    *,
    context_steps: list[ExecutionStep] | None = None,
) -> bool:
    context = context_steps or steps
    for step in steps:
        if step.kind != "tool_result" or step.metadata.get("is_error"):
            continue

        if _step_evidence_satisfies_requirement(requirement, step):
            return True
    if requirement in {"running_http_service", "running_tcp_service"}:
        return _http_response_matches_known_service(steps, context)
    return False


def _step_evidence_satisfies_requirement(
    requirement: str, step: ExecutionStep
) -> bool:
    tool_name = step.metadata.get("tool_name")
    if _step_declares_evidence_requirement(requirement, step):
        return True
    if _observation_evidence_satisfies_requirement(requirement, step.content):
        return True
    if requirement == "running_http_service":
        return bool(
            tool_name == "expose_local_http_service"
            and _expose_local_http_service_evidence_is_positive(step.content)
        )
    if requirement == "running_tcp_service":
        if tool_name == "get_filesystem_process_evidence" and _tcp_service_evidence_is_positive(step.content):
            return True
        if tool_name == "wait_for_port" and _wait_for_port_evidence_is_positive(step.content):
            return True
    if requirement == "filesystem_artifact":
        if tool_name == "get_filesystem_process_evidence" and _filesystem_artifact_evidence_is_positive(step.content):
            return True
        if tool_name == "write_text_file" and _write_text_file_evidence_is_positive(step.content):
            return True
    if requirement == "database_mutation":
        return tool_name in {"create_table", "write_query"}
    if requirement == "command_output":
        return _successful_command_output_evidence(str(tool_name), step.content)
    if requirement == "artifact_quality" and tool_name == "write_text_file":
        try:
            data = json.loads(step.content)
        except (TypeError, json.JSONDecodeError):
            return False
        return _artifact_quality_payload_is_acceptable(data)
    return requirement == "none"


def _step_declares_evidence_requirement(
    requirement: str,
    step: ExecutionStep,
) -> bool:
    evidence_types = step.metadata.get("evidence_types") or step.metadata.get(
        "evidence_requirements"
    )
    if not isinstance(evidence_types, list) or requirement not in evidence_types:
        return False
    payload = _step_json_payload(step)
    if payload:
        if payload.get("success") is False or payload.get("ok") is False:
            return False
        status = str(payload.get("status") or "").lower()
        if status in {"error", "failed", "failure", "denied", "blocked"}:
            return False
        if payload.get("error"):
            return False
    return True


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


def _wait_for_port_evidence_is_positive(content: str) -> bool:
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False
    return bool(data.get("open") and data.get("port"))


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
        if step.metadata.get("tool_name") != "write_text_file":
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


_OBSERVATION_TAGS_BY_REQUIREMENT: dict[str, frozenset[str]] = {
    "filesystem_artifact": frozenset({"filesystem_artifact"}),
    "running_http_service": frozenset({"service_exposed"}),
    "running_tcp_service": frozenset(
        {"service_exposed", "tcp_open", "process_running"}
    ),
    "command_output": frozenset({"command_success"}),
}


def _observation_evidence_satisfies_requirement(
    requirement: str, content: str
) -> bool:
    needed = _OBSERVATION_TAGS_BY_REQUIREMENT.get(requirement)
    if not needed:
        return False
    return bool(_evidence_observation_tags(content) & needed)


def _payload_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _evidence_observation_tags(content: str) -> set[str]:
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return set()
    if not isinstance(data, dict):
        return set()

    status = str(data.get("status") or "").lower()
    if status in {"error", "failed", "failure", "denied", "blocked"}:
        return set()

    tags: set[str] = set()

    if data.get("written") and data.get("exists") and _payload_int(data.get("size_bytes")) > 0:
        tags.add("filesystem_artifact")
    if data.get("exposed") and data.get("connectable") and data.get("url"):
        tags.add("service_exposed")
        tags.add("tcp_open")
    if data.get("open") and data.get("port"):
        tags.add("tcp_open")
    if data.get("connectable"):
        tags.add("tcp_open")

    status_code = _payload_int(data.get("status_code"))
    if 200 <= status_code < 400:
        tags.add("http_response")

    if data.get("exit_code") == 0 and str(
        data.get("stdout") or data.get("stderr") or ""
    ).strip():
        tags.add("command_success")

    paths = data.get("paths") or []
    if (
        isinstance(paths, list)
        and paths
        and all(isinstance(path, dict) and path.get("exists") for path in paths)
    ):
        tags.add("filesystem_artifact")
    for pid in data.get("pids") or []:
        if isinstance(pid, dict) and pid.get("running"):
            tags.add("process_running")
            break
    for process in data.get("process_names") or []:
        if isinstance(process, dict) and _payload_int(process.get("count")) > 0:
            tags.add("process_running")
            break
    for port in data.get("ports") or []:
        if isinstance(port, dict) and (port.get("connectable") or port.get("open")):
            tags.add("tcp_open")
            break

    background_log = data.get("background_log") or {}
    if (
        isinstance(background_log, dict)
        and background_log.get("exists")
        and background_log.get("tail")
    ):
        tags.add("process_running")

    return tags


def _step_json_payload(step: ExecutionStep) -> dict[str, Any]:
    try:
        data = json.loads(step.content)
    except (TypeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _step_command(step: ExecutionStep) -> str:
    arguments = step.metadata.get("arguments") or {}
    if isinstance(arguments, dict):
        return normalize_command(arguments.get("command"))
    return ""


def _step_url(step: ExecutionStep) -> str:
    payload = _step_json_payload(step)
    if payload.get("url"):
        return str(payload.get("url") or "")
    arguments = step.metadata.get("arguments") or {}
    if isinstance(arguments, dict):
        return str(arguments.get("url") or "")
    return ""


def _url_port(url: str) -> int | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "http":
        return 80
    if parsed.scheme == "https":
        return 443
    return None


def _url_is_local(url: str) -> bool:
    if not url:
        return False
    try:
        hostname = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return bool(
        hostname == "localhost"
        or hostname == "0.0.0.0"
        or hostname == "::1"
        or hostname.startswith("127.")
    )


def _url_matches_prefix(url: str, expected: str) -> bool:
    if not url or not expected:
        return False
    normalized = url.rstrip("/") + "/"
    expected_normalized = expected.rstrip("/") + "/"
    return normalized.startswith(expected_normalized) or expected_normalized.startswith(normalized)


def _command_ports(command: str) -> set[int]:
    ports: set[int] = set()
    patterns = (
        r"(?:^|\s)(?:--port|-p)\s*=?\s*(\d{2,5})(?:\s|$)",
        r"(?:^|\s)PORT=(\d{2,5})(?:\s|$)",
        r"\bhttp\.server\s+(\d{2,5})\b",
        r"\b(?:localhost|127\.0\.0\.1|0\.0\.0\.0):(\d{2,5})\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, command, flags=re.IGNORECASE):
            port = _payload_int(match.group(1))
            if 1 <= port <= 65535:
                ports.add(port)
    return ports


def _known_service_ports(steps: list[ExecutionStep]) -> set[int]:
    ports: set[int] = set()
    for step in steps:
        if step.kind != "tool_result" or step.metadata.get("is_error"):
            continue
        payload = _step_json_payload(step)
        status = str(payload.get("status") or "").lower()
        if status in {"error", "failed", "failure", "denied", "blocked"}:
            continue
        for key in ("port",):
            port = _payload_int(payload.get(key))
            if 1 <= port <= 65535 and (
                payload.get("open")
                or payload.get("connectable")
                or payload.get("exposed")
                or status in {"launched", "running", "started", "ok", "success"}
            ):
                ports.add(port)
        for port_info in payload.get("ports") or []:
            if isinstance(port_info, dict) and (port_info.get("connectable") or port_info.get("open")):
                port = _payload_int(port_info.get("port"))
                if 1 <= port <= 65535:
                    ports.add(port)
        if step.metadata.get("tool_name") == "execute_background_service" and status not in {
            "error",
            "failed",
        }:
            ports.update(_command_ports(_step_command(step)))
    return ports


def _known_service_urls(steps: list[ExecutionStep]) -> set[str]:
    urls: set[str] = set()
    for step in steps:
        if step.kind != "tool_result" or step.metadata.get("is_error"):
            continue
        payload = _step_json_payload(step)
        if payload.get("exposed") and payload.get("connectable") and payload.get("url"):
            urls.add(str(payload["url"]))
    return urls


def _http_response_matches_known_service(
    candidate_steps: list[ExecutionStep],
    context_steps: list[ExecutionStep],
) -> bool:
    known_ports = _known_service_ports(context_steps)
    known_urls = _known_service_urls(context_steps)
    for step in candidate_steps:
        if step.kind != "tool_result" or step.metadata.get("is_error"):
            continue
        payload = _step_json_payload(step)
        status_code = _payload_int(payload.get("status_code"))
        if status_code < 200 or status_code >= 400:
            continue
        url = _step_url(step)
        if any(_url_matches_prefix(url, known_url) for known_url in known_urls):
            return True
        port = _url_port(url)
        if port is not None and port in known_ports and _url_is_local(url):
            return True
    return False


def _filesystem_artifact_evidence_is_positive(content: str) -> bool:
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False

    paths = data.get("paths", [])
    if not isinstance(paths, list) or not paths:
        return False
    path_entries = [path for path in paths if isinstance(path, dict)]
    return bool(path_entries) and all(bool(path.get("exists")) for path in path_entries)


def _tcp_service_evidence_is_positive(content: str) -> bool:
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False

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


def _filesystem_process_evidence_is_positive(content: str) -> bool:
    """Backward-compatible broad evidence check for older tests/callers."""
    return _filesystem_artifact_evidence_is_positive(content) or _tcp_service_evidence_is_positive(content)


# ---------------------------------------------------------------------------
# Command deduplication helpers
# ---------------------------------------------------------------------------

def normalize_command(command: Any) -> str:
    """Collapse whitespace so trivially-different spellings compare equal."""
    if not command:
        return ""
    return " ".join(str(command).split())


_LONG_RUNNING_SERVICE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bpython(?:\d+(?:\.\d+)?)?\s+-m\s+http\.server\b",
        r"\buvicorn\b",
        r"\bgunicorn\b",
        r"\bhypercorn\b",
        r"\bflask\s+run\b",
        r"\bstreamlit\s+run\b",
        r"\b(?:npm|pnpm|yarn)\s+(?:run\s+)?dev\b",
        r"\b(?:npm|pnpm|yarn)\s+start\b",
        r"\b(?:vite|next\s+dev|astro\s+dev|remix\s+dev)\b",
        r"\b(?:serve|http-server)\b",
        (
            r"\b(?:node|python(?:\d+(?:\.\d+)?)?)\s+[^;&|]*"
            r"(?:server|app|main)\.(?:js|mjs|cjs|ts|py)\b"
        ),
    )
)

_FINITE_BACKGROUND_PROGRAMS: frozenset[str] = frozenset(
    {
        "awk",
        "cat",
        "curl",
        "df",
        "du",
        "echo",
        "false",
        "find",
        "grep",
        "head",
        "hostname",
        "ifconfig",
        "ip",
        "lsof",
        "ls",
        "netstat",
        "pgrep",
        "ps",
        "pwd",
        "sed",
        "ss",
        "stat",
        "tail",
        "test",
        "true",
        "wc",
        "wget",
        "which",
        "where",
    }
)

_FINITE_NPM_COMMANDS: frozenset[str] = frozenset(
    {"audit", "build", "ci", "install", "lint", "pack", "publish", "test"}
)


def _looks_like_long_running_service_command(command: str) -> bool:
    return any(pattern.search(command) for pattern in _LONG_RUNNING_SERVICE_PATTERNS)


def _split_shell_segments(command: str) -> list[str]:
    return [
        segment.strip()
        for segment in re.split(r"\s*(?:&&|\|\||[|;])\s*", command)
        if segment.strip()
    ]


def _command_words(segment: str) -> list[str]:
    cleaned = segment.strip().strip("()")
    if not cleaned:
        return []
    return cleaned.split()


def _program_basename(token: str) -> str:
    token = token.strip("\"'").rstrip(":")
    token = token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return token.lower()


def _first_effective_program(words: list[str]) -> tuple[str, list[str]]:
    index = 0
    while index < len(words):
        word = words[index]
        lowered = _program_basename(word)
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", word):
            index += 1
            continue
        if lowered in {"sudo", "command", "env", "time", "nohup"}:
            index += 1
            continue
        if lowered == "timeout":
            index += 2 if index + 1 < len(words) else 1
            continue
        return lowered, words[index + 1 :]
    return "", []


def _python_background_invocation_is_finite(program: str, args: list[str]) -> bool:
    if not re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", program):
        return False
    if not args:
        return True
    first = args[0].lower()
    if first in {"-c", "-v", "-vv", "-h", "--help", "--version"}:
        return True
    if first == "-m" and len(args) > 1:
        module = args[1].lower()
        return module in {"pip", "pytest", "unittest", "compileall", "venv"}
    return False


def _node_background_invocation_is_finite(program: str, args: list[str]) -> bool:
    if program != "node":
        return False
    if not args:
        return True
    return args[0].lower() in {"-e", "-p", "--eval", "--print", "-v", "--version"}


def _npm_background_invocation_is_finite(program: str, args: list[str]) -> bool:
    if program not in {"npm", "pnpm", "yarn", "npx"} or not args:
        return False
    first = args[0].lower()
    if first == "run" and len(args) > 1:
        return args[1].lower() in _FINITE_NPM_COMMANDS
    return first in _FINITE_NPM_COMMANDS


def looks_like_finite_background_command(command: Any) -> bool:
    """Return True when a background-service command is clearly finite/probing.

    The classifier is intentionally conservative: unknown commands are allowed
    because they may be custom daemons, while common one-shot diagnostics and
    build/test/install commands are blocked from ``execute_background_service``.
    """
    normalized = normalize_command(command)
    if not normalized:
        return True
    if _looks_like_long_running_service_command(normalized):
        return False

    for segment in _split_shell_segments(normalized):
        program, args = _first_effective_program(_command_words(segment))
        if not program:
            continue
        if program in _FINITE_BACKGROUND_PROGRAMS:
            return True
        if _python_background_invocation_is_finite(program, args):
            return True
        if _node_background_invocation_is_finite(program, args):
            return True
        if _npm_background_invocation_is_finite(program, args):
            return True
    return False


def background_service_misuse_message(command: Any) -> str | None:
    if not looks_like_finite_background_command(command):
        return None
    return (
        "[execute_background_service blocked: finite/probe command] This command "
        "looks like a diagnostic, build, test, install, or other finite command, "
        "so it was NOT started as a background service:\n"
        f"  {normalize_command(command)[:200]}\n"
        "Use execute_terminal_command for commands that should finish. Reserve "
        "execute_background_service only for non-terminating servers, daemons, "
        "watchers, and development servers."
    )


def last_host_command(steps: list[ExecutionStep]) -> str:
    """Return the most recent host command (terminal/background) run so far."""
    for step in reversed(steps):
        if step.kind != "tool_result":
            continue
        if step.metadata.get("tool_name") not in HOST_EXECUTION_TOOLS:
            continue
        arguments = step.metadata.get("arguments") or {}
        return normalize_command(arguments.get("command"))
    return ""


def duplicate_command_message(tool_name: str) -> str:
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


def terminal_command_declares_state_change(arguments: dict[str, Any] | None) -> bool:
    """Whether a terminal command was explicitly marked as changing host state."""
    if not isinstance(arguments, dict):
        return False
    return arguments.get("changes_state") is True


def tool_call_changes_state(tool_name: str, arguments: dict[str, Any] | None) -> bool:
    """Whether a tool call should reset no-progress verification counters."""
    if tool_name == "execute_terminal_command":
        return terminal_command_declares_state_change(arguments)
    return tool_name in _STATE_CHANGING_TOOLS or tool_name == "execute_background_service"


# Anti-thrash cap: how many times one identical host command may run since the
# last host-state change before further repeats are short-circuited. Re-running
# the same command with nothing changed in between cannot produce a new result;
# past this many attempts the model must change approach instead of spinning.
# A state change resets the count, so a legitimate edit-then-rebuild or
# start-then-poll loop is never blocked.
MAX_IDENTICAL_COMMAND_RUNS = max(
    1, int(os.getenv("AGENT_MAX_IDENTICAL_COMMAND_RUNS", "3"))
)

MAX_TERMINAL_FAILURES_BEFORE_DIAGNOSTIC = max(
    2, int(os.getenv("AGENT_MAX_TERMINAL_FAILURES_BEFORE_DIAGNOSTIC", "2"))
)


def host_command_runs_since_state_change(
    steps: list[ExecutionStep], command: Any
) -> int:
    """Count prior runs of *command* since the most recent state-changing success.

    Only host-execution results count, and any successful ``_STATE_CHANGING_TOOLS``
    call resets the window, so commands repeated with no intervening host-state
    change are the only ones that accumulate toward the cap.
    """
    target = normalize_command(command)
    if not target:
        return 0

    start = 0
    for index, step in enumerate(steps):
        if (
            step.kind == "tool_result"
            and not step.metadata.get("is_error")
            and (
                step.metadata.get("changes_state") is True
                or step.metadata.get("tool_name") in _STATE_CHANGING_TOOLS
            )
        ):
            start = index + 1

    count = 0
    for step in steps[start:]:
        if step.kind != "tool_result":
            continue
        if step.metadata.get("tool_name") not in HOST_EXECUTION_TOOLS:
            continue
        arguments = step.metadata.get("arguments") or {}
        if normalize_command(arguments.get("command")) == target:
            count += 1
    return count


def repeated_command_message(tool_name: str, command: Any) -> str:
    return (
        f"[skipped: repeated command] You have already run this exact {tool_name} "
        "command several times in this task with nothing changed in between, so it "
        "was NOT executed again:\n"
        f"  {normalize_command(command)[:200]}\n"
        "Repeating an identical command cannot produce a new result. If it FAILED "
        "before, it is wrong for the active execution environment -- re-read the "
        "execution-environment summary from the start of the session (OS, shell, "
        "available package managers) and switch to a command that fits this platform, "
        "or change your approach. If it already SUCCEEDED, stop re-running it and "
        "move on to the next step. Do not issue this command again."
    )


def terminal_failure_since_diagnostic(steps: list[ExecutionStep]) -> bool:
    """True after repeated terminal failures without a diagnostic step.

    One failed command can be corrected directly. After repeated failures, force
    a non-terminal diagnostic step to prevent platform-guessing loops like
    java/java -version/which/cmd/powershell variants.
    """
    consecutive_failures = 0
    for step in reversed(steps):
        if step.kind != "tool_result":
            continue
        tool_name = step.metadata.get("tool_name")
        if tool_name in _RECOVERY_DIAGNOSTIC_TOOLS:
            return False
        if (
            not step.metadata.get("is_error")
            and (
                tool_name in _STATE_CHANGING_TOOLS
                or step.metadata.get("changes_state") is True
            )
        ):
            return False
        if tool_name == "execute_terminal_command":
            if step.metadata.get("is_error"):
                consecutive_failures += 1
                if consecutive_failures >= MAX_TERMINAL_FAILURES_BEFORE_DIAGNOSTIC:
                    return True
                continue
            return False
    return False


def terminal_failure_recovery_message(command: Any) -> str:
    return (
        "[blocked: repeated terminal failures] Several terminal commands failed "
        "without a diagnostic step, so this new terminal command was NOT executed:\n"
        f"  {normalize_command(command)[:200]}\n"
        "First diagnose the failure with a non-terminal tool: call "
        "get_system_environment to confirm OS/shell/runtimes, "
        "get_filesystem_process_evidence to inspect actual files/processes/ports, "
        "expand_tool_output for truncated logs, or web_search/web_fetch for an "
        "unfamiliar error. Then choose a different command that matches the active "
        "execution environment. Do not keep trying shell/path variants."
    )


# ---------------------------------------------------------------------------
# Generic identical-tool-call anti-spin guard
#
# Applies to EVERY tool (not just host commands or port checks): re-issuing the
# same tool with the same arguments, with no host/world state change in between,
# cannot return new information — it just burns API calls and rate-limit budget.
# The host-command guard above is the special case for terminal/background
# commands; this covers all the read/evidence/research tools (filesystem &
# process checks, web_fetch/web_search, get_system_environment, db reads, etc.).
# ---------------------------------------------------------------------------

# Tools exempt from the generic cap: repeating them with identical args is
# legitimate paging / internal blocking / one-shot control rather than a spin.
_REPEAT_CAP_EXEMPT_TOOLS: frozenset[str] = frozenset(
    {
        "expand_tool_output",  # deliberate paging through a large output handle
        "set_task_contract",   # at most once per task anyway
        "update_plan",         # plan-loop control; capping risks a contract deadlock
        "wait_for_port",       # blocks internally; a repeat is a deliberate longer wait
        "delegate_task",       # sub-agent dispatch; each run can differ
    }
)

# Host-execution tools have their own command-signature guard
# (host_command_runs_since_state_change); exclude them here to avoid double
# counting and to preserve that path's specialised escalation messaging.
GENERIC_CAP_SKIP_TOOLS: frozenset[str] = _REPEAT_CAP_EXEMPT_TOOLS | HOST_EXECUTION_TOOLS

# How many times one tool may be called with identical arguments since the last
# state change before further identical calls are short-circuited. Env-tunable.
MAX_IDENTICAL_TOOL_CALLS = max(
    1, int(os.getenv("AGENT_MAX_IDENTICAL_TOOL_CALLS", "3"))
)

# Maximum number of consecutive execute_terminal_command calls allowed without
# any other tool type (evidence check, file write, etc.) in between.
#
# A long unbroken run of terminal-only calls — varied or identical — means the
# agent is trying many command variants without pausing to inspect state. The
# identical-call guard above catches *exact* repeats; this cap catches the
# harder "varied spin" pattern where each command is slightly different but the
# agent is still not making progress. Tunable via env var.
MAX_CONSECUTIVE_TERMINAL_COMMANDS = max(
    4, int(os.getenv("AGENT_MAX_CONSECUTIVE_TERMINAL_COMMANDS", "8"))
)


def _tool_call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    """Stable signature for a (tool, args) pair so identical calls compare equal."""
    try:
        args_repr = json.dumps(arguments or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        args_repr = str(arguments)
    return f"{tool_name}::{args_repr}"


def _is_state_change_reset(step: ExecutionStep) -> bool:
    """True if *step* is a successful state-changing event that should reset the
    identical-call window (a write, db mutation, or background-service launch).

    A changed world means a previously-identical read can now legitimately
    return something new, so the counter restarts after such an event.
    """
    if step.kind != "tool_result" or step.metadata.get("is_error"):
        return False
    if step.metadata.get("changes_state") is True:
        return True
    name = str(step.metadata.get("tool_name") or "")
    arguments = step.metadata.get("arguments") or {}
    return tool_call_changes_state(name, arguments)


def identical_tool_call_runs_since_state_change(
    steps: list[ExecutionStep],
    tool_name: str,
    arguments: dict[str, Any],
) -> int:
    """Count prior calls of *tool_name* with identical *arguments* since the most
    recent state-changing success.

    Generalises the host-command anti-thrash guard to every tool. Any successful
    write / db mutation / service launch resets the window, so only calls
    repeated with no intervening state change accumulate toward the cap.
    """
    signature = _tool_call_signature(tool_name, arguments)

    start = 0
    for index, step in enumerate(steps):
        if _is_state_change_reset(step):
            start = index + 1

    count = 0
    for step in steps[start:]:
        if step.kind != "tool_result":
            continue
        if step.metadata.get("tool_name") != tool_name:
            continue
        if _tool_call_signature(tool_name, step.metadata.get("arguments") or {}) == signature:
            count += 1
    return count


def repeated_tool_call_message(tool_name: str, arguments: dict[str, Any]) -> str:
    """Block message for a repeated identical tool call.

    The redirect is keyed off the *resource being polled* (the ``ports``
    argument), not off any single tool name, so it applies to every tool that
    can poll a port. Anything else gets the general "wait once / fix the cause"
    guidance below.
    """
    args = arguments or {}
    # A repeated port check is the most common poll-loop and it has a dedicated
    # blocking primitive. Trigger on the argument, regardless of which tool.
    ports = args.get("ports")
    if ports:
        port_list = ", ".join(str(p) for p in ports)
        return (
            f"[skipped: poll loop] You have checked port(s) {port_list} repeatedly with "
            "no change, so this was NOT run again. Polling burns API calls and the "
            "rate-limit budget. Call wait_for_port instead — it blocks internally "
            "until the port opens (or times out) in a single turn. If the service "
            "failed to start, read the background-service log first (the start "
            "call returned its exact path)."
        )
    return (
        f"[skipped: repeated call] You have already called {tool_name} with these "
        "exact arguments several times with nothing changing in between, so it was "
        "NOT run again — an identical call cannot return new information. Use the "
        "result you already have and move on. If you are waiting for something to "
        "finish, wait ONCE (wait_for_port for a service port, or a single blocking "
        "command) instead of polling; if a previous step failed, fix the underlying "
        "cause rather than re-checking the same thing."
    )


# ---------------------------------------------------------------------------
# Consecutive terminal-command run-length guard
#
# The identical-call guard (above) catches EXACT repeats. This guard catches
# the complementary "varied spin" pattern: many DIFFERENT commands in a row
# with no evidence check, file write, or other tool type in between. Any task
# where the agent runs N consecutive execute_terminal_command calls without
# pausing to inspect state or create an artifact is almost certainly stuck in
# a trial-and-error loop that will hit rate limits before making real progress.
# ---------------------------------------------------------------------------

def consecutive_terminal_run_length(steps: list[ExecutionStep]) -> int:
    """Count how many consecutive ``execute_terminal_command`` tool results appear
    at the tail of *steps* with no other tool type in between.

    Only ``execute_terminal_command`` is counted (not ``execute_background_service``
    — that is a one-shot service launch, never a spin target). ``tool_result``
    steps from other tools break the streak.
    """
    count = 0
    for step in reversed(steps):
        if step.kind != "tool_result":
            continue
        if step.metadata.get("tool_name") == "execute_terminal_command":
            count += 1
        else:
            break
    return count


def consecutive_terminal_cap_message(run_length: int) -> str:
    """Block message for a consecutive terminal-command run that exceeded the cap."""
    return (
        f"[blocked: command-only run ({run_length})] You have issued {run_length} "
        "execute_terminal_command calls in a row without any other tool call in "
        "between. This almost always means a trial-and-error loop: each new command "
        "variant is guesswork, not a reasoned next step. "
        "STOP and do ONE of the following before running another command:\n"
        "  1. Call get_system_environment if you have not confirmed the OS, shell, "
        "     and available runtimes — never guess the platform.\n"
        "  2. Call get_filesystem_process_evidence to verify what actually exists "
        "     on disk or is running (do not assume a previous command succeeded).\n"
        "  3. Read any error output you already have and reason about the root cause "
        "     before choosing a new approach — do not just rephrase the same command.\n"
        "Once you have inspected state and know what to do, you may run commands again."
    )


# ---------------------------------------------------------------------------
# Consecutive verification-tool cycling guard
#
# Catches the pattern where the agent cycles between DIFFERENT verification
# tools (expose -> web_fetch -> wait_for_port -> expose ...) without any
# state-changing tool in between.  The identical-call guard catches exact
# repeats; the terminal-run guard catches terminal-only runs; this covers
# the remaining gap: read-only verification cycling.
# ---------------------------------------------------------------------------

VERIFICATION_TOOLS: frozenset[str] = frozenset(
    {
        "expose_local_http_service",
        "wait_for_port",
        "get_filesystem_process_evidence",
        "web_fetch",
        "browser_navigate",
        "browser_screenshot",
        "browser_get_text",
    }
)


def tool_call_is_verification_probe(
    tool_name: str,
    arguments: dict[str, Any] | None,
) -> bool:
    """Whether a call observes prior work rather than creating new durable state."""
    if tool_name == "execute_terminal_command":
        return not terminal_command_declares_state_change(arguments)
    return tool_name in VERIFICATION_TOOLS


MAX_CONSECUTIVE_VERIFICATION_CALLS = max(
    4, int(os.getenv("AGENT_MAX_CONSECUTIVE_VERIFICATION_CALLS", "6"))
)


def consecutive_verification_run_length(steps: list[ExecutionStep]) -> int:
    """Count consecutive verification-only tool results at the tail of *steps*.

    A "verification tool" is any read-only tool whose purpose is checking
    whether prior state-changing work succeeded (port open? file exists?
    page loads?).  State-changing tools (writes, commands, service launches)
    break the streak.
    """
    count = 0
    for step in reversed(steps):
        if step.kind != "tool_result":
            continue
        name = str(step.metadata.get("tool_name") or "")
        arguments = step.metadata.get("arguments") or {}
        if tool_call_is_verification_probe(name, arguments):
            count += 1
        else:
            break
    return count


def consecutive_verification_cap_message(run_length: int) -> str:
    """Block message for a verification-only run that exceeded the cap."""
    return (
        f"[blocked: verification loop ({run_length})] You have called {run_length} "
        "verification/evidence tools in a row without any state-changing tool "
        "(write_text_file, execute_terminal_command, execute_background_service) "
        "in between. Repeating verification checks cannot make a failing service "
        "start working. STOP and do ONE of the following:\n"
        "  1. Read the background-service log (cat /tmp/background_task.log or the "
        "     path from the execute_background_service result) to see if the "
        "     service crashed or failed to start.\n"
        "  2. Call get_system_environment to confirm the runtime is available.\n"
        "  3. Kill the old process and re-launch the service with a corrected "
        "     command via execute_background_service.\n"
        "  4. If the service is running but verification probes fail due to "
        "     networking, update the task graph/plan to mark verification as "
        "     best-effort and close out the task.\n"
        "Do not call another verification tool until you have taken a "
        "state-changing action."
    )


# ---------------------------------------------------------------------------
# Action-task tool blocking
# ---------------------------------------------------------------------------

def should_block_tool_for_action_task(
    contract: dict[str, Any] | None,
    messages: list[dict[str, Any]],
    steps: list[ExecutionStep],
    tool_name: str,
) -> bool:
    return (
        tool_name == "delegate_task"
        and contract is not None
        and contract.get("mode") == "execute"
        and not contract_completion_status(
            contract, messages, steps, contract_required=False
        )["complete"]
    )


def blocked_action_tool_message(tool_name: str) -> str:
    return (
        f"[{tool_name} blocked] This is an environment-changing task and the requested "
        "artifacts have not been verified yet. Use execute_terminal_command or "
        "execute_background_service directly, then verify the result."
    )


def attempted_tool_names(steps: list[ExecutionStep]) -> list[str]:
    return sorted({
        str(step.metadata.get("tool_name"))
        for step in steps
        if step.kind == "tool_result" and step.metadata.get("tool_name")
    })


# ---------------------------------------------------------------------------
# Instruction builders (contract-only; plan-aware builders live in planning.py)
# ---------------------------------------------------------------------------

def build_task_contract_instruction() -> str:
    return (
        "Before doing any other work, call set_task_contract for the current user "
        "task. Choose mode='answer' for pure text answers. Choose mode='execute' "
        "when success requires changing or verifying the active tool environment. "
        "Do not claim a host target unless the user explicitly asked to bypass the "
        "sandbox and host fallback is enabled. For execute mode, "
        "choose the evidence requirement(s) that would prove completion: "
        "filesystem_artifact, running_http_service, "
        "running_tcp_service, database_mutation, or command_output. Use "
        "'running_http_service' only for HTTP servers that can be exposed through "
        "the proxy; use 'running_tcp_service' for non-HTTP services such as game "
        "servers, databases, queues, SSH-like daemons, or any process whose proof "
        "is a listening TCP port. Use 'none' only with answer mode."
    )


def build_incomplete_contract_cap_message(
    original_prompt: str,
    status: dict[str, Any],
    steps: list[ExecutionStep],
) -> str:
    return (
        "**Task paused before completion**\n\n"
        f"I could not verify that this request was completed: `{original_prompt}`.\n\n"
        f"**Missing evidence:** {', '.join(status.get('missing', [])) or 'none'}\n"
        f"**Tools attempted:** {', '.join(attempted_tool_names(steps)) or 'none'}\n\n"
        "Send `continue` and I will resume from the current contract."
    )
