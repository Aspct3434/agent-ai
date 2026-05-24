"""Plan management, executive summary, and cross-domain instruction builders.

Sits one layer above contract.py: it imports the plan-lookup helpers from
there so _contract_completion_status can reference plan state without creating
a circular import, and adds the richer rendering and summary logic that needs
both plan and contract data.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from contract import (
    _attempted_tool_names,
    _is_continuation_signal,
    _latest_artifact_quality,
    _latest_plan,
)
from evaluator import _SIDE_EFFECT_TOOLS, ExecutionStep

logger = logging.getLogger(__name__)

_LEDGER_MAX_ITEMS = int(os.getenv("AGENT_LEDGER_MAX_ITEMS", "12"))
_PLAN_MAX_RENDER = int(os.getenv("AGENT_PLAN_MAX_RENDER", "25"))
_MAX_HISTORY_MESSAGES = int(os.getenv("AGENT_MAX_HISTORY_MESSAGES", "40"))
_KEEP_RECENT_MESSAGES = int(os.getenv("AGENT_KEEP_RECENT_MESSAGES", "16"))

_PLAN_STATUS_ICON = {
    "done": "[x]",
    "in_progress": "[~]",
    "pending": "[ ]",
    "failed": "[!]",
}

# ---------------------------------------------------------------------------
# Plan management
# ---------------------------------------------------------------------------


def _run_update_plan(arguments: dict[str, Any]) -> tuple[str, bool]:
    """Execute the ``update_plan`` builtin. Returns ``(content, is_error)``.

    The plan content itself is carried by the tool-call arguments already stored
    in history; this just validates the input and echoes a status line so the
    model gets confirmation of what it set.
    """
    from contract import _plan_steps_from_args  # local import avoids name clash

    steps = _plan_steps_from_args(arguments)
    if not steps:
        return (
            "[update_plan error] Provide a non-empty 'steps' array, each item "
            "{title, status} with status one of pending/in_progress/done/failed.",
            True,
        )
    counts = {"done": 0, "in_progress": 0, "pending": 0, "failed": 0}
    for step in steps:
        status = step.get("status") if isinstance(step, dict) else None
        if status in counts:
            counts[status] += 1
    total = len(steps)
    open_left = counts["pending"] + counts["in_progress"]
    tail = (
        "All steps are done/failed -- you may give your final answer."
        if open_left == 0
        else (
            "Now execute the next not-done step. "
            "Do NOT call update_plan again in this same turn — proceed "
            "immediately with write_text_file, execute_terminal_command, "
            "execute_background_service, or expose_local_http_service to do the actual work."
        )
    )
    return (
        f"Plan updated: {total} step(s) -- {counts['done']} done, "
        f"{counts['in_progress']} in progress, {counts['pending']} pending, "
        f"{counts['failed']} failed. {tail}",
        False,
    )


def _count_done_plan_steps(messages: list[dict[str, Any]]) -> int:
    """Number of steps marked 'done' in the current plan (0 if no plan)."""
    plan = _latest_plan(messages) or []
    return sum(
        1 for step in plan if isinstance(step, dict) and step.get("status") == "done"
    )


def _count_successful_side_effects(steps: list[ExecutionStep]) -> int:
    """Number of successful host-changing tool results recorded in *steps*."""
    return sum(
        1
        for step in steps
        if step.kind == "tool_result"
        and not step.metadata.get("is_error")
        and step.metadata.get("tool_name") in _SIDE_EFFECT_TOOLS
    )


def _render_plan(plan: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for step in plan[:_PLAN_MAX_RENDER]:
        if not isinstance(step, dict):
            continue
        icon = _PLAN_STATUS_ICON.get(step.get("status", "pending"), "[ ]")
        title = " ".join(str(step.get("title", "")).split())[:120]
        lines.append(f"  {icon} {title}")
    if len(plan) > _PLAN_MAX_RENDER:
        lines.append(f"  ... (+{len(plan) - _PLAN_MAX_RENDER} more steps)")
    return "\n".join(lines) or "  (empty)"


def _build_plan_continuation_instruction(messages: list[dict[str, Any]]) -> str:
    plan = _latest_plan(messages) or []
    return (
        "Your plan still has open steps:\n"
        f"{_render_plan(plan)}\n"
        "Continue now with tool calls: execute the next not-done step, and call "
        "update_plan to mark steps done/failed as you go. Do not give a final answer "
        "until every step is done or failed."
    )


# ---------------------------------------------------------------------------
# Cross-domain instruction builders (need both contract and plan state)
# ---------------------------------------------------------------------------

def _successful_tool_names(steps: list[ExecutionStep]) -> set[str]:
    """Names of tools that have produced at least one non-error result."""
    return {
        str(step.metadata.get("tool_name"))
        for step in steps
        if step.kind == "tool_result"
        and step.metadata.get("tool_name")
        and not step.metadata.get("is_error")
    }


def _build_quality_rejection_detail(steps: list[ExecutionStep]) -> str:
    """Return a human-readable summary of why the artifact quality gate failed."""
    quality = _latest_artifact_quality(steps)
    if not quality:
        return "Quality metrics: unavailable (no artifact quality data found)."
    lines = ["Quality metrics from the last artifact:"]
    if quality.get("placeholder_detected"):
        matches = quality.get("placeholder_matches", [])
        lines.append(f"  PLACEHOLDER TEXT DETECTED: {matches[:4]}")
    lines.append(f"  CSS rules: {quality.get('style_rule_count', 0)} (minimum 5 required)")
    lines.append(f"  Content words: {quality.get('content_word_count', 0)} (minimum 30 required)")
    lines.append(f"  Interactive signals: {quality.get('interactive_signal_count', 0)}")
    return "\n".join(lines)


def _build_contract_execution_instruction(
    contract: dict[str, Any],
    status: dict[str, Any],
    messages: list[dict[str, Any]],
    steps: list[ExecutionStep] | None = None,
) -> str:
    _steps = steps or []
    attempted = set(_attempted_tool_names(_steps))     # all attempts (inc. errors)
    succeeded = _successful_tool_names(_steps)          # only successful results
    plan = _latest_plan(messages)
    plan_guidance = (
        "No update_plan checklist exists yet; call update_plan before continuing."
        if plan is None
        else "Your update_plan checklist still has open steps; execute the work now, then update it when done."
        if status.get("plan_open")
        else "Your update_plan checklist is closed."
    )
    missing = set(status.get("missing", []))
    if "plan" in missing:
        next_action = "Call update_plan now with the concrete checklist of steps."
    else:
        evidence_missing = [
            item for item in status.get("missing", [])
            if item not in {"plan", "plan_open_steps"}
        ]
        if evidence_missing:
            if "artifact_quality" in evidence_missing:
                quality_detail = _build_quality_rejection_detail(_steps)
                next_action = (
                    "The artifact exists, but quality validation FAILED.\n"
                    f"{quality_detail}\n"
                    "Call write_text_file NOW and REPLACE the file with a COMPLETE, "
                    "production-quality deliverable. The file must include:\n"
                    "  • A full CSS design: colour palette, typography, spacing, layout\n"
                    "  • 100+ words of real, substantive content (not placeholders)\n"
                    "  • Visual polish: backgrounds, shadows, rounded corners, transitions\n"
                    "  • Working JavaScript if interaction was requested\n"
                    "Write the ENTIRE file content in one write_text_file call — do not "
                    "hold back or truncate. Then verify the file or serve it if access is required."
                )
            elif "running_tcp_service" in evidence_missing:
                next_action = (
                    "A non-HTTP TCP service still needs proof. First read the latest "
                    "evidence: if it shows a missing runtime, missing binary/JAR, "
                    "missing config, or an empty port/process list, repair that with "
                    "execute_terminal_command. Only after prerequisites exist should "
                    "you use execute_background_service once, then wait_for_port for "
                    "the target port. Do NOT call expose_local_http_service unless "
                    "the service is actually HTTP."
                )
            else:
                next_action = (
                    "The available tools have been narrowed to tools that can produce "
                    "the missing structured evidence. Call one of those tools now. "
                    "Use write_text_file for concrete text artifacts, execute/background "
                    "tools for runtime work, expose_local_http_service for HTTP access, "
                    "and get_filesystem_process_evidence for verification. Do not call "
                    "update_plan again until after the missing evidence is produced."
                )
        elif "plan_open_steps" in missing:
            next_action = (
                "All structured evidence is present; call update_plan now to close "
                "the remaining open steps based on that evidence."
            )
        else:
            next_action = "Continue with the next required tool call."
    return (
        "TASK CONTRACT STILL OPEN:\n"
        f"Mode: {contract.get('mode')}\n"
        f"Summary: {contract.get('summary')}\n"
        f"Success criteria: {json.dumps(contract.get('success_criteria', []))}\n"
        f"Missing evidence: {', '.join(status.get('missing', [])) or 'none'}\n"
        f"Tools called so far: {', '.join(sorted(attempted)) or 'none'}\n"
        f"{plan_guidance}\n"
        f"{next_action}\n"
        "Do not emit final plain text until the contract is complete."
    )


def _build_contract_continuation_instruction(
    contract: dict[str, Any] | None,
    status: dict[str, Any],
    final_response: str,
    messages: list[dict[str, Any]],
    steps: list[ExecutionStep],
) -> str:
    if contract is None:
        return (
            "The previous assistant text was rejected because no task contract was "
            "set for the current user request. Call set_task_contract now, then "
            "continue according to that contract. Do not repeat the rejected text.\n"
            f"Rejected text: {final_response[:500]}"
        )
    return (
        "The previous assistant text was rejected because the task contract is not "
        "complete.\n"
        f"Contract summary: {contract.get('summary')}\n"
        f"Missing evidence: {', '.join(status.get('missing', [])) or 'none'}\n"
        f"Open plan: {status.get('plan_open')}\n"
        f"Tools attempted: {', '.join(_attempted_tool_names(steps)) or 'none'}\n"
        f"Current plan:\n{_render_plan(_latest_plan(messages) or [])}\n"
        f"Rejected text: {final_response[:500]}\n\n"
        "Continue with tool calls that close the plan and produce the missing "
        "evidence. Do not emit final text until the contract is complete."
    )


# ---------------------------------------------------------------------------
# Tool result classification (needed by executive summary and agent.py)
# ---------------------------------------------------------------------------

def _classify_tool_result(tool_name: str, content: str) -> tuple[bool, str]:
    """Return ``(is_error, text)`` for a tool result content string.

    *text* is the human-meaningful payload to display or trim, in both the
    success and failure cases.  For ``execute_terminal_command`` the JSON
    envelope is unwrapped to the merged stdout/stderr (so the JSON scaffolding
    is dropped) and ``exit_code`` is the authoritative error signal.  For all
    other tools the raw content is returned verbatim and any content that
    begins with a bracketed error marker is treated as a failure.
    """
    if tool_name == "execute_terminal_command":
        try:
            data = json.loads(content)
            # Merge stdout and stderr so downstream trimming covers both streams.
            merged = "\n".join(
                filter(None, [data.get("stdout", ""), data.get("stderr", "")])
            )
            return data.get("exit_code", 0) > 0, merged
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass  # fall through to generic heuristic

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            if data.get("error"):
                return True, str(data.get("error"))
            if tool_name == "wait_for_port" and data.get("open") is False:
                return True, json.dumps(data)
    except (json.JSONDecodeError, TypeError):
        pass

    # Generic heuristic: bracketed error prefix produced by all builtin
    # dispatch branches and _extract_tool_text.
    stripped = content.lstrip()
    is_error = stripped.startswith("[") and "error" in stripped[:100].lower()
    return is_error, content


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------

def _action_label(tool_name: str, raw_arguments: str) -> str:
    """One-line, dedupe-friendly label for a completed action in the ledger."""
    try:
        args = json.loads(raw_arguments) if raw_arguments else {}
    except (json.JSONDecodeError, TypeError):
        args = {}
    if not isinstance(args, dict):
        args = {}
    command = args.get("command")
    if command:
        return f"{tool_name}: {' '.join(str(command).split())[:160]}"
    if args:
        compact = ", ".join(f"{key}={value}" for key, value in list(args.items())[:3])
        return f"{tool_name}({compact[:140]})"
    return tool_name


def _build_executive_summary(messages: list[dict[str, Any]]) -> str:
    """Derive a concise executive summary from the current message history.

    Appended as a trailing system message every iteration (after the cacheable
    prefix) so the model always sees, close to generation: the current
    objective, the actions it has ALREADY completed, and recent blockers.

    The Completed_Actions ledger is the key anti-thrash signal: on long
    multi-step tasks the raw transcript can be pruned or simply long, and the
    model otherwise re-runs finished steps. The ledger is built from the full
    durable history (never pruned), so it stays complete.
    """
    # Current_Objective - last concrete user message, capped at 300 chars.
    objective = "Not yet defined."
    user_texts = [
        str(msg.get("content") or "").strip()
        for msg in messages
        if msg.get("role") == "user"
    ]
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = (msg.get("content") or "").strip()
            if _is_continuation_signal(text):
                prior = next(
                    (
                        candidate
                        for candidate in reversed(user_texts[:-1])
                        if candidate and not _is_continuation_signal(candidate)
                    ),
                    "",
                )
                if prior:
                    text = prior
            objective = text[:300] + ("..." if len(text) > 300 else "")
            break

    # Map every tool_call_id to (tool_name, raw_arguments) from assistant turns.
    call_meta: dict[str, tuple[str, str]] = {}
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                try:
                    call_meta[tc["id"]] = (
                        tc["function"]["name"],
                        tc["function"].get("arguments", "") or "",
                    )
                except (KeyError, TypeError):
                    pass

    # Completed_Actions - successful side-effecting calls, in order, deduped.
    completed: list[str] = []
    seen: set[str] = set()
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        name, raw_args = call_meta.get(
            msg.get("tool_call_id", ""), ("unknown_tool", "")
        )
        if name not in _SIDE_EFFECT_TOOLS:
            continue
        is_error, _ = _classify_tool_result(name, msg.get("content", ""))
        if is_error:
            continue
        label = _action_label(name, raw_args)
        if label not in seen:
            seen.add(label)
            completed.append(label)
    completed_str = (
        "\n".join(f"  - {item}" for item in completed[-_LEDGER_MAX_ITEMS:])
        if completed
        else "None yet."
    )

    # Known_System_Blockers -- errors in the last 6 tool messages (~2 iterations)
    call_id_to_name = {cid: meta[0] for cid, meta in call_meta.items()}
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    blockers: list[str] = []
    for msg in tool_msgs[-6:]:
        tool_name = call_id_to_name.get(msg.get("tool_call_id", ""), "unknown_tool")
        content = msg.get("content", "")
        is_error, error_text = _classify_tool_result(tool_name, content)
        if not is_error:
            continue
        label = next(
            (ln.strip() for ln in error_text.splitlines() if ln.strip()),
            error_text[:120],
        )
        blockers.append(f"{tool_name}: {label[:120]}")

    blockers_str = "; ".join(blockers) if blockers else "None detected."

    # Plan - the model's own update_plan checklist, the primary progress signal.
    plan = _latest_plan(messages)
    if plan:
        plan_section = (
            "Plan (your update_plan checklist -- work the first not-done step; "
            "update_plan as steps finish):\n"
            f"{_render_plan(plan)}\n"
        )
    else:
        plan_section = (
            "Plan: none yet. For a multi-step task, call update_plan first to lay "
            "out the ordered steps, then keep it updated.\n"
        )

    return (
        "=== EXECUTIVE SUMMARY (refreshed every iteration) ===\n"
        f"Current_Objective: {objective}\n"
        f"{plan_section}"
        f"Completed_Actions (already done -- do NOT repeat these):\n{completed_str}\n"
        f"Known_System_Blockers: {blockers_str}\n"
        "=====================================================\n"
        "Work the plan: execute the first not-done step, build on Completed_Actions "
        "instead of redoing them, respect ordering (do not start a service before its "
        "prerequisites are in place), and clear any Known_System_Blockers. "
        "If the user's latest or next input is a short affirmation (e.g. 'yes', "
        "'ok', 'go', 'continue', 'do it'), treat it as confirmation to immediately "
        "execute Current_Objective. Do NOT ask for clarification. Act."
    )


# ---------------------------------------------------------------------------
# Memory storage for iteration cap
# ---------------------------------------------------------------------------

async def _store_iteration_cap_memory(
    memory: Any,
    session_id: str,
    original_prompt: str,
    tools_attempted: list[str],
    final_response: str,
    max_iterations: int,
) -> None:
    if not hasattr(memory, "store_event"):
        return
    raw_text = (
        f"Task: {original_prompt}\n"
        f"Tools attempted: {', '.join(tools_attempted) or 'none'}\n"
        f"Outcome: hit {max_iterations}-iteration cap\n"
        f"Final response: {final_response}"
    )
    entities = {
        "nodes": [{"label": "Tool", "name": name} for name in tools_attempted],
        "relationships": [],
    }
    await asyncio.to_thread(
        memory.store_event,
        session_id,
        raw_text,
        entities,
    )


# ---------------------------------------------------------------------------
# Message-window pruning
# ---------------------------------------------------------------------------

def _prune_message_window(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the stable prefix and recent turns, summarising older middle history."""
    if len(messages) <= _MAX_HISTORY_MESSAGES:
        return messages

    first_user_idx = next(
        (idx for idx, msg in enumerate(messages) if msg.get("role") == "user"),
        min(2, len(messages)),
    )
    prefix_end = max(first_user_idx, min(2, len(messages)))
    recent_start = max(prefix_end, len(messages) - _KEEP_RECENT_MESSAGES)

    # Do not start the retained recent window with an orphaned tool result.
    while recent_start < len(messages) and messages[recent_start].get("role") == "tool":
        recent_start += 1
    if recent_start >= len(messages):
        recent_start = max(prefix_end, len(messages) - _KEEP_RECENT_MESSAGES)

    middle = messages[prefix_end:recent_start]
    if not middle:
        return messages

    return [
        *messages[:prefix_end],
        {"role": "system", "content": _build_history_compaction_summary(middle)},
        *messages[recent_start:],
    ]


def _build_history_compaction_summary(messages: list[dict[str, Any]]) -> str:
    user_turns: list[str] = []
    assistant_turns: list[str] = []
    tool_names: list[str] = []

    for msg in messages:
        role = msg.get("role")
        if role == "user":
            text = str(msg.get("content") or "").strip()
            if text:
                user_turns.append(text[:180])
        elif role == "assistant":
            text = str(msg.get("content") or "").strip()
            if text:
                assistant_turns.append(text[:180])
            for tc in msg.get("tool_calls") or []:
                try:
                    tool_names.append(tc["function"]["name"])
                except (KeyError, TypeError):
                    pass

    recent_users = "; ".join(user_turns[-3:]) or "None recorded."
    recent_assistant = "; ".join(assistant_turns[-3:]) or "None recorded."
    tools = ", ".join(sorted(set(tool_names))) or "none"

    return (
        "Earlier conversation was compacted to reduce model input tokens.\n"
        f"Recent older user requests: {recent_users}\n"
        f"Recent older assistant answers: {recent_assistant}\n"
        f"Tools used in compacted history: {tools}"
    )
