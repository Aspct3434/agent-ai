"""Proof-carrying task graph orchestration.

The task graph is stored in normal conversation history through tool calls,
similar to ``update_plan``.  That keeps checkpoints replayable and avoids a
second runtime database while still giving the agent a typed execution graph
with proof-linked nodes.
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from contract import (
    _expose_local_http_service_evidence_is_positive,
    _filesystem_artifact_evidence_is_positive,
    _plan_steps_from_args,
    _successful_command_output_evidence,
    _tcp_service_evidence_is_positive,
    _wait_for_port_evidence_is_positive,
    _write_text_file_evidence_is_positive,
)
from evaluator import ExecutionStep

SET_TASK_GRAPH_TOOL_NAME = "set_task_graph"
INSPECT_TASK_GRAPH_TOOL_NAME = "inspect_task_graph"
UPDATE_TASK_NODE_TOOL_NAME = "update_task_node"
REPAIR_TASK_GRAPH_TOOL_NAME = "repair_task_graph"
VERIFY_TASK_GRAPH_TOOL_NAME = "verify_task_graph"

TASK_GRAPH_TOOL_NAMES: frozenset[str] = frozenset(
    {
        SET_TASK_GRAPH_TOOL_NAME,
        INSPECT_TASK_GRAPH_TOOL_NAME,
        UPDATE_TASK_NODE_TOOL_NAME,
        REPAIR_TASK_GRAPH_TOOL_NAME,
        VERIFY_TASK_GRAPH_TOOL_NAME,
    }
)

GRAPH_CONTROL_TOOLS: frozenset[str] = frozenset(
    {
        SET_TASK_GRAPH_TOOL_NAME,
        INSPECT_TASK_GRAPH_TOOL_NAME,
        UPDATE_TASK_NODE_TOOL_NAME,
        REPAIR_TASK_GRAPH_TOOL_NAME,
        VERIFY_TASK_GRAPH_TOOL_NAME,
        "update_plan",
        "expand_tool_output",
    }
)

RECOVERY_DIAGNOSTIC_TOOLS: frozenset[str] = frozenset(
    {
        "get_system_environment",
        "get_filesystem_process_evidence",
        "web_search",
        "web_fetch",
        "expand_tool_output",
        "wait_for_port",
    }
)

VALID_NODE_STATUSES: frozenset[str] = frozenset(
    {"pending", "ready", "in_progress", "done", "failed", "blocked"}
)
TERMINAL_NODE_STATUSES: frozenset[str] = frozenset({"done", "failed"})
OPEN_NODE_STATUSES: frozenset[str] = frozenset(
    {"pending", "ready", "in_progress", "blocked"}
)
VALID_NODE_KINDS: frozenset[str] = frozenset(
    {"plan", "diagnose", "write", "command", "service", "verify", "delegate", "finalize"}
)
VALID_PROOF_REQUIREMENTS: frozenset[str] = frozenset(
    {
        "filesystem_artifact",
        "running_http_service",
        "running_tcp_service",
        "database_mutation",
        "command_output",
        "artifact_quality",
        "none",
    }
)

DEFAULT_ALLOWED_TOOLS_BY_KIND: dict[str, frozenset[str]] = {
    "plan": frozenset({SET_TASK_GRAPH_TOOL_NAME, "update_plan"}),
    "diagnose": RECOVERY_DIAGNOSTIC_TOOLS,
    "write": frozenset(
        {
            "write_text_file",
            "write_file",
            "edit_file",
            "create_directory",
            "execute_terminal_command",
            "get_filesystem_process_evidence",
        }
    ),
    "command": frozenset(
        {"execute_terminal_command", "get_system_environment", "get_filesystem_process_evidence"}
    ),
    "service": frozenset(
        {
            "execute_background_service",
            "execute_terminal_command",
            "wait_for_port",
            "get_filesystem_process_evidence",
            "expose_local_http_service",
        }
    ),
    "verify": frozenset(
        {
            "get_filesystem_process_evidence",
            "wait_for_port",
            "browser_navigate",
            "browser_get_text",
            "browser_screenshot",
            "read_file",
            "read_query",
        }
    ),
    "delegate": frozenset({"delegate_task"}),
    "finalize": frozenset({VERIFY_TASK_GRAPH_TOOL_NAME, UPDATE_TASK_NODE_TOOL_NAME}),
}


def _json_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _as_str_list(value: Any, *, field: str) -> tuple[list[str], str | None]:
    if value in (None, ""):
        return [], None
    if not isinstance(value, list):
        return [], f"{field} must be an array of strings"
    items: list[str] = []
    for item in value:
        text = " ".join(str(item).split())
        if text:
            items.append(text)
    return list(dict.fromkeys(items)), None


def _normalise_status(value: Any) -> str:
    text = str(value or "pending").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "todo": "pending",
        "waiting": "pending",
        "active": "in_progress",
        "started": "in_progress",
        "complete": "done",
        "completed": "done",
        "ok": "done",
        "error": "failed",
    }
    return aliases.get(text, text)


def _normalise_node(raw: Any, index: int) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw, dict):
        return None, f"node {index} must be an object"

    node_id = " ".join(str(raw.get("id") or f"node_{index + 1}").split())
    if not node_id:
        return None, f"node {index} id must be non-empty"
    title = " ".join(str(raw.get("title") or raw.get("step") or raw.get("name") or "").split())
    if not title:
        return None, f"node {node_id!r} title must be non-empty"

    kind = str(raw.get("kind") or "verify").strip().lower()
    if kind not in VALID_NODE_KINDS:
        return None, f"node {node_id!r} has unknown kind {kind!r}"

    status = _normalise_status(raw.get("status"))
    if status not in VALID_NODE_STATUSES:
        return None, f"node {node_id!r} has unknown status {status!r}"

    depends_on, err = _as_str_list(raw.get("depends_on", []), field="depends_on")
    if err:
        return None, f"node {node_id!r}: {err}"
    allowed_tools, err = _as_str_list(raw.get("allowed_tools", []), field="allowed_tools")
    if err:
        return None, f"node {node_id!r}: {err}"
    proof_requirements, err = _as_str_list(
        raw.get("proof_requirements", []), field="proof_requirements"
    )
    if err:
        return None, f"node {node_id!r}: {err}"
    unknown_proofs = sorted(set(proof_requirements) - VALID_PROOF_REQUIREMENTS)
    if unknown_proofs:
        return None, (
            f"node {node_id!r} has unknown proof requirement(s): "
            f"{', '.join(unknown_proofs)}"
        )
    evidence_refs, err = _as_str_list(raw.get("evidence_refs", []), field="evidence_refs")
    if err:
        return None, f"node {node_id!r}: {err}"

    retry_count = raw.get("retry_count", 0)
    try:
        retry_count_int = max(0, int(retry_count or 0))
    except (TypeError, ValueError):
        return None, f"node {node_id!r} retry_count must be an integer"

    return (
        {
            "id": node_id,
            "title": title,
            "kind": kind,
            "status": status,
            "depends_on": depends_on,
            "allowed_tools": allowed_tools,
            "proof_requirements": proof_requirements,
            "evidence_refs": evidence_refs,
            "failure_reason": " ".join(str(raw.get("failure_reason") or "").split()),
            "retry_count": retry_count_int,
        },
        None,
    )


def normalise_task_graph(nodes: Any) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(nodes, str):
        try:
            nodes = json.loads(nodes)
        except json.JSONDecodeError:
            return [], "nodes must be an array, not an invalid JSON string"
    if not isinstance(nodes, list) or not nodes:
        return [], "nodes must be a non-empty array"

    normalised: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(nodes):
        node, error = _normalise_node(raw, index)
        if error is not None:
            return [], error
        assert node is not None
        if node["id"] in seen:
            return [], f"duplicate node id {node['id']!r}"
        seen.add(str(node["id"]))
        normalised.append(node)

    unknown_deps = sorted(
        {
            dep
            for node in normalised
            for dep in node.get("depends_on", [])
            if dep not in seen
        }
    )
    if unknown_deps:
        return [], f"unknown dependency id(s): {', '.join(unknown_deps)}"

    cycle = _find_cycle(normalised)
    if cycle:
        return [], f"cycle detected: {' -> '.join(cycle)}"

    return normalised, None


def _find_cycle(nodes: list[dict[str, Any]]) -> list[str]:
    by_id = {str(node["id"]): node for node in nodes}
    visiting: set[str] = set()
    visited: set[str] = set()
    path: list[str] = []

    def visit(node_id: str) -> list[str]:
        if node_id in visited:
            return []
        if node_id in visiting:
            start = path.index(node_id) if node_id in path else 0
            return [*path[start:], node_id]
        visiting.add(node_id)
        path.append(node_id)
        for dep in by_id[node_id].get("depends_on", []):
            cycle = visit(str(dep))
            if cycle:
                return cycle
        path.pop()
        visiting.remove(node_id)
        visited.add(node_id)
        return []

    for node in nodes:
        cycle = visit(str(node["id"]))
        if cycle:
            return cycle
    return []


def _current_task_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    user_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "user"
    ]
    if not user_indices:
        return messages
    latest = user_indices[-1]
    latest_text = str(messages[latest].get("content") or "").strip().lower().strip(".!?")
    continuation = latest_text in {
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
    if continuation and len(user_indices) > 1:
        latest = user_indices[-2]
    return messages[latest:]


def _assistant_tool_calls(messages: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    for message in _current_task_messages(messages):
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            try:
                name = str(tool_call["function"]["name"])
                raw_args = tool_call["function"].get("arguments") or "{}"
                args = json.loads(raw_args)
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(args, dict):
                calls.append((name, args))
    return calls


def _latest_plan_nodes(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    latest_steps: list[dict[str, Any]] | None = None
    for name, args in _assistant_tool_calls(messages):
        if name != "update_plan":
            continue
        steps = _plan_steps_from_args(args)
        if steps:
            latest_steps = steps
    if not latest_steps:
        return None

    nodes: list[dict[str, Any]] = []
    previous_id = ""
    for index, step in enumerate(latest_steps):
        title = str(step.get("title") or f"Step {index + 1}")
        node_id = f"plan_{index + 1}"
        status = _normalise_status(step.get("status"))
        kind = _infer_kind_from_title(title)
        nodes.append(
            {
                "id": node_id,
                "title": title,
                "kind": kind,
                "status": status,
                "depends_on": [previous_id] if previous_id else [],
                "allowed_tools": sorted(DEFAULT_ALLOWED_TOOLS_BY_KIND.get(kind, frozenset())),
                "proof_requirements": [],
                "evidence_refs": [],
                "failure_reason": "",
                "retry_count": 0,
            }
        )
        previous_id = node_id
    return nodes


def _infer_kind_from_title(title: str) -> str:
    low = title.lower()
    if any(token in low for token in ("diagnose", "inspect", "investigate", "read")):
        return "diagnose"
    if any(token in low for token in ("write", "create", "edit", "save", "file")):
        return "write"
    if any(token in low for token in ("serve", "server", "port", "service", "launch")):
        return "service"
    if any(token in low for token in ("verify", "test", "check", "confirm")):
        return "verify"
    if any(token in low for token in ("delegate", "audit", "review")):
        return "delegate"
    if any(token in low for token in ("run", "install", "build", "command")):
        return "command"
    return "verify"


def _apply_node_update(
    nodes: list[dict[str, Any]], args: dict[str, Any]
) -> list[dict[str, Any]]:
    node_id = str(args.get("node_id") or "")
    if not node_id:
        return nodes
    updated = deepcopy(nodes)
    for node in updated:
        if node.get("id") != node_id:
            continue
        if "status" in args:
            status = _normalise_status(args.get("status"))
            if status in VALID_NODE_STATUSES:
                prior = node.get("status")
                node["status"] = status
                if status in {"failed", "blocked"} and prior != status:
                    node["retry_count"] = int(node.get("retry_count") or 0) + 1
        if "evidence_refs" in args:
            refs, error = _as_str_list(args.get("evidence_refs"), field="evidence_refs")
            if error is None:
                node["evidence_refs"] = refs
        if "failure_reason" in args:
            node["failure_reason"] = " ".join(str(args.get("failure_reason") or "").split())
        return updated
    return nodes


def _repair_nodes(
    nodes: list[dict[str, Any]], reason: str
) -> list[dict[str, Any]]:
    repaired = deepcopy(nodes)
    terminal = {
        str(node["id"])
        for node in repaired
        if node.get("status") in TERMINAL_NODE_STATUSES
    }
    for node in repaired:
        if node.get("status") not in {"failed", "blocked"}:
            continue
        if node.get("evidence_refs") and node.get("status") == "done":
            continue
        if all(str(dep) in terminal for dep in node.get("depends_on", [])):
            node["status"] = "pending"
            node["failure_reason"] = reason
    return repaired


class TaskGraphEngine:
    """Reconstruct, validate, verify, and guide session task graphs."""

    def run_set_task_graph(self, arguments: dict[str, Any]) -> tuple[str, bool]:
        nodes, error = normalise_task_graph(arguments.get("nodes"))
        if error is not None:
            return f"[set_task_graph error] {error}", True
        snapshot = self._snapshot(nodes, source="explicit")
        return _json_response({"graph_set": True, **snapshot}), False

    def run_update_task_node(
        self,
        arguments: dict[str, Any],
        messages: list[dict[str, Any]],
        steps: list[ExecutionStep],
    ) -> tuple[str, bool]:
        node_id = str(arguments.get("node_id") or "")
        if not node_id:
            return "[update_task_node error] node_id is required", True
        status = _normalise_status(arguments.get("status"))
        if status not in VALID_NODE_STATUSES:
            return f"[update_task_node error] unknown status {status!r}", True
        graph = self.latest_graph(messages)
        if graph is None:
            return "[update_task_node error] no task graph exists", True
        if node_id not in {node["id"] for node in graph["nodes"]}:
            return f"[update_task_node error] node {node_id!r} not found", True
        verification = self.verify(messages, steps)
        return _json_response({"node_updated": node_id, "status": status, **verification}), False

    def run_repair_task_graph(
        self,
        arguments: dict[str, Any],
        messages: list[dict[str, Any]],
        steps: list[ExecutionStep],
    ) -> tuple[str, bool]:
        reason = " ".join(str(arguments.get("reason") or "").split())
        if not reason:
            return "[repair_task_graph error] reason is required", True
        graph = self.latest_graph(messages)
        if graph is None:
            return "[repair_task_graph error] no task graph exists", True
        return _json_response({"graph_repaired": True, "reason": reason, **self.inspect(messages, steps)}), False

    def inspect(
        self, messages: list[dict[str, Any]], steps: list[ExecutionStep] | None = None
    ) -> dict[str, Any]:
        graph = self.latest_graph(messages)
        if graph is None:
            return {
                "has_graph": False,
                "source": None,
                "nodes": [],
                "active_node": None,
                "ready_nodes": [],
                "blocked_nodes": [],
                "verifier": {
                    "passed": False,
                    "missing_nodes": ["task_graph"],
                    "invalid_evidence_refs": [],
                    "blocked_nodes": [],
                    "proof_report": [],
                },
            }
        nodes = graph["nodes"]
        snapshot = self._snapshot(nodes, source=str(graph["source"]))
        if steps is not None:
            snapshot["verifier"] = self.verify(messages, steps)
        return snapshot

    def verify(self, messages: list[dict[str, Any]], steps: list[ExecutionStep]) -> dict[str, Any]:
        graph = self.latest_graph(messages)
        if graph is None:
            return {
                "passed": False,
                "missing_nodes": ["task_graph"],
                "invalid_evidence_refs": [],
                "ignored_invalid_evidence_refs": [],
                "blocked_nodes": [],
                "proof_report": [],
            }
        return self.verify_nodes(graph["nodes"], steps)

    def verify_nodes(
        self, nodes: list[dict[str, Any]], steps: list[ExecutionStep]
    ) -> dict[str, Any]:
        by_id = {str(node["id"]): node for node in nodes}
        terminal = {
            str(node["id"])
            for node in nodes
            if node.get("status") in TERMINAL_NODE_STATUSES
        }
        missing_nodes = [
            str(node["id"])
            for node in nodes
            if node.get("status") not in TERMINAL_NODE_STATUSES
        ]
        blocked_nodes = [
            str(node["id"])
            for node in nodes
            if node.get("status") == "blocked"
            or any(str(dep) not in terminal for dep in node.get("depends_on", []))
        ]

        invalid_refs: list[dict[str, str]] = []
        ignored_invalid_refs: list[dict[str, str]] = []
        proof_report: list[dict[str, Any]] = []
        for node in nodes:
            node_id = str(node["id"])
            if node.get("status") == "failed":
                proof_report.append(
                    {
                        "node_id": node_id,
                        "passed": True,
                        "terminal_status": "failed",
                        "detail": node.get("failure_reason") or "explicitly failed",
                    }
                )
                continue
            if node.get("status") != "done":
                proof_report.append(
                    {
                        "node_id": node_id,
                        "passed": False,
                        "detail": f"node status is {node.get('status')}",
                    }
                )
                continue

            bad_deps = [dep for dep in node.get("depends_on", []) if dep not in terminal]
            if bad_deps:
                proof_report.append(
                    {
                        "node_id": node_id,
                        "passed": False,
                        "detail": f"dependencies not terminal: {', '.join(bad_deps)}",
                    }
                )
                continue

            requirements = [
                req for req in node.get("proof_requirements", []) if req != "none"
            ]
            if not requirements:
                proof_report.append(
                    {"node_id": node_id, "passed": True, "detail": "no proof required"}
                )
                continue

            candidate_steps = self._candidate_steps_for_node(node, steps)
            inferred_candidate_steps = self._candidate_steps_for_node(
                node, steps, ignore_evidence_refs=True
            )
            ref_lookup = self._step_lookup(steps)
            node_invalid_refs: list[dict[str, str]] = []
            for ref in node.get("evidence_refs", []):
                step = ref_lookup.get(str(ref))
                if step is None:
                    node_invalid_refs.append(
                        {
                            "node_id": node_id,
                            "evidence_ref": str(ref),
                            "reason": "not found",
                        }
                    )
                elif step.metadata.get("is_error"):
                    node_invalid_refs.append(
                        {
                            "node_id": node_id,
                            "evidence_ref": str(ref),
                            "reason": "tool result is an error",
                        }
                    )
            requirement_results = []
            for requirement in requirements:
                ref_passed = any(
                    _step_satisfies_requirement(step, requirement)
                    for step in candidate_steps
                )
                inferred_passed = any(
                    _step_satisfies_requirement(step, requirement)
                    for step in inferred_candidate_steps
                )
                passed = any(
                    _step_satisfies_requirement(step, requirement)
                    for step in [*candidate_steps, *inferred_candidate_steps]
                )
                requirement_results.append(
                    {
                        "requirement": requirement,
                        "passed": passed,
                        "passed_by_evidence_ref": ref_passed,
                        "passed_by_inference": inferred_passed,
                    }
                )
            requirements_passed = all(item["passed"] for item in requirement_results)
            if node_invalid_refs and requirements_passed:
                ignored_invalid_refs.extend(node_invalid_refs)
            else:
                invalid_refs.extend(node_invalid_refs)
            inferred_refs = [
                str(ref)
                for ref in (
                    step.metadata.get("tool_call_id")
                    for step in inferred_candidate_steps
                    if any(
                        _step_satisfies_requirement(step, requirement)
                        for requirement in requirements
                    )
                )
                if ref
            ]
            proof_report.append(
                {
                    "node_id": node_id,
                    "passed": requirements_passed,
                    "requirements": requirement_results,
                    "evidence_refs": list(node.get("evidence_refs", [])),
                    "suggested_evidence_refs": inferred_refs,
                    "ignored_invalid_evidence_refs": node_invalid_refs
                    if requirements_passed
                    else [],
                }
            )

        passed = (
            not missing_nodes
            and not blocked_nodes
            and not invalid_refs
            and all(item.get("passed") for item in proof_report)
        )
        return {
            "passed": passed,
            "missing_nodes": missing_nodes,
            "invalid_evidence_refs": invalid_refs,
            "ignored_invalid_evidence_refs": ignored_invalid_refs,
            "blocked_nodes": blocked_nodes,
            "proof_report": proof_report,
            "node_count": len(by_id),
        }

    def latest_graph(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        nodes: list[dict[str, Any]] | None = None
        source = "plan"
        for name, args in _assistant_tool_calls(messages):
            if name == SET_TASK_GRAPH_TOOL_NAME:
                parsed, error = normalise_task_graph(args.get("nodes"))
                if error is None:
                    nodes = parsed
                    source = "explicit"
            elif name == UPDATE_TASK_NODE_TOOL_NAME and nodes is not None:
                nodes = _apply_node_update(nodes, args)
            elif name == REPAIR_TASK_GRAPH_TOOL_NAME and nodes is not None:
                reason = " ".join(str(args.get("reason") or "").split()) or "repair requested"
                nodes = _repair_nodes(nodes, reason)

        if nodes is not None:
            return {"source": source, "nodes": nodes}

        plan_nodes = _latest_plan_nodes(messages)
        if plan_nodes is None:
            return None
        for name, args in _assistant_tool_calls(messages):
            if name == UPDATE_TASK_NODE_TOOL_NAME:
                plan_nodes = _apply_node_update(plan_nodes, args)
            elif name == REPAIR_TASK_GRAPH_TOOL_NAME:
                reason = " ".join(str(args.get("reason") or "").split()) or "repair requested"
                plan_nodes = _repair_nodes(plan_nodes, reason)
        return {"source": "plan", "nodes": plan_nodes}

    def completion_status(
        self, messages: list[dict[str, Any]], steps: list[ExecutionStep]
    ) -> dict[str, Any]:
        graph = self.latest_graph(messages)
        if graph is None:
            return {
                "has_graph": False,
                "source": None,
                "complete": False,
                "missing": ["task_graph"],
                "active_node": None,
                "ready_nodes": [],
                "blocked_nodes": [],
                "verifier": self.verify(messages, steps),
            }
        verifier = self.verify_nodes(graph["nodes"], steps)
        snapshot = self._snapshot(graph["nodes"], source=str(graph["source"]))
        missing: list[str] = []
        if verifier["missing_nodes"] or verifier["blocked_nodes"]:
            missing.append("task_graph_open")
        if verifier["invalid_evidence_refs"] or not all(
            item.get("passed") for item in verifier["proof_report"]
        ):
            missing.append("task_graph_proof")
        return {
            "has_graph": True,
            "source": graph["source"],
            "complete": verifier["passed"],
            "missing": missing,
            "active_node": snapshot["active_node"],
            "ready_nodes": snapshot["ready_nodes"],
            "blocked_nodes": snapshot["blocked_nodes"],
            "verifier": verifier,
        }

    def allowed_tools_for_next(
        self, messages: list[dict[str, Any]], steps: list[ExecutionStep]
    ) -> set[str]:
        graph = self.latest_graph(messages)
        if graph is None:
            return set(GRAPH_CONTROL_TOOLS)
        snapshot = self._snapshot(graph["nodes"], source=str(graph["source"]))
        active = snapshot["active_node"]
        allowed = set(GRAPH_CONTROL_TOOLS)
        if not active:
            verifier = self.verify_nodes(graph["nodes"], steps)
            if not verifier.get("passed"):
                allowed.update(RECOVERY_DIAGNOSTIC_TOOLS)
                allowed.add(REPAIR_TASK_GRAPH_TOOL_NAME)
            return allowed
        active_allowed = active.get("allowed_tools") or sorted(
            DEFAULT_ALLOWED_TOOLS_BY_KIND.get(str(active.get("kind")), frozenset())
        )
        allowed.update(str(tool) for tool in active_allowed)
        if int(active.get("retry_count") or 0) >= 2 or active.get("status") == "blocked":
            allowed.update(RECOVERY_DIAGNOSTIC_TOOLS)
            allowed.add(REPAIR_TASK_GRAPH_TOOL_NAME)
        if any(step.metadata.get("is_error") for step in steps[-3:]):
            allowed.update(RECOVERY_DIAGNOSTIC_TOOLS)
        return allowed

    def build_instruction(
        self, messages: list[dict[str, Any]], steps: list[ExecutionStep]
    ) -> str:
        status = self.completion_status(messages, steps)
        if not status["has_graph"]:
            return (
                "TASK GRAPH REQUIRED:\n"
                "Create a typed execution DAG before doing more work. Prefer "
                "set_task_graph with nodes that include id, title, kind, status, "
                "depends_on, allowed_tools, proof_requirements, evidence_refs, "
                "failure_reason, and retry_count. update_plan remains accepted "
                "for backward compatibility, but set_task_graph gives stronger "
                "proof-carrying execution control."
            )
        active = status.get("active_node") or {}
        ready = ", ".join(node["id"] for node in status.get("ready_nodes", [])) or "none"
        blocked = ", ".join(status.get("blocked_nodes", [])) or "none"
        proof = status.get("verifier", {})
        return (
            "TASK GRAPH STATUS:\n"
            f"Source: {status.get('source')}\n"
            f"Active node: {active.get('id', 'none')} - {active.get('title', '')}\n"
            f"Ready nodes: {ready}\n"
            f"Blocked nodes: {blocked}\n"
            f"Graph proof passed: {proof.get('passed')}\n"
            "Work only the active ready node. Use tools from that node's allowed_tools. "
            "After a tool produces evidence, call update_task_node with the node_id, "
            "status, and evidence_refs using the successful tool_call_id. If the node "
            "fails repeatedly or becomes impossible, call repair_task_graph with the "
            "specific blocker."
        )

    def _snapshot(self, nodes: list[dict[str, Any]], *, source: str) -> dict[str, Any]:
        ready_nodes = [
            node
            for node in nodes
            if node.get("status") in {"ready", "in_progress"}
            or (
                node.get("status") == "pending"
                and all(
                    self._node_status(nodes, str(dep)) in TERMINAL_NODE_STATUSES
                    for dep in node.get("depends_on", [])
                )
            )
        ]
        active = next((node for node in nodes if node.get("status") == "in_progress"), None)
        if active is None:
            active = ready_nodes[0] if ready_nodes else None
        blocked_nodes = [
            str(node["id"])
            for node in nodes
            if node.get("status") == "blocked"
            or (
                any(
                    self._node_status(nodes, str(dep)) not in TERMINAL_NODE_STATUSES
                    for dep in node.get("depends_on", [])
                )
                and node.get("status") not in TERMINAL_NODE_STATUSES
            )
        ]
        return {
            "has_graph": True,
            "source": source,
            "nodes": deepcopy(nodes),
            "active_node": deepcopy(active),
            "ready_nodes": deepcopy(ready_nodes),
            "blocked_nodes": blocked_nodes,
            "summary": {
                "total": len(nodes),
                "done": sum(1 for node in nodes if node.get("status") == "done"),
                "failed": sum(1 for node in nodes if node.get("status") == "failed"),
                "open": sum(1 for node in nodes if node.get("status") in OPEN_NODE_STATUSES),
            },
        }

    @staticmethod
    def _node_status(nodes: list[dict[str, Any]], node_id: str) -> str:
        for node in nodes:
            if node.get("id") == node_id:
                return str(node.get("status"))
        return "missing"

    @staticmethod
    def _step_lookup(steps: list[ExecutionStep]) -> dict[str, ExecutionStep]:
        lookup: dict[str, ExecutionStep] = {}
        for step in steps:
            if step.kind != "tool_result":
                continue
            call_id = step.metadata.get("tool_call_id")
            if call_id:
                lookup[str(call_id)] = step
        return lookup

    def _candidate_steps_for_node(
        self,
        node: dict[str, Any],
        steps: list[ExecutionStep],
        *,
        ignore_evidence_refs: bool = False,
    ) -> list[ExecutionStep]:
        refs = [str(ref) for ref in node.get("evidence_refs", [])]
        lookup = self._step_lookup(steps)
        if refs and not ignore_evidence_refs:
            return [lookup[ref] for ref in refs if ref in lookup]
        allowed = {str(tool) for tool in node.get("allowed_tools", [])}
        if not allowed:
            allowed = set(DEFAULT_ALLOWED_TOOLS_BY_KIND.get(str(node.get("kind")), frozenset()))
        return [
            step
            for step in steps
            if step.kind == "tool_result"
            and not step.metadata.get("is_error")
            and (not allowed or step.metadata.get("tool_name") in allowed)
        ]


def _step_satisfies_requirement(step: ExecutionStep, requirement: str) -> bool:
    if step.kind != "tool_result" or step.metadata.get("is_error"):
        return False
    tool_name = step.metadata.get("tool_name")
    content = step.content
    if requirement == "filesystem_artifact":
        if tool_name == "write_text_file":
            return _write_text_file_evidence_is_positive(content)
        if tool_name == "get_filesystem_process_evidence":
            return _filesystem_artifact_evidence_is_positive(content)
    if requirement == "running_http_service":
        return bool(
            tool_name == "expose_local_http_service"
            and _expose_local_http_service_evidence_is_positive(content)
        )
    if requirement == "running_tcp_service":
        if tool_name == "wait_for_port":
            return _wait_for_port_evidence_is_positive(content)
        if tool_name == "get_filesystem_process_evidence":
            return _tcp_service_evidence_is_positive(content)
    if requirement == "database_mutation":
        return tool_name in {"create_table", "write_query"}
    if requirement == "command_output":
        return _successful_command_output_evidence(str(tool_name), content)
    if requirement == "artifact_quality":
        if tool_name != "write_text_file":
            return False
        try:
            data = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return False
        quality = data.get("artifact_quality")
        return not isinstance(quality, dict) or not bool(quality.get("placeholder_detected"))
    return requirement == "none"
