from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from evaluator import ExecutionStep  # noqa: E402
from task_graph import TaskGraphEngine, normalise_task_graph  # noqa: E402


def _assistant_tool(name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def _tool_step(
    tool_name: str,
    call_id: str,
    content: dict,
    *,
    is_error: bool = False,
    arguments: dict[str, object] | None = None,
) -> ExecutionStep:
    return ExecutionStep(
        kind="tool_result",
        content=json.dumps(content),
        metadata={
            "tool_name": tool_name,
            "tool_call_id": call_id,
            "is_error": is_error,
            "arguments": arguments or {},
        },
    )


def test_graph_schema_rejects_duplicate_unknown_dep_and_cycle() -> None:
    _, error = normalise_task_graph(
        [
            {"id": "a", "title": "A", "kind": "write", "status": "pending"},
            {"id": "a", "title": "B", "kind": "write", "status": "pending"},
        ]
    )
    assert "duplicate" in str(error)

    _, error = normalise_task_graph(
        [{"id": "a", "title": "A", "kind": "write", "status": "pending", "depends_on": ["b"]}]
    )
    assert "unknown dependency" in str(error)

    _, error = normalise_task_graph(
        [
            {"id": "a", "title": "A", "kind": "write", "status": "pending", "depends_on": ["b"]},
            {"id": "b", "title": "B", "kind": "verify", "status": "pending", "depends_on": ["a"]},
        ]
    )
    assert "cycle detected" in str(error)


def test_update_plan_auto_converts_to_linear_graph() -> None:
    engine = TaskGraphEngine()
    messages = [
        {"role": "user", "content": "build it"},
        _assistant_tool(
            "update_plan",
            {
                "steps": [
                    {"title": "Write files", "status": "done"},
                    {"title": "Verify files", "status": "pending"},
                ]
            },
        ),
    ]

    snapshot = engine.inspect(messages, [])

    assert snapshot["source"] == "plan"
    assert [node["id"] for node in snapshot["nodes"]] == ["plan_1", "plan_2"]
    assert snapshot["nodes"][1]["depends_on"] == ["plan_1"]
    assert snapshot["active_node"]["id"] == "plan_2"


def test_update_task_node_status_transition_and_evidence_refs() -> None:
    engine = TaskGraphEngine()
    messages = [
        {"role": "user", "content": "write file"},
        _assistant_tool(
            "set_task_graph",
            {
                "nodes": [
                    {
                        "id": "write",
                        "title": "Write artifact",
                        "kind": "write",
                        "status": "in_progress",
                        "allowed_tools": ["write_text_file"],
                        "proof_requirements": ["filesystem_artifact"],
                    }
                ]
            },
        ),
        _assistant_tool(
            "update_task_node",
            {"node_id": "write", "status": "done", "evidence_refs": ["call_write"]},
        ),
    ]

    graph = engine.latest_graph(messages)

    assert graph is not None
    assert graph["nodes"][0]["status"] == "done"
    assert graph["nodes"][0]["evidence_refs"] == ["call_write"]


def test_verifier_rejects_fake_evidence_ref() -> None:
    engine = TaskGraphEngine()
    messages = [
        {"role": "user", "content": "write file"},
        _assistant_tool(
            "set_task_graph",
            {
                "nodes": [
                    {
                        "id": "write",
                        "title": "Write artifact",
                        "kind": "write",
                        "status": "done",
                        "allowed_tools": ["write_text_file"],
                        "proof_requirements": ["filesystem_artifact"],
                        "evidence_refs": ["missing_call"],
                    }
                ]
            },
        ),
    ]

    result = engine.verify(messages, [])

    assert result["passed"] is False
    assert result["invalid_evidence_refs"][0]["evidence_ref"] == "missing_call"


def test_verifier_infers_real_evidence_when_ref_name_is_wrong() -> None:
    engine = TaskGraphEngine()
    messages = [
        {"role": "user", "content": "write file"},
        _assistant_tool(
            "set_task_graph",
            {
                "nodes": [
                    {
                        "id": "write",
                        "title": "Write artifact",
                        "kind": "write",
                        "status": "done",
                        "allowed_tools": ["write_text_file"],
                        "proof_requirements": ["filesystem_artifact"],
                        "evidence_refs": ["hallucinated_ref"],
                    }
                ]
            },
        ),
    ]
    steps = [
        _tool_step(
            "write_text_file",
            "call_write",
            {"written": True, "exists": True, "size_bytes": 12},
        )
    ]

    result = engine.verify(messages, steps)

    assert result["passed"] is True
    assert result["invalid_evidence_refs"] == []
    assert result["ignored_invalid_evidence_refs"][0]["evidence_ref"] == "hallucinated_ref"
    assert result["proof_report"][0]["suggested_evidence_refs"] == ["call_write"]


def test_verifier_accepts_real_file_and_command_evidence() -> None:
    engine = TaskGraphEngine()
    messages = [
        {"role": "user", "content": "write and run"},
        _assistant_tool(
            "set_task_graph",
            {
                "nodes": [
                    {
                        "id": "write",
                        "title": "Write artifact",
                        "kind": "write",
                        "status": "done",
                        "allowed_tools": ["write_text_file"],
                        "proof_requirements": ["filesystem_artifact"],
                        "evidence_refs": ["call_write"],
                    },
                    {
                        "id": "command",
                        "title": "Run check",
                        "kind": "command",
                        "status": "done",
                        "depends_on": ["write"],
                        "allowed_tools": ["execute_terminal_command"],
                        "proof_requirements": ["command_output"],
                        "evidence_refs": ["call_cmd"],
                    },
                ]
            },
        ),
    ]
    steps = [
        _tool_step(
            "write_text_file",
            "call_write",
            {"written": True, "exists": True, "size_bytes": 12},
        ),
        _tool_step(
            "execute_terminal_command",
            "call_cmd",
            {"exit_code": 0, "stdout": "ok", "stderr": ""},
        ),
    ]

    result = engine.verify(messages, steps)

    assert result["passed"] is True


def test_verifier_accepts_generic_http_observation_as_service_evidence() -> None:
    engine = TaskGraphEngine()
    messages = [
        {"role": "user", "content": "serve site"},
        _assistant_tool(
            "set_task_graph",
            {
                "nodes": [
                    {
                        "id": "verify",
                        "title": "Verify served site",
                        "kind": "verify",
                        "status": "done",
                        "allowed_tools": ["custom_probe"],
                        "proof_requirements": ["running_http_service"],
                        "evidence_refs": ["call_probe"],
                    }
                ]
            },
        ),
    ]
    steps = [
        _tool_step(
            "execute_background_service",
            "call_launch",
            {"status": "launched", "pid": 123},
            arguments={"command": "python3 -m http.server 8080"},
        ),
        _tool_step(
            "custom_probe",
            "call_probe",
            {
                "url": "http://127.0.0.1:8080/",
                "status_code": 200,
                "content_type": "text/html",
            },
        )
    ]

    result = engine.verify(messages, steps)

    assert result["passed"] is True


def test_repair_preserves_completed_evidence_nodes() -> None:
    engine = TaskGraphEngine()
    messages = [
        {"role": "user", "content": "write then serve"},
        _assistant_tool(
            "set_task_graph",
            {
                "nodes": [
                    {
                        "id": "write",
                        "title": "Write artifact",
                        "kind": "write",
                        "status": "done",
                        "allowed_tools": ["write_text_file"],
                        "proof_requirements": ["filesystem_artifact"],
                        "evidence_refs": ["call_write"],
                    },
                    {
                        "id": "serve",
                        "title": "Serve artifact",
                        "kind": "service",
                        "status": "blocked",
                        "depends_on": ["write"],
                        "allowed_tools": ["execute_background_service"],
                        "proof_requirements": ["running_http_service"],
                        "failure_reason": "port busy",
                    },
                ]
            },
        ),
        _assistant_tool("repair_task_graph", {"reason": "choose a different port"}),
    ]

    graph = engine.latest_graph(messages)

    assert graph is not None
    by_id = {node["id"]: node for node in graph["nodes"]}
    assert by_id["write"]["status"] == "done"
    assert by_id["write"]["evidence_refs"] == ["call_write"]
    assert by_id["serve"]["status"] == "pending"
    assert by_id["serve"]["failure_reason"] == "choose a different port"


def test_failed_proof_with_no_active_node_allows_recovery_diagnostics() -> None:
    engine = TaskGraphEngine()
    messages = [
        {"role": "user", "content": "write then verify"},
        _assistant_tool(
            "set_task_graph",
            {
                "nodes": [
                    {
                        "id": "write",
                        "title": "Write artifact",
                        "kind": "write",
                        "status": "done",
                        "allowed_tools": ["write_text_file"],
                        "proof_requirements": ["filesystem_artifact"],
                        "evidence_refs": ["missing_write_ref"],
                    }
                ]
            },
        ),
    ]

    allowed = engine.allowed_tools_for_next(messages, [])

    assert "get_filesystem_process_evidence" in allowed
    assert "update_task_node" in allowed
    assert "repair_task_graph" in allowed
    assert "execute_terminal_command" in allowed


@pytest.mark.asyncio
async def test_gateway_task_graph_endpoints_use_engine() -> None:
    from gateway import app, get_task_graph, verify_task_graph

    snapshot = {"has_graph": True, "nodes": [{"id": "n1"}]}
    verified = {"passed": True, "missing_nodes": []}
    app.state.engine = SimpleNamespace(
        task_graph_snapshot=lambda session_id: snapshot if session_id == "s1" else None,
        verify_task_graph=lambda session_id: verified if session_id == "s1" else None,
    )

    assert await get_task_graph("s1") == snapshot
    assert await verify_task_graph("s1") == verified

    with pytest.raises(HTTPException):
        await get_task_graph("missing")
