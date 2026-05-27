from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import agent as agent_module  # noqa: E402
from agent import AgentEngine, ExecutionStep, NormalizedMessage  # noqa: E402
from tools import (  # noqa: E402
    EXECUTE_TERMINAL_COMMAND_TOOL,
    EXPOSE_LOCAL_HTTP_SERVICE_TOOL,
    GET_SYSTEM_ENVIRONMENT_TOOL,
    INSPECT_TASK_GRAPH_TOOL,
    REPAIR_TASK_GRAPH_TOOL,
    SET_TASK_CONTRACT_TOOL,
    SET_TASK_GRAPH_TOOL,
    UPDATE_PLAN_TOOL,
    UPDATE_TASK_NODE_TOOL,
    VERIFY_TASK_GRAPH_TOOL,
    WRITE_TEXT_FILE_TOOL,
)


class _EmptyMemory:
    def retrieve_context(self, query: str, query_type: str) -> dict[str, Any]:
        return {"query_type": query_type, "results": []}

    def store_event(
        self, session_id: str, raw_text: str, entities: dict[str, Any]
    ) -> str:
        return ""


class _FakeTools:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.written: list[dict[str, Any]] = []
        self.served: list[dict[str, Any]] = []

    async def list_all_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "server": "filesystem",
                "name": "create_directory",
                "description": "Create a directory.",
                "inputSchema": {
                    "type": "object",
                    "required": ["path"],
                    "properties": {"path": {"type": "string"}},
                },
            },
            {
                "server": "filesystem",
                "name": "list_allowed_directories",
                "description": "List allowed directories.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            SET_TASK_CONTRACT_TOOL,
            UPDATE_PLAN_TOOL,
            SET_TASK_GRAPH_TOOL,
            INSPECT_TASK_GRAPH_TOOL,
            UPDATE_TASK_NODE_TOOL,
            REPAIR_TASK_GRAPH_TOOL,
            VERIFY_TASK_GRAPH_TOOL,
            GET_SYSTEM_ENVIRONMENT_TOOL,
            WRITE_TEXT_FILE_TOOL,
            EXECUTE_TERMINAL_COMMAND_TOOL,
            EXPOSE_LOCAL_HTTP_SERVICE_TOOL,
        ]

    async def execute_terminal_command(self, command: str) -> dict[str, Any]:
        self.commands.append(command)
        if "fail-on-purpose" in command:
            return {
                "exit_code": 1,
                "stdout": "SYSTEM ALERT: Command execution failed with error code 1.",
                "stderr": "simulated failure",
                "current_working_directory": "/tmp",
            }
        return {
            "exit_code": 0,
            "stdout": f"ok: {command}",
            "stderr": "",
            "current_working_directory": "/tmp",
        }

    def get_system_environment(self) -> str:
        return json.dumps(
            {
                "os": "Linux",
                "shell": {"shell": "bash", "posix": True},
                "runtimes": {"python3": True, "java": False},
                "user": {"is_root": True, "sudo_available": False},
                "sandbox": {"mode": "docker", "container_workdir": "/workspace"},
            }
        )

    def write_text_file(self, path: str, content: str) -> str:
        payload = {
            "written": True,
            "path": path,
            "exists": True,
            "size_bytes": len(content.encode("utf-8")),
        }
        self.written.append(payload)
        return json.dumps(payload)

    def expose_local_http_service(
        self, port: int, path: str = "", name: str | None = None
    ) -> str:
        payload = {
            "exposed": True,
            "connectable": True,
            "port": port,
            "path": path or "/",
            "name": name or f"local-http-{port}",
            "url": f"http://localhost:8000/proxy/{port}/{path.lstrip('/')}",
        }
        self.served.append(payload)
        return json.dumps(payload)


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _completion(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls" if tool_calls else "stop",
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ]
    )


def _chunk(response: Any, *, content: str | None = None, tool_calls: Any = None) -> Any:
    return SimpleNamespace(
        _response=response,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ],
    )


async def _stream_response(response: Any) -> AsyncIterator[Any]:
    message = response.choices[0].message
    if message.tool_calls:
        yield _chunk(response, tool_calls=message.tool_calls)
        return
    for token in str(message.content or "").split(" "):
        yield _chunk(response, content=token + " ")


class _ScriptedStreamingModel:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls = 0
        self.request_messages: list[list[dict[str, Any]]] = []
        self.request_tool_names: list[list[str]] = []
        self.request_tool_choices: list[Any] = []
        self.request_parallel_tool_calls: list[Any] = []

    async def __call__(self, **kwargs: Any) -> Any:
        messages = kwargs["messages"]
        _assert_provider_safe_messages(messages)
        self.request_messages.append(messages)
        self.request_tool_names.append(
            [
                tool.get("function", {}).get("name")
                for tool in kwargs.get("tools", [])
            ]
        )
        self.request_tool_choices.append(kwargs.get("tool_choice"))
        self.request_parallel_tool_calls.append(kwargs.get("parallel_tool_calls"))
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return _stream_response(response)


def _stream_chunk_builder(chunks: list[Any], messages: Any = None) -> Any:
    return chunks[-1]._response


def _assert_provider_safe_messages(messages: list[dict[str, Any]]) -> None:
    assert any(
        message.get("role") != "system" for message in messages
    ), f"provider request has no non-system messages: {messages}"
    for message in messages:
        content = message.get("content")
        role = message.get("role")
        if isinstance(content, str):
            assert content.strip(), f"empty string content sent for {role}: {message}"
            continue
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    assert str(block.get("text") or "").strip(), (
                        f"empty text block sent for {role}: {message}"
                    )
            continue
        if content is None:
            assert role == "assistant" and message.get("tool_calls"), (
                f"null content sent without assistant tool_calls: {message}"
            )


async def _run_engine_with_model(
    script: list[Any],
) -> tuple[list[dict[str, Any]], _FakeTools, _ScriptedStreamingModel]:
    model = _ScriptedStreamingModel(script)
    tools = _FakeTools()
    original_completion = agent_module.litellm.acompletion
    original_builder = agent_module.litellm.stream_chunk_builder
    agent_module.litellm.acompletion = model
    agent_module.litellm.stream_chunk_builder = _stream_chunk_builder
    try:
        engine = AgentEngine(memory=_EmptyMemory(), tools=tools, model="test-model")
        events: list[dict[str, Any]] = []
        async for event in engine.stream_task(
            NormalizedMessage(
                session_id="contract-test",
                role="user",
                content="make a simple website about the importance of sleep",
            )
        ):
            events.append(event)
        return events, tools, model
    finally:
        agent_module.litellm.acompletion = original_completion
        agent_module.litellm.stream_chunk_builder = original_builder


async def _run_engine_with_script(script: list[Any]) -> tuple[list[dict[str, Any]], _FakeTools]:
    events, tools, _ = await _run_engine_with_model(script)
    return events, tools


def _contract_tool(mode: str, evidence: list[str]) -> Any:
    summary = (
        "Serve an HTTP result for the user request."
        if "running_http_service" in evidence
        else "Handle the user request."
    )
    return _tool_call(
        "call_contract",
        "set_task_contract",
        {
            "mode": mode,
            "summary": summary,
            "success_criteria": ["The requested outcome is complete."],
            "evidence_requirements": evidence,
        },
    )


def _plan_tool(status: str) -> Any:
    return _tool_call(
        "call_plan",
        "update_plan",
        {
            "steps": [
                {"title": "Create website files", "status": status},
                {"title": "Serve static site", "status": status},
            ]
        },
    )


def test_execute_contract_suppresses_rejected_streamed_prose() -> None:
    script = [
        _completion(tool_calls=[_contract_tool("execute", ["running_http_service"])]),
        _completion(tool_calls=[_plan_tool("in_progress")]),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_mkdir",
                    "execute_terminal_command",
                    {"command": "mkdir -p /tmp/sleep-website"},
                )
            ]
        ),
        _completion(content="I'll create the sleep website now."),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_write",
                    "execute_terminal_command",
                    {"command": "cat > /tmp/sleep-website/index.html"},
                )
            ]
        ),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_serve",
                    "expose_local_http_service",
                    {"port": 8765},
                )
            ]
        ),
        _completion(tool_calls=[_plan_tool("done")]),
        _completion(content="Served: http://localhost:8000/proxy/8765/sleep-importance/"),
    ]

    events, tools = asyncio.run(_run_engine_with_script(script))

    streamed = "".join(
        event.get("content", "") for event in events if event.get("type") == "token"
    )
    assert "I'll create the sleep website now" not in streamed
    assert any(
        event.get("type") == "tool_call"
        and event.get("tool") == "expose_local_http_service"
        for event in events
    )
    assert any(
        event.get("type") == "text"
        and "sleep-importance" in event.get("content", "")
        for event in events
    )
    assert any("mkdir -p /tmp/sleep-website" in command for command in tools.commands)
    assert tools.served and tools.served[-1]["connectable"] is True


def test_answer_contract_streams_tokens_immediately() -> None:
    script = [
        _completion(tool_calls=[_contract_tool("answer", ["none"])]),
        _completion(content="Sleep supports memory and recovery."),
    ]

    events, _ = asyncio.run(_run_engine_with_script(script))
    streamed = "".join(
        event.get("content", "") for event in events if event.get("type") == "token"
    )

    assert "Sleep supports memory" in streamed
    assert any(
        event.get("type") == "text"
        and "Sleep supports memory" in event.get("content", "")
        for event in events
    )


def test_empty_tool_call_content_is_sanitized_before_next_request() -> None:
    script = [
        _completion(content="", tool_calls=[_contract_tool("answer", ["none"])]),
        _completion(content="Done."),
    ]

    events, _ = asyncio.run(_run_engine_with_script(script))

    assert any(
        event.get("type") == "text" and event.get("content") == "Done."
        for event in events
    )


def test_empty_rejected_final_text_does_not_poison_next_request() -> None:
    script = [
        _completion(tool_calls=[_contract_tool("execute", ["running_http_service"])]),
        _completion(tool_calls=[_plan_tool("in_progress")]),
        _completion(content=""),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_serve",
                    "expose_local_http_service",
                    {"port": 8765},
                )
            ]
        ),
        _completion(tool_calls=[_plan_tool("done")]),
        _completion(content="Served: http://localhost:8000/proxy/8765/sleep-importance/"),
    ]

    events, _ = asyncio.run(_run_engine_with_script(script))

    assert any(
        event.get("type") == "text"
        and "sleep-importance" in event.get("content", "")
        for event in events
    )


def test_provider_request_gets_transient_user_when_history_sanitizes_to_system_only() -> None:
    prepared = agent_module._prepare_llm_request_messages(
        [
            {"role": "system", "content": "System directive."},
            {"role": "assistant", "content": ""},
            {"role": "system", "content": "Continue silently."},
        ],
        "create a simple website about sleep",
    )

    _assert_provider_safe_messages(prepared)
    assert prepared[-1]["role"] == "user"
    assert "create a simple website about sleep" in prepared[-1]["content"]


def test_update_plan_accepts_json_string_steps() -> None:
    content, is_error = agent_module._run_update_plan(
        {
            "steps": json.dumps(
                [
                    {"title": "Create files", "status": "in_progress"},
                    {"title": "Serve", "status": "pending"},
                ]
            )
        }
    )

    assert is_error is False
    assert "2 step(s)" in content


def test_execute_contract_narrows_tools_to_evidence_producers() -> None:
    script = [
        _completion(tool_calls=[_contract_tool("execute", ["running_http_service"])]),
        _completion(tool_calls=[_plan_tool("in_progress")]),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_write",
                    "write_text_file",
                    {
                        "path": "sleep-website/index.html",
                        "content": "<!doctype html><title>Sleep</title>",
                    },
                )
            ]
        ),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_serve",
                    "expose_local_http_service",
                    {"port": 8765},
                )
            ]
        ),
        _completion(tool_calls=[_plan_tool("done")]),
        _completion(content="Served: http://localhost:8000/proxy/8765/sleep-importance/"),
    ]

    events, tools, model = asyncio.run(_run_engine_with_model(script))

    assert model.request_tool_names[0] == ["set_task_contract"]
    assert model.request_tool_names[1] == [
        "update_plan",
        "set_task_graph",
        "inspect_task_graph",
    ]
    evidence_tool_names = set(model.request_tool_names[2])
    assert "write_text_file" in evidence_tool_names
    assert "expose_local_http_service" in evidence_tool_names
    assert "execute_terminal_command" in evidence_tool_names
    assert "create_directory" not in evidence_tool_names
    assert "list_allowed_directories" not in evidence_tool_names
    assert model.request_tool_choices[2] == "required"
    assert tools.written
    assert any(
        event.get("type") == "text" and "sleep-importance" in event.get("content", "")
        for event in events
    )


def test_explicit_task_graph_replaces_plan_and_narrows_to_active_node() -> None:
    script = [
        _completion(tool_calls=[_contract_tool("execute", ["filesystem_artifact"])]),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_graph",
                    "set_task_graph",
                    {
                        "nodes": [
                            {
                                "id": "write",
                                "title": "Write sleep artifact",
                                "kind": "write",
                                "status": "in_progress",
                                "allowed_tools": ["write_text_file"],
                                "proof_requirements": ["filesystem_artifact"],
                            }
                        ]
                    },
                )
            ]
        ),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_write",
                    "write_text_file",
                    {
                        "path": "sleep-graph/index.html",
                        "content": "<!doctype html><title>Sleep graph</title>",
                    },
                )
            ]
        ),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_done",
                    "update_task_node",
                    {
                        "node_id": "write",
                        "status": "done",
                        "evidence_refs": ["call_write"],
                    },
                )
            ]
        ),
        _completion(content="Created sleep-graph/index.html."),
    ]

    events, tools, model = asyncio.run(_run_engine_with_model(script))

    assert model.request_tool_names[1] == [
        "update_plan",
        "set_task_graph",
        "inspect_task_graph",
    ]
    graph_limited = set(model.request_tool_names[2])
    assert "write_text_file" in graph_limited
    assert "execute_terminal_command" in graph_limited
    assert tools.written
    assert any(
        event.get("type") == "text" and "sleep-graph" in event.get("content", "")
        for event in events
    )


def _assistant_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def _contract_messages(plan_status: str = "done") -> list[dict[str, Any]]:
    contract_args = {
        "mode": "execute",
        "summary": "Serve a sleep website.",
        "success_criteria": ["A browser URL serves the site."],
        "evidence_requirements": ["running_http_service"],
    }
    return [
        {"role": "user", "content": "make a simple website about sleep"},
        _assistant_tool("set_task_contract", contract_args),
        _assistant_tool(
            "update_plan",
            {"steps": [{"title": "Serve", "status": plan_status}]},
        ),
    ]


def test_execute_contract_requires_matching_structured_evidence() -> None:
    messages = _contract_messages()
    contract = agent_module.latest_task_contract(messages)
    mkdir_step = ExecutionStep(
        kind="tool_result",
        content=json.dumps({"exit_code": 0, "stdout": "created", "stderr": ""}),
        metadata={
            "tool_name": "execute_terminal_command",
            "is_error": False,
            "arguments": {"command": "mkdir -p /tmp/sleep-website"},
        },
    )
    serve_step = ExecutionStep(
        kind="tool_result",
        content=json.dumps(
            {
                "exposed": True,
                "connectable": True,
                "port": 8765,
                "url": "http://localhost:8000/proxy/8765/",
            }
        ),
        metadata={
            "tool_name": "expose_local_http_service",
            "is_error": False,
            "arguments": {"port": 8765},
        },
    )
    write_step = ExecutionStep(
        kind="tool_result",
        content=json.dumps({"written": True, "exists": True, "size_bytes": 10}),
        metadata={"tool_name": "write_text_file", "is_error": False},
    )

    mkdir_status = agent_module.contract_completion_status(
        contract, messages, [mkdir_step], contract_required=True
    )
    serve_status = agent_module.contract_completion_status(
        contract, messages, [mkdir_step, serve_step], contract_required=True
    )
    open_plan_status = agent_module.contract_completion_status(
        contract, _contract_messages(plan_status="in_progress"), [serve_step], contract_required=True
    )
    dual_evidence_status = agent_module.contract_completion_status(
        {
            **contract,
            "evidence_requirements": [
                "filesystem_artifact",
                "running_http_service",
            ],
        },
        messages,
        [write_step, serve_step],
        contract_required=True,
    )

    assert mkdir_status["complete"] is False
    assert "running_http_service" in mkdir_status["missing"]
    assert serve_status["complete"] is True
    assert dual_evidence_status["complete"] is True
    assert open_plan_status["complete"] is False
    assert "plan_open_steps" in open_plan_status["missing"]


def test_parallel_tool_calls_disabled_during_contract_execution() -> None:
    """parallel_tool_calls=False must be sent to the model on every turn where
    must_set_contract or needs_execution is true.

    Regression: the model previously emitted two update_plan calls in the same
    turn (parallel tool calls), collapsing a 3-step plan to a 1-step plan before
    any result was seen.  Setting parallel_tool_calls=False forces sequential
    execution so only one call is made per turn.
    """
    script = [
        _completion(tool_calls=[_contract_tool("execute", ["running_http_service"])]),
        _completion(tool_calls=[_plan_tool("in_progress")]),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_write",
                    "write_text_file",
                    {"path": "/workspace/index.html", "content": "<html>Sleep</html>"},
                )
            ]
        ),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_serve",
                    "expose_local_http_service",
                    {"port": 8765},
                )
            ]
        ),
        _completion(tool_calls=[_plan_tool("done")]),
        _completion(content="Served: http://localhost:8000/proxy/8765/sleep/"),
    ]

    events, tools, model = asyncio.run(_run_engine_with_model(script))

    # Every contract-enforced turn must have parallel_tool_calls=False.
    # Turn 0: must_set_contract -> False
    # Turn 1: needs_execution (plan missing) -> False
    # Turn 2: needs_execution (evidence missing) -> False
    # Turn 3: needs_execution (evidence missing) -> False
    # Turn 4: needs_execution (plan still open) -> False
    # Turn 5: final answer, not enforced -> None (absent)
    for i, ptc in enumerate(model.request_parallel_tool_calls[:-1]):
        assert ptc is False, (
            f"Turn {i}: expected parallel_tool_calls=False during contract enforcement, "
            f"got {ptc!r}"
        )
    # Last turn (final answer generation) must NOT have the flag set.
    assert model.request_parallel_tool_calls[-1] is None, (
        "Final answer turn must not force parallel_tool_calls=False"
    )
    # The task must still complete successfully end-to-end.
    assert tools.served
    assert any(
        event.get("type") == "text" and "sleep" in event.get("content", "")
        for event in events
    )


def test_contract_recovery_forces_next_evidence_tool_after_rejected_prose() -> None:
    """After rejected prose, the next execute turn is narrowed to one evidence tool."""
    script = [
        _completion(
            tool_calls=[
                _tool_call(
                    "call_contract",
                    "set_task_contract",
                    {
                        "mode": "execute",
                        "summary": "Create a simple website about sleep.",
                        "success_criteria": [
                            "Website is served and accessible at a public URL",
                            "Content about sleep is displayed and interactive",
                        ],
                        "evidence_requirements": [
                            "running_http_service",
                            "filesystem_artifact",
                        ],
                    },
                )
            ]
        ),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_plan",
                    "update_plan",
                    {
                        "steps": [
                            {
                                "title": "Create index.html with sleep content",
                                "status": "in_progress",
                            },
                            {"title": "Serve the website", "status": "pending"},
                            {"title": "Verify the served URL", "status": "pending"},
                        ]
                    },
                )
            ]
        ),
        _completion(content="I'll create the sleep website now."),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_write",
                    "write_text_file",
                    {
                        "path": "generated_sites/sleep/index.html",
                        "content": "<!doctype html><title>Sleep</title>",
                    },
                )
            ]
        ),
        _completion(content="I'll serve it next."),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_serve",
                    "expose_local_http_service",
                    {"port": 8765},
                )
            ]
        ),
        _completion(tool_calls=[_plan_tool("done")]),
        _completion(content="Served: http://localhost:8000/proxy/8765/sleep/"),
    ]

    events, tools, model = asyncio.run(_run_engine_with_model(script))

    assert model.request_tool_names[3] in (
        ["write_text_file"],
        ["expose_local_http_service"],
    )
    assert tools.written
    assert tools.served
    assert any("sleep" in item["path"] for item in tools.written)
    event_tools = [event.get("tool") for event in events if event.get("type") == "tool_call"]
    assert "write_text_file" in event_tools
    assert "expose_local_http_service" in event_tools
    assert any(
        event.get("type") == "text" and "sleep" in event.get("content", "").lower()
        for event in events
    )


def test_completed_static_site_final_answer_includes_literal_url() -> None:
    script = [
        _completion(tool_calls=[_contract_tool("execute", ["running_http_service"])]),
        _completion(tool_calls=[_plan_tool("in_progress")]),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_write",
                    "write_text_file",
                    {
                        "path": "generated_sites/site/index.html",
                        "content": "<!doctype html><title>Sleep</title>",
                    },
                )
            ]
        ),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_serve",
                    "expose_local_http_service",
                    {"port": 8765},
                )
            ]
        ),
        _completion(tool_calls=[_plan_tool("done")]),
        _completion(content="The website is ready: Importance of Sleep."),
    ]

    events, tools, _ = asyncio.run(_run_engine_with_model(script))

    assert tools.served
    final_texts = [
        event.get("content", "")
        for event in events
        if event.get("type") in {"text", "final_answer"}
    ]
    assert final_texts
    assert "Evidence:" in final_texts[-1]
    assert "Service URL: http://localhost:8000/proxy/8765/" in final_texts[-1]
    assert "Port: 8765" in final_texts[-1]


def test_url_followup_recovers_served_url_from_history() -> None:
    messages = [
        {"role": "user", "content": "create a website"},
        {
            "role": "tool",
            "tool_call_id": "call_serve",
            "content": json.dumps(
                {
                    "exposed": True,
                    "connectable": True,
                    "port": 8765,
                    "url": "http://localhost:8000/proxy/8765/",
                }
            ),
        },
        {"role": "assistant", "content": "I served the website."},
        {"role": "user", "content": "give me the url"},
    ]

    final = agent_module._ensure_evidence_in_final_response(
        "The URL is: Importance of Sleep.",
        contract={"mode": "answer", "evidence_requirements": ["none"]},
        original_prompt="give me the url",
        steps=[],
        messages=messages,
    )

    assert "Service URL: http://localhost:8000/proxy/8765/" in final


def test_final_answer_appends_general_evidence_literals() -> None:
    steps = [
        ExecutionStep(
            kind="tool_result",
            content=json.dumps(
                {
                    "written": True,
                    "path": "/tmp/report.txt",
                    "exists": True,
                    "size_bytes": 2,
                }
            ),
            metadata={
                "tool_name": "write_text_file",
                "is_error": False,
                "arguments": {"path": "/tmp/report.txt"},
            },
        ),
        ExecutionStep(
            kind="tool_result",
            content=json.dumps(
                {
                    "exit_code": 0,
                    "stdout": "installed ok\n",
                    "stderr": "",
                    "current_working_directory": "/tmp",
                }
            ),
            metadata={
                "tool_name": "execute_terminal_command",
                "is_error": False,
                "arguments": {"command": "echo installed ok"},
            },
        ),
    ]

    final = agent_module._ensure_evidence_in_final_response(
        "Done.",
        contract={
            "mode": "execute",
            "evidence_requirements": ["filesystem_artifact", "command_output"],
        },
        original_prompt="create the file and show command output",
        steps=steps,
        messages=[],
    )

    assert "File: /tmp/report.txt" in final
    assert "Command: echo installed ok" in final
    assert "Command output: installed ok" in final


def test_contract_recovery_forces_command_tool_for_command_output_evidence() -> None:
    script = [
        _completion(tool_calls=[_contract_tool("execute", ["command_output"])]),
        _completion(tool_calls=[_plan_tool("in_progress")]),
        _completion(content="I'll run the command now."),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_command",
                    "execute_terminal_command",
                    {"command": "echo recovered"},
                )
            ]
        ),
        _completion(tool_calls=[_plan_tool("done")]),
        _completion(content="Command output captured."),
    ]

    events, tools, model = asyncio.run(_run_engine_with_model(script))

    assert model.request_tool_names[3] == ["execute_terminal_command"]
    assert any("echo recovered" in command for command in tools.commands)
    assert any(
        event.get("type") == "text" and "Command output" in event.get("content", "")
        for event in events
    )


def test_failed_tool_result_is_streamed_and_triggers_self_repair_instruction() -> None:
    script = [
        _completion(tool_calls=[_contract_tool("execute", ["command_output"])]),
        _completion(tool_calls=[_plan_tool("in_progress")]),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_fail",
                    "execute_terminal_command",
                    {"command": "fail-on-purpose"},
                )
            ]
        ),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_env",
                    "get_system_environment",
                    {},
                )
            ]
        ),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_recover",
                    "execute_terminal_command",
                    {"command": "echo recovered"},
                )
            ]
        ),
        _completion(tool_calls=[_plan_tool("done")]),
        _completion(content="Recovered after reading the tool failure."),
    ]

    events, tools, model = asyncio.run(_run_engine_with_model(script))

    failed_events = [
        event
        for event in events
        if event.get("type") == "tool_result" and event.get("is_error")
    ]
    assert failed_events
    assert "simulated failure" in failed_events[0].get("content", "")
    assert any("echo recovered" in command for command in tools.commands)
    assert any(
        "SELF-REPAIR MODE" in str(message.get("content", ""))
        for request in model.request_messages
        for message in request
    )
