from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import agent as agent_module
from agent import AgentEngine, ExecutionStep, NormalizedMessage
from tools import (
    EXECUTE_TERMINAL_COMMAND_TOOL,
    PUBLISH_STATIC_SITE_TOOL,
    SET_TASK_CONTRACT_TOOL,
    UPDATE_PLAN_TOOL,
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
        self.published: list[dict[str, Any]] = []

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
            WRITE_TEXT_FILE_TOOL,
            EXECUTE_TERMINAL_COMMAND_TOOL,
            PUBLISH_STATIC_SITE_TOOL,
        ]

    async def execute_terminal_command(self, command: str) -> dict[str, Any]:
        self.commands.append(command)
        return {
            "exit_code": 0,
            "stdout": f"ok: {command}",
            "stderr": "",
            "current_working_directory": "/tmp",
        }

    def write_text_file(self, path: str, content: str) -> str:
        payload = {
            "written": True,
            "path": path,
            "exists": True,
            "size_bytes": len(content.encode("utf-8")),
        }
        self.written.append(payload)
        return json.dumps(payload)

    def publish_static_site(
        self, source_path: str | None = None, slug: str | None = None
    ) -> str:
        payload = {
            "published": True,
            "source_path": source_path,
            "published_path": f"/published/{slug or 'site'}",
            "url": f"http://localhost:8000/sites/{slug or 'site'}/",
            "index_exists": True,
            "files": ["index.html"],
        }
        self.published.append(payload)
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
    return _tool_call(
        "call_contract",
        "set_task_contract",
        {
            "mode": mode,
            "summary": "Handle the user request.",
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
                {"title": "Publish static site", "status": status},
            ]
        },
    )


def test_execute_contract_suppresses_rejected_streamed_prose() -> None:
    script = [
        _completion(tool_calls=[_contract_tool("execute", ["published_static_site_url"])]),
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
                    "call_publish",
                    "publish_static_site",
                    {
                        "source_path": "/tmp/sleep-website",
                        "slug": "sleep-importance",
                    },
                )
            ]
        ),
        _completion(tool_calls=[_plan_tool("done")]),
        _completion(content="Published: http://localhost:8000/sites/sleep-importance/"),
    ]

    events, tools = asyncio.run(_run_engine_with_script(script))

    streamed = "".join(
        event.get("content", "") for event in events if event.get("type") == "token"
    )
    assert "I'll create the sleep website now" not in streamed
    assert any(
        event.get("type") == "tool_call"
        and event.get("tool") == "publish_static_site"
        for event in events
    )
    assert any(
        event.get("type") == "text"
        and "sleep-importance" in event.get("content", "")
        for event in events
    )
    assert any("mkdir -p /tmp/sleep-website" in command for command in tools.commands)
    assert tools.published and tools.published[-1]["index_exists"] is True


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
        _completion(tool_calls=[_contract_tool("execute", ["published_static_site_url"])]),
        _completion(tool_calls=[_plan_tool("in_progress")]),
        _completion(content=""),
        _completion(
            tool_calls=[
                _tool_call(
                    "call_publish",
                    "publish_static_site",
                    {
                        "source_path": "/tmp/sleep-website",
                        "slug": "sleep-importance",
                    },
                )
            ]
        ),
        _completion(tool_calls=[_plan_tool("done")]),
        _completion(content="Published: http://localhost:8000/sites/sleep-importance/"),
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
                    {"title": "Publish", "status": "pending"},
                ]
            )
        }
    )

    assert is_error is False
    assert "2 step(s)" in content


def test_execute_contract_narrows_tools_to_evidence_producers() -> None:
    script = [
        _completion(tool_calls=[_contract_tool("execute", ["published_static_site_url"])]),
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
                    "call_publish",
                    "publish_static_site",
                    {
                        "source_path": "sleep-website",
                        "slug": "sleep-importance",
                    },
                )
            ]
        ),
        _completion(tool_calls=[_plan_tool("done")]),
        _completion(content="Published: http://localhost:8000/sites/sleep-importance/"),
    ]

    events, tools, model = asyncio.run(_run_engine_with_model(script))

    assert model.request_tool_names[0] == ["set_task_contract"]
    assert model.request_tool_names[1] == ["update_plan"]
    evidence_tool_names = set(model.request_tool_names[2])
    assert "write_text_file" in evidence_tool_names
    assert "publish_static_site" in evidence_tool_names
    assert "execute_terminal_command" in evidence_tool_names
    assert "create_directory" not in evidence_tool_names
    assert "list_allowed_directories" not in evidence_tool_names
    assert model.request_tool_choices[2] == "required"
    assert tools.written
    assert any(
        event.get("type") == "text" and "sleep-importance" in event.get("content", "")
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
        "summary": "Publish a sleep website.",
        "success_criteria": ["A browser URL serves the site."],
        "evidence_requirements": ["published_static_site_url"],
    }
    return [
        {"role": "user", "content": "make a simple website about sleep"},
        _assistant_tool("set_task_contract", contract_args),
        _assistant_tool(
            "update_plan",
            {"steps": [{"title": "Publish", "status": plan_status}]},
        ),
    ]


def test_execute_contract_requires_matching_structured_evidence() -> None:
    messages = _contract_messages()
    contract = agent_module._latest_task_contract(messages)
    mkdir_step = ExecutionStep(
        kind="tool_result",
        content=json.dumps({"exit_code": 0, "stdout": "created", "stderr": ""}),
        metadata={
            "tool_name": "execute_terminal_command",
            "is_error": False,
            "arguments": {"command": "mkdir -p /tmp/sleep-website"},
        },
    )
    publish_step = ExecutionStep(
        kind="tool_result",
        content=json.dumps(
            {
                "published": True,
                "index_exists": True,
                "url": "http://localhost:8000/sites/sleep/",
            }
        ),
        metadata={
            "tool_name": "publish_static_site",
            "is_error": False,
            "arguments": {"source_path": "/tmp/sleep-website"},
        },
    )

    mkdir_status = agent_module._contract_completion_status(
        contract, messages, [mkdir_step], contract_required=True
    )
    publish_status = agent_module._contract_completion_status(
        contract, messages, [mkdir_step, publish_step], contract_required=True
    )
    open_plan_status = agent_module._contract_completion_status(
        contract, _contract_messages(plan_status="in_progress"), [publish_step], contract_required=True
    )
    dual_evidence_status = agent_module._contract_completion_status(
        {
            **contract,
            "evidence_requirements": [
                "filesystem_artifact",
                "published_static_site_url",
            ],
        },
        messages,
        [publish_step],
        contract_required=True,
    )

    assert mkdir_status["complete"] is False
    assert "published_static_site_url" in mkdir_status["missing"]
    assert publish_status["complete"] is True
    assert dual_evidence_status["complete"] is True
    assert open_plan_status["complete"] is False
    assert "plan_open_steps" in open_plan_status["missing"]
