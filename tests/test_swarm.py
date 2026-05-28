from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import agent as agent_module  # noqa: E402
from agent import (  # noqa: E402
    NormalizedMessage,
    SubAgentOrchestrator,
    SubAgentTask,
    TypeSafeAgentEngine,
)
from tools import ToolManager  # noqa: E402

PROMPT = (
    "Write an optimized SQL query to pull the top users, then spawn a sub-agent "
    "auditor to verify that the query doesn't contain any SQL injection vulnerabilities."
)


class _EmptyMemory:
    def retrieve_context(self, query: str, query_type: str) -> dict[str, Any]:
        return {"query_type": query_type, "results": []}

    def store_event(
        self,
        session_id: str,
        raw_text: str,
        entities: dict[str, Any],
    ) -> str:
        return ""


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        self.messages.append(message)
        print(message)


def _completion_response(
    *,
    finish_reason: str,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                ),
            )
        ]
    )


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


async def _fake_completion(**kwargs: Any) -> Any:
    messages = kwargs["messages"]
    system_directive = messages[0]["content"]

    if "specialist security and correctness auditor" in system_directive:
        _fake_completion.auditor_invoked = True
        print("AUDITOR_SUB_AGENT_INVOKED")
        return _completion_response(
            finish_reason="stop",
            content=(
                "Audit result: PASS. The proposed query uses a fixed ORDER BY and "
                "LIMIT clause and contains no string interpolation or user-controlled "
                "SQL fragments."
            ),
        )

    contract_seen = any(
        message.get("role") == "tool"
        and message.get("tool_call_id") == "call_contract"
        for message in messages
    )
    if not contract_seen:
        return _completion_response(
            finish_reason="tool_calls",
            tool_calls=[
                _tool_call(
                    "call_contract",
                    "set_task_contract",
                    {
                        "mode": "answer",
                        "summary": "Write a SQL query and audit it with a sub-agent.",
                        "success_criteria": [
                            "The final answer includes the query and audit verdict."
                        ],
                        "evidence_requirements": ["none"],
                    },
                )
            ],
        )

    delegate_result_seen = any(
        message.get("role") == "tool"
        and message.get("tool_call_id") == "call_delegate_auditor"
        for message in messages
    )

    if not delegate_result_seen:
        query = (
            "SELECT id, name, score FROM users "
            "ORDER BY score DESC LIMIT 10"
        )
        return _completion_response(
            finish_reason="tool_calls",
            tool_calls=[
                _tool_call(
                    "call_delegate_auditor",
                    "delegate_task",
                    {
                        "agent_type": "auditor",
                        "task_description": (
                            "Verify that this SQL query contains no SQL injection "
                            "vulnerabilities and report any findings."
                        ),
                        "context_payload": {
                            "query": query,
                            "notes": (
                                "The parent generated this top-users query and needs "
                                "a security audit before presenting it."
                            ),
                        },
                    },
                )
            ],
        )

    return _completion_response(
        finish_reason="stop",
        content=(
            "Optimized query: SELECT id, name, score FROM users ORDER BY score DESC "
            "LIMIT 10. Auditor sub-agent verdict: PASS, no SQL injection "
            "vulnerabilities found."
        ),
    )


_fake_completion.auditor_invoked = False


async def run_swarm_test() -> None:
    original_completion = agent_module.litellm.acompletion
    agent_module.litellm.acompletion = _fake_completion
    _fake_completion.auditor_invoked = False

    log_capture = _LogCapture()
    log_capture.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    agent_module.logger.addHandler(log_capture)
    agent_module.logger.setLevel(logging.DEBUG)

    try:
        async with ToolManager() as tools:
            all_tools = await tools.list_all_tools()
            tool_names = {tool["name"] for tool in all_tools}
            print(f"TOOLS_AVAILABLE: {sorted(tool_names)}")
            assert "delegate_task" in tool_names

            engine = TypeSafeAgentEngine(
                memory=_EmptyMemory(),
                tools=tools,
                model="test-model",
            )

            events: list[dict[str, Any]] = []
            async for event in engine.stream_task(
                NormalizedMessage(
                    session_id="swarm-test-session",
                    role="user",
                    content=PROMPT,
                )
            ):
                events.append(event)
                print(f"EVENT: {event}")

        assert any(
            event.get("type") == "tool_call"
            and event.get("tool") == "delegate_task"
            and event.get("params", {}).get("agent_type") == "auditor"
            for event in events
        ), "parent agent did not call delegate_task with agent_type='auditor'"
        assert _fake_completion.auditor_invoked, "auditor sub-agent was not invoked"
        assert any(
            "Delegating to 'auditor' sub-agent" in message
            for message in log_capture.messages
        ), "expected delegation log for auditor sub-agent"
        assert any(
            event.get("type") == "text"
            and "Auditor sub-agent verdict: PASS" in event.get("content", "")
            for event in events
        ), "expected parent final answer to include auditor verdict"

        print("ALL SWARM CHECKS PASSED")
    finally:
        agent_module.logger.removeHandler(log_capture)
        agent_module.litellm.acompletion = original_completion


def test_parent_agent_invokes_auditor_sub_agent() -> None:
    asyncio.run(run_swarm_test())


async def _run_parallel_orchestrator_test() -> None:
    active = 0
    max_active = 0

    async def runner(task: SubAgentTask) -> str:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        return f"done:{task.task_id}"

    orchestrator = SubAgentOrchestrator(
        runner,
        max_concurrency=2,
        timeout_seconds=1,
    )
    tasks = [
        SubAgentTask(
            task_id=f"task-{i}",
            agent_type="researcher",
            task_description=f"task {i}",
            context_payload={},
        )
        for i in range(2)
    ]

    results = await orchestrator.run(tasks, mode="parallel")

    assert [result.result for result in results] == ["done:task-0", "done:task-1"]
    assert max_active == 2


def test_sub_agent_orchestrator_runs_parallel_tasks() -> None:
    asyncio.run(_run_parallel_orchestrator_test())


if __name__ == "__main__":
    asyncio.run(run_swarm_test())
