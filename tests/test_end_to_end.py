from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import agent as agent_module  # noqa: E402
from agent import AgentEngine, NormalizedMessage  # noqa: E402
from gateway import Gateway, Message  # noqa: E402
from tools import ToolManager  # noqa: E402

SERVER_NAME = "sqlite"
USER_QUESTION = "What tables exist in the database?"


class _EmptyMemory:
    def retrieve_context(self, query: str, query_type: str) -> dict[str, Any]:
        print(f"Agent Engine fetched memory context: query={query!r}, type={query_type}")
        return {"query_type": query_type, "results": []}


class _ObservedToolManager(ToolManager):
    async def list_all_tools(self) -> list[dict[str, Any]]:
        tools = await super().list_all_tools()
        names = ", ".join(tool["name"] for tool in tools)
        print(f"Agent Engine fetched tools: {names}")
        return tools

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        print(f"Agent Engine executing tool: {server_name}.{tool_name}({arguments})")
        result = await super().call_tool(server_name, tool_name, arguments)
        rendered = "\n".join(
            getattr(block, "text", str(block)) for block in result.content
        )
        print(f"SQLite tool output: {rendered}")
        return result


def _find_sqlite_server() -> str:
    if command := shutil.which("mcp-server-sqlite"):
        return command

    scripts_dirs = sorted(
        (Path.home() / "AppData" / "Local" / "Python").glob(
            "pythoncore-*-64/Scripts/mcp-server-sqlite.exe"
        ),
        reverse=True,
    )
    if scripts_dirs:
        return str(scripts_dirs[0])

    pytest.skip(
        "mcp-server-sqlite not found. Install it with: python -m pip install mcp-server-sqlite"
    )


def _assert_users_table_exists(db_path: Path) -> None:
    assert db_path.exists(), f"Expected SQLite database at {db_path}"
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'users'"
        ).fetchall()
    assert rows, "Expected root test.db to contain a users table"


async def _fake_llm_completion(**kwargs: Any) -> Any:
    call_number = _fake_llm_completion.call_count + 1
    _fake_llm_completion.call_count = call_number

    if call_number == 1:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call-contract",
                                function=SimpleNamespace(
                                    name="set_task_contract",
                                    arguments=json.dumps(
                                        {
                                            "mode": "answer",
                                            "summary": "Answer a database schema question.",
                                            "success_criteria": [
                                                "The reply lists the database tables."
                                            ],
                                            "evidence_requirements": ["none"],
                                        }
                                    ),
                                ),
                            )
                        ],
                    ),
                )
            ]
        )

    if call_number == 2:
        tool_names = [
            tool["function"]["name"] for tool in kwargs.get("tools", [])
        ]
        print(f"Agent Engine ReAct loop 1: model sees tools={tool_names}")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call-list-tables",
                                function=SimpleNamespace(
                                    name="list_tables",
                                    arguments=json.dumps({}),
                                ),
                            )
                        ],
                    ),
                )
            ]
        )

    tool_messages = [
        message["content"]
        for message in kwargs["messages"]
        if message.get("role") == "tool"
    ]
    assert len(tool_messages) == 2, f"Expected two tool results, got {len(tool_messages)}"
    assert "users" in tool_messages[-1], f"Expected users table in tool output: {tool_messages[-1]}"

    final_answer = "The database contains one table: users."
    print(f"Agent Engine final answer: {final_answer}")
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=final_answer, tool_calls=None),
            )
        ]
    )


_fake_llm_completion.call_count = 0


async def run_end_to_end_test() -> None:
    db_path = PROJECT_ROOT / "test.db"
    _assert_users_table_exists(db_path)

    original_completion = agent_module.litellm.acompletion
    agent_module.litellm.acompletion = _fake_llm_completion
    _fake_llm_completion.call_count = 0

    async with _ObservedToolManager() as tools:
        print(f"Connecting SQLite MCP server to {db_path} ...")
        await tools.connect_server(
            name=SERVER_NAME,
            command=_find_sqlite_server(),
            args=["--db-path", str(db_path)],
        )
        print("SQLite MCP server connected.")

        engine = AgentEngine(memory=_EmptyMemory(), tools=tools, model="test-model")

        async def handler(message: Message) -> str:
            content = message.payload["content"]
            print(f"Gateway received message: {content!r}")
            return await engine.process_task(
                NormalizedMessage(
                    session_id=message.session_id,
                    role="user",
                    content=content,
                )
            )

        gateway = Gateway(handler)
        try:
            result = await gateway.send(
                "session-e2e",
                {"content": USER_QUESTION},
            )
        finally:
            await gateway.shutdown()
            agent_module.litellm.acompletion = original_completion

    assert result.error is None, result.error
    assert result.output == "The database contains one table: users."
    assert _fake_llm_completion.call_count == 3

    print(f"Gateway final output: {result.output}")
    print("ALL END-TO-END CHECKS PASSED")


def test_gateway_agent_sqlite_mcp_end_to_end() -> None:
    asyncio.run(run_end_to_end_test())


if __name__ == "__main__":
    asyncio.run(run_end_to_end_test())
