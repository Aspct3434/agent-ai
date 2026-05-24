from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools import ToolManager  # noqa: E402

SERVER_NAME = "sqlite"
REQUIRED_TOOLS = {"read_query", "list_tables"}


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


async def run_tools_test() -> None:
    db_dir = Path(tempfile.mkdtemp(prefix="mcp-sqlite-test-"))
    db_path = db_dir / "test.db"
    command = _find_sqlite_server()

    try:
        async with ToolManager() as tm:
            print(f"Connecting to MCP SQLite server (db={db_path}) ...")
            await tm.connect_server(
                name=SERVER_NAME,
                command=command,
                args=["--db-path", str(db_path)],
            )
            print("Connected.\n")

            tools = await tm.list_all_tools()

            assert tools, "Expected at least one tool from the SQLite MCP server"
            tool_names = {tool["name"] for tool in tools}
            missing = REQUIRED_TOOLS - tool_names
            assert not missing, f"Missing expected SQLite MCP tools: {sorted(missing)}"

            print(f"Found {len(tools)} tool(s) on server '{SERVER_NAME}':\n")
            for tool in tools:
                print(f"  [{tool['server']}] {tool['name']}")
                if desc := tool.get("description"):
                    print(f"    description : {desc}")
                print(
                    "    inputSchema : "
                    + json.dumps(tool["inputSchema"], indent=6).replace(
                        "\n", "\n" + " " * 18
                    )
                )
                if out := tool.get("outputSchema"):
                    print(
                        "    outputSchema: "
                        + json.dumps(out, indent=6).replace("\n", "\n" + " " * 18)
                    )
                print()

            print("ALL TOOL CHECKS PASSED")
    finally:
        shutil.rmtree(db_dir, ignore_errors=True)


def test_sqlite_mcp_server_lists_tools() -> None:
    asyncio.run(run_tools_test())


if __name__ == "__main__":
    asyncio.run(run_tools_test())
