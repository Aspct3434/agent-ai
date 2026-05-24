"""Run live agentic smoke tasks through the same WebSocket path as the UI.

This is intentionally not a mocked unit test. It connects to the running
FastAPI backend, sends real chat messages, records the streamed tool trace, and
checks for framework-level regressions such as host-shell commands, host-scope
execution, stale negative evidence without a repair action, and wrong evidence
types for non-HTTP services.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import websockets


HOST_COMMAND_RE = re.compile(
    r"(^|\s)(cmd|powershell|pwsh)(\.exe)?(\s|$)|"
    r"[A-Za-z]:\\|%USERPROFILE%|%APPDATA%|\\Users\\",
    re.IGNORECASE,
)

STATE_CHANGING_TOOLS = {
    "execute_terminal_command",
    "execute_background_service",
    "write_text_file",
    "create_table",
    "write_query",
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    prompt: str
    max_seconds: int
    forbid_tools: set[str] = field(default_factory=set)
    require_any_tool: set[str] = field(default_factory=set)
    require_contract_evidence: set[str] = field(default_factory=set)


TASKS: list[TaskSpec] = [
    TaskSpec(
        name="file-artifact",
        prompt=(
            "Create /workspace/live-smoke/file-task/hello.txt containing exactly "
            "'hello from live smoke' and verify the file exists. Use the active "
            "execution environment only."
        ),
        max_seconds=180,
        require_any_tool={"execute_terminal_command", "write_text_file"},
        require_contract_evidence={"filesystem_artifact"},
    ),
    TaskSpec(
        name="tcp-web-service",
        prompt=(
            "Create /workspace/live-smoke/web/index.html, start a simple Python "
            "HTTP server on TCP port 8765 serving that directory, and verify the "
            "port is listening. Use the active execution environment only."
        ),
        max_seconds=240,
        require_any_tool={"execute_background_service"},
    ),
]

def _event_line(event: dict[str, Any]) -> str:
    etype = event.get("type")
    if etype == "tool_call":
        params = event.get("params") or {}
        command = params.get("command")
        suffix = f" command={command!r}" if command else ""
        return f"tool_call {event.get('tool')}{suffix}"
    if etype == "tool_result":
        meta = event.get("metadata") or {}
        return (
            f"tool_result {event.get('tool')} error={event.get('is_error')} "
            f"meta={meta}"
        )
    if etype == "final_answer":
        return f"final_answer reason={event.get('reason')}"
    if etype == "error":
        return f"error {event.get('detail')}"
    return str(etype)


def _contract_evidence_from_events(events: list[dict[str, Any]]) -> set[str]:
    evidence: set[str] = set()
    for event in events:
        if event.get("type") != "tool_call" or event.get("tool") != "set_task_contract":
            if event.get("type") == "tool_result" and event.get("tool") == "set_task_contract":
                try:
                    payload = json.loads(str(event.get("content") or "{}"))
                except json.JSONDecodeError:
                    payload = {}
                for item in payload.get("evidence_requirements") or []:
                    evidence.add(str(item))
            continue
        params = event.get("params") or {}
        for item in params.get("evidence_requirements") or []:
            evidence.add(str(item))
    return evidence


def _validate_events(spec: TaskSpec, events: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    tools_called = {
        str(event.get("tool"))
        for event in events
        if event.get("type") == "tool_call" and event.get("tool")
    }

    for event in events:
        if event.get("type") == "tool_call":
            tool = str(event.get("tool"))
            if tool in spec.forbid_tools:
                errors.append(f"forbidden tool used: {tool}")
            params = event.get("params") or {}
            command = str(params.get("command") or "")
            if command and HOST_COMMAND_RE.search(command):
                errors.append(f"host-shaped command leaked through: {command}")
        elif event.get("type") == "tool_result":
            meta = event.get("metadata") or {}
            if meta.get("scope") == "host":
                errors.append(f"tool ran with host scope: {event.get('tool')}")
        elif event.get("type") == "error":
            errors.append(f"websocket error event: {event.get('detail')}")

    if spec.require_any_tool and not (tools_called & spec.require_any_tool):
        required = ", ".join(sorted(spec.require_any_tool))
        errors.append(f"none of required tools were called: {required}")

    contract_evidence = _contract_evidence_from_events(events)
    missing_evidence = spec.require_contract_evidence - contract_evidence
    if missing_evidence:
        errors.append(
            "contract missed required evidence: "
            + ", ".join(sorted(missing_evidence))
        )

    for index, event in enumerate(events):
        if (
            event.get("type") == "tool_result"
            and event.get("tool") == "get_filesystem_process_evidence"
            and event.get("is_error") is True
        ):
            later_tools = {
                str(later.get("tool"))
                for later in events[index + 1 :]
                if later.get("type") == "tool_call"
            }
            if not (later_tools & STATE_CHANGING_TOOLS):
                errors.append(
                    "negative filesystem/process evidence was not followed by a "
                    "state-changing repair tool"
                )

    if not any(event.get("type") in {"text", "final_answer"} for event in events):
        errors.append("task ended without a terminal text/final_answer event")

    return errors


async def run_task(
    spec: TaskSpec,
    *,
    url: str,
    out_dir: Path,
    verbose: bool,
) -> dict[str, Any]:
    session_id = f"live-smoke-{spec.name}-{uuid.uuid4().hex[:8]}"
    events: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{session_id}.jsonl"
    started = time.monotonic()

    async with websockets.connect(url, max_size=None) as websocket:
        await websocket.send(json.dumps({"session_id": session_id, "text": spec.prompt}))
        while True:
            remaining = spec.max_seconds - (time.monotonic() - started)
            if remaining <= 0:
                events.append(
                    {
                        "type": "error",
                        "detail": f"live smoke timeout after {spec.max_seconds}s",
                    }
                )
                break
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                events.append(
                    {
                        "type": "error",
                        "detail": f"live smoke timeout after {spec.max_seconds}s",
                    }
                )
                break
            event = json.loads(raw)
            events.append(event)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            if verbose:
                print(f"[{spec.name}] {_event_line(event)}", flush=True)
            if event.get("type") in {"text", "final_answer"}:
                break

    errors = _validate_events(spec, events)
    duration = round(time.monotonic() - started, 2)
    return {
        "name": spec.name,
        "session_id": session_id,
        "duration_seconds": duration,
        "events": len(events),
        "log_path": str(log_path),
        "ok": not errors,
        "errors": errors,
        "tools": [
            event.get("tool")
            for event in events
            if event.get("type") == "tool_call"
        ],
    }


async def async_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/stream")
    parser.add_argument("--out-dir", default="live-agent-runs")
    parser.add_argument(
        "--only",
        choices=["file-artifact", "tcp-web-service"],
        help="Run just one task from the suite.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    suite = list(TASKS)
    if args.only:
        available = {task.name: task for task in TASKS}
        suite = [available[args.only]]

    results = []
    for spec in suite:
        print(f"\n=== live task: {spec.name} ===", flush=True)
        result = await run_task(
            spec,
            url=args.url,
            out_dir=Path(args.out_dir),
            verbose=args.verbose,
        )
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    print("\n=== summary ===")
    print(json.dumps(results, indent=2))
    return 0 if all(item["ok"] for item in results) else 1


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
