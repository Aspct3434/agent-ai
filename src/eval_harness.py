"""Evaluation harness for the agent.

Runs a suite of tasks through ``AgentEngine`` and records, per task, whether a
deterministic success check passed plus the efficiency metrics that matter:
LLM calls, token usage (incl. cache reads when the provider reports them), tool
calls, wall-clock, and the terminal outcome.

The point is to turn "this change feels better" into numbers you can diff across
runs. Write results with ``--out results.json`` before a change and pass that
file as ``--baseline`` after, to see per-task deltas.

Usage::

    # live run against the configured model (needs an API key + AGENT_MODEL)
    python src/eval_harness.py --out before.json
    # ... make a change ...
    python src/eval_harness.py --baseline before.json --out after.json

    # validate the harness itself with a deterministic mock model (no API key)
    python src/eval_harness.py --self-test

Metrics are captured non-invasively by wrapping ``litellm.acompletion`` for the
duration of a run, so the agent code under test is unmodified.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow running both as `python src/eval_harness.py` and as an import from tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent as agent_module
from agent import AgentEngine, NormalizedMessage

# ---------------------------------------------------------------------------
# Result / task model
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    task_id: str
    passed: bool = False
    outcome: str = "unknown"  # completed | iteration_limit | exception | rate_limited | critical_failure
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    tool_calls: int = 0
    wall_seconds: float = 0.0
    final_text: str = ""
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def to_record(self) -> dict[str, Any]:
        """JSON-serialisable summary (events excluded -- too bulky for diffs)."""
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "outcome": self.outcome,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "tool_calls": self.tool_calls,
            "wall_seconds": self.wall_seconds,
            "final_text": self.final_text[:500],
            "error": self.error,
        }


@dataclass
class EvalTask:
    id: str
    prompt: str
    check: Callable[[EvalResult], bool]
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Token / call metering (wraps litellm.acompletion non-invasively)
# ---------------------------------------------------------------------------

def _usage_get(usage: Any, *names: str) -> int:
    for name in names:
        value = None
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


class _UsageMeter:
    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cache_read_tokens = 0

    def record(self, response: Any) -> None:
        self.calls += 1
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return
        self.prompt_tokens += _usage_get(usage, "prompt_tokens", "input_tokens")
        self.completion_tokens += _usage_get(usage, "completion_tokens", "output_tokens")
        self.total_tokens += _usage_get(usage, "total_tokens")
        # Anthropic prompt-cache fields (names vary by litellm version).
        self.cache_read_tokens += _usage_get(
            usage, "cache_read_input_tokens", "cache_read_tokens"
        )


@contextmanager
def _meter_litellm(meter: _UsageMeter) -> Iterator[None]:
    original = agent_module.litellm.acompletion

    async def wrapped(**kwargs: Any) -> Any:
        response = await original(**kwargs)
        try:
            meter.record(response)
        except Exception:  # never let metering break a run
            pass
        return response

    agent_module.litellm.acompletion = wrapped
    try:
        yield
    finally:
        agent_module.litellm.acompletion = original


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_task(engine: AgentEngine, task: EvalTask, session_id: str | None = None) -> EvalResult:
    """Run one task end-to-end and score it. Never raises -- failures are captured."""
    meter = _UsageMeter()
    result = EvalResult(task_id=task.id)
    sid = session_id or f"eval-{task.id}"
    start = time.perf_counter()
    try:
        with _meter_litellm(meter):
            async for event in engine.stream_task(
                NormalizedMessage(session_id=sid, role="user", content=task.prompt)
            ):
                result.events.append(event)
                etype = event.get("type")
                if etype == "tool_call":
                    result.tool_calls += 1
                elif etype == "text":
                    result.final_text = event.get("content", "")
                    result.outcome = "completed"
                elif etype == "final_answer":
                    result.final_text = event.get("content", "")
                    result.outcome = event.get("reason", "final_answer")
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.outcome = "harness_error"

    result.wall_seconds = round(time.perf_counter() - start, 3)
    result.llm_calls = meter.calls
    result.prompt_tokens = meter.prompt_tokens
    result.completion_tokens = meter.completion_tokens
    result.total_tokens = meter.total_tokens or (meter.prompt_tokens + meter.completion_tokens)
    result.cache_read_tokens = meter.cache_read_tokens

    try:
        result.passed = bool(task.check(result))
    except Exception as exc:
        result.passed = False
        result.error = (result.error or "") + f" check_error: {exc}"
    return result


async def run_suite(engine: AgentEngine, tasks: list[EvalTask]) -> list[EvalResult]:
    results: list[EvalResult] = []
    for task in tasks:
        results.append(await run_task(engine, task))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_table(results: list[EvalResult]) -> str:
    header = ("task", "pass", "outcome", "llm", "tok", "cache", "tools", "secs")
    rows = [header]
    for r in results:
        rows.append(
            (
                r.task_id[:28],
                "PASS" if r.passed else "FAIL",
                r.outcome[:14],
                str(r.llm_calls),
                str(r.total_tokens),
                str(r.cache_read_tokens),
                str(r.tool_calls),
                f"{r.wall_seconds:.1f}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        for row in rows
    ]
    sep = "  ".join("-" * w for w in widths)
    lines.insert(1, sep)

    passed = sum(1 for r in results if r.passed)
    total = len(results) or 1
    summary = (
        f"\nPASS {passed}/{len(results)} ({100 * passed / total:.0f}%)  |  "
        f"mean llm_calls {statistics.mean([r.llm_calls for r in results]):.1f}  |  "
        f"mean total_tokens {statistics.mean([r.total_tokens for r in results]):.0f}  |  "
        f"mean secs {statistics.mean([r.wall_seconds for r in results]):.1f}"
    )
    return "\n".join(lines) + "\n" + summary


def format_comparison(baseline: list[dict[str, Any]], results: list[EvalResult]) -> str:
    base_by_id = {b["task_id"]: b for b in baseline}
    lines = ["\nDelta vs baseline (negative tokens/llm = improvement):"]
    for r in results:
        b = base_by_id.get(r.task_id)
        if b is None:
            lines.append(f"  {r.task_id}: (new)")
            continue
        d_tok = r.total_tokens - b.get("total_tokens", 0)
        d_llm = r.llm_calls - b.get("llm_calls", 0)
        pass_change = ""
        if bool(b.get("passed")) != r.passed:
            pass_change = "  PASS->FAIL" if not r.passed else "  FAIL->PASS"
        lines.append(
            f"  {r.task_id[:28]:28}  tok {d_tok:+d}  llm {d_llm:+d}{pass_change}"
        )
    return "\n".join(lines)


def write_json(path: str | Path, results: list[EvalResult]) -> None:
    Path(path).write_text(
        json.dumps([r.to_record() for r in results], indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Default live suite (deterministic checkers; runs against the real model)
# ---------------------------------------------------------------------------

def _check_contains(*substrings: str) -> Callable[[EvalResult], bool]:
    def check(r: EvalResult) -> bool:
        text = r.final_text.lower()
        return all(s.lower() in text for s in substrings)
    return check


def _check_file(path: str, content_substr: str | None = None) -> Callable[[EvalResult], bool]:
    def check(r: EvalResult) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        if content_substr is not None:
            try:
                return content_substr in p.read_text(errors="ignore")
            except OSError:
                return False
        return True
    return check


LIVE_TASKS: list[EvalTask] = [
    EvalTask(
        id="qa_math",
        prompt="What is 2 + 2? Reply with just the number.",
        check=_check_contains("4"),
        tags=["informational"],
    ),
    EvalTask(
        id="create_file_verify",
        prompt=(
            "Create a file at /tmp/eval_hello.txt whose exact contents are "
            "HELLO_EVAL, then verify it exists."
        ),
        check=_check_file("/tmp/eval_hello.txt", "HELLO_EVAL"),
        tags=["action", "filesystem"],
    ),
    EvalTask(
        id="static_site",
        prompt=(
            "Make a simple static website about cats in the directory "
            "/tmp/eval-cats, then verify the files exist."
        ),
        check=_check_file("/tmp/eval-cats/index.html"),
        tags=["action", "website"],
    ),
    # Regression: the agent previously stalled after update_plan without calling
    # write_text_file when the user asked for an "interactive website" without
    # naming a framework. Fixed by giving concrete step-by-step guidance.
    EvalTask(
        id="interactive_sleep_site",
        prompt="Create an interactive website about the importance of sleep.",
        check=_check_file("/workspace/index.html"),
        tags=["action", "website", "regression"],
    ),
]


class _EmptyMemory:
    def retrieve_context(self, query: str, query_type: str) -> dict[str, Any]:
        return {"query_type": query_type, "results": []}

    def store_event(self, session_id: str, raw_text: str, entities: dict[str, Any]) -> str:
        return ""


async def _build_live_engine() -> tuple[AgentEngine, Any]:
    import os

    from tools import ToolManager

    tools = ToolManager()
    model = os.getenv("AGENT_MODEL", "gpt-4o-mini")
    engine = AgentEngine(
        memory=_EmptyMemory(),
        tools=tools,
        model=model,
        fast_model=os.getenv("FAST_AGENT_MODEL", model),
        strong_model=os.getenv("STRONG_AGENT_MODEL", model),
    )
    return engine, tools


async def _main_live(args: argparse.Namespace) -> int:
    engine, tools = await _build_live_engine()
    try:
        results = await run_suite(engine, LIVE_TASKS)
    finally:
        if hasattr(tools, "close"):
            await tools.close()

    print(format_table(results))
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        print(format_comparison(baseline, results))
    if args.out:
        write_json(args.out, results)
        print(f"\nWrote {args.out}")
    return 0 if all(r.passed for r in results) else 1


# ---------------------------------------------------------------------------
# Self-test: validate the harness plumbing with a deterministic mock model so it
# can be exercised without an API key (also imported by tests/).
# ---------------------------------------------------------------------------

class _ScriptedModel:
    """Returns canned litellm-shaped responses (with usage) from a script."""

    def __init__(self, script: list[Any]) -> None:
        self._script = script
        self.calls = 0

    async def __call__(self, **kwargs: Any) -> Any:
        response = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return response


def _mk_response(*, content: str | None = None, tool: tuple[str, dict] | None = None) -> Any:
    from types import SimpleNamespace

    tool_calls = None
    finish = "stop"
    if tool is not None:
        name, args = tool
        finish = "tool_calls"
        tool_calls = [
            SimpleNamespace(
                id=f"call_{name}", function=SimpleNamespace(name=name, arguments=json.dumps(args))
            )
        ]
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content, tool_calls=tool_calls))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def run_self_test() -> int:
    """Exercise the harness with a mock model + fake tools. Returns 0 on success."""

    class FakeTools:
        async def list_all_tools(self) -> list[dict[str, Any]]:
            return []

        async def execute_terminal_command(self, command: str) -> dict[str, Any]:
            return {"exit_code": 0, "stdout": "ok", "stderr": "", "current_working_directory": "/"}

    # Mock model: run one tool call, then answer "4".
    script = [
        _mk_response(tool=("execute_terminal_command", {"command": "echo compute"})),
        _mk_response(content="The answer is 4."),
    ]
    model = _ScriptedModel(script)
    original = agent_module.litellm.acompletion
    agent_module.litellm.acompletion = model
    try:
        engine = AgentEngine(memory=_EmptyMemory(), tools=FakeTools(), model="gpt-4o-mini")
        task = EvalTask(id="selftest_qa", prompt="What is 2 + 2?", check=_check_contains("4"))
        result = asyncio.run(run_task(engine, task))
    finally:
        agent_module.litellm.acompletion = original

    print(format_table([result]))
    checks = {
        "passed": result.passed is True,
        "outcome_completed": result.outcome == "completed",
        "llm_calls==2": result.llm_calls == 2,
        "tool_calls==1": result.tool_calls == 1,
        "prompt_tokens==20": result.prompt_tokens == 20,
        "completion_tokens==10": result.completion_tokens == 10,
        "json_record": "events" not in result.to_record(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print(f"SELF-TEST FAILED: {failed}\n{result.to_record()}")
        return 1
    print("EVAL HARNESS SELF-TEST PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent evaluation harness")
    parser.add_argument("--self-test", action="store_true", help="Run the harness against a mock model (no API key).")
    parser.add_argument("--out", help="Write results JSON to this path.")
    parser.add_argument("--baseline", help="Compare against a previously written results JSON.")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()
    return asyncio.run(_main_live(args))


if __name__ == "__main__":
    raise SystemExit(main())
