"""Unit tests for the contract and planning modules extracted from agent.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contract import (
    _build_task_contract_instruction,
    _filter_tool_schemas,
    _is_continuation_signal,
    _latest_plan,
    _normalise_task_contract,
    _normalize_command,
    _plan_has_open_steps,
    _run_set_task_contract,
)
from evaluator import ExecutionStep
from llm_utils import (
    _is_async_iterable,
    _is_rate_limit_error,
    _make_final_answer,
    _sanitize_messages_for_llm,
)
from planning import (
    _build_contract_execution_instruction,
    _build_executive_summary,
    _classify_tool_result,
    _count_done_plan_steps,
    _render_plan,
    _run_update_plan,
)

# ---------------------------------------------------------------------------
# llm_utils
# ---------------------------------------------------------------------------

class TestMakeFinalAnswer:
    def test_structure(self):
        event = _make_final_answer("iteration_limit", "Task paused.")
        assert event["type"] == "final_answer"
        assert event["reason"] == "iteration_limit"
        assert event["content"] == "Task paused."


class TestSanitizeMessages:
    def test_removes_empty_user_message(self):
        msgs = [{"role": "user", "content": "hello"}, {"role": "user", "content": ""}]
        result = _sanitize_messages_for_llm(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "hello"

    def test_keeps_assistant_with_tool_calls_no_content(self):
        msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}]
        result = _sanitize_messages_for_llm(msgs)
        assert len(result) == 1
        assert result[0]["content"] is None

    def test_fills_empty_tool_result(self):
        msgs = [{"role": "tool", "content": ""}]
        result = _sanitize_messages_for_llm(msgs)
        assert result[0]["content"] == "(empty result)"

    def test_strips_empty_text_blocks(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "  "}, {"type": "text", "text": "hi"}]}]
        result = _sanitize_messages_for_llm(msgs)
        assert len(result[0]["content"]) == 1


class TestIsAsyncIterable:
    def test_async_gen_is_iterable(self):
        async def gen():
            yield 1
        assert _is_async_iterable(gen())

    def test_plain_list_is_not(self):
        assert not _is_async_iterable([1, 2, 3])


class TestIsRateLimitError:
    def test_matches_ratelimit_in_message(self):
        exc = RuntimeError("rate limit exceeded")
        assert _is_rate_limit_error(exc)

    def test_does_not_match_unrelated(self):
        exc = RuntimeError("connection refused")
        assert not _is_rate_limit_error(exc)


# ---------------------------------------------------------------------------
# contract: task contract validation
# ---------------------------------------------------------------------------

class TestNormaliseTaskContract:
    def test_valid_answer_mode(self):
        contract, err = _normalise_task_contract({
            "mode": "answer",
            "summary": "Explain X",
            "success_criteria": ["Clear explanation provided"],
            "evidence_requirements": ["none"],
        })
        assert err is None
        assert contract["mode"] == "answer"
        assert contract["evidence_requirements"] == ["none"]

    def test_valid_execute_mode(self):
        contract, err = _normalise_task_contract({
            "mode": "execute",
            "summary": "Build a site",
            "success_criteria": ["index.html written"],
            "evidence_requirements": ["filesystem_artifact"],
        })
        assert err is None
        assert contract["mode"] == "execute"

    def test_invalid_mode(self):
        _, err = _normalise_task_contract({"mode": "magic", "summary": "x",
            "success_criteria": ["x"], "evidence_requirements": ["none"]})
        assert err is not None and "mode" in err

    def test_empty_summary(self):
        _, err = _normalise_task_contract({"mode": "answer", "summary": "",
            "success_criteria": ["ok"], "evidence_requirements": ["none"]})
        assert err is not None

    def test_missing_success_criteria(self):
        _, err = _normalise_task_contract({"mode": "answer", "summary": "x",
            "success_criteria": [], "evidence_requirements": ["none"]})
        assert err is not None

    def test_execute_requires_non_none_evidence(self):
        _, err = _normalise_task_contract({"mode": "execute", "summary": "do stuff",
            "success_criteria": ["done"], "evidence_requirements": ["none"]})
        assert err is not None and "non-'none'" in err

    def test_unknown_evidence(self):
        _, err = _normalise_task_contract({"mode": "execute", "summary": "x",
            "success_criteria": ["ok"], "evidence_requirements": ["flying_unicorn"]})
        assert err is not None and "unknown" in err

    def test_deduplication(self):
        contract, err = _normalise_task_contract({
            "mode": "execute",
            "summary": "make files",
            "success_criteria": ["files exist"],
            "evidence_requirements": ["filesystem_artifact", "filesystem_artifact"],
        })
        assert err is None
        assert contract["evidence_requirements"].count("filesystem_artifact") == 1

    def test_running_tcp_service_evidence_is_valid(self):
        contract, err = _normalise_task_contract({
            "mode": "execute",
            "summary": "start a generic TCP service",
            "success_criteria": ["port listens"],
            "evidence_requirements": ["running_tcp_service"],
        })
        assert err is None
        assert contract["evidence_requirements"] == ["running_tcp_service"]

    def test_non_http_service_contract_corrects_http_evidence_to_tcp(self):
        contract, err = _normalise_task_contract({
            "mode": "execute",
            "summary": "Install a generic server process",
            "success_criteria": ["Server binds to port 25565"],
            "evidence_requirements": ["running_http_service"],
        })
        assert err is None
        assert contract["evidence_requirements"] == ["running_tcp_service"]

    def test_contract_sanitizes_host_wording(self):
        contract, err = _normalise_task_contract({
            "mode": "execute",
            "summary": "Install a simple server on the host",
            "success_criteria": ["host system has a listening port"],
            "evidence_requirements": ["running_tcp_service"],
        })
        assert err is None
        assert "host" not in contract["summary"].lower()
        assert "host" not in contract["success_criteria"][0].lower()

    def test_downloaded_artifact_adds_filesystem_evidence(self):
        contract, err = _normalise_task_contract({
            "mode": "execute",
            "summary": "Install and start a server",
            "success_criteria": ["Server JAR is downloaded", "TCP port listens"],
            "evidence_requirements": ["running_tcp_service"],
        })
        assert err is None
        assert "running_tcp_service" in contract["evidence_requirements"]
        assert "filesystem_artifact" in contract["evidence_requirements"]

    def test_run_set_task_contract_ok(self):
        result, is_error = _run_set_task_contract({
            "mode": "answer",
            "summary": "Count to ten",
            "success_criteria": ["counted"],
            "evidence_requirements": ["none"],
        })
        assert not is_error
        data = json.loads(result)
        assert data["contract_set"] is True

    def test_run_set_task_contract_error(self):
        result, is_error = _run_set_task_contract({"mode": "bad"})
        assert is_error
        assert "error" in result


# ---------------------------------------------------------------------------
# contract: continuation signal detection
# ---------------------------------------------------------------------------

class TestIsContinuationSignal:
    SIGNALS = ["yes", "y", "ok", "okay", "go", "continue", "do it",
               "proceed", "start", "run it", "carry on",
               "yes.", "OK!", "CONTINUE"]
    NOT_SIGNALS = ["build a website", "what is 2+2", "hello", "no", "stop"]

    def test_signals(self):
        for sig in self.SIGNALS:
            assert _is_continuation_signal(sig), f"Expected continuation: {sig!r}"

    def test_not_signals(self):
        for text in self.NOT_SIGNALS:
            assert not _is_continuation_signal(text), f"Should NOT be continuation: {text!r}"


# ---------------------------------------------------------------------------
# contract: plan lookups
# ---------------------------------------------------------------------------

def _make_plan_message(steps: list[dict]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "tc1",
            "type": "function",
            "function": {
                "name": "update_plan",
                "arguments": json.dumps({"steps": steps}),
            },
        }],
    }


class TestLatestPlan:
    def test_no_plan(self):
        messages = [{"role": "user", "content": "hello"}]
        assert _latest_plan(messages) is None

    def test_finds_plan_in_history(self):
        steps = [{"title": "step 1", "status": "pending"}]
        messages = [
            {"role": "user", "content": "do task"},
            _make_plan_message(steps),
        ]
        plan = _latest_plan(messages)
        assert plan is not None
        assert plan[0]["title"] == "step 1"

    def test_returns_most_recent_plan(self):
        first = [{"title": "old step", "status": "done"}]
        second = [{"title": "new step", "status": "pending"}]
        messages = [
            {"role": "user", "content": "do task"},
            _make_plan_message(first),
            _make_plan_message(second),
        ]
        plan = _latest_plan(messages)
        assert plan[0]["title"] == "new step"


class TestPlanHasOpenSteps:
    def test_no_plan_returns_false(self):
        assert not _plan_has_open_steps([{"role": "user", "content": "hi"}])

    def test_all_done_returns_false(self):
        steps = [{"title": "s", "status": "done"}]
        messages = [{"role": "user", "content": "x"}, _make_plan_message(steps)]
        assert not _plan_has_open_steps(messages)

    def test_pending_step_returns_true(self):
        steps = [{"title": "s", "status": "done"}, {"title": "s2", "status": "pending"}]
        messages = [{"role": "user", "content": "x"}, _make_plan_message(steps)]
        assert _plan_has_open_steps(messages)


# ---------------------------------------------------------------------------
# contract: tool schema filtering
# ---------------------------------------------------------------------------

class TestFilterToolSchemas:
    SCHEMAS = [
        {"function": {"name": "tool_a"}},
        {"function": {"name": "tool_b"}},
        {"function": {"name": "tool_c"}},
    ]

    def test_keeps_named(self):
        result = _filter_tool_schemas(self.SCHEMAS, {"tool_a", "tool_c"})
        names = [s["function"]["name"] for s in result]
        assert names == ["tool_a", "tool_c"]

    def test_empty_set_returns_empty(self):
        result = _filter_tool_schemas(self.SCHEMAS, set())
        assert result == []


# ---------------------------------------------------------------------------
# contract: normalize_command
# ---------------------------------------------------------------------------

class TestNormalizeCommand:
    def test_collapses_spaces(self):
        assert _normalize_command("  rm  -rf  /  ") == "rm -rf /"

    def test_empty(self):
        assert _normalize_command("") == ""
        assert _normalize_command(None) == ""


# ---------------------------------------------------------------------------
# planning: render plan
# ---------------------------------------------------------------------------

class TestRenderPlan:
    def test_icons(self):
        plan = [
            {"title": "A", "status": "done"},
            {"title": "B", "status": "in_progress"},
            {"title": "C", "status": "pending"},
            {"title": "D", "status": "failed"},
        ]
        rendered = _render_plan(plan)
        assert "[x]" in rendered
        assert "[~]" in rendered
        assert "[ ]" in rendered
        assert "[!]" in rendered

    def test_empty_plan(self):
        assert _render_plan([]) == "  (empty)"

    def test_truncates_long_plan(self):
        plan = [{"title": f"step {i}", "status": "pending"} for i in range(30)]
        rendered = _render_plan(plan)
        assert "more steps" in rendered


# ---------------------------------------------------------------------------
# planning: classify_tool_result
# ---------------------------------------------------------------------------

class TestClassifyToolResult:
    def test_terminal_exit_0_ok(self):
        content = json.dumps({"exit_code": 0, "stdout": "ok", "stderr": ""})
        is_err, _ = _classify_tool_result("execute_terminal_command", content)
        assert not is_err

    def test_terminal_exit_1_error(self):
        content = json.dumps({"exit_code": 1, "stdout": "", "stderr": "fail"})
        is_err, _ = _classify_tool_result("execute_terminal_command", content)
        assert is_err

    def test_generic_bracketed_error(self):
        is_err, _ = _classify_tool_result("some_tool", "[error] something went wrong")
        assert is_err

    def test_generic_success(self):
        is_err, _ = _classify_tool_result("some_tool", "all good")
        assert not is_err


# ---------------------------------------------------------------------------
# planning: run_update_plan
# ---------------------------------------------------------------------------

class TestRunUpdatePlan:
    def test_valid_plan(self):
        steps = [{"title": "do x", "status": "done"}, {"title": "do y", "status": "pending"}]
        result, is_error = _run_update_plan({"steps": steps})
        assert not is_error
        assert "1 done" in result

    def test_empty_steps(self):
        _, is_error = _run_update_plan({"steps": []})
        assert is_error

    def test_missing_steps_key(self):
        _, is_error = _run_update_plan({})
        assert is_error

    def test_all_done_tail(self):
        steps = [{"title": "x", "status": "done"}]
        result, _ = _run_update_plan({"steps": steps})
        assert "All steps are done" in result


# ---------------------------------------------------------------------------
# planning: count_done_plan_steps
# ---------------------------------------------------------------------------

class TestCountDonePlanSteps:
    def test_no_plan(self):
        assert _count_done_plan_steps([{"role": "user", "content": "hi"}]) == 0

    def test_counts_done(self):
        steps = [
            {"title": "a", "status": "done"},
            {"title": "b", "status": "done"},
            {"title": "c", "status": "pending"},
        ]
        messages = [{"role": "user", "content": "x"}, _make_plan_message(steps)]
        assert _count_done_plan_steps(messages) == 2


# ---------------------------------------------------------------------------
# planning: executive summary smoke test
# ---------------------------------------------------------------------------

class TestBuildExecutiveSummary:
    def test_shows_objective(self):
        messages = [{"role": "user", "content": "Build a REST API"}]
        summary = _build_executive_summary(messages)
        assert "Build a REST API" in summary

    def test_shows_completed_actions(self):
        messages = [
            {"role": "user", "content": "install deps"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "execute_terminal_command",
                        "arguments": json.dumps({"command": "pip install flask"}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "tc1",
                "content": json.dumps({"exit_code": 0, "stdout": "installed", "stderr": ""}),
            },
        ]
        summary = _build_executive_summary(messages)
        assert "pip install flask" in summary

    def test_shows_none_when_no_actions(self):
        messages = [{"role": "user", "content": "explain X"}]
        summary = _build_executive_summary(messages)
        assert "None yet" in summary


# ---------------------------------------------------------------------------
# contract: instruction builders smoke tests
# ---------------------------------------------------------------------------

class TestBuildTaskContractInstruction:
    def test_mentions_modes(self):
        instruction = _build_task_contract_instruction()
        assert "answer" in instruction
        assert "execute" in instruction
        assert "set_task_contract" in instruction


def _tool_result_step(tool_name: str, is_error: bool = False) -> ExecutionStep:
    return ExecutionStep(
        kind="tool_result",
        content="ok",
        metadata={"tool_name": tool_name, "is_error": is_error},
    )


class TestBuildContractExecutionInstruction:
    _HTTP_CONTRACT = {
        "mode": "execute",
        "summary": "serve sleep website",
        "success_criteria": ["site is served over HTTP"],
        "evidence_requirements": ["running_http_service"],
    }
    _MSGS = [{"role": "user", "content": "build site"}]

    def test_incomplete_status_has_next_action(self):
        contract = {
            "mode": "execute",
            "summary": "build site",
            "success_criteria": ["index.html exists"],
            "evidence_requirements": ["filesystem_artifact"],
        }
        status = {"complete": False, "missing": ["plan"], "plan_open": True}
        messages = [{"role": "user", "content": "build site"}]
        instruction = _build_contract_execution_instruction(contract, status, messages)
        assert "update_plan" in instruction.lower()

    def test_http_service_evidence_guides_runtime_work(self):
        status = {
            "complete": False,
            "missing": ["plan_open_steps", "running_http_service"],
            "plan_open": True,
        }
        steps = [_tool_result_step("set_task_contract"), _tool_result_step("update_plan")]
        instruction = _build_contract_execution_instruction(
            self._HTTP_CONTRACT, status, self._MSGS, steps
        )
        low = instruction.lower()
        assert "execute_background_service" in low or "expose_local_http_service" in low
        assert "do not call update_plan" in low or "not call update_plan" in low

    def test_tools_called_so_far_appears_in_instruction(self):
        """The instruction should surface which tools have been called so the
        model can see its own progress without re-reading the full history."""
        status = {
            "complete": False,
            "missing": ["plan_open_steps", "running_http_service"],
            "plan_open": True,
        }
        steps = [_tool_result_step("update_plan")]
        instruction = _build_contract_execution_instruction(
            self._HTTP_CONTRACT, status, self._MSGS, steps
        )
        assert "update_plan" in instruction

    def test_update_plan_response_discourages_repeat_call(self):
        """_run_update_plan must tell the model not to call update_plan again
        in the same turn when there are still open steps."""
        result, is_error = _run_update_plan(
            {"steps": [{"title": "Build", "status": "in_progress"}]}
        )
        assert not is_error
        assert "do not call update_plan" in result.lower()
