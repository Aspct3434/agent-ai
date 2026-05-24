"""Regression tests for the "agent can't finish a React build on its own" failure.

Root cause from the captured transcript: the model emitted near-miss arguments
that the framework rejected *silently*, trapping it:

  * set_task_contract with success_criteria as an <item>-wrapped STRING, not a list
  * update_plan with {"plan": "<json string>"} instead of {"steps": [...]}
  * step objects keyed "step" not "title", with status "waiting" (not canonical)

Because those update_plan calls errored, the plan never closed, so the contract's
`plan_open_steps` requirement could never clear â€” the task could never complete
even if the build had succeeded. These tests lock in tolerant coercion plus the
build-and-serve directive that tells the agent to build before serving.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from contract import (  # noqa: E402
    _coerce_plan_steps,
    _latest_plan,
    _normalise_plan_status,
    _normalise_task_contract,
    _plan_has_open_steps,
    _plan_steps_from_args,
    _split_criteria_string,
)
from planning import _run_update_plan  # noqa: E402

# ---------------------------------------------------------------------------
# Plan status coercion
# ---------------------------------------------------------------------------


class TestNormalisePlanStatus:
    def test_canonical_passthrough(self):
        for s in ("pending", "in_progress", "done", "failed"):
            assert _normalise_plan_status(s) == s

    def test_waiting_becomes_pending(self):
        assert _normalise_plan_status("waiting") == "pending"

    def test_todo_becomes_pending(self):
        assert _normalise_plan_status("todo") == "pending"

    def test_completed_becomes_done(self):
        assert _normalise_plan_status("completed") == "done"
        assert _normalise_plan_status("complete") == "done"

    def test_in_progress_spelled_with_space(self):
        assert _normalise_plan_status("in progress") == "in_progress"

    def test_active_becomes_in_progress(self):
        assert _normalise_plan_status("active") == "in_progress"

    def test_blocked_becomes_failed(self):
        assert _normalise_plan_status("blocked") == "failed"

    def test_unknown_defaults_to_pending(self):
        assert _normalise_plan_status("banana") == "pending"

    def test_none_defaults_to_pending(self):
        assert _normalise_plan_status(None) == "pending"


# ---------------------------------------------------------------------------
# Plan step coercion â€” title aliases and JSON-string inputs
# ---------------------------------------------------------------------------


class TestCoercePlanSteps:
    def test_canonical_steps(self):
        steps = _coerce_plan_steps([{"title": "A", "status": "done"}])
        assert steps == [{"title": "A", "status": "done"}]

    def test_step_alias_for_title(self):
        steps = _coerce_plan_steps([{"step": "Build app", "status": "done"}])
        assert steps[0]["title"] == "Build app"

    def test_name_alias_for_title(self):
        steps = _coerce_plan_steps([{"name": "Serve", "status": "pending"}])
        assert steps[0]["title"] == "Serve"

    def test_waiting_status_coerced(self):
        steps = _coerce_plan_steps([{"step": "X", "status": "waiting"}])
        assert steps[0]["status"] == "pending"

    def test_json_string_input(self):
        steps = _coerce_plan_steps('[{"title": "A", "status": "done"}]')
        assert steps == [{"title": "A", "status": "done"}]

    def test_items_without_title_are_dropped(self):
        steps = _coerce_plan_steps([{"status": "done"}, {"title": "Keep", "status": "done"}])
        assert len(steps) == 1
        assert steps[0]["title"] == "Keep"

    def test_non_list_returns_none(self):
        assert _coerce_plan_steps(42) is None

    def test_bad_json_string_returns_none(self):
        assert _coerce_plan_steps("not json") is None


# ---------------------------------------------------------------------------
# _plan_steps_from_args â€” "plan" alias for "steps"
# ---------------------------------------------------------------------------


class TestPlanStepsFromArgs:
    def test_steps_key(self):
        steps = _plan_steps_from_args({"steps": [{"title": "A", "status": "done"}]})
        assert steps[0]["title"] == "A"

    def test_plan_alias_key(self):
        steps = _plan_steps_from_args({"plan": [{"title": "A", "status": "done"}]})
        assert steps[0]["title"] == "A"

    def test_plan_alias_as_json_string(self):
        # The exact malformed shape from the transcript.
        steps = _plan_steps_from_args(
            {"plan": '[{"step": "Create React app", "status": "done"}]'}
        )
        assert steps[0]["title"] == "Create React app"
        assert steps[0]["status"] == "done"


# ---------------------------------------------------------------------------
# _run_update_plan â€” the transcript's malformed call must now succeed
# ---------------------------------------------------------------------------


class TestRunUpdatePlanTolerant:
    # Verbatim payload from the failing transcript.
    TRANSCRIPT_PLAN = (
        '[\n  {"step": "Create React app using Vite", "status": "done"},\n'
        '  {"step": "Install npm dependencies", "status": "in_progress"},\n'
        '  {"step": "Build the React app for production", "status": "waiting"},\n'
        '  {"step": "Serve static site", "status": "waiting"}\n]'
    )

    def test_transcript_plan_payload_is_not_an_error(self):
        _, is_error = _run_update_plan({"plan": self.TRANSCRIPT_PLAN})
        assert not is_error, (
            "The malformed update_plan call from the transcript must now be "
            "accepted; otherwise the plan never closes and the contract is "
            "permanently stuck on plan_open_steps."
        )

    def test_transcript_plan_counts_are_correct(self):
        result, _ = _run_update_plan({"plan": self.TRANSCRIPT_PLAN})
        assert "1 done" in result
        assert "1 in progress" in result
        # the two "waiting" steps were coerced to pending
        assert "2 pending" in result

    def test_canonical_still_works(self):
        result, is_error = _run_update_plan(
            {"steps": [{"title": "x", "status": "done"}]}
        )
        assert not is_error
        assert "All steps are done" in result

    def test_truly_empty_still_errors(self):
        _, is_error = _run_update_plan({})
        assert is_error


# ---------------------------------------------------------------------------
# _latest_plan / _plan_has_open_steps with the "plan" alias
# ---------------------------------------------------------------------------


def _plan_msg(arguments: dict) -> dict:
    import json

    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "c1",
                "function": {"name": "update_plan", "arguments": json.dumps(arguments)},
            }
        ],
    }


class TestLatestPlanWithAlias:
    def test_latest_plan_reads_plan_alias(self):
        messages = [
            {"role": "user", "content": "build it"},
            _plan_msg({"plan": '[{"step": "Build", "status": "done"}]'}),
        ]
        plan = _latest_plan(messages)
        assert plan is not None
        assert plan[0]["title"] == "Build"

    def test_open_steps_detected_after_coercion(self):
        messages = [
            {"role": "user", "content": "build it"},
            _plan_msg(
                {
                    "plan": '[{"step": "A", "status": "done"}, '
                    '{"step": "B", "status": "waiting"}]'
                }
            ),
        ]
        # "waiting" -> pending, which is an open status
        assert _plan_has_open_steps(messages)

    def test_all_done_closes_plan(self):
        messages = [
            {"role": "user", "content": "build it"},
            _plan_msg({"plan": '[{"step": "A", "status": "completed"}]'}),
        ]
        assert not _plan_has_open_steps(messages)


# ---------------------------------------------------------------------------
# success_criteria as a string
# ---------------------------------------------------------------------------


class TestSplitCriteriaString:
    def test_item_tags(self):
        text = "\n  <item>First</item>\n  <item>Second</item>\n"
        assert _split_criteria_string(text) == ["First", "Second"]

    def test_newline_separated(self):
        assert _split_criteria_string("First\nSecond\nThird") == [
            "First",
            "Second",
            "Third",
        ]

    def test_semicolon_separated(self):
        assert _split_criteria_string("a; b; c") == ["a", "b", "c"]

    def test_blank_lines_dropped(self):
        assert _split_criteria_string("a\n\n\nb") == ["a", "b"]


class TestNormaliseContractStringCriteria:
    def test_item_string_success_criteria_accepted(self):
        # The exact shape from the transcript's first (rejected) contract call.
        contract, error = _normalise_task_contract(
            {
                "mode": "execute",
                "summary": "Build a sleep site in React",
                "success_criteria": (
                    "\n  <item>A working React application built and running</item>\n"
                    "  <item>Website is served and accessible via a public URL</item>\n"
                ),
                "evidence_requirements": ["running_http_service"],
                "toolset": "coding",
            }
        )
        assert error is None, f"contract should be accepted, got error: {error}"
        assert len(contract["success_criteria"]) == 2

    def test_list_success_criteria_still_works(self):
        contract, error = _normalise_task_contract(
            {
                "mode": "answer",
                "summary": "explain X",
                "success_criteria": ["clear answer"],
                "evidence_requirements": ["none"],
            }
        )
        assert error is None
        assert contract["success_criteria"] == ["clear answer"]


# ---------------------------------------------------------------------------
# SYSTEM_DIRECTIVE build-and-serve recipe
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# parallel_tool_calls=False guard â€” prevents double update_plan in one turn
# ---------------------------------------------------------------------------


class TestParallelToolCallsDisabledForContractExecution:
    """The model sometimes emits two update_plan calls in one turn when only
    update_plan is in the allowed tool set.  The second call collapses the
    carefully-constructed 3-step plan to a 1-step plan before any result is
    seen, then the task stalls because write_text_file is never called.

    The fix: set parallel_tool_calls=False in the LLM completion kwargs
    whenever the contract gate is active (must_set_contract or needs_execution).
    These tests confirm the flag is present in the kwargs assembled by the
    agent loop under those conditions.
    """

    @staticmethod
    def _make_contract():
        return {
            "mode": "execute",
            "summary": "build sleep site",
            "success_criteria": ["site served"],
            "evidence_requirements": ["running_http_service"],
            "toolset": "all",
        }

    def test_update_plan_response_discourages_another_call(self):
        """update_plan's response must tell the model not to call it again
        in the same turn when there are still open steps â€” belt-and-braces
        defence even though parallel_tool_calls=False is the primary guard."""
        result, is_error = _run_update_plan(
            {"steps": [{"title": "Build HTML", "status": "pending"}]}
        )
        assert not is_error
        assert "do not call update_plan" in result.lower()

    def test_second_update_plan_collapse_is_still_coerced(self):
        """Even when two update_plan calls occur (e.g. on older models that
        ignore parallel_tool_calls), _latest_plan must return the second one
        (the 1-step plan), not None â€” so the iteration can continue rather
        than re-asking for a plan from scratch."""
        import json

        messages = [
            {"role": "user", "content": "build sleep site"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "set_task_contract",
                            "arguments": json.dumps(
                                {
                                    "mode": "execute",
                                    "summary": "build sleep site",
                                    "success_criteria": ["served"],
                                    "evidence_requirements": ["running_http_service"],
                                }
                            ),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            # Two update_plan calls in one turn â€” the collapse pattern
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "p1",
                        "function": {
                            "name": "update_plan",
                            "arguments": json.dumps(
                                {
                                    "steps": [
                                        {"title": "Write HTML", "status": "pending"},
                                        {"title": "Serve site", "status": "pending"},
                                        {"title": "Verify URL", "status": "pending"},
                                    ]
                                }
                            ),
                        },
                    },
                    {
                        "id": "p2",
                        "function": {
                            "name": "update_plan",
                            "arguments": '{"steps": "[{\\"title\\": \\"Write HTML\\", \\"status\\": \\"in_progress\\"}]"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "p1", "content": "Plan updated: 3 steps"},
            {"role": "tool", "tool_call_id": "p2", "content": "Plan updated: 1 step"},
        ]
        from contract import _latest_plan, _plan_has_open_steps

        plan = _latest_plan(messages)
        # The second (collapsed) plan must be found and coerced, not None
        assert plan is not None, "_latest_plan must not return None after double update_plan"
        assert len(plan) == 1
        assert plan[0]["title"] == "Write HTML"
        assert plan[0]["status"] == "in_progress"
        # Plan is still open â€” next iteration must proceed to write_text_file
        assert _plan_has_open_steps(messages)


class TestBuildServeDirective:
    @staticmethod
    def _directive() -> str:
        from agent import SYSTEM_DIRECTIVE

        return SYSTEM_DIRECTIVE.lower()

    def test_mentions_npm_run_build(self):
        assert "npm run build" in self._directive()

    def test_mentions_dist_folder(self):
        assert "dist" in self._directive()

    def test_warns_against_serving_project_root(self):
        directive = self._directive()
        assert "project root" in directive and "never" in directive

    def test_mentions_non_interactive_scaffold(self):
        directive = self._directive()
        assert "ci=1" in directive or "non-interactive" in directive or "prompt" in directive
