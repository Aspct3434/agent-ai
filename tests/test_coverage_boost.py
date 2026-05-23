"""Targeted tests to push coverage over 70%.

Covers: evidence parsers in contract.py, prune_message_window/history
compaction in planning.py, llm_utils retry logic, and checkpointer async I/O.
"""
from __future__ import annotations

import json
import sys
import unittest.mock
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contract import (
    _blocked_action_tool_message,
    _build_incomplete_contract_cap_message,
    _can_stream_text_before_final,
    _contract_completion_status,
    _duplicate_command_message,
    _evidence_requirement_satisfied,
    _expose_local_http_service_evidence_is_positive,
    _filesystem_process_evidence_is_positive,
    _last_host_command,
    _publish_static_site_evidence_is_positive,
    _should_block_tool_for_action_task,
    _successful_command_output_evidence,
    _tool_names_for_contract_status,
    _write_text_file_evidence_is_positive,
)
from evaluator import ExecutionStep
from llm_utils import (
    _acompletion_stream_with_retry,
    _acompletion_with_retry,
    _is_rate_limit_error,
    _prepare_llm_request_messages,
    _rate_limit_user_message,
)
from planning import (
    _KEEP_RECENT_MESSAGES,
    _MAX_HISTORY_MESSAGES,
    _build_contract_continuation_instruction,
    _build_history_compaction_summary,
    _build_plan_continuation_instruction,
    _count_successful_side_effects,
    _prune_message_window,
    _store_iteration_cap_memory,
)

# ===========================================================================
# contract.py — evidence parsers
# ===========================================================================

class TestPublishStaticSiteEvidence:
    def test_positive(self):
        content = json.dumps({"published": True, "index_exists": True, "url": "http://x"})
        assert _publish_static_site_evidence_is_positive(content)

    def test_missing_url(self):
        content = json.dumps({"published": True, "index_exists": True})
        assert not _publish_static_site_evidence_is_positive(content)

    def test_invalid_json(self):
        assert not _publish_static_site_evidence_is_positive("not-json")

    def test_false_published(self):
        content = json.dumps({"published": False, "index_exists": True, "url": "http://x"})
        assert not _publish_static_site_evidence_is_positive(content)

    def test_placeholder_quality_rejected(self):
        content = json.dumps(
            {
                "published": True,
                "index_exists": True,
                "url": "http://x",
                "artifact_quality": {"placeholder_detected": True},
            }
        )
        assert not _publish_static_site_evidence_is_positive(content)


class TestWriteTextFileEvidence:
    def test_positive(self):
        content = json.dumps({"written": True, "exists": True, "size_bytes": 42})
        assert _write_text_file_evidence_is_positive(content)

    def test_zero_bytes(self):
        content = json.dumps({"written": True, "exists": True, "size_bytes": 0})
        assert not _write_text_file_evidence_is_positive(content)

    def test_invalid_json(self):
        assert not _write_text_file_evidence_is_positive("{bad")

    def test_placeholder_quality_rejected(self):
        content = json.dumps(
            {
                "written": True,
                "exists": True,
                "size_bytes": 42,
                "artifact_quality": {"placeholder_detected": True},
            }
        )
        assert not _write_text_file_evidence_is_positive(content)

    def test_highly_interactive_contract_requires_interaction_signals(self):
        contract = {
            "mode": "execute",
            "summary": "Create a highly interactive website about sleep.",
            "success_criteria": ["Highly interactive website is published"],
            "evidence_requirements": ["published_static_site_url"],
        }
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "update_plan",
                            "arguments": json.dumps(
                                {"steps": [{"title": "Publish", "status": "done"}]}
                            ),
                        }
                    }
                ],
            }
        ]
        steps = [
            ExecutionStep(
                kind="tool_result",
                content=json.dumps(
                    {
                        "published": True,
                        "index_exists": True,
                        "url": "http://x",
                        "artifact_quality": {
                            "placeholder_detected": False,
                            "interactive_signal_count": 0,
                            "style_rule_count": 1,
                            "content_word_count": 12,
                        },
                    }
                ),
                metadata={"tool_name": "publish_static_site", "is_error": False},
            )
        ]

        status = _contract_completion_status(
            contract, messages, steps, contract_required=True
        )

        assert status["complete"] is False
        assert "artifact_quality" in status["missing"]


class TestExposeLocalHttpServiceEvidence:
    def test_positive(self):
        content = json.dumps({"exposed": True, "connectable": True, "url": "http://localhost:5000"})
        assert _expose_local_http_service_evidence_is_positive(content)

    def test_not_connectable(self):
        content = json.dumps({"exposed": True, "connectable": False, "url": "http://x"})
        assert not _expose_local_http_service_evidence_is_positive(content)

    def test_invalid_json(self):
        assert not _expose_local_http_service_evidence_is_positive("nope")


class TestFilesystemProcessEvidence:
    def test_path_exists(self):
        content = json.dumps({"paths": [{"exists": True}]})
        assert _filesystem_process_evidence_is_positive(content)

    def test_pid_running(self):
        content = json.dumps({"pids": [{"running": True}]})
        assert _filesystem_process_evidence_is_positive(content)

    def test_process_count(self):
        content = json.dumps({"process_names": [{"count": 2}]})
        assert _filesystem_process_evidence_is_positive(content)

    def test_port_connectable(self):
        content = json.dumps({"ports": [{"connectable": True}]})
        assert _filesystem_process_evidence_is_positive(content)

    def test_background_log(self):
        content = json.dumps({"background_log": {"exists": True, "tail": "running"}})
        assert _filesystem_process_evidence_is_positive(content)

    def test_all_empty(self):
        content = json.dumps({"paths": [{"exists": False}], "pids": [], "ports": []})
        assert not _filesystem_process_evidence_is_positive(content)

    def test_invalid_json(self):
        assert not _filesystem_process_evidence_is_positive("??")


class TestSuccessfulCommandOutputEvidence:
    def test_terminal_exit0_with_stdout(self):
        content = json.dumps({"exit_code": 0, "stdout": "ok", "stderr": ""})
        assert _successful_command_output_evidence("execute_terminal_command", content)

    def test_terminal_exit0_no_output(self):
        content = json.dumps({"exit_code": 0, "stdout": "", "stderr": ""})
        assert not _successful_command_output_evidence("execute_terminal_command", content)

    def test_terminal_exit1(self):
        content = json.dumps({"exit_code": 1, "stdout": "err"})
        assert not _successful_command_output_evidence("execute_terminal_command", content)

    def test_background_service_ok(self):
        content = json.dumps({"status": "running", "pid": 123})
        assert _successful_command_output_evidence("execute_background_service", content)

    def test_background_service_error(self):
        content = json.dumps({"status": "error"})
        assert not _successful_command_output_evidence("execute_background_service", content)

    def test_background_service_failed(self):
        content = json.dumps({"status": "failed"})
        assert not _successful_command_output_evidence("execute_background_service", content)

    def test_unknown_tool(self):
        assert not _successful_command_output_evidence("some_other_tool", "anything")

    def test_invalid_json_terminal(self):
        assert not _successful_command_output_evidence("execute_terminal_command", "bad")

    def test_invalid_json_background(self):
        assert not _successful_command_output_evidence("execute_background_service", "bad")


def _make_step(tool_name: str, content: str, is_error: bool = False) -> ExecutionStep:
    return ExecutionStep(
        kind="tool_result",
        content=content,
        metadata={"tool_name": tool_name, "is_error": is_error},
    )


class TestEvidenceRequirementSatisfied:
    def test_published_static_site_url(self):
        content = json.dumps({"published": True, "index_exists": True, "url": "http://x"})
        steps = [_make_step("publish_static_site", content)]
        assert _evidence_requirement_satisfied("published_static_site_url", steps)

    def test_running_http_service(self):
        content = json.dumps({"exposed": True, "connectable": True, "url": "http://localhost:5000"})
        steps = [_make_step("expose_local_http_service", content)]
        assert _evidence_requirement_satisfied("running_http_service", steps)

    def test_filesystem_artifact_via_write_text_file(self):
        content = json.dumps({"written": True, "exists": True, "size_bytes": 10})
        steps = [_make_step("write_text_file", content)]
        assert _evidence_requirement_satisfied("filesystem_artifact", steps)

    def test_filesystem_artifact_via_filesystem_evidence(self):
        content = json.dumps({"paths": [{"exists": True}]})
        steps = [_make_step("get_filesystem_process_evidence", content)]
        assert _evidence_requirement_satisfied("filesystem_artifact", steps)

    def test_filesystem_artifact_via_publish(self):
        content = json.dumps({"published": True, "index_exists": True, "url": "http://x"})
        steps = [_make_step("publish_static_site", content)]
        assert _evidence_requirement_satisfied("filesystem_artifact", steps)

    def test_database_mutation(self):
        steps = [_make_step("create_table", "ok")]
        assert _evidence_requirement_satisfied("database_mutation", steps)

    def test_database_mutation_write_query(self):
        steps = [_make_step("write_query", "ok")]
        assert _evidence_requirement_satisfied("database_mutation", steps)

    def test_command_output(self):
        content = json.dumps({"exit_code": 0, "stdout": "done"})
        steps = [_make_step("execute_terminal_command", content)]
        assert _evidence_requirement_satisfied("command_output", steps)

    def test_error_step_ignored(self):
        content = json.dumps({"exit_code": 0, "stdout": "done"})
        steps = [_make_step("execute_terminal_command", content, is_error=True)]
        assert not _evidence_requirement_satisfied("command_output", steps)

    def test_wrong_tool_not_satisfied(self):
        steps = [_make_step("some_tool", "ok")]
        assert not _evidence_requirement_satisfied("published_static_site_url", steps)


class TestToolNamesForContractStatus:
    def test_missing_plan(self):
        names = _tool_names_for_contract_status(
            {"evidence_requirements": ["filesystem_artifact"]},
            {"missing": ["plan"]},
        )
        assert names == {"update_plan"}

    def test_missing_evidence(self):
        names = _tool_names_for_contract_status(
            {"evidence_requirements": ["filesystem_artifact"]},
            {"missing": ["filesystem_artifact"]},
        )
        assert "write_text_file" in names or "execute_terminal_command" in names

    def test_missing_plan_open_steps(self):
        names = _tool_names_for_contract_status(
            {"evidence_requirements": ["filesystem_artifact"]},
            {"missing": ["plan_open_steps"]},
        )
        assert names == {"update_plan"}

    def test_complete(self):
        names = _tool_names_for_contract_status(
            {"evidence_requirements": ["filesystem_artifact"]},
            {"missing": []},
        )
        assert len(names) > 0


class TestShouldBlockToolForActionTask:
    def _make_contract(self):
        return {
            "mode": "execute",
            "summary": "do stuff",
            "success_criteria": ["done"],
            "evidence_requirements": ["filesystem_artifact"],
        }

    def test_blocks_delegate_when_execute_incomplete(self):
        contract = self._make_contract()
        messages = [{"role": "user", "content": "do something"}]
        steps: list[ExecutionStep] = []
        assert _should_block_tool_for_action_task(contract, messages, steps, "delegate_task")

    def test_allows_delegate_when_no_contract(self):
        assert not _should_block_tool_for_action_task(None, [], [], "delegate_task")

    def test_allows_non_delegate_tool(self):
        contract = self._make_contract()
        messages = [{"role": "user", "content": "do something"}]
        assert not _should_block_tool_for_action_task(contract, messages, [], "execute_terminal_command")


class TestLastHostCommand:
    def test_returns_last_command(self):
        steps = [
            _make_step("execute_terminal_command", "ok"),
        ]
        steps[0].metadata["arguments"] = {"command": "ls -la"}
        cmd = _last_host_command(steps)
        assert cmd == "ls -la"

    def test_empty_steps(self):
        assert _last_host_command([]) == ""


class TestBlockedActionToolMessage:
    def test_message_contains_tool_name(self):
        msg = _blocked_action_tool_message("delegate_task")
        assert "delegate_task" in msg


class TestDuplicateCommandMessage:
    def test_message_contains_tool_name(self):
        msg = _duplicate_command_message("execute_terminal_command")
        assert "execute_terminal_command" in msg


class TestBuildIncompleteContractCapMessage:
    def test_contains_prompt_and_missing(self):
        steps: list[ExecutionStep] = []
        msg = _build_incomplete_contract_cap_message(
            "build a site",
            {"missing": ["filesystem_artifact"]},
            steps,
        )
        assert "build a site" in msg
        assert "filesystem_artifact" in msg


class TestCanStreamTextBeforeFinal:
    def test_answer_mode_always_can_stream(self):
        contract = {"mode": "answer"}
        assert _can_stream_text_before_final(contract, [], [])

    def test_none_contract_cannot_stream(self):
        assert not _can_stream_text_before_final(None, [], [])

    def test_execute_incomplete_cannot_stream(self):
        contract = {
            "mode": "execute",
            "evidence_requirements": ["filesystem_artifact"],
        }
        assert not _can_stream_text_before_final(contract, [{"role": "user", "content": "go"}], [])


# ===========================================================================
# planning.py — pruning and compaction
# ===========================================================================

class TestPruneMessageWindow:
    def _make_messages(self, n: int) -> list[dict]:
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(n):
            msgs.append({"role": "user", "content": f"user {i}"})
            msgs.append({"role": "assistant", "content": f"assistant {i}"})
        return msgs

    def test_no_pruning_when_short(self):
        msgs = self._make_messages(5)
        result = _prune_message_window(msgs)
        assert result == msgs

    def test_prunes_long_history(self):
        # Build a list longer than _MAX_HISTORY_MESSAGES
        msgs = self._make_messages(_MAX_HISTORY_MESSAGES + 5)
        result = _prune_message_window(msgs)
        assert len(result) < len(msgs)
        # Should contain a compaction summary
        contents = [m.get("content", "") for m in result if m.get("role") == "system"]
        assert any("compacted" in str(c) for c in contents)

    def test_does_not_orphan_tool_result(self):
        """recent_start must not point at a tool message."""
        msgs = self._make_messages(_MAX_HISTORY_MESSAGES + 5)
        # Insert a tool message right where the cutpoint would land
        msgs.insert(len(msgs) - _KEEP_RECENT_MESSAGES, {"role": "tool", "content": "result"})
        result = _prune_message_window(msgs)
        # First message in the kept-recent window should not be a tool
        recent = [m for m in result if m.get("role") in ("user", "assistant", "tool")]
        if recent:
            assert recent[-_KEEP_RECENT_MESSAGES]["role"] != "tool"


class TestBuildHistoryCompactionSummary:
    def test_captures_user_and_assistant(self):
        msgs = [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there", "tool_calls": [
                {"function": {"name": "some_tool"}}
            ]},
        ]
        summary = _build_history_compaction_summary(msgs)
        assert "hello world" in summary
        assert "hi there" in summary
        assert "some_tool" in summary

    def test_empty_messages(self):
        summary = _build_history_compaction_summary([])
        assert "None recorded" in summary

    def test_malformed_tool_call_ignored(self):
        msgs = [
            {"role": "assistant", "content": "ok", "tool_calls": [{"no_function": True}]},
        ]
        # Should not raise
        summary = _build_history_compaction_summary(msgs)
        assert "compacted" in summary


class TestBuildPlanContinuationInstruction:
    def _make_plan_msg(self, steps):
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

    def test_contains_pending_steps(self):
        steps = [{"title": "run tests", "status": "pending"}]
        msgs = [{"role": "user", "content": "go"}, self._make_plan_msg(steps)]
        instruction = _build_plan_continuation_instruction(msgs)
        assert "run tests" in instruction
        assert "update_plan" in instruction.lower() or "not-done" in instruction


class TestCountSuccessfulSideEffects:
    def test_counts_correct(self):
        steps = [
            _make_step("execute_terminal_command", json.dumps({"exit_code": 0, "stdout": "ok"})),
            _make_step("write_text_file", json.dumps({"written": True, "exists": True, "size_bytes": 10})),
            _make_step("execute_terminal_command", "fail", is_error=True),
        ]
        count = _count_successful_side_effects(steps)
        assert count == 2  # only non-error side-effect tools


@pytest.mark.asyncio
class TestStoreIterationCapMemory:
    async def test_no_store_event_attr_skips(self):
        memory = object()  # no store_event method
        await _store_iteration_cap_memory(memory, "s1", "prompt", ["tool_a"], "response", 16)
        # Should not raise

    async def test_calls_store_event(self):
        events = []

        class FakeMemory:
            def store_event(self, session_id, raw_text, entities):
                events.append((session_id, raw_text))

        memory = FakeMemory()
        await _store_iteration_cap_memory(memory, "s1", "my task", ["tool_a"], "response", 16)
        assert events
        assert "my task" in events[0][1]
        assert "16" in events[0][1]


class TestBuildContractContinuationInstruction:
    def test_no_contract(self):
        instruction = _build_contract_continuation_instruction(
            None, {"missing": []}, "some text", [], []
        )
        assert "set_task_contract" in instruction

    def test_with_contract(self):
        contract = {"mode": "execute", "summary": "build site",
                    "success_criteria": ["done"], "evidence_requirements": ["filesystem_artifact"]}
        status = {"missing": ["filesystem_artifact"], "plan_open": False}
        instruction = _build_contract_continuation_instruction(
            contract, status, "partial answer", [{"role": "user", "content": "go"}], []
        )
        assert "filesystem_artifact" in instruction
        assert "partial answer" in instruction


# ===========================================================================
# llm_utils.py — retry logic
# ===========================================================================

class TestLlmRetry:
    @pytest.mark.asyncio
    async def test_acompletion_with_retry_success_on_first_try(self):
        with unittest.mock.patch("litellm.acompletion", return_value="response") as mock:
            result = await _acompletion_with_retry(model="m", messages=[])
        assert result == "response"
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_acompletion_with_retry_raises_non_rate_limit(self):
        with unittest.mock.patch("litellm.acompletion", side_effect=ValueError("bad")):
            with pytest.raises(ValueError, match="bad"):
                await _acompletion_with_retry(model="m", messages=[])

    @pytest.mark.asyncio
    async def test_acompletion_with_retry_retries_on_rate_limit(self):
        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("rate limit exceeded")
            return "ok"

        with (
            unittest.mock.patch("litellm.acompletion", side_effect=side_effect),
            unittest.mock.patch("llm_utils._RATE_LIMIT_BASE_DELAY", 0.001),
            unittest.mock.patch("asyncio.sleep", return_value=None),
        ):
            result = await _acompletion_with_retry(model="m", messages=[])
        assert result == "ok"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_acompletion_stream_with_retry_success(self):
        mock_stream = unittest.mock.AsyncMock()
        with unittest.mock.patch("litellm.acompletion", return_value=mock_stream):
            result = await _acompletion_stream_with_retry(model="m", messages=[])
        assert result is mock_stream

    def test_rate_limit_user_message_is_string(self):
        msg = _rate_limit_user_message()
        assert isinstance(msg, str)
        assert "rate limit" in msg.lower()

    def test_is_rate_limit_error_generic_string(self):
        assert _is_rate_limit_error(RuntimeError("rate_limit exceeded"))

    def test_is_rate_limit_error_ratelimit(self):
        assert _is_rate_limit_error(Exception("RateLimitError: quota"))


class TestPrepareMessages:
    def test_injection_when_only_system_messages(self):
        msgs = [{"role": "system", "content": "you are..."}]
        result = _prepare_llm_request_messages(msgs, original_prompt="do something")
        assert any(m["role"] == "user" for m in result)

    def test_no_injection_when_user_present(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        result = _prepare_llm_request_messages(msgs)
        user_msgs = [m for m in result if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "hi"

    def test_injection_without_prompt(self):
        msgs = [{"role": "system", "content": "sys"}]
        result = _prepare_llm_request_messages(msgs)
        assert any(m["role"] == "user" for m in result)
        injected = next(m for m in result if m["role"] == "user")
        assert "continue" in injected["content"].lower()


# ===========================================================================
# checkpointer.py — async SQLite operations
# ===========================================================================

class TestStateCheckpointer:
    @pytest.mark.asyncio
    async def test_save_and_load_checkpoint(self, tmp_path):
        from checkpointer import StateCheckpointer, initialize_checkpoints_db

        db = tmp_path / "test.db"
        await initialize_checkpoints_db(db)
        cp = StateCheckpointer(db)

        checkpoint_id = await cp.save_checkpoint(
            session_id="sess1",
            step_number=1,
            state_payload={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert checkpoint_id

        loaded = await cp.load_checkpoint(checkpoint_id)
        assert loaded is not None
        assert loaded["messages"][0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_load_nonexistent_raises_key_error(self, tmp_path):
        from checkpointer import StateCheckpointer, initialize_checkpoints_db

        db = tmp_path / "test.db"
        await initialize_checkpoints_db(db)
        cp = StateCheckpointer(db)
        with pytest.raises(KeyError, match="nonexistent-id"):
            await cp.load_checkpoint("nonexistent-id")

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, tmp_path):
        from checkpointer import StateCheckpointer, initialize_checkpoints_db

        db = tmp_path / "test.db"
        await initialize_checkpoints_db(db)
        cp = StateCheckpointer(db)

        for i in range(3):
            await cp.save_checkpoint("sess-list", i, {"step": i})

        checkpoints = await cp.list_checkpoints("sess-list")
        assert len(checkpoints) == 3
        assert checkpoints[0]["step_number"] < checkpoints[1]["step_number"]

    @pytest.mark.asyncio
    async def test_pruning_retains_max(self, tmp_path):
        from checkpointer import _RETAIN_PER_SESSION, StateCheckpointer, initialize_checkpoints_db

        db = tmp_path / "test.db"
        await initialize_checkpoints_db(db)
        cp = StateCheckpointer(db)

        save_count = _RETAIN_PER_SESSION + 5
        for i in range(save_count):
            await cp.save_checkpoint("sess-prune", i, {"step": i})

        checkpoints = await cp.list_checkpoints("sess-prune")
        assert len(checkpoints) <= _RETAIN_PER_SESSION
