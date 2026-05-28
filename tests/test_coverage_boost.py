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

from agent import (
    AgentEngine,
    _filesystem_process_evidence_has_negative_findings,
    _should_emit_tool_call_progress,
)
from contract import (
    MAX_CONSECUTIVE_VERIFICATION_CALLS,
    _evidence_observation_tags,
    _evidence_requirement_satisfied,
    _expose_local_http_service_evidence_is_positive,
    _filesystem_process_evidence_is_positive,
    _successful_command_output_evidence,
    _write_text_file_evidence_is_positive,
    background_service_misuse_message,
    blocked_action_tool_message,
    build_incomplete_contract_cap_message,
    can_stream_text_before_final,
    consecutive_verification_run_length,
    contract_completion_status,
    duplicate_command_message,
    evidence_requirement_satisfied_by_steps,
    last_host_command,
    looks_like_finite_background_command,
    should_block_tool_for_action_task,
    terminal_failure_recovery_message,
    terminal_failure_since_diagnostic,
    tool_call_is_verification_probe,
    tool_names_for_contract_status,
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
            "success_criteria": ["Highly interactive website file exists"],
            "evidence_requirements": ["filesystem_artifact"],
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
                        "written": True,
                        "exists": True,
                        "size_bytes": 42,
                        "artifact_quality": {
                            "placeholder_detected": False,
                            "interactive_signal_count": 0,
                            "style_rule_count": 1,
                            "content_word_count": 12,
                        },
                    }
                ),
                metadata={"tool_name": "write_text_file", "is_error": False},
            )
        ]

        status = contract_completion_status(
            contract, messages, steps, contract_required=True
        )

        assert status["complete"] is False
        assert "artifact_quality" in status["missing"]

    def test_generated_skill_metadata_satisfies_contract_evidence(self):
        step = ExecutionStep(
            kind="tool_result",
            content=json.dumps(
                {
                    "success": True,
                    "changes_state": True,
                    "evidence_types": ["filesystem_artifact"],
                    "path": "/tmp/generated/index.html",
                    "exists": True,
                }
            ),
            metadata={
                "tool_name": "write_topic_site",
                "server": "skills",
                "is_error": False,
                "changes_state": True,
                "evidence_types": ["filesystem_artifact"],
            },
        )

        assert evidence_requirement_satisfied_by_steps("filesystem_artifact", [step])
        assert _count_successful_side_effects([step]) == 1


class TestExposeLocalHttpServiceEvidence:
    def test_positive(self):
        content = json.dumps({"exposed": True, "connectable": True, "url": "http://localhost:5000"})
        assert _expose_local_http_service_evidence_is_positive(content)

    def test_not_connectable(self):
        content = json.dumps({"exposed": True, "connectable": False, "url": "http://x"})
        assert not _expose_local_http_service_evidence_is_positive(content)

    def test_invalid_json(self):
        assert not _expose_local_http_service_evidence_is_positive("nope")


class TestObservationEvidenceTags:
    def test_extracts_http_response_without_tool_name(self):
        content = json.dumps({"status_code": 204, "content_type": "application/json"})
        assert "http_response" in _evidence_observation_tags(content)

    def test_extracts_file_process_port_and_command_observations(self):
        content = json.dumps(
            {
                "paths": [{"exists": True}],
                "pids": [{"running": True}],
                "ports": [{"connectable": True}],
                "exit_code": 0,
                "stdout": "ok",
            }
        )
        tags = _evidence_observation_tags(content)
        assert {"filesystem_artifact", "process_running", "tcp_open", "command_success"} <= tags

    def test_error_status_suppresses_positive_observations(self):
        content = json.dumps({"status": "error", "status_code": 200, "paths": [{"exists": True}]})
        assert _evidence_observation_tags(content) == set()


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


class TestFilesystemProcessEvidenceNegativeFindings:
    def test_missing_path_is_negative_finding(self):
        content = json.dumps({"paths": [{"path": "/usr/bin/java", "exists": False}]})
        assert _filesystem_process_evidence_has_negative_findings(content)

    def test_open_port_is_not_negative_finding(self):
        content = json.dumps({"ports": [{"port": 25565, "connectable": True}]})
        assert not _filesystem_process_evidence_has_negative_findings(content)

    def test_mixed_ports_are_not_negative_when_one_target_is_open(self):
        content = json.dumps(
            {
                "ports": [
                    {"port": 25565, "connectable": True},
                    {"port": 25575, "connectable": False},
                ]
            }
        )
        assert not _filesystem_process_evidence_has_negative_findings(content)

    def test_all_closed_ports_are_negative(self):
        content = json.dumps({"ports": [{"port": 25565, "connectable": False}]})
        assert _filesystem_process_evidence_has_negative_findings(content)


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


def _make_step(
    tool_name: str,
    content: str,
    is_error: bool = False,
    arguments: dict[str, object] | None = None,
) -> ExecutionStep:
    return ExecutionStep(
        kind="tool_result",
        content=content,
        metadata={"tool_name": tool_name, "is_error": is_error, "arguments": arguments or {}},
    )


class TestEvidenceRequirementSatisfied:
    def test_running_http_service(self):
        content = json.dumps({"exposed": True, "connectable": True, "url": "http://localhost:5000"})
        steps = [_make_step("expose_local_http_service", content)]
        assert _evidence_requirement_satisfied("running_http_service", steps)

    def test_http_observation_requires_service_provenance(self):
        content = json.dumps(
            {
                "url": "https://example.com",
                "status_code": 200,
                "content_type": "text/html",
            }
        )
        steps = [_make_step("custom_probe", content)]
        assert not _evidence_requirement_satisfied("running_http_service", steps)

    def test_http_observation_matches_launched_service_port(self):
        launch = _make_step(
            "execute_background_service",
            json.dumps({"status": "launched", "pid": 123}),
            arguments={"command": "python3 -m http.server 8080"},
        )
        probe = _make_step(
            "custom_probe",
            json.dumps({"url": "http://127.0.0.1:8080/", "status_code": 200}),
        )
        assert evidence_requirement_satisfied_by_steps(
            "running_http_service",
            [probe],
            context_steps=[launch, probe],
        )

    def test_running_tcp_service_via_wait_for_port(self):
        content = json.dumps({"open": True, "port": 25565})
        steps = [_make_step("wait_for_port", content)]
        assert _evidence_requirement_satisfied("running_tcp_service", steps)

    def test_running_tcp_service_via_process_evidence(self):
        content = json.dumps({"ports": [{"port": 25565, "connectable": True}]})
        steps = [_make_step("get_filesystem_process_evidence", content)]
        assert _evidence_requirement_satisfied("running_tcp_service", steps)

    def test_running_tcp_service_not_satisfied_by_existing_file(self):
        content = json.dumps({"paths": [{"path": "/workspace/eula.txt", "exists": True}], "ports": []})
        steps = [_make_step("get_filesystem_process_evidence", content)]
        assert not _evidence_requirement_satisfied("running_tcp_service", steps)

    def test_filesystem_artifact_via_write_text_file(self):
        content = json.dumps({"written": True, "exists": True, "size_bytes": 10})
        steps = [_make_step("write_text_file", content)]
        assert _evidence_requirement_satisfied("filesystem_artifact", steps)

    def test_filesystem_artifact_via_filesystem_evidence(self):
        content = json.dumps({"paths": [{"exists": True}]})
        steps = [_make_step("get_filesystem_process_evidence", content)]
        assert _evidence_requirement_satisfied("filesystem_artifact", steps)

    def test_filesystem_artifact_requires_all_checked_paths(self):
        content = json.dumps({"paths": [{"path": "/usr/bin/java", "exists": False}, {"path": "/workspace/eula.txt", "exists": True}]})
        steps = [_make_step("get_filesystem_process_evidence", content)]
        assert not _evidence_requirement_satisfied("filesystem_artifact", steps)

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
        assert not _evidence_requirement_satisfied("running_http_service", steps)

    def test_failed_http_verification_never_satisfies_service_evidence(self):
        steps = [
            _make_step(
                "execute_background_service",
                json.dumps({"status": "launched", "pid": 123}),
            ),
            _make_step(
                "write_text_file",
                json.dumps({"written": True, "exists": True, "size_bytes": 10}),
            ),
            *[
                _make_step(
                    "expose_local_http_service",
                    "[expose_local_http_service error] connection refused",
                    is_error=True,
                )
                for _ in range(4)
            ],
        ]
        assert not _evidence_requirement_satisfied("running_http_service", steps)


class TestConsecutiveVerificationRun:
    def test_counts_varied_verification_tools(self):
        steps = [
            _make_step("write_text_file", "{}"),
            _make_step("web_fetch", "{}", is_error=True),
            _make_step("get_filesystem_process_evidence", "{}"),
        ]
        assert consecutive_verification_run_length(steps) == 2

    def test_state_changing_tool_breaks_streak(self):
        steps = [
            _make_step("web_fetch", "{}", is_error=True),
            _make_step("write_text_file", "{}"),
        ]
        assert consecutive_verification_run_length(steps) == 0

    def test_observation_terminal_command_counts_as_verification(self):
        steps = [
            _make_step("web_fetch", "{}", is_error=True),
            _make_step(
                "execute_terminal_command",
                json.dumps({"exit_code": 0, "stdout": "200", "stderr": ""}),
                arguments={"command": "curl -sI http://127.0.0.1:8080/"},
            ),
        ]
        assert consecutive_verification_run_length(steps) == 2
        assert tool_call_is_verification_probe(
            "execute_terminal_command",
            {"command": "curl -sI http://127.0.0.1:8080/"},
        )

    def test_declared_state_changing_terminal_command_breaks_streak(self):
        steps = [
            _make_step("web_fetch", "{}", is_error=True),
            _make_step(
                "execute_terminal_command",
                json.dumps({"exit_code": 0, "stdout": "built", "stderr": ""}),
                arguments={"command": "npm run build", "changes_state": True},
            ),
        ]
        assert consecutive_verification_run_length(steps) == 0


class TestBackgroundServiceCommandClassifier:
    def test_blocks_shell_probe_chain(self):
        command = "ss -tlnp | grep 3000 || netstat -tlnp | grep 3000 || lsof -i :3000"
        assert looks_like_finite_background_command(command)
        assert "finite/probe" in str(background_service_misuse_message(command))

    def test_blocks_common_finite_commands(self):
        for command in (
            "curl -s http://localhost:3000",
            "ls -la site && wc -l site/index.html",
            "npm run build",
            "python -m pytest",
        ):
            assert looks_like_finite_background_command(command), command

    def test_allows_common_long_running_servers(self):
        for command in (
            "cd site && python3 -m http.server 3000",
            "uvicorn app:app --host 0.0.0.0 --port 8000",
            "npm run dev -- --host 0.0.0.0",
            "python app.py",
        ):
            assert not looks_like_finite_background_command(command), command
            assert background_service_misuse_message(command) is None


class TestVerificationLoopDispatchGuard:
    @pytest.mark.asyncio
    async def test_parallel_verification_tool_is_blocked_after_cap(self):
        class DummyEngine:
            def _task_graph_tool_block(self, *_args, **_kwargs):
                return None

            async def _execute_single_tool(self, *_args, **_kwargs):
                raise AssertionError("verification guard should block before execution")

        steps = [
            _make_step("web_fetch", "{}", is_error=True)
            for _ in range(MAX_CONSECUTIVE_VERIFICATION_CALLS)
        ]
        messages: list[dict[str, object]] = []

        error_count, events = await AgentEngine._dispatch_tool_calls(
            DummyEngine(),
            calls=[("tc-next", "web_fetch", {"url": "http://127.0.0.1:9999"})],
            messages=messages,
            steps=steps,
            tool_index={},
            session_id="test",
        )

        assert error_count == 1
        assert "verification loop" in steps[-1].content
        assert events[-1]["is_error"] is True

    @pytest.mark.asyncio
    async def test_background_probe_is_blocked_before_execution(self):
        class DummyEngine:
            def _task_graph_tool_block(self, *_args, **_kwargs):
                return None

            async def _execute_single_tool(self, *_args, **_kwargs):
                raise AssertionError("background probe guard should block before execution")

        steps: list[ExecutionStep] = []
        messages: list[dict[str, object]] = []

        error_count, events = await AgentEngine._dispatch_tool_calls(
            DummyEngine(),
            calls=[
                (
                    "tc-probe",
                    "execute_background_service",
                    {"command": "ss -tlnp | grep 3000 || lsof -i :3000"},
                )
            ],
            messages=messages,
            steps=steps,
            tool_index={},
            session_id="test",
        )

        assert error_count == 1
        assert "finite/probe" in steps[-1].content
        assert events[-1]["is_error"] is True

    @pytest.mark.asyncio
    async def test_terminal_observation_probe_does_not_reset_verification_cap(self):
        class DummyEngine:
            def _task_graph_tool_block(self, *_args, **_kwargs):
                return None

            def _task_graph_explicitly_allows(self, *_args, **_kwargs):
                return False

            async def _execute_single_tool(self, *_args, **_kwargs):
                raise AssertionError("verification guard should block before execution")

        steps = []
        for i in range(MAX_CONSECUTIVE_VERIFICATION_CALLS):
            if i % 2:
                steps.append(
                    _make_step(
                        "execute_terminal_command",
                        json.dumps({"exit_code": 0, "stdout": "ok", "stderr": ""}),
                        arguments={"command": f"curl -s http://127.0.0.1:{8000 + i}/"},
                    )
                )
            else:
                steps.append(_make_step("web_fetch", "{}", is_error=True))

        error_count, events = await AgentEngine._dispatch_tool_calls(
            DummyEngine(),
            calls=[
                (
                    "tc-next",
                    "execute_terminal_command",
                    {"command": "curl -s http://127.0.0.1:8080/"},
                )
            ],
            messages=[],
            steps=steps,
            tool_index={},
            session_id="test",
        )

        assert error_count == 1
        assert "verification loop" in steps[-1].content
        assert events[-1]["is_error"] is True


class TestToolCallProgressEmission:
    def test_suppresses_duplicate_background_service_in_same_batch(self):
        prior = [
            (
                "tc-1",
                "execute_background_service",
                {"command": "python3 -m http.server 3000"},
            )
        ]

        assert not _should_emit_tool_call_progress(
            "execute_background_service",
            {"command": "  python3   -m   http.server   3000  "},
            prior,
        )

    def test_allows_non_duplicate_background_service_progress(self):
        assert _should_emit_tool_call_progress(
            "execute_background_service",
            {"command": "python3 -m http.server 3001"},
            [],
        )


class TestToolNamesForContractStatus:
    def test_missing_plan(self):
        names = tool_names_for_contract_status(
            {"evidence_requirements": ["filesystem_artifact"]},
            {"missing": ["plan"]},
        )
        assert names == {"update_plan"}

    def test_missing_evidence(self):
        names = tool_names_for_contract_status(
            {"evidence_requirements": ["filesystem_artifact"]},
            {"missing": ["filesystem_artifact"]},
        )
        assert "write_text_file" in names or "execute_terminal_command" in names

    def test_missing_tcp_evidence(self):
        names = tool_names_for_contract_status(
            {"evidence_requirements": ["running_tcp_service"]},
            {"missing": ["running_tcp_service"]},
        )
        assert "wait_for_port" in names
        assert "get_filesystem_process_evidence" in names

    def test_missing_plan_open_steps(self):
        names = tool_names_for_contract_status(
            {"evidence_requirements": ["filesystem_artifact"]},
            {"missing": ["plan_open_steps"]},
        )
        assert names == {"update_plan"}

    def test_complete(self):
        names = tool_names_for_contract_status(
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
        assert should_block_tool_for_action_task(contract, messages, steps, "delegate_task")

    def test_allows_delegate_when_no_contract(self):
        assert not should_block_tool_for_action_task(None, [], [], "delegate_task")

    def test_allows_non_delegate_tool(self):
        contract = self._make_contract()
        messages = [{"role": "user", "content": "do something"}]
        assert not should_block_tool_for_action_task(contract, messages, [], "execute_terminal_command")


class TestLastHostCommand:
    def test_returns_last_command(self):
        steps = [
            _make_step("execute_terminal_command", "ok"),
        ]
        steps[0].metadata["arguments"] = {"command": "ls -la"}
        cmd = last_host_command(steps)
        assert cmd == "ls -la"

    def test_empty_steps(self):
        assert last_host_command([]) == ""


class TestBlockedActionToolMessage:
    def test_message_contains_tool_name(self):
        msg = blocked_action_tool_message("delegate_task")
        assert "delegate_task" in msg


class TestDuplicateCommandMessage:
    def test_message_contains_tool_name(self):
        msg = duplicate_command_message("execute_terminal_command")
        assert "execute_terminal_command" in msg


class TestTerminalFailureRecovery:
    def test_allows_one_terminal_failure_before_requiring_diagnostic(self):
        failed = _make_step(
            "execute_terminal_command",
            json.dumps({"exit_code": 127, "stderr": "java: not found"}),
            is_error=True,
        )
        assert not terminal_failure_since_diagnostic([failed])

        second_failed = _make_step(
            "execute_terminal_command",
            json.dumps({"exit_code": 127, "stderr": "which: not found"}),
            is_error=True,
        )
        assert terminal_failure_since_diagnostic([failed, second_failed])

        diagnostic = _make_step("get_system_environment", "{}", is_error=False)
        assert not terminal_failure_since_diagnostic([failed, second_failed, diagnostic])

    def test_recovery_message_mentions_diagnostic_tools(self):
        msg = terminal_failure_recovery_message("java -version")
        assert "get_system_environment" in msg
        assert "java -version" in msg


class TestBuildIncompleteContractCapMessage:
    def test_contains_prompt_and_missing(self):
        steps: list[ExecutionStep] = []
        msg = build_incomplete_contract_cap_message(
            "build a site",
            {"missing": ["filesystem_artifact"]},
            steps,
        )
        assert "build a site" in msg
        assert "filesystem_artifact" in msg


class TestCanStreamTextBeforeFinal:
    def test_answer_mode_always_can_stream(self):
        contract = {"mode": "answer"}
        assert can_stream_text_before_final(contract, [], [])

    def test_none_contract_cannot_stream(self):
        assert not can_stream_text_before_final(None, [], [])

    def test_execute_incomplete_cannot_stream(self):
        contract = {
            "mode": "execute",
            "evidence_requirements": ["filesystem_artifact"],
        }
        assert not can_stream_text_before_final(contract, [{"role": "user", "content": "go"}], [])


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
            _make_step(
                "execute_terminal_command",
                json.dumps({"exit_code": 0, "stdout": "ok"}),
                arguments={"command": "pip install flask", "changes_state": True},
            ),
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
