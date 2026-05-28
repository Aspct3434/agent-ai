"""Tests for evidence-gated skill evolution + the auto skill maker.

Covers the unique self-improvement mechanism: per-version metrics, the
validation gate on candidates, archive + promotion, and automatic rollback
of a regressed version.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluator import SkillRegistry, _skill_function_names

_GOOD = (
    "from _skill import skill\n\n\n"
    "@skill\n"
    "def greet(name: str) -> str:\n"
    '    "Greet a person."\n'
    '    return f"Hi {name}"\n'
)
_IMPROVED = (
    "from _skill import skill\n\n\n"
    "@skill\n"
    'def greet(name: str = "world") -> str:\n'
    '    "Greet a person, improved."\n'
    '    return f"Hello {name}!"\n'
)
_DROPS_FUNCTION = (
    "from _skill import skill\n\n\n"
    "@skill\n"
    "def something_else() -> str:\n"
    '    "Different function."\n'
    '    return "x"\n'
)
_BAD_SYNTAX = "from _skill import skill\n@skill\ndef greet(:\n"
_NO_DECORATOR = "def greet(name):\n    return name\n"
_STATEFUL = (
    "from _skill import skill\n\n\n"
    "@skill(name=\"write_report\", description=\"Write a report.\", "
    "changes_state=True, evidence_types=[\"filesystem_artifact\"])\n"
    "def write_report(path: str) -> dict[str, object]:\n"
    '    "Write a report file."\n'
    "    return {\"success\": True, \"path\": path}\n"
)


def _reg(tmp_path, **kw) -> SkillRegistry:
    return SkillRegistry(skills_dir=tmp_path, model=kw.pop("model", None), **kw)


class TestFunctionNames:
    def test_extracts_top_level_functions(self) -> None:
        assert _skill_function_names(_GOOD) == {"greet"}

    def test_invalid_syntax_returns_empty(self) -> None:
        assert _skill_function_names(_BAD_SYNTAX) == set()


class TestCreateSkill:
    def test_writes_valid_skill_and_registers(self, tmp_path) -> None:
        reg = _reg(tmp_path)
        path = reg.create_skill("My Tool", _GOOD, description="greets")
        assert Path(path).exists()
        assert Path(path).stem == "my_tool"
        assert reg._stats["my_tool"]["created_by"] == "agent"

    def test_lists_stateful_skill_metadata_by_tool_name(self, tmp_path) -> None:
        reg = _reg(tmp_path)
        (tmp_path / "write_report_file.py").write_text(_STATEFUL, encoding="utf-8")

        listed = reg.list_skills()
        metadata = reg.skill_metadata_for_tool("write_report")

        assert listed[0]["tool_names"] == ["write_report"]
        assert listed[0]["changes_state"] is True
        assert listed[0]["evidence_types"] == ["filesystem_artifact"]
        assert metadata is not None
        assert metadata["changes_state"] is True

    def test_rejects_bad_syntax(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            _reg(tmp_path).create_skill("x", _BAD_SYNTAX)

    def test_requires_skill_decorator(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            _reg(tmp_path).create_skill("x", _NO_DECORATOR)


class TestRecordUse:
    def test_tracks_per_version_and_failures(self, tmp_path) -> None:
        reg = _reg(tmp_path)
        reg.record_use("s", success=True)
        reg.record_use("s", success=False)
        st = reg._stats["s"]
        assert st["versions"]["1"] == {"uses": 2, "successes": 1}
        assert st["failures_since_improve"] == 1
        assert st["success_rate"] == 0.5


class TestValidationGate:
    def test_accepts_valid_candidate(self, tmp_path) -> None:
        ok, _ = _reg(tmp_path)._validate_candidate(_GOOD, _IMPROVED)
        assert ok

    def test_rejects_invalid_syntax(self, tmp_path) -> None:
        ok, reason = _reg(tmp_path)._validate_candidate(_GOOD, _BAD_SYNTAX)
        assert not ok and "syntax" in reason

    def test_rejects_missing_decorator(self, tmp_path) -> None:
        ok, reason = _reg(tmp_path)._validate_candidate(_GOOD, _NO_DECORATOR)
        assert not ok and "@skill" in reason

    def test_rejects_dropped_function(self, tmp_path) -> None:
        ok, reason = _reg(tmp_path)._validate_candidate(_GOOD, _DROPS_FUNCTION)
        assert not ok and "function" in reason


class TestMaybeImprove:
    @pytest.mark.asyncio
    async def test_promotes_valid_candidate_and_archives(self, tmp_path) -> None:
        reg = _reg(tmp_path, model="x", improve_after_uses=2)
        (tmp_path / "greet.py").write_text(_GOOD, encoding="utf-8")
        reg.record_use("greet", success=True)
        reg.record_use("greet", success=True)
        reg._synthesize_improvement = AsyncMock(return_value=_IMPROVED)

        assert await reg.maybe_improve("greet") is True
        assert reg._stats["greet"]["version"] == 2
        assert "Hello" in (tmp_path / "greet.py").read_text(encoding="utf-8")
        assert (tmp_path / ".versions" / "greet" / "v1.py").exists()
        assert reg._stats["greet"]["versions"]["2"] == {"uses": 0, "successes": 0}

    @pytest.mark.asyncio
    async def test_rejects_candidate_that_drops_function(self, tmp_path) -> None:
        reg = _reg(tmp_path, model="x", improve_after_uses=2)
        (tmp_path / "greet.py").write_text(_GOOD, encoding="utf-8")
        reg.record_use("greet", success=True)
        reg.record_use("greet", success=True)
        reg._synthesize_improvement = AsyncMock(return_value=_DROPS_FUNCTION)

        assert await reg.maybe_improve("greet") is False
        assert reg._stats["greet"]["version"] == 1
        assert "greet" in (tmp_path / "greet.py").read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_failure_threshold_triggers_improvement(self, tmp_path) -> None:
        reg = _reg(tmp_path, model="x", improve_after_uses=999)  # never matures by use-count
        reg._improve_after_failures = 2
        (tmp_path / "greet.py").write_text(_GOOD, encoding="utf-8")
        reg.record_use("greet", success=False)
        reg.record_use("greet", success=False)
        reg._synthesize_improvement = AsyncMock(return_value=_IMPROVED)
        assert await reg.maybe_improve("greet") is True


class TestRollback:
    def _setup_v2(self, tmp_path, v1_stats, v2_stats) -> SkillRegistry:
        reg = _reg(tmp_path)
        reg._rollback_min_samples = 3
        (tmp_path / "greet.py").write_text(_IMPROVED, encoding="utf-8")  # live = v2
        vdir = tmp_path / ".versions" / "greet"
        vdir.mkdir(parents=True)
        (vdir / "v1.py").write_text(_GOOD, encoding="utf-8")
        reg._stats["greet"] = {"version": 2, "versions": {"1": v1_stats, "2": v2_stats}}
        return reg

    def test_rolls_back_on_regression(self, tmp_path) -> None:
        reg = self._setup_v2(
            tmp_path,
            v1_stats={"uses": 5, "successes": 5},
            v2_stats={"uses": 4, "successes": 1},
        )
        assert reg.maybe_rollback("greet") is True
        assert reg._stats["greet"]["version"] == 1
        assert "Hi " in (tmp_path / "greet.py").read_text(encoding="utf-8")  # v1 restored
        assert reg._stats["greet"]["rollback_count"] == 1

    def test_no_rollback_when_not_worse(self, tmp_path) -> None:
        reg = self._setup_v2(
            tmp_path,
            v1_stats={"uses": 5, "successes": 4},
            v2_stats={"uses": 4, "successes": 4},
        )
        assert reg.maybe_rollback("greet") is False
        assert reg._stats["greet"]["version"] == 2

    def test_no_rollback_without_enough_samples(self, tmp_path) -> None:
        reg = self._setup_v2(
            tmp_path,
            v1_stats={"uses": 5, "successes": 5},
            v2_stats={"uses": 1, "successes": 0},  # below min_samples
        )
        assert reg.maybe_rollback("greet") is False
        assert reg._stats["greet"]["version"] == 2


class TestMaybeEvolve:
    @pytest.mark.asyncio
    async def test_prefers_rollback(self, tmp_path) -> None:
        reg = _reg(tmp_path, model="x")
        reg._rollback_min_samples = 3
        (tmp_path / "greet.py").write_text(_IMPROVED, encoding="utf-8")
        vdir = tmp_path / ".versions" / "greet"
        vdir.mkdir(parents=True)
        (vdir / "v1.py").write_text(_GOOD, encoding="utf-8")
        reg._stats["greet"] = {
            "version": 2,
            "versions": {"1": {"uses": 5, "successes": 5}, "2": {"uses": 4, "successes": 0}},
        }
        assert await reg.maybe_evolve("greet") == "rolled_back"

    @pytest.mark.asyncio
    async def test_improves_when_no_regression(self, tmp_path) -> None:
        reg = _reg(tmp_path, model="x", improve_after_uses=2)
        (tmp_path / "greet.py").write_text(_GOOD, encoding="utf-8")
        reg.record_use("greet", success=True)
        reg.record_use("greet", success=True)
        reg._synthesize_improvement = AsyncMock(return_value=_IMPROVED)
        assert await reg.maybe_evolve("greet") == "improved"

    @pytest.mark.asyncio
    async def test_unchanged_when_immature(self, tmp_path) -> None:
        reg = _reg(tmp_path, model="x", improve_after_uses=5)
        (tmp_path / "greet.py").write_text(_GOOD, encoding="utf-8")
        reg.record_use("greet", success=True)
        assert await reg.maybe_evolve("greet") == "unchanged"
