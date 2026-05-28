"""Proof-carrying self-improvement for agent-ai.

The evolution layer stages candidate improvements, verifies them, promotes only
with a proof bundle, and keeps rollback data. V1 handles executable skills plus
additive prompt/toolset policy snippets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from evaluator import (
    ExecutionTrajectory,
    SkillRegistry,
    _extract_skill_description,
    _extract_skill_tags,
    _skill_function_names,
    _slugify,
    _validate_python_syntax,
)
from sqlite_migrations import SQLiteMigration, apply_sqlite_migrations

logger = logging.getLogger(__name__)

_EVOLUTION_MIGRATIONS = (
    SQLiteMigration(
        version=1,
        name="create_evolution_ledger",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS evolution_runs (
                trace_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                final_output TEXT NOT NULL,
                success INTEGER NOT NULL,
                trace_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS evolution_candidates (
                candidate_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                baseline_json TEXT NOT NULL,
                source_trace_ids_json TEXT NOT NULL,
                proof_json TEXT NOT NULL DEFAULT '{}',
                rejection_reason TEXT NOT NULL DEFAULT '',
                rollback_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                promoted_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS evolution_rollbacks (
                rollback_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                rollback_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
        ),
    ),
)

CandidateKind = Literal["skill", "prompt_policy", "toolset_policy"]
CandidateStatus = Literal["staged", "promoted", "rejected", "rolled_back"]

VERIFIER_VERSION = "proof-carrying-evolution-v1"

_POLICY_FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(ignore|skip|bypass|disable|remove)\b.*\b(task[- ]?contract|contract)\b", re.I),
    re.compile(r"\b(ignore|skip|bypass|disable|remove)\b.*\bevidence\b", re.I),
    re.compile(r"\b(ignore|skip|bypass|disable|remove)\b.*\bapproval", re.I),
    re.compile(r"\b(ignore|skip|bypass|disable|remove)\b.*\bsandbox", re.I),
    re.compile(r"\b(no|never)\b.*\b(evidence|approval|task[- ]?contract|sandbox)\b", re.I),
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(data: Any) -> str:
    return json.dumps(data, sort_keys=True)


def _load_json(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _candidate_id(kind: str) -> str:
    return f"{kind}_{uuid.uuid4().hex[:12]}"


class EvolutionLedger:
    """SQLite ledger for traces, candidates, proofs, and rollbacks."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            apply_sqlite_migrations(conn, "evolution", _EVOLUTION_MIGRATIONS)

    def record_trajectory(self, trajectory: ExecutionTrajectory) -> str:
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        tool_results = [s for s in trajectory.steps if s.kind == "tool_result"]
        failed_tools = [
            s.metadata.get("tool_name", "unknown")
            for s in tool_results
            if s.metadata.get("is_error")
        ]
        successful_tools = [
            s.metadata.get("tool_name", "unknown")
            for s in tool_results
            if not s.metadata.get("is_error")
        ]
        evidence = {
            "successful_tools": successful_tools,
            "failed_tools": failed_tools,
            "tool_result_count": len(tool_results),
            "final_output_present": bool(trajectory.final_output.strip()),
        }
        trace = {
            "metadata": trajectory.metadata,
            "steps": [
                {
                    "kind": step.kind,
                    "content": step.content[:4000],
                    "metadata": step.metadata,
                }
                for step in trajectory.steps
            ],
        }
        success = bool(trajectory.final_output.strip()) and not trajectory.metadata.get(
            "hit_iteration_cap", False
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evolution_runs
                (trace_id, session_id, prompt, final_output, success, trace_json, evidence_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    str(trajectory.metadata.get("session_id") or "unknown"),
                    trajectory.prompt,
                    trajectory.final_output,
                    int(success),
                    _json(trace),
                    _json(evidence),
                    _now(),
                ),
            )
        return trace_id

    def create_candidate(
        self,
        *,
        kind: CandidateKind,
        name: str,
        payload: dict[str, Any],
        baseline: dict[str, Any] | None = None,
        source_trace_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        candidate_id = _candidate_id(kind)
        created_at = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evolution_candidates
                (candidate_id, kind, name, status, payload_json, baseline_json,
                 source_trace_ids_json, created_at, updated_at)
                VALUES (?, ?, ?, 'staged', ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    kind,
                    name,
                    _json(payload),
                    _json(baseline or {}),
                    _json(source_trace_ids or []),
                    created_at,
                    created_at,
                ),
            )
        candidate = self.get_candidate(candidate_id)
        assert candidate is not None
        return candidate

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evolution_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return self._row_to_candidate(row) if row else None

    def list_candidates(
        self,
        status: str | None = None,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM evolution_candidates"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(limit, 500)))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    def count_candidates(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM evolution_candidates GROUP BY status"
            ).fetchall()
        counts = {"staged": 0, "promoted": 0, "rejected": 0, "rolled_back": 0}
        counts.update({str(row["status"]): int(row["count"]) for row in rows})
        return counts

    def set_candidate_status(
        self,
        candidate_id: str,
        *,
        status: CandidateStatus,
        proof: dict[str, Any] | None = None,
        rejection_reason: str = "",
        rollback: dict[str, Any] | None = None,
    ) -> None:
        promoted_at = _now() if status == "promoted" else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE evolution_candidates
                SET status = ?, proof_json = ?, rejection_reason = ?,
                    rollback_json = ?, updated_at = ?, promoted_at = COALESCE(?, promoted_at)
                WHERE candidate_id = ?
                """,
                (
                    status,
                    _json(proof or {}),
                    rejection_reason,
                    _json(rollback or {}),
                    _now(),
                    promoted_at,
                    candidate_id,
                ),
            )

    def latest_promoted_policy(self, kind: Literal["prompt_policy", "toolset_policy"]) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM evolution_candidates
                WHERE kind = ? AND status = 'promoted'
                ORDER BY promoted_at DESC, updated_at DESC
                LIMIT 1
                """,
                (kind,),
            ).fetchone()
        return self._row_to_candidate(row) if row else None

    def record_rollback(self, candidate: dict[str, Any], rollback: dict[str, Any]) -> str:
        rollback_id = f"rollback_{uuid.uuid4().hex[:12]}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evolution_rollbacks
                (rollback_id, candidate_id, kind, name, rollback_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    rollback_id,
                    candidate["candidate_id"],
                    candidate["kind"],
                    candidate["name"],
                    _json(rollback),
                    _now(),
                ),
            )
        return rollback_id

    def _row_to_candidate(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "candidate_id": row["candidate_id"],
            "kind": row["kind"],
            "name": row["name"],
            "status": row["status"],
            "payload": _load_json(row["payload_json"], {}),
            "baseline": _load_json(row["baseline_json"], {}),
            "source_trace_ids": _load_json(row["source_trace_ids_json"], []),
            "proof": _load_json(row["proof_json"], {}),
            "rejection_reason": row["rejection_reason"],
            "rollback": _load_json(row["rollback_json"], {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "promoted_at": row["promoted_at"],
        }


class EvolutionEngine:
    """Stages, verifies, promotes, and rolls back candidate improvements."""

    def __init__(
        self,
        *,
        ledger_path: str | Path,
        skills_dir: str | Path,
        skill_registry: SkillRegistry,
        model: str | None = None,
        staging_dir: str | Path | None = None,
    ) -> None:
        self.ledger = EvolutionLedger(ledger_path)
        self._skills_dir = Path(skills_dir)
        self._skill_registry = skill_registry
        self._model = model
        self._staging_dir = Path(
            staging_dir
            or os.getenv("EVOLUTION_STAGING_DIR")
            or (self._skills_dir / ".evolution")
        )

    @property
    def staging_dir(self) -> Path:
        return self._staging_dir

    def status(self) -> dict[str, Any]:
        counts = self.ledger.count_candidates()
        return {
            "db_path": str(self.ledger.db_path),
            "staging_dir": str(self._staging_dir),
            "verifier_version": VERIFIER_VERSION,
            "candidates": counts,
            "active_policies": {
                "prompt_policy": bool(self.active_prompt_policy()),
                "toolset_policy": bool(self.active_toolset_policy()),
            },
        }

    def record_trajectory(self, trajectory: ExecutionTrajectory) -> str:
        trace_id = self.ledger.record_trajectory(trajectory)
        self._stage_policy_candidates_from_trace(trace_id, trajectory)
        return trace_id

    def stage_skill_candidate(
        self,
        *,
        name: str,
        code: str,
        source_trace_ids: list[str] | None = None,
        reason: str = "distilled",
    ) -> dict[str, Any]:
        slug = _slugify(name)
        baseline = self._skill_baseline(slug)
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        staged_path = self._staging_dir / f"{slug}_{uuid.uuid4().hex[:8]}.py"
        staged_path.write_text(code, encoding="utf-8")
        return self.ledger.create_candidate(
            kind="skill",
            name=slug,
            payload={
                "code": code,
                "code_hash": _sha256_text(code),
                "staged_path": str(staged_path),
                "reason": reason,
                "description": _extract_skill_description(code),
                "tags": _extract_skill_tags(code),
            },
            baseline=baseline,
            source_trace_ids=source_trace_ids or [],
        )

    def stage_policy_candidate(
        self,
        *,
        kind: Literal["prompt_policy", "toolset_policy"],
        name: str,
        content: str,
        source_trace_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.ledger.create_candidate(
            kind=kind,
            name=_slugify(name),
            payload={"content": content, "content_hash": _sha256_text(content)},
            baseline=self._policy_baseline(kind),
            source_trace_ids=source_trace_ids or [],
        )

    def _stage_policy_candidates_from_trace(
        self,
        trace_id: str,
        trajectory: ExecutionTrajectory,
    ) -> None:
        tool_results = [step for step in trajectory.steps if step.kind == "tool_result"]
        failed_tools = [
            str(step.metadata.get("tool_name") or "unknown")
            for step in tool_results
            if step.metadata.get("is_error")
        ]
        if trajectory.metadata.get("hit_iteration_cap"):
            self.stage_policy_candidate(
                kind="prompt_policy",
                name=f"iteration_recovery_{trace_id}",
                content=(
                    "For tasks that approach the iteration limit, update the plan, "
                    "separate completed work from remaining work, and verify current "
                    "filesystem/process/service state before summarising progress."
                ),
                source_trace_ids=[trace_id],
            )
        if failed_tools:
            unique = ", ".join(sorted(set(failed_tools))[:5])
            self.stage_policy_candidate(
                kind="toolset_policy",
                name=f"failure_diagnostics_{trace_id}",
                content=(
                    "For tasks with repeated tool failures, narrow to the smallest "
                    "relevant toolset, use read-only diagnostics first, then retry "
                    f"state-changing tools after inspecting recent failures: {unique}."
                ),
                source_trace_ids=[trace_id],
            )

    async def observe_skill_use(
        self,
        skill_name: str,
        *,
        success: bool,
        arguments: dict[str, Any],
        outcome: str,
    ) -> str:
        self._skill_registry.record_use(skill_name, success=success)
        self._skill_registry.record_use_example(
            skill_name,
            task=json.dumps(arguments, sort_keys=True)[:200],
            outcome=outcome[:200],
        )
        candidate = await self._build_skill_improvement_candidate(skill_name)
        return "staged" if candidate else "unchanged"

    async def _build_skill_improvement_candidate(self, skill_name: str) -> dict[str, Any] | None:
        if self._model is None:
            return None
        stats = self._skill_registry._stats.get(skill_name, {})
        use_count = stats.get("use_count", 0)
        last_improved_count = stats.get("improved_at_count", 0)
        failures = stats.get("failures_since_improve", 0)
        improve_after_uses = self._skill_registry._improve_after_uses
        matured = (
            use_count >= improve_after_uses
            and use_count - last_improved_count >= improve_after_uses
        )
        failing = failures >= self._skill_registry._improve_after_failures
        if not (matured or failing):
            return None
        skill_file = self._skills_dir / f"{skill_name}.py"
        if not skill_file.exists():
            return None
        original_code = skill_file.read_text(encoding="utf-8")
        usage_log = stats.get("usage_log", [])
        usage_text = "\n".join(
            f"- Task: {ex['task']} -> {ex['outcome']}" for ex in usage_log[-10:]
        ) or "No usage examples recorded."
        improved = await self._skill_registry._synthesize_improvement(
            original_code, usage_text, use_count
        )
        if not improved:
            return None
        return self.stage_skill_candidate(
            name=skill_name,
            code=improved,
            reason="usage_improvement",
        )

    def list_candidates(self, status: str | None = None) -> list[dict[str, Any]]:
        return self.ledger.list_candidates(status=status)

    def inspect_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        return self.ledger.get_candidate(candidate_id)

    def run_cycle(self, limit: int = 5) -> dict[str, Any]:
        staged = self.ledger.list_candidates(status="staged", limit=max(1, limit))
        results: list[dict[str, Any]] = []
        for candidate in staged:
            proof = self._verify_candidate(candidate)
            if proof["passed"]:
                rollback = self._promote_candidate(candidate, proof)
                self.ledger.set_candidate_status(
                    candidate["candidate_id"],
                    status="promoted",
                    proof=proof,
                    rollback=rollback,
                )
                results.append({
                    "candidate_id": candidate["candidate_id"],
                    "status": "promoted",
                    "score_delta": proof["score_delta"],
                })
            else:
                reason = "; ".join(
                    check["name"] for check in proof["checks"] if not check["passed"]
                ) or "verification failed"
                self.ledger.set_candidate_status(
                    candidate["candidate_id"],
                    status="rejected",
                    proof=proof,
                    rejection_reason=reason,
                )
                results.append({
                    "candidate_id": candidate["candidate_id"],
                    "status": "rejected",
                    "reason": reason,
                })
        return {"processed": len(results), "results": results}

    def rollback(self, candidate_id: str) -> dict[str, Any]:
        candidate = self.ledger.get_candidate(candidate_id)
        if candidate is None:
            return {"rolled_back": False, "error": "candidate not found"}
        if candidate["status"] != "promoted":
            return {"rolled_back": False, "error": "candidate is not promoted"}
        rollback_data = candidate.get("rollback") or {}
        if candidate["kind"] == "skill":
            result = self._rollback_skill(candidate, rollback_data)
        elif candidate["kind"] in {"prompt_policy", "toolset_policy"}:
            result = {"rolled_back": True, "kind": candidate["kind"]}
        else:
            return {"rolled_back": False, "error": f"unknown candidate kind {candidate['kind']}"}
        if result.get("rolled_back"):
            rollback_id = self.ledger.record_rollback(candidate, result)
            self.ledger.set_candidate_status(
                candidate_id,
                status="rolled_back",
                proof=candidate.get("proof") or {},
                rollback={**rollback_data, "rollback_id": rollback_id},
            )
            result["rollback_id"] = rollback_id
        return result

    def active_prompt_policy(self) -> str:
        candidate = self.ledger.latest_promoted_policy("prompt_policy")
        if not candidate:
            return ""
        return str(candidate["payload"].get("content") or "").strip()

    def active_toolset_policy(self) -> str:
        candidate = self.ledger.latest_promoted_policy("toolset_policy")
        if not candidate:
            return ""
        return str(candidate["payload"].get("content") or "").strip()

    def _verify_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        if candidate["kind"] == "skill":
            checks, candidate_score = self._verify_skill(candidate)
        elif candidate["kind"] in {"prompt_policy", "toolset_policy"}:
            checks, candidate_score = self._verify_policy(candidate)
        else:
            checks = [{"name": "known_candidate_kind", "passed": False, "detail": candidate["kind"]}]
            candidate_score = 0.0
        baseline_score = float(candidate["baseline"].get("score", 0.0))
        passed = all(check["passed"] for check in checks) and candidate_score >= baseline_score
        proof = {
            "source_trace_ids": candidate["source_trace_ids"],
            "baseline_artifact": candidate["baseline"],
            "candidate_artifact": {
                "kind": candidate["kind"],
                "name": candidate["name"],
                "hash": candidate["payload"].get("code_hash")
                or candidate["payload"].get("content_hash"),
            },
            "verifier_version": VERIFIER_VERSION,
            "checks": checks,
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "score_delta": round(candidate_score - baseline_score, 4),
            "rollback_target": candidate["baseline"],
            "passed": passed,
            "verified_at": _now(),
        }
        return proof

    def _verify_skill(self, candidate: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
        code = str(candidate["payload"].get("code") or "")
        baseline = candidate["baseline"]
        checks: list[dict[str, Any]] = []

        syntax_ok = _validate_python_syntax(code)
        checks.append({"name": "valid_python_syntax", "passed": syntax_ok})

        decorator_ok = "@skill" in code or "_is_skill" in code
        checks.append({"name": "has_skill_decorator", "passed": decorator_ok})

        fns = _skill_function_names(code)
        checks.append({"name": "defines_public_function", "passed": bool(fns), "detail": sorted(fns)})

        baseline_fns = set(baseline.get("functions") or [])
        preserves_api = not baseline_fns or bool(fns & baseline_fns)
        checks.append({
            "name": "preserves_public_api",
            "passed": preserves_api,
            "detail": {"baseline": sorted(baseline_fns), "candidate": sorted(fns)},
        })

        import_ok, import_detail = self._sandbox_import_skill(code, candidate["name"])
        checks.append({"name": "sandbox_import", "passed": import_ok, "detail": import_detail[:500]})

        reason = str(candidate["payload"].get("reason") or "")
        replay_ok = (
            bool(candidate["source_trace_ids"])
            or bool(baseline.get("exists"))
            or reason in {"auto_skill_maker", "manual"}
        )
        checks.append({
            "name": "evidence_replay_context",
            "passed": replay_ok,
            "detail": candidate["source_trace_ids"] or reason or "baseline skill improvement",
        })

        score = 1.0 if all(check["passed"] for check in checks) else 0.0
        return checks, score

    def _verify_policy(self, candidate: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
        content = str(candidate["payload"].get("content") or "").strip()
        forbidden = [
            pattern.pattern
            for pattern in _POLICY_FORBIDDEN_PATTERNS
            if pattern.search(content)
        ]
        checks = [
            {"name": "non_empty_policy", "passed": bool(content)},
            {"name": "bounded_policy_size", "passed": len(content) <= 4000},
            {"name": "does_not_weaken_contracts", "passed": not forbidden, "detail": forbidden},
            {
                "name": "additive_policy",
                "passed": not re.search(r"\b(replace|overwrite)\b.*\b(system|policy|contract)\b", content, re.I),
            },
        ]
        return checks, 1.0 if all(check["passed"] for check in checks) else 0.0

    def _promote_candidate(self, candidate: dict[str, Any], proof: dict[str, Any]) -> dict[str, Any]:
        if candidate["kind"] == "skill":
            return self._promote_skill(candidate, proof)
        if candidate["kind"] in {"prompt_policy", "toolset_policy"}:
            return {
                "kind": candidate["kind"],
                "previous": candidate["baseline"],
                "candidate_hash": proof["candidate_artifact"]["hash"],
            }
        raise ValueError(f"Unknown candidate kind {candidate['kind']!r}")

    def _promote_skill(self, candidate: dict[str, Any], proof: dict[str, Any]) -> dict[str, Any]:
        name = candidate["name"]
        code = str(candidate["payload"].get("code") or "")
        skill_file = self._skills_dir / f"{name}.py"
        baseline = candidate["baseline"]
        skill_file.parent.mkdir(parents=True, exist_ok=True)
        previous_code = skill_file.read_text(encoding="utf-8") if skill_file.exists() else ""
        if previous_code:
            version = int(baseline.get("version") or 1)
            self._skill_registry._archive_version(name, version, previous_code)
        skill_file.write_text(code, encoding="utf-8")

        stats = self._skill_registry._stats.setdefault(
            name,
            {
                "use_count": 0,
                "success_count": 0,
                "usage_log": [],
                "last_used": None,
                "created_by": "evolution",
                "created_at": _now(),
            },
        )
        previous_version = int(stats.get("version", baseline.get("version") or 0) or 0)
        stats["version"] = previous_version + 1 if baseline.get("exists") else 1
        stats["evolution_status"] = "promoted"
        stats["last_proof_id"] = candidate["candidate_id"]
        stats["promoted_at"] = _now()
        stats["version_hash"] = proof["candidate_artifact"]["hash"]
        stats["improved_at"] = stats["promoted_at"]
        stats["improved_at_count"] = stats.get("use_count", 0)
        stats["failures_since_improve"] = 0
        stats.setdefault("versions", {})[str(stats["version"])] = {"uses": 0, "successes": 0}
        self._skill_registry._save_stats()
        return {
            "kind": "skill",
            "path": str(skill_file),
            "previous_exists": bool(baseline.get("exists")),
            "previous_code": previous_code,
            "previous_hash": baseline.get("hash", ""),
            "previous_version": baseline.get("version", 0),
        }

    def _rollback_skill(self, candidate: dict[str, Any], rollback: dict[str, Any]) -> dict[str, Any]:
        name = candidate["name"]
        skill_file = self._skills_dir / f"{name}.py"
        if rollback.get("previous_exists"):
            skill_file.write_text(str(rollback.get("previous_code") or ""), encoding="utf-8")
        elif skill_file.exists():
            skill_file.unlink()
        stats = self._skill_registry._stats.setdefault(name, {})
        stats["version"] = int(rollback.get("previous_version") or max(1, stats.get("version", 1) - 1))
        stats["evolution_status"] = "rolled_back"
        stats["rollback_count"] = stats.get("rollback_count", 0) + 1
        stats["rolled_back_at"] = _now()
        if skill_file.exists():
            stats["version_hash"] = _sha256_text(skill_file.read_text(encoding="utf-8"))
        else:
            stats["version_hash"] = ""
        self._skill_registry._save_stats()
        return {"rolled_back": True, "kind": "skill", "path": str(skill_file)}

    def _skill_baseline(self, name: str) -> dict[str, Any]:
        skill_file = self._skills_dir / f"{name}.py"
        stats = self._skill_registry._stats.get(name, {})
        if not skill_file.exists():
            return {"exists": False, "score": 0.0, "version": 0, "hash": ""}
        code = skill_file.read_text(encoding="utf-8")
        return {
            "exists": True,
            "kind": "skill",
            "path": str(skill_file),
            "hash": _sha256_text(code),
            "version": stats.get("version", 1),
            "score": stats.get("success_rate", 1.0),
            "functions": sorted(_skill_function_names(code)),
        }

    def _policy_baseline(self, kind: Literal["prompt_policy", "toolset_policy"]) -> dict[str, Any]:
        active = self.ledger.latest_promoted_policy(kind)
        if not active:
            return {"exists": False, "score": 0.0, "hash": ""}
        content = str(active["payload"].get("content") or "")
        return {
            "exists": True,
            "kind": kind,
            "candidate_id": active["candidate_id"],
            "hash": _sha256_text(content),
            "score": 1.0,
        }

    def _sandbox_import_skill(self, code: str, name: str) -> tuple[bool, str]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            skill_bootstrap = self._skills_dir / "_skill.py"
            if skill_bootstrap.exists():
                shutil.copy2(skill_bootstrap, tmp_path / "_skill.py")
            else:
                (tmp_path / "_skill.py").write_text(
                    "def skill(fn=None, **kwargs):\n"
                    "    def deco(f):\n"
                    "        f._is_skill = True\n"
                    "        return f\n"
                    "    return deco(fn) if fn else deco\n",
                    encoding="utf-8",
                )
            module_path = tmp_path / f"{_slugify(name)}.py"
            module_path.write_text(code, encoding="utf-8")
            script = (
                "import importlib.util, pathlib\n"
                f"path = pathlib.Path({str(module_path)!r})\n"
                "spec = importlib.util.spec_from_file_location('candidate_skill', path)\n"
                "module = importlib.util.module_from_spec(spec)\n"
                "assert spec and spec.loader\n"
                "spec.loader.exec_module(module)\n"
            )
            try:
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=tmp,
                    text=True,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except Exception as exc:
                return False, str(exc)
            if result.returncode != 0:
                return False, (result.stderr or result.stdout or "").strip()
            return True, "imported"
