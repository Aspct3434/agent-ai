"""Skill distiller: turns successful agent trajectories into reusable skills.

An LLM synthesizes each trajectory into a parameterized Python function decorated
with @skill so it is automatically registered on the MCP skills server. Only
trajectories that pass the quality gate (at least two successful side-effecting
tool calls, non-trivial prompt) are submitted for synthesis.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import litellm

logger = logging.getLogger(__name__)

_MIN_SIDE_EFFECT_STEPS = 2

# Tools that produce observable side effects (filesystem, network, db).
# Single source of truth — planning.py imports this set rather than redeclaring.
_SIDE_EFFECT_TOOLS: frozenset[str] = frozenset(
    {
        "execute_terminal_command",
        "execute_background_service",
        "write_text_file",
        "expose_local_http_service",
        "create_table",
        "write_query",
    }
)


def _temperature_for_synthesis_model(model: str) -> float:
    """Return a provider-compatible temperature for background skill synthesis."""
    normalized = model.lower()
    if normalized.startswith("moonshot/") or normalized.startswith("kimi-"):
        return 1.0
    return 0.2

_SKILL_SYNTHESIS_PROMPT = """\
You are a skill synthesizer for an AI agent. An agent successfully completed this task:

TASK: {task}

SUCCESSFUL TOOL CALLS (in order):
{steps}

FINAL OUTPUT SUMMARY:
{output}

Generate a reusable Python skill using the @skill decorator. Requirements:
1. Accept parameters that generalize the task (e.g. `topic: str` not a hardcoded value)
2. Contain real executable Python -- use subprocess for shell ops, pathlib for files
3. Return a meaningful string describing what was accomplished
4. Have a clear, snake_case name and a one-sentence description

Exact format to use:
```python
from __future__ import annotations
from _skill import skill
import subprocess
from pathlib import Path

@skill(name="<snake_case_name>", description="<one sentence: what it does and its parameters>")
def <function_name>(<param1>: <type>, <param2>: <type> = <default>) -> str:
    \"\"\"<one-line docstring>\"\"\"
    # Real implementation using subprocess / pathlib / etc.
    return "<meaningful result string>"
```

Output ONLY the Python code block. Do NOT just return a string constant -- write logic that \
actually executes the task. If this task cannot be generalized into a reusable parameterized \
skill (e.g. it is a one-off query, a status check, or has no generalizable parameters), \
output exactly: NOT_DISTILLABLE
"""


@dataclass(frozen=True)
class ExecutionStep:
    """One action or observation from an agent execution."""

    kind: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionTrajectory:
    """Complete record of an AgentEngine run for later evaluation."""

    prompt: str
    steps: list[ExecutionStep]
    final_output: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillDistiller:
    """Background evaluator that synthesizes reusable skills from successful trajectories."""

    def __init__(
        self,
        max_queue_size: int = 0,
        skills_dir: str | Path = "skills",
        model: str | None = None,
        evolution_engine: Any = None,
    ) -> None:
        self._queue: asyncio.Queue[ExecutionTrajectory | None] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._skills_dir = Path(skills_dir)
        self._model = model
        self._evolution_engine = evolution_engine
        self._task: asyncio.Task[None] | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._task = asyncio.create_task(self._run(), name="skill-distiller")
        self._started = True

    async def submit(self, trajectory: ExecutionTrajectory) -> None:
        if not self._started:
            await self.start()
        await self._queue.put(trajectory)

    async def shutdown(self) -> None:
        if not self._started:
            return
        await self._queue.put(None)
        if self._task is not None:
            await self._task
        self._task = None
        self._started = False

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    async def _run(self) -> None:
        while True:
            trajectory = await self._queue.get()
            try:
                if trajectory is None:
                    return
                await self._distill(trajectory)
            except Exception:
                logger.exception("Unexpected error in skill distiller")
            finally:
                self._queue.task_done()

    async def _distill(self, trajectory: ExecutionTrajectory) -> None:
        if not trajectory.final_output.strip():
            return
        if _is_trivial_prompt(trajectory.prompt):
            return
        if not _has_meaningful_execution(trajectory.steps):
            logger.debug(
                "Skipping distillation: insufficient side-effect evidence for %r",
                trajectory.prompt[:60],
            )
            return
        if self._model is None:
            logger.debug("Skipping distillation: no synthesis model configured")
            return

        skill_code = await self._synthesize_skill_code(trajectory)
        if skill_code is None:
            return

        if self._evolution_engine is not None:
            self._stage_skill_candidate(trajectory, skill_code)
        else:
            await asyncio.to_thread(self._write_skill_file, trajectory, skill_code)

    async def _synthesize_skill_code(self, trajectory: ExecutionTrajectory) -> str | None:
        # Only reached after the caller verifies a synthesis model is configured.
        assert self._model is not None
        step_summary = _summarize_steps_for_prompt(trajectory.steps)
        prompt = _SKILL_SYNTHESIS_PROMPT.format(
            task=trajectory.prompt,
            steps=step_summary,
            output=trajectory.final_output[:400],
        )
        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=900,
                temperature=_temperature_for_synthesis_model(self._model),
            )
            raw = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("Skill synthesis LLM call failed: %s", exc)
            return None

        if "NOT_DISTILLABLE" in raw:
            logger.debug(
                "LLM determined prompt not distillable: %r", trajectory.prompt[:60]
            )
            return None

        code = _extract_python_block(raw)
        if not code:
            logger.warning(
                "No Python block in synthesis response for %r", trajectory.prompt[:60]
            )
            return None

        if not _validate_python_syntax(code):
            logger.warning(
                "Generated skill has invalid syntax for %r", trajectory.prompt[:60]
            )
            return None

        return code

    def _write_skill_file(self, trajectory: ExecutionTrajectory, skill_code: str) -> None:
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        stem, digest = self._skill_stem_and_digest(trajectory)
        path = self._skills_dir / f"{stem}_{digest}.py"
        path.write_text(skill_code, encoding="utf-8")
        logger.info("Distilled skill written: %s", path.name)

    def _stage_skill_candidate(self, trajectory: ExecutionTrajectory, skill_code: str) -> None:
        stem, digest = self._skill_stem_and_digest(trajectory)
        trace_id = str(trajectory.metadata.get("evolution_trace_id") or "")
        source_trace_ids = [trace_id] if trace_id else []
        candidate = self._evolution_engine.stage_skill_candidate(
            name=f"{stem}_{digest}",
            code=skill_code,
            source_trace_ids=source_trace_ids,
            reason="distilled_trajectory",
        )
        logger.info(
            "Distilled skill staged for proof-carrying evolution: %s",
            candidate.get("candidate_id"),
        )

    def _skill_stem_and_digest(self, trajectory: ExecutionTrajectory) -> tuple[str, str]:
        stem = _slugify(trajectory.prompt)
        digest = hashlib.sha256(
            trajectory.prompt.strip().lower().encode("utf-8")
        ).hexdigest()[:8]
        return stem, digest


# ---------------------------------------------------------------------------
# Quality-gate helpers
# ---------------------------------------------------------------------------

def _has_meaningful_execution(steps: list[ExecutionStep]) -> bool:
    """True when the trajectory has at least _MIN_SIDE_EFFECT_STEPS successful side effects."""
    count = sum(
        1
        for step in steps
        if (
            step.kind == "tool_result"
            and not step.metadata.get("is_error")
            and step.metadata.get("tool_name") in _SIDE_EFFECT_TOOLS
        )
    )
    return count >= _MIN_SIDE_EFFECT_STEPS


def _summarize_steps_for_prompt(steps: list[ExecutionStep]) -> str:
    """Summarize the successful tool calls for the LLM synthesis prompt."""
    lines: list[str] = []
    for step in steps:
        if step.kind != "tool_result" or step.metadata.get("is_error"):
            continue
        tool_name = step.metadata.get("tool_name", "unknown")
        args = step.metadata.get("arguments") or {}
        cmd = args.get("command", "")
        if cmd:
            lines.append(f"- {tool_name}: {cmd[:100]}")
        elif args:
            compact = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:2])
            lines.append(f"- {tool_name}({compact[:80]})")
        else:
            lines.append(f"- {tool_name}")
        if len(lines) >= 12:
            lines.append("  … (more steps omitted)")
            break
    return "\n".join(lines) or "No significant tool calls recorded."


def _extract_python_block(text: str) -> str:
    """Extract the first ```python ... ``` code block from the LLM response."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    if stripped.startswith(("from ", "import ", "@skill", "def ")):
        return stripped
    return ""


def _validate_python_syntax(code: str) -> bool:
    """Return True only when the code compiles without syntax errors."""
    try:
        compile(code, "<skill>", "exec")
        return True
    except SyntaxError:
        return False


def _skill_function_names(code: str) -> set[str]:
    """Return the names of top-level functions defined in *code*.

    Used to guarantee a self-improvement candidate preserves the skill's
    public function(s), so callers and the MCP server don't break.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


_TRIVIAL_PROMPTS: frozenset[str] = frozenset(
    {
        "hi", "hello", "hey", "yo", "sup", "hiya",
        "yes", "y", "no", "n", "ok", "okay", "k",
        "go", "continue", "proceed", "do it", "run it", "carry on",
        "thanks", "thank you", "done", "cool", "nice",
    }
)


def _is_trivial_prompt(prompt: str) -> bool:
    normalized = prompt.strip().lower().strip(".!?")
    if not normalized or normalized in _TRIVIAL_PROMPTS:
        return True
    return len(normalized.split()) < 3


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug[:64] or "distilled_skill"


# ---------------------------------------------------------------------------
# Skill registry: usage tracking + self-improvement
# ---------------------------------------------------------------------------

_SKILL_IMPROVEMENT_PROMPT = """\
You are improving an existing AI agent skill. The skill has been used {use_count} times.

ORIGINAL SKILL CODE:
```python
{original_code}
```

RECENT USAGE EXAMPLES (task → outcome):
{usage_examples}

Improve the skill to:
1. Handle edge cases observed in usage
2. Be more robust and parameterized
3. Return richer, more informative results
4. Fix any issues seen in failed runs

Output ONLY the improved Python code block (same @skill format). If the original
is already optimal, output it unchanged. Do NOT output NOT_DISTILLABLE.
"""

_PROFILE_EXTRACTION_PROMPT = """\
Extract user information from this agent conversation. Return a JSON object with
these fields (omit fields you cannot infer, do not guess):

{
  "name": "user's name if mentioned",
  "expertise": ["domain1", "domain2"],
  "communication_style": "one of: technical/casual/formal/brief",
  "preferences": ["prefers X", "dislikes Y"],
  "recurring_topics": ["topic1", "topic2"],
  "goals": ["short goal description"]
}

CONVERSATION:
{conversation}

Return ONLY the JSON object. If no meaningful info can be extracted, return {{}}.
"""


class SkillRegistry:
    """Tracks skill usage and triggers self-improvement when skills mature.

    Usage stats are stored in a JSON sidecar file alongside each skill.
    After ``improve_after_uses`` successful invocations, the skill code is
    re-synthesized with usage context to produce a better version.
    """

    def __init__(
        self,
        skills_dir: str | Path,
        model: str | None = None,
        improve_after_uses: int = 5,
    ) -> None:
        self._skills_dir = Path(skills_dir)
        self._model = model
        self._improve_after_uses = improve_after_uses
        # Evidence-gated evolution thresholds (unique to agent-ai):
        # also improve after repeated failures, and roll a promoted version
        # back if it measurably regresses against its predecessor.
        self._improve_after_failures = int(os.getenv("SKILL_IMPROVE_AFTER_FAILURES", "2"))
        self._rollback_min_samples = int(os.getenv("SKILL_ROLLBACK_MIN_SAMPLES", "4"))
        self._rollback_margin = float(os.getenv("SKILL_ROLLBACK_MARGIN", "0.15"))
        self._stats_file = self._skills_dir / "_registry.json"
        self._stats: dict[str, Any] = self._load_stats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_skills(self) -> list[dict[str, Any]]:
        """Return metadata for all installed skills."""
        skills: list[dict[str, Any]] = []
        for py_file in sorted(self._skills_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            code = ""
            try:
                code = py_file.read_text(encoding="utf-8")
            except OSError:
                continue
            name = py_file.stem
            stats = self._stats.get(name, {})
            version_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
            skills.append({
                "name": name,
                "file": py_file.name,
                "description": _extract_skill_description(code),
                "tags": _extract_skill_tags(code),
                "use_count": stats.get("use_count", 0),
                "success_rate": stats.get("success_rate", 1.0),
                "version": stats.get("version", 1),
                "last_used": stats.get("last_used"),
                "improved_at": stats.get("improved_at"),
                "evolution_status": stats.get("evolution_status", "live"),
                "version_hash": stats.get("version_hash", version_hash),
                "last_proof_id": stats.get("last_proof_id"),
                "promoted_at": stats.get("promoted_at"),
                "rollback_count": stats.get("rollback_count", 0),
            })
        return skills

    def record_use(self, skill_name: str, *, success: bool) -> None:
        """Record one invocation of *skill_name* (overall + per-version)."""
        s = self._stats.setdefault(skill_name, {
            "use_count": 0, "success_count": 0, "version": 1,
            "usage_log": [], "last_used": None, "improved_at": None,
        })
        s["use_count"] += 1
        if success:
            s["success_count"] += 1
        else:
            s["failures_since_improve"] = s.get("failures_since_improve", 0) + 1
        s["last_used"] = datetime.now(UTC).isoformat()
        total = s["use_count"]
        s["success_rate"] = s["success_count"] / total if total else 1.0
        # Per-version metrics power the evidence-gated rollback decision.
        version = str(s.get("version", 1))
        vstats = s.setdefault("versions", {}).setdefault(version, {"uses": 0, "successes": 0})
        vstats["uses"] += 1
        if success:
            vstats["successes"] += 1
        self._save_stats()

    def record_use_example(self, skill_name: str, task: str, outcome: str) -> None:
        s = self._stats.setdefault(skill_name, {"usage_log": []})
        log: list[dict[str, str]] = s.setdefault("usage_log", [])
        log.append({"task": task[:200], "outcome": outcome[:200]})
        if len(log) > 20:
            s["usage_log"] = log[-20:]
        self._save_stats()

    async def maybe_evolve(self, skill_name: str) -> str:
        """Evidence-gated evolution step. Returns the action taken.

        This is the public trigger the engine calls after a skill runs. Unlike
        Hermes (which overwrites a skill on use), agent-ai:
          1. rolls back first if the current version has measurably regressed
             against its predecessor (auto-repair of a bad improvement), then
          2. synthesises an improvement — but only *promotes* it if it passes a
             validation gate (valid Python, keeps @skill, preserves the public
             function), archiving the prior version so a regression is reversible.

        Returns one of: ``"rolled_back"``, ``"improved"``, ``"unchanged"``.
        """
        if self.maybe_rollback(skill_name):
            return "rolled_back"
        if await self.maybe_improve(skill_name):
            return "improved"
        return "unchanged"

    async def maybe_improve(self, skill_name: str) -> bool:
        """Synthesise + evidence-gate a new skill version. Returns True if promoted.

        Triggers when the skill matures (``improve_after_uses`` new uses) OR has
        accumulated repeated failures since the last improvement. A candidate is
        promoted only if it passes :meth:`_validate_candidate`; the prior version
        is archived under ``.versions/`` for rollback.
        """
        if self._model is None:
            return False
        s = self._stats.get(skill_name, {})
        use_count = s.get("use_count", 0)
        last_improved_count = s.get("improved_at_count", 0)
        failures = s.get("failures_since_improve", 0)

        matured = (
            use_count >= self._improve_after_uses
            and use_count - last_improved_count >= self._improve_after_uses
        )
        failing = failures >= self._improve_after_failures
        if not (matured or failing):
            return False

        skill_file = self._skills_dir / f"{skill_name}.py"
        if not skill_file.exists():
            return False

        original_code = skill_file.read_text(encoding="utf-8")
        usage_log = s.get("usage_log", [])
        usage_text = "\n".join(
            f"- Task: {ex['task']} → {ex['outcome']}" for ex in usage_log[-10:]
        ) or "No usage examples recorded."

        improved = await self._synthesize_improvement(
            original_code, usage_text, use_count
        )
        if not improved:
            return False

        ok, reason = self._validate_candidate(original_code, improved)
        if not ok:
            logger.info("Rejected self-improvement for %s: %s", skill_name, reason)
            return False

        version = s.get("version", 1)
        self._archive_version(skill_name, version, original_code)
        try:
            skill_file.write_text(improved, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write improved skill %s: %s", skill_name, exc)
            return False

        new_version = version + 1
        s["version"] = new_version
        s["improved_at"] = datetime.now(UTC).isoformat()
        s["improved_at_count"] = use_count
        s["failures_since_improve"] = 0
        s.setdefault("versions", {})[str(new_version)] = {"uses": 0, "successes": 0}
        self._save_stats()
        logger.info("Skill %s improved to version %d (gate passed)", skill_name, new_version)
        return True

    def maybe_rollback(self, skill_name: str) -> bool:
        """Roll back to the previous version if the current one regressed.

        Evidence-based: only fires once the current version has at least
        ``rollback_min_samples`` uses and its success rate is more than
        ``rollback_margin`` below the predecessor's. Restores the archived
        previous version. Returns True if a rollback happened.
        """
        s = self._stats.get(skill_name, {})
        version = s.get("version", 1)
        if version < 2:
            return False
        versions = s.get("versions", {})
        cur = versions.get(str(version), {})
        prev = versions.get(str(version - 1), {})
        cur_uses = cur.get("uses", 0)
        prev_uses = prev.get("uses", 0)
        if cur_uses < self._rollback_min_samples or prev_uses == 0:
            return False
        cur_rate = cur.get("successes", 0) / cur_uses
        prev_rate = prev.get("successes", 0) / prev_uses
        if cur_rate + self._rollback_margin >= prev_rate:
            return False

        archive = self._versions_dir(skill_name) / f"v{version - 1}.py"
        if not archive.exists():
            return False
        skill_file = self._skills_dir / f"{skill_name}.py"
        try:
            skill_file.write_text(archive.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not roll back skill %s: %s", skill_name, exc)
            return False

        s["version"] = version - 1
        s["rollback_count"] = s.get("rollback_count", 0) + 1
        s["rolled_back_at"] = datetime.now(UTC).isoformat()
        s["failures_since_improve"] = 0
        self._save_stats()
        logger.info(
            "Rolled back skill %s: v%d (%.0f%%) regressed below v%d (%.0f%%)",
            skill_name, version, cur_rate * 100, version - 1, prev_rate * 100,
        )
        return True

    def create_skill(
        self,
        name: str,
        code: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """Author a new skill on demand (the auto skill maker).

        Validates that *code* is syntactically valid, uses the ``@skill``
        decorator, and defines at least one function, then writes it to the
        skills directory and registers initial stats. Returns the file path.
        """
        slug = _slugify(name)
        if not _validate_python_syntax(code):
            raise ValueError("Skill code has invalid Python syntax")
        if "@skill" not in code:
            raise ValueError("Skill code must decorate a function with @skill")
        if not _skill_function_names(code):
            raise ValueError("Skill code defines no function")
        dest = self._skills_dir / f"{slug}.py"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(code, encoding="utf-8")
        self._stats.setdefault(slug, {
            "use_count": 0, "success_count": 0, "version": 1,
            "usage_log": [], "last_used": None, "improved_at": None,
            "created_by": "agent", "created_at": datetime.now(UTC).isoformat(),
        })
        self._save_stats()
        logger.info("Auto skill maker created skill %s (%s)", slug, description[:60])
        return str(dest)

    # ------------------------------------------------------------------
    # Versioning helpers
    # ------------------------------------------------------------------

    def _versions_dir(self, skill_name: str) -> Path:
        return self._skills_dir / ".versions" / skill_name

    def _archive_version(self, skill_name: str, version: int, code: str) -> None:
        """Snapshot a skill's *version* code so a later version can be rolled back."""
        vdir = self._versions_dir(skill_name)
        vdir.mkdir(parents=True, exist_ok=True)
        try:
            (vdir / f"v{version}.py").write_text(code, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not archive %s v%d: %s", skill_name, version, exc)

    def _validate_candidate(self, original: str, candidate: str) -> tuple[bool, str]:
        """Evidence gate: a candidate must be valid and preserve the skill's API."""
        if not candidate.strip():
            return False, "empty candidate"
        if not _validate_python_syntax(candidate):
            return False, "invalid Python syntax"
        if "@skill" not in candidate and "_is_skill" not in candidate:
            return False, "candidate dropped the @skill decorator"
        original_fns = _skill_function_names(original)
        candidate_fns = _skill_function_names(candidate)
        if original_fns and not (original_fns & candidate_fns):
            return False, f"candidate dropped the public function(s) {sorted(original_fns)}"
        return True, "ok"

    def export_skill(self, skill_name: str) -> dict[str, Any] | None:
        """Return a portable dict for sharing/importing this skill."""
        skill_file = self._skills_dir / f"{skill_name}.py"
        if not skill_file.exists():
            return None
        code = skill_file.read_text(encoding="utf-8")
        stats = self._stats.get(skill_name, {})
        return {
            "name": skill_name,
            "description": _extract_skill_description(code),
            "tags": _extract_skill_tags(code),
            "code": code,
            "version": stats.get("version", 1),
            "use_count": stats.get("use_count", 0),
            "success_rate": stats.get("success_rate", 1.0),
            "exported_at": datetime.now(UTC).isoformat(),
        }

    def import_skill(self, payload: dict[str, Any]) -> str:
        """Write an imported skill to the skills directory.

        Returns the file path written.
        """
        name = _slugify(payload.get("name", "imported_skill"))
        code = payload.get("code", "")
        if not code:
            raise ValueError("Imported skill has no code")
        if not _validate_python_syntax(code):
            raise ValueError("Imported skill has invalid Python syntax")
        dest = self._skills_dir / f"{name}.py"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(code, encoding="utf-8")
        logger.info("Imported skill: %s", dest.name)
        return str(dest)

    # ------------------------------------------------------------------
    # agentskills.io (SKILL.md) interop
    # ------------------------------------------------------------------

    def export_skill_md(self, skill_name: str) -> str | None:
        """Render a skill as an agentskills.io-compatible SKILL.md document."""
        data = self.export_skill(skill_name)
        if data is None:
            return None
        from skill_standard import to_skill_md

        return to_skill_md(
            name=data["name"],
            description=data["description"],
            code=data["code"],
            tags=data.get("tags") or [],
        )

    def import_skill_md(self, text: str) -> str:
        """Import a skill from an agentskills.io SKILL.md document.

        The standard supports instruction-only skills, but agent-ai skills are
        executable Python, so a fenced ``python`` block is required.
        """
        from skill_standard import parse_skill_md

        parsed = parse_skill_md(text)
        if not parsed["code"]:
            raise ValueError(
                "SKILL.md has no ```python``` block; agent-ai skills must be executable."
            )
        return self.import_skill(
            {
                "name": parsed["name"] or "imported_skill",
                "description": parsed["description"],
                "code": parsed["code"],
                "tags": parsed["metadata"].get("tags", []),
            }
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _synthesize_improvement(
        self, original_code: str, usage_examples: str, use_count: int
    ) -> str | None:
        assert self._model is not None
        prompt = _SKILL_IMPROVEMENT_PROMPT.format(
            use_count=use_count,
            original_code=original_code[:2000],
            usage_examples=usage_examples,
        )
        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=_temperature_for_synthesis_model(self._model),
            )
            raw = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("Skill improvement LLM call failed: %s", exc)
            return None
        code = _extract_python_block(raw)
        if not code or not _validate_python_syntax(code):
            return None
        return code

    def _load_stats(self) -> dict[str, Any]:
        try:
            if self._stats_file.exists():
                data = json.loads(self._stats_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _save_stats(self) -> None:
        try:
            self._stats_file.parent.mkdir(parents=True, exist_ok=True)
            self._stats_file.write_text(
                json.dumps(self._stats, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Could not save skill registry stats: %s", exc)


def _extract_skill_description(code: str) -> str:
    m = re.search(r'description\s*=\s*["\']([^"\']+)["\']', code)
    return m.group(1) if m else ""


def _extract_skill_tags(code: str) -> list[str]:
    """Heuristically tag a skill by keywords in its code."""
    tags: list[str] = []
    lower = code.lower()
    tag_map = {
        "web": ["requests", "httpx", "http", "url", "html", "scrape"],
        "shell": ["subprocess", "execute", "terminal", "bash", "shell"],
        "files": ["pathlib", "open(", "write_text", "read_text"],
        "data": ["pandas", "csv", "json", "sqlite", "database"],
        "git": ["git clone", "git commit", "git push"],
        "docker": ["docker run", "docker build"],
        "python": ["pip install", "import ", "python"],
    }
    for tag, keywords in tag_map.items():
        if any(kw in lower for kw in keywords):
            tags.append(tag)
    return tags


# ---------------------------------------------------------------------------
# User profile extraction
# ---------------------------------------------------------------------------


async def extract_and_update_user_profile(
    profile_store: Any,
    conversation_text: str,
    model: str,
) -> None:
    """Extract user info from conversation text and merge into profile_store.

    Runs as a fire-and-forget coroutine after each completed task.
    """
    if not conversation_text.strip() or not model:
        return
    prompt = _PROFILE_EXTRACTION_PROMPT.format(conversation=conversation_text[:3000])
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.1,
        )
        raw = (response.choices[0].message.content or "").strip()
        # Strip markdown code blocks if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        updates = json.loads(raw)
        if isinstance(updates, dict) and updates:
            profile_store.update(updates)
            logger.debug("User profile updated from conversation")
    except Exception as exc:
        logger.debug("Profile extraction skipped: %s", exc)
