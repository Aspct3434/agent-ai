"""Skill distiller: turns successful agent trajectories into reusable skills.

An LLM synthesizes each trajectory into a parameterized Python function decorated
with @skill so it is automatically registered on the MCP skills server. Only
trajectories that pass the quality gate (at least two successful side-effecting
tool calls, non-trivial prompt) are submitted for synthesis.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
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
        "publish_static_site",
        "expose_local_http_service",
        "create_table",
        "write_query",
    }
)

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
    ) -> None:
        self._queue: asyncio.Queue[ExecutionTrajectory | None] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._skills_dir = Path(skills_dir)
        self._model = model
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

        await asyncio.to_thread(self._write_skill_file, trajectory, skill_code)

    async def _synthesize_skill_code(self, trajectory: ExecutionTrajectory) -> str | None:
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
                temperature=0.2,
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
        stem = _slugify(trajectory.prompt)
        digest = hashlib.sha256(
            trajectory.prompt.strip().lower().encode("utf-8")
        ).hexdigest()[:8]
        path = self._skills_dir / f"{stem}_{digest}.py"
        path.write_text(skill_code, encoding="utf-8")
        logger.info("Distilled skill written: %s", path.name)


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
