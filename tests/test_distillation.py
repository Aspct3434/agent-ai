from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import evaluator as evaluator_module
from evaluator import ExecutionStep, ExecutionTrajectory, SkillDistiller


def _trajectory() -> ExecutionTrajectory:
    return ExecutionTrajectory(
        prompt="Create and publish a reusable static site for a topic",
        steps=[
            ExecutionStep(
                kind="tool_result",
                content=json.dumps({"exit_code": 0, "stdout": "created"}),
                metadata={
                    "tool_name": "execute_terminal_command",
                    "is_error": False,
                    "arguments": {"command": "create site files"},
                },
            ),
            ExecutionStep(
                kind="tool_result",
                content=json.dumps(
                    {
                        "published": True,
                        "index_exists": True,
                        "url": "http://localhost:8000/sites/example/",
                    }
                ),
                metadata={
                    "tool_name": "publish_static_site",
                    "is_error": False,
                    "arguments": {"source_path": "/tmp/site", "slug": "example"},
                },
            ),
        ],
        final_output="Published http://localhost:8000/sites/example/",
    )


async def _distill_without_model() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        distiller = SkillDistiller(skills_dir=tmp)
        await distiller._distill(_trajectory())
        created = [
            path.name
            for path in Path(tmp).glob("*.py")
            if path.name not in {"server.py", "_skill.py"}
        ]
        assert created == []


def test_skill_distiller_without_model_creates_no_skill() -> None:
    asyncio.run(_distill_without_model())


async def _fake_synthesis_completion(**kwargs: Any) -> Any:
    code = """```python
from __future__ import annotations
from _skill import skill
from pathlib import Path

@skill(name="write_topic_site", description="Create a tiny static site for a topic.")
def write_topic_site(topic: str, output_dir: str = "/tmp/topic-site") -> str:
    \"\"\"Create an index.html file for a topic site.\"\"\"
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "index.html").write_text(f"<h1>{topic}</h1>", encoding="utf-8")
    return f"Created topic site for {topic} at {path}"
```"""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=code))]
    )


async def _distill_with_model() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        original_completion = evaluator_module.litellm.acompletion
        evaluator_module.litellm.acompletion = _fake_synthesis_completion
        try:
            trajectory = _trajectory()
            distiller = SkillDistiller(skills_dir=tmp, model="test-model")
            await distiller._distill(trajectory)
        finally:
            evaluator_module.litellm.acompletion = original_completion

        created = [
            path
            for path in Path(tmp).glob("*.py")
            if path.name not in {"server.py", "_skill.py"}
        ]
        assert len(created) == 1
        source = created[0].read_text(encoding="utf-8")
        assert "@skill" in source
        assert "def write_topic_site(topic: str" in source
        assert "Path(output_dir)" in source
        assert trajectory.final_output not in source


def test_skill_distiller_writes_parameterized_skill_from_model() -> None:
    asyncio.run(_distill_with_model())


if __name__ == "__main__":
    test_skill_distiller_without_model_creates_no_skill()
    test_skill_distiller_writes_parameterized_skill_from_model()
