"""Tests for agentskills.io (SKILL.md) interop."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skill_standard import parse_skill_md, to_skill_md

_SKILL_CODE = 'from _skill import skill\n\n\n@skill\ndef greet(name: str) -> str:\n    "Greet."\n    return f"Hi {name}"\n'


class TestParseSkillMd:
    def test_frontmatter_and_code(self) -> None:
        text = (
            "---\n"
            "name: greet\n"
            "description: Greets a person\n"
            "tags: [demo, hello]\n"
            "---\n"
            "# greet\n\nGreets a person.\n\n"
            "```python\nprint('hi')\n```\n"
        )
        parsed = parse_skill_md(text)
        assert parsed["name"] == "greet"
        assert parsed["description"] == "Greets a person"
        assert parsed["metadata"]["tags"] == ["demo", "hello"]
        assert parsed["code"] == "print('hi')"

    def test_no_frontmatter(self) -> None:
        parsed = parse_skill_md("just some text, no frontmatter")
        assert parsed["name"] == ""
        assert parsed["metadata"] == {}
        assert parsed["body"] == "just some text, no frontmatter"

    def test_no_code_block(self) -> None:
        parsed = parse_skill_md("---\nname: x\n---\nNo code here.")
        assert parsed["code"] == ""

    def test_py_fence_alias(self) -> None:
        parsed = parse_skill_md("```py\nx = 1\n```")
        assert parsed["code"] == "x = 1"


class TestToSkillMd:
    def test_contains_frontmatter_and_code(self) -> None:
        md = to_skill_md("greet", "Greets a person", code="print('hi')", tags=["demo"])
        assert md.startswith("---\n")
        assert "name: greet" in md
        assert "description: Greets a person" in md
        assert "```python" in md
        assert "print('hi')" in md

    def test_roundtrip_preserves_core_fields(self) -> None:
        md = to_skill_md("mytool", "Does a thing", code=_SKILL_CODE, tags=["a", "b"])
        parsed = parse_skill_md(md)
        assert parsed["name"] == "mytool"
        assert parsed["description"] == "Does a thing"
        assert parsed["metadata"]["tags"] == ["a", "b"]
        assert "def greet" in parsed["code"]


class TestRegistryRoundTrip:
    def test_export_then_import_md(self, tmp_path) -> None:
        from evaluator import SkillRegistry

        src_dir = tmp_path / "src_skills"
        src_dir.mkdir()
        (src_dir / "greet.py").write_text(_SKILL_CODE, encoding="utf-8")

        src_reg = SkillRegistry(skills_dir=src_dir, model="x", improve_after_uses=5)
        md = src_reg.export_skill_md("greet")
        assert md is not None
        assert "name: greet" in md
        assert "def greet" in md

        dest_dir = tmp_path / "dest_skills"
        dest_reg = SkillRegistry(skills_dir=dest_dir, model="x", improve_after_uses=5)
        path = dest_reg.import_skill_md(md)
        assert Path(path).exists()
        assert "def greet" in Path(path).read_text(encoding="utf-8")

    def test_import_md_without_code_rejected(self, tmp_path) -> None:
        from evaluator import SkillRegistry

        reg = SkillRegistry(skills_dir=tmp_path, model="x", improve_after_uses=5)
        with pytest.raises(ValueError):
            reg.import_skill_md("---\nname: noop\n---\nJust instructions, no code.")
