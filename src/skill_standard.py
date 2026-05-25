"""agentskills.io / Agent Skills (SKILL.md) interoperability.

A skill in the open `agentskills.io <https://agentskills.io>`_ standard is a
directory containing a ``SKILL.md`` file: YAML frontmatter (``name``,
``description``, optional ``license``/``version``/``tags``) followed by a
Markdown body of instructions, plus optional bundled scripts.

agent-ai's own skills are ``@skill``-decorated Python functions. This module
converts between the two so skills can be published to / imported from the
Skills Hub: the standard frontmatter carries discovery metadata, and the
executable Python rides along in a fenced ``python`` code block.
"""
from __future__ import annotations

import re
from typing import Any

import yaml

# Frontmatter: a leading "---\n … \n---" block, then the body.
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
# First fenced python code block in the body.
_PY_BLOCK_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL)


def parse_skill_md(text: str) -> dict[str, Any]:
    """Parse a SKILL.md document into its parts.

    Returns ``{name, description, metadata, body, code}``. ``code`` is the
    contents of the first fenced ``python`` block (empty string if none).
    """
    stripped = text.lstrip("﻿")
    match = _FRONTMATTER_RE.match(stripped)
    if match:
        loaded = yaml.safe_load(match.group(1))
        metadata: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
        body = match.group(2).strip()
    else:
        metadata, body = {}, stripped.strip()

    code_match = _PY_BLOCK_RE.search(body)
    code = code_match.group(1).strip() if code_match else ""

    return {
        "name": str(metadata.get("name", "")).strip(),
        "description": str(metadata.get("description", "")).strip(),
        "metadata": metadata,
        "body": body,
        "code": code,
    }


def to_skill_md(
    name: str,
    description: str,
    code: str = "",
    tags: list[str] | None = None,
    *,
    instructions: str = "",
) -> str:
    """Render an agent-ai skill as an agentskills.io-compatible SKILL.md."""
    frontmatter: dict[str, Any] = {"name": name, "description": description}
    if tags:
        frontmatter["tags"] = list(tags)
    front = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()

    parts: list[str] = [f"---\n{front}\n---", "", f"# {name}", "", description or ""]
    if instructions:
        parts += ["", instructions.strip()]
    if code:
        parts += ["", "## Implementation", "", "```python", code.strip(), "```"]
    return "\n".join(parts).strip() + "\n"
