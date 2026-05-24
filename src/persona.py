"""Persona loader: reads SOUL.md, AGENTS.md, and TOOLS.md from a workspace.

Directory layout (default: $PROJECT_ROOT/persona/):
    SOUL.md         — Core personality, tone, and values
    AGENTS.md       — Sub-agent role definitions
    TOOLS.md        — Tool usage guidelines
    <name>/         — Named persona sub-directories (switch at runtime)

All present files are combined and prepended to the agent's SYSTEM_DIRECTIVE.
Supports runtime persona switching and per-session persona overrides.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PERSONA_FILES = ("SOUL.md", "AGENTS.md", "TOOLS.md")


class PersonaLoader:
    """Loads agent persona definition files and injects them into the system prompt."""

    def __init__(self, persona_dir: str | Path | None = None) -> None:
        self._dir: Path | None = Path(persona_dir) if persona_dir else None
        self._active_persona: str | None = None
        self._content: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, persona_name: str | None = None) -> str:
        """Build and return the persona prompt injection string.

        Named sub-directory files take priority over root-level files.
        Returns empty string when no persona directory is configured.
        """
        if self._dir is None or not self._dir.exists():
            return ""

        search_dirs: list[Path] = []
        if persona_name:
            named = self._dir / persona_name
            if named.exists():
                search_dirs.append(named)
        search_dirs.append(self._dir)

        parts: list[str] = []
        for fname in _PERSONA_FILES:
            for d in search_dirs:
                fpath = d / fname
                if fpath.exists():
                    try:
                        text = fpath.read_text(encoding="utf-8").strip()
                        if text:
                            section = fname[:-3]  # strip .md
                            parts.append(f"## {section}\n{text}")
                    except OSError as exc:
                        logger.warning("Could not read %s: %s", fpath, exc)
                    break  # first match wins

        combined = "\n\n".join(parts)
        self._active_persona = persona_name
        self._content = combined
        if combined:
            logger.info(
                "Persona loaded: %s (%d chars)",
                persona_name or "default",
                len(combined),
            )
        return combined

    def reload(self) -> str:
        """Re-read files from disk with the currently active persona."""
        return self.load(self._active_persona)

    def list_personas(self) -> list[str]:
        """Return the names of all named persona sub-directories."""
        if self._dir is None or not self._dir.exists():
            return []
        return sorted(
            p.name
            for p in self._dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    @property
    def active_persona(self) -> str | None:
        return self._active_persona

    @property
    def content(self) -> str:
        return self._content

    def describe(self) -> dict[str, Any]:
        return {
            "persona_dir": str(self._dir) if self._dir else None,
            "active_persona": self._active_persona,
            "available_personas": self.list_personas(),
            "content_length": len(self._content),
            "loaded": bool(self._content),
        }
