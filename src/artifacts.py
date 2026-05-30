"""Per-session artifact ledger.

Records the concrete things the agent produces -- files written, services
served, processes started -- as typed records the moment they happen, so a later
"give me the url again" / "show me the config file" can be answered by querying
structured state instead of re-scanning the conversation transcript.

Querying typed, recency-ordered records is both more reliable than pattern
matching a transcript (an artifact from an unrelated earlier turn can no longer
leak into the current answer) and cheaper (the model's context no longer has to
retain full tool outputs just to recall what was produced).

In-memory and per session, mirroring the engine's conversation histories.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Artifact:
    """One concrete thing the agent produced."""

    label: str  # evidence-kind label, e.g. "Service URL", "File", "Port"
    value: str  # the concrete value, e.g. the URL or absolute path
    turn: int  # 1-based user-turn index in which it was produced
    tool_name: str


class ArtifactLedger:
    """Deduplicated, recency-ordered store of artifacts for a single session."""

    def __init__(self) -> None:
        self._items: list[Artifact] = []
        self._turn = 0

    def begin_turn(self) -> int:
        """Advance to the next user turn. Call once per incoming user message."""
        self._turn += 1
        return self._turn

    @property
    def current_turn(self) -> int:
        return self._turn

    def record(self, items: Iterable[tuple[str, str]], tool_name: str) -> None:
        """Record (label, value) pairs a tool produced during the current turn.

        Re-recording a known (label, value) refreshes its recency instead of
        duplicating it, so the freshest occurrence wins on recall.
        """
        for label, value in items:
            if not label or not value:
                continue
            self._items = [
                a for a in self._items if not (a.label == label and a.value == value)
            ]
            self._items.append(Artifact(label, value, self._turn, tool_name))

    def current_turn_items(self) -> list[tuple[str, str]]:
        """Artifacts produced in the turn now in progress, in production order."""
        return [(a.label, a.value) for a in self._items if a.turn == self._turn]

    def recent_first(self) -> list[tuple[str, str]]:
        """All artifacts, most recently produced first."""
        return [(a.label, a.value) for a in reversed(self._items)]

    def latest(self, label: str) -> str | None:
        """Most recent value recorded under *label*, or None."""
        for a in reversed(self._items):
            if a.label == label:
                return a.value
        return None
