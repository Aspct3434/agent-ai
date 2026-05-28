from __future__ import annotations

_STOP_WORDS = frozenset({"stop", "cancel", "interrupt", "abort"})


def is_stop_command(text: str) -> bool:
    """True for short user control messages that should cancel the active turn."""
    normalized = text.strip().lower().strip(".! \t\r\n")
    return normalized in _STOP_WORDS or normalized.startswith("/stop")
