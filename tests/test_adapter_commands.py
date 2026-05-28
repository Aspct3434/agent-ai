from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapters._commands import is_stop_command


def test_stop_command_accepts_plain_short_control_words():
    assert is_stop_command("Stop")
    assert is_stop_command("cancel!")
    assert is_stop_command(" /stop now")


def test_stop_command_does_not_capture_normal_sentences():
    assert not is_stop_command("stop adding so many animations")
    assert not is_stop_command("how do I stop a server?")
