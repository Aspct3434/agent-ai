from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artifacts import ArtifactLedger  # noqa: E402


def test_records_and_scopes_to_current_turn() -> None:
    led = ArtifactLedger()
    led.begin_turn()
    led.record([("File", "/tmp/a.py")], "write_text_file")
    led.begin_turn()
    led.record([("Service URL", "http://x/")], "expose_local_http_service")

    # current_turn_items reflects only the turn in progress.
    assert led.current_turn_items() == [("Service URL", "http://x/")]
    # recent_first spans all turns, newest first.
    assert led.recent_first() == [("Service URL", "http://x/"), ("File", "/tmp/a.py")]


def test_dedup_refreshes_recency() -> None:
    led = ArtifactLedger()
    led.begin_turn()
    led.record([("File", "/tmp/a.py"), ("File", "/tmp/b.py")], "write_text_file")
    led.begin_turn()
    # Re-touching /tmp/a.py should move it to the front, not duplicate it.
    led.record([("File", "/tmp/a.py")], "write_text_file")

    recent = led.recent_first()
    assert recent == [("File", "/tmp/a.py"), ("File", "/tmp/b.py")]
    assert recent.count(("File", "/tmp/a.py")) == 1


def test_latest_by_label_and_empties_ignored() -> None:
    led = ArtifactLedger()
    led.begin_turn()
    led.record([("Service URL", "http://old/"), ("", "skip"), ("File", "")], "t")
    led.begin_turn()
    led.record([("Service URL", "http://new/")], "t")

    assert led.latest("Service URL") == "http://new/"
    assert led.latest("Port") is None
    # Blank label/value pairs are never recorded.
    assert all(label and value for label, value in led.recent_first())
