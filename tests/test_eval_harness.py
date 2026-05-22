from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eval_harness import run_self_test  # noqa: E402


def test_eval_harness_self_test() -> None:
    assert run_self_test() == 0


if __name__ == "__main__":
    raise SystemExit(run_self_test())
