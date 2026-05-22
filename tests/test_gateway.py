"""
Load test: 1,000 mock webhooks across 50 sessions fired simultaneously.

Checks:
  1. All 1,000 messages complete with no errors.
  2. No concurrent handler execution within a single session (race condition).
  3. Each session's messages are processed in submission order.
  4. All 50 sessions are represented in results.
"""

import asyncio
import random
import time
from collections import defaultdict

from gateway import Gateway, Message

NUM_MESSAGES = 1_000
NUM_SESSIONS = 50
MESSAGES_PER_SESSION = NUM_MESSAGES // NUM_SESSIONS  # 20 per session

# ---------------------------------------------------------------------------
# Shared tracking state
# Safe without locks: asyncio is single-threaded; we only mutate between awaits.
# ---------------------------------------------------------------------------
active_per_session: dict[str, int] = defaultdict(int)
processed_order: dict[str, list[int]] = defaultdict(list)
race_violations: list[str] = []


async def mock_handler(message: Message) -> str:
    sid = message.session_id
    step: int = message.payload["step"]

    # Increment before the await — no switch can happen here.
    active_per_session[sid] += 1
    if active_per_session[sid] > 1:
        race_violations.append(
            f"session={sid} step={step} concurrent={active_per_session[sid]}"
        )

    await asyncio.sleep(random.uniform(0.001, 0.010))  # simulate variable I/O

    processed_order[sid].append(step)
    active_per_session[sid] -= 1
    return "ok"


def build_sends() -> list[tuple[str, dict]]:
    """
    Generate all (session_id, payload) pairs, then shuffle so sessions
    interleave on arrival — mimicking real concurrent webhook traffic.
    Within each session the steps are generated 0..N-1 before the shuffle,
    so submission_order below records the exact per-session arrival sequence.
    """
    sends: list[tuple[str, dict]] = []
    for s in range(NUM_SESSIONS):
        sid = f"session-{s:03d}"
        for step in range(MESSAGES_PER_SESSION):
            sends.append((sid, {"step": step}))
    random.shuffle(sends)
    return sends


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if not ok and detail:
        line += f"\n         {detail}"
    print(line)
    return ok


async def run() -> None:
    random.seed(42)
    gateway = Gateway(mock_handler)

    sends = build_sends()

    # Record per-session submission order (the sequence gateway.send is called).
    submission_order: dict[str, list[int]] = defaultdict(list)
    for sid, payload in sends:
        submission_order[sid].append(payload["step"])

    print(f"Firing {NUM_MESSAGES:,} messages across {NUM_SESSIONS} sessions...\n")
    t0 = time.perf_counter()

    results = await asyncio.gather(*(gateway.send(sid, p) for sid, p in sends))

    elapsed = time.perf_counter() - t0
    await gateway.shutdown()

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------
    all_passed = True

    ok = _check(
        f"All {NUM_MESSAGES:,} messages returned a result",
        len(results) == NUM_MESSAGES,
        f"got {len(results)}",
    )
    all_passed &= ok

    errored = [r for r in results if r.error is not None]
    ok = _check(
        "No handler errors",
        len(errored) == 0,
        f"{len(errored)} error(s): {errored[:2]}",
    )
    all_passed &= ok

    ok = _check(
        "No concurrent handler execution within any session",
        len(race_violations) == 0,
        f"{len(race_violations)} violation(s): {race_violations[:3]}",
    )
    all_passed &= ok

    order_failures: list[str] = []
    for sid in sorted(submission_order):
        expected = submission_order[sid]
        actual = processed_order.get(sid, [])
        if actual != expected:
            order_failures.append(
                f"{sid}: expected={expected[:6]}… actual={actual[:6]}…"
            )
    ok = _check(
        f"Processing order matches submission order for all {NUM_SESSIONS} sessions",
        len(order_failures) == 0,
        "; ".join(order_failures[:3]),
    )
    all_passed &= ok

    missing = {f"session-{s:03d}" for s in range(NUM_SESSIONS)} - set(processed_order)
    ok = _check(
        f"All {NUM_SESSIONS} sessions covered",
        len(missing) == 0,
        f"missing: {missing}",
    )
    all_passed &= ok

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\nTime : {elapsed:.3f}s  ({NUM_MESSAGES / elapsed:,.0f} msg/s)")
    print(f"Peak : up to {NUM_SESSIONS} sessions running in parallel")

    sequential_estimate = NUM_MESSAGES * 0.0055  # midpoint of [0.001, 0.010]
    print(
        f"Speedup vs sequential: ~{sequential_estimate / elapsed:.1f}x "
        f"(est. sequential={sequential_estimate:.1f}s)"
    )

    print()
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(run())
