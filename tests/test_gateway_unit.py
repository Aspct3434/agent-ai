"""Unit tests for gateway.py: session lane, manager, and model classes.

These tests cover the pure-Python FIFO concurrency logic without requiring a
live database, MCP server, or LLM API key.  The conftest.py stubs out the
optional backend imports (chromadb, neo4j, sentence_transformers) so gateway.py
and agent.py can import cleanly.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from gateway import (
    Gateway,
    Message,
    Result,
    SessionLaneManager,
    WebhookPayload,
    WebhookResponse,
    _EmptyMemory,
    _SessionLane,
)

# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class TestModelClasses:
    def test_message_default_id(self):
        msg = Message(session_id="s1", payload={"x": 1})
        assert msg.message_id  # auto-generated UUID
        assert msg.session_id == "s1"
        assert msg.payload == {"x": 1}

    def test_result_success(self):
        r = Result(message_id="m1", session_id="s1", output="done")
        assert r.error is None
        assert r.output == "done"

    def test_result_error(self):
        r = Result(message_id="m1", session_id="s1", error="boom")
        assert r.error == "boom"
        assert r.output is None

    def test_webhook_payload(self):
        wp = WebhookPayload(chat_id="c1", user_id="u1", text="hello")
        assert wp.text == "hello"

    def test_webhook_response(self):
        wr = WebhookResponse(chat_id="c1", user_id="u1", reply="hi")
        assert wr.reply == "hi"
        assert wr.error is None


# ---------------------------------------------------------------------------
# _EmptyMemory
# ---------------------------------------------------------------------------

class TestEmptyMemory:
    def test_retrieve_returns_empty_results(self):
        mem = _EmptyMemory()
        ctx = mem.retrieve_context("query", "semantic")
        assert ctx["results"] == []
        assert ctx["query_type"] == "semantic"

    def test_store_event_returns_empty_string(self):
        mem = _EmptyMemory()
        result = mem.store_event("session1", "raw text", {"nodes": []})
        assert result == ""


# ---------------------------------------------------------------------------
# _SessionLane
# ---------------------------------------------------------------------------

class TestSessionLane:
    @pytest.mark.asyncio
    async def test_single_message_processed(self):
        results = []

        async def handler(msg: Message) -> str:
            results.append(msg.payload["val"])
            return f"ok:{msg.payload['val']}"

        lane = _SessionLane("s1", handler)
        lane.start()

        result = await lane.submit(Message(session_id="s1", payload={"val": 42}))
        await lane.drain_and_stop()

        assert result.output == "ok:42"
        assert results == [42]

    @pytest.mark.asyncio
    async def test_messages_processed_in_order(self):
        order = []

        async def handler(msg: Message) -> str:
            order.append(msg.payload["n"])
            return "ok"

        lane = _SessionLane("s1", handler)
        lane.start()

        futs = [
            asyncio.create_task(
                lane.submit(Message(session_id="s1", payload={"n": i}))
            )
            for i in range(10)
        ]
        await asyncio.gather(*futs)
        await lane.drain_and_stop()

        assert order == list(range(10))

    @pytest.mark.asyncio
    async def test_handler_exception_becomes_error_result(self):
        async def bad_handler(msg: Message) -> str:
            raise ValueError("intentional error")

        lane = _SessionLane("s1", bad_handler)
        lane.start()

        result = await lane.submit(Message(session_id="s1", payload={}))
        await lane.drain_and_stop()

        assert result.error is not None
        assert "intentional error" in result.error

    @pytest.mark.asyncio
    async def test_no_concurrent_execution_in_one_lane(self):
        """Concurrent .submit() calls on ONE lane must serialise execution."""
        concurrency_count = [0]
        max_concurrent = [0]

        async def handler(msg: Message) -> str:
            concurrency_count[0] += 1
            max_concurrent[0] = max(max_concurrent[0], concurrency_count[0])
            await asyncio.sleep(0)  # yield, giving other tasks a chance to run
            concurrency_count[0] -= 1
            return "ok"

        lane = _SessionLane("s1", handler)
        lane.start()

        await asyncio.gather(
            *(lane.submit(Message(session_id="s1", payload={"n": i})) for i in range(20))
        )
        await lane.drain_and_stop()

        assert max_concurrent[0] == 1, (
            f"Lane executed {max_concurrent[0]} messages concurrently"
        )


# ---------------------------------------------------------------------------
# SessionLaneManager
# ---------------------------------------------------------------------------

class TestSessionLaneManager:
    @pytest.mark.asyncio
    async def test_creates_separate_lane_per_session(self):
        seen_sessions = []

        async def handler(msg: Message) -> str:
            seen_sessions.append(msg.session_id)
            return "ok"

        manager = SessionLaneManager(handler)
        await asyncio.gather(
            manager.submit(Message(session_id="A", payload={})),
            manager.submit(Message(session_id="B", payload={})),
        )
        await manager.shutdown()

        assert set(seen_sessions) == {"A", "B"}

    @pytest.mark.asyncio
    async def test_active_sessions_property(self):
        async def handler(msg: Message) -> str:
            await asyncio.sleep(0.01)
            return "ok"

        manager = SessionLaneManager(handler)
        # Submit but don't await, so the lane is created immediately
        task = asyncio.create_task(
            manager.submit(Message(session_id="sess-x", payload={}))
        )
        # Yield to let the lane creation happen
        await asyncio.sleep(0)
        assert "sess-x" in manager.active_sessions
        await task
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_different_sessions_run_concurrently(self):
        """Two sessions should run in parallel, not sequentially."""
        start_times: dict[str, float] = {}

        async def slow_handler(msg: Message) -> str:
            start_times[msg.session_id] = asyncio.get_event_loop().time()
            await asyncio.sleep(0.05)
            return "ok"

        manager = SessionLaneManager(slow_handler)
        t0 = asyncio.get_event_loop().time()
        await asyncio.gather(
            manager.submit(Message(session_id="A", payload={})),
            manager.submit(Message(session_id="B", payload={})),
        )
        elapsed = asyncio.get_event_loop().time() - t0
        await manager.shutdown()

        # Both should finish in ~50ms (concurrent), not ~100ms (sequential).
        assert elapsed < 0.09, f"Sessions ran sequentially: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------

class TestGateway:
    @pytest.mark.asyncio
    async def test_send_returns_result(self):
        async def handler(msg: Message) -> str:
            return "pong"

        gw = Gateway(handler)
        result = await gw.send("session-1", {"ping": True})
        await gw.shutdown()

        assert result.output == "pong"
        assert result.session_id == "session-1"

    @pytest.mark.asyncio
    async def test_active_sessions(self):
        received = asyncio.Event()
        done = asyncio.Event()

        async def blocking_handler(msg: Message) -> str:
            received.set()
            await done.wait()
            return "ok"

        gw = Gateway(blocking_handler)
        task = asyncio.create_task(gw.send("my-session", {}))

        await received.wait()
        assert "my-session" in gw.active_sessions

        done.set()
        await task
        await gw.shutdown()

    @pytest.mark.asyncio
    async def test_load_all_sessions_covered(self):
        """50 sessions x 10 messages; all results OK, no race within a session."""
        import random
        random.seed(99)

        concurrency: dict[str, int] = {}
        race_violations: list[str] = []

        async def handler(msg: Message) -> str:
            sid = msg.session_id
            concurrency[sid] = concurrency.get(sid, 0) + 1
            if concurrency[sid] > 1:
                race_violations.append(sid)
            await asyncio.sleep(random.uniform(0.001, 0.005))
            concurrency[sid] -= 1
            return "ok"

        sessions = [f"sess-{i:03d}" for i in range(50)]
        msgs = [
            (sid, {"step": step})
            for sid in sessions
            for step in range(10)
        ]
        random.shuffle(msgs)

        gw = Gateway(handler)
        results = await asyncio.gather(*(gw.send(sid, p) for sid, p in msgs))
        await gw.shutdown()

        assert len(results) == 500
        assert all(r.error is None for r in results)
        assert race_violations == [], f"Race conditions: {race_violations[:3]}"
        assert {r.session_id for r in results} == set(sessions)
