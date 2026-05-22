from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import memory as memory_module  # noqa: E402

RAW_TEXT = "Deployed a Python script that monitors API endpoints"
ENTITIES = {
    "Subject": "Python Script",
    "Action": "Monitors",
    "Object": "API Endpoints",
}


class _Vector(list[float]):
    def tolist(self) -> list[float]:
        return list(self)


class _DeterministicEncoder:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def encode(self, text: str) -> _Vector:
        normalized = text.lower()
        return _Vector(
            [
                float("python" in normalized),
                float("script" in normalized),
                float("api" in normalized),
                float("endpoint" in normalized),
            ]
        )


def _as_graph_entities(entities: dict[str, str]) -> dict[str, Any]:
    return {
        "nodes": [
            {"label": "Subject", "name": entities["Subject"]},
            {"label": "Object", "name": entities["Object"]},
        ],
        "relationships": [
            {
                "source": entities["Subject"],
                "type": entities["Action"],
                "target": entities["Object"],
            }
        ],
    }


async def _delete_event(memory: memory_module.HybridMemory, event_id: str) -> None:
    def delete() -> None:
        with memory._neo4j.session() as session:
            session.execute_write(
                lambda tx: tx.run(
                    "MATCH (e:Event {id: $event_id}) DETACH DELETE e",
                    event_id=event_id,
                ).consume()
            )

    await asyncio.to_thread(delete)


async def run_memory_test() -> None:
    memory_module.SentenceTransformer = _DeterministicEncoder

    chroma_path = Path(tempfile.mkdtemp(prefix="hybrid-memory-chroma-"))
    session_id = f"test-session-{uuid.uuid4()}"
    event_id: str | None = None
    memory = memory_module.HybridMemory(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="",
        chroma_path=str(chroma_path),
    )

    try:
        print("Storing test memory in vector and graph backends...")
        event_id = await asyncio.to_thread(
            memory.store_event,
            session_id,
            RAW_TEXT,
            _as_graph_entities(ENTITIES),
        )
        print(f"Stored event_id={event_id}")

        print("Querying vector database...")
        vector_result = await asyncio.to_thread(
            memory.retrieve_context,
            "Python script monitoring API endpoints",
            "semantic",
            1,
        )
        vector_hits = vector_result["results"]
        assert vector_hits, "expected at least one vector result"
        top_hit = vector_hits[0]
        assert top_hit["id"] == event_id
        assert top_hit["text"] == RAW_TEXT
        assert top_hit["session_id"] == session_id
        print("  [PASS] Vector result contains the stored raw text and session id")

        print("Querying graph database...")
        cypher = f"""
        MATCH (e:Event {{id: {json.dumps(event_id)}}})-[:CONTAINS]->(subject:Subject)
        MATCH (e)-[:CONTAINS]->(object:Object)
        MATCH (subject)-[action]->(object)
        RETURN
            e.id AS event_id,
            e.session_id AS session_id,
            subject.name AS subject,
            type(action) AS action,
            object.name AS object
        """
        graph_result = await asyncio.to_thread(
            memory.retrieve_context,
            cypher,
            "factual",
            1,
        )
        graph_rows = graph_result["results"]
        assert graph_rows, "expected at least one graph result"
        row = graph_rows[0]
        assert row["event_id"] == event_id
        assert row["session_id"] == session_id
        assert row["subject"] == ENTITIES["Subject"]
        assert row["action"] == ENTITIES["Action"]
        assert row["object"] == ENTITIES["Object"]
        print("  [PASS] Graph result contains the stored subject/action/object triple")

        print("\nALL MEMORY CHECKS PASSED")
    finally:
        if event_id is not None:
            await _delete_event(memory, event_id)
        memory.close()
        shutil.rmtree(chroma_path, ignore_errors=True)


def test_hybrid_memory_stores_vector_and_graph_data() -> None:
    asyncio.run(run_memory_test())


if __name__ == "__main__":
    asyncio.run(run_memory_test())
