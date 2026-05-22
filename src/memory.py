from __future__ import annotations

import re
import uuid
from typing import Any, Literal

import chromadb
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


def _safe_identifier(value: str) -> str:
    """Strip characters that are illegal in Cypher labels / relationship types."""
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not sanitized or sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


class HybridMemory:
    """
    Dual-backend memory store.

    ChromaDB holds semantic embeddings of raw event text.
    Neo4j holds a knowledge graph of entities extracted from those events.

    Expected ``entities`` schema for ``store_event``::

        {
            "nodes": [
                {"label": "Person", "name": "Alice", "properties": {"age": 30}},
                {"label": "Company", "name": "Acme"},
            ],
            "relationships": [
                {"source": "Alice", "type": "WORKS_AT", "target": "Acme",
                 "properties": {"since": 2020}},
            ],
        }

    ``nodes[].label`` and ``relationships[].type`` become Cypher node labels and
    relationship types respectively. Both are sanitized before interpolation.
    ``nodes[].name`` is the merge key; ``properties`` are optional extra attributes.
    """

    _COLLECTION = "events"
    _EMBED_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        chroma_path: str = "./chroma_db",
    ) -> None:
        self._encoder = SentenceTransformer(self._EMBED_MODEL)

        self._chroma = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._chroma.get_or_create_collection(self._COLLECTION)

        try:
            self._neo4j = GraphDatabase.driver(
                neo4j_uri, auth=(neo4j_user, neo4j_password)
            )
            self._neo4j.verify_connectivity()
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "Neo4j unavailable at %s; graph storage disabled.", neo4j_uri
            )
            self._neo4j = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_event(
        self,
        session_id: str,
        raw_text: str,
        entities: dict[str, Any],
    ) -> str:
        """Embed *raw_text* into ChromaDB and write *entities* into Neo4j.

        Returns the generated event UUID that links both stores.
        """
        event_id = str(uuid.uuid4())

        self._store_embedding(event_id, session_id, raw_text)
        if self._neo4j is not None:
            with self._neo4j.session() as session:
                session.execute_write(
                    self._write_graph, event_id, session_id, entities
                )

        return event_id

    def retrieve_context(
        self,
        query: str,
        query_type: Literal["semantic", "factual"],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Query one of the two backends based on *query_type*.

        ``'semantic'``: embeds *query* and returns the *top_k* closest
        documents from ChromaDB. Each result includes the stored text, its
        L2 distance (lower = more similar), and the originating session_id.

        ``'factual'``: treats *query* as a read-only Cypher statement and
        executes it against Neo4j, returning each result row as a plain dict.

        Return shape::

            {
                "query_type": "semantic" | "factual",
                "results": [ ... ],
            }
        """
        if query_type == "semantic":
            return {
                "query_type": "semantic",
                "results": self._semantic_search(query, top_k),
            }
        if query_type == "factual":
            if self._neo4j is None:
                return {"query_type": "factual", "results": []}
            return {
                "query_type": "factual",
                "results": self._factual_query(query),
            }
        raise ValueError(
            f"query_type must be 'semantic' or 'factual', got {query_type!r}"
        )

    def close(self) -> None:
        if self._neo4j is not None:
            self._neo4j.close()

    def __enter__(self) -> HybridMemory:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _store_embedding(
        self, event_id: str, session_id: str, raw_text: str
    ) -> None:
        embedding = self._encoder.encode(raw_text).tolist()
        self._collection.add(
            ids=[event_id],
            embeddings=[embedding],
            documents=[raw_text],
            metadatas=[{"session_id": session_id}],
        )

    def _semantic_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        embedding = self._encoder.encode(query).tolist()
        response = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = response["ids"][0]
        docs = response["documents"][0]
        distances = response["distances"][0]
        metas = response["metadatas"][0]
        return [
            {
                "id": event_id,
                "text": text,
                "distance": dist,
                "session_id": meta.get("session_id"),
            }
            for event_id, text, dist, meta in zip(ids, docs, distances, metas, strict=False)
        ]

    def _factual_query(self, cypher: str) -> list[dict[str, Any]]:
        # execute_read enforces a read-only transaction; writes in the Cypher
        # will raise a ClientError from the driver.
        with self._neo4j.session() as session:
            return list(session.execute_read(lambda tx: tx.run(cypher).data()))

    @staticmethod
    def _write_graph(
        tx, event_id: str, session_id: str, entities: dict[str, Any]
    ) -> None:
        tx.run(
            "MERGE (e:Event {id: $id}) SET e.session_id = $sid",
            id=event_id,
            sid=session_id,
        )

        for node in entities.get("nodes", []):
            label = _safe_identifier(node.get("label", "Entity"))
            name = node.get("name", "")
            props = node.get("properties", {})
            tx.run(
                f"MERGE (n:{label} {{name: $name}}) "
                "SET n += $props "
                "WITH n "
                "MATCH (e:Event {id: $event_id}) "
                "MERGE (e)-[:CONTAINS]->(n)",
                name=name,
                props=props,
                event_id=event_id,
            )

        for rel in entities.get("relationships", []):
            rel_type = _safe_identifier(rel.get("type", "RELATED_TO"))
            props = rel.get("properties", {})
            tx.run(
                f"MATCH (a {{name: $source}}), (b {{name: $target}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                "SET r += $props",
                source=rel.get("source", ""),
                target=rel.get("target", ""),
                props=props,
            )
