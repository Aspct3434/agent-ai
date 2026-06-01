from __future__ import annotations

import copy
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


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
        # Imported lazily so the lightweight install (no ML extra) can still
        # import this module for UserProfileStore. These backends are only
        # needed when AGENT_USE_HYBRID_MEMORY is enabled.
        import chromadb
        from neo4j import GraphDatabase
        from sentence_transformers import SentenceTransformer

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


# ---------------------------------------------------------------------------
# User profile store
# ---------------------------------------------------------------------------


_DEFAULT_PROFILE: dict[str, Any] = {
    "name": "",
    "expertise": [],
    "communication_style": "",
    "preferences": [],
    "recurring_topics": [],
    "goals": [],
    "pending_updates": [],
    "last_updated": "",
    "interaction_count": 0,
}


def _fresh_default_profile() -> dict[str, Any]:
    return copy.deepcopy(_DEFAULT_PROFILE)


class UserProfileStore:
    """Persists a structured user profile to a JSON file.

    The profile is built incrementally from conversations by an LLM extractor
    (see ``evaluator.extract_and_update_user_profile``). The stored data is
    injected into each new session to personalize the agent's responses.

    File location: ``{profile_dir}/user_profile.json``
    """

    def __init__(self, profile_dir: str | Path = "./chroma_db") -> None:
        self._path = Path(profile_dir) / "user_profile.json"
        self._profile: dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self) -> dict[str, Any]:
        return dict(self._profile)

    def update(self, updates: dict[str, Any]) -> None:
        """Merge *updates* into the stored profile and persist."""
        self._merge_updates(updates)
        self._profile["last_updated"] = datetime.now(UTC).isoformat()
        self._profile["interaction_count"] = self._profile.get("interaction_count", 0) + 1
        self._save()

    def propose_update(self, updates: dict[str, Any]) -> str | None:
        """Store extracted profile changes for review instead of injecting them."""
        filtered = self._filtered_updates(updates)
        if not filtered:
            return None
        pending = self._profile.setdefault("pending_updates", [])
        if not isinstance(pending, list):
            pending = []
            self._profile["pending_updates"] = pending
        update_id = uuid.uuid4().hex
        pending.append(
            {
                "id": update_id,
                "updates": filtered,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        self._profile["last_updated"] = datetime.now(UTC).isoformat()
        self._save()
        return update_id

    def approve_pending(self, update_id: str | None = None) -> int:
        """Approve one pending update, or all pending updates when id is omitted."""
        pending = self._pending_updates()
        kept: list[dict[str, Any]] = []
        approved = 0
        for item in pending:
            if update_id is not None and item.get("id") != update_id:
                kept.append(item)
                continue
            updates = item.get("updates")
            if isinstance(updates, dict):
                self._merge_updates(updates)
                approved += 1
        self._profile["pending_updates"] = kept
        if approved:
            self._profile["last_updated"] = datetime.now(UTC).isoformat()
            self._profile["interaction_count"] = self._profile.get("interaction_count", 0) + approved
            self._save()
        return approved

    def reject_pending(self, update_id: str | None = None) -> int:
        """Reject one pending update, or all pending updates when id is omitted."""
        pending = self._pending_updates()
        if update_id is None:
            rejected = len(pending)
            self._profile["pending_updates"] = []
        else:
            kept = [item for item in pending if item.get("id") != update_id]
            rejected = len(pending) - len(kept)
            self._profile["pending_updates"] = kept
        if rejected:
            self._profile["last_updated"] = datetime.now(UTC).isoformat()
            self._save()
        return rejected

    def _merge_updates(self, updates: dict[str, Any]) -> None:
        for key, value in self._filtered_updates(updates).items():
            current = self._profile.get(key)
            if isinstance(current, list) and isinstance(value, list):
                seen = set(current)
                for item in value:
                    if item and item not in seen:
                        current.append(item)
                        seen.add(item)
                self._profile[key] = current[-20:]
            elif value:
                self._profile[key] = value

    def _filtered_updates(self, updates: dict[str, Any]) -> dict[str, Any]:
        filtered: dict[str, Any] = {}
        for key, value in updates.items():
            if key not in _DEFAULT_PROFILE or key in {
                "pending_updates",
                "last_updated",
                "interaction_count",
            }:
                continue
            if isinstance(value, list):
                cleaned = [item for item in value if item]
                if cleaned:
                    filtered[key] = cleaned
            elif value:
                filtered[key] = value
        return filtered

    def _pending_updates(self) -> list[dict[str, Any]]:
        pending = self._profile.get("pending_updates")
        if not isinstance(pending, list):
            return []
        return [item for item in pending if isinstance(item, dict)]

    def as_context_string(self) -> str:
        """Return a concise single-line summary for system prompt injection."""
        p = self._profile
        parts: list[str] = []
        if p.get("name"):
            parts.append(f"Name: {p['name']}")
        if p.get("expertise"):
            parts.append(f"Expertise: {', '.join(p['expertise'][:4])}")
        if p.get("communication_style"):
            parts.append(f"Style: {p['communication_style']}")
        if p.get("preferences"):
            parts.append(f"Preferences: {', '.join(p['preferences'][:3])}")
        if p.get("recurring_topics"):
            parts.append(f"Topics: {', '.join(p['recurring_topics'][:4])}")
        return " | ".join(parts) if parts else ""

    def clear(self) -> None:
        self._profile = _fresh_default_profile()
        self._save()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                return {**_fresh_default_profile(), **data}
        except Exception:
            pass
        return _fresh_default_profile()

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._profile, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Could not save user profile: %s", exc)
