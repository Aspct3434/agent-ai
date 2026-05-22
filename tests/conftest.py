"""Pytest configuration: stub optional backend dependencies so every test
module can import agent.py, gateway.py, and memory.py without needing a live
ChromaDB / Neo4j / sentence-transformers installation.

All functional correctness for those backends is exercised by the integration
tests (test_memory.py, test_end_to_end.py) which deliberately skip when the
real packages are absent.  This conftest only handles the *import-time*
dependency so unit tests can exercise the pure-Python logic.
"""
from __future__ import annotations

import sys
import types
import unittest.mock


def _stub_module(name: str) -> unittest.mock.MagicMock:
    """Return a MagicMock registered under *name* in sys.modules."""
    mod = unittest.mock.MagicMock()
    sys.modules[name] = mod
    return mod


# Only inject stubs when the real packages are absent.
if "chromadb" not in sys.modules:
    try:
        import chromadb  # noqa: F401
    except ModuleNotFoundError:
        _stub_module("chromadb")

if "neo4j" not in sys.modules:
    try:
        import neo4j  # noqa: F401
    except ModuleNotFoundError:
        neo4j_stub = types.ModuleType("neo4j")
        neo4j_stub.GraphDatabase = unittest.mock.MagicMock()  # type: ignore[attr-defined]
        sys.modules["neo4j"] = neo4j_stub

if "sentence_transformers" not in sys.modules:
    try:
        import sentence_transformers  # noqa: F401
    except ModuleNotFoundError:
        _stub_module("sentence_transformers")
