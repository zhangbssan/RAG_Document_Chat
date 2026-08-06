#!/usr/bin/env python
"""Test agent/tools.py: search_uploaded_docs tool in isolation, no LLM involved."""
from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_TEST_COLLECTION = "test_agent_tools_collection"
os.environ["REALTIME_PDF_COLLECTION_NAME"] = _TEST_COLLECTION

from app.rag import embeddings as embedding_module


def _test_embed(texts: list[str], dimensions: int = 384) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dimensions
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


embedding_module.embed_texts = _test_embed
embedding_module.embed_query = lambda q: _test_embed([q])[0]

from app.rag.types import Chunk
from app.rag.vector_store import add_chunks, get_collection
from app.agent.tools import UserContext, _search_uploaded_docs_impl, make_search_tool


def test_search_uploaded_docs_tool() -> bool:
    print("=" * 70)
    print("SEARCH_UPLOADED_DOCS TOOL TEST")
    print("=" * 70)

    chunks = [
        Chunk(
            id="hash1:p1:c1",
            text="The office WiFi password is SkyBlue42.",
            metadata={"document_name": "office_handbook.pdf", "file_hash": "hash1", "page": 3, "chunk_index": 1},
        ),
    ]

    try:
        print("\n[1/3] Seeding one chunk...")
        add_chunks(chunks)
        print("   OK")

        print("\n[2/3] Invoking the LLM-bindable tool (only `query` in its schema)...")
        user_context = UserContext(user_id="u1", department_id=None, session_id="s1")
        search_tool = make_search_tool(user_context, top_k=3)
        assert list(search_tool.args.keys()) == ["query"], f"expected only 'query' exposed, got {search_tool.args}"

        result = search_tool.invoke({"query": "What is the WiFi password?"})
        assert result["status"] == "success", f"expected success, got {result}"
        assert "SkyBlue42" in result["content"], f"expected password in content: {result}"
        assert len(result["citations"]) > 0, f"expected citations: {result}"
        citation = result["citations"][0]
        for key in ("source_id", "source_name", "page", "chunk_id", "text"):
            assert key in citation, f"citation missing '{key}': {citation}"
        assert citation["source_name"] == "office_handbook.pdf"
        assert citation["source_id"] == "hash1"
        assert result["metadata"] == {"tool": "search_uploaded_docs", "result_count": len(result["citations"])}
        assert result["error"] is None
        print(f"   OK: {result}")

        print("\n[3/3] Error path: empty query...")
        error_result = _search_uploaded_docs_impl("", user_context, top_k=3)
        assert error_result["status"] == "error", f"expected error status: {error_result}"
        assert error_result["error"], f"expected a non-null error message: {error_result}"
        assert error_result["content"] == ""
        assert error_result["citations"] == []
        print(f"   OK: {error_result}")

        print("\nPASSED: search_uploaded_docs tool works correctly, in isolation.")
        return True

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return False

    finally:
        client = get_collection()
        if client.has_collection(_TEST_COLLECTION):
            client.drop_collection(_TEST_COLLECTION)
            print(f"\n[cleanup] dropped {_TEST_COLLECTION}")


if __name__ == "__main__":
    success = test_search_uploaded_docs_tool()
    sys.exit(0 if success else 1)