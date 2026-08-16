#!/usr/bin/env python
"""Test retriever.py: search_sources() maps hybrid_search() blocks into Source objects."""
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

_TEST_COLLECTION = "test_retriever_hybrid_collection"
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

from app.rag.retriever import search_sources
from app.rag.types import Chunk
from app.rag.vector_store import add_chunks, get_collection


def test_retriever_hybrid() -> bool:
    print("=" * 70)
    print("RETRIEVER (Path A) HYBRID SEARCH TEST")
    print("=" * 70)

    chunks = [
        Chunk(
            id="hash1:p1:c1",
            text="Section about firmware rollback overlap sentence marker here",
            metadata={"document_name": "manual.pdf", "file_hash": "hash1", "page": 1, "chunk_index": 1, "chunk_seq": 1},
        ),
        Chunk(
            id="hash1:p1:c2",
            text="overlap sentence marker here continues describing quartzite calibration steps",
            metadata={"document_name": "manual.pdf", "file_hash": "hash1", "page": 1, "chunk_index": 2, "chunk_seq": 2},
        ),
    ]

    try:
        print("\n[1/2] Seeding 2 same-page chunks...")
        added = add_chunks(chunks)
        assert added == 2, f"expected 2 chunks added, got {added}"
        print(f"   OK: added {added}")

        print("\n[2/2] search_sources() returns a Source with merged text, pages, chunk range, and link...")
        sources = search_sources("quartzite calibration", top_k=5)
        assert len(sources) >= 1, f"expected at least one source, got {sources}"
        source = sources[0]
        assert source.document == "manual.pdf"
        assert source.page == 1
        assert source.pages == [1]
        assert source.chunk == "1-2", f"expected merged chunk_seq range '1-2', got {source.chunk!r}"
        assert source.link == "doc:hash1#p1", f"got {source.link!r}"
        assert "overlap sentence marker here" in source.text
        assert source.text.count("overlap sentence marker here") == 1, f"expected the shared overlap stripped, got: {source.text!r}"
        assert source.score is not None
        print(f"   OK: document={source.document!r} pages={source.pages} chunk={source.chunk!r} link={source.link!r}")

        print("\nPASSED: retriever.py's search_sources() correctly maps hybrid_search() blocks to Source objects.")
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
    success = test_retriever_hybrid()
    sys.exit(0 if success else 1)
