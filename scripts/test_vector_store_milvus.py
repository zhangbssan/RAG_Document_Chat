#!/usr/bin/env python
"""Test the Milvus-backed vector_store.py: add -> query -> list -> delete round trip."""
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

_TEST_COLLECTION = "test_realtime_pdf_collection"
os.environ["REALTIME_PDF_COLLECTION_NAME"] = _TEST_COLLECTION

from app.rag import embeddings as embedding_module


def _test_embed(texts: list[str], dimensions: int = 384) -> list[list[float]]:
    """Deterministic local embeddings for this script; avoids model downloads."""
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
from app.rag.vector_store import (
    add_chunks,
    delete_document,
    get_collection,
    indexed_file_hashes,
    list_documents,
    query_chunks,
)


def test_vector_store_milvus() -> bool:
    print("=" * 70)
    print("MILVUS VECTOR STORE TEST")
    print("=" * 70)

    chunks = [
        Chunk(
            id="hash1:p1:c1",
            text="The service agreement covers annual maintenance for industrial equipment.",
            metadata={"document_name": "service_agreement.pdf", "file_hash": "hash1", "page": 1, "chunk_index": 1},
        ),
        Chunk(
            id="hash1:p2:c1",
            text="Termination requires 30 days written notice from either party.",
            metadata={"document_name": "service_agreement.pdf", "file_hash": "hash1", "page": 2, "chunk_index": 1},
        ),
    ]

    try:
        print("\n[1/5] Adding chunks...")
        added = add_chunks(chunks)
        assert added == 2, f"expected 2 chunks added, got {added}"
        print(f"   OK: added {added} chunks")

        print("\n[2/5] indexed_file_hashes()...")
        hashes = indexed_file_hashes()
        assert "hash1" in hashes, f"expected hash1 in {hashes}"
        print(f"   OK: {hashes}")

        print("\n[3/5] query_chunks()...")
        results = query_chunks("What is the termination notice period?", top_k=2)
        assert len(results) > 0, "expected at least one result"
        assert any("30 days" in r["text"] for r in results), f"expected termination chunk in results: {results}"
        print(f"   OK: {len(results)} results, top text: {results[0]['text'][:60]}...")

        print("\n[4/5] list_documents()...")
        docs = list_documents()
        assert len(docs) == 1, f"expected 1 document, got {docs}"
        assert docs[0]["document_name"] == "service_agreement.pdf"
        assert docs[0]["chunks"] == 2
        print(f"   OK: {docs}")

        print("\n[5/5] delete_document()...")
        deleted = delete_document("hash1")
        assert deleted == 2, f"expected 2 deleted, got {deleted}"
        remaining = list_documents()
        assert remaining == [], f"expected no documents left, got {remaining}"
        print(f"   OK: deleted {deleted}, {len(remaining)} documents remain")

        print("\nPASSED: vector_store.py works against Milvus.")
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
    success = test_vector_store_milvus()
    sys.exit(0 if success else 1)