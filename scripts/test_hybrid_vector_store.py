#!/usr/bin/env python
"""Test vector_store.py's hybrid-search primitives: chunk_seq on dense search,
sparse_search() (BM25 FTS), and get_chunks_by_seq() (anchor window fetch)."""
from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import time
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_TEST_COLLECTION = "test_hybrid_vector_store_collection"
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
from app.rag.vector_store import add_chunks, get_chunks_by_seq, get_collection, query_chunks, sparse_search


def _get_chunks_by_seq_until(file_hash: str, chunk_seqs: list[int], expected_rows: int, attempts: int = 20, delay: float = 0.25) -> list[dict]:
    """Poll get_chunks_by_seq() until every row carries chunk_seq.

    Right after add_chunks()'s insert+flush, a query for dynamic fields can race the
    write on Milvus's default consistency level and momentarily return rows missing
    dynamic fields like chunk_seq (documented in test_vector_store_milvus.py's
    _query_until()). Condition-based polling, not a fixed sleep or a single attempt.
    """
    last_rows: list[dict] = []
    for _ in range(attempts):
        last_rows = get_chunks_by_seq(file_hash, chunk_seqs)
        if len(last_rows) == expected_rows and all("chunk_seq" in row for row in last_rows):
            return last_rows
        time.sleep(delay)
    return last_rows


def test_hybrid_vector_store() -> bool:
    print("=" * 70)
    print("VECTOR_STORE HYBRID PRIMITIVES TEST")
    print("=" * 70)

    chunks = [
        Chunk(
            id="hash1:p1:c1",
            text="The onboarding checklist covers laptop setup and badge issuance.",
            metadata={"document_name": "onboarding.pdf", "file_hash": "hash1", "page": 1, "chunk_index": 1, "chunk_seq": 1},
        ),
        Chunk(
            id="hash1:p1:c2",
            text="Badge issuance requires photo ID verification at the front desk.",
            metadata={"document_name": "onboarding.pdf", "file_hash": "hash1", "page": 1, "chunk_index": 2, "chunk_seq": 2},
        ),
        Chunk(
            id="hash1:p2:c1",
            text="Parking permits are issued separately by facilities management.",
            metadata={"document_name": "onboarding.pdf", "file_hash": "hash1", "page": 2, "chunk_index": 1, "chunk_seq": 3},
        ),
    ]

    try:
        print("\n[1/4] Seeding 3 chunks...")
        added = add_chunks(chunks)
        assert added == 3, f"expected 3 chunks added, got {added}"
        print(f"   OK: added {added}")

        print("\n[2/4] query_chunks() (dense) now returns chunk_seq in metadata...")
        dense_results = query_chunks("laptop setup", top_k=3)
        assert dense_results, "expected at least one dense result"
        assert all("chunk_seq" in r["metadata"] for r in dense_results), f"expected chunk_seq on every hit: {dense_results}"
        top = next(r for r in dense_results if "laptop" in r["text"])
        assert top["metadata"]["chunk_seq"] == 1, f"expected chunk_seq=1 on the laptop chunk, got: {top}"
        print(f"   OK: {len(dense_results)} dense hits, all carry chunk_seq")

        print("\n[3/4] sparse_search() (BM25 FTS) finds a chunk by its distinguishing keyword...")
        sparse_results = sparse_search("parking", top_k=3)
        assert sparse_results, "expected at least one sparse result"
        assert sparse_results[0]["metadata"]["chunk_seq"] == 3, f"expected the parking chunk (chunk_seq=3) to rank first, got: {sparse_results[0]}"
        assert "Parking permits" in sparse_results[0]["text"]
        print(f"   OK: top sparse hit is chunk_seq=3: {sparse_results[0]['text'][:50]}...")

        print("\n[4/4] get_chunks_by_seq() fetches an anchor's neighbours by chunk_seq...")
        window = _get_chunks_by_seq_until("hash1", [1, 2], expected_rows=2)
        assert len(window) == 2, f"expected 2 rows, got {window}"
        window.sort(key=lambda r: r["chunk_seq"])
        assert [r["chunk_seq"] for r in window] == [1, 2], f"expected chunk_seq [1, 2], got: {window}"
        assert window[0]["page"] == 1 and window[1]["page"] == 1
        assert "id" in window[0] and "text" in window[0]
        empty = get_chunks_by_seq("hash1", [])
        assert empty == [], f"expected [] for an empty chunk_seqs list, got {empty}"
        print(f"   OK: fetched chunk_seq=[1, 2], empty-input short-circuit works")

        print("\nPASSED: vector_store.py hybrid-search primitives work against Milvus.")
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
    success = test_hybrid_vector_store()
    sys.exit(0 if success else 1)
