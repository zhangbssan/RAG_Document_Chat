#!/usr/bin/env python
"""Test the Milvus-backed vector_store.py: add -> query -> list -> delete round trip."""
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
from app.rag.vector_store import get_collection as _get_collection_for_raw_query


def _query_until(client, collection_name: str, output_fields: list[str], expected_rows: int, attempts: int = 20, delay: float = 0.25) -> list[dict]:
    """Poll client.query() until every row carries every requested field.

    Right after add_chunks()'s insert+flush, a query for dynamic fields can race the
    write on Milvus's default consistency level and momentarily return rows missing
    fields like `page`/`chunk_seq` (confirmed against a real Milvus instance during
    implementation: intermittent even with consistency_level="Strong" requested).
    Condition-based polling is the reliable fix, not a fixed sleep or a single attempt.
    """
    last_rows: list[dict] = []
    for _ in range(attempts):
        last_rows = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=output_fields,
            limit=16384,
            consistency_level="Strong",
        )
        if len(last_rows) == expected_rows and all(
            all(field in row for field in output_fields) for row in last_rows
        ):
            return last_rows
        time.sleep(delay)
    return last_rows


def test_vector_store_milvus() -> bool:
    print("=" * 70)
    print("MILVUS VECTOR STORE TEST")
    print("=" * 70)

    chunks = [
        Chunk(
            id="hash1:p1:c1",
            text="The service agreement covers annual maintenance for industrial equipment.",
            metadata={"document_name": "service_agreement.pdf", "file_hash": "hash1", "page": 1, "chunk_index": 1, "chunk_seq": 1},
        ),
        Chunk(
            id="hash1:p2:c1",
            text="Termination requires 30 days written notice from either party.",
            metadata={"document_name": "service_agreement.pdf", "file_hash": "hash1", "page": 2, "chunk_index": 1, "chunk_seq": 2},
        ),
        Chunk(
            id="hash1:p3:c1",
            text="Payment is due within 15 days of invoice receipt.",
            metadata={"document_name": "service_agreement.pdf", "file_hash": "hash1", "page": 3, "chunk_index": 1, "chunk_seq": 3},
        ),
    ]

    try:
        print("\n[1/7] Adding chunks...")
        added = add_chunks(chunks)
        assert added == 3, f"expected 3 chunks added, got {added}"
        print(f"   OK: added {added} chunks")

        # Milvus does not allow retrieving raw data for a Function output field
        # (sparse_vector is derived by the BM25 Function, not a directly stored field) —
        # client.query(output_fields=["sparse_vector"]) raises
        # "not allowed to retrieve raw data of field sparse_vector". So instead of checking
        # non-emptiness directly, we prove every row got a working sparse vector by searching
        # each chunk's one distinguishing keyword and confirming it ranks its own chunk first —
        # a chunk with an empty/missing sparse vector could not be found this way at all.
        print("\n[2/7] Sparse (BM25) search finds each chunk by its distinguishing keyword...")
        client = _get_collection_for_raw_query()
        for keyword, expected_snippet in [
            ("maintenance", "annual maintenance"),
            ("termination", "30 days"),
            ("invoice", "15 days"),
        ]:
            sparse_hits = client.search(
                collection_name=_TEST_COLLECTION,
                data=[keyword],
                anns_field="sparse_vector",
                limit=1,
                output_fields=["text"],
            )
            assert sparse_hits and sparse_hits[0], f"expected sparse search hits for {keyword!r}, got {sparse_hits}"
            top_text = sparse_hits[0][0]["entity"]["text"]
            assert expected_snippet in top_text, (
                f"expected keyword {keyword!r} to rank the chunk containing {expected_snippet!r} first, got: {top_text}"
            )
        print("   OK: all 3 chunks have working sparse vectors (each found by its own keyword)")

        print("\n[3/7] Checking chunk_seq is monotonic across pages, chunk_index resets per page...")
        seq_rows = _query_until(
            client,
            _TEST_COLLECTION,
            output_fields=["chunk_seq", "chunk_index", "page"],
            expected_rows=3,
        )
        seq_rows.sort(key=lambda r: r["page"])
        assert [r["chunk_seq"] for r in seq_rows] == [1, 2, 3], f"expected chunk_seq 1,2,3, got: {seq_rows}"
        assert all(r["chunk_index"] == 1 for r in seq_rows), f"expected chunk_index==1 on every row (each page has one chunk), got: {seq_rows}"
        print(f"   OK: chunk_seq strictly increasing (1,2,3), chunk_index resets to 1 per page")

        print("\n[4/7] indexed_file_hashes()...")
        hashes = indexed_file_hashes()
        assert "hash1" in hashes, f"expected hash1 in {hashes}"
        print(f"   OK: {hashes}")

        print("\n[5/7] query_chunks()...")
        results = query_chunks("What is the termination notice period?", top_k=2)
        assert len(results) > 0, "expected at least one result"
        assert any("30 days" in r["text"] for r in results), f"expected termination chunk in results: {results}"
        print(f"   OK: {len(results)} results, top text: {results[0]['text'][:60]}...")

        print("\n[6/7] list_documents()...")
        docs = list_documents()
        assert len(docs) == 1, f"expected 1 document, got {docs}"
        assert docs[0]["document_name"] == "service_agreement.pdf"
        assert docs[0]["chunks"] == 3
        print(f"   OK: {docs}")

        print("\n[7/7] delete_document()...")
        deleted = delete_document("hash1")
        assert deleted == 3, f"expected 3 deleted, got {deleted}"
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
