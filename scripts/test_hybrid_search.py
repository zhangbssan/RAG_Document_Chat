#!/usr/bin/env python
"""Test hybrid_search.py::hybrid_search() end to end against live Milvus: dense+BM25
fusion, anchor selection, ±1 chunk_seq window fetch, interval merge, same-page
overlap stripping, cross-page no-strip, and citation links."""
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

_TEST_COLLECTION = "test_hybrid_search_collection"
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

from app.rag.hybrid_search import hybrid_search
from app.rag.types import Chunk
from app.rag.vector_store import add_chunks, get_collection

_OVERLAP_PHRASE = "trailing context marker phrase"


def test_hybrid_search() -> bool:
    print("=" * 70)
    print("HYBRID_SEARCH END-TO-END TEST")
    print("=" * 70)

    try:
        print("\n[1/4] hybrid_search() on an empty collection returns []...")
        cold = hybrid_search("anything", anchor_top_n=1)
        assert cold == [], f"expected [] on an empty collection, got {cold}"
        print("   OK")

        print("\n[2/4] Seeding a 3-chunk document (2 same-page + 1 cross-page) and a 1-chunk other document...")
        chunks = [
            Chunk(
                id="hash_a:p1:c1",
                text=f"Alpha section about gemstone sourcing ethics and mining audits {_OVERLAP_PHRASE}",
                metadata={"document_name": "doc_a.pdf", "file_hash": "hash_a", "page": 1, "chunk_index": 1, "chunk_seq": 1},
            ),
            Chunk(
                id="hash_a:p1:c2",
                text=f"{_OVERLAP_PHRASE} introduces the vermilion pigment grading section on the same page",
                metadata={"document_name": "doc_a.pdf", "file_hash": "hash_a", "page": 1, "chunk_index": 2, "chunk_seq": 2},
            ),
            Chunk(
                id="hash_a:p2:c1",
                text="Gamma section on page two about export tariffs and customs classification.",
                metadata={"document_name": "doc_a.pdf", "file_hash": "hash_a", "page": 2, "chunk_index": 1, "chunk_seq": 3},
            ),
            Chunk(
                id="hash_b:p1:c1",
                text="An entirely unrelated giraffe habitat brochure page.",
                metadata={"document_name": "doc_b.pdf", "file_hash": "hash_b", "page": 1, "chunk_index": 1, "chunk_seq": 1},
            ),
        ]
        added = add_chunks(chunks)
        assert added == 4, f"expected 4 chunks added, got {added}"
        print(f"   OK: added {added}")

        print("\n[3/4] Anchor in the middle of doc_a: window merges all 3 chunks, same-page overlap stripped once, cross-page break not stripped...")
        blocks = hybrid_search("vermilion", anchor_top_n=1, window=1)
        assert len(blocks) == 1, f"expected exactly 1 block, got {len(blocks)}: {blocks}"
        block = blocks[0]
        assert block["metadata"]["file_hash"] == "hash_a"
        assert block["metadata"]["page_start"] == 1 and block["metadata"]["page_end"] == 2
        assert block["metadata"]["chunk_seq_start"] == 1 and block["metadata"]["chunk_seq_end"] == 3
        assert block["text"].count(_OVERLAP_PHRASE) == 1, f"expected the same-page overlap stripped to one occurrence, got: {block['text']!r}"
        assert "\n\n" in block["text"], "expected a paragraph break between the page-1 and page-2 content"
        assert block["text"].rstrip().endswith("customs classification."), f"expected page-two text intact at the end, got: {block['text']!r}"
        assert block["link"] == "doc:hash_a#p1-2", f"got {block['link']!r}"
        print(f"   OK: 1 block, pages 1-2, chunk_seq 1-3, overlap stripped once, link={block['link']!r}")

        print("\n[4/4] A query unique to the other document returns a separate, single-chunk block...")
        blocks_b = hybrid_search("giraffe", anchor_top_n=1, window=1)
        assert len(blocks_b) == 1, f"expected exactly 1 block, got {len(blocks_b)}: {blocks_b}"
        block_b = blocks_b[0]
        assert block_b["metadata"]["file_hash"] == "hash_b"
        assert block_b["metadata"]["page_start"] == 1 and block_b["metadata"]["page_end"] == 1
        assert block_b["metadata"]["chunk_seq_start"] == 1 and block_b["metadata"]["chunk_seq_end"] == 1
        assert "giraffe" in block_b["text"]
        assert block_b["link"] == "doc:hash_b#p1", f"got {block_b['link']!r}"
        print(f"   OK: 1 separate block for doc_b, link={block_b['link']!r}")

        print("\nPASSED: hybrid_search() fuses, dedups, expands, merges, and links correctly end to end.")
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
    success = test_hybrid_search()
    sys.exit(0 if success else 1)
