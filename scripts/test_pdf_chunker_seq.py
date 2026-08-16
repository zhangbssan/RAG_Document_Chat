#!/usr/bin/env python
"""Test chunker.py: chunk_seq is monotonic per-document, chunk_index resets per page."""
from __future__ import annotations

import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.rag.chunker import build_chunks_from_pages
from app.rag.types import PageText


def test_chunk_seq() -> bool:
    print("=" * 70)
    print("CHUNKER chunk_seq TEST")
    print("=" * 70)

    # Each page's text is short enough to produce exactly one chunk per page,
    # so this fixture directly exercises the 3-page, one-chunk-per-page case.
    pages = [
        PageText(document_name="doc.pdf", file_hash="h1", page=1, text="Page one content."),
        PageText(document_name="doc.pdf", file_hash="h1", page=2, text="Page two content."),
        PageText(document_name="doc.pdf", file_hash="h1", page=3, text="Page three content."),
    ]

    try:
        print("\n[1/2] Building chunks from 3 pages...")
        chunks = build_chunks_from_pages(pages, chunk_size=950, overlap=180)
        assert len(chunks) == 3, f"expected 3 chunks (one per page), got {len(chunks)}"
        print(f"   OK: {len(chunks)} chunks")

        print("\n[2/2] Checking chunk_seq is 1..N, chunk_index resets to 1 per page...")
        seqs = [c.metadata["chunk_seq"] for c in chunks]
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert seqs == [1, 2, 3], f"expected chunk_seq [1, 2, 3], got {seqs}"
        assert indices == [1, 1, 1], f"expected chunk_index [1, 1, 1] (resets each page), got {indices}"
        print(f"   OK: chunk_seq={seqs}, chunk_index={indices}")

        print("\nPASSED: chunk_seq is monotonic per document, chunk_index still resets per page.")
        return True

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return False


if __name__ == "__main__":
    success = test_chunk_seq()
    sys.exit(0 if success else 1)