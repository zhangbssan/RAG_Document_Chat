#!/usr/bin/env python
"""Test hybrid_search.py's pure fusion/merge/anchor functions in isolation — no Milvus needed."""
from __future__ import annotations

import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.rag.hybrid_search import (
    _anchor_to_row,
    _anchor_windows,
    _append_with_overlap_removed,
    _build_block,
    _citation_link,
    _merge_chunk_text,
    _merge_intervals,
    _rrf_fuse,
    _select_anchors,
)


def _hit(id_, text, file_hash, page, chunk_index, chunk_seq):
    return {
        "id": id_,
        "text": text,
        "metadata": {
            "document_name": "doc.pdf",
            "file_hash": file_hash,
            "page": page,
            "chunk_index": chunk_index,
            "chunk_seq": chunk_seq,
        },
    }


def test_hybrid_search_merge() -> bool:
    print("=" * 70)
    print("HYBRID_SEARCH PURE-FUNCTION TEST")
    print("=" * 70)

    try:
        print("\n[1/7] _rrf_fuse() dedups by id and fuses two ranked lists...")
        dense = [_hit(1, "a", "h1", 1, 1, 1), _hit(2, "b", "h1", 1, 2, 2)]
        sparse = [_hit(2, "b", "h1", 1, 2, 2), _hit(3, "c", "h1", 2, 1, 3)]
        fused = _rrf_fuse(dense, sparse)
        assert len(fused) == 3, f"expected 3 unique chunks (1,2,3), got {len(fused)}: {fused}"
        by_id = {e["id"]: e for e in fused}
        assert by_id[2]["dense_rank"] == 2 and by_id[2]["sparse_rank"] == 1, f"chunk 2 should carry both ranks: {by_id[2]}"
        assert by_id[1].get("sparse_rank") is None, "chunk 1 only came from dense"
        assert by_id[3].get("dense_rank") is None, "chunk 3 only came from sparse"
        assert fused[0]["id"] == 2, f"chunk 2 (in both lists) should score highest, got order: {[e['id'] for e in fused]}"
        print(f"   OK: 3 unique chunks, id=2 correctly carries both ranks and ranks first")

        print("\n[2/7] _select_anchors() takes the top N...")
        anchors = _select_anchors(fused, anchor_top_n=2)
        assert len(anchors) == 2
        print(f"   OK: {[a['id'] for a in anchors]}")

        print("\n[3/7] _anchor_windows() dedups overlapping ±1 windows across anchors...")
        anchors2 = [_hit(10, "x", "h1", 1, 1, 5), _hit(11, "y", "h1", 1, 2, 6)]
        needed = _anchor_windows(anchors2, window=1)
        assert needed == {"h1": {4, 5, 6, 7}}, f"expected union of {{4,5,6}} and {{5,6,7}}, got {needed}"
        print(f"   OK: {needed}")

        print("\n[4/7] _anchor_windows() never requests chunk_seq < 1...")
        edge = _anchor_windows([_hit(20, "z", "h1", 1, 1, 1)], window=1)
        assert edge == {"h1": {1, 2}}, f"expected {{1,2}} (0 excluded), got {edge}"
        print(f"   OK: {edge}")

        print("\n[5/7] _merge_intervals() splits non-adjacent runs, merges adjacent ones...")
        rows = [
            {"id": 1, "chunk_seq": 1}, {"id": 2, "chunk_seq": 2}, {"id": 3, "chunk_seq": 3},
            {"id": 4, "chunk_seq": 9}, {"id": 5, "chunk_seq": 10},
        ]
        blocks = _merge_intervals(rows)
        assert [[r["id"] for r in b] for b in blocks] == [[1, 2, 3], [4, 5]], f"got {blocks}"
        print(f"   OK: {[[r['id'] for r in b] for b in blocks]}")

        print("\n[6/7] Text merge: same-page overlap stripped once, cross-page not stripped...")
        prev = "the quarterly report covers revenue growth and cost containment measures"
        overlap = prev[-30:]
        next_same_page = overlap + " plus a forward-looking outlook section"
        merged = _append_with_overlap_removed(prev, next_same_page, max_overlap=180)
        assert merged == prev + " plus a forward-looking outlook section", f"expected overlap stripped, got: {merged!r}"

        rows_text = [
            {"text": prev, "page": 1},
            {"text": next_same_page, "page": 1},
            {"text": "Unrelated page-two heading with no shared text.", "page": 2},
        ]
        full = _merge_chunk_text(rows_text)
        assert full.count(overlap) == 1, f"same-page overlap must be stripped to one occurrence: {full!r}"
        assert "\n\n" in full, "expected a paragraph break between the page-1 and page-2 chunks"
        assert full.endswith("Unrelated page-two heading with no shared text."), f"got: {full!r}"

        no_overlap = _append_with_overlap_removed("first sentence.", "second sentence.", max_overlap=180)
        assert no_overlap == "first sentence. second sentence.", f"expected plain join fallback, got: {no_overlap!r}"
        print("   OK: same-page overlap stripped once, cross-page joined with a break and left intact, no-overlap case falls back to a plain join")

        print("\n[7/7] _citation_link() and _build_block() / _anchor_to_row() end to end...")
        assert _citation_link("h1", 3, 3) == "doc:h1#p3"
        assert _citation_link("h1", 3, 5) == "doc:h1#p3-5"

        context_rows = [
            {"id": 1, "text": "Alpha.", "document_name": "doc.pdf", "file_hash": "h1", "page": 1, "chunk_index": 1, "chunk_seq": 4},
            {"id": 2, "text": "Beta.", "document_name": "doc.pdf", "file_hash": "h1", "page": 1, "chunk_index": 2, "chunk_seq": 5},
        ]
        block = _build_block(context_rows, anchor_scores={1: 0.03})
        assert block["id"] == "h1:seq4-5"
        assert block["text"] == "Alpha. Beta."
        assert block["score"] == 0.03
        assert block["metadata"]["page_start"] == 1 and block["metadata"]["page_end"] == 1
        assert block["metadata"]["chunk_seq_start"] == 4 and block["metadata"]["chunk_seq_end"] == 5
        assert block["link"] == "doc:h1#p1"

        bare_anchor = _hit(99, "Standalone.", "h2", 7, 1, None)
        bare_row = _anchor_to_row(bare_anchor)
        assert bare_row["chunk_seq"] is None
        bare_block = _build_block([bare_row], anchor_scores={99: 0.01})
        assert bare_block["id"] == "h2:p7:c1", f"expected chunk_index-based id fallback, got {bare_block['id']}"
        assert bare_block["metadata"]["chunk_seq_start"] is None
        assert bare_block["link"] == "doc:h2#p7"
        print(f"   OK: {block['id']}, {bare_block['id']}")

        print("\nPASSED: hybrid_search.py pure functions behave correctly in isolation.")
        return True

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return False


if __name__ == "__main__":
    success = test_hybrid_search_merge()
    sys.exit(0 if success else 1)
