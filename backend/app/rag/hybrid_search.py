from __future__ import annotations

from typing import Any

from app.config import CHUNK_OVERLAP

RRF_K = 60
_MIN_TEXT_OVERLAP_CHARS = 20


def _rrf_fuse(dense_hits: list[dict[str, Any]], sparse_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge dense + sparse hit lists, dedup by chunk id, score with Reciprocal Rank Fusion."""
    by_id: dict[Any, dict[str, Any]] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        entry = by_id.setdefault(hit["id"], dict(hit))
        entry["dense_rank"] = rank

    for rank, hit in enumerate(sparse_hits, start=1):
        entry = by_id.setdefault(hit["id"], dict(hit))
        entry["sparse_rank"] = rank

    fused: list[dict[str, Any]] = []
    for entry in by_id.values():
        score = 0.0
        if entry.get("dense_rank") is not None:
            score += 1.0 / (RRF_K + entry["dense_rank"])
        if entry.get("sparse_rank") is not None:
            score += 1.0 / (RRF_K + entry["sparse_rank"])
        entry["rrf_score"] = score
        fused.append(entry)

    fused.sort(key=lambda e: e["rrf_score"], reverse=True)
    return fused


def _select_anchors(fused: list[dict[str, Any]], anchor_top_n: int) -> list[dict[str, Any]]:
    return fused[:anchor_top_n]


def _anchor_windows(anchors: list[dict[str, Any]], window: int) -> dict[str, set[int]]:
    """Per document, the set of chunk_seq values needed to build every anchor's
    ±window view. Only anchors that carry a chunk_seq should be passed in —
    hybrid_search() filters those out first."""
    needed: dict[str, set[int]] = {}
    for anchor in anchors:
        metadata = anchor["metadata"]
        file_hash = metadata["file_hash"]
        anchor_seq = metadata["chunk_seq"]
        seqs = needed.setdefault(file_hash, set())
        for offset in range(-window, window + 1):
            seq = anchor_seq + offset
            if seq >= 1:
                seqs.add(seq)
    return needed


def _merge_intervals(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group chunk_seq-sorted rows of one document into contiguous runs — two
    rows merge into the same block when their chunk_seq differ by 1 or less."""
    if not rows:
        return []

    blocks: list[list[dict[str, Any]]] = [[rows[0]]]
    for row in rows[1:]:
        if row["chunk_seq"] - blocks[-1][-1]["chunk_seq"] <= 1:
            blocks[-1].append(row)
        else:
            blocks.append([row])
    return blocks


def _append_with_overlap_removed(prev_text: str, next_text: str, max_overlap: int = CHUNK_OVERLAP) -> str:
    """Strip the duplicated substring chunk_text() leaves between two adjacent
    same-page chunks (up to max_overlap characters) before concatenating. Falls
    back to a plain space-joined concatenation when no real overlap is found."""
    search_len = min(max_overlap, len(prev_text), len(next_text))
    for size in range(search_len, _MIN_TEXT_OVERLAP_CHARS - 1, -1):
        if prev_text[-size:] == next_text[:size]:
            return prev_text + next_text[size:]
    return f"{prev_text} {next_text}"


def _merge_chunk_text(rows: list[dict[str, Any]]) -> str:
    """Concatenate a contiguous run of chunks into one block's text. Adjacent
    chunks that share a page get overlap-stripped; adjacent chunks on
    different pages are joined with a paragraph break and never overlap-
    stripped, since chunk_text() runs per-page and leaves no shared substring
    across a page boundary."""
    merged = rows[0]["text"]
    for previous, current in zip(rows, rows[1:]):
        if previous["page"] == current["page"]:
            merged = _append_with_overlap_removed(merged, current["text"])
        else:
            merged = f"{merged}\n\n{current['text']}"
    return merged


def _citation_link(file_hash: str, page_start: int | None, page_end: int | None) -> str:
    """The reference handed back to the agent/user to point at the exact source
    span — same idea as session_search_tool.py's _session_link(), resolving to
    a document + page range instead of a chat session."""
    if page_start is None:
        return f"doc:{file_hash}"
    if page_start == page_end:
        return f"doc:{file_hash}#p{page_start}"
    return f"doc:{file_hash}#p{page_start}-{page_end}"


def _build_block(rows: list[dict[str, Any]], anchor_scores: dict[Any, float]) -> dict[str, Any]:
    """Assemble one final context block from a contiguous run of chunk rows."""
    pages = sorted({row["page"] for row in rows if row.get("page") is not None})
    chunk_indexes = [row["chunk_index"] for row in rows]
    chunk_seqs = [row["chunk_seq"] for row in rows if row.get("chunk_seq") is not None]
    document_name = rows[0]["document_name"]
    file_hash = rows[0]["file_hash"]
    page_start = pages[0] if pages else None
    page_end = pages[-1] if pages else None
    chunk_seq_start = chunk_seqs[0] if chunk_seqs else None
    chunk_seq_end = chunk_seqs[-1] if chunk_seqs else None

    block_score = max(
        (anchor_scores[row["id"]] for row in rows if row["id"] in anchor_scores),
        default=0.0,
    )
    block_id = (
        f"{file_hash}:seq{chunk_seq_start}-{chunk_seq_end}"
        if chunk_seq_start is not None
        else f"{file_hash}:p{page_start}:c{rows[0].get('chunk_index')}"
    )

    return {
        "id": block_id,
        "text": _merge_chunk_text(rows),
        "score": block_score,
        "metadata": {
            "document_name": document_name,
            "file_hash": file_hash,
            "page_start": page_start,
            "page_end": page_end,
            "pages": pages,
            "chunk_seq_start": chunk_seq_start,
            "chunk_seq_end": chunk_seq_end,
            "chunk_indexes": chunk_indexes,
        },
        "link": _citation_link(file_hash, page_start, page_end),
    }


def _anchor_to_row(anchor: dict[str, Any]) -> dict[str, Any]:
    """Adapt a nested search-hit (id/text/metadata{...}) into the flat row shape
    _build_block() expects — used for anchors with no chunk_seq: no window can
    be built, so the anchor becomes its own single-chunk block."""
    metadata = anchor.get("metadata", {})
    return {
        "id": anchor.get("id"),
        "text": anchor.get("text"),
        "document_name": metadata.get("document_name"),
        "file_hash": metadata.get("file_hash"),
        "page": metadata.get("page"),
        "chunk_index": metadata.get("chunk_index"),
        "chunk_seq": metadata.get("chunk_seq"),
    }
