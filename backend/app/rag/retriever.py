from __future__ import annotations

from app.config import TOP_K
from app.rag.hybrid_search import hybrid_search
from app.schemas import Source


def retrieve_chunks(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Hybrid (dense + BM25) retrieval, RRF-fused, expanded into anchor-centered
    context blocks. See app.rag.hybrid_search.hybrid_search() and
    docs/superpowers/specs/2026-08-16-hybrid-query-retrieval-design.md.

    Args:
        query: User query string.
        top_k: Candidate breadth for both the dense and sparse searches.

    Returns:
        List of context blocks (not raw chunks) with metadata, text, and score.
    """
    return hybrid_search(query, dense_top_k=top_k, sparse_top_k=top_k)


def search_sources(question: str, top_k: int = TOP_K) -> list[Source]:
    """
    Search and return sources for the chat response.
    """
    blocks = retrieve_chunks(question, top_k=top_k)

    sources: list[Source] = []
    for block in blocks:
        metadata = block.get("metadata", {})
        sources.append(
            Source(
                text=block.get("text", ""),
                document=metadata.get("document_name", "Unknown"),
                page=metadata.get("page_start", "?"),
                pages=metadata.get("pages") or None,
                chunk=_chunk_label(metadata),
                score=block.get("score"),
                link=block.get("link"),
            )
        )

    return sources


def _chunk_label(metadata: dict) -> int | str:
    start = metadata.get("chunk_seq_start")
    end = metadata.get("chunk_seq_end")
    if start is None:
        indexes = metadata.get("chunk_indexes") or []
        return indexes[0] if indexes else "?"
    return start if start == end else f"{start}-{end}"
