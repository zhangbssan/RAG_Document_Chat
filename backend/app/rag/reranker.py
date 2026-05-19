from __future__ import annotations

import re
from math import sqrt
from typing import Any


RRF_K = 60
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    rerank_top_k: int,
) -> list[dict[str, Any]]:
    """
    Rerank vector search results with RRF over dense and lexical ranks.
    """
    if rerank_top_k <= 0:
        return []

    query_terms = _keywords(query)
    candidates: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks):
        text_terms = _keywords(str(chunk.get("text", "")))
        overlap = query_terms & text_terms
        if query_terms and text_terms:
            lexical_score = len(overlap) / sqrt(len(query_terms) * len(text_terms))
        else:
            lexical_score = 0.0
        vector_score = float(chunk.get("score") or 0.0)

        candidate = dict(chunk)
        candidate["dense_rank"] = index + 1
        candidate["vector_score"] = vector_score
        candidate["keyword_overlap"] = len(overlap)
        candidate["matched_keywords"] = sorted(overlap)
        candidate["lexical_score"] = lexical_score

        candidates.append(candidate)

    lexical_order = sorted(
        candidates,
        key=lambda chunk: (
            chunk["keyword_overlap"],
            chunk["lexical_score"],
            chunk["vector_score"],
            -chunk["dense_rank"],
        ),
        reverse=True,
    )

    for lexical_rank, candidate in enumerate(lexical_order, start=1):
        candidate["lexical_rank"] = lexical_rank
        rrf_score = _rrf_score(
            dense_rank=candidate["dense_rank"],
            lexical_rank=lexical_rank,
        )
        candidate["rerank_score"] = rrf_score
        candidate["score"] = rrf_score

    reranked = sorted(
        candidates,
        key=lambda chunk: (
            chunk["rerank_score"],
            chunk["keyword_overlap"],
            chunk["vector_score"],
            -chunk["dense_rank"],
        ),
        reverse=True,
    )
    return reranked[:rerank_top_k]


def _rrf_score(dense_rank: int, lexical_rank: int) -> float:
    return (1 / (RRF_K + dense_rank)) + (1 / (RRF_K + lexical_rank))


def _keywords(text: str) -> set[str]:
    terms: set[str] = set()

    for token in TOKEN_PATTERN.findall(text.lower()):
        if token in STOPWORDS:
            continue
        if token.isascii() and token.isalpha() and len(token) == 1:
            continue
        terms.add(token)

    return terms
