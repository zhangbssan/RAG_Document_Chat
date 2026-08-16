from __future__ import annotations

import re

from app.schemas import Source


def evaluate_retrieval(test_case: dict, retrieved_chunks: list[Source]) -> dict:
    expected_document = test_case.get("expected_document")
    expected_page = test_case.get("expected_page")
    expected_keywords = test_case.get("expected_keywords", [])

    document_hit_score = _document_hit_score(
        retrieved_chunks=retrieved_chunks,
        expected_document=expected_document,
    )
    page_hit_score = _page_hit_score(
        retrieved_chunks=retrieved_chunks,
        expected_document=expected_document,
        expected_page=expected_page,
    )
    keyword_score = _keyword_score(
        retrieved_chunks=retrieved_chunks,
        expected_keywords=expected_keywords,
    )
    final_score = (
        0.5 * document_hit_score
        + 0.2 * page_hit_score
        + 0.3 * keyword_score
    )

    return {
        "document_hit_score": round(document_hit_score, 4),
        "page_hit_score": round(page_hit_score, 4),
        "keyword_score": round(keyword_score, 4),
        "final_score": round(final_score, 4),
    }


def _document_hit_score(
    retrieved_chunks: list[Source],
    expected_document: str | None,
) -> float:
    if not expected_document:
        return 0.0

    for chunk in retrieved_chunks:
        if chunk.document == expected_document:
            return 1.0

    return 0.0


def _page_hit_score(
    retrieved_chunks: list[Source],
    expected_document: str | None,
    expected_page: int | str | None,
) -> float:
    if not expected_document:
        return 0.0

    if expected_page is None:
        return _document_hit_score(retrieved_chunks, expected_document)

    expected_page_text = str(expected_page)
    for chunk in retrieved_chunks:
        if chunk.document != expected_document:
            continue
        pages = chunk.pages if chunk.pages else [chunk.page]
        if any(str(page) == expected_page_text for page in pages):
            return 1.0

    return 0.0


def _keyword_score(
    retrieved_chunks: list[Source],
    expected_keywords: list[str],
) -> float:
    if not expected_keywords:
        return 0.0

    retrieved_text = _normalize_text(" ".join(chunk.text for chunk in retrieved_chunks))
    matched_count = sum(
        1
        for keyword in expected_keywords
        if _normalize_text(keyword) in retrieved_text
    )

    return matched_count / len(expected_keywords)


def _normalize_text(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[-_/]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()
