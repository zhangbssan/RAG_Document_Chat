from __future__ import annotations

from app.schemas import Source


def evaluate_retrieval(test_case: dict, retrieved_chunks: list[Source]) -> dict:
    expected_document = test_case.get("expected_document")
    expected_page = test_case.get("expected_page")
    expected_keywords = test_case.get("expected_keywords", [])

    source_hit_score = _source_hit_score(
        retrieved_chunks=retrieved_chunks,
        expected_document=expected_document,
        expected_page=expected_page,
    )
    keyword_score = _keyword_score(
        retrieved_chunks=retrieved_chunks,
        expected_keywords=expected_keywords,
    )
    final_score = 0.7 * source_hit_score + 0.3 * keyword_score

    return {
        "source_hit_score": round(source_hit_score, 4),
        "keyword_score": round(keyword_score, 4),
        "final_score": round(final_score, 4),
    }


def _source_hit_score(
    retrieved_chunks: list[Source],
    expected_document: str | None,
    expected_page: int | str | None,
) -> float:
    if not expected_document or expected_page is None:
        return 0.0

    expected_page_text = str(expected_page)
    for chunk in retrieved_chunks:
        if chunk.document == expected_document and str(chunk.page) == expected_page_text:
            return 1.0

    return 0.0


def _keyword_score(
    retrieved_chunks: list[Source],
    expected_keywords: list[str],
) -> float:
    if not expected_keywords:
        return 0.0

    retrieved_text = " ".join(chunk.text for chunk in retrieved_chunks).lower()
    matched_count = sum(
        1
        for keyword in expected_keywords
        if keyword.lower() in retrieved_text
    )

    return matched_count / len(expected_keywords)
