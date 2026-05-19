from __future__ import annotations

from fastapi import APIRouter

from app.data.test_cases import test_cases
from app.rag.evaluator import evaluate_retrieval
from app.rag.retriever import search_sources
from app.rag.vector_store import list_documents



router = APIRouter()


@router.post("/evaluate")
async def run_evaluation() -> dict:
    """Run evaluation on test cases."""
    try:
        required_documents = sorted(
            {
                test_case["expected_document"]
                for test_case in test_cases
                if test_case.get("expected_document")
            }
        )
        indexed_document_records = list_documents()
        indexed_documents = sorted(
            {
                str(document["document_name"])
                for document in indexed_document_records
                if document.get("document_name")
            }
        )
        missing_documents = sorted(set(required_documents) - set(indexed_documents))

        results = []

        for test_case in test_cases:
            retrieved_chunks = search_sources(test_case["question"])
            scores = evaluate_retrieval(test_case, retrieved_chunks)
            retrieved_sources = [
                {
                    "text": source.text,
                    "document": source.document,
                    "page": source.page,
                    "chunk": source.chunk,
                    "score": source.score,
                }
                for source in retrieved_chunks
            ]

            results.append(
                {
                    "id": test_case["id"],
                    "question": test_case["question"],
                    "expected_answer": test_case["expected_answer"],
                    "expected_document": test_case["expected_document"],
                    "expected_page": test_case["expected_page"],
                    "expected_keywords": test_case["expected_keywords"],
                    "retrieved_sources": retrieved_sources,
                    **scores,
                }
            )

        average_final_score = (
            sum(result["final_score"] for result in results) / len(results)
            if results
            else 0.0
        )

        return {
            "status": "evaluation completed",
            "tests_run": len(results),
            "average_final_score": round(average_final_score, 4),
            "required_documents": required_documents,
            "indexed_documents": indexed_documents,
            "missing_documents": missing_documents,
            "results": results,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
