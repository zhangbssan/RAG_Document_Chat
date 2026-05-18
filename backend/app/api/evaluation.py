from __future__ import annotations

from fastapi import APIRouter

from app.data.test_cases import test_cases
from app.rag.evaluator import evaluate_retrieval
from app.rag.retriever import search_sources

# 运行 hardcoded tests。
# 对应 endpoint：POST /evaluate
# 职责：1. 读取 TEST_CASES
#      2. 对每个 question 调用 retriever
#      3. 计算 retrieval score
#      4. 返回每个测试问题的 expected answer、retrieved sources、score
# UI show：Question
#          Expected Answer
#          Retrieved Sources
#          Retrieval Score


router = APIRouter()


@router.post("/evaluate")
async def run_evaluation() -> dict:
    """Run evaluation on test cases."""
    try:
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
            "results": results,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
