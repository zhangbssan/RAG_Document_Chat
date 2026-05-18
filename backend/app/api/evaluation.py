from __future__ import annotations

from fastapi import APIRouter

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
async def run_evaluation() -> dict[str, str]:
    """Run evaluation on test cases."""
    try:
        # Placeholder for evaluation logic
        # TODO: Implement evaluation against test cases
        return {"status": "evaluation completed", "tests_run": 0}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
