from __future__ import annotations

from fastapi import APIRouter

from app.config import TOP_K
from app.rag.generator import answer_question
from app.rag.retriever import search_sources
from app.schemas import ChatRequest, ChatResponse

# 对应 endpoint：POST /chat
# 1. 接收用户 question
# 2. 调用 retriever.retrieve(question)
# 3. 可选调用 reranker.rerank(question, chunks)
# 4. 调用 generator.generate_answer(question, chunks)
# 5. 返回 answer + sources

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    sources = search_sources(request.question, top_k=request.top_k or TOP_K)
    answer = answer_question(request.question, sources)
    return ChatResponse(answer=answer, sources=sources)
