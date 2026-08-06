from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json

from app.agent.chat_agent import run_agent_chat
from app.config import TOP_K
from app.rag.generator import answer_question_stream
from app.rag.retriever import search_sources
from app.schemas import ChatRequest, ChatResponse, Source

router = APIRouter()


def _json_event(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _source_to_dict(source: Source) -> dict:
    if hasattr(source, "model_dump"):
        return source.model_dump()
    return source.dict()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Agent-backed chat endpoint - the agent decides whether to search uploaded
    documents (search_uploaded_docs tool) or answer directly.

    Args:
        request: ChatRequest with question and optional top_k, openai_api_key

    Returns:
        ChatResponse with answer and sources (empty if the tool wasn't used)

    Raises:
        HTTPException: If question is empty or other errors occur
    """
    try:
        if not request.question or not request.question.strip():
            raise ValueError("Question cannot be empty")

        result = run_agent_chat(
            question=request.question,
            user_context={},
            api_key=request.openai_api_key,
            top_k=request.top_k or TOP_K,
        )

        return ChatResponse(answer=result["answer"], sources=result["sources"])

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}") from e


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint - unchanged, still uses direct retrieval + generation
    (not the agent). Streams sources first, then answer tokens.

    Returns:
        NDJSON stream with sources, token chunks, and a done marker.
    """
    try:
        if not request.question or not request.question.strip():
            raise ValueError("Question cannot be empty")

        sources = search_sources(
            question=request.question,
            top_k=request.top_k or TOP_K
        )

        def generate():
            yield _json_event({
                "type": "sources",
                "data": [_source_to_dict(source) for source in sources],
            })

            for text_chunk in answer_question_stream(
                question=request.question,
                sources=sources,
                api_key=request.openai_api_key,
            ):
                yield _json_event({"type": "token", "data": text_chunk})

            yield _json_event({"type": "done"})

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stream processing failed: {str(e)}") from e