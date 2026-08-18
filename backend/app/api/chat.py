from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.agent.runtime import get_agent_runtime
from app.config import OPENAI_API_KEY, TOP_K
from app.rag.generator import answer_question_stream, fallback_answer
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
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Run one turn of the persistent, single-tool Agent."""
    try:
        if not payload.question or not payload.question.strip():
            raise ValueError("Question cannot be empty")

        api_key = payload.openai_api_key or OPENAI_API_KEY
        if not api_key:
            sources = search_sources(
                question=payload.question,
                top_k=payload.top_k or TOP_K,
            )
            return ChatResponse(answer=fallback_answer(sources), sources=sources)

        runtime = get_agent_runtime(request)
        result = runtime.run_chat(
            question=payload.question,
            conversation_id=payload.conversation_id,
            user_context={"session_id": payload.conversation_id},
            api_key=api_key,
            top_k=payload.top_k or TOP_K,
        )
        runtime.conversations.ensure(payload.conversation_id, payload.question)
        runtime.conversations.touch(payload.conversation_id)
        return ChatResponse(answer=result["answer"], sources=result["sources"])

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Request processing failed: {str(exc)}") from exc


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest, request: Request):
    """Stream one persistent Agent turn as newline-delimited JSON."""
    try:
        if not payload.question or not payload.question.strip():
            raise ValueError("Question cannot be empty")

        api_key = payload.openai_api_key or OPENAI_API_KEY
        if api_key:
            runtime = get_agent_runtime(request)

            def generate_agent():
                completed = False
                for event in runtime.stream_chat(
                    question=payload.question,
                    conversation_id=payload.conversation_id,
                    user_context={"session_id": payload.conversation_id},
                    api_key=api_key,
                    top_k=payload.top_k or TOP_K,
                ):
                    if event.get("type") == "sources":
                        event = {
                            "type": "sources",
                            "data": [_source_to_dict(source) for source in event.get("data", [])],
                        }
                    if event.get("type") == "done":
                        completed = True
                    yield _json_event(event)
                if completed:
                    runtime.conversations.ensure(payload.conversation_id, payload.question)
                    runtime.conversations.touch(payload.conversation_id)

            return StreamingResponse(generate_agent(), media_type="application/x-ndjson")

        # No LLM means no Agent memory; preserve the stateless extractive stream.
        sources = search_sources(
            question=payload.question,
            top_k=payload.top_k or TOP_K,
        )

        def generate():
            yield _json_event(
                {
                    "type": "notice",
                    "data": "No LLM API key is configured; this fallback answer is not saved to conversation memory.",
                }
            )
            yield _json_event(
                {
                    "type": "sources",
                    "data": [_source_to_dict(source) for source in sources],
                }
            )
            for text_chunk in answer_question_stream(
                question=payload.question,
                sources=sources,
                api_key=payload.openai_api_key,
            ):
                yield _json_event({"type": "token", "data": text_chunk})
            yield _json_event({"type": "done"})

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stream processing failed: {str(exc)}") from exc
