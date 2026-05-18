from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json

from app.config import TOP_K
from app.rag.generator import answer_question, answer_question_stream
from app.rag.retriever import search_sources
from app.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Standard chat endpoint - returns complete answer with sources.
    
    Args:
        request: ChatRequest with question and optional top_k
        
    Returns:
        ChatResponse with answer and sources
        
    Raises:
        HTTPException: If question is empty or other errors occur
    """
    try:
        if not request.question or not request.question.strip():
            raise ValueError("Question cannot be empty")
        
        # Retrieve relevant sources
        sources = search_sources(
            request.question, 
            top_k=request.top_k or TOP_K
        )
        
        # Generate answer
        answer = answer_question(request.question, sources)
        
        return ChatResponse(answer=answer, sources=sources)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}") from e


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint - streams the answer token by token.
    
    Returns:
        Server-Sent Events stream with:
        - "sources": Initial source list (JSON)
        - "content": Answer content (text chunks)
        - "done": Final marker
    """
    try:
        if not request.question or not request.question.strip():
            raise ValueError("Question cannot be empty")
        
        # Retrieve relevant sources
        sources = search_sources(
            request.question,
            top_k=request.top_k or TOP_K
        )
        
        # Create streaming response
        def generate():
            # First, send the metadata with sources
            sources_json = [
                {
                    "text": source.text,
                    "document": source.document,
                    "page": source.page,
                    "chunk": source.chunk,
                    "score": source.score,
                }
                for source in sources
            ]
            
            yield f"data: {json.dumps({'type': 'sources', 'data': sources_json})}\n\n"
            
            # Then stream the answer
            for text_chunk in answer_question_stream(request.question, sources):
                yield f"data: {json.dumps({'type': 'content', 'data': text_chunk})}\n\n"
            
            # Send done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stream processing failed: {str(e)}") from e

