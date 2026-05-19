from __future__ import annotations

from pydantic import BaseModel


class Source(BaseModel):
    text: str
    document: str
    page: int | str
    chunk: int | str
    score: float | None = None


class ChatRequest(BaseModel):
    question: str
    top_k: int | None = None
    openai_api_key: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    added_chunks: int
    messages: list[str]


class DocumentStats(BaseModel):
    """Statistics about indexed documents."""
    chunk_count: int
    document_count: int = 0
    documents: list[str] = []


class DocumentListResponse(BaseModel):
    """Response for document list request."""
    total_chunks: int
    total_documents: int
    documents: list[str]



