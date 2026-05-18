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


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    added_chunks: int
    messages: list[str]


class DocumentStats(BaseModel):
    chunk_count: int
