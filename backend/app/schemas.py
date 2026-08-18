from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class Source(BaseModel):
    text: str
    document: str
    page: int | str
    chunk: int | str
    score: float | None = None
    pages: list[int] | None = None
    link: str | None = None


class ChatRequest(BaseModel):
    question: str
    conversation_id: str = Field(min_length=1, max_length=128)
    top_k: int | None = Field(default=None, ge=1, le=20)
    openai_api_key: str | None = None

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("conversation_id cannot be empty")
        try:
            return str(UUID(normalized))
        except ValueError as exc:
            raise ValueError("conversation_id must be a valid UUID") from exc


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]


class ConversationMessage(BaseModel):
    role: str
    content: str
    sources: list[Source] = Field(default_factory=list)


class ConversationHistoryResponse(BaseModel):
    conversation: ConversationSummary
    messages: list[ConversationMessage]


class ConversationRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("title cannot be empty")
        return normalized


class ConversationDeleteResponse(BaseModel):
    conversation_id: str
    deleted: bool


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

