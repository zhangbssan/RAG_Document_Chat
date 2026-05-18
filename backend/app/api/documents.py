from __future__ import annotations

from fastapi import APIRouter

from app.rag.vector_store import get_collection
from app.schemas import DocumentStats


router = APIRouter()


@router.get("/documents/stats", response_model=DocumentStats)
def document_stats() -> DocumentStats:
    return DocumentStats(chunk_count=get_collection().count())
