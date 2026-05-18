from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.rag.vector_store import list_documents
from app.schemas import DocumentListResponse, DocumentStats


router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/stats", response_model=DocumentStats)
def document_stats() -> DocumentStats:
    """
    Get statistics about indexed documents.

    Returns:
        DocumentStats with chunk count, document count, and document names.
    """
    try:
        documents = list_documents()

        return DocumentStats(
            chunk_count=sum(doc.get("chunks", 0) for doc in documents),
            document_count=len(documents),
            documents=[doc["document_name"] for doc in documents],
        )

    except Exception as e:
        raise HTTPException(
            status_code=499,
            detail=f"Failed to get statistics: {str(e)}",
        ) from e


@router.get("/list", response_model=DocumentListResponse)
def document_list() -> DocumentListResponse:
    """
    Get full list of indexed documents.

    Returns:
        DocumentListResponse with total stats and document names.
    """
    try:
        documents = list_documents()

        return DocumentListResponse(
            total_chunks=sum(doc.get("chunks", 0) for doc in documents),
            total_documents=len(documents),
            documents=[doc["document_name"] for doc in documents],
        )

    except Exception as e:
        raise HTTPException(
            status_code=501,
            detail=f"Failed to get document list: {str(e)}",
        ) from e