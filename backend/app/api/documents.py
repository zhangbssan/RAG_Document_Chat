from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.rag.vector_store import delete_document, list_documents
from app.schemas import DocumentStats
from app.utils.file_utils import delete_uploaded_file


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


@router.get("/list")
def document_list() -> dict:
    """
    Get full list of indexed documents.

    Returns:
        DocumentListResponse with total stats and document names.
    """
    try:
        documents = list_documents()

        return {
            "total_chunks": sum(doc.get("chunks", 0) for doc in documents),
            "total_documents": len(documents),
            "documents": documents,
        }

    except Exception as e:
        raise HTTPException(
            status_code=501,
            detail=f"Failed to get document list: {str(e)}",
        ) from e


@router.delete("/{file_hash}")
def document_delete(file_hash: str) -> dict:
    """
    Delete one indexed document by file hash.
    """
    try:
        documents = list_documents()
        document = next(
            (doc for doc in documents if doc.get("file_hash") == file_hash),
            None,
        )

        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {file_hash}",
            )

        document_name = str(document["document_name"])
        deleted_chunks = delete_document(file_hash)
        deleted_file = delete_uploaded_file(document_name)

        return {
            "message": "Document deleted.",
            "document_name": document_name,
            "file_hash": file_hash,
            "deleted_chunks": deleted_chunks,
            "deleted_file": deleted_file,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}",
        ) from e
