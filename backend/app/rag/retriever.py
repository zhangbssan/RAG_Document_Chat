from __future__ import annotations

from app.config import TOP_K
from app.schemas import Source
from app.rag.embeddings import VectorStore, embed_query


# Global vector store instance
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def retrieve_chunks(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve similar chunks from the vector store.
    
    Args:
        query: User query string
        top_k: Number of chunks to retrieve
        
    Returns:
        List of retrieved chunks with metadata and scores
    """
    vector_store = get_vector_store()
    retrieved = vector_store.query_chunks(query, top_k=top_k)
    return retrieved


def search_sources(question: str, top_k: int = TOP_K) -> list[Source]:
    """Search and return sources (for backward compatibility)."""
    retrieved = retrieve_chunks(question, top_k=top_k)
    
    sources: list[Source] = []
    for chunk in retrieved:
        metadata = chunk.get("metadata", {})
        sources.append(
            Source(
                text=chunk.get("text", ""),
                document=metadata.get("document_name", "Unknown"),
                page=metadata.get("page", "?"),
                chunk=metadata.get("chunk_index", "?"),
                score=chunk.get("score"),
            )
        )
    
    return sources
