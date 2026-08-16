from __future__ import annotations

from pydantic import BaseModel

from langchain_core.tools import BaseTool, tool

from app.rag.hybrid_search import hybrid_search


class UserContext(BaseModel):
    user_id: str | None = None
    department_id: str | None = None
    session_id: str | None = None


def _search_uploaded_docs_impl(query: str, user_context: UserContext, top_k: int = 5) -> dict:
    """Core search logic. Hybrid (dense + BM25) retrieval, RRF-fused, and expanded
    into anchor-centered context blocks — see app.rag.hybrid_search.hybrid_search().
    user_context is accepted for a future permission/scoping layer but is NOT used
    to filter results yet — search always queries the single shared
    realtime_pdf_collection, per the current storage-layer design."""
    try:
        blocks = hybrid_search(query, dense_top_k=top_k, sparse_top_k=top_k)
    except Exception as e:
        return {
            "status": "error",
            "content": "",
            "citations": [],
            "metadata": {"tool": "search_uploaded_docs", "result_count": 0},
            "error": str(e),
        }

    citations = []
    content_parts = []
    for block in blocks:
        metadata = block.get("metadata", {})
        text = block.get("text", "")

        citations.append(
            {
                "source_id": metadata.get("file_hash"),
                "source_name": metadata.get("document_name"),
                "page": metadata.get("page_start"),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "pages": metadata.get("pages"),
                "chunk_id": block.get("id"),
                "link": block.get("link"),
                "text": text,
            }
        )
        content_parts.append(text)

    return {
        "status": "success",
        "content": "\n\n---\n\n".join(content_parts),
        "citations": citations,
        "metadata": {"tool": "search_uploaded_docs", "result_count": len(citations)},
        "error": None,
    }


def make_search_tool(user_context: UserContext, top_k: int = 5) -> BaseTool:
    """Build a search_uploaded_docs tool bound to this request's real user_context.

    Only `query` is exposed in the tool's LLM-visible schema — user_context is
    captured via closure, never something the model fills in itself.
    """

    @tool
    def search_uploaded_docs(query: str) -> dict:
        """Search the user's uploaded PDF documents for passages relevant to the query.
        Use this whenever the question could be answered from documents the user has uploaded."""
        return _search_uploaded_docs_impl(query, user_context, top_k=top_k)

    return search_uploaded_docs
