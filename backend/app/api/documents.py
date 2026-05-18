from __future__ import annotations

from fastapi import APIRouter

from app.rag.vector_store import get_collection
from app.schemas import DocumentStats

# 列出已上传/已索引文档。
# GET /documents
#responsibility: - 从 vector store metadata 中统计 document_name
#                - 返回 document list
# show in UI: Uploaded documents:
#               - employee_handbook.pdf
#               - product_manual.pdf

router = APIRouter()


@router.get("/documents", response_model=DocumentStats)
def document_stats() -> DocumentStats:
    return DocumentStats(chunk_count=get_collection().count())
