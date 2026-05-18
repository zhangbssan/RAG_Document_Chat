from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.rag.vector_store import index_pdf_uploads
from app.schemas import UploadResponse


router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdfs(files: list[UploadFile] = File(...)) -> UploadResponse:
    try:
        added_chunks, messages = await index_pdf_uploads(files)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UploadResponse(added_chunks=added_chunks, messages=messages)
