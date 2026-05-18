from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.rag.vector_store import index_pdf_uploads
from app.schemas import UploadResponse

# 处理 PDF 上传。

# 对应 endpoint：POST /upload

# 1. 接收多个 PDF 文件
# 2. 检查是否是 PDF
# 3. 保存到 data/uploads/
# 4. 调用 pdf_loader.extract_pages()
# 5. 调用 chunker.create_chunks()
# 6. 调用 vector_store.add_chunks()
# 7. 返回处理结果

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdfs(files: list[UploadFile] = File(...)) -> UploadResponse:
    try:
        added_chunks, messages = await index_pdf_uploads(files)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UploadResponse(added_chunks=added_chunks, messages=messages)
