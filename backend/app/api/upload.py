from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import UPLOAD_DIR
from app.rag.chunker import build_chunks_from_pages
from app.rag.pdf_loader import pdf_extraction
from app.rag.vector_store import add_chunks, indexed_file_hashes
from app.schemas import UploadResponse
from app.utils.file_utils import save_upload_file, validate_pdf_filename

# Upload PDF files endpoint
# POST /upload - receive files, extract text, chunk, embed and index

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdfs(files: list[UploadFile] = File(...)) -> UploadResponse:
    """
    Upload and index PDF files.
    
    Args:
        files: List of PDF files to upload
        
    Returns:
        UploadResponse with number of chunks added and status messages
        
    Raises:
        HTTPException: If there's an error processing the files
    """
    try:
        known_hashes = indexed_file_hashes()
        added_chunks = 0
        messages: list[str] = []

        for uploaded_file in files:
            file_name = validate_pdf_filename(uploaded_file.filename)
            pdf_path = await save_upload_file(uploaded_file, UPLOAD_DIR)

            pages = pdf_extraction(file_name=file_name, pdf_path=pdf_path)
            file_hash = pages[0].file_hash if pages else None

            if file_hash and file_hash in known_hashes:
                messages.append(f"📄 {file_name}: Already indexed")
                continue

            chunks = build_chunks_from_pages(pages)
            if not chunks:
                messages.append(f"⚠️ {file_name}: Failed to extract text content")
                continue

            indexed_chunks = add_chunks(chunks)
            added_chunks += indexed_chunks

            if file_hash:
                known_hashes.add(file_hash)

            messages.append(f"✅ {file_name}: Indexed {indexed_chunks} text clauses")
    
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    
    return UploadResponse(added_chunks=added_chunks, messages=messages)

