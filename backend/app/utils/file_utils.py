from __future__ import annotations

from pathlib import Path

from fastapi import UploadFile

from app.config import MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB, UPLOAD_DIR


def validate_pdf_filename(filename: str | None) -> str:
    if not filename:
        raise ValueError("Uploaded file has no filename.")

    safe_name = Path(filename).name
    if not safe_name:
        raise ValueError("Uploaded file has no filename.")

    if Path(safe_name).suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF files are supported: {safe_name}")

    return safe_name


async def save_upload_file(uploaded_file: UploadFile, upload_dir: Path) -> Path:
    file_name = validate_pdf_filename(uploaded_file.filename)
    validate_upload_size(uploaded_file, file_name)
    upload_dir.mkdir(parents=True, exist_ok=True)

    target_path = upload_dir / file_name
    target_path.write_bytes(await uploaded_file.read())
    return target_path


def validate_upload_size(uploaded_file: UploadFile, file_name: str) -> None:
    uploaded_file.file.seek(0, 2)
    file_size = uploaded_file.file.tell()
    uploaded_file.file.seek(0)

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(
            f"File too large: {file_name}. "
            f"Maximum allowed size is {MAX_UPLOAD_SIZE_MB} MB."
        )


def delete_uploaded_file(document_name: str) -> bool:
    safe_name = Path(document_name).name
    if not safe_name:
        return False

    file_path = UPLOAD_DIR / safe_name
    if not file_path.exists():
        return False

    file_path.unlink()
    return True
