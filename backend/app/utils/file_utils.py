from __future__ import annotations

from pathlib import Path

from fastapi import UploadFile


def validate_pdf_filename(filename: str | None) -> str:
    if not filename:
        raise ValueError("Uploaded file is missing a filename")

    safe_name = Path(filename).name
    if not safe_name:
        raise ValueError("Uploaded file is missing a filename")

    if Path(safe_name).suffix.lower() != ".pdf":
        raise ValueError(f"{safe_name}: only PDF files are supported")

    return safe_name


async def save_upload_file(uploaded_file: UploadFile, upload_dir: Path) -> Path:
    file_name = validate_pdf_filename(uploaded_file.filename)
    upload_dir.mkdir(parents=True, exist_ok=True)

    target_path = upload_dir / file_name
    target_path.write_bytes(await uploaded_file.read())
    return target_path
