from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.rag.chunker import chunk_text, normalize_text


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, str | int]


def build_chunks(file_name: str, file_digest: str, pdf_path: Path) -> list[Chunk]:
    reader = PdfReader(str(pdf_path))
    chunks: list[Chunk] = []

    for page_index, page in enumerate(reader.pages, start=1):
        page_text = normalize_text(page.extract_text() or "")
        for chunk_index, text in enumerate(chunk_text(page_text), start=1):
            chunks.append(
                Chunk(
                    id=f"{file_digest}:{page_index}:{chunk_index}",
                    text=text,
                    metadata={
                        "document": file_name,
                        "file_hash": file_digest,
                        "page": page_index,
                        "chunk": chunk_index,
                    },
                )
            )

    return chunks
