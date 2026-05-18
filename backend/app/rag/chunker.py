from __future__ import annotations

import re

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.rag.types import Chunk, PageText


def normalize_text(text: str) -> str:
    """
    Normalize text extracted from PDFs.

    Keeps paragraph structure while removing noisy whitespace.
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_split_point(window: str, min_split: int) -> int:
    """
    Find a natural split point inside the current text window.

    Priority:
    1. paragraph break
    2. line break
    3. sentence boundary
    4. word boundary

    If no suitable boundary is found, fall back to the end of the window.
    """
    separators = ["\n\n", "\n", ". ", " "]

    for separator in separators:
        split_at = window.rfind(separator)

        if split_at > min_split:
            return split_at + len(separator)

    return len(window)


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks.

    The splitter first creates a character-length window and then tries to
    move the split point to a natural text boundary. Overlap helps preserve
    context across neighboring chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must not be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = normalize_text(text)

    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        window = text[start:end]

        if end < len(text):
            split_at = find_split_point(
                window=window,
                min_split=int(chunk_size * 0.55),
            )
            end = start + split_at
            window = text[start:end]

        chunk = window.strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def build_chunks_from_pages(
    pages: list[PageText],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Convert page-level PDF text into source-aware chunks.

    Input:
        list[PageText] from pdf_loader.py

    Output:
        list[Chunk] for embedding and vector storage

    Each chunk keeps metadata for source citation:
    document name, file hash, page number, and chunk index.
    """
    chunks: list[Chunk] = []

    for page in pages:
        page_chunks = chunk_text(
            text=page.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_index, chunk_content in enumerate(page_chunks, start=1):
            chunk_id = f"{page.file_hash}:p{page.page}:c{chunk_index}"

            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=chunk_content,
                    metadata={
                        "document_name": page.document_name,
                        "file_hash": page.file_hash,
                        "page": page.page,
                        "chunk_index": chunk_index,
                    },
                )
            )

    return chunks