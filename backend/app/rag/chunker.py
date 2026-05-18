from __future__ import annotations

import re

from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        window = text[start:end]

        if end < len(text):
            split_at = max(window.rfind("\n\n"), window.rfind(". "), window.rfind(" "))
            if split_at > chunk_size * 0.55:
                end = start + split_at + 1
                window = text[start:end]

        chunks.append(window.strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)

    return [chunk for chunk in chunks if chunk]
