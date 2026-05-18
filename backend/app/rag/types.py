from __future__ import annotations

from pydantic import BaseModel

class PageText(BaseModel):
    document_name: str
    file_hash: str
    page: int
    text: str

class Chunk(BaseModel):
    id: str
    text: str
    metadata: dict[str, str | int]