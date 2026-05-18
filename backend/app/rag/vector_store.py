from __future__ import annotations

import chromadb
from chromadb.config import Settings

from app.config import CHROMA_DIR, COLLECTION_NAME, UPLOAD_DIR
from app.rag.embeddings import LocalEmbeddingFunction
from app.rag.types import Chunk


def get_collection():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=LocalEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def indexed_file_hashes() -> set[str]:
    result = get_collection().get(include=["metadatas"])
    hashes: set[str] = set()
    for metadata in result.get("metadatas") or []:
        if metadata and metadata.get("file_hash"):
            hashes.add(str(metadata["file_hash"]))
    return hashes


def add_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0

    get_collection().upsert(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )
    return len(chunks)
