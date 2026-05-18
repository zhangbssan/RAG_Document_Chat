from __future__ import annotations

import chromadb
from chromadb.config import Settings
from fastapi import UploadFile

from app.config import CHROMA_DIR, COLLECTION_NAME, UPLOAD_DIR
from app.rag.embeddings import LocalEmbeddingFunction
from app.rag.pdf_loader import build_chunks
from app.utils.file_utils import file_hash


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


async def index_pdf_uploads(files: list[UploadFile]) -> tuple[int, list[str]]:
    collection = get_collection()
    known_hashes = indexed_file_hashes()
    added_chunks = 0
    messages: list[str] = []

    for uploaded_file in files:
        data = await uploaded_file.read()
        digest = file_hash(data)
        file_name = uploaded_file.filename or "uploaded.pdf"
        target = UPLOAD_DIR / file_name
        target.write_bytes(data)

        if digest in known_hashes:
            messages.append(f"{file_name}: bereits indexiert")
            continue

        chunks = build_chunks(file_name, digest, target)
        if not chunks:
            messages.append(f"{file_name}: kein extrahierbarer Text gefunden")
            continue

        collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )
        added_chunks += len(chunks)
        known_hashes.add(digest)
        messages.append(f"{file_name}: {len(chunks)} Textabschnitte indexiert")

    return added_chunks, messages
