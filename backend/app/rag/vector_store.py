from __future__ import annotations
import os
from functools import lru_cache


os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings

from app.config import CHROMA_DIR, COLLECTION_NAME, UPLOAD_DIR
from app.rag.embeddings import LocalEmbeddingFunction
from app.rag.types import Chunk


@lru_cache(maxsize=1)
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
def query_chunks(query: str, top_k: int = 5) -> list[dict]:
    if not query.strip():
        raise ValueError("Query must not be empty.")

    collection = get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: list[dict] = []

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for chunk_id, text, metadata, distance in zip(ids, docs, metas, dists):
        retrieved.append(
            {
                "id": chunk_id,
                "text": text,
                "metadata": metadata,
                "distance": float(distance) if distance is not None else None,
                "score": 1 / (1 + float(distance)) if distance is not None else 0.0,
            }
        )

    return retrieved

def list_documents() -> list[dict]:
    """
    List indexed documents based on stored chunk metadata.
    Works even when the collection is empty.
    """
    result = get_collection().get(include=["metadatas"])
    metadatas = result.get("metadatas") or []

    documents: dict[str, dict] = {}

    for metadata in metadatas:
        if not metadata:
            continue

        document_name = metadata.get("document_name")
        file_hash = metadata.get("file_hash")
        page = metadata.get("page")

        if not document_name:
            continue

        if document_name not in documents:
            documents[document_name] = {
                "document_name": document_name,
                "file_hash": file_hash,
                "pages": set(),
                "chunks": 0,
            }

        if page is not None:
            documents[document_name]["pages"].add(page)

        documents[document_name]["chunks"] += 1

    return [
        {
            "document_name": doc["document_name"],
            "file_hash": doc["file_hash"],
            "pages": len(doc["pages"]),
            "chunks": doc["chunks"],
        }
        for doc in documents.values()
    ]
