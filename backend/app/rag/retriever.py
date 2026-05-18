from __future__ import annotations

from app.config import TOP_K
from app.schemas import Source
from app.rag.vector_store import get_collection


def search_sources(question: str, top_k: int = TOP_K) -> list[Source]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    result = collection.query(query_texts=[question], n_results=top_k)
    sources: list[Source] = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        sources.append(
            Source(
                text=text,
                document=metadata.get("document", "Unbekannt"),
                page=metadata.get("page", "?"),
                chunk=metadata.get("chunk", "?"),
                score=1 - float(distance) if distance is not None else None,
            )
        )

    return sources
