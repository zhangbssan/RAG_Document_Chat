from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load the embedding model once and reuse it during the backend process."""
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Create normalized dense embeddings for a list of texts."""
    if not texts:
        return []

    model = load_embedding_model(model_name=EMBEDDING_MODEL)

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Create a normalized dense embedding for a single query."""
    if not query.strip():
        raise ValueError("Query must not be empty.")

    return embed_texts([query])[0]

