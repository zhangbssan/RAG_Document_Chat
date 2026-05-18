from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, UPLOAD_DIR
from app.rag.types import Chunk


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


class VectorStore:
    """ChromaDB vector store for RAG chunks."""
    
    def __init__(self):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    
    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks to the vector store."""
        if not chunks:
            return 0
        
        # Embed all texts
        embeddings = embed_texts([chunk.text for chunk in chunks])
        
        # Add to collection
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )
        
        return len(chunks)
    
    def query_chunks(self, query: str, top_k: int = 5) -> list[dict]:
        """Query the vector store for similar chunks."""
        if self.collection.count() == 0:
            return []
        
        # Embed query
        query_embedding = embed_query(query)
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        retrieved = []
        if results and results.get("documents"):
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]
            
            for chunk_id, text, metadata, distance in zip(ids, docs, metas, dists):
                retrieved.append({
                    "id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                    "score": 1 - float(distance) if distance is not None else 0.0,
                })
        
        return retrieved
    
    def list_documents(self) -> set[str]:
        """List all unique documents in the store."""
        result = self.collection.get(include=["metadatas"])
        documents: set[str] = set()
        
        for metadata in result.get("metadatas") or []:
            if metadata and metadata.get("document_name"):
                documents.add(str(metadata["document_name"]))
        
        return documents
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        count = self.collection.count()
        documents = self.list_documents()
        
        return {
            "total_chunks": count,
            "total_documents": len(documents),
            "documents": sorted(documents),
        }
