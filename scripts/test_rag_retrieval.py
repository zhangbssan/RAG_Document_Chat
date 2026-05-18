#!/usr/bin/env python
"""Test RAG retrieval: Load PDFs, embed chunks, and query."""
from __future__ import annotations

import os
import atexit
import hashlib
import logging
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_test_data_dir = Path(tempfile.mkdtemp(prefix="rag-retrieval-test-"))
atexit.register(shutil.rmtree, _test_data_dir, ignore_errors=True)
os.environ["CHROMA_DIR"] = str(_test_data_dir / "chroma")
os.environ["UPLOAD_DIR"] = str(_test_data_dir / "uploads")

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

try:
    import posthog

    posthog.disabled = True
    posthog.capture = lambda *args, **kwargs: None
except ImportError:
    pass

from app.rag.pdf_loader import pdf_extraction
from app.rag.chunker import build_chunks_from_pages
from app.rag import embeddings as embedding_module


def _test_embed_texts(texts: list[str], dimensions: int = 128) -> list[list[float]]:
    """Deterministic local embeddings for this script; avoids model downloads."""
    vectors: list[list[float]] = []

    for text in texts:
        vector = [0.0] * dimensions
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        vectors.append([value / norm for value in vector])

    return vectors


embedding_module.embed_texts = _test_embed_texts

from app.rag.vector_store import add_chunks, get_collection


def query_chunks(query: str, top_k: int = 3) -> list[dict]:
    """Query chunks from the current vector_store.py Chroma collection."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: list[dict] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    for chunk_id, text, metadata, distance in zip(ids, docs, metas, dists):
        retrieved.append(
            {
                "id": chunk_id,
                "text": text,
                "metadata": metadata,
                "score": 1 - float(distance) if distance is not None else 0.0,
            }
        )

    return retrieved


def collection_stats() -> dict:
    """Return basic stats for the current vector_store.py Chroma collection."""
    collection = get_collection()
    result = collection.get(include=["metadatas"])
    documents = {
        str(metadata["document_name"])
        for metadata in result.get("metadatas") or []
        if metadata and metadata.get("document_name")
    }

    return {
        "total_chunks": collection.count(),
        "total_documents": len(documents),
        "documents": sorted(documents),
    }


def test_rag_retrieval():
    """Test complete RAG pipeline: load PDFs -> chunk -> embed -> retrieve."""
    sample_docs_dir = Path(__file__).resolve().parents[1] / "sample_docs"
    
    if not sample_docs_dir.exists():
        print(f"❌ sample_docs not found")
        return False
    
    print("=" * 80)
    print("🔍 RAG RETRIEVAL TEST")
    print("=" * 80 + "\n")
    
    # Step 1: Load and chunk PDFs
    print("📚 [1/3] Loading and chunking PDFs...")
    all_chunks = []
    
    pdf_files = list(sample_docs_dir.glob("*.pdf"))[:2]  # Test with 2 PDFs
    if not pdf_files:
        print("❌ No PDFs found")
        return False
    
    for pdf_path in pdf_files:
        print(f"   📄 {pdf_path.name}")
        pages = pdf_extraction(pdf_path.name, pdf_path)
        chunks = build_chunks_from_pages(pages)
        all_chunks.extend(chunks)
        print(f"      ✓ {len(chunks)} chunks")
    
    print(f"\n   Total chunks: {len(all_chunks)}\n")
    
    # Step 2: Add chunks to vector store
    print("💾 [2/3] Embedding and storing chunks...")
    added = add_chunks(all_chunks)
    print(f"   ✓ Stored {added} chunks\n")
    
    # Step 3: Query and retrieve
    print("🔎 [3/3] Testing queries...\n")
    
    test_queries = [
        "What is the service agreement about?",
        "How do I use the smartwatch?",
        "What are employee benefits?",
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        retrieved = query_chunks(query, top_k=3)
        
        if retrieved:
            for i, chunk in enumerate(retrieved, 1):
                score = chunk.get("score", 0)
                meta = chunk.get("metadata", {})
                doc = meta.get("document_name", "?")
                page = meta.get("page", "?")
                text_preview = chunk.get("text", "")[:60].replace("\n", " ")
                
                print(f"  [{i}] Score: {score:.3f} | Doc: {doc} | Page: {page}")
                print(f"      Text: {text_preview}...")
        else:
            print(f"  No results found")
        
        print()
    
    # Step 4: Show statistics
    print("📊 Vector Store Statistics:")
    stats = collection_stats()
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Total documents: {stats['total_documents']}")
    for doc in stats['documents']:
        print(f"      - {doc}")
    
    print("\n✅ RAG retrieval test completed!")
    return True


if __name__ == "__main__":
    success = test_rag_retrieval()
    sys.exit(0 if success else 1)
