#!/usr/bin/env python
"""Test RAG retrieval: Load PDFs, embed chunks, and query."""
from __future__ import annotations

import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.rag.pdf_loader import pdf_extraction
from app.rag.chunker import build_chunks_from_pages
from app.rag.embeddings import VectorStore
from app.rag.retriever import retrieve_chunks


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
    vector_store = VectorStore()
    all_chunks = []
    
    pdf_files = list(sample_docs_dir.glob("*.pdf"))[:2]  # Test with 2 PDFs
    
    for pdf_path in pdf_files:
        print(f"   📄 {pdf_path.name}")
        pages = pdf_extraction(pdf_path.name, pdf_path)
        chunks = build_chunks_from_pages(pages)
        all_chunks.extend(chunks)
        print(f"      ✓ {len(chunks)} chunks")
    
    print(f"\n   Total chunks: {len(all_chunks)}\n")
    
    # Step 2: Add chunks to vector store
    print("💾 [2/3] Embedding and storing chunks...")
    added = vector_store.add_chunks(all_chunks)
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
        retrieved = retrieve_chunks(query, top_k=3)
        
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
    stats = vector_store.get_stats()
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Total documents: {stats['total_documents']}")
    for doc in stats['documents']:
        print(f"      - {doc}")
    
    print("\n✅ RAG retrieval test completed!")
    return True


if __name__ == "__main__":
    success = test_rag_retrieval()
    sys.exit(0 if success else 1)
