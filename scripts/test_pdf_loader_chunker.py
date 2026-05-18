#!/usr/bin/env python
"""Integration test: Load PDFs from sample_docs and chunk them."""
from __future__ import annotations

import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.rag.pdf_loader import pdf_extraction
from app.rag.chunker import build_chunks_from_pages


def main():
    """Load PDFs and chunk them."""
    sample_docs_dir = Path(__file__).resolve().parents[1] / "sample_docs"
    
    if not sample_docs_dir.exists():
        print(f"❌ sample_docs not found")
        return False
    
    pdf_files = list(sample_docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDFs found")
        return False
    
    print(f"📚 Found {len(pdf_files)} PDFs\n")
    
    total_chunks = 0
    
    for pdf_path in pdf_files:
        print(f"📄 {pdf_path.name}")
        
        try:
            # Step 1: Extract pages from PDF
            pages = pdf_extraction(pdf_path.name, pdf_path)
            print(f"   ✓ Extracted {len(pages)} page(s)")
            
            # Step 2: Chunk the pages
            chunks = build_chunks_from_pages(pages)
            total_chunks += len(chunks)
            print(f"   ✓ Created {len(chunks)} chunk(s)")
            
            # Show sample chunk
            if chunks:
                sample = chunks[0]
                print(f"   Sample: {sample.text[:60]}...")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}\n")
            return False
    
    print(f"✅ Complete! Total chunks created: {total_chunks}")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

