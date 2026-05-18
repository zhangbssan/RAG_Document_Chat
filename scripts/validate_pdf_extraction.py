#!/usr/bin/env python
"""
Validation script to verify PageText extraction accuracy.
Checks that all required fields are correctly extracted from PDFs.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add backend directory to Python path
_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.rag.pdf_loader import pdf_extraction
from app.schemas import PageText


def validate_pagetext_extraction():
    """Validate that PageText extraction works correctly."""
    # Find sample PDFs
    sample_docs_dir = Path(__file__).resolve().parents[1] / "sample_docs"
    
    if not sample_docs_dir.exists():
        print(f"❌ sample_docs directory not found at {sample_docs_dir}")
        return False
    
    pdf_files = list(sample_docs_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {sample_docs_dir}")
        return False
    
    print("🔍 Validating PageText Extraction\n")
    print(f"Testing with {len(pdf_files)} PDF file(s)\n")
    
    all_valid = True
    
    for pdf_path in pdf_files:
        print(f"📄 Testing: {pdf_path.name}")
        print("-" * 70)
        
        try:
            pages = pdf_extraction(pdf_path.name, pdf_path)
            
            if not pages:
                print(f"  ⚠️  No pages extracted")
                print()
                continue
            
            print(f"  ✓ Pages extracted: {len(pages)}")
            
            # Validate each page
            for idx, page in enumerate(pages[:3]):  # Check first 3 pages
                print(f"\n  📑 Page {idx + 1} ({page.page}):")
                
                # Check all required fields
                checks = {
                    "document_name": (isinstance(page.document_name, str) and len(page.document_name) > 0, page.document_name),
                    "file_hash": (isinstance(page.file_hash, str) and len(page.file_hash) == 32, f"{page.file_hash[:16]}..."),
                    "page": (isinstance(page.page, int) and page.page > 0, page.page),
                    "text": (isinstance(page.text, str) and len(page.text) > 0, f"{len(page.text)} chars"),
                }
                
                for field_name, (is_valid, value) in checks.items():
                    status = "✓" if is_valid else "✗"
                    print(f"    {status} {field_name}: {value}")
                    if not is_valid:
                        all_valid = False
                
                # Sample text preview
                text_preview = page.text[:80].replace("\n", " ")
                print(f"    📝 Text preview: {text_preview}...")
            
            if len(pages) > 3:
                print(f"\n  ... and {len(pages) - 3} more pages")
            
            print(f"\n  ✅ {pdf_path.name}: PASSED\n")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}\n")
            all_valid = False
    
    print("=" * 70)
    if all_valid:
        print("✅ All PageText extraction checks PASSED!")
        return True
    else:
        print("❌ Some PageText extraction checks FAILED")
        return False


if __name__ == "__main__":
    success = validate_pagetext_extraction()
    sys.exit(0 if success else 1)
