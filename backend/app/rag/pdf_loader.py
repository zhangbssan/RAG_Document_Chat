from __future__ import annotations

import hashlib
from pathlib import Path

import fitz

from app.rag.types import PageText



# PyMuPDF (fitz) key points:
# - Extract text content from PDF for further processing


def pdf_extraction(file_name: str, pdf_path: Path) -> list[PageText]:
    """
    Extract text content from PDF by page.
    
    Args:
        file_name: Name of the PDF file
        pdf_path: Path to the PDF file
        
    Returns:
        List of PageText objects containing extracted page content
    """
    pages: list[PageText] = []
    
    try:
        # Calculate file hash
        file_digest = hashlib.md5(pdf_path.read_bytes()).hexdigest()
        
        # Open PDF document using PyMuPDF
        pdf_document = fitz.open(str(pdf_path))
        total_pages = pdf_document.page_count
        
        for page_index in range(total_pages):
            # Get page (0-indexed, but we use 1-indexed for display)
            page = pdf_document[page_index]
            page_number = page_index + 1
            
            # Extract text from page
            page_text = page.get_text()
            
            # Skip empty pages
            if page_text and page_text.strip():
                pages.append(
                    PageText(
                        document_name=file_name,
                        file_hash=file_digest,
                        page=page_number,
                        text=page_text,
                    )
                )
        
        pdf_document.close()
        
    except Exception as e:
        raise ValueError(f"Error processing PDF '{file_name}': {str(e)}") from e
    
    return pages
