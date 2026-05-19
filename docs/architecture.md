# Architecture

This document describes the repository layout and the high-level RAG flow for RAG Document Chat.

## 1. Project Structure

```text
RAG_Document_Chat/
├── README.md
├── .env
├── .gitignore
├── docker-compose.yml
├── docs/
│   ├── architecture.md
│   └── screenshots/
│       ├── .gitkeep
│       └── evaluation_score.png
├── sample_docs/
│   ├── employee_handbook_de.pdf
│   ├── employee_handbook_en.pdf
│   ├── product_manual_de.pdf
│   ├── product_manual_en.pdf
│   ├── service_agreement_de.pdf
│   └── service_agreement_en.pdf
├── scripts/
│   ├── test_pdf_loader_chunker.py
│   ├── test_rag_reranker.py
│   ├── test_rag_retrieval.py
│   ├── validate_pdf_extraction.py
│   └── validate_setup.py
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── data/
│   │   ├── chroma/
│   │   ├── huggingface/
│   │   └── uploads/
│   │       └── .gitkeep
│   ├── scripts/
│   │   └── smoke_test_pdf_loader.py
│   └── app/
│       ├── main.py
│       ├── config.py
│       ├── schemas.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── chat.py
│       │   ├── documents.py
│       │   ├── evaluation.py
│       │   └── upload.py
│       ├── data/
│       │   └── test_cases.py
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── chunker.py
│       │   ├── embeddings.py
│       │   ├── evaluator.py
│       │   ├── generator.py
│       │   ├── pdf_loader.py
│       │   ├── prompts.py
│       │   ├── reranker.py
│       │   ├── retriever.py
│       │   ├── types.py
│       │   └── vector_store.py
│       └── utils/
│           ├── __init__.py
│           └── file_utils.py
└── frontend/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py
```

### Directory Overview

- `backend/`: FastAPI backend service, including API routes, RAG pipeline modules, configuration, schemas, and utilities.
- `backend/app/api/`: HTTP API endpoints for upload, chat, streaming chat, document management, and evaluation.
- `backend/app/data/`: Hardcoded evaluation test cases.
- `backend/app/rag/`: Core RAG implementation: PDF loading, chunking, embeddings, ChromaDB storage, retrieval, reranking, prompts, generation, and evaluation scoring.
- `backend/app/utils/`: Shared backend helper code.
- `backend/data/`: Runtime data directory for uploaded PDFs, persistent ChromaDB files, and Hugging Face model cache. Docker Compose mounts this directory into the backend container.
- `backend/scripts/`: Backend-specific smoke tests and helper scripts.
- `frontend/`: Streamlit frontend application and Docker/dependency configuration.
- `sample_docs/`: Sample PDFs used by the fixed retrieval evaluation benchmark.
- `scripts/`: Project-level validation and RAG/PDF test scripts.
- `docs/`: Architecture documentation and local run screenshots.

Generated Python caches, virtual environments, local screenshots, ChromaDB files, and model-cache files are intentionally not listed in detail.

## 2. High-Level RAG Flow Diagram

```text
User uploads PDFs in Streamlit
        |
        v
FastAPI /api/upload
        |
        v
PyMuPDF extracts text page by page
        |
        v
Chunker splits page text into overlapping chunks
        |
        v
sentence-transformers creates embeddings
        |
        v
ChromaDB stores chunks, embeddings, and metadata
        |
        v
User asks a question in Streamlit chat
        |
        v
FastAPI /api/chat or /api/chat/stream
        |
        v
ChromaDB dense retrieval gets candidate chunks
        |
        v
Lexical reranker applies keyword overlap + RRF
        |
        v
Top ranked chunks become answer context
        |
        v
OpenAI answer generation or extractive fallback
        |
        v
Streamlit displays answer, citations, excerpts, and rank scores
```

## Backend and Frontend Responsibilities

### Backend

- Validates PDF uploads and file size limits.
- Extracts page-level text from PDFs.
- Builds overlapping chunks with document/page/chunk metadata.
- Stores chunks in a persistent ChromaDB collection.
- Retrieves and reranks chunks for chat and evaluation.
- Generates answers with OpenAI when an API key is available.
- Falls back to extractive answers when no API key is available.

### Frontend

- Provides PDF upload and document management UI.
- Provides chat and streaming answer UI.
- Shows source document, page, chunk, excerpt, and rank score.
- Allows request-scoped OpenAI API key input.
- Provides the fixed evaluation panel and missing-sample-document warning.

## Storage

- Uploaded PDFs are stored under `backend/data/uploads/`.
- ChromaDB persistence is stored under `backend/data/chroma/`.
- Hugging Face model cache is stored under `backend/data/huggingface/` in Docker.
- Docker Compose mounts `backend/data/` into the backend container so indexed data survives container restarts.
