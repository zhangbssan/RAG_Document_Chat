# Project Structure

```text
RAG_Document_Chat/
├── README.md
├── QUICK_START.md
├── LANGUAGE_CONVERSION.md
├── ENGLISH_CONVERSION_COMPLETE.md
├── docker-compose.yml
├── docs/
│   └── architecture.md
├── scripts/
│   ├── test_pdf_loader_chunker.py
│   ├── test_rag_reranker.py
│   ├── test_rag_retrieval.py
│   ├── validate_pdf_extraction.py
│   └── validate_setup.py
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
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

## Directory Overview

- `backend/`: FastAPI backend service, including API routes, RAG pipeline modules, configuration, schemas, and backend utilities.
- `backend/app/api/`: HTTP API endpoints for chat, document handling, upload, and evaluation.
- `backend/app/rag/`: Core RAG implementation, including PDF loading, chunking, embeddings, vector storage, retrieval, reranking, prompt construction, generation, and evaluation.
- `backend/app/utils/`: Shared backend utility code.
- `backend/scripts/`: Backend-specific smoke tests and helper scripts.
- `frontend/`: Frontend application and its Docker/dependency configuration.
- `scripts/`: Project-level validation and test scripts for RAG behavior and PDF processing.
- `docs/`: Architecture and project documentation.

## Root Files

- `README.md`: Main project documentation.
- `QUICK_START.md`: Quick setup and usage guide.
- `docker-compose.yml`: Container orchestration for running project services.
- `LANGUAGE_CONVERSION.md`: Language conversion notes.
- `ENGLISH_CONVERSION_COMPLETE.md`: English conversion completion notes.
