# Architecture

The project is split into a FastAPI backend and a Streamlit frontend.

## Backend

- `app/api`: HTTP endpoints for upload, chat, documents, and future evaluation.
- `app/rag`: PDF loading, chunking, embeddings, ChromaDB storage, retrieval, and answer generation.
- `app/data`: prompts and future evaluation test cases.
- `app/utils`: shared helpers.

## Frontend

The Streamlit app uploads PDFs and sends chat requests to the backend API.

## Storage

Uploaded PDFs and ChromaDB persistence live under `backend/data/`.
