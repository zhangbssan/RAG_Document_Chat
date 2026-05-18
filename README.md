# RAG Document Chat

RAG application for chatting with multiple uploaded PDF files. The project is
split into a FastAPI backend and a Streamlit frontend.

## Features

- Upload multiple PDFs at the same time
- Extract PDF text page by page
- Chunk PDF text with overlap
- Embed chunks with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Store vectors and metadata in persistent ChromaDB
- Chat interface with source references by document, page, and text section
- Optional OpenAI answer generation via `OPENAI_API_KEY`

Without `OPENAI_API_KEY`, the backend still returns the most relevant source
snippets.

## Structure

```text
backend/   FastAPI API, RAG pipeline, ChromaDB storage
frontend/  Streamlit UI
docs/      architecture notes and screenshots
sample_docs/
```

## Local Setup

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Docker

```bash
cp .env.example .env
docker compose up --build
```

Frontend: `http://127.0.0.1:8501`

Backend: `http://127.0.0.1:8000`
