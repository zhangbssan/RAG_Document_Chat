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

Upload-Feld für PDFs (mehrere gleichzeitig)

Chunking, Embedding und Vektordatenbank (FAISS, ChromaDB oder pgvector)

Chat-Interface mit Antworten inkl. Quellenangabe (Seite / Textabschnitt), der Chat weiß aus welchem Dokument die Antwort kommt

5 hartcodierte Testfragen mit erwarteten Antworten und automatisch berechnetem Retrieval-Score (z.B. via RAGAS oder eigener Metrik), Score sichtbar im UI

FastAPI Backend, Frontend frei wählbar (Streamlit, React, Next.js)

Lauffähig via Docker Compose

README mit Erklärung der gewählten Chunking-Strategie und deren Auswirkung auf die Retrieval-Qualität



Evaluation

The app includes a small fixed retrieval benchmark with five hardcoded questions. 

The benchmark is designed for the sample PDFs in `sample_docs/`.

Before running evaluation:
1. Start the app.
2. Upload the sample PDFs through the UI.
3. Open the Evaluation section and click Run Evaluation.

The evaluation is read-only. It does not clear, upload, or modify documents.

The chat feature works with any uploaded text-based PDF, while the evaluation benchmark is tied to the sample documents because expected answers and source pages must be known in advance.