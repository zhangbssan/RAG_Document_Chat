# RAG Document Chat

## Short Project Overview

RAG Document Chat is a lightweight Business RAG application for querying uploaded business PDFs with source-grounded answers.

It is designed for business-style PDF documents such as agreements, manuals, handbooks, and policies. The backend is built with FastAPI and the frontend is built with Streamlit. PDF text is extracted with PyMuPDF, split into overlapping chunks, embedded with sentence-transformers, stored in Milvus, retrieved with hybrid dense+BM25 search fused via Reciprocal Rank Fusion and expanded into anchor-centered context blocks, and then used for OpenAI-based or extractive fallback answers.

The chat feature works with uploaded text-based PDFs. The evaluation feature is a fixed retrieval benchmark for the included sample documents.

For the detailed repository structure and high-level RAG flow diagram, see `docs/architecture.md`.
Local run example screenshots are stored in `docs/screenshots/`, including the evaluation example screenshot and a short local demo recording.

A public demo is available here: [Cloud Run Demo](https://ragdocumentchatfrontend-710350614808.europe-west1.run.app/). The demo is intended for quick testing. For the most reliable local setup, use Docker Compose as described below.

## Feature Checklist

| Requirement | Status | Implementation |
| --- | --- | --- |
| Upload-Feld fur PDFs, mehrere gleichzeitig | Implemented | Streamlit sidebar supports uploading multiple PDF files. Backend validates `.pdf` filenames and enforces `MAX_UPLOAD_SIZE_MB`. |
| Chunking, Embedding und Vektordatenbank | Implemented | PyMuPDF extracts text page by page. Text is split into overlapping chunks. Chunks are embedded with sentence-transformers and stored in Milvus with metadata. |
| Chat-Interface mit Antworten inkl. Quellenangabe | Implemented | Streamlit chat sends questions to FastAPI. Answers include sources with document name, page, chunk/section, excerpt, and rank score. |
| 5 hartcodierte Testfragen mit erwarteten Antworten und Retrieval-Score | Implemented | Evaluation uses five hardcoded test cases with expected answer, expected document, expected page, and expected keywords. Scores are visible in the Streamlit Evaluation tab. |
| FastAPI Backend, Frontend frei wahlbar | Implemented | FastAPI backend and Streamlit frontend. |
| Lauffahig via Docker Compose | Implemented | `docker compose up --build` starts backend and frontend together. |
| README explains chunking strategy and retrieval quality | Implemented | See "Chunking Strategy and Retrieval Quality". |
| Bonus: Streaming-Antworten | Implemented | `/api/chat/stream` sends sources first, then streams answer tokens as NDJSON. |
| Bonus: Re-Ranking der retrieved Chunks | Implemented | Dense and BM25-sparse Milvus candidates are fused with Reciprocal Rank Fusion, then expanded into anchor-centered ±1 chunk context blocks. |
| Bonus: Public deployment | Implemented | A Cloud Run demo is available for quick testing. Docker Compose remains the recommended local setup. |

## Architecture

The detailed project structure and high-level RAG flow diagram are documented in `docs/architecture.md`.

Each indexed chunk stores source metadata:

- `document_name`
- `file_hash`
- `page`
- `chunk_index`

This metadata allows the app to show which document, page, and text section each source came from.

## Tech Stack

- FastAPI backend
- Streamlit frontend
- Milvus vector database
- sentence-transformers embeddings
- PyMuPDF for PDF text extraction
- OpenAI API for optional LLM answer generation
- Docker Compose for running backend and frontend together

## Quick Start with Docker Compose

Requires a Milvus server reachable at `MILVUS_HOST`/`MILVUS_PORT` (see [Environment Variables](#environment-variables)); `docker-compose.yml` in this repo only defines the `backend` and `frontend` services, so Milvus must already be running separately (e.g. via its own `docker compose up` from a Milvus install) and reachable from the backend container.

From the repository root:

```bash
docker compose up --build
```

Then open:

- Frontend: `http://localhost:8501`
- Backend API docs: `http://localhost:8000/docs`
- Backend health check: `http://localhost:8000/health`

To stop the app:

```bash
docker compose down
```

## Local Development Setup

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
API_BASE_URL=http://127.0.0.1:8000 streamlit run app.py
```

The frontend talks to the backend through `API_BASE_URL`.

## Environment Variables

The backend loads environment variables from `.env` and Docker Compose also passes selected values.

| Variable | Default / Notes |
| --- | --- |
| `OPENAI_API_KEY` | Optional. If unset, the app uses extractive fallback answers unless the user enters a request-scoped key in the frontend. |
| `OPENAI_MODEL` | Defaults to `gpt-4o-mini`. |
| `EMBEDDING_MODEL` | Defaults to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. |
| `CHUNK_SIZE` | Defaults to `950` characters. |
| `CHUNK_OVERLAP` | Defaults to `180` characters. |
| `TOP_K` | Defaults to `5`; number of dense candidates retrieved from Milvus. |
| `SPARSE_TOP_K` | Defaults to `TOP_K`'s value; number of BM25 sparse candidates retrieved from Milvus. |
| `ANCHOR_TOP_N` | Defaults to `2`; number of RRF-fused anchor chunks expanded into context blocks. |
| `ANCHOR_WINDOW` | Defaults to `1`; how many chunk_seq neighbours are fetched on each side of an anchor. |
| `MAX_UPLOAD_SIZE_MB` | Defaults to `200`; backend upload size limit per file. |
| `UPLOAD_DIR` | Defaults to backend data directory; Docker uses `/app/data/uploads`. |
| `MILVUS_HOST` | Defaults to `127.0.0.1`; host of the Milvus server used for chunk storage/retrieval. |
| `MILVUS_PORT` | Defaults to `19530`. |
| `REALTIME_PDF_COLLECTION_NAME` | Defaults to `realtime_pdf_collection`; Milvus collection used for uploaded PDF chunks. |

There is no `USE_RERANKER` switch in the current code — hybrid retrieval (dense + BM25, RRF-fused) is the normal retrieval pipeline. `TOP_K`/`SPARSE_TOP_K` control the initial dense/sparse candidate breadth; `ANCHOR_TOP_N`/`ANCHOR_WINDOW` control how many anchors are expanded into context blocks and how wide each block's window is.

## Usage

1. Open the Streamlit frontend at `http://localhost:8501`.
2. Upload one or more PDF files from the sidebar.
3. Enter an OpenAI API key in the sidebar if you want LLM-generated answers. Without a key, the app uses extractive fallback answers based on the retrieved chunks.
4. Ask questions in the Chat tab.
5. Review the answer and the displayed sources.
6. Delete indexed documents from the sidebar if needed.

The OpenAI API key is request-scoped/session-only. It is sent with chat requests when entered and is not stored by the app.

The upload flow accepts PDF files only. Non-PDF filenames are rejected by the backend. Uploaded files are also checked against the configured file size limit.

## Evaluation

The app includes a small fixed retrieval benchmark in the Evaluation tab.

Evaluation uses five hardcoded test cases. Each test case includes:

- question
- expected answer
- expected document
- expected page
- expected keywords

The benchmark is designed for the sample English PDFs in `sample_docs/`:

- `sample_docs/employee_handbook_en.pdf`
- `sample_docs/product_manual_en.pdf`
- `sample_docs/service_agreement_en.pdf`

**Upload these sample PDFs before running evaluation.** The UI displays the required sample documents and warns if any are missing. Scores may be low or zero when the sample documents are not indexed.

Evaluation is read-only. It uses the same current Milvus collection as chat, does not upload sample PDFs automatically, does not delete documents, and does not create a separate collection.

With the three English sample PDFs uploaded, the current local evaluation example score is `0.84` average final score across the five hardcoded questions.

### Retrieval Score

The score is a lightweight custom retrieval metric, not an LLM answer-quality judge.

- `document_hit_score`: `1.0` if at least one retrieved source matches the expected document, otherwise `0.0`.
- `page_hit_score`: `1.0` if at least one retrieved source matches both the expected document and expected page, otherwise `0.0`.
- `keyword_score`: fraction of expected keywords found in the retrieved source text.
- `final_score`: weighted score using `0.5 * document_hit_score + 0.2 * page_hit_score + 0.3 * keyword_score`.

## Chunking Strategy and Retrieval Quality

PDF text is extracted page by page. Chunking is then applied within each page, which preserves page-level citation accuracy.

The default chunking configuration is:

- `CHUNK_SIZE=950`
- `CHUNK_OVERLAP=180`

The chunker uses character-length windows and attempts to split at natural boundaries in this order:

1. paragraph break
2. line break
3. sentence boundary
4. word boundary

Overlap helps preserve context across chunk boundaries. This reduces boundary loss, where an answer-relevant phrase might otherwise be split across two chunks.

Trade-offs:

- Smaller chunks can improve retrieval precision because each chunk is more focused, but they may lose surrounding context.
- Larger chunks preserve more context, but can reduce retrieval precision because irrelevant text is mixed into the same vector.
- More overlap improves continuity across chunks, but increases index size and duplicate text.

## Hybrid Retrieval

The retrieval pipeline is one shared hybrid core (`backend/app/rag/hybrid_search.py`), used by both the streaming chat path and the agent's search tool:

1. Milvus dense (embedding) search and Milvus-native BM25 full-text search each return their own ranked candidate set, independently.
2. The two lists are deduplicated by chunk id and fused with Reciprocal Rank Fusion.
3. The top `ANCHOR_TOP_N` fused chunks become anchors; each anchor's `±ANCHOR_WINDOW` neighbouring chunks (by document-global `chunk_seq`, not page-local `chunk_index`) are fetched and merged into a context block, with duplicate text stripped between same-page neighbours.

Each block is passed to the answer generator and shown as a source in the UI, with a citable `doc:<file_hash>#p<page_range>` link back to its exact span.

Displayed source scores are RRF fusion scores from the hybrid retrieval step. They are not similarity percentages.

## Streaming Answers

The streaming endpoint is:

```text
POST /api/chat/stream
```

It returns newline-delimited JSON. The backend first sends the retrieved sources and then streams answer tokens. Sources and streamed answer tokens are based on the same retrieved context.

## Fallback Behavior

OpenAI answer generation is optional.

If no API key is available from the backend environment and no request-scoped API key is entered in the frontend, the app returns an extractive fallback answer based on the top retrieved source chunks.

The fallback still includes citations and rank scores.

## API Endpoints

Current API routes:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Health check. |
| `POST` | `/api/upload` | Upload and index one or more PDF files. |
| `POST` | `/api/chat` | Non-streaming chat answer with sources. |
| `POST` | `/api/chat/stream` | Streaming chat response; sources first, then tokens. |
| `GET` | `/api/documents/stats` | Indexed document statistics. |
| `GET` | `/api/documents/list` | Full indexed document list with file hash, pages, and chunks. |
| `DELETE` | `/api/documents/{file_hash}` | Delete one indexed document and its uploaded file by file hash. |
| `POST` | `/api/evaluate` | Run the fixed retrieval evaluation benchmark. |

## Limitations and Future Improvements

Current limitations:

- Text-based PDFs only. OCR is not implemented.
- Table extraction is limited to whatever text PyMuPDF extracts.
- Multi-turn chat history is only kept in the Streamlit UI for display. The backend processes each question independently; it does not perform intent detection, conversation-aware retrieval, or query rewriting based on previous turns.
- Page-level citation can still be imperfect for some PDFs because PDF text extraction and chunk boundaries may not always align exactly with the expected page-level answer location.
- The evaluation includes page-level scoring as a strict diagnostic signal. In some cases, the retriever may find the correct document but not the exact expected page.
- The evaluation benchmark is fixed to the sample documents and should not be interpreted as a general benchmark for arbitrary PDFs.
- A public Cloud Run demo is available for quick testing, but Docker Compose is the recommended setup for reliable local use.

Possible improvements:

- OCR support for scanned PDFs.
- Better table-aware extraction and chunking.
- Conversation-aware query rewriting for follow-up questions.
- Stronger reranking with a cross-encoder.
- More evaluation cases and answer-quality evaluation.
- Public deployment with persistent storage and authentication.
