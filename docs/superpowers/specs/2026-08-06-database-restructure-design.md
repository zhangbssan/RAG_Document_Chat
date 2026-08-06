# Database Restructure Design — Realtime PDF / Offline Docs / Numeric Data

**Status:** Approved pending final user sign-off on this document
**Scope:** `00_RAG_Document_Chat/backend/app/` only. `06_Local_RAG_Agent` is not modified.

## Purpose

Today `00_RAG_Document_Chat` stores everything in one shared ChromaDB collection, and has no
connection at all to the enterprise data (MariaDB company docs, employee numbers) that lives in
`06_Local_RAG_Agent`. This restructures storage into three clean, independently-testable stores —
no user/session isolation, no permissions, no auth. Simplicity over completeness.

| Data type | Store(s) | Notes |
|---|---|---|
| Realtime PDF (user upload) | Milvus — **one shared collection** | Replaces ChromaDB entirely. Mutable: add on upload, delete by file hash. |
| Offline docs (`company_documents/*.md`) | MariaDB (ground truth) **+** Milvus — **one shared collection** | Department stored as metadata on each Milvus chunk, not a separate collection. |
| Numeric data (employee records) | MariaDB only | Replaces SQLite (`enterprise_erp.db`) entirely — no fallback. |

Explicitly out of scope: the leave-policy PDF / `rag_leave_policy_collection` (confirmed not useful
right now), any session/user binding, and rewiring `ai_orchestrator.py`'s routing or `chat.py` to
call the enterprise stores — this plan only replaces the storage layer and gives each store one
tested read/write function.

## Why this is simpler than it first looked

Reading `app/rag/chunker.py` and `app/rag/embeddings.py` shows both are already generic, not
PDF-specific:
- `chunk_text(text, chunk_size, overlap)` — plain string in, list of chunk strings out.
- `embed_texts(list[str])` / `embed_query(str)` — sentence-transformers, no PDF or Chroma coupling.

So both new stores (realtime PDF and offline docs) can reuse this existing chunking/embedding
code directly. Neither needs the files copied from `06_Local_RAG_Agent` in Step 1
(`ai_orchestrator.py`, `Chunks.py`, `Hybrid_retrieval.py`, `Biencoder_dense_retrieval.py`,
`BM25_sparse_retrieval.py`) — those stay exactly as copied, untouched and still dormant, matching
the leave-policy path being deprioritized. This also means no new LangChain/Ollama dependencies:
numeric_data_store's NL→SQL step reuses the plain OpenAI SDK pattern `rag/generator.py` already
uses, not LangChain's `SQLDatabase`.

## Components

### 1. `backend/app/rag/vector_store.py` — rewritten in place (Chroma → Milvus)

Every current caller (`api/upload.py`, `api/chat.py` via `rag/retriever.py`, `api/documents.py`,
`api/evaluation.py`) only depends on this module's six functions. Keeping their exact signatures
and return shapes means **none of those four files change** — only this module's internals swap
from ChromaDB to Milvus:

- `add_chunks(chunks: list[Chunk]) -> int` — embed via `embed_texts()`, insert into the
  `realtime_pdf_collection` Milvus collection (auto id, `text` varchar, `embedding` float vector,
  dynamic metadata fields for `document_name`/`file_hash`/`page`/`chunk_index`).
- `delete_document(file_hash: str) -> int` — `MilvusClient.delete(filter=f'file_hash == "{file_hash}"')`.
- `query_chunks(query: str, top_k: int) -> list[dict]` — embed via `embed_query()`, dense search,
  return `{id, text, metadata, distance, score}` per hit — same shape consumed by `reranker.py` today.
- `list_documents() -> list[dict]` and `indexed_file_hashes() -> set[str]` — `MilvusClient.query()`
  over all rows' metadata, grouped the same way the current Chroma-based versions do.
- Collection is created lazily on first use if it doesn't exist (schema: id, text, dense vector
  dim = the sentence-transformers model's output dim, dynamic fields enabled for metadata).

### 2. `backend/app/config.py` — add settings (no existing values changed)

```python
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
REALTIME_PDF_COLLECTION_NAME = os.getenv("REALTIME_PDF_COLLECTION_NAME", "realtime_pdf_collection")
OFFLINE_DOCS_COLLECTION_NAME = os.getenv("OFFLINE_DOCS_COLLECTION_NAME", "offline_docs_collection")

MARIADB_HOST = os.getenv("MARIADB_HOST", "127.0.0.1")
MARIADB_PORT = int(os.getenv("MARIADB_PORT", "3307"))
MARIADB_USER = os.getenv("MARIADB_USER", "root")
MARIADB_PASSWORD = os.getenv("MARIADB_PASSWORD", "offerishere")
MARIADB_DATABASE = os.getenv("MARIADB_DATABASE", "wiki_db")
```

Defaults match `06_Local_RAG_Agent/docker-compose.yml`'s MariaDB service (port 3307, same
credentials/db name) so this connects to the same server 06 already runs, per your instruction to
keep the data/services owned by `06_Local_RAG_Agent`.

### 3. `backend/app/orchestrator/offline_docs_store.py` — new

- `ensure_tables() -> None` — `CREATE TABLE IF NOT EXISTS departments/articles` (same schema as
  `06/data_ingestion/knowledge_ingest.py`, so it's compatible with data already in `wiki_db` if 06
  has run its own ingestion, and self-sufficient if not).
- `ingest_offline_docs(base_folder: Path) -> dict` — walk `.md` files under `base_folder`;
  department = top-level folder name; clean markdown (strip images, unwrap links — same regexes
  as `knowledge_ingest.py`); insert one `articles` row per file (MariaDB, ground truth); chunk the
  cleaned text via `chunk_text()`, embed via `embed_texts()`, write each chunk into the single
  `offline_docs_collection` Milvus collection with metadata `{article_id, department}`. Returns
  `{articles_inserted, chunks_indexed}`.
- `search_offline_docs(query: str, top_k: int) -> list[dict]` — embed via `embed_query()`, dense
  search `offline_docs_collection`, return hits with `article_id`, `department`, `text`, `score`.

### 4. `backend/app/orchestrator/numeric_data_store.py` — new

- `ensure_users_table() -> None` — `CREATE TABLE IF NOT EXISTS users` matching the existing
  `wiki_db.users` schema (`name`, `department`, `salary_grade`, `remaining_vacation_day`).
- `seed_test_employees(rows: list[dict]) -> int` — idempotent direct insert into `users`, used by
  tests (and available for manual seeding) — no dependency on SQLite or `migrate_employee.py`.
- `query_numeric_data(nl_question: str) -> dict` — reflects the `users` table's columns via
  SQLAlchemy, builds a prompt (schema + question) asking the OpenAI chat model for one SQL
  `SELECT` statement (mirrors `rag/generator.py`'s existing OpenAI-call style, not LangChain),
  cleans the response the same way `ai_orchestrator.py`'s `clean_sql()` does, executes it via the
  SQLAlchemy engine, returns `{raw_sql, result}`.

### 5. New dependencies (`backend/requirements.txt`)

`pymilvus` (Milvus client), `SQLAlchemy`, `PyMySQL` (MariaDB driver). Nothing else — no LangChain,
no Ollama, no BM25/pymilvus.model. Everything else reuses what's already in the file.

## Data Flow

- **Upload:** `POST /api/upload` → `pdf_loader` → `chunker.build_chunks_from_pages` →
  `vector_store.add_chunks` → `realtime_pdf_collection` (Milvus). Unchanged from today except the
  storage backend.
- **Offline ingestion (manual script run, like today's `knowledge_ingest.py`):**
  `ingest_offline_docs("company_documents/")` → MariaDB `articles` row + `offline_docs_collection`
  chunks per file.
- **Numeric query:** `query_numeric_data("How many people are in Engineering?")` → OpenAI generates
  SQL against `users` → SQLAlchemy executes against MariaDB → result string.

## Error Handling

- Milvus collection missing on query (nothing uploaded/ingested yet) → return `[]`, not an error
  (matches today's `collection.count() == 0` early-return behavior in the Chroma version).
- MariaDB unreachable → the calling function raises; no SQLite or Chroma fallback anywhere.
- Malformed/empty markdown file during offline ingestion → skip that file, log it, continue the
  batch (don't abort the whole ingestion run over one bad file).

## Testing

Integration tests requiring a live Milvus + the MariaDB from `06_Local_RAG_Agent/docker-compose.yml`
(both already required for the leave-policy path today, so this isn't a new operational burden):

1. **`test_vector_store_milvus.py`** — `add_chunks` → `query_chunks` → `delete_document` round trip
   against `realtime_pdf_collection`; confirms the six-function contract still holds so
   `upload.py`/`chat.py`/`documents.py`/`evaluation.py` keep working untouched.
2. **`test_offline_docs_store.py`** — ingest 2–3 sample `.md` files under a temp department folder;
   assert MariaDB `articles` row count and Milvus chunk count match expectations; run
   `search_offline_docs()` and assert the top hit's `article_id` resolves back to the correct
   MariaDB row and carries the right `department`.
3. **`test_numeric_data_store.py`** — `seed_test_employees()` a few rows; ask a natural-language
   question; assert the generated SQL targets `users` and the returned result matches the seed data.

## Self-Review

- **Placeholders:** none — every component lists concrete function signatures and exact schema/config values.
- **Consistency:** all three stores use the same embedding function (`app/rag/embeddings.py`,
  sentence-transformers) and the same MariaDB server (06's), so there's no mixed-backend surprise.
- **Scope:** confirmed bounded to `00_RAG_Document_Chat/backend/app/` — the Step-1-copied
  orchestrator files are left untouched, leave-policy is untouched, no session/auth work, no
  `chat.py` routing expansion beyond what's needed to keep it working against the new Milvus store.
- **Ambiguity resolved:** department is stored as Milvus metadata (not a separate collection) per
  your confirmation; leave-policy explicitly deprioritized per your confirmation.