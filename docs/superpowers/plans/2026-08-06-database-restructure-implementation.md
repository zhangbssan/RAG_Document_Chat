# Database Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ChromaDB with Milvus for realtime PDF uploads (one shared collection), and give offline company documents (MariaDB ground truth + one shared Milvus collection, department as metadata) and numeric employee data (MariaDB only) their own tested storage/retrieval functions — per `docs/superpowers/specs/2026-08-06-database-restructure-design.md`.

**Architecture:** `backend/app/rag/vector_store.py` is rewritten in place (same 6-function contract, Milvus instead of Chroma) so `upload.py`/`chat.py`/`documents.py`/`evaluation.py` need no changes. Two new self-contained modules go under `backend/app/orchestrator/` (`offline_docs_store.py`, `numeric_data_store.py`), reusing the existing generic `chunker.chunk_text()` and `embeddings.embed_texts()`/`embed_query()` — no LangChain/Ollama dependency added. Tests follow this repo's existing convention: standalone scripts under `scripts/` (not pytest — there is no pytest in this project).

**Tech Stack:** `pymilvus` (Milvus client), `SQLAlchemy` + `PyMySQL` (MariaDB), existing `sentence-transformers` embeddings, existing `openai` SDK for NL→SQL.

## Global Constraints

- No session/user isolation, no auth, no permissions — confirmed explicitly out of scope.
- Leave-policy PDF / `rag_leave_policy_collection` and the Step-1-copied orchestrator files (`ai_orchestrator.py`, `Chunks.py`, `Hybrid_retrieval.py`, `Biencoder_dense_retrieval.py`, `BM25_sparse_retrieval.py`) are untouched — confirmed deprioritized.
- Department is stored as Milvus **metadata** on offline-doc chunks, not a separate collection per department — confirmed.
- MariaDB is the only store for numeric data — no SQLite fallback.
- `06_Local_RAG_Agent` is not modified by this plan.
- **Environment reality check (verified this session, not assumed):** `backend/.venv` existed but had nothing installed beyond `pip`/`setuptools` (its `pip` launcher script also had a stale shebang from before this project was moved — use `python3 -m pip`, not `pip`, to avoid that). `pymilvus==2.6.3`, `SQLAlchemy==2.0.44`, and `PyMySQL==1.1.1` have already been installed into `backend/.venv` during design verification. Docker was installed but the daemon was not running, so MariaDB (`06_Local_RAG_Agent/docker-compose.yml`) and Milvus were not reachable at design time.
- **Before running any test task below (Tasks 2–4), start services:**
  ```bash
  # 1. Start Docker Desktop (GUI), then:
  cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/06_Local_RAG_Agent
  docker compose up -d mariadb
  # 2. Ensure a Milvus instance is reachable at MILVUS_HOST:MILVUS_PORT (default 127.0.0.1:19530).
  #    No Milvus compose file exists in this repo — start it however you normally do for this project.
  # 3. Confirm 00_RAG_Document_Chat/.env has a valid OPENAI_API_KEY (Task 4's test calls OpenAI).
  ```
  If a task's test can't reach a required service, the test script will raise a clear connection
  error — that's expected; start the missing service and re-run rather than treating it as a bug.

---

## File Structure

```
00_RAG_Document_Chat/
├── backend/
│   ├── requirements.txt                   # MODIFY: +pymilvus +SQLAlchemy +PyMySQL
│   └── app/
│       ├── config.py                      # MODIFY: +Milvus/MariaDB settings
│       ├── rag/
│       │   └── vector_store.py            # REWRITE: Chroma -> Milvus, same 6-function API
│       └── orchestrator/
│           ├── offline_docs_store.py      # NEW
│           └── numeric_data_store.py      # NEW
└── scripts/
    ├── test_rag_retrieval.py              # MODIFY: stop touching raw Chroma API, add cleanup
    ├── test_vector_store_milvus.py        # NEW
    ├── test_offline_docs_store.py         # NEW
    └── test_numeric_data_store.py         # NEW
```

---

### Task 1: Add Milvus + MariaDB dependencies and config settings

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/app/config.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `app.config.MILVUS_HOST`, `MILVUS_PORT`, `REALTIME_PDF_COLLECTION_NAME`, `OFFLINE_DOCS_COLLECTION_NAME`, `MARIADB_HOST`, `MARIADB_PORT`, `MARIADB_USER`, `MARIADB_PASSWORD`, `MARIADB_DATABASE`, `MARIADB_SQLALCHEMY_URI` — used by every later task.

- [ ] **Step 1: Append new dependencies to `backend/requirements.txt`**

Add these three lines at the end of the file:
```
pymilvus==2.6.3
SQLAlchemy==2.0.44
PyMySQL==1.1.1
```

- [ ] **Step 2: Install them**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 -m pip install -r backend/requirements.txt
```
Expected: installs succeed (already verified working in this environment during design; `pymilvus`/`SQLAlchemy`/`PyMySQL` are already present, this re-run just confirms/completes the rest of `requirements.txt`).

- [ ] **Step 3: Add settings to `backend/app/config.py`**

Append at the end of the file (after the existing `CHUNK_OVERLAP` line):
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
MARIADB_SQLALCHEMY_URI = (
    f"mysql+pymysql://{MARIADB_USER}:{MARIADB_PASSWORD}@{MARIADB_HOST}:{MARIADB_PORT}/{MARIADB_DATABASE}"
)
```

- [ ] **Step 4: Verify**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat/backend
.venv/bin/python3 -c "
from app.config import (
    MILVUS_HOST, MILVUS_PORT, REALTIME_PDF_COLLECTION_NAME,
    OFFLINE_DOCS_COLLECTION_NAME, MARIADB_SQLALCHEMY_URI,
)
print(MILVUS_HOST, MILVUS_PORT, REALTIME_PDF_COLLECTION_NAME, OFFLINE_DOCS_COLLECTION_NAME)
print(MARIADB_SQLALCHEMY_URI)
"
```
Expected: prints `127.0.0.1 19530 realtime_pdf_collection offline_docs_collection` and
`mysql+pymysql://root:offerishere@127.0.0.1:3307/wiki_db`, no errors.

- [ ] **Step 5: Commit**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
git add backend/requirements.txt backend/app/config.py
git commit -m "Add Milvus and MariaDB settings/dependencies for the database restructure"
```

---

### Task 2: Rewrite `vector_store.py` for Milvus, fix `test_rag_retrieval.py`

**Files:**
- Modify: `backend/app/rag/vector_store.py`
- Modify: `scripts/test_rag_retrieval.py`
- Test: `scripts/test_vector_store_milvus.py` (new)

**Interfaces:**
- Consumes: `app.config.MILVUS_HOST/PORT/REALTIME_PDF_COLLECTION_NAME` (Task 1), `app.rag.embeddings.embed_texts`/`embed_query` (existing, unchanged), `app.rag.types.Chunk` (existing, unchanged).
- Produces: same six functions as before — `get_collection()`, `indexed_file_hashes() -> set[str]`, `add_chunks(chunks: list[Chunk]) -> int`, `delete_document(file_hash: str) -> int`, `query_chunks(query: str, top_k: int) -> list[dict]` (each dict: `{id, text, metadata, distance, score}`), `list_documents() -> list[dict]` (each dict: `{document_name, file_hash, pages, chunks}`). `api/upload.py`, `api/chat.py`, `api/documents.py`, `api/evaluation.py` all consume these six functions and are **not modified** in this task — their behavior must be identical to before.

- [ ] **Step 1: Write the failing test — `scripts/test_vector_store_milvus.py`**

```python
#!/usr/bin/env python
"""Test the Milvus-backed vector_store.py: add -> query -> list -> delete round trip."""
from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_TEST_COLLECTION = "test_realtime_pdf_collection"
os.environ["REALTIME_PDF_COLLECTION_NAME"] = _TEST_COLLECTION

from app.rag import embeddings as embedding_module


def _test_embed(texts: list[str], dimensions: int = 384) -> list[list[float]]:
    """Deterministic local embeddings for this script; avoids model downloads."""
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dimensions
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


embedding_module.embed_texts = _test_embed
embedding_module.embed_query = lambda q: _test_embed([q])[0]

from app.rag.types import Chunk
from app.rag.vector_store import (
    add_chunks,
    delete_document,
    get_collection,
    indexed_file_hashes,
    list_documents,
    query_chunks,
)


def test_vector_store_milvus() -> bool:
    print("=" * 70)
    print("MILVUS VECTOR STORE TEST")
    print("=" * 70)

    chunks = [
        Chunk(
            id="hash1:p1:c1",
            text="The service agreement covers annual maintenance for industrial equipment.",
            metadata={"document_name": "service_agreement.pdf", "file_hash": "hash1", "page": 1, "chunk_index": 1},
        ),
        Chunk(
            id="hash1:p2:c1",
            text="Termination requires 30 days written notice from either party.",
            metadata={"document_name": "service_agreement.pdf", "file_hash": "hash1", "page": 2, "chunk_index": 1},
        ),
    ]

    try:
        print("\n[1/5] Adding chunks...")
        added = add_chunks(chunks)
        assert added == 2, f"expected 2 chunks added, got {added}"
        print(f"   OK: added {added} chunks")

        print("\n[2/5] indexed_file_hashes()...")
        hashes = indexed_file_hashes()
        assert "hash1" in hashes, f"expected hash1 in {hashes}"
        print(f"   OK: {hashes}")

        print("\n[3/5] query_chunks()...")
        results = query_chunks("What is the termination notice period?", top_k=2)
        assert len(results) > 0, "expected at least one result"
        assert any("30 days" in r["text"] for r in results), f"expected termination chunk in results: {results}"
        print(f"   OK: {len(results)} results, top text: {results[0]['text'][:60]}...")

        print("\n[4/5] list_documents()...")
        docs = list_documents()
        assert len(docs) == 1, f"expected 1 document, got {docs}"
        assert docs[0]["document_name"] == "service_agreement.pdf"
        assert docs[0]["chunks"] == 2
        print(f"   OK: {docs}")

        print("\n[5/5] delete_document()...")
        deleted = delete_document("hash1")
        assert deleted == 2, f"expected 2 deleted, got {deleted}"
        remaining = list_documents()
        assert remaining == [], f"expected no documents left, got {remaining}"
        print(f"   OK: deleted {deleted}, {len(remaining)} documents remain")

        print("\nPASSED: vector_store.py works against Milvus.")
        return True

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return False

    finally:
        client = get_collection()
        if client.has_collection(_TEST_COLLECTION):
            client.drop_collection(_TEST_COLLECTION)
            print(f"\n[cleanup] dropped {_TEST_COLLECTION}")


if __name__ == "__main__":
    success = test_vector_store_milvus()
    sys.exit(0 if success else 1)
```

- [ ] **Step 2: Run it, confirm it fails for the right reason**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_vector_store_milvus.py
```
Expected: `ImportError` or `AttributeError` — `vector_store.py` still imports `chromadb`, not `pymilvus`, so it can't satisfy this test yet.

- [ ] **Step 3: Rewrite `backend/app/rag/vector_store.py`**

Replace the entire file with:
```python
from __future__ import annotations

from pymilvus import DataType, MilvusClient

from app.config import MILVUS_HOST, MILVUS_PORT, REALTIME_PDF_COLLECTION_NAME
from app.rag import embeddings
from app.rag.types import Chunk

_client: MilvusClient | None = None


def get_collection() -> MilvusClient:
    """Return the shared MilvusClient, creating the realtime PDF collection if needed."""
    global _client
    if _client is None:
        _client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

    if not _client.has_collection(REALTIME_PDF_COLLECTION_NAME):
        dim = len(embeddings.embed_query("dimension probe"))

        schema = _client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
        _client.create_collection(collection_name=REALTIME_PDF_COLLECTION_NAME, schema=schema)

        index_params = _client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        _client.create_index(REALTIME_PDF_COLLECTION_NAME, index_params)
        _client.load_collection(REALTIME_PDF_COLLECTION_NAME)

    return _client


def indexed_file_hashes() -> set[str]:
    client = get_collection()
    rows = client.query(
        collection_name=REALTIME_PDF_COLLECTION_NAME,
        filter="",
        output_fields=["file_hash"],
    )
    return {row["file_hash"] for row in rows if row.get("file_hash")}


def add_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0

    client = get_collection()
    texts = [chunk.text for chunk in chunks]
    vectors = embeddings.embed_texts(texts)

    data = [
        {"text": chunk.text, "embedding": vector, **chunk.metadata}
        for chunk, vector in zip(chunks, vectors)
    ]
    client.insert(collection_name=REALTIME_PDF_COLLECTION_NAME, data=data)
    return len(chunks)


def delete_document(file_hash: str) -> int:
    client = get_collection()
    matches = client.query(
        collection_name=REALTIME_PDF_COLLECTION_NAME,
        filter=f'file_hash == "{file_hash}"',
        output_fields=["id"],
    )
    ids = [row["id"] for row in matches]

    if not ids:
        return 0

    client.delete(collection_name=REALTIME_PDF_COLLECTION_NAME, ids=ids)
    return len(ids)


def query_chunks(query: str, top_k: int = 5) -> list[dict]:
    if not query.strip():
        raise ValueError("Query must not be empty.")

    client = get_collection()
    stats = client.get_collection_stats(REALTIME_PDF_COLLECTION_NAME)
    if int(stats.get("row_count", 0)) == 0:
        return []

    query_vector = embeddings.embed_query(query)
    results = client.search(
        collection_name=REALTIME_PDF_COLLECTION_NAME,
        data=[query_vector],
        anns_field="embedding",
        limit=top_k,
        output_fields=["text", "document_name", "file_hash", "page", "chunk_index"],
    )

    retrieved: list[dict] = []
    for hits in results:
        for hit in hits:
            entity = hit.get("entity", {})
            distance = hit.get("distance")
            retrieved.append(
                {
                    "id": hit.get("id"),
                    "text": entity.get("text"),
                    "metadata": {
                        "document_name": entity.get("document_name"),
                        "file_hash": entity.get("file_hash"),
                        "page": entity.get("page"),
                        "chunk_index": entity.get("chunk_index"),
                    },
                    "distance": float(distance) if distance is not None else None,
                    "score": float(distance) if distance is not None else 0.0,
                }
            )

    return retrieved


def list_documents() -> list[dict]:
    client = get_collection()
    rows = client.query(
        collection_name=REALTIME_PDF_COLLECTION_NAME,
        filter="",
        output_fields=["document_name", "file_hash", "page"],
    )

    documents: dict[str, dict] = {}
    for row in rows:
        document_name = row.get("document_name")
        if not document_name:
            continue

        if document_name not in documents:
            documents[document_name] = {
                "document_name": document_name,
                "file_hash": row.get("file_hash"),
                "pages": set(),
                "chunks": 0,
            }

        page = row.get("page")
        if page is not None:
            documents[document_name]["pages"].add(page)
        documents[document_name]["chunks"] += 1

    return [
        {
            "document_name": doc["document_name"],
            "file_hash": doc["file_hash"],
            "pages": len(doc["pages"]),
            "chunks": doc["chunks"],
        }
        for doc in documents.values()
    ]
```

Note: `list_documents()` no longer needs the `collection.count() == 0` early-return `query_chunks` had in the
Chroma version — Milvus's `query()` with an empty result set just returns `[]` naturally.

- [ ] **Step 4: Fix `scripts/test_rag_retrieval.py` to stop touching raw Chroma APIs**

The script currently imports `get_collection` and calls Chroma-specific `.count()`/`.query()`/`.get()`
directly on it — those methods don't exist on a `MilvusClient`. Replace the entire file with:

```python
#!/usr/bin/env python
"""Test RAG retrieval: Load PDFs, embed chunks, and query (Milvus-backed vector_store.py)."""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_TEST_COLLECTION = "test_rag_retrieval_collection"
_test_upload_dir = Path(tempfile.mkdtemp(prefix="rag-retrieval-test-"))
os.environ["UPLOAD_DIR"] = str(_test_upload_dir)
os.environ["REALTIME_PDF_COLLECTION_NAME"] = _TEST_COLLECTION

from app.rag import embeddings as embedding_module


def _test_embed_texts(texts: list[str], dimensions: int = 384) -> list[list[float]]:
    """Deterministic local embeddings for this script; avoids model downloads."""
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dimensions
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        vectors.append([value / norm for value in vector])
    return vectors


embedding_module.embed_texts = _test_embed_texts
embedding_module.embed_query = lambda q: _test_embed_texts([q])[0]

from app.rag.pdf_loader import pdf_extraction
from app.rag.chunker import build_chunks_from_pages
from app.rag.vector_store import add_chunks, get_collection, list_documents, query_chunks


def collection_stats() -> dict:
    """Return basic stats for the current vector_store.py collection."""
    documents = list_documents()
    return {
        "total_chunks": sum(doc["chunks"] for doc in documents),
        "total_documents": len(documents),
        "documents": sorted(doc["document_name"] for doc in documents),
    }


def test_rag_retrieval():
    """Test complete RAG pipeline: load PDFs -> chunk -> embed -> retrieve."""
    sample_docs_dir = Path(__file__).resolve().parents[1] / "sample_docs"

    if not sample_docs_dir.exists():
        print(f"❌ sample_docs not found")
        return False

    print("=" * 80)
    print("🔍 RAG RETRIEVAL TEST")
    print("=" * 80 + "\n")

    try:
        print("📚 [1/3] Loading and chunking PDFs...")
        all_chunks = []

        pdf_files = list(sample_docs_dir.glob("*.pdf"))[:2]
        if not pdf_files:
            print("❌ No PDFs found")
            return False

        for pdf_path in pdf_files:
            print(f"   📄 {pdf_path.name}")
            pages = pdf_extraction(pdf_path.name, pdf_path)
            chunks = build_chunks_from_pages(pages)
            all_chunks.extend(chunks)
            print(f"      ✓ {len(chunks)} chunks")

        print(f"\n   Total chunks: {len(all_chunks)}\n")

        print("💾 [2/3] Embedding and storing chunks...")
        added = add_chunks(all_chunks)
        print(f"   ✓ Stored {added} chunks\n")

        print("🔎 [3/3] Testing queries...\n")
        test_queries = [
            "What is the service agreement about?",
            "How do I use the smartwatch?",
            "What are employee benefits?",
        ]

        for query in test_queries:
            print(f"Query: '{query}'")
            retrieved = query_chunks(query, top_k=3)

            if retrieved:
                for i, chunk in enumerate(retrieved, 1):
                    score = chunk.get("score", 0)
                    meta = chunk.get("metadata", {})
                    doc = meta.get("document_name", "?")
                    page = meta.get("page", "?")
                    text_preview = chunk.get("text", "")[:60].replace("\n", " ")
                    print(f"  [{i}] Score: {score:.3f} | Doc: {doc} | Page: {page}")
                    print(f"      Text: {text_preview}...")
            else:
                print(f"  No results found")
            print()

        print("📊 Vector Store Statistics:")
        stats = collection_stats()
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Total documents: {stats['total_documents']}")
        for doc in stats['documents']:
            print(f"      - {doc}")

        print("\n✅ RAG retrieval test completed!")
        return True

    finally:
        client = get_collection()
        if client.has_collection(_TEST_COLLECTION):
            client.drop_collection(_TEST_COLLECTION)
            print(f"\n[cleanup] dropped Milvus collection {_TEST_COLLECTION}")
        shutil.rmtree(_test_upload_dir, ignore_errors=True)


if __name__ == "__main__":
    success = test_rag_retrieval()
    sys.exit(0 if success else 1)
```

- [ ] **Step 5: Run both, confirm they pass**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_vector_store_milvus.py
backend/.venv/bin/python3 scripts/test_rag_retrieval.py
```
Expected: both print `PASSED`/`✅ RAG retrieval test completed!` and exit 0. Requires Milvus reachable
(see Global Constraints prerequisites).

- [ ] **Step 6: Commit**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
git add backend/app/rag/vector_store.py scripts/test_rag_retrieval.py scripts/test_vector_store_milvus.py
git commit -m "Replace ChromaDB with Milvus in vector_store.py, keep the same 6-function API"
```

---

### Task 3: Build `offline_docs_store.py`

**Files:**
- Create: `backend/app/orchestrator/offline_docs_store.py`
- Test: `scripts/test_offline_docs_store.py` (new)

**Interfaces:**
- Consumes: `app.config.MARIADB_*`/`MILVUS_HOST`/`MILVUS_PORT`/`OFFLINE_DOCS_COLLECTION_NAME` (Task 1), `app.rag.chunker.chunk_text` (existing), `app.rag.embeddings.embed_texts`/`embed_query` (existing).
- Produces: `ensure_tables() -> None`, `ingest_offline_docs(base_folder: Path) -> dict` (`{articles_inserted, chunks_indexed}`), `search_offline_docs(query: str, top_k: int) -> list[dict]` (`{text, article_id, department, score}`). Also exposes `_get_db_connection()` and `_get_milvus_client()` for test/inspection use.

- [ ] **Step 1: Write the failing test — `scripts/test_offline_docs_store.py`**

```python
#!/usr/bin/env python
"""Test offline_docs_store.py: ingest markdown -> MariaDB + Milvus, then search + verify back-reference."""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_TEST_COLLECTION = "test_offline_docs_collection"
_TEST_DEPARTMENT = "test-restructure-dept"
os.environ["OFFLINE_DOCS_COLLECTION_NAME"] = _TEST_COLLECTION

from app.rag import embeddings as embedding_module


def _test_embed(texts: list[str], dimensions: int = 384) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dimensions
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


embedding_module.embed_texts = _test_embed
embedding_module.embed_query = lambda q: _test_embed([q])[0]

from app.orchestrator.offline_docs_store import (
    _get_db_connection,
    _get_milvus_client,
    ingest_offline_docs,
    search_offline_docs,
)


def test_offline_docs_store() -> bool:
    print("=" * 70)
    print("OFFLINE DOCS STORE TEST")
    print("=" * 70)

    temp_dir = Path(tempfile.mkdtemp(prefix="offline-docs-test-"))
    dept_dir = temp_dir / _TEST_DEPARTMENT
    dept_dir.mkdir(parents=True)

    (dept_dir / "onboarding.md").write_text(
        "# Onboarding\n\nNew hires get a laptop and a company badge in week one.",
        encoding="utf-8",
    )
    (dept_dir / "security.md").write_text(
        "# Security\n\nAll staff must enable two-factor authentication within 24 hours of joining.",
        encoding="utf-8",
    )

    conn = None
    try:
        print("\n[1/3] Ingesting 2 markdown files...")
        stats = ingest_offline_docs(temp_dir)
        assert stats["articles_inserted"] == 2, f"expected 2 articles, got {stats}"
        assert stats["chunks_indexed"] >= 2, f"expected >=2 chunks, got {stats}"
        print(f"   OK: {stats}")

        print("\n[2/3] Verifying MariaDB ground truth...")
        conn = _get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT a.id, a.title, a.breadcrumbs, d.name AS department "
                "FROM articles a JOIN departments d ON a.dept_id = d.id "
                "WHERE d.name = %s",
                (_TEST_DEPARTMENT,),
            )
            rows = cursor.fetchall()
        assert len(rows) == 2, f"expected 2 MariaDB rows, got {rows}"
        titles = {row["title"] for row in rows}
        assert titles == {"onboarding", "security"}, f"unexpected titles: {titles}"
        print(f"   OK: {rows}")

        print("\n[3/3] Searching Milvus + verifying article_id back-reference...")
        hits = search_offline_docs("How do new hires get set up with equipment?", top_k=3)
        assert len(hits) > 0, "expected at least one hit"
        top_hit = hits[0]
        assert top_hit["department"] == _TEST_DEPARTMENT, f"unexpected department: {top_hit}"
        matching_row = next((r for r in rows if r["id"] == top_hit["article_id"]), None)
        assert matching_row is not None, f"article_id {top_hit['article_id']} not found in {rows}"
        print(f"   OK: top hit article_id={top_hit['article_id']} -> MariaDB title='{matching_row['title']}'")

        print("\nPASSED: offline_docs_store.py works end to end.")
        return True

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return False

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

        client = _get_milvus_client()
        if client.has_collection(_TEST_COLLECTION):
            client.drop_collection(_TEST_COLLECTION)
            print(f"[cleanup] dropped Milvus collection {_TEST_COLLECTION}")

        if conn is None:
            conn = _get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE a FROM articles a JOIN departments d ON a.dept_id = d.id WHERE d.name = %s",
                (_TEST_DEPARTMENT,),
            )
            cursor.execute("DELETE FROM departments WHERE name = %s", (_TEST_DEPARTMENT,))
        conn.commit()
        conn.close()
        print(f"[cleanup] removed MariaDB test rows for department '{_TEST_DEPARTMENT}'")


if __name__ == "__main__":
    success = test_offline_docs_store()
    sys.exit(0 if success else 1)
```

- [ ] **Step 2: Run it, confirm it fails**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_offline_docs_store.py
```
Expected: `ModuleNotFoundError: No module named 'app.orchestrator.offline_docs_store'`.

- [ ] **Step 3: Create `backend/app/orchestrator/offline_docs_store.py`**

```python
from __future__ import annotations

import os
import re
from pathlib import Path

import pymysql
from pymilvus import DataType, MilvusClient

from app.config import (
    MARIADB_DATABASE,
    MARIADB_HOST,
    MARIADB_PASSWORD,
    MARIADB_PORT,
    MARIADB_USER,
    MILVUS_HOST,
    MILVUS_PORT,
    OFFLINE_DOCS_COLLECTION_NAME,
)
from app.rag import embeddings
from app.rag.chunker import chunk_text

_milvus_client: MilvusClient | None = None


def _get_db_connection():
    return pymysql.connect(
        host=MARIADB_HOST,
        port=MARIADB_PORT,
        user=MARIADB_USER,
        password=MARIADB_PASSWORD,
        database=MARIADB_DATABASE,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def _get_milvus_client() -> MilvusClient:
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

    if not _milvus_client.has_collection(OFFLINE_DOCS_COLLECTION_NAME):
        dim = len(embeddings.embed_query("dimension probe"))

        schema = _milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
        _milvus_client.create_collection(collection_name=OFFLINE_DOCS_COLLECTION_NAME, schema=schema)

        index_params = _milvus_client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        _milvus_client.create_index(OFFLINE_DOCS_COLLECTION_NAME, index_params)
        _milvus_client.load_collection(OFFLINE_DOCS_COLLECTION_NAME)

    return _milvus_client


def clean_markdown_content(raw_text: str) -> str:
    cleaned_text = re.sub(r"!\[.*?\]\(.*?\)", "", raw_text)
    cleaned_text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", cleaned_text)
    return cleaned_text.strip()


def ensure_tables() -> None:
    conn = _get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS departments (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(50) UNIQUE NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    content MEDIUMTEXT NOT NULL,
                    breadcrumbs VARCHAR(500) NOT NULL,
                    dept_id INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dept_id) REFERENCES departments(id)
                )
                """
            )
        conn.commit()
    finally:
        conn.close()


def ingest_offline_docs(base_folder: Path) -> dict:
    """Ingest every .md file under base_folder: MariaDB ground truth + Milvus chunks."""
    ensure_tables()
    milvus_client = _get_milvus_client()

    conn = _get_db_connection()
    articles_inserted = 0
    chunks_indexed = 0

    try:
        with conn.cursor() as cursor:
            for root, _dirs, files in os.walk(base_folder):
                for file_name in files:
                    if not file_name.endswith(".md"):
                        continue

                    full_path = Path(root) / file_name
                    rel_path = Path(root).relative_to(base_folder)

                    if str(rel_path) == ".":
                        department = "General"
                        breadcrumbs = "General"
                    else:
                        parts = rel_path.parts
                        department = parts[0]
                        breadcrumbs = " > ".join(parts)

                    cursor.execute("INSERT IGNORE INTO departments (name) VALUES (%s)", (department,))
                    cursor.execute("SELECT id FROM departments WHERE name = %s", (department,))
                    dept_id = cursor.fetchone()["id"]

                    raw_content = full_path.read_text(encoding="utf-8")
                    cleaned_content = clean_markdown_content(raw_content)
                    title = full_path.stem.replace("_", " ")

                    cursor.execute(
                        "INSERT INTO articles (title, content, breadcrumbs, dept_id) VALUES (%s, %s, %s, %s)",
                        (title, cleaned_content, breadcrumbs, dept_id),
                    )
                    article_id = cursor.lastrowid
                    articles_inserted += 1

                    chunk_texts = chunk_text(cleaned_content)
                    if not chunk_texts:
                        continue

                    vectors = embeddings.embed_texts(chunk_texts)
                    data = [
                        {"text": chunk, "embedding": vector, "article_id": article_id, "department": department}
                        for chunk, vector in zip(chunk_texts, vectors)
                    ]
                    milvus_client.insert(collection_name=OFFLINE_DOCS_COLLECTION_NAME, data=data)
                    chunks_indexed += len(data)

        conn.commit()
    finally:
        conn.close()

    return {"articles_inserted": articles_inserted, "chunks_indexed": chunks_indexed}


def search_offline_docs(query: str, top_k: int = 5) -> list[dict]:
    if not query.strip():
        raise ValueError("Query must not be empty.")

    client = _get_milvus_client()
    stats = client.get_collection_stats(OFFLINE_DOCS_COLLECTION_NAME)
    if int(stats.get("row_count", 0)) == 0:
        return []

    query_vector = embeddings.embed_query(query)
    results = client.search(
        collection_name=OFFLINE_DOCS_COLLECTION_NAME,
        data=[query_vector],
        anns_field="embedding",
        limit=top_k,
        output_fields=["text", "article_id", "department"],
    )

    hits: list[dict] = []
    for result_group in results:
        for hit in result_group:
            entity = hit.get("entity", {})
            distance = hit.get("distance")
            hits.append(
                {
                    "text": entity.get("text"),
                    "article_id": entity.get("article_id"),
                    "department": entity.get("department"),
                    "score": float(distance) if distance is not None else 0.0,
                }
            )

    return hits
```

- [ ] **Step 4: Run the test, confirm it passes**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_offline_docs_store.py
```
Expected: `PASSED: offline_docs_store.py works end to end.`, exit 0. Requires Milvus and MariaDB
(`06_Local_RAG_Agent/docker-compose.yml`, port 3307) both reachable.

- [ ] **Step 5: Commit**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
git add backend/app/orchestrator/offline_docs_store.py scripts/test_offline_docs_store.py
git commit -m "Add offline_docs_store.py: MariaDB ground truth + shared Milvus collection for company docs"
```

---

### Task 4: Build `numeric_data_store.py`

**Files:**
- Create: `backend/app/orchestrator/numeric_data_store.py`
- Test: `scripts/test_numeric_data_store.py` (new)

**Interfaces:**
- Consumes: `app.config.MARIADB_SQLALCHEMY_URI`, `app.config.OPENAI_API_KEY`, `app.config.OPENAI_MODEL` (all existing/Task 1).
- Produces: `ensure_users_table() -> None`, `seed_test_employees(rows: list[dict]) -> int`, `query_numeric_data(nl_question: str) -> dict` (`{raw_sql, result}`, `result` is `list[dict]` of row mappings). Exposes `_engine` (SQLAlchemy engine) for test/cleanup use.

- [ ] **Step 1: Write the failing test — `scripts/test_numeric_data_store.py`**

```python
#!/usr/bin/env python
"""Test numeric_data_store.py: seed MariaDB users -> ask NL question -> verify generated SQL + result."""
from __future__ import annotations

import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from sqlalchemy import text

from app.orchestrator.numeric_data_store import _engine, query_numeric_data, seed_test_employees

_TEST_DEPARTMENT = "test-restructure-numeric"


def test_numeric_data_store() -> bool:
    print("=" * 70)
    print("NUMERIC DATA STORE TEST")
    print("=" * 70)

    seed_rows = [
        {"name": "Test Alpha", "department": _TEST_DEPARTMENT, "salary_grade": "G4", "remaining_vacation_day": 12},
        {"name": "Test Beta", "department": _TEST_DEPARTMENT, "salary_grade": "G5", "remaining_vacation_day": 20},
    ]

    try:
        print("\n[1/2] Seeding test employees...")
        seeded = seed_test_employees(seed_rows)
        assert seeded == 2, f"expected 2 seeded, got {seeded}"
        print(f"   OK: seeded {seeded} rows")

        print("\n[2/2] Asking a natural-language question...")
        question = f"How many employees are in the {_TEST_DEPARTMENT} department?"
        answer = query_numeric_data(question)
        print(f"   Generated SQL: {answer['raw_sql']}")
        print(f"   Result: {answer['result']}")

        assert "users" in answer["raw_sql"].lower(), f"expected query against users table: {answer['raw_sql']}"
        assert answer["raw_sql"].strip().upper().startswith("SELECT"), f"expected a SELECT: {answer['raw_sql']}"
        assert len(answer["result"]) > 0, f"expected a non-empty result: {answer}"

        first_row = answer["result"][0]
        count_value = next(iter(first_row.values()))
        assert int(count_value) == 2, f"expected count 2, got {first_row}"

        print("\nPASSED: numeric_data_store.py generates correct SQL against MariaDB.")
        return True

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return False

    finally:
        with _engine.begin() as conn:
            conn.execute(
                text("DELETE FROM users WHERE department = :department"),
                {"department": _TEST_DEPARTMENT},
            )
        print(f"[cleanup] removed MariaDB test rows for department '{_TEST_DEPARTMENT}'")


if __name__ == "__main__":
    success = test_numeric_data_store()
    sys.exit(0 if success else 1)
```

- [ ] **Step 2: Run it, confirm it fails**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_numeric_data_store.py
```
Expected: `ModuleNotFoundError: No module named 'app.orchestrator.numeric_data_store'`.

- [ ] **Step 3: Create `backend/app/orchestrator/numeric_data_store.py`**

```python
from __future__ import annotations

import re

from openai import OpenAI
from sqlalchemy import create_engine, text

from app.config import MARIADB_SQLALCHEMY_URI, OPENAI_API_KEY, OPENAI_MODEL

_engine = create_engine(MARIADB_SQLALCHEMY_URI)

_USERS_TABLE_SCHEMA = """
Table: users
Columns:
  - id INT PRIMARY KEY
  - name VARCHAR(100)
  - department VARCHAR(50)
  - salary_grade VARCHAR(10)
  - remaining_vacation_day INT
"""


def ensure_users_table() -> None:
    with _engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    department VARCHAR(50) NOT NULL,
                    salary_grade VARCHAR(10) NOT NULL,
                    remaining_vacation_day INT NOT NULL
                )
                """
            )
        )


def seed_test_employees(rows: list[dict]) -> int:
    if not rows:
        return 0

    ensure_users_table()
    with _engine.begin() as conn:
        for row in rows:
            conn.execute(
                text(
                    """
                    INSERT INTO users (name, department, salary_grade, remaining_vacation_day)
                    VALUES (:name, :department, :salary_grade, :remaining_vacation_day)
                    """
                ),
                row,
            )
    return len(rows)


def _clean_sql(raw_sql: str) -> str:
    cleaned = re.sub(r"```sql", "", raw_sql, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    match = re.search(r"(SELECT\s.*)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if match:
        cleaned = match.group(1)
    cleaned = cleaned.split(";")[0] + ";"
    return cleaned.strip()


def query_numeric_data(nl_question: str) -> dict:
    ensure_users_table()

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "You are a strict MariaDB architect. Given the table schema below, translate the "
        "question into a single MariaDB SELECT statement. Rules: no explanation, no markdown, "
        "output only one statement starting with SELECT.\n\n"
        f"Schema:\n{_USERS_TABLE_SCHEMA}\n\n"
        f"Question: {nl_question}\n\nSQL:"
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw_sql = response.choices[0].message.content or ""
    sql = _clean_sql(raw_sql)

    with _engine.connect() as conn:
        result = conn.execute(text(sql))
        rows = [dict(row._mapping) for row in result]

    return {"raw_sql": sql, "result": rows}
```

- [ ] **Step 4: Run the test, confirm it passes**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_numeric_data_store.py
```
Expected: `PASSED: numeric_data_store.py generates correct SQL against MariaDB.`, exit 0. Requires
MariaDB reachable and a valid `OPENAI_API_KEY` in `00_RAG_Document_Chat/.env`.

- [ ] **Step 5: Commit**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
git add backend/app/orchestrator/numeric_data_store.py scripts/test_numeric_data_store.py
git commit -m "Add numeric_data_store.py: MariaDB-backed NL-to-SQL for employee data"
```

---

## Self-Review

**Spec coverage:**
- Realtime PDF → Milvus, one shared collection, mutable (add/delete) → Task 2, fully replacing Chroma with the same 6-function contract so no caller changes.
- Offline docs → MariaDB ground truth + one shared Milvus collection, department as metadata, `article_id` back-reference → Task 3, tested explicitly for the back-reference.
- Numeric data → MariaDB only, no SQLite → Task 4, `numeric_data_store.py` never touches SQLite.
- "Fully replace Chroma, change anywhere it's used" → `vector_store.py` rewritten (Task 2) and `test_rag_retrieval.py`, the one other file directly touching Chroma's raw API, fixed in the same task; `upload.py`/`documents.py`/`evaluation.py` need no changes since they only use the six wrapper functions.
- Testing (your requested step 3) → every task is test-first, with real assertions and cleanup, following this repo's existing standalone-script convention rather than introducing pytest.
- Leave-policy / Step-1-copied orchestrator files / session-user work → untouched, not referenced by any new module.

**Placeholder scan:** every step has complete, runnable code — no "add error handling" or "TBD" stand-ins.

**Type/naming consistency:** `Chunk`/`PageText` (from `app/rag/types.py`), `chunk_text()` (from `app/rag/chunker.py`), `embed_texts()`/`embed_query()` (from `app/rag/embeddings.py`) are used with their existing signatures throughout — no renames. The six `vector_store.py` function names and return shapes are preserved exactly from the pre-existing Chroma version.

**Environment risk called out explicitly** (not assumed): `backend/.venv` was empty/stale before this session; Docker daemon and Milvus were not running at design time. These are stated as prerequisites in Global Constraints rather than silently assumed to work.

---

**Plan complete and saved to `00_RAG_Document_Chat/docs/superpowers/plans/2026-08-06-database-restructure-implementation.md`.** Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — I execute the 4 tasks in this session using `executing-plans`, with checkpoints for you to review.

Which approach?
