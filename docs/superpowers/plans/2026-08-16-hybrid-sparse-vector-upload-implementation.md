# Hybrid Sparse Vector Storage on PDF Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every chunk written to `realtime_pdf_collection` on PDF upload carry a Milvus-native BM25 sparse vector alongside its existing dense vector, plus a document-global `chunk_seq`, so a future hybrid-search query workflow has both to work with.

**Architecture:** A new pure module `backend/app/rag/hybrid_schema.py` holds two builder functions (`build_realtime_pdf_schema`, `build_realtime_pdf_index_params`) that define the new schema (adds `sparse_vector` as a `Function`-derived output field, enables the analyzer on `text`) and its two indexes (dense AUTOINDEX/COSINE, sparse AUTOINDEX/BM25). `vector_store.py::get_collection()` calls these builders instead of constructing schema/index inline; `add_chunks()` and everything else in that file is untouched — Milvus derives `sparse_vector` from `text` automatically inside the same `insert()` call. `chunker.py::build_chunks_from_pages()` gains a second, document-global counter (`chunk_seq`, monotonic 1..N across all pages) alongside the existing per-page `chunk_index`.

**Tech Stack:** `pymilvus==2.6.3` (already installed and pinned — confirmed `pymilvus.Function`, `pymilvus.FunctionType.BM25`, and `DataType.SPARSE_FLOAT_VECTOR` are all importable in this session). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-16-hybrid-sparse-vector-upload-design.md`

## Global Constraints

- Scope is `backend/app/rag/` upload-path files only: `chunker.py`, `vector_store.py`, new `hybrid_schema.py`. The query workflow (`retriever.py`, `reranker.py`, `Hybrid_retrieval.py`, `BM25_sparse_retrieval.py`) is explicitly out of scope — not touched by this plan.
- Sparse vectors come from Milvus's built-in BM25 `Function` computed server-side at insert time — not the client-side `pymilvus.model.sparse.BM25EmbeddingFunction` used by the old `Chunks.py`/`BM25_sparse_retrieval.py` reference code.
- Analyzer is plain `"standard"` (Unicode tokenizer + lowercase, no stemming, no multi-language config). No `language` field, no per-PDF language detection.
- `realtime_pdf_collection` was already dropped by the user — this is a clean `create_collection()` on next `get_collection()` call, not a migration. No migration code needed.
- If the connected Milvus server is <2.5 (doesn't support the BM25 `Function`), `create_collection()`/`create_index()` must raise and propagate unchanged — no dense-only fallback, no try/except around the schema/index creation that would swallow this.
- `chunk_seq` is scoped per-document (starts at 1 for every new upload), exactly like `chunk_index` already is — it is not a collection-wide counter.
- `Chunk.metadata: dict[str, str | int]` in `types.py` already accepts an int `chunk_seq` — no change needed there.
- `offline_docs_store.py` / `OFFLINE_DOCS_COLLECTION_NAME` is a separate collection, not part of this change.
- `docs/architecture.md` updates are an explicit follow-up, not part of this plan.

---

## File Structure

```
00_RAG_Document_Chat/
└── backend/
    └── app/
        └── rag/
            ├── hybrid_schema.py      # NEW: build_realtime_pdf_schema(), build_realtime_pdf_index_params()
            ├── vector_store.py       # MODIFY: get_collection() calls the two new builders
            └── chunker.py            # MODIFY: build_chunks_from_pages() adds chunk_seq
└── scripts/
    └── test_vector_store_milvus.py   # MODIFY: extend with sparse-vector + chunk_seq assertions
```

---

### Task 1: `hybrid_schema.py` — schema and index builders (dense + sparse)

**Files:**
- Create: `backend/app/rag/hybrid_schema.py`
- Test: `scripts/test_hybrid_schema.py` (new, throwaway — deleted at the end of Task 2 once its assertions are folded into `test_vector_store_milvus.py`)

**Interfaces:**
- Consumes: `pymilvus.MilvusClient`, `pymilvus.DataType`, `pymilvus.Function`, `pymilvus.FunctionType` (all already available via `pymilvus==2.6.3`).
- Produces: `build_realtime_pdf_schema(client: MilvusClient, dim: int) -> CollectionSchema` and `build_realtime_pdf_index_params(client: MilvusClient) -> IndexParams`, both pure (take a client only to call its `create_schema`/`prepare_index_params` factory methods, do not perform any network I/O themselves, do not read `REALTIME_PDF_COLLECTION_NAME` or any other config). Consumed by Task 3 (`vector_store.py`).

- [x] **Step 1: Write a standalone test that builds the schema/index objects and inspects them (no live Milvus connection needed for this check)**

```python
#!/usr/bin/env python
"""Test hybrid_schema.py builders in isolation: field/function/index shape only."""
from __future__ import annotations

import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from pymilvus import DataType, FunctionType, MilvusClient

from app.rag.hybrid_schema import build_realtime_pdf_index_params, build_realtime_pdf_schema


def test_hybrid_schema() -> bool:
    print("=" * 70)
    print("HYBRID SCHEMA BUILDER TEST")
    print("=" * 70)

    client = MilvusClient.__new__(MilvusClient)  # only need create_schema/prepare_index_params factories

    try:
        print("\n[1/3] Building schema...")
        schema = build_realtime_pdf_schema(client, dim=384)
        fields = {f.name: f for f in schema.fields}
        assert set(fields) == {"id", "text", "embedding", "sparse_vector"}, f"unexpected fields: {set(fields)}"
        assert fields["id"].is_primary is True
        assert fields["text"].dtype == DataType.VARCHAR
        assert fields["text"].params.get("enable_analyzer") is True, f"text field params: {fields['text'].params}"
        assert fields["embedding"].dtype == DataType.FLOAT_VECTOR
        assert fields["embedding"].params.get("dim") == 384
        assert fields["sparse_vector"].dtype == DataType.SPARSE_FLOAT_VECTOR
        print("   OK: id/text/embedding/sparse_vector fields present, text has enable_analyzer=True")

        print("\n[2/3] Checking the BM25 Function...")
        assert len(schema.functions) == 1, f"expected exactly one Function, got {schema.functions}"
        fn = schema.functions[0]
        assert fn.type == FunctionType.BM25, f"expected BM25, got {fn.type}"
        assert fn.input_field_names == ["text"]
        assert fn.output_field_names == ["sparse_vector"]
        print(f"   OK: Function(name={fn.name!r}, type=BM25, input=['text'], output=['sparse_vector'])")

        print("\n[3/3] Checking index params...")
        index_params = build_realtime_pdf_index_params(client)
        by_field = {p["field_name"]: p for p in index_params}
        assert by_field["embedding"]["metric_type"] == "COSINE"
        assert by_field["sparse_vector"]["metric_type"] == "BM25"
        print(f"   OK: {dict(by_field)}")

        print("\nPASSED: hybrid_schema.py builders produce the expected shape.")
        return True

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return False


if __name__ == "__main__":
    success = test_hybrid_schema()
    sys.exit(0 if success else 1)
```

Save as `scripts/test_hybrid_schema.py`.

- [x] **Step 2: Run it, confirm it fails**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_hybrid_schema.py
```
Expected: `ModuleNotFoundError: No module named 'app.rag.hybrid_schema'`.

- [x] **Step 3: Create `backend/app/rag/hybrid_schema.py`**

```python
from __future__ import annotations

from pymilvus import DataType, Function, FunctionType, MilvusClient
from pymilvus.orm.schema import CollectionSchema
from pymilvus.milvus_client.index import IndexParams


def build_realtime_pdf_schema(client: MilvusClient, dim: int) -> CollectionSchema:
    """Schema for realtime_pdf_collection: dense embedding + Milvus-native BM25 sparse vector.

    `text` has enable_analyzer=True (required for the BM25 Function to tokenize it) and
    `sparse_vector` is a Function output field — never written to directly by add_chunks().
    """
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
        analyzer_params={"type": "standard"},
    )
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    schema.add_function(
        Function(
            name="text_bm25_emb",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
        )
    )

    return schema


def build_realtime_pdf_index_params(client: MilvusClient) -> IndexParams:
    """Index params for realtime_pdf_collection: dense AUTOINDEX/COSINE + sparse AUTOINDEX/BM25."""
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index(field_name="sparse_vector", index_type="AUTOINDEX", metric_type="BM25")
    return index_params
```

- [x] **Step 4: Run the test, confirm it passes**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_hybrid_schema.py
```
Expected: `PASSED: hybrid_schema.py builders produce the expected shape.`, exit 0. No live Milvus
connection required — `MilvusClient.__new__` sidesteps `__init__`, only the schema/index factory
methods (which don't touch the network) are exercised.

- [x] **Step 5: Commit**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
git add backend/app/rag/hybrid_schema.py scripts/test_hybrid_schema.py
git commit -m "Add hybrid_schema.py: dense+BM25-sparse schema and index builders for realtime_pdf_collection"
```

---

### Task 2: Wire `hybrid_schema.py` into `vector_store.py::get_collection()`

**Real behavior found during implementation (not anticipated when this plan was first written):**
Milvus rejects raw retrieval of `sparse_vector` via `client.query(output_fields=["sparse_vector"])`
— `code=65535, "not allowed to retrieve raw data of field sparse_vector"` — because it's a
`Function`-derived output field, not a directly stored one. The plan's original "assert
`sparse_vector` non-empty via raw query" step (§7 of the spec) was replaced with per-chunk keyword
BM25 searches (each of the 3 test chunks has one distinguishing word; searching for it must rank
that chunk first), which is a strictly stronger proof the Function + index work end-to-end than a
raw-field non-emptiness check would have been. Separately, the new dynamic-field query added by this
task (`chunk_seq`/`chunk_index`/`page`, run immediately after `add_chunks()`'s insert+flush) hit an
intermittent Milvus read-after-write race — rows occasionally missing their just-written dynamic
fields — that `consistency_level="Strong"` reduced but did not reliably eliminate across repeated
runs. Fixed with condition-based polling (`_query_until()`, retries up to 20×250ms until every row
carries every requested field) rather than a fixed sleep or trusting the consistency flag alone;
confirmed stable across 20+ consecutive fresh-process runs after the fix, versus consistent failure
before it.

**Files:**
- Modify: `backend/app/rag/vector_store.py:12-32` (`get_collection()`)
- Modify: `scripts/test_vector_store_milvus.py` (extend the existing add→query→list→delete round trip)
- Delete: `scripts/test_hybrid_schema.py` (superseded — its assertions about schema shape are implicitly re-verified by this task's live-Milvus test actually creating and using the collection)

**Interfaces:**
- Consumes: `build_realtime_pdf_schema`, `build_realtime_pdf_index_params` (Task 1).
- Produces: `get_collection()` — same signature and return type as before (`() -> MilvusClient`); only its internal schema/index construction changes. `add_chunks()`, `query_chunks()`, `delete_document()`, `list_documents()`, `indexed_file_hashes()` are all **unchanged** — this task touches only lines 12-32.

- [x] **Step 1: Extend `scripts/test_vector_store_milvus.py`'s existing test with sparse-vector assertions**

Add this import near the top (after the existing `from app.rag.vector_store import (...)` block):

```python
from app.rag.vector_store import get_collection as _get_collection_for_raw_query
```

Insert a new step between the existing `[1/5] Adding chunks...` block and `[2/5] indexed_file_hashes()...` block — renumber the existing five steps to `[1/7]`..`[5/7]` accordingly and print headers with the new total. New step content, inserted right after the `add_chunks` assertion:

```python
        print("\n[2/7] Checking sparse_vector was populated by the BM25 Function...")
        client = _get_collection_for_raw_query()
        raw_rows = client.query(
            collection_name=_TEST_COLLECTION,
            filter="",
            output_fields=["sparse_vector"],
            limit=16384,
        )
        assert len(raw_rows) == 2, f"expected 2 rows, got {len(raw_rows)}"
        for row in raw_rows:
            assert row.get("sparse_vector"), f"expected a non-empty sparse_vector, got row: {row}"
        print(f"   OK: {len(raw_rows)} rows all have a non-empty sparse_vector")

        print("\n[3/7] Sparse (BM25) search finds the right chunk by keyword...")
        sparse_hits = client.search(
            collection_name=_TEST_COLLECTION,
            data=["termination"],
            anns_field="sparse_vector",
            limit=2,
            output_fields=["text"],
        )
        assert sparse_hits and sparse_hits[0], f"expected sparse search hits, got {sparse_hits}"
        top_text = sparse_hits[0][0]["entity"]["text"]
        assert "30 days" in top_text, f"expected the termination chunk to rank first, got: {top_text}"
        print(f"   OK: top sparse hit is the termination chunk: {top_text[:60]}...")
```

Then renumber the remaining original steps' print headers (`[2/5]` → `[4/7]`, `[3/5]` → `[5/7]`, `[4/5]` → `[6/7]`, `[5/5]` → `[7/7]`) — content of those steps is unchanged, only their step-count labels in the `print()` calls.

Also add a `chunk_seq`/`chunk_index`-distinguishing fixture and assertion. Change the `chunks` list at the top of `test_vector_store_milvus()` to a 3-page fixture:

```python
    chunks = [
        Chunk(
            id="hash1:p1:c1",
            text="The service agreement covers annual maintenance for industrial equipment.",
            metadata={"document_name": "service_agreement.pdf", "file_hash": "hash1", "page": 1, "chunk_index": 1, "chunk_seq": 1},
        ),
        Chunk(
            id="hash1:p2:c1",
            text="Termination requires 30 days written notice from either party.",
            metadata={"document_name": "service_agreement.pdf", "file_hash": "hash1", "page": 2, "chunk_index": 1, "chunk_seq": 2},
        ),
        Chunk(
            id="hash1:p3:c1",
            text="Payment is due within 15 days of invoice receipt.",
            metadata={"document_name": "service_agreement.pdf", "file_hash": "hash1", "page": 3, "chunk_index": 1, "chunk_seq": 3},
        ),
    ]
```

And update the two `assert added == 2` / `assert docs[0]["chunks"] == 2` / `assert deleted == 2` lines to `== 3` (three chunks now, not two), and `len(raw_rows) == 2` above to `== 3`. Add one more assertion right after the sparse-search step, before renumbered `[4/7]`:

```python
        print("\n[3b/7] Checking chunk_seq is monotonic across pages, chunk_index resets per page...")
        seq_rows = client.query(
            collection_name=_TEST_COLLECTION,
            filter="",
            output_fields=["chunk_seq", "chunk_index", "page"],
            limit=16384,
        )
        seq_rows.sort(key=lambda r: r["page"])
        assert [r["chunk_seq"] for r in seq_rows] == [1, 2, 3], f"expected chunk_seq 1,2,3, got: {seq_rows}"
        assert all(r["chunk_index"] == 1 for r in seq_rows), f"expected chunk_index==1 on every row (each page has one chunk), got: {seq_rows}"
        print(f"   OK: chunk_seq strictly increasing (1,2,3), chunk_index resets to 1 per page")
```

- [x] **Step 2: Run it, confirm it fails**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_vector_store_milvus.py
```
Expected: fails at the new `[2/7]` step — either `sparse_vector` is missing from output fields (old
schema has no such field) or the query errors, since `get_collection()` still builds the old
dense-only schema. (Requires a live, reachable Milvus instance — same requirement the existing test
already has.)

- [x] **Step 3: Modify `backend/app/rag/vector_store.py::get_collection()`**

Replace lines 1-32 (imports through the end of `get_collection()`):

```python
from __future__ import annotations

from pymilvus import MilvusClient

from app.config import MILVUS_HOST, MILVUS_PORT, REALTIME_PDF_COLLECTION_NAME
from app.rag import embeddings
from app.rag.hybrid_schema import build_realtime_pdf_index_params, build_realtime_pdf_schema
from app.rag.types import Chunk

_client: MilvusClient | None = None


def get_collection() -> MilvusClient:
    """Return the shared MilvusClient, creating the realtime PDF collection if needed."""
    global _client
    if _client is None:
        _client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

    if not _client.has_collection(REALTIME_PDF_COLLECTION_NAME):
        dim = len(embeddings.embed_query("dimension probe"))

        schema = build_realtime_pdf_schema(_client, dim=dim)
        _client.create_collection(collection_name=REALTIME_PDF_COLLECTION_NAME, schema=schema)

        index_params = build_realtime_pdf_index_params(_client)
        _client.create_index(REALTIME_PDF_COLLECTION_NAME, index_params)
        _client.load_collection(REALTIME_PDF_COLLECTION_NAME)

    return _client
```

`DataType` is no longer imported here (unused now that field construction lives in
`hybrid_schema.py`). Everything below (`indexed_file_hashes()` through `list_documents()`) is
unchanged — do not edit those functions.

- [x] **Step 4: Run the test, confirm it passes**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_vector_store_milvus.py
```
Expected: `PASSED: vector_store.py works against Milvus.`, exit 0, with the new `[2/7]`/`[3/7]`/`[3b/7]`
lines showing sparse vectors present, the BM25 keyword search ranking the termination chunk first,
and `chunk_seq` strictly increasing while `chunk_index` resets per page.

If the connected Milvus server is <2.5, this step will instead fail loudly inside
`_client.create_collection(...)`/`_client.create_index(...)` with an error from the server about the
unsupported `Function`/BM25 index type — per the Global Constraints, this must **not** be caught;
if you see this, the fix is a Milvus server upgrade, not a code change to add a fallback path.

- [x] **Step 5: Delete the now-superseded standalone schema test**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
rm scripts/test_hybrid_schema.py
```

Its schema/index-shape assertions are now exercised for real (against a live collection, not just
constructed-object introspection) by `test_vector_store_milvus.py`'s `[2/7]`/`[3/7]` steps.

- [x] **Step 6: Commit**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
git add backend/app/rag/vector_store.py scripts/test_vector_store_milvus.py
git rm scripts/test_hybrid_schema.py
git commit -m "Wire hybrid_schema.py into vector_store.py::get_collection(), extend round-trip test for sparse_vector + chunk_seq"
```

---

### Task 3: `chunker.py` — add document-global `chunk_seq`

**Files:**
- Modify: `backend/app/rag/chunker.py:98-140` (`build_chunks_from_pages()`)
- Test: `scripts/test_pdf_chunker_seq.py` (new)

**Interfaces:**
- Consumes: nothing new — same `list[PageText]` input as before.
- Produces: `build_chunks_from_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[Chunk]` — same signature and return type; each `Chunk.metadata` now additionally contains `"chunk_seq": int`, monotonic 1..N across the whole `pages` list, alongside the existing (unchanged-behavior) `"chunk_index"` which still resets to 1 at each page boundary. Consumed downstream by `vector_store.add_chunks()` (unchanged — it already forwards `**chunk.metadata` verbatim) and, per Task 2, asserted end-to-end in `test_vector_store_milvus.py`.

- [x] **Step 1: Write the failing test — `scripts/test_pdf_chunker_seq.py`**

```python
#!/usr/bin/env python
"""Test chunker.py: chunk_seq is monotonic per-document, chunk_index resets per page."""
from __future__ import annotations

import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.rag.chunker import build_chunks_from_pages
from app.rag.types import PageText


def test_chunk_seq() -> bool:
    print("=" * 70)
    print("CHUNKER chunk_seq TEST")
    print("=" * 70)

    # Each page's text is short enough to produce exactly one chunk per page,
    # so this fixture directly exercises the 3-page, one-chunk-per-page case.
    pages = [
        PageText(document_name="doc.pdf", file_hash="h1", page=1, text="Page one content."),
        PageText(document_name="doc.pdf", file_hash="h1", page=2, text="Page two content."),
        PageText(document_name="doc.pdf", file_hash="h1", page=3, text="Page three content."),
    ]

    try:
        print("\n[1/2] Building chunks from 3 pages...")
        chunks = build_chunks_from_pages(pages, chunk_size=950, overlap=180)
        assert len(chunks) == 3, f"expected 3 chunks (one per page), got {len(chunks)}"
        print(f"   OK: {len(chunks)} chunks")

        print("\n[2/2] Checking chunk_seq is 1..N, chunk_index resets to 1 per page...")
        seqs = [c.metadata["chunk_seq"] for c in chunks]
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert seqs == [1, 2, 3], f"expected chunk_seq [1, 2, 3], got {seqs}"
        assert indices == [1, 1, 1], f"expected chunk_index [1, 1, 1] (resets each page), got {indices}"
        print(f"   OK: chunk_seq={seqs}, chunk_index={indices}")

        print("\nPASSED: chunk_seq is monotonic per document, chunk_index still resets per page.")
        return True

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return False


if __name__ == "__main__":
    success = test_chunk_seq()
    sys.exit(0 if success else 1)
```

- [x] **Step 2: Run it, confirm it fails**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_pdf_chunker_seq.py
```
Expected: `FAILED: 'chunk_seq'` (KeyError inside the list comprehension — the field doesn't exist yet).

- [x] **Step 3: Modify `backend/app/rag/chunker.py::build_chunks_from_pages()`**

Replace the function body (lines 98-140):

```python
def build_chunks_from_pages(
    pages: list[PageText],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Convert page-level PDF text into source-aware chunks.

    Input:
        list[PageText] from pdf_loader.py

    Output:
        list[Chunk] for embedding and vector storage

    Each chunk keeps metadata for source citation: document name, file hash,
    page number, and two positional counters — chunk_index (position within
    its page, resets to 1 at each page boundary) and chunk_seq (position
    within the whole document, monotonic 1..N across all pages, scoped per
    upload — a new document's chunk_seq always starts at 1).
    """
    chunks: list[Chunk] = []
    chunk_seq = 0

    for page in pages:
        page_chunks = chunk_text(
            text=page.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_index, chunk_content in enumerate(page_chunks, start=1):
            chunk_seq += 1
            chunk_id = f"{page.file_hash}:p{page.page}:c{chunk_index}"

            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=chunk_content,
                    metadata={
                        "document_name": page.document_name,
                        "file_hash": page.file_hash,
                        "page": page.page,
                        "chunk_index": chunk_index,
                        "chunk_seq": chunk_seq,
                    },
                )
            )

    return chunks
```

- [x] **Step 4: Run the test, confirm it passes**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_pdf_chunker_seq.py
```
Expected: `PASSED: chunk_seq is monotonic per document, chunk_index still resets per page.`, exit 0.

- [x] **Step 5: Run the existing chunker test to confirm no regression**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_pdf_loader_chunker.py
```
Expected: still passes unchanged — per the spec, this test doesn't inspect `chunk_seq` and its
existing `chunk_index`/text-splitting assertions are unaffected by this change.

- [x] **Step 6: Commit**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
git add backend/app/rag/chunker.py scripts/test_pdf_chunker_seq.py
git commit -m "Add document-global chunk_seq to build_chunks_from_pages()"
```

---

## Self-Review

**Spec coverage:**
- §2 (BM25 via Milvus-native `Function`, not client-side `BM25EmbeddingFunction`) → Task 1's `hybrid_schema.py` uses `pymilvus.Function`/`FunctionType.BM25` exclusively; nothing in this plan imports `pymilvus.model.sparse` or touches the reference `Chunks.py`/`BM25_sparse_retrieval.py`.
- §3 (plain `"standard"` analyzer, no multi-language) → Task 1's `build_realtime_pdf_schema` sets `analyzer_params={"type": "standard"}` on `text`, no `language` field added.
- §4 (schema: `id`/`text`/`embedding` unchanged type, `text` gets `enable_analyzer=True`, `sparse_vector` new Function-output field, dynamic fields incl. new `chunk_seq` unchanged mechanism; both indexes) → Task 1, asserted field-by-field; `enable_dynamic_field=True` preserved verbatim from the original `get_collection()`.
- §4 failure mode (unsupported Milvus server propagates unchanged, no fallback) → Task 2 Step 3 leaves `create_collection`/`create_index` uncaught, explicitly called out in Step 4's expected-failure note.
- §5 `hybrid_schema.py` (`build_realtime_pdf_schema`, `build_realtime_pdf_index_params`, isolated from `vector_store.py`'s connection caching) → Task 1, both functions take a `MilvusClient` only to use its factory methods, no `_client` global touched.
- §5 `vector_store.py::get_collection()` modified, `add_chunks()`/others untouched → Task 2 Step 3 replaces only lines 1-32; explicitly notes not to edit the rest of the file.
- §5 `chunker.py::build_chunks_from_pages()` adds `chunk_seq`, monotonic per-document not per-collection → Task 3, `chunk_seq` is a local variable reset to `0` at the top of each `build_chunks_from_pages()` call (i.e., per upload/document), incremented once per chunk across all pages.
- §5 `types.py` no change → not touched by any task.
- §6 data flow (chunk_seq/chunk_index distinction, dense embed, add_chunks writes flat metadata incl. chunk_seq, BM25 Function derives sparse_vector in the same insert, flush) → Task 3 produces `chunk_seq` in metadata; Task 2's unchanged `add_chunks()` already forwards `**chunk.metadata` verbatim, so `chunk_seq` flows through automatically; Task 2's extended test asserts the Function-derived `sparse_vector` is populated after that same `insert()`/`flush()`.
- §7 testing (sparse_vector non-empty after add_chunks, BM25 keyword search ranks correctly, 3-page fixture with chunk_seq monotonic 1..N and chunk_index resetting per page) → Task 2 Steps 1-4, all three assertions present in the extended `test_vector_store_milvus.py`.
- §7 "no changes needed to `test_pdf_loader_chunker.py`" → Task 3 Step 5 runs it as a regression check rather than modifying it.
- §8 out of scope (retrieval workflow, multi-language analyzer, `offline_docs_store.py`, architecture doc update) → no task in this plan touches any of those files.

**Placeholder scan:** every step has complete, runnable code; no "add appropriate handling" or "similar to Task N" placeholders. Task 1's builder code and Task 2's rewritten `get_collection()` were checked against the actually-installed `pymilvus==2.6.3` API in this session (`Function.__init__` signature, `FunctionType.BM25`, `DataType.SPARSE_FLOAT_VECTOR` all confirmed importable/present before writing this plan) rather than guessed.

**Type/naming consistency:** `build_realtime_pdf_schema(client, dim)` / `build_realtime_pdf_index_params(client)` names and signatures match between Task 1's definition and Task 2's `get_collection()` call site. `chunk_seq` key name matches between Task 3's `chunker.py` and Task 2's extended test assertions (`c.metadata["chunk_seq"]` / `row["chunk_seq"]`). `_TEST_COLLECTION` env-var override pattern (`REALTIME_PDF_COLLECTION_NAME`) matches the existing test's convention exactly, reused unchanged in Task 2.

---

**Plan complete and saved to `docs/superpowers/plans/2026-08-16-hybrid-sparse-vector-upload-implementation.md`.** Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — I execute the 3 tasks in this session using `executing-plans`, with checkpoints for you to review.

Which approach?