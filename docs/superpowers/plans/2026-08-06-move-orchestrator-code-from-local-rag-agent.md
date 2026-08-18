# Move Orchestrator Code from 06_Local_RAG_Agent into 00_RAG_Document_Chat — Step 1 (Relocation Only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Copy the query-time orchestrator code (`ai_orchestrator.py` and its retrieval dependencies) from `06_Local_RAG_Agent` into a new `backend/app/orchestrator/` package inside `00_RAG_Document_Chat`, with zero logic changes, so a later step can wire the chat endpoint to it.

**Architecture:** This is step 1 of a larger merge. It is pure file relocation (copy, not move): the five orchestrator-related Python modules plus `config.py` are duplicated byte-for-byte into `00_RAG_Document_Chat/backend/app/orchestrator/`. `06_Local_RAG_Agent` keeps its own untouched copies — it remains the standalone owner of the data/services (SQLite `enterprise_erp.db`, Milvus, MariaDB `wiki_db`, `company_documents/`, `data/leave_policy.pdf`) and its own scripts (`api_server.py`, `check_data/*`, `evaluate_breakdown.py`, `ai_orchestrator_walkthrough.ipynb`) continue to work unmodified because their imports still resolve against the originals. A later step (not this one) will fix imports in the new copies, point their config at 06's live data/services, and swap `backend/app/api/chat.py` to call the orchestrator instead of `rag/retriever.py` + `rag/generator.py`.

**Tech Stack:** Python (FastAPI backend, no new dependencies installed in this step), `cp`/`diff` for verification, `git` for commits inside the `00_RAG_Document_Chat` repo.

## Global Constraints

- No source code is edited, reformatted, or renamed in content — every copied file must be byte-identical to its source (verified with `diff`).
- Do not fix imports, do not touch `backend/app/api/chat.py`, do not merge dependencies into the active `backend/requirements.txt`, do not attempt to run or import the new package. Getting it running is explicitly out of scope for this step.
- Do not modify, move, or delete anything under `06_Local_RAG_Agent/` — it must remain fully functional as-is (its own `api_server.py`, `check_data/*`, `evaluate_breakdown.py`, and `ai_orchestrator_walkthrough.ipynb` all import these same modules from their current location and must keep working).
- All new files live under `00_RAG_Document_Chat/backend/app/orchestrator/`, a new Python package sibling to the existing `api/`, `rag/`, `data/`, and `utils/` packages (each of which has its own empty `__init__.py`, so this one does too).
- Every task ends with a `git commit` inside the `00_RAG_Document_Chat` repo (it has its own `.git`, separate from the rest of `Agentic_RAG`).

---

## File Structure

```
00_RAG_Document_Chat/backend/app/orchestrator/          # NEW package
├── __init__.py                          # empty, matches sibling package convention
├── ai_orchestrator.py                   # copy of 06/ai_orchestrator.py — the plan/route/SQL+RAG/merge brain
├── Chunks.py                            # copy of 06/Chunks.py — PDF/JSON chunking + Ollama embeddings + Milvus writes
├── Hybrid_retrieval.py                  # copy of 06/Hybrid_retrieval.py — combines dense + sparse search
├── Biencoder_dense_retrieval.py         # copy of 06/Biencoder_dense_retrieval.py — Milvus dense (semantic) search
├── BM25_sparse_retrieval.py             # copy of 06/BM25_sparse_retrieval.py — Milvus sparse (BM25) search
├── config.py                            # copy of 06/config.py — LLM_MODEL, MILVUS_HOST/PORT, get_db_connection(), etc.
├── requirements-local-rag-agent.txt     # copy of 06/requirements.txt — reference only, NOT merged into backend/requirements.txt
└── MOVE_NOTES.md                        # manifest: what moved, what's still external, what Step 2 must fix
```

Why these six `.py` files and no others: tracing `ai_orchestrator.py`'s imports, `run_enterprise_orchestrator()` needs `Chunks_Embedding` (to rebuild the leave-policy chunks + BM25 encoder at query time) and `HybridSearcher` (to actually query Milvus); `Hybrid_retrieval.py` in turn imports `Biencoder_dense_retrieval.DenseRetriever` and `BM25_sparse_retrieval.SparseRetriever`; everything imports `config` for model names, hosts, and paths. Nothing else in `06_Local_RAG_Agent` (ingestion scripts, `api_server.py`, `check_data/*`, the notebook, `evaluate_breakdown.py`, `enterprise_erp.db`, `company_documents/`, `data/`) is on that import chain, and per your instructions the data/services stay owned by `06_Local_RAG_Agent` — only the code that *queries* them moves.

---

### Task 1: Scaffold the orchestrator package and copy the core orchestrator module

**Files:**
- Create: `00_RAG_Document_Chat/backend/app/orchestrator/__init__.py`
- Create: `00_RAG_Document_Chat/backend/app/orchestrator/ai_orchestrator.py`
- Source: `06_Local_RAG_Agent/ai_orchestrator.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: the `orchestrator` package directory that Tasks 2–6 add files into. `ai_orchestrator.py` still contains its original (currently unresolvable) imports (`import config`, `from Chunks import Chunks_Embedding`, `from Hybrid_retrieval import HybridSearcher`) — Step 2 fixes these, not this task.

- [ ] **Step 1: Create the package directory and empty `__init__.py`**

```bash
mkdir -p /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat/backend/app/orchestrator
touch /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat/backend/app/orchestrator/__init__.py
```

- [ ] **Step 2: Copy `ai_orchestrator.py` verbatim**

```bash
cp /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/06_Local_RAG_Agent/ai_orchestrator.py \
   /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat/backend/app/orchestrator/ai_orchestrator.py
```

- [ ] **Step 3: Verify byte-identical copy**

Run:
```bash
diff /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/06_Local_RAG_Agent/ai_orchestrator.py \
     /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat/backend/app/orchestrator/ai_orchestrator.py
```
Expected: no output (files identical).

- [ ] **Step 4: Commit**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
git add backend/app/orchestrator/__init__.py backend/app/orchestrator/ai_orchestrator.py
git commit -m "Add orchestrator package skeleton, copy ai_orchestrator.py from 06_Local_RAG_Agent"
```

---

### Task 2: Copy the chunking/embedding module (`Chunks.py`)

**Files:**
- Create: `00_RAG_Document_Chat/backend/app/orchestrator/Chunks.py`
- Source: `06_Local_RAG_Agent/Chunks.py`

**Interfaces:**
- Consumes: the `orchestrator/` directory from Task 1.
- Produces: `Chunks_Embedding` class (unmodified) that `ai_orchestrator.py`'s `_policy_processor_for_hybrid()` instantiates. Still imports `config` and `pymilvus`/`langchain_ollama` — unresolved until Step 2.

- [ ] **Step 1: Copy verbatim**

```bash
cp /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/06_Local_RAG_Agent/Chunks.py \
   /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat/backend/app/orchestrator/Chunks.py
```

- [ ] **Step 2: Verify byte-identical copy**

Run:
```bash
diff /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/06_Local_RAG_Agent/Chunks.py \
     /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat/backend/app/orchestrator/Chunks.py
```
Expected: no output.

- [ ] **Step 3: Commit**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
git add backend/app/orchestrator/Chunks.py
git commit -m "Copy Chunks.py from 06_Local_RAG_Agent into orchestrator package"
```

---

### Task 3: Copy the hybrid retrieval subsystem (3 files)

These three files are one coherent subsystem (`Hybrid_retrieval.py` directly imports the other two), so they move together as one deliverable.

**Files:**
- Create: `00_RAG_Document_Chat/backend/app/orchestrator/Hybrid_retrieval.py`
- Create: `00_RAG_Document_Chat/backend/app/orchestrator/Biencoder_dense_retrieval.py`
- Create: `00_RAG_Document_Chat/backend/app/orchestrator/BM25_sparse_retrieval.py`
- Sources: `06_Local_RAG_Agent/Hybrid_retrieval.py`, `06_Local_RAG_Agent/Biencoder_dense_retrieval.py`, `06_Local_RAG_Agent/BM25_sparse_retrieval.py`

**Interfaces:**
- Consumes: the `orchestrator/` directory from Task 1.
- Produces: `HybridSearcher` (used by `ai_orchestrator.py`), `DenseRetriever`, `SparseRetriever` — all unmodified, still importing bare `config` and each other by bare module name (`from Biencoder_dense_retrieval import DenseRetriever`, etc.), unresolved until Step 2.

- [ ] **Step 1: Copy all three files verbatim**

```bash
SRC=/Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/06_Local_RAG_Agent
DST=/Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat/backend/app/orchestrator
cp "$SRC/Hybrid_retrieval.py" "$DST/Hybrid_retrieval.py"
cp "$SRC/Biencoder_dense_retrieval.py" "$DST/Biencoder_dense_retrieval.py"
cp "$SRC/BM25_sparse_retrieval.py" "$DST/BM25_sparse_retrieval.py"
```

- [ ] **Step 2: Verify all three are byte-identical**

Run:
```bash
SRC=/Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/06_Local_RAG_Agent
DST=/Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat/backend/app/orchestrator
diff "$SRC/Hybrid_retrieval.py" "$DST/Hybrid_retrieval.py"
diff "$SRC/Biencoder_dense_retrieval.py" "$DST/Biencoder_dense_retrieval.py"
diff "$SRC/BM25_sparse_retrieval.py" "$DST/BM25_sparse_retrieval.py"
```
Expected: no output from any of the three.

- [ ] **Step 3: Commit**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
git add backend/app/orchestrator/Hybrid_retrieval.py \
        backend/app/orchestrator/Biencoder_dense_retrieval.py \
        backend/app/orchestrator/BM25_sparse_retrieval.py
git commit -m "Copy hybrid retrieval subsystem (dense + sparse) from 06_Local_RAG_Agent"
```

---

### Task 4: Copy the orchestrator's config module and flag the naming collision

`00_RAG_Document_Chat` already has its own `backend/app/config.py` (CHROMA_DIR, TOP_K, OPENAI settings for the plain RAG pipeline). `06_Local_RAG_Agent`'s `config.py` is a *different* module (LLM_MODEL, OLLAMA_BASE_URL, MILVUS_HOST/PORT, LEAVE_POLICY_*, `get_db_connection()` for MariaDB). Copying it in verbatim under the same filename `config.py`, just in a different directory (`orchestrator/` vs `app/`), avoids an actual filesystem collision today, but every file copied in Tasks 1–3 does a bare `import config` — which is ambiguous the moment both are importable on the same path. This task copies the file as-is (per the "no code changes" constraint) and records the collision as a named follow-up rather than silently resolving it.

**Files:**
- Create: `00_RAG_Document_Chat/backend/app/orchestrator/config.py`
- Source: `06_Local_RAG_Agent/config.py`

**Interfaces:**
- Consumes: the `orchestrator/` directory from Task 1.
- Produces: `config.LLM_MODEL`, `config.OLLAMA_BASE_URL`, `config.MILVUS_HOST`, `config.MILVUS_PORT`, `config.LEAVE_POLICY_FILE_PATH`, `config.LEAVE_POLICY_COLLECTION_NAME`, `config.get_db_connection()` — all referenced by the files from Tasks 1–3. Not yet resolvable by Python's import system from this location (Step 2 problem, see `MOVE_NOTES.md` in Task 6).

- [ ] **Step 1: Copy verbatim**

```bash
cp /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/06_Local_RAG_Agent/config.py \
   /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat/backend/app/orchestrator/config.py
```

- [ ] **Step 2: Verify byte-identical copy**

Run:
```bash
diff /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/06_Local_RAG_Agent/config.py \
     /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat/backend/app/orchestrator/config.py
```
Expected: no output.

- [ ] **Step 3: Commit**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
git add backend/app/orchestrator/config.py
git commit -m "Copy config.py from 06_Local_RAG_Agent into orchestrator package"
```

---

### Task 5: Copy 06's `requirements.txt` as a reference file (not merged)

The orchestrator code depends on packages `00_RAG_Document_Chat/backend/requirements.txt` doesn't have yet (`langchain`, `langchain-core`, `langchain-community`, `langchain-ollama`, `langgraph`*, `pymilvus`, `pymilvus.model`, `rank-bm25`, `ollama`, `SQLAlchemy`). Per the Global Constraints, the active `backend/requirements.txt` is not touched in this step — this task only preserves 06's full pinned list alongside the new code so the diff is available when Step 2 does the merge.

*(`langgraph` is present in 06's `requirements.txt` but unused by `ai_orchestrator.py` — see `doc/adr/001-use-lcel-instead-of-langgraph.md` in `06_Local_RAG_Agent`; worth noting in `MOVE_NOTES.md` so Step 2 doesn't pull it in unnecessarily.)

**Files:**
- Create: `00_RAG_Document_Chat/backend/app/orchestrator/requirements-local-rag-agent.txt`
- Source: `06_Local_RAG_Agent/requirements.txt`

**Interfaces:**
- Consumes: the `orchestrator/` directory from Task 1.
- Produces: a reference file Step 2 will diff against `backend/requirements.txt` to decide what to add.

- [ ] **Step 1: Copy verbatim under the new filename**

```bash
cp /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/06_Local_RAG_Agent/requirements.txt \
   /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat/backend/app/orchestrator/requirements-local-rag-agent.txt
```

- [ ] **Step 2: Verify byte-identical copy**

Run:
```bash
diff /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/06_Local_RAG_Agent/requirements.txt \
     /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat/backend/app/orchestrator/requirements-local-rag-agent.txt
```
Expected: no output.

- [ ] **Step 3: Confirm the active backend requirements.txt was NOT touched**

Run:
```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
git status --short backend/requirements.txt
```
Expected: no output (file not modified — only the new reference file under `orchestrator/` is staged).

- [ ] **Step 4: Commit**

```bash
git add backend/app/orchestrator/requirements-local-rag-agent.txt
git commit -m "Add 06_Local_RAG_Agent's requirements.txt as a reference for the future dependency merge"
```

---

### Task 6: Write `MOVE_NOTES.md` — the manifest for Step 2

This file is the handoff document: what was copied, what's deliberately left in `06_Local_RAG_Agent`, and the exact list of things Step 2 (wiring) must fix before anything can run. Nothing here is code — it's the map.

**Files:**
- Create: `00_RAG_Document_Chat/backend/app/orchestrator/MOVE_NOTES.md`

**Interfaces:**
- Consumes: the file set produced by Tasks 1–5.
- Produces: documentation only; no other task depends on this file's content, but it is the spec Step 2's plan should be written against.

- [ ] **Step 1: Write the manifest**

```markdown
# Orchestrator Move Notes (Step 1 of the 00 + 06 merge)

## What was copied here (byte-identical to 06_Local_RAG_Agent, verified with `diff`)
- `ai_orchestrator.py` — plan/route/SQL+RAG/merge orchestrator (`run_enterprise_orchestrator()`)
- `Chunks.py` — PDF/JSON chunking, Ollama embeddings, Milvus writes (`Chunks_Embedding`)
- `Hybrid_retrieval.py` — combines dense + sparse search (`HybridSearcher`)
- `Biencoder_dense_retrieval.py` — Milvus dense/semantic search (`DenseRetriever`)
- `BM25_sparse_retrieval.py` — Milvus sparse/BM25 search (`SparseRetriever`)
- `config.py` — LLM_MODEL, OLLAMA_BASE_URL, MILVUS_HOST/PORT, LEAVE_POLICY_*, `get_db_connection()`
- `requirements-local-rag-agent.txt` — reference copy of 06's full pinned dependency list (not merged)

## What was deliberately NOT copied — stays owned by 06_Local_RAG_Agent
- `enterprise_erp.db` (SQLite) — employee/department ground-truth data queried by the SQL branch
- `data/leave_policy.pdf` — source PDF re-chunked at query time by `_policy_processor_for_hybrid()`
- `company_documents/` — markdown corpus ingested into MariaDB (`wiki_db`) by `data_ingestion/knowledge_ingest.py`
- Milvus (policy PDF vector store) and MariaDB (`wiki_db`, via `docker-compose.yml` in 06) — both services keep running out of 06
- `data_ingestion/*.py`, `api_server.py`, `check_data/*.py`, `evaluate_breakdown.py`, `ai_orchestrator_walkthrough.ipynb` — 06-only tooling, untouched, still import the originals in place

## Known breakage — NOT fixed in this step, required before anything here can run
1. **Bare imports won't resolve.** `ai_orchestrator.py`, `Chunks.py`, `Hybrid_retrieval.py`, `Biencoder_dense_retrieval.py`, and `BM25_sparse_retrieval.py` all use bare imports (`import config`, `from Chunks import Chunks_Embedding`, `from Hybrid_retrieval import HybridSearcher`, `from Biencoder_dense_retrieval import DenseRetriever`, `from BM25_sparse_retrieval import SparseRetriever`) that assumed all files sat flat in one directory on `sys.path`. Inside `backend/app/orchestrator/`, a FastAPI app that imports `app.main`, these need to become relative imports (`from . import config`, `from .Chunks import Chunks_Embedding`, etc.) or the package needs a `sys.path` shim.
2. **Two `config.py` modules will coexist**: `backend/app/config.py` (existing RAG pipeline settings) and `backend/app/orchestrator/config.py` (copied here). They serve different purposes and must both be imported unambiguously (e.g. `from app import config` vs `from app.orchestrator import config`) — no rename has been done yet.
3. **`enterprise_erp.db` path is hardcoded relative to `__file__`.** `run_enterprise_orchestrator()` does `os.path.join(os.path.dirname(os.path.abspath(__file__)), 'enterprise_erp.db')` — from its new location that resolves inside `00_RAG_Document_Chat`, not to 06's actual database file. Step 2 must point this at 06's file (e.g. via an env var / absolute path), since the data intentionally stays in `06_Local_RAG_Agent`.
4. **`config.LEAVE_POLICY_FILE_PATH` and `config.DATA_DIR` are relative to 06's `PROJECT_ROOT`.** Same issue as #3 — Step 2 must repoint these at 06's `data/leave_policy.pdf` rather than duplicating the PDF.
5. **MILVUS_HOST/MILVUS_PORT and MariaDB credentials** in the copied `config.py` currently assume "same machine as 06." Confirm these still resolve from wherever `00_RAG_Document_Chat`'s backend actually runs (same host vs. Docker container — see `00_RAG_Document_Chat/docker-compose.yml`, which runs the backend inside a container without network access to 06's services unless added to the same Docker network or reached via `host.docker.internal`).
6. **`backend/requirements.txt` is missing** everything the orchestrator needs: `langchain`, `langchain-core`, `langchain-community`, `langchain-ollama`, `pymilvus`, `pymilvus.model`, `rank-bm25`, `ollama`, `SQLAlchemy`. See `requirements-local-rag-agent.txt` for exact pinned versions. `langgraph`/`langgraph-*` can likely be skipped — 06's own ADR (`doc/adr/001-use-lcel-instead-of-langgraph.md`) says the orchestrator uses plain LCEL, not LangGraph.
7. **`backend/app/api/chat.py` still calls the old pipeline** (`app.rag.retriever.search_sources` + `app.rag.generator.answer_question`), not the orchestrator. Swapping this is the actual behavior change and is out of scope for this step.

## Suggested scope for Step 2 (not part of this plan)
- Fix imports (#1), resolve the `config.py` naming collision (#2)
- Make the SQLite path, leave-policy PDF path, and Milvus/MariaDB connection targets configurable via env vars pointing back at `06_Local_RAG_Agent`'s files/services (#3, #4, #5)
- Merge the missing dependencies into `backend/requirements.txt` (#6)
- Replace the direct-embedding call in `backend/app/api/chat.py` with a call into `orchestrator.ai_orchestrator.run_enterprise_orchestrator(question)` (#7)
- Decide Docker networking between the two projects' services (#5)
```

- [ ] **Step 2: Write the file**

Use the content above verbatim as the file body.

- [ ] **Step 3: Commit**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
git add backend/app/orchestrator/MOVE_NOTES.md
git commit -m "Document Step 1 orchestrator move and Step 2 follow-ups in MOVE_NOTES.md"
```

---

## Self-Review

**Spec coverage:**
- "Move necessary code from Local_RAG_Agent to RAG_Document_Chat" → Tasks 1–4 (the 6 files on `ai_orchestrator.py`'s actual import chain).
- "Do not change any code" → every task copies verbatim and verifies with `diff`; no edits anywhere.
- "Keep the frontend and upload PDF part" of `00_RAG_Document_Chat` → untouched; nothing in this plan touches `frontend/`, `backend/app/api/upload.py`, or `backend/app/rag/`.
- "I don't want the plain RAG pipeline, I want ai_orchestrator.py" → orchestrator code is now physically present in `00_RAG_Document_Chat`; the actual swap in `chat.py` is explicitly logged as out-of-scope (item #7 in `MOVE_NOTES.md`) since the user said no code changes this step.
- "Keep the data and Milvus/MariaDB in Local_RAG_Agent" → nothing under `06_Local_RAG_Agent/data*`, `enterprise_erp.db`, `company_documents/`, or `docker-compose.yml` is touched or copied; `MOVE_NOTES.md` explicitly enumerates this.
- "You don't need to make the project run" → no import fixes, no dependency install, no wiring; `MOVE_NOTES.md` documents exactly why it can't run yet.

**Placeholder scan:** No task contains "TBD"/"handle appropriately"/etc. — every step is an exact, copy-pasteable shell command with an exact expected output.

**Type/naming consistency:** File and function names referenced (`Chunks_Embedding`, `HybridSearcher`, `DenseRetriever`, `SparseRetriever`, `run_enterprise_orchestrator`, `get_db_connection`) match their originals exactly, since these are unmodified copies — no renames introduced.

---

**Plan complete and saved to `00_RAG_Document_Chat/docs/superpowers/plans/2026-08-06-move-orchestrator-code-from-local-rag-agent.md`.** Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — I execute the 6 tasks in this session using `executing-plans`, with checkpoints for you to review.

Which approach?