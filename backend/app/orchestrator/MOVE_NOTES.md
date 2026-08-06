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