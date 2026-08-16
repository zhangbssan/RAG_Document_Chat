# Hybrid Query Retrieval (Dense + BM25 + Anchor/Context Assembly) — Design

**Date:** 2026-08-16
**Scope:** The query workflow only — `backend/app/rag/retriever.py`, `backend/app/rag/reranker.py` (deleted), a new `backend/app/rag/hybrid_search.py`, `backend/app/agent/tools.py`, `backend/app/agent/chat_agent.py`, `backend/app/schemas.py`, `backend/app/rag/evaluator.py`, `backend/app/rag/generator.py`, `frontend/app.py`. Builds directly on top of the storage laid down by `docs/superpowers/specs/2026-08-16-hybrid-sparse-vector-upload-design.md` (native BM25 `sparse_vector` field, document-global `chunk_seq`) — that spec explicitly deferred this work.

## 1. Goal

Replace today's two retrieval paths — Path A (`retriever.py`: dense search → lexical-overlap RRF `reranker.py`) and Path B (`tools.py`'s `search_uploaded_docs` tool: dense search only, no reranking) — with one shared hybrid retrieval pipeline used by both:

```
question
  ├── dense search (embedding, top-k)
  └── BM25 sparse search (Milvus-native FTS, top-k)
        ↓
  merge + dedup by chunk id
        ↓
  Reciprocal Rank Fusion (RRF)
        ↓
  select top-N anchor chunks (default 2)
        ↓
  build the (document, chunk_seq) list each anchor needs (±1 window)
        ↓
  dedup that list (so overlapping anchor windows aren't fetched twice)
        ↓
  fetch each anchor's ±1 neighbours (chunk_seq-addressed, page-independent)
        ↓
  merge overlapping/adjacent context intervals into blocks
        ↓
  dedup block text (strip the character overlap between same-page neighbours;
  do NOT strip across a page boundary — chunk_text() never overlaps across pages)
        ↓
  sort blocks by (document, chunk_seq)
        ↓
  attach a citable link per block
        ↓
  return context blocks (not raw chunks)
```

This is a straight port of the anchor/hydration design already proven in `07_Hermes_Memory_Context`'s session-search tool (`get_anchored_view()` in `hermes_state_search.py:895`, `_session_link()` in `tools/session_search_tool.py:321`) — same idea (an FTS/vector hit is a bare fragment; expand it into its immediate neighbourhood before handing it to the model, and always attach a resolvable reference back to the source), applied to `chunk_seq` instead of message id and to document pages instead of chat sessions.

## 2. Why delete `reranker.py`

`reranker.py`'s lexical-overlap + RRF only ever reranked chunks *already retrieved by dense search* — it could never surface a chunk that dense search missed but that matched the query's exact keywords, because it never issued an independent lexical search. It also wasn't used by Path B at all (only Path A).

The new design replaces it with a real two-source fusion: dense ANN search and a genuine BM25 full-text search (Milvus's native `sparse_vector` Function field, see the upload spec) run independently, each producing their own ranked list, fused by RRF over *rank in each list* — the same RRF math `reranker.py` used, but over two real retrieval methods instead of one retrieval + one heuristic. This subsumes and replaces `reranker.py`'s purpose entirely, so it is deleted rather than kept alongside.

## 3. Both chat paths converge on one retrieval core

Per project decision, both chat paths adopt the new pipeline:

- **Path A** (`/api/chat/stream`, default Streamlit UI): `retriever.py::search_sources()` now calls `hybrid_search()` and maps context blocks to `Source` objects.
- **Path B** (`/api/chat`, agent + `search_uploaded_docs` tool): `tools.py::_search_uploaded_docs_impl()` now calls `hybrid_search()` and maps context blocks to tool citations.

Both consume the same `backend/app/rag/hybrid_search.py::hybrid_search()`. The two paths still differ in everything *around* retrieval (Path A always retrieves and streams; Path B is agent-gated, the LLM decides whether to call the tool at all, possibly more than once) — only the retrieval core is unified.

## 4. `chunk.id` for dedup

Milvus's auto-generated primary-key `id` (an `INT64`, returned as `hit["id"]` by both `client.search()` calls) is the same underlying row identifier regardless of which index (`embedding` or `sparse_vector`) found it, since both indexes live on the same collection/rows. Dedup-by-id is therefore a plain dict keyed by that id — a chunk present in both the dense and sparse result lists collapses into one fused entry carrying both `dense_rank` and `sparse_rank`.

## 5. Anchor windows are `chunk_seq`-addressed, not `chunk_index`-addressed

`chunk_index` resets to 1 at every page boundary (page-local); `chunk_seq` is monotonic across the whole document (document-global, added by the upload spec specifically to make this possible). An anchor's "±1 chunk" window is defined in `chunk_seq` space, so it can cross a page boundary transparently — e.g. the last chunk of page 2 and the first chunk of page 3 are `chunk_seq` neighbours even though their `page` differs. This is why the text-merge step (§7) has to branch on `page` equality rather than assume every neighbour pair shares a page.

## 6. Two-stage dedup around the ±1 window fetch, plus a third at text level

The spec's flow lists dedup three times; each is a different operation, not a repeated step:

1. **Fetch-set dedup** — the set of `chunk_seq` values needed across *all* anchors in one document (union of each anchor's `{seq-1, seq, seq+1}`) is built as a Python `set`, which is itself the dedup: if anchor A (seq 5) and anchor B (seq 6) are both selected, their windows overlap at 5/6, and the union naturally fetches each `chunk_seq` value once, not twice.
2. **Interval merge** — after fetching, the *ranges* each anchor's window covers are merged if they touch or overlap (standard sorted-interval merge: next run starts a new block only when its `chunk_seq` is more than 1 past the current block's end). Two anchors 5 apart in the same document (with window=1) produce two separate blocks; two anchors 2 apart produce one merged block.
3. **Text-level dedup** — even after interval merging, concatenating two adjacent chunks' `text` naively would duplicate the `CHUNK_OVERLAP` (180 char default) substring `chunk_text()` leaves between them *when they share a page*. This step finds the longest suffix-of-previous/prefix-of-next match (bounded by `CHUNK_OVERLAP`, floored at 20 chars to avoid accidental short matches) and strips it before joining. Cross-page neighbours are joined with a `"\n\n"` break and never overlap-stripped, because `chunk_text()` operates per-page — there is no shared substring to strip across a page boundary in the first place.

## 7. Citable link

Modeled on `_session_link()` (`tools/session_search_tool.py:321`) — a short, stable string the agent/user can be given as "this is where this came from", without loading the whole document. Format: `doc:<file_hash>#p<page_start>` (single page) or `doc:<file_hash>#p<page_start>-<page_end>` (block spans pages). No PDF-viewer deep-linking exists yet in the frontend — the link is a stable citation identifier now, and a future frontend change could turn it into a real jump-to-page action without changing this format.

## 8. Backward compatibility for chunks without `chunk_seq`

Chunks indexed before the upload-spec change (or hand-built fixtures in existing test scripts that don't set `chunk_seq`) won't have it. `hybrid_search()` treats a missing `chunk_seq` as "can't build a window" and falls back to a single-chunk block built directly from the anchor itself (no neighbour fetch) — not an error. This keeps `scripts/test_chat_agent.py`'s existing fixtures (which omit `chunk_seq`) working unmodified.

## 9. `Source` schema changes

A block can now span multiple pages and multiple original chunks, so `Source` (`schemas.py`) gains two fields without removing any existing one:

- `pages: list[int] | None` — every page the block's text was drawn from (`page` keeps its existing meaning: the block's first page, `page_start`, for backward compatibility with `evaluator.py`'s exact-match scoring and the frontend's existing rendering).
- `link: str | None` — the citable link (§7).
- `chunk` keeps its existing `int | str` type but now holds a `chunk_seq` range label (e.g. `"12-14"`) instead of a single `chunk_index`, for blocks that span more than one chunk.

`evaluator.py::_page_hit_score()` is updated to check `expected_page in (chunk.pages or [chunk.page])` instead of exact `page` equality, since a block legitimately "hits" the expected page if that page is anywhere in the merged range — this is a strictly more accurate check under the new multi-page-block reality, not a scoring regression.

## 10. Components

### New file: `backend/app/rag/hybrid_search.py`

Pure fusion/merge functions (no I/O), plus the top-level orchestrator that calls `vector_store.py`:

- `_rrf_fuse(dense_hits, sparse_hits) -> list[dict]` — dedup by id (§4) + RRF scoring.
- `_select_anchors(fused, anchor_top_n) -> list[dict]`
- `_anchor_windows(anchors, window) -> dict[file_hash, set[chunk_seq]]` — §6 stage 1.
- `_fetch_context_chunks(needed) -> dict[file_hash, list[row]]` — calls `vector_store.get_chunks_by_seq()`.
- `_merge_intervals(rows) -> list[list[row]]` — §6 stage 2.
- `_append_with_overlap_removed(prev_text, next_text, max_overlap) -> str` / `_merge_chunk_text(rows) -> str` — §6 stage 3.
- `_citation_link(file_hash, page_start, page_end) -> str` — §7.
- `_build_block(rows, anchor_scores) -> dict` — assembles one final context block.
- `_anchor_to_row(anchor) -> dict` — adapts a nested search-hit into the flat row shape `_build_block` expects, used for the §8 fallback.
- `hybrid_search(query, dense_top_k, sparse_top_k, anchor_top_n, window) -> list[dict]` — the orchestrator described in §1.

### Modified: `backend/app/rag/vector_store.py`

- `query_chunks()` (dense): add `chunk_seq` to `output_fields` / returned `metadata`. Refactor the search→dict conversion shared with the new `sparse_search()` into one helper (`_hits_from_search_results()`), avoiding duplicated code.
- New `sparse_search(query, top_k) -> list[dict]` — `client.search(anns_field="sparse_vector", data=[query], ...)`. Milvus tokenizes and BM25-scores the raw query text itself via the Function attached to `sparse_vector`; no local embedding call, unlike dense search.
- New `get_chunks_by_seq(file_hash, chunk_seqs) -> list[dict]` — `client.query(filter='file_hash == "..." && chunk_seq in [...]', output_fields=[...])`, flat row shape (matches `indexed_file_hashes()`/`list_documents()`'s existing `query()`-based convention, as opposed to `query_chunks()`'s nested `search()`-based convention).

### Modified: `backend/app/rag/retriever.py` (Path A)

`retrieve_chunks()` / `search_sources()` call `hybrid_search()` and map blocks to `Source` objects. `rerank_top_k` parameter removed (no longer meaningful — anchor count is now `ANCHOR_TOP_N`, an internal `hybrid_search()` default, not part of the public retrieval API).

### Deleted: `backend/app/rag/reranker.py`

Superseded — see §2.

### Modified: `backend/app/agent/tools.py` / `backend/app/agent/chat_agent.py` (Path B)

`_search_uploaded_docs_impl()` calls `hybrid_search()`; citations gain `page_start`/`page_end`/`link`, keep `page` (= `page_start`) for backward compatibility. `chat_agent.py::_citation_to_source()` threads `link` through to the returned `Source`.

### Modified: `backend/app/config.py`

- Add `SPARSE_TOP_K` (defaults to `TOP_K`'s value), `ANCHOR_TOP_N` (default `2`), `ANCHOR_WINDOW` (default `1`).
- Remove `RERANK_TOP_K` (no longer referenced anywhere once `reranker.py` and its one caller are gone).

### Modified: `backend/app/schemas.py`, `backend/app/rag/evaluator.py`, `backend/app/rag/generator.py`, `frontend/app.py`

Presentation/scoring layer threads the new `pages`/`link` fields through — §9.

## 11. Explicitly out of scope

- Changing what `search_uploaded_docs` exposes to the LLM — still only `query` (top_k, anchor_top_n, window stay Python-side defaults, not model-controllable).
- A real PDF-viewer deep link behind the citable link — out of scope, see §7.
- User/session-scoped filtering (`UserContext` is still accepted but unused for filtering, unchanged from today).
- `offline_docs_store.py` / `OFFLINE_DOCS_COLLECTION_NAME` / `backend/app/orchestrator/` — untouched, separate collection and still-unwired code per `docs/superpowers/plans/2026-08-06-move-orchestrator-code-from-local-rag-agent.md`.
- `docs/architecture.md` / `ARCHITECTURE.md` / `README.md` prose updates reflecting this change are part of the implementation plan (last task), not deferred this time, since they directly document the two chat paths this spec changes.
