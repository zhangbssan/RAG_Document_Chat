# frontend-web ↔ Streamlit Parity Checklist

**Status: partially verified.** This build ran in a headless environment with no browser automation tool available, so the interactive/visual half of this checklist could not be completed by the agent. Everything below marked "API-verified" was checked by exercising the real backend directly (`curl`) and confirming the exact response shape matches what the corresponding React hook/component consumes — that is strong evidence the plumbing is correct, but it is **not** the same as clicking through the rendered UI. Everything marked "needs browser pass" still needs a human (or a browser-automation tool) to actually do.

Run both apps against the same backend/data to complete the unchecked rows:

```bash
cd backend && uvicorn app.main:app --reload --port 8000
cd frontend && streamlit run app.py   # http://localhost:8501
cd frontend-web && npm run dev        # http://localhost:5173
```

## API-verified during the build (evidence, not a substitute for the checklist below)

- `POST /api/chat/stream` NDJSON event shapes (`tool_start`, `sources`, `token`, `done`) match `ChatStreamEvent` exactly — confirmed with two live calls (small talk, and a document-lookup question), including checking that a small-talk question produces zero tool calls.
- `Source` fields returned by the tool (`text`, `document`, `page`, `chunk`, `score`, `pages`, `link`) match the `Source` type exactly.
- A completed streamed turn correctly persists: `GET /api/conversations` showed both test conversations afterward, with titles derived from the first question — matching `ConversationStore.ensure()`'s lazy-creation behavior. Both were deleted via `DELETE /api/conversations/{id}` afterward to avoid leaving test data in your sidebar.
- `GET /api/documents/list` shape matches `DocumentListResponse`/`DocumentRecord` exactly (checked against the one real indexed document currently in Milvus).
- `POST /api/evaluate` response shape matches `EvaluationResponse`/`EvaluationResult` exactly, including the `missing_documents` list behaving as expected (the three `sample_docs/*_en.pdf` aren't uploaded yet, so scores were 0 — expected, not a bug).
- `frontend-web`'s Docker image builds and serves; confirmed the built JS bundle has the correct `http://localhost:8000` base URL baked in (not `http://backend:8000`, which would be unreachable from a browser); confirmed the container itself can also reach the host backend.

## Needs a browser pass (not yet done)

- [ ] New chat creates a fresh UUID thread; no sidebar entry appears until the first successful turn.
- [ ] Conversation list: select switches the main panel; rename updates the title in place; delete removes it and falls back to the next conversation (or a fresh chat if none remain).
- [ ] Reloading the page restores the previously selected conversation's history.
- [ ] Uploading multiple PDFs at once shows one status line per file (indexed / already indexed / unreadable), and a re-upload of the same file is reported as already indexed.
- [ ] Document stats (total chunks/documents) and the per-document list (name, pages, chunks) match between both UIs after the same uploads.
- [ ] Deleting a document removes it from both the list and the stats.
- [ ] Typing an OpenAI API key in Settings is used for the next request; clearing it falls back to the backend's configured key (or extractive fallback if neither exists) — confirm via the `notice` banner in the no-key case.
- [ ] Sending a question that needs document lookup shows a searching indicator, then streams the answer, then shows citation cards matching the sources Streamlit displays for the same question.
- [ ] Sending small talk / arithmetic does not trigger a tool call in either UI (no citation cards, no searching indicator) — API-level evidence above supports this but the rendered pill/cards need a visual check.
- [ ] Sending an empty question is rejected client-side (or shows the backend's 400) in both UIs.
- [ ] Asking a multi-part question that forces several tool calls in a row eventually hits the recursion limit (or completes within it) with the same clear limit message/behavior in both UIs — no infinite spinner in either.
- [ ] Loading `/conversations/{id}/messages` for a conversation that was just deleted surfaces a clear "not found" state in both UIs rather than a blank screen or unhandled crash.
- [ ] Evaluation tab: required/missing sample-document lists match; running evaluation produces the same `average_final_score` and per-question scores.
- [ ] Stopping the backend mid-session and sending a question shows a clear error in both UIs instead of an unhandled crash.
- [ ] Dark mode toggle (Light/Dark/System) visually flips the whole app correctly and persists across reload.
- [ ] Sidebar collapses to an off-canvas drawer with a scrim below ~768px width, and sits inline above it.

## Result

Not yet summarized — the browser pass above hasn't run. Once you (or a connected browser-automation tool) complete the unchecked rows, note any `❌` findings here and whether they block deleting `frontend/`.
