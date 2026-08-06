# LangChain Agent Layer Design — Realtime PDF Only

**Status:** Approved
**Scope:** `00_RAG_Document_Chat/backend/app/` only. Offline docs and numeric data are explicitly
out of scope. `POST /api/chat/stream` is untouched.

## Purpose

Replace the direct retrieval+generation call in `POST /api/chat` with a LangChain tool-calling
agent. The agent has access to one tool, `search_uploaded_docs`, which wraps the existing
Milvus-backed `vector_store.py` (realtime PDF uploads). The agent decides whether the tool is
needed for a given question — it is not called unconditionally. This is the first "agent layer"
in the project and is intentionally scoped to a single tool so the pattern can be extended later
(offline docs, numeric data) without redesigning the agent itself.

## Architecture

```
Frontend (unchanged)
     │ POST /api/chat  {question, top_k, openai_api_key}
     ▼
chat.py  (rewritten)
     │
     ▼
agent/chat_agent.py :: run_agent_chat(question, user_context, api_key)
     │  ChatOpenAI.bind_tools([search_uploaded_docs])
     │  system prompt: "use the tool when the question needs the user's uploaded docs"
     ▼
 ┌───────────────┴───────────────┐
 │ LLM decides: call tool?       │
 └───────────────┬───────────────┘
        yes ──────┼────── no
         ▼                  ▼
  agent/tools.py       LLM answers directly
  search_uploaded_docs
         │  embeds query (reuses embeddings.embed_query)
         │  vector_store.query_chunks() -> Milvus realtime_pdf_collection
         ▼
  returns fixed JSON schema (status/content/citations/metadata/error)
         │
         ▼
  ToolMessage fed back to LLM -> final answer
         │
         ▼
chat.py maps citations -> existing Source[] list -> ChatResponse(answer, sources)
```

Upload path (`api/upload.py` → `vector_store.add_chunks`) is untouched — embedding on upload
already happens automatically, unrelated to this change.

## Components

### `backend/app/agent/tools.py` (new)

Pydantic input schema, matching the user-specified shape exactly:

```python
class UserContext(BaseModel):
    user_id: str | None = None
    department_id: str | None = None
    session_id: str | None = None

class ToolInput(BaseModel):
    query: str
    user_context: UserContext
```

`search_uploaded_docs(query, user_context) -> dict` — a LangChain `StructuredTool` wrapping
`vector_store.query_chunks()`. `user_context` is accepted per the schema but **not used** to
filter results — search still queries the single shared `realtime_pdf_collection`, consistent
with the earlier decision to skip session/user isolation for the storage layer itself. The query
is embedded inside the tool (reuses `app.rag.embeddings.embed_query`).

Returns exactly:
```python
{
  "status": "success" | "error",
  "content": str,          # concatenated retrieved text, for the LLM to read
  "citations": [
    {"source_id": str, "source_name": str, "page": int | str, "chunk_id": str, "text": str},
    ...
  ],
  "metadata": {"tool": "search_uploaded_docs", "result_count": int},
  "error": None | str,
}
```

Any exception (Milvus unreachable, empty query, etc.) is caught inside the tool and turned into
`{"status": "error", "error": str(e), "content": "", "citations": [], "metadata": {...}}` — never
raised up into the agent loop, so a storage failure degrades to a clear message instead of a 500.

### `backend/app/agent/chat_agent.py` (new)

`run_agent_chat(question: str, user_context: dict, api_key: str | None = None) -> dict` returning
`{"answer": str, "sources": list[Source]}`.

- Builds `ChatOpenAI(model=OPENAI_MODEL, api_key=api_key or OPENAI_API_KEY).bind_tools([search_uploaded_docs])`.
- Sends a system prompt ("you have a tool that searches the user's uploaded documents; use it
  when the question could be answered from them, otherwise answer directly") plus the user's
  question.
- If the model's response includes a tool call, executes `search_uploaded_docs`, appends the
  result as a `ToolMessage`, and asks the model for a final answer grounded in that content.
- If no tool call, the model's direct response is the answer and `sources` is `[]`.
- Maps each citation to the existing `Source` schema: `document=source_name`, `page=page`,
  `chunk=chunk_id`, `text=text`, `score=None` (the tool's citation shape has no score field, and
  `Source.score` is already `float | None = None`, so this is a valid no-op mapping, not a
  regression).
- `api_key` parameter preserves the existing "bring your own OpenAI key" behavior from
  `ChatRequest.openai_api_key` — no regression versus the current direct-call path.

### `backend/app/api/chat.py` (rewritten)

`POST /api/chat` calls `run_agent_chat(question, user_context={}, api_key=request.openai_api_key)`
instead of `search_sources()` + `answer_question()`. `POST /api/chat/stream` is **not modified** —
it keeps calling the old direct retrieval+generation path, per the earlier decision to keep
streaming out of scope for this pass.

### Dependencies

New: `langchain`, `langchain-core`, `langchain-openai`. Not needed: `langchain-community`,
`langgraph` — a single bound-tools call doesn't need the heavier agent-executor/graph machinery.

## Error Handling

- Tool-level errors are caught and returned as structured `status: "error"` JSON (see above) —
  never raised to the agent loop.
- If the LLM doesn't call the tool (e.g., a greeting), `sources` is `[]` — expected, not an error.

## Testing

Same standalone-script convention as the rest of this project (no pytest):

- **`scripts/test_search_uploaded_docs_tool.py`** — tool in isolation: add chunks via
  `vector_store.add_chunks`, call `search_uploaded_docs` directly, assert the exact JSON shape
  (`status`, `content`, `citations[]` fields, `metadata`, `error`), plus an error-path case
  (e.g., empty query) asserting `status == "error"` and a non-null `error` message.
- **`scripts/test_chat_agent.py`** — end-to-end: seed a document, ask a question that needs it →
  assert the tool was invoked (non-empty `sources`) and the answer reflects the seeded content;
  ask an unrelated question (e.g., "what's 2+2") → assert the tool was *not* invoked (`sources == []`).
  Requires a live `OPENAI_API_KEY` — user will provide it before this test is run.

## Self-Review

- **Placeholders:** none — schemas, function signatures, and JSON shapes are all concrete.
- **Consistency:** citation-to-`Source` mapping uses fields that already exist on both sides
  (no frontend changes needed); `user_context` handling matches the earlier storage-layer decision
  to skip isolation for now.
- **Scope:** confirmed bounded to realtime PDF + `/api/chat` only — offline docs, numeric data,
  and `/api/chat/stream` are explicitly untouched.
- **Ambiguity resolved:** LLM = OpenAI (matches existing config); integration = replace
  `/api/chat` in place; `user_context` accepted but unused; streaming stays on the old path.