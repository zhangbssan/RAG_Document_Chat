# Persistent Multi-Conversation Single-Tool Agent Implementation Plan

> **Execution gate:** This document is a plan only. Do not implement any task until the user has reviewed and explicitly approved the plan. After approval, execute it task by task and stop on any failed acceptance check rather than silently changing the design.

**Goal:** Turn the existing one-tool, one-request LangChain agent into a practical daily PDF-analysis assistant with persistent multi-turn conversations, multiple isolated chat threads, agent-controlled use of `search_uploaded_docs`, and citations limited to sources retrieved during the current turn.

**Current baseline:** The PDF upload and hybrid retrieval pipeline already exists. `backend/app/agent/tools.py` already exposes exactly one LangChain tool, `search_uploaded_docs`, and its Milvus-backed dense + BM25 + RRF retrieval has passed live integration tests. `backend/app/agent/chat_agent.py` already uses LangChain's `create_agent`; it must be extended, not replaced. The current working-tree version of `frontend/app.py` already calls the non-streaming agent endpoint `POST /api/chat` directly, so the earlier “make the frontend use the agent” phase is already complete and must be preserved. The legacy `POST /api/chat/stream` route remains in the backend but is not the UI's primary path.

**Architecture:** Continue using LangChain's high-level `create_agent` API for model/tool orchestration and LangGraph's built-in checkpoint/streaming capabilities for conversation state. Add a SQLite checkpointer keyed by `conversation_id`/LangGraph `thread_id`; add a minimal conversation catalog for listing, titling, switching, and deleting chats; and keep Milvus responsible only for uploaded-PDF chunks and retrieval. No custom graph nodes, edges, reducers, planner, router, or hand-written agent loop will be introduced.

**Tech stack:** Existing `langchain==1.3.14`, `langchain-core==1.5.3`, `langchain-openai==1.4.1`, and transitive `langgraph==1.2.10`; add the version-compatible `langgraph-checkpoint-sqlite` package. Continue using FastAPI, Streamlit, Pydantic, and Milvus. Use Python's standard `sqlite3` only for the small application-owned conversation catalog.

---

## Why Both LangChain and LangGraph Are Used

### LangChain responsibilities

- `ChatOpenAI` remains the model adapter.
- The existing `@tool`/`BaseTool` integration remains the only LLM-visible capability.
- `create_agent(...)` remains the high-level agent constructor.
- `create_agent` owns the model → tool → model loop and decides whether zero, one, or several tool calls are needed.
- Optional structured response handling may be added through `create_agent(response_format=...)`; no separate output parser loop will be written.

### LangGraph responsibilities

- `SqliteSaver` persists the compiled agent graph's message state.
- `configurable.thread_id` isolates one conversation from another.
- Graph state restoration provides multi-turn memory without manually re-sending or rebuilding history.
- `get_state()` provides persisted message history for the UI.
- `delete_thread()` removes a conversation's checkpoints.
- In the final streaming task, `agent.stream(...)` provides graph/tool/model events; no manual `while tool_calls` loop will be added.

### Rationale

`create_agent` already returns a compiled LangGraph state graph. Using it together with a LangGraph checkpointer gives the project the desired agent loop and memory while keeping project-owned code limited to lifecycle, API contracts, citations, and UI concerns. Re-implementing Hermes-style state machines would duplicate framework behavior and is explicitly outside this plan.

---

## Global Constraints

- **Do not modify `backend/app/agent/tools.py`.** Its current query-only LLM schema, request-bound `UserContext`, retrieval implementation, error envelope, and citation fields are the accepted tool contract.
- Keep exactly one LLM-visible tool: `search_uploaded_docs`.
- Do not add a custom `StateGraph`, graph nodes, graph edges, message reducer, planner, router, executor loop, or `while` loop around tool calls.
- Continue using `langchain.agents.create_agent` as the only agent-loop constructor.
- Use LangGraph checkpoint state as the single source of truth for Human/AI/Tool messages. Do not create a duplicate application messages table.
- A small application-owned `conversations` table may store only discoverability metadata: id, title, and timestamps.
- Milvus remains the source of PDF chunks; SQLite must not duplicate PDF text or vectors beyond ToolMessages naturally contained in checkpoints.
- Preserve the current frontend behavior that directly calls `POST /api/chat`.
- Preserve request-scoped OpenAI API key support. Never write a request-provided API key to SQLite, checkpoints, logs, conversation metadata, or Streamlit message history.
- The no-key extractive fallback remains available, but it is stateless because no LLM/agent run occurs. The UI must make this limitation clear if the fallback is used.
- Only sources produced by successful `search_uploaded_docs` calls in the **current turn** may be returned in that turn's `ChatResponse.sources`.
- Do not return ToolMessages from earlier turns merely because the checkpointer restored them.
- Deduplicate current-turn sources by stable `chunk_id`, preserving first-seen order.
- Unknown or model-invented citation ids must be discarded server-side.
- Keep the existing recursion limit as a safety bound. It configures the framework agent; it must not become a new project-owned loop.
- Do not implement OCR, table extraction, cross-user access policy, department tools, SQL tools, cross-conversation long-term memory, or automatic summarization in this plan.
- Preserve all unrelated dirty-worktree changes. In particular, do not overwrite the current uncommitted frontend/API fallback changes.
- All tests that create Milvus data must use throwaway `test_*` collections and drop them in `finally` blocks.
- All tests that create SQLite files must use a temporary directory and remove it afterward.

---

## Final Data Ownership

```text
backend/data/chat_history.sqlite
├── LangGraph checkpoint tables       # owned by SqliteSaver
│   └── per-thread messages/state
└── app_conversations                 # owned by this application
    └── id/title/created_at/updated_at

backend/data/uploads/
└── original uploaded PDF files

Milvus: realtime_pdf_collection
└── PDF text chunks + dense vectors + BM25 sparse vectors + citation metadata
```

The Docker backend already mounts `./backend/data:/app/data`; therefore the SQLite file and uploaded PDFs survive backend container replacement without adding another volume.

---

## Target Request and Response Contracts

### Chat request

```json
{
  "question": "What does it say about the trial period?",
  "conversation_id": "9b9339b5-1f64-4ef8-9932-5fead3ad6df7",
  "top_k": 5,
  "openai_api_key": "request-only; never persisted"
}
```

### Non-streaming chat response

```json
{
  "answer": "...",
  "sources": [
    {
      "document": "employee_handbook.pdf",
      "page": 3,
      "pages": [3, 4],
      "chunk": "<stable block id>",
      "link": "doc:<hash>#p3-4",
      "text": "...",
      "score": null
    }
  ]
}
```

### Conversation summary

```json
{
  "id": "9b9339b5-1f64-4ef8-9932-5fead3ad6df7",
  "title": "What does the handbook say about annual leave?",
  "created_at": "2026-08-17T12:00:00Z",
  "updated_at": "2026-08-17T12:04:00Z"
}
```

### Agent streaming events (final phase)

```json
{"type":"tool_start","tool":"search_uploaded_docs","query":"trial period annual leave"}
{"type":"sources","data":[...]}
{"type":"token","data":"Employees"}
{"type":"token","data":" ..."}
{"type":"done"}
```

---

## Planned File Structure

```text
00_RAG_Document_Chat/
├── backend/
│   ├── requirements.txt                         # MODIFY: SQLite checkpointer dependency
│   └── app/
│       ├── config.py                            # MODIFY: CHAT_DB_PATH
│       ├── main.py                              # MODIFY: runtime lifespan + conversation router
│       ├── schemas.py                           # MODIFY: conversation_id + conversation DTOs
│       ├── agent/
│       │   ├── tools.py                         # UNCHANGED
│       │   ├── chat_agent.py                    # MODIFY: checkpointer/thread id/current-turn sources
│       │   ├── runtime.py                       # NEW: framework lifecycle, invoke/state/delete
│       │   └── conversation_store.py            # NEW: metadata-only catalog
│       └── api/
│           ├── chat.py                          # MODIFY: conversation id + runtime use
│           └── conversations.py                 # NEW: list/history/delete endpoints
├── frontend/app.py                              # MODIFY: conversation ids/list/switch/delete; later stream
├── scripts/
│   ├── test_conversation_store.py               # NEW
│   ├── test_agent_memory.py                     # NEW
│   ├── test_current_turn_citations.py            # NEW
│   ├── test_conversation_api.py                 # NEW
│   └── test_chat_agent.py                       # MODIFY: new conversation_id contract
├── docs/
│   ├── architecture.md                          # MODIFY
│   └── query-workflow.md                        # MODIFY
├── README.md                                    # MODIFY
└── .gitignore                                   # MODIFY: SQLite runtime files
```

---

## Task 0: Reconfirm and Freeze the Existing Baseline

**Files:** No production changes.

**Purpose:** Confirm the plan starts from the actual working tree, including the user's current uncommitted change that made the UI call the agent endpoint directly.

- [ ] Verify `frontend/app.py::ask_question()` posts to `/api/chat`.
- [ ] Verify `frontend/app.py::render_chat()` calls `ask_question()` directly and does not call `stream_answer()`.
- [ ] Verify `backend/app/api/chat.py::chat()` delegates to `run_agent_chat()` when an API key is available.
- [ ] Verify `backend/app/api/chat.py::chat()` retains the extractive fallback when no key is available.
- [ ] Verify `backend/app/agent/tools.py::make_search_tool()` exposes only `query` in the LLM-visible schema.
- [ ] Run the existing import and pure-function tests:

```bash
backend/.venv/bin/python scripts/validate_setup.py
backend/.venv/bin/python scripts/test_pdf_loader_chunker.py
backend/.venv/bin/python scripts/test_pdf_chunker_seq.py
backend/.venv/bin/python scripts/test_hybrid_search_merge.py
backend/.venv/bin/python scripts/test_generator_context_format.py
```

- [ ] With live Milvus available, run:

```bash
backend/.venv/bin/python scripts/test_hybrid_vector_store.py
backend/.venv/bin/python scripts/test_hybrid_search.py
backend/.venv/bin/python scripts/test_search_uploaded_docs_tool.py
```

**Expected:** All tests exit 0. The tool test confirms that no Agent-memory work is needed inside `tools.py`.

**Stop condition:** If the frontend no longer calls `/api/chat` or the Tool schema is not query-only, update this plan before implementation instead of silently broadening scope.

---

## Task 1: Add the Supported SQLite Checkpointer Dependency and Configuration

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/app/config.py`
- Modify: `.gitignore`

**Interfaces produced:**
- `CHAT_DB_PATH: Path`
- Importable `langgraph.checkpoint.sqlite.SqliteSaver`

- [ ] Add a `langgraph-checkpoint-sqlite` version compatible with the installed `langgraph==1.2.10`. Resolve and pin the installed version rather than leaving it unbounded.
- [ ] Install the updated backend requirements.
- [ ] Inspect the installed `SqliteSaver` constructor, `setup`, `get_tuple/list`, and `delete_thread` signatures before writing runtime code. The package API is authoritative; do not assume an older tutorial's context-manager usage.
- [ ] Add to `backend/app/config.py`:

```python
CHAT_DB_PATH = Path(os.getenv("CHAT_DB_PATH", DATA_DIR / "chat_history.sqlite"))
```

- [ ] Add Docker environment configuration only if necessary; the existing `/app/data` volume must remain the persistence boundary.
- [ ] Ignore runtime SQLite artifacts without ignoring the uploads directory:

```gitignore
backend/data/*.sqlite
backend/data/*.sqlite-shm
backend/data/*.sqlite-wal
```

- [ ] Verify imports and the resolved version:

```bash
backend/.venv/bin/python -c "from langgraph.checkpoint.sqlite import SqliteSaver; import importlib.metadata as m; print(m.version('langgraph-checkpoint-sqlite'))"
```

**Expected:** Import succeeds and the printed version is added as an exact pin in `backend/requirements.txt`.

**Commit after approval/execution:** `Add SQLite checkpoint persistence dependency and config`

---

## Task 2: Define Conversation API Schemas Without Changing the Tool

**Files:**
- Modify: `backend/app/schemas.py`
- Create: `scripts/test_conversation_schemas.py`

**Interfaces produced:**

```python
class ChatRequest(BaseModel):
    question: str
    conversation_id: str
    top_k: int | None = None
    openai_api_key: str | None = None

class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str

class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]

class ConversationMessage(BaseModel):
    role: str
    content: str
    sources: list[Source] = Field(default_factory=list)

class ConversationHistoryResponse(BaseModel):
    conversation: ConversationSummary
    messages: list[ConversationMessage]
```

- [ ] Add `conversation_id` as a required non-blank string. Treat it as an opaque UUID-shaped identifier at the API boundary; do not allow it to become a filesystem path.
- [ ] Add response models for list and history endpoints.
- [ ] Ensure `openai_api_key` remains request-only and is absent from every response model.
- [ ] Test valid requests, blank conversation ids, and serialization of source page ranges/links.
- [ ] Update all direct `ChatRequest` construction sites and test fixtures.

**Expected:** Schema tests exit 0; existing `ChatResponse` remains backward-compatible apart from the now-required `conversation_id` request field.

**Commit after approval/execution:** `Add conversation identifiers and API response schemas`

---

## Task 3: Add a Metadata-Only Conversation Catalog

**Files:**
- Create: `backend/app/agent/conversation_store.py`
- Create: `scripts/test_conversation_store.py`

**Purpose:** LangGraph checkpoints can restore a known thread but are not a user-facing conversation catalog. Store only the fields needed to discover and label threads; do not duplicate message history.

**Table:**

```sql
CREATE TABLE IF NOT EXISTS app_conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

**Interfaces produced:**

```python
class ConversationStore:
    def ensure(self, conversation_id: str, first_question: str) -> dict: ...
    def list(self) -> list[dict]: ...
    def get(self, conversation_id: str) -> dict | None: ...
    def touch(self, conversation_id: str) -> None: ...
    def rename(self, conversation_id: str, title: str) -> dict: ...
    def delete(self, conversation_id: str) -> bool: ...
```

- [ ] Use the same `CHAT_DB_PATH`, but use an application-owned table name prefixed with `app_` so it cannot collide with LangGraph tables.
- [ ] Enable WAL mode and a reasonable SQLite busy timeout on the catalog connection.
- [ ] Generate the initial title deterministically from the first non-empty question: normalize whitespace and truncate to 60 characters. Do not spend an LLM call on title generation.
- [ ] Store UTC ISO-8601 timestamps.
- [ ] Return conversations ordered by `updated_at DESC`.
- [ ] Reject empty ids and empty rename titles.
- [ ] Test create/idempotent ensure/list ordering/get/rename/delete against a temporary SQLite file.

**Expected:** Tests prove that calling `ensure` twice does not erase the original title or creation time and that only metadata, not messages or API keys, is stored in `app_conversations`.

**Commit after approval/execution:** `Add minimal persistent conversation catalog`

---

## Task 4: Introduce a Framework-Owned Agent Runtime With SQLite Memory

**Files:**
- Create: `backend/app/agent/runtime.py`
- Modify: `backend/app/agent/chat_agent.py`
- Modify: `backend/app/main.py`
- Create: `scripts/test_agent_memory.py`

**Purpose:** Own the SQLite connection/checkpointer lifecycle and expose small application methods while leaving the agent loop inside `create_agent`.

**Runtime responsibilities:**

```python
class AgentRuntime:
    def __init__(self, db_path: Path): ...
    def start(self) -> None: ...
    def close(self) -> None: ...
    def run_chat(
        self,
        question: str,
        conversation_id: str,
        user_context: dict | None,
        api_key: str,
        top_k: int,
    ) -> dict: ...
    def get_display_history(self, conversation_id: str) -> list[dict]: ...
    def delete_conversation_state(self, conversation_id: str) -> None: ...
```

- [ ] Open one process-owned SQLite connection/checkpointer using the installed package's supported API.
- [ ] Make startup idempotent and close the connection during FastAPI shutdown.
- [ ] Attach runtime startup/shutdown using FastAPI's `lifespan` mechanism; do not create/close SQLite once per request.
- [ ] Store the runtime on `app.state` or expose one explicit dependency accessor. Avoid hidden import-time side effects.
- [ ] In `run_chat`, keep the existing `make_search_tool(UserContext(...), top_k)` call unchanged.
- [ ] Build the agent with:

```python
create_agent(
    model=ChatOpenAI(...),
    tools=[search_tool],
    system_prompt=_SYSTEM_PROMPT,
    checkpointer=sqlite_checkpointer,
)
```

- [ ] Invoke using:

```python
config = {
    "configurable": {"thread_id": conversation_id},
    "recursion_limit": _RECURSION_LIMIT,
}
```

- [ ] Send only the new `HumanMessage`; allow LangGraph to restore old messages from the checkpoint.
- [ ] Preserve per-request API key behavior by constructing the model/compiled agent per request while reusing the same process-owned checkpointer. Never cache a request key.
- [ ] Keep `GraphRecursionError` handling.
- [ ] Move orchestration entry points from `chat_agent.py` into the runtime only as needed; retain citation mapping/prompt helpers in `chat_agent.py` rather than rewriting the module wholesale.
- [ ] `get_display_history` must read the latest checkpoint through the checkpointer's supported public API, so reopening history does not require an OpenAI key or a cached per-request graph instance.
- [ ] Reconstruct display turns from persisted messages: expose HumanMessage and final AIMessage content, associate each final AIMessage with citations from ToolMessages belonging to that turn, and keep raw ToolMessages/intermediate tool-calling AIMessage objects internal.
- [ ] Tests must use a deterministic fake chat model or injected model factory. They must not require a paid OpenAI call to prove persistence.
- [ ] Memory test sequence:
  1. Run conversation A turn 1.
  2. Run conversation A turn 2 and assert the fake model receives turn 1 in restored history.
  3. Run conversation B and assert it cannot see A.
  4. Close runtime, open a new runtime on the same temporary SQLite file, and assert A can continue.

**Expected:** Multi-turn state survives runtime recreation, and conversation ids isolate state without any custom Agent state definition.

**Commit after approval/execution:** `Persist create_agent conversation state with LangGraph SQLite checkpoints`

---

## Task 5: Return Only Current-Turn, Validated Sources

**Files:**
- Modify: `backend/app/agent/chat_agent.py`
- Modify: `backend/app/agent/runtime.py`
- Create: `scripts/test_current_turn_citations.py`

**Problem:** Once checkpoint memory is enabled, `result["messages"]` contains historical ToolMessages. The current implementation scans all ToolMessages and would therefore leak old-turn citations into every later response.

**Interfaces produced:**

```python
def sources_from_new_messages(messages: list, previous_message_count: int) -> list[Source]: ...
def deduplicate_sources(sources: list[Source]) -> list[Source]: ...
```

- [ ] Before invocation, read the existing checkpoint and record the number or ids of persisted messages.
- [ ] After invocation, inspect only messages appended by this invocation.
- [ ] Parse only successful `search_uploaded_docs` ToolMessages.
- [ ] Ignore malformed payloads and failed tool results without crashing the answer.
- [ ] Deduplicate by `chunk_id`, preserving first-seen order across repeated tool calls.
- [ ] Never scan ToolMessages from earlier turns.
- [ ] Update the system prompt to require citations using only source names/pages/links returned by the tool. Do not change the tool implementation.
- [ ] Add a server-side citation validator: every returned `Source.chunk` must occur in a current-turn tool citation.
- [ ] Keep the first implementation honest: `ChatResponse.sources` means “sources retrieved and made available during this answer.” If strict “actually cited in prose” filtering is implemented, use LangChain `response_format` with a Pydantic answer schema and validate `cited_source_ids`; do not regex arbitrary model text.
- [ ] Add tests containing an old ToolMessage plus a new ToolMessage and assert only the new source is returned.
- [ ] Add a test where the same chunk appears in two current-turn tool calls and assert it appears once.
- [ ] Add a test with an invented citation id and assert it is discarded.

**Expected:** A follow-up that does not call the tool returns `sources=[]`, even if an earlier turn used the tool.

**Commit after approval/execution:** `Scope agent citations to validated current-turn tool results`

---

## Task 6: Wire Conversation Persistence Into the Chat API

**Files:**
- Modify: `backend/app/api/chat.py`
- Create: `backend/app/api/conversations.py`
- Modify: `backend/app/main.py`
- Create: `scripts/test_conversation_api.py`

**Endpoints:**

```text
POST   /api/chat
GET    /api/conversations
GET    /api/conversations/{conversation_id}/messages
PATCH  /api/conversations/{conversation_id}
DELETE /api/conversations/{conversation_id}
```

- [ ] `POST /api/chat` validates `conversation_id`; on the Agent path it calls `ConversationStore.ensure(...)`, runs the agent with the same id as LangGraph `thread_id`, and touches `updated_at` after a successful answer.
- [ ] Preserve the existing no-key fallback. Check for this before creating persistent conversation metadata: the fallback may return an answer and sources, but must not create an empty persistent conversation or pretend to have persisted agent memory.
- [ ] `GET /api/conversations` returns metadata ordered by most recently updated.
- [ ] `GET .../messages` combines catalog metadata with displayable Human/AI messages loaded from LangGraph state.
- [ ] `PATCH` changes title only.
- [ ] `DELETE` deletes both catalog metadata and LangGraph thread checkpoints. If either side is already absent, deletion remains idempotent.
- [ ] Reject unknown ids with 404 for history/rename; keep delete idempotent with a clear response.
- [ ] Do not expose ToolMessages, checkpoint internals, API keys, system prompts, or raw graph configuration.
- [ ] Register the conversations router in `main.py`.
- [ ] API tests use a temporary database and an injected fake runtime; they must cover A/B isolation, list ordering, history filtering, rename, and delete.

**Expected:** The backend supports discoverable, persistent, isolated conversations while retaining the existing `ChatResponse` format.

**Commit after approval/execution:** `Expose persistent conversation management APIs`

---

## Task 7: Add Multi-Conversation UX While Preserving Direct Agent Calls

**Files:**
- Modify: `frontend/app.py`

**Important baseline:** The current working tree already calls `/api/chat` directly. Do not reintroduce the old direct-retrieval UI path during this task.

**Frontend state:**

```python
st.session_state.current_conversation_id: str
st.session_state.conversations: list[dict]
st.session_state.messages: list[dict]  # display cache for selected conversation only
```

- [ ] Add `conversation_id` to `_chat_payload()` and `ask_question()`.
- [ ] Generate a UUID with `uuid.uuid4()` for a new conversation.
- [ ] On app initialization, fetch `/api/conversations`.
- [ ] If no conversations exist, create a local empty conversation id; the backend catalog entry is created on its first submitted question.
- [ ] Add a “New chat” button.
- [ ] Render conversation titles in the sidebar, ordered by backend `updated_at`.
- [ ] On selection, load `/api/conversations/{id}/messages` and replace the display cache.
- [ ] Add delete support with confirmation appropriate to Streamlit's available UI primitives.
- [ ] Keep the existing document upload/list/delete controls intact.
- [ ] Rename the existing “Clear Chat” behavior: it must start a new conversation rather than only clearing browser state while leaving backend memory active.
- [ ] After a successful first response, refresh the conversation list so the generated title appears.
- [ ] Never store the OpenAI API key in conversation metadata or message dictionaries.
- [ ] Continue displaying per-answer sources using the current source renderer.
- [ ] Manually verify:
  1. Create A and ask two related questions.
  2. Create B and ask an unrelated question.
  3. Switch to A and see its prior two turns.
  4. Refresh the browser and recover both conversations.
  5. Delete B and verify it cannot be reopened.

**Expected:** The UI behaves like a minimal multi-chat application, and every question still goes through the one-tool agent endpoint.

**Commit after approval/execution:** `Add persistent multi-conversation Streamlit chat UX`

---

## Task 8: Add Native Agent Streaming After Non-Streaming Memory Is Stable

**Files:**
- Modify: `backend/app/agent/runtime.py`
- Modify: `backend/app/api/chat.py`
- Modify: `frontend/app.py`
- Create: `scripts/test_agent_stream_events.py`

**Reason for ordering:** Persistence, thread isolation, and current-turn citation scoping are easier to validate with `invoke()` first. Streaming is added only after those invariants pass. This avoids debugging checkpoint semantics and event parsing simultaneously.

- [ ] Add `AgentRuntime.stream_chat(...)` implemented with LangGraph's native `agent.stream(...)` API.
- [ ] Use supported stream modes from the installed LangGraph version to observe model messages and graph/tool updates. Inspect the installed signature before coding.
- [ ] Do not implement a manual tool-call loop.
- [ ] Emit application NDJSON events through a thin adapter:
  - `tool_start`: tool name and safe query only.
  - `sources`: deduplicated current-turn validated sources.
  - `token`: answer text fragments only.
  - `done`: exactly once.
  - `error`: structured safe message; never include API keys or full internal traces.
- [ ] Replace the legacy `/api/chat/stream` direct retrieval implementation with the agent stream when a key exists.
- [ ] Preserve direct retrieval/generator streaming only as the no-key fallback.
- [ ] Restore a frontend stream reader that posts the same `conversation_id` and renders Tool status, tokens, and sources.
- [ ] Ensure the UI does not call both `/api/chat` and `/api/chat/stream` for the same user turn.
- [ ] Test event ordering with a deterministic fake agent stream:

```text
tool_start → sources → token* → done
```

- [ ] Test a no-tool answer:

```text
token* → done
```

- [ ] Test that sources from earlier checkpointed turns are absent.

**Expected:** The UI gains progressive output and visible Tool activity while the agent loop remains fully framework-owned.

**Commit after approval/execution:** `Stream LangGraph agent activity and answers to the frontend`

---

## Task 9: End-to-End Daily-Use Acceptance Tests

**Files:**
- Modify: `scripts/test_chat_agent.py`
- Create: `scripts/test_persistent_agent_e2e.py`

### Automated acceptance

- [ ] Run all pure/unit tests without Milvus/OpenAI.
- [ ] Run live Milvus retrieval tests against throwaway collections.
- [ ] Run the persistent-memory test against temporary SQLite.
- [ ] Run API tests with an injected fake model/runtime.
- [ ] Run one explicitly marked live OpenAI smoke test only when `OPENAI_API_KEY` is present.

### Required scenarios

- [ ] **Tool decision:** “What is 2 + 2?” returns an answer with no Tool source.
- [ ] **Document decision:** A question about an indexed PDF calls `search_uploaded_docs` and returns sources.
- [ ] **Follow-up:** “What about employees in probation?” resolves against the preceding document question in the same conversation.
- [ ] **Isolation:** The same follow-up in a fresh conversation does not inherit the other conversation's context.
- [ ] **Persistence:** Close and recreate the backend runtime; the original conversation continues.
- [ ] **Current-turn citations:** A no-tool follow-up does not repeat the previous turn's sources.
- [ ] **Repeated tool calls:** Duplicate chunks are returned once.
- [ ] **Deletion:** Deleting a conversation removes catalog and checkpoint state.
- [ ] **No key:** The extractive fallback remains usable and clearly stateless.
- [ ] **Tool regression:** `list(make_search_tool(...).args.keys()) == ["query"]` remains true.
- [ ] **Upload regression:** Upload/index/search behavior remains unchanged.

### Real PDF manual acceptance

Use one text-based sample PDF, not a scanned/OCR-only file:

```text
Upload PDF
→ verify document list/chunk count
→ create conversation A
→ ask a document-grounded question
→ inspect cited document/page/link
→ ask a pronoun-based follow-up
→ create conversation B
→ verify isolation
→ restart backend
→ reopen A and continue
```

**Expected:** All automated tests exit 0 and the manual flow succeeds without editing SQLite or manually supplying message history.

**Commit after approval/execution:** `Add persistent single-tool agent end-to-end coverage`

---

## Task 10: Update Documentation to Match the Implemented System

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/query-workflow.md`
- Modify: `README.md`

- [ ] Replace the current statement that backend requests are independent.
- [ ] Document `conversation_id` → LangGraph `thread_id` mapping.
- [ ] Document SQLite checkpoint state versus the metadata-only conversation table.
- [ ] Document that Milvus stores PDF retrieval data, not chat history.
- [ ] Update the query diagram to show:

```text
Frontend conversation
→ FastAPI
→ LangGraph restores thread checkpoint
→ LangChain create_agent decides Tool use
→ search_uploaded_docs (optional)
→ final answer + current-turn sources
→ LangGraph persists updated state
```

- [ ] Document no-key fallback as stateless.
- [ ] Document that the frontend's primary path is the Agent path.
- [ ] Document current limits: text PDFs only, one shared document collection, no user access isolation, no cross-conversation memory, no automatic long-history summarization yet.
- [ ] Add operational notes for backing up/removing `backend/data/chat_history.sqlite`.

**Expected:** README and workflow docs describe the actual final code paths and no longer claim that Streamlit display history alone is multi-turn memory.

**Commit after approval/execution:** `Document persistent multi-conversation agent workflow`

---

## Verification Command Matrix

Run from the repository root:

```bash
backend/.venv/bin/python -m compileall -q backend/app frontend/app.py
backend/.venv/bin/python scripts/validate_setup.py
backend/.venv/bin/python scripts/test_conversation_schemas.py
backend/.venv/bin/python scripts/test_conversation_store.py
backend/.venv/bin/python scripts/test_agent_memory.py
backend/.venv/bin/python scripts/test_current_turn_citations.py
backend/.venv/bin/python scripts/test_conversation_api.py
backend/.venv/bin/python scripts/test_agent_stream_events.py
backend/.venv/bin/python scripts/test_pdf_loader_chunker.py
backend/.venv/bin/python scripts/test_pdf_chunker_seq.py
backend/.venv/bin/python scripts/test_hybrid_search_merge.py
backend/.venv/bin/python scripts/test_generator_context_format.py
```

With live Milvus:

```bash
backend/.venv/bin/python scripts/test_hybrid_vector_store.py
backend/.venv/bin/python scripts/test_hybrid_search.py
backend/.venv/bin/python scripts/test_retriever_hybrid.py
backend/.venv/bin/python scripts/test_search_uploaded_docs_tool.py
```

With live Milvus and an explicitly available OpenAI key:

```bash
backend/.venv/bin/python scripts/test_chat_agent.py
backend/.venv/bin/python scripts/test_persistent_agent_e2e.py
```

---

## Completion Criteria

This plan is complete only when all of the following are true:

- The existing `search_uploaded_docs` implementation remains unchanged.
- The LLM sees exactly one tool with exactly one argument: `query`.
- Production code contains no hand-written agent loop or custom LangGraph topology.
- LangChain `create_agent` owns autonomous Tool selection and repeated Tool calls.
- LangGraph SQLite checkpoints provide persistent multi-turn state.
- Two conversation ids remain isolated.
- Conversation history survives backend restart.
- The frontend can create, list, switch, reopen, and delete conversations.
- The frontend continues to call the Agent path by default.
- Sources returned for a turn come only from Tool calls made in that turn.
- Source ids are validated against actual Tool results and deduplicated.
- Uploaded-PDF ingestion and hybrid retrieval regressions still pass.
- No request-provided API key is persisted.
- Documentation accurately reflects storage ownership and runtime behavior.

---

## Explicitly Deferred Work

The following should be separate future plans triggered by observed need, not bundled into this implementation:

- Conversation summarization or trimming middleware for very long threads.
- User authentication and authorization filters in `search_uploaded_docs`.
- Per-user or per-conversation Milvus collections.
- Cross-conversation semantic memory.
- Additional department-document or business-database tools.
- OCR and table-aware PDF extraction.
- Production multi-worker checkpoint backends such as PostgreSQL.
- Background runs, resumable jobs, approvals, or Hermes-style task state.
