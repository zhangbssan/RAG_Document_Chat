# Persistent Multi-Conversation Single-Tool Agent — Design

**Date:** 2026-08-17  
**Status:** Proposed — awaiting user review and approval  
**Scope:** The conversational Agent layer, conversation persistence, conversation APIs, and Streamlit chat UX. The existing PDF upload pipeline, hybrid retrieval implementation, and `search_uploaded_docs` Tool are dependencies and are not redesigned here.  
**Implementation plan:** `docs/superpowers/plans/2026-08-17-persistent-multi-conversation-single-tool-agent-implementation.md`

## 1. Goal

Turn the current request-scoped, one-tool Agent into a practical daily PDF-analysis assistant that supports:

1. Multiple independent conversations.
2. Persistent multi-turn context within each conversation.
3. Autonomous LLM decisions about whether uploaded-document retrieval is needed.
4. Zero, one, or multiple calls to the existing `search_uploaded_docs` Tool.
5. Source citations that are restricted to successful Tool results from the current turn.
6. Conversation recovery after browser refresh and backend restart.
7. A minimal ChatGPT-like workflow: create, list, switch, reopen, rename, and delete conversations.
8. Native streaming as a later layer over the same persistent Agent, without introducing a custom Agent loop.

The system remains intentionally small: one Agent, one read-only Tool, one local SQLite conversation database, and one Milvus collection for uploaded PDFs.

## 2. Current State

The project already has the expensive and domain-specific parts:

- PDF upload, page extraction, chunking, dense embedding, and Milvus indexing.
- Milvus-native BM25 sparse vectors.
- Dense + BM25 retrieval, Reciprocal Rank Fusion, anchor selection, neighboring-chunk expansion, text-overlap removal, and citable document/page links.
- A LangChain Tool named `search_uploaded_docs` whose LLM-visible schema contains only `query`.
- A LangChain `create_agent` call with that single Tool and a recursion limit.
- Collection of Tool citations into the existing `Source` response model.
- A frontend working-tree change that now calls the Agent-backed `POST /api/chat` endpoint directly instead of using the direct-retrieval streaming route as its primary path.

What is missing is conversation state:

- `ChatRequest` has no conversation identifier.
- Each Agent invocation receives only the current question.
- `create_agent` has no checkpointer.
- The Streamlit `st.session_state.messages` list is display state, not backend Agent memory.
- The backend cannot list, reopen, rename, or delete conversations.
- Once memory is added, the current “scan all ToolMessages” citation logic would incorrectly repeat sources from earlier turns.

## 3. Design Principles

### 3.1 Use framework-owned orchestration

The project will not implement its own Agent loop. LangChain's `create_agent` remains the only Agent-loop constructor and owns:

```text
model decision
  → optional Tool call(s)
  → ToolMessage(s)
  → further model decision
  → final answer
```

There will be no project-owned loop that inspects `AIMessage.tool_calls`, executes tools, appends ToolMessages, and calls the model again.

### 3.2 Use framework-owned short-term memory

LangGraph's SQLite checkpointer stores the Agent graph state. The application supplies a stable `thread_id`; it does not manually reload and concatenate message history into prompts.

### 3.3 Keep state minimal

No custom Agent state schema is required. The default `messages` state produced by `create_agent` is sufficient for this phase.

The application owns only:

- Conversation identity.
- Conversation-list metadata.
- Runtime lifecycle.
- Current-turn source validation.
- API and UI presentation.

### 3.4 Preserve the Tool boundary

`backend/app/agent/tools.py` is treated as a stable dependency. The Agent can choose the query string, but it cannot choose Milvus collection names, user ids, `top_k`, anchor counts, page windows, filters, or lower-level database operations.

### 3.5 One source of truth per data type

- LangGraph checkpoints own conversation messages and Agent execution state.
- The application conversation table owns only list/title/timestamp metadata.
- Milvus owns searchable PDF chunks and retrieval metadata.
- The filesystem owns original uploaded PDFs.
- Streamlit state is only a UI cache for the selected conversation.

## 4. Framework Boundary: LangChain vs. LangGraph

### 4.1 LangChain

LangChain provides the high-level Agent interface:

- `ChatOpenAI` for model access.
- Existing `@tool`/`BaseTool` support for `search_uploaded_docs`.
- `create_agent(...)` for tool binding and the Agent loop.
- Optionally, `response_format` if strict model-selected citation ids are added later.

### 4.2 LangGraph

LangGraph provides stateful execution under `create_agent`:

- `SqliteSaver` for checkpoint persistence.
- `configurable.thread_id` for conversation isolation.
- State restoration before a new turn.
- State checkpointing after model and Tool steps.
- Public checkpoint/state access for conversation history.
- Thread deletion.
- Native graph/model/Tool streaming events.

### 4.3 Why both are appropriate

`create_agent` compiles to a LangGraph graph. LangChain supplies the convenient Agent abstraction; LangGraph supplies durable execution state. This combination gives the desired behavior without either copying Hermes's runtime or dropping down to low-level graph construction.

## 5. High-Level Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│ Streamlit                                                    │
│  conversation list · selected conversation · message view   │
└──────────────────────────────┬───────────────────────────────┘
                               │ conversation_id + question
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ FastAPI                                                      │
│  chat API · conversation API · process-owned AgentRuntime    │
└──────────────┬───────────────────────────────┬───────────────┘
               │                               │
               ▼                               ▼
┌─────────────────────────────┐   ┌────────────────────────────┐
│ LangChain create_agent      │   │ SQLite                     │
│  ChatOpenAI                 │   │  LangGraph checkpoints     │
│  search_uploaded_docs       │   │  app_conversations         │
│  system prompt              │   └────────────────────────────┘
└──────────────┬──────────────┘
               │ optional Tool call
               ▼
┌──────────────────────────────────────────────────────────────┐
│ Existing search_uploaded_docs Tool — unchanged               │
│  hybrid dense + BM25 → RRF → anchor context → citations     │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Milvus realtime_pdf_collection                               │
└──────────────────────────────────────────────────────────────┘
```

## 6. Conversation Identity

Every chat thread has one opaque `conversation_id`, generated as a UUID by the frontend.

The same value is used as LangGraph's `thread_id`:

```python
config = {
    "configurable": {
        "thread_id": conversation_id,
    },
    "recursion_limit": RECURSION_LIMIT,
}
```

Semantics:

- Same `conversation_id`: restore and continue that conversation.
- Different `conversation_id`: independent state with no inherited messages.
- Reusing the id after backend restart: restore the persisted checkpoint.
- Deleting the id: remove both discovery metadata and LangGraph checkpoint state.

The identifier is not used as a path and does not select a Milvus collection. It is only a conversation/thread key.

## 7. Agent State and Turn Lifecycle

The default Agent message state contains:

```text
HumanMessage
AIMessage with optional tool_calls
ToolMessage
AIMessage final answer
```

No custom state fields are needed initially.

For a new turn:

```text
1. Receive question + conversation_id.
2. Resolve the existing checkpoint for thread_id.
3. Record the pre-turn message boundary.
4. Submit only the new HumanMessage to create_agent.
5. LangGraph restores historical messages automatically.
6. LangChain/LLM decides whether to call search_uploaded_docs.
7. LangChain executes zero, one, or multiple Tool calls.
8. The model produces a final AIMessage.
9. LangGraph persists the updated state.
10. The application extracts sources only from ToolMessages after the
    pre-turn boundary.
11. Return the final answer and validated current-turn sources.
```

The application does not send all prior messages again. It also does not maintain a separate summary or memory object.

## 8. SQLite Storage Design

### 8.1 File location

Default:

```text
backend/data/chat_history.sqlite
```

Container path:

```text
/app/data/chat_history.sqlite
```

The existing `./backend/data:/app/data` volume makes it persistent across container replacement.

### 8.2 LangGraph-owned tables

`SqliteSaver` creates and manages its checkpoint tables. Their schema is treated as framework-owned and must not be queried with application-specific SQL when a public saver API exists.

Checkpoint data includes the message state required to resume a thread, including HumanMessages, AIMessages, and ToolMessages.

### 8.3 Application-owned table

The application adds one metadata-only table:

```sql
CREATE TABLE IF NOT EXISTS app_conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

This table exists because a checkpoint saver can restore a known thread but is not a complete product-facing conversation catalog.

It does not contain:

- User or assistant message text.
- Tool results.
- PDF chunks.
- API keys.
- Model configuration.
- Serialized Agent state.

### 8.4 Title policy

The initial title is derived without an LLM call:

1. Normalize whitespace in the first question.
2. Truncate to 60 characters.
3. Preserve the original title on later turns.
4. Allow explicit rename through the API/UI.

### 8.5 SQLite operating assumptions

- This design targets the current single-backend-process daily-use deployment.
- WAL mode and a busy timeout reduce contention between checkpoint and catalog connections.
- The SQLite connection/checkpointer is opened once per backend process and closed at shutdown.
- Multi-worker production deployment is not promised by this design; a shared server database such as PostgreSQL would be a later migration.

## 9. Runtime Lifecycle

A small `AgentRuntime` owns infrastructure lifecycle; it is not an Agent loop.

Responsibilities:

- Open the SQLite connection/checkpointer at FastAPI startup.
- Close it at FastAPI shutdown.
- Build `create_agent` with the existing Tool and the shared checkpointer.
- Invoke/stream with `thread_id` and recursion limit.
- Read the latest checkpoint through supported public APIs.
- Convert persisted messages into display history.
- Delete thread state.

FastAPI's lifespan hook owns `AgentRuntime.start()` and `AgentRuntime.close()`.

### 9.1 Request-scoped API keys

The current application allows an OpenAI API key on each request. Such keys must never be cached or persisted.

Therefore:

- The SQLite checkpointer is process-owned and reused.
- The `ChatOpenAI` instance and compiled `create_agent` graph may be constructed per request.
- The request key exists only while handling that request.
- A newly constructed but structurally identical graph can resume the same `thread_id` through the shared checkpointer.

If the project later standardizes on a server-side key only, the compiled Agent can be cached without changing checkpoint semantics.

## 10. Conversation History Projection

Persisted graph state contains internal execution messages that should not be exposed directly.

The history endpoint projects checkpoint messages into UI turns:

- HumanMessage → `{role: "user", content: ...}`.
- Final AIMessage without pending Tool calls → `{role: "assistant", content: ..., sources: [...]}`.
- Tool-calling AIMessage → internal only.
- Raw ToolMessage → internal only, but its validated citations are associated with the following final AIMessage from the same turn.
- System messages and graph metadata → never returned.

This projection lets a reopened conversation reproduce source panels without introducing a duplicate messages table.

## 11. Tool Use

The Agent has exactly one Tool:

```text
search_uploaded_docs(query: str)
```

The existing Tool remains unchanged:

- The model supplies only `query`.
- `UserContext` is injected server-side.
- `top_k` is controlled by application configuration/request bounds.
- Retrieval remains dense + BM25 + RRF + anchor expansion.
- Tool failures remain structured Tool results rather than unhandled storage exceptions.

Agent behavior:

- General knowledge, arithmetic, and small talk may be answered without the Tool.
- Questions about uploaded documents should call the Tool.
- Multi-part questions may cause multiple Tool calls within the existing recursion bound.
- A follow-up may use conversation history to formulate a better Tool query.

## 12. Citation Semantics

### 12.1 Current-turn scope

With persistent memory, the final graph state includes all historical ToolMessages. Returning citations by scanning the entire state would be incorrect.

The response source set is therefore:

```text
successful search_uploaded_docs citations
∩ messages produced after the current turn boundary
```

Sources are deduplicated by stable `chunk_id` while preserving first-seen order.

### 12.2 Validation

The backend is authoritative about source identity:

- A returned source must originate in a successfully parsed current-turn ToolMessage.
- Its chunk id, document name, pages, text, and link come from the Tool payload.
- Model-invented ids, file names, pages, or links are not accepted.
- Malformed Tool payloads are skipped rather than crashing the response.

### 12.3 Meaning of `ChatResponse.sources`

For the initial implementation, `sources` means:

> Evidence retrieved by the Agent and made available while producing this answer.

The system prompt instructs the model to cite only those sources. This is not mathematically identical to proving that every returned source was mentioned in the prose.

If strict prose-level selection becomes necessary, the approved extension is LangChain structured output:

```python
class AgentAnswer(BaseModel):
    answer: str
    cited_source_ids: list[str]
```

The server would intersect `cited_source_ids` with current-turn Tool results. Regex parsing of arbitrary answer text is explicitly rejected.

### 12.4 No-Tool turns

If the current turn does not call `search_uploaded_docs`, it returns:

```json
{"sources": []}
```

Earlier-turn sources are not repeated merely because they remain in memory.

## 13. API Design

### 13.1 Chat

```text
POST /api/chat
```

Request:

```json
{
  "question": "And what about employees in probation?",
  "conversation_id": "<uuid>",
  "top_k": 5,
  "openai_api_key": "optional request-only value"
}
```

Response remains:

```json
{
  "answer": "...",
  "sources": []
}
```

`conversation_id` is required and non-blank.

### 13.2 Conversation management

```text
GET    /api/conversations
GET    /api/conversations/{conversation_id}/messages
PATCH  /api/conversations/{conversation_id}
DELETE /api/conversations/{conversation_id}
```

- List returns metadata ordered by `updated_at DESC`.
- History returns the projected Human/AI turns with per-answer sources.
- Patch changes the title only.
- Delete removes catalog metadata and LangGraph thread state.
- Delete is idempotent.
- Unknown ids return 404 for history and rename.

The frontend generates a new UUID locally. A catalog row is created lazily on the first successful Agent-backed question, so abandoned empty chats do not accumulate in SQLite.

## 14. Frontend Design

The existing working-tree frontend already uses the non-streaming Agent endpoint. This behavior is the baseline.

The sidebar gains:

- New chat.
- Conversation list.
- Selected-conversation indicator.
- Rename.
- Delete.

Streamlit state contains only the currently selected id, the fetched conversation list, and a display cache of the selected conversation's messages.

Behavior:

```text
App opens
  → fetch conversation list
  → select most recent conversation, or create a local empty id

New chat
  → generate UUID
  → clear selected display cache

Submit first question
  → send UUID as conversation_id
  → backend creates catalog/checkpoint state
  → refresh conversation list

Switch chat
  → fetch projected history from backend

Delete chat
  → delete catalog + checkpoint
  → select next chat or create a new local id
```

The current “Clear Chat” button cannot merely clear `st.session_state.messages`, because that would leave backend memory active under the same id. It becomes “New chat” or explicitly deletes the selected conversation.

## 15. Streaming Design

Streaming is deliberately added after non-streaming persistence and citation scoping are proven.

The streaming implementation uses `agent.stream(...)`, not a manual model/Tool loop.

The backend converts framework events into a stable application NDJSON protocol:

```text
tool_start → sources → token* → done
```

or, when no Tool is used:

```text
token* → done
```

Event meanings:

- `tool_start`: safe Tool name and query for user feedback.
- `sources`: current-turn validated/deduplicated sources.
- `token`: final answer text fragments, excluding intermediate Tool-call content.
- `done`: one terminal success event.
- `error`: safe terminal error without secrets or internal checkpoint data.

The same conversation id/thread id and SQLite checkpointer are used for both `invoke` and `stream`; streaming does not create a second memory system.

## 16. No-Key Fallback

`create_agent` requires an LLM. If neither a request key nor server key is available, the existing direct retrieval + extractive answer remains available.

This path is explicitly stateless:

- It does not run the Agent.
- It does not create a LangGraph checkpoint.
- It does not create an empty conversation catalog row.
- The UI should communicate that conversational memory requires an LLM key.

This preserves usefulness without pretending that extractive retrieval has Agent memory.

## 17. Error Handling

### Invalid input

- Empty question → HTTP 400.
- Empty/invalid conversation id → HTTP 400.
- Invalid `top_k` → HTTP 422 or explicit bounded validation.

### Tool/storage errors

The existing Tool returns a structured error payload to the Agent. The Agent may explain that document search failed without the API endpoint crashing.

### Agent recursion limit

`GraphRecursionError` produces the existing bounded-failure response. It does not trigger a second custom loop.

### SQLite errors

- Startup failure to open/setup the checkpoint database prevents the Agent service from claiming readiness.
- Per-request write failures return a server error and do not report the turn as persisted.
- Catalog `updated_at` changes only after a successful Agent answer.

### Conversation deletion

Catalog deletion and framework thread deletion use public APIs and are idempotent. Full cross-table atomicity is not guaranteed because LangGraph owns its tables; a repeated delete repairs partially completed deletion.

## 18. Privacy and Security Boundaries

- Request-provided API keys are never stored in checkpoints, conversation metadata, Tool payloads, logs, or frontend message history.
- Raw ToolMessages are not exposed through conversation-history APIs.
- System prompts and graph configuration are not returned.
- Conversation ids are opaque and non-path-like.
- This phase remains a single-user local application: it does not add authentication or authorization.
- `UserContext` remains reserved but does not yet filter Milvus results.
- Anyone with network access to the unprotected backend and a conversation id could access that conversation; production authentication is required before multi-user deployment.

## 19. Context Growth

The initial version retains the complete message history for each thread. This is acceptable for the intended daily-use prototype but has a finite model context limit.

This design intentionally does not add manual compression or summary state.

When real usage demonstrates long-thread pressure, the next design should prefer LangChain/LangGraph-supported trimming or summarization middleware. That must be driven by observed context length and tested separately, rather than copied from Hermes preemptively.

## 20. Alternatives Considered

### 20.1 Streamlit `session_state` as memory

Rejected because it is display-only, disappears with the UI session, cannot reliably survive refresh/backend restart, and does not make prior messages available to the backend Agent.

### 20.2 `InMemorySaver`

Rejected for the target daily-use behavior because backend restart loses all conversations. It remains useful in unit tests.

### 20.3 Custom Hermes-style Agent runtime

Rejected because the project currently needs only one Tool and short-term conversation state. `create_agent` plus a checkpointer already provides the required loop and persistence.

### 20.4 Low-level custom LangGraph `StateGraph`

Rejected because there are no custom routing or state-transition requirements. It would expose implementation details without adding user value.

### 20.5 Separate application messages table

Rejected because it would duplicate LangGraph checkpoint messages and create synchronization problems. Only catalog metadata is stored separately.

### 20.6 Store conversations in Milvus

Rejected because chat messages require ordered transactional state and exact thread lookup, not semantic vector search. Milvus remains dedicated to document retrieval.

### 20.7 One Milvus collection per conversation

Rejected because uploaded documents are shared application data in the current single-user design; conversation memory and document scope are different concerns.

### 20.8 Parse citations from arbitrary Markdown

Rejected as the source of truth because model text can be malformed or invented. Tool results remain authoritative; strict model-selected citations would use structured output.

## 21. Testing Strategy

### Framework-independent tests

- Conversation schema validation.
- Conversation catalog CRUD and timestamp ordering.
- Current-turn ToolMessage boundary filtering.
- Citation deduplication and validation.
- Conversation-history projection.

### LangGraph persistence tests

Using a temporary SQLite file and deterministic fake model:

- Turn 2 sees turn 1 in the same thread.
- Conversation B cannot see conversation A.
- State survives runtime close/reopen.
- Deleted thread cannot be restored.

### Existing Tool/retrieval regression tests

- Tool schema still exposes only `query`.
- Dense and BM25 retrieval still work.
- Hybrid context and links still work.
- Upload/index/delete behavior is unchanged.

### API/UI scenarios

- Create/list/switch/rename/delete conversations.
- Browser refresh restores list and selected history.
- Document question produces current-turn sources.
- General question produces no sources.
- Pronoun-based follow-up uses prior context.
- Fresh conversation does not inherit that context.
- No-key fallback remains available and is marked stateless.

### Live smoke test

One explicitly marked test uses a real OpenAI key and live Milvus to verify actual model Tool choice. Deterministic persistence tests do not depend on paid or nondeterministic model calls.

## 22. Success Criteria

The design is successful when:

- The frontend's primary chat path is the Agent.
- Exactly one LLM-visible Tool exists and `tools.py` is unchanged.
- LangChain `create_agent` owns all Tool-loop behavior.
- LangGraph SQLite checkpoints own multi-turn message state.
- No custom Agent state or graph topology exists.
- Conversations are isolated by `thread_id` and survive backend restart.
- The UI can discover and reopen conversations through the metadata catalog.
- Reopened assistant turns retain their source panels.
- Current responses never include citations solely from older turns.
- Unknown citation ids are not trusted.
- Request API keys are never persisted.
- Existing PDF upload and hybrid retrieval behavior remains intact.

## 23. Explicitly Out of Scope

- Changes to `backend/app/agent/tools.py`.
- Additional Tools.
- Authentication and per-user authorization.
- User/session filtering in Milvus.
- Per-conversation document collections.
- Cross-conversation long-term semantic memory.
- Manual context compaction or Hermes-style summaries.
- OCR and table-aware PDF extraction.
- Background jobs, human approval steps, resumable tasks, or distributed runs.
- Production multi-worker SQLite guarantees.
- PostgreSQL checkpoint migration.

## 24. Relationship to Earlier Specs

- `2026-08-06-langchain-agent-layer-design.md` introduced the one-tool Agent and remains authoritative for the Tool boundary and autonomous Tool decision.
- `2026-08-16-hybrid-sparse-vector-upload-design.md` remains authoritative for upload-time dense/BM25 storage.
- `2026-08-16-hybrid-query-retrieval-design.md` remains authoritative for hybrid retrieval and citation-block construction.
- This spec supersedes the earlier Agent spec only where it said requests were stateless, the frontend was unchanged, or streaming was out of scope. It does not supersede its one-tool design.

