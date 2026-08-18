# React Frontend (`frontend-web/`) — Design Spec

## 1. Goal

Build a second, professional-looking frontend for RAG Document Chat as a new `frontend-web/` React + TypeScript app, reaching full feature parity with the existing Streamlit `frontend/`, without touching or removing `frontend/`. Once the new frontend is verified item-by-item against Streamlit, `frontend/` will be deleted in a later, separate change (not part of this spec).

Both frontends talk to the same unmodified FastAPI backend (`backend/`), documented in `docs/architecture.md`, `docs/query-workflow.md`, `docs/agent-runtime-workflow.md`, and `docs/pdf-upload-workflow.md`. This spec does not change any backend behavior or API contract.

## 2. Stack

| Concern | Choice | Why |
| --- | --- | --- |
| Build tool | Vite | Fast dev server, native TS, simple static build for Docker/nginx. |
| Language | TypeScript (strict) | Backend contracts are typed (Pydantic); mirror that on the client. |
| UI framework | React 18 | Team-standard, large ecosystem, matches the requested `src/` shape. |
| Styling | Tailwind CSS | Utility-first, fast iteration, easy dark-mode variants (`dark:` prefix). |
| Components | shadcn/ui (Radix primitives) | Accessible Dialog/DropdownMenu/Tabs/Tooltip/Toast primitives that ship as owned source, not an opaque dependency — themeable to a professional, non-generic look. |
| Server state | TanStack Query | Caching/refetch for conversations, history, documents, evaluation; avoids hand-rolled loading/error bookkeeping. |
| UI state | Zustand | Small store for `activeConversationId`, theme, sidebar-collapsed, session API key. |
| Markdown | `react-markdown` + `remark-gfm` | Agent answers may contain lists/tables/code; render them properly instead of raw text. |
| Package manager | npm | Zero extra tooling; matches default Node Docker images. |
| Testing | None automated (v1) | Per user decision: rely on the manual Streamlit-parity pass (build step 8). Can add Vitest/RTL later. |

## 3. Directory Structure

```text
00_RAG_Document_Chat/
├── backend/                     # unchanged
├── frontend/                    # unchanged, kept until manual verification passes
├── frontend-web/
│   ├── src/
│   │   ├── api/
│   │   │   ├── client.ts        # fetch wrapper, VITE_API_BASE_URL, error normalization
│   │   │   ├── chat.ts          # postChat, streamChat (NDJSON)
│   │   │   ├── conversations.ts # list/history/rename/delete
│   │   │   ├── documents.ts     # upload/list/stats/delete
│   │   │   └── evaluation.ts    # runEvaluation
│   │   ├── components/
│   │   │   ├── layout/          # AppShell, Sidebar, MainPanel, TopTabs
│   │   │   ├── chat/            # MessageList, MessageBubble, ToolStatusPill,
│   │   │   │                    # CitationCard, ChatInput
│   │   │   ├── conversations/   # ConversationList, ConversationItem, RenameDialog
│   │   │   ├── documents/       # UploadDropzone, DocumentList, DocumentStats
│   │   │   ├── settings/        # ApiKeyField, ThemeToggle
│   │   │   ├── evaluation/      # EvaluationPanel, EvaluationResultCard
│   │   │   └── ui/              # shadcn primitives (button, dialog, tabs, toast, ...)
│   │   ├── hooks/
│   │   │   ├── useChatStream.ts # fetch+ReadableStream NDJSON state machine
│   │   │   ├── useConversations.ts
│   │   │   ├── useDocuments.ts
│   │   │   └── useTheme.ts
│   │   ├── pages/
│   │   │   ├── ChatPage.tsx
│   │   │   └── EvaluationPage.tsx
│   │   ├── store/
│   │   │   └── uiStore.ts       # Zustand: activeConversationId, theme, sidebarOpen, apiKey
│   │   ├── types/
│   │   │   ├── chat.ts
│   │   │   ├── conversation.ts
│   │   │   ├── document.ts
│   │   │   └── evaluation.ts
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── postcss.config.js
│   ├── .env.example             # VITE_API_BASE_URL=http://127.0.0.1:8000
│   ├── nginx.conf                # static-serve config for the Docker image
│   └── Dockerfile                # multi-stage: node build -> nginx runtime
└── docker-compose.yml            # gains a frontend-web service
```

`hooks/` was in the user's requested tree without `store/`; Zustand's store is small enough to live in one file and is listed explicitly here for clarity, but it is an implementation detail under `src/`, not a structural deviation from the requested top-level layout (`api/ components/ hooks/ pages/ types/ App.tsx` all present as requested).

## 4. Types (`src/types/`)

Mirror `backend/app/schemas.py` field-for-field so drift is visible immediately:

```ts
// chat.ts
export interface Source {
  text: string;
  document: string;
  page: number | string;
  chunk: number | string;
  score: number | null;
  pages: number[] | null;
  link: string | null;
}
export interface ChatRequest {
  question: string;
  conversation_id: string;
  top_k?: number;
  openai_api_key?: string;
}
export interface ChatResponse {
  answer: string;
  sources: Source[];
}
export type ChatStreamEvent =
  | { type: "tool_start"; tool: string; query: string }
  | { type: "sources"; data: Source[] }
  | { type: "token"; data: string }
  | { type: "notice"; data: string }
  | { type: "error"; data: string }
  | { type: "done" };

// conversation.ts
export interface ConversationSummary {
  id: string; title: string; created_at: string; updated_at: string;
}
export interface ConversationMessage {
  role: "user" | "assistant"; content: string; sources: Source[];
}
export interface ConversationHistoryResponse {
  conversation: ConversationSummary; messages: ConversationMessage[];
}

// document.ts
export interface DocumentRecord {
  document_name: string; file_hash: string; pages: number; chunks: number;
}
export interface DocumentListResponse {
  total_chunks: number; total_documents: number; documents: DocumentRecord[];
}
export interface UploadResponse { added_chunks: number; messages: string[]; }

// evaluation.ts
export interface EvaluationResult {
  id: string; question: string; expected_answer: string;
  expected_document: string; expected_page: number;
  expected_keywords: string[]; retrieved_sources: Source[];
  document_hit_score: number; page_hit_score: number;
  keyword_score: number; final_score: number;
}
export interface EvaluationResponse {
  status: string; tests_run: number; average_final_score: number;
  required_documents: string[]; indexed_documents: string[];
  missing_documents: string[]; results: EvaluationResult[];
}
```

`DocumentRecord` is inferred from actual `document_list()`/`list_documents()` shapes (`backend/app/rag/vector_store.py`), which return more fields than the currently-narrow `DocumentStats` Pydantic model — the frontend type follows the real JSON, not the incomplete schema.

## 5. Streaming Design (`useChatStream`)

`/api/chat/stream` returns `application/x-ndjson` from a synchronous-looking POST; `EventSource` cannot send a POST body, so:

1. `fetch(url, { method: "POST", body: JSON.stringify(req) })`, read `response.body.getReader()`.
2. Decode chunks with `TextDecoder`, split on `\n`, buffer any incomplete trailing line, `JSON.parse` each complete line into a `ChatStreamEvent`.
3. Reduce events into local state: `tool_start` → show a `ToolStatusPill` with the query; `sources` → replace the pending citation set (dedup already done server-side); `token` → append to the streaming draft string; `notice` → show a dismissible inline banner (no-key fallback case); `error` → replace the draft with an error bubble; `done` → finalize the message into the conversation's message list and invalidate the `["conversations"]` query so the sidebar re-sorts/re-titles.
4. Abort support: an `AbortController` tied to the input box, so navigating away or starting a new question cancels the in-flight stream cleanly.

This hook is the direct client-side counterpart of the event table in `docs/query-workflow.md` §Step 6 — no new event types, no reinterpretation.

## 6. State Management

- **Server state (TanStack Query)**: `conversations`, `conversation:{id}` history, `documents`, `documentStats`. Mutations (`upload`, `deleteDocument`, `renameConversation`, `deleteConversation`) invalidate the relevant query keys, matching Streamlit's `refresh_conversations()` / `st.rerun()` pattern but declaratively.
- **UI state (Zustand `uiStore`)**: `activeConversationId`, `theme: "light" | "dark" | "system"`, `sidebarOpen`, `apiKey` (kept in memory + mirrored to `sessionStorage` only, cleared on tab close — never `localStorage`, matching the backend doc's "request-scoped key" model).
- Streaming draft state (current in-flight tokens/sources/tool-status) lives locally in `useChatStream`, not in global state — it's ephemeral and only one stream runs at a time.

## 7. Layout & Routing

No router library; two views toggled by a top `Tabs` bar (`Chat`, `Evaluation`), mirroring Streamlit's `st.tabs(["Chat", "Evaluation"])` exactly.

- **Sidebar** (collapsible via `sidebarOpen`): New Chat button → Conversation list (click to select, inline rename via dialog, delete with confirm) → Document panel (dropzone upload with per-file result toasts, stats, per-document delete) → Settings (API key field, theme toggle).
- **Main / Chat tab**: conversation title (editable inline, same rename action as sidebar) → scrollable `MessageList` (`MessageBubble` for user/assistant, `ToolStatusPill` shown transiently while `tool_start` is active and no `sources`/`done` yet, `CitationCard` list collapsible under an assistant message) → `ChatInput` pinned to the bottom, disabled while a stream is in flight, with the same empty-question guard as the backend (`HTTP 400`).
- **Evaluation tab**: required/indexed/missing sample-doc lists, "Run Evaluation" button, summary metrics, per-question expandable result cards with the four scores and retrieved sources — full parity with `render_evaluation_panel()`.

## 8. Theming, Responsiveness, Errors

- Dark mode: Tailwind `class` strategy, `ThemeToggle` writes `theme` to `uiStore` (persisted to `localStorage` — theme preference, not a secret) and toggles a `dark` class on `<html>`; defaults to `system` via `prefers-color-scheme` on first load.
- Responsive: sidebar collapses to an off-canvas drawer below a `md` breakpoint; message list and citation cards use fluid widths, no fixed px layouts.
- Errors: every API call surfaces failures through a shared toast (shadcn `Toast`) plus inline state where Streamlit currently uses `st.error`/`st.warning` (e.g., failed document stats, failed rename) — same failure semantics as `docs/query-workflow.md` §7, just rendered as toasts/banners instead of Streamlit alerts.

## 9. Docker & Compose

- `frontend-web/Dockerfile`: stage 1 `node:20-alpine` → `npm ci && npm run build`; stage 2 `nginx:alpine` serving `/dist` with `nginx.conf` doing SPA fallback (`try_files ... /index.html`) and proxying `/api/*` is **not** needed — the client calls the backend directly via `VITE_API_BASE_URL` baked in at build time (same pattern as Streamlit's `API_BASE_URL` env var).
- `docker-compose.yml`: add a `frontend-web` service (new host port, e.g. `5173:80`) alongside the existing `frontend` service — both run side by side until the user deletes `frontend/` after verification. No changes to `backend` or `milvus` services.

## 10. Feature Parity Checklist

This is the literal checklist for the user's step 8 ("和 Streamlit 逐项对比验证"):

- [ ] New chat creates a fresh UUID thread, empty message list, no premature sidebar entry (catalog row appears only after first successful turn — matches backend `ConversationStore.ensure()`).
- [ ] Conversation list: select, rename, delete (delete falls back to next available or a fresh chat).
- [ ] History restores correctly after page reload / switching conversations.
- [ ] PDF upload: multi-file, per-file status message (indexed / already indexed / unreadable), disabled while empty selection.
- [ ] Document stats (chunk/document counts) and per-document list with page/chunk counts and delete.
- [ ] Session-only OpenAI API key input, never persisted to disk/localStorage.
- [ ] Chat send → streaming tokens → tool-status indicator → citation cards → persisted after `done`.
- [ ] No-key fallback: notice banner, non-persisted stateless answer.
- [ ] Error states: empty question, request failure, agent recursion-limit message, unknown-conversation 404.
- [ ] Evaluation: required/missing sample docs warning, run button, summary metrics, per-question score breakdown + sources.

## 11. Explicit Non-Goals

- No changes to `backend/`, its API contracts, or its docs.
- No removal of `frontend/` in this change — deletion happens later, by the user, after manual verification.
- No authentication/user accounts — parity target is the existing single-user Streamlit app.
- No automated frontend test suite in v1 (explicit user decision).