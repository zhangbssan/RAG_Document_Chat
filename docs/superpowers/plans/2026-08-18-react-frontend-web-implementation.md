# React Frontend (`frontend-web/`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new `frontend-web/` React + TypeScript app that reaches full feature parity with the existing Streamlit `frontend/`, talking to the same unmodified FastAPI backend, without touching or removing `frontend/`.

**Architecture:** Vite + React 18 + TypeScript SPA. Two views (`Chat`, `Evaluation`) toggled by top tabs, no router. TanStack Query owns server state (conversations, history, documents, evaluation); a small Zustand store owns UI state (active conversation, sidebar, theme, session API key). `/api/chat/stream` NDJSON is consumed via `fetch` + `ReadableStream` (not `EventSource`, which cannot send a POST body). Tailwind CSS + shadcn/ui (Radix) provide the visual system.

**Tech Stack:** Vite 5, React 18, TypeScript 5, Tailwind CSS 3, shadcn/ui, TanStack Query 5, Zustand 4, react-markdown 9 + remark-gfm 4, npm.

**Spec:** `docs/superpowers/specs/2026-08-18-react-frontend-design.md`

## Global Constraints

- Do not modify anything under `backend/` or `frontend/` in this plan — `frontend-web/` is additive only.
- All new frontend code lives under `frontend-web/` at the repo root (`00_RAG_Document_Chat/frontend-web/`).
- No automated test suite in this pass — verification is `tsc --noEmit`, `npm run build`, and manual checks against a running backend, per the user's explicit decision.
- API base URL is configured via `VITE_API_BASE_URL` (build-time env var), mirroring Streamlit's `API_BASE_URL` pattern — never hardcode `http://127.0.0.1:8000` outside `.env.example`/`api/client.ts`'s fallback default.
- The OpenAI API key typed into Settings is session-scoped: kept in memory + `sessionStorage` only, never `localStorage`.
- NDJSON stream event vocabulary is fixed by the backend and already documented in `docs/query-workflow.md` §Step 6: `tool_start`, `sources`, `token`, `notice`, `error`, `done`. Do not invent new event types.
- Every task must leave `npm run build` (in `frontend-web/`) passing before it is considered done.

---

## Task 1: Scaffold `frontend-web/` (Vite + React + TS + Tailwind + shadcn/ui)

**Files:**
- Create: `frontend-web/` (via `npm create vite@latest`)
- Create: `frontend-web/tailwind.config.ts`
- Create: `frontend-web/postcss.config.js`
- Create: `frontend-web/src/index.css`
- Modify: `frontend-web/vite.config.ts`
- Modify: `frontend-web/tsconfig.json`, `frontend-web/tsconfig.app.json`
- Create: `frontend-web/.env.example`
- Create: `frontend-web/components.json` (via shadcn init)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a working Vite dev/build pipeline, Tailwind utility classes available in any `.tsx`, the `@/*` → `frontend-web/src/*` path alias, and shadcn/ui's `cn()` helper at `frontend-web/src/lib/utils.ts` — every later task's components rely on `@/` imports and `cn()`.

- [ ] **Step 1: Scaffold the Vite project**

Run from `00_RAG_Document_Chat/`:

```bash
npm create vite@latest frontend-web -- --template react-ts
cd frontend-web
npm install
```

- [ ] **Step 2: Install runtime dependencies**

```bash
npm install @tanstack/react-query zustand react-markdown remark-gfm lucide-react
```

- [ ] **Step 3: Install and configure Tailwind CSS**

```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

Replace `frontend-web/tailwind.config.ts` (rename the generated `.js` to `.ts`) with:

```ts
import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [],
};

export default config;
```

Replace `frontend-web/src/index.css` with:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222 47% 11%;
    --card: 0 0% 100%;
    --card-foreground: 222 47% 11%;
    --primary: 221 83% 53%;
    --primary-foreground: 210 40% 98%;
    --muted: 210 40% 96%;
    --muted-foreground: 215 16% 47%;
    --border: 214 32% 91%;
    --destructive: 0 72% 51%;
    --destructive-foreground: 210 40% 98%;
    --radius: 0.5rem;
  }

  :root.dark {
    --background: 222 47% 8%;
    --foreground: 210 40% 98%;
    --card: 222 47% 11%;
    --card-foreground: 210 40% 98%;
    --primary: 217 91% 60%;
    --primary-foreground: 222 47% 11%;
    --muted: 217 33% 17%;
    --muted-foreground: 215 20% 65%;
    --border: 217 33% 20%;
    --destructive: 0 63% 51%;
    --destructive-foreground: 210 40% 98%;
  }

  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground;
  }
}
```

- [ ] **Step 4: Configure the `@/` path alias**

Add to `frontend-web/tsconfig.json` (inside the `compilerOptions` of `tsconfig.app.json` if the scaffold split configs — check which file has `"include": ["src"]` and edit that one):

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

Install the node types needed for `vite.config.ts` path resolution and update `frontend-web/vite.config.ts`:

```bash
npm install -D @types/node
```

```ts
import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
  },
});
```

- [ ] **Step 5: Initialize shadcn/ui**

```bash
npx shadcn@latest init -d
npx shadcn@latest add button dialog dropdown-menu tabs tooltip toast input textarea avatar scroll-area separator switch badge
```

This creates `frontend-web/components.json`, `frontend-web/src/lib/utils.ts` (exporting `cn()`), and `frontend-web/src/components/ui/*.tsx` for each primitive listed.

- [ ] **Step 6: Add the backend base-URL env var**

Create `frontend-web/.env.example`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Create `frontend-web/.env` with the same content (this file should already be covered by Vite's default `.gitignore` entry for `.env*.local`; explicitly add `.env` to `frontend-web/.gitignore` if the scaffold didn't already ignore it, since it may later hold a real backend URL).

- [ ] **Step 7: Verify the scaffold builds**

Run: `cd frontend-web && npm run build`
Expected: build succeeds, `frontend-web/dist/` is created, no TypeScript errors.

Run: `cd frontend-web && npm run dev`
Expected: dev server starts on port 5173; visiting it in a browser shows the default Vite/React starter page with no console errors.

- [ ] **Step 8: Commit**

```bash
git add frontend-web
git commit -m "Scaffold frontend-web: Vite + React + TS + Tailwind + shadcn/ui"
```

---

## Task 2: TypeScript API Types

**Files:**
- Create: `frontend-web/src/types/chat.ts`
- Create: `frontend-web/src/types/conversation.ts`
- Create: `frontend-web/src/types/document.ts`
- Create: `frontend-web/src/types/evaluation.ts`
- Create: `frontend-web/src/types/index.ts` (barrel re-export)

**Interfaces:**
- Consumes: nothing beyond the scaffold from Task 1.
- Produces: `Source`, `ChatRequest`, `ChatResponse`, `ChatStreamEvent` (chat.ts); `ConversationSummary`, `ConversationMessage`, `ConversationHistoryResponse`, `ConversationListResponse` (conversation.ts); `DocumentRecord`, `DocumentListResponse`, `UploadResponse` (document.ts); `EvaluationResult`, `EvaluationResponse` (evaluation.ts). Every later task imports these from `@/types`.

- [ ] **Step 1: Write `frontend-web/src/types/chat.ts`**

```ts
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
```

- [ ] **Step 2: Write `frontend-web/src/types/conversation.ts`**

```ts
import type { Source } from "./chat";

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
  sources: Source[];
}

export interface ConversationHistoryResponse {
  conversation: ConversationSummary;
  messages: ConversationMessage[];
}

export interface ConversationListResponse {
  conversations: ConversationSummary[];
}
```

- [ ] **Step 3: Write `frontend-web/src/types/document.ts`**

```ts
export interface DocumentRecord {
  document_name: string;
  file_hash: string;
  pages: number;
  chunks: number;
}

export interface DocumentListResponse {
  total_chunks: number;
  total_documents: number;
  documents: DocumentRecord[];
}

export interface UploadResponse {
  added_chunks: number;
  messages: string[];
}

export interface DocumentDeleteResponse {
  message: string;
  document_name: string;
  file_hash: string;
  deleted_chunks: number;
  deleted_file: boolean;
}
```

- [ ] **Step 4: Write `frontend-web/src/types/evaluation.ts`**

```ts
import type { Source } from "./chat";

export interface EvaluationResult {
  id: string;
  question: string;
  expected_answer: string;
  expected_document: string;
  expected_page: number;
  expected_keywords: string[];
  retrieved_sources: Source[];
  document_hit_score: number;
  page_hit_score: number;
  keyword_score: number;
  final_score: number;
}

export interface EvaluationResponse {
  status: string;
  tests_run: number;
  average_final_score: number;
  required_documents: string[];
  indexed_documents: string[];
  missing_documents: string[];
  results: EvaluationResult[];
}
```

- [ ] **Step 5: Write the barrel file `frontend-web/src/types/index.ts`**

```ts
export * from "./chat";
export * from "./conversation";
export * from "./document";
export * from "./evaluation";
```

- [ ] **Step 6: Verify with a type-check**

Run: `cd frontend-web && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add frontend-web/src/types
git commit -m "Add frontend-web TypeScript types mirroring backend schemas"
```

---

## Task 3: API Client Layer

**Files:**
- Create: `frontend-web/src/api/client.ts`
- Create: `frontend-web/src/api/chat.ts`
- Create: `frontend-web/src/api/conversations.ts`
- Create: `frontend-web/src/api/documents.ts`
- Create: `frontend-web/src/api/evaluation.ts`

**Interfaces:**
- Consumes: types from `@/types` (Task 2).
- Produces: `apiFetch`, `ApiError`, `getApiBaseUrl` (client.ts); `postChat`, `streamChat` (chat.ts); `listConversations`, `getConversationHistory`, `renameConversation`, `deleteConversation` (conversations.ts); `uploadPdfs`, `listDocuments`, `deleteDocument` (documents.ts); `runEvaluation` (evaluation.ts). All later hooks (Tasks 5, 6, 8, 9) call exactly these function names/signatures.

- [ ] **Step 1: Write `frontend-web/src/api/client.ts`**

```ts
export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function getApiBaseUrl(): string {
  const base = import.meta.env.VITE_API_BASE_URL as string | undefined;
  return (base || "http://127.0.0.1:8000").replace(/\/$/, "");
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body?.detail || detail;
    } catch {
      // response had no JSON body
    }
    throw new ApiError(response.status, detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}
```

- [ ] **Step 2: Write `frontend-web/src/api/chat.ts`**

```ts
import { ApiError, getApiBaseUrl } from "./client";
import type { ChatRequest, ChatResponse, ChatStreamEvent } from "@/types";

export async function postChat(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw new ApiError(response.status, response.statusText);
  }
  return (await response.json()) as ChatResponse;
}

export async function streamChat(
  request: ChatRequest,
  onEvent: (event: ChatStreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${getApiBaseUrl()}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new ApiError(response.status, response.statusText || "Stream request failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        onEvent(JSON.parse(trimmed) as ChatStreamEvent);
      } catch {
        // ignore malformed NDJSON lines
      }
    }
  }

  const remaining = buffer.trim();
  if (remaining) {
    try {
      onEvent(JSON.parse(remaining) as ChatStreamEvent);
    } catch {
      // ignore
    }
  }
}
```

- [ ] **Step 3: Write `frontend-web/src/api/conversations.ts`**

```ts
import { apiFetch } from "./client";
import type {
  ConversationHistoryResponse,
  ConversationListResponse,
  ConversationSummary,
} from "@/types";

export async function listConversations(): Promise<ConversationSummary[]> {
  const result = await apiFetch<ConversationListResponse>("/api/conversations");
  return result.conversations;
}

export async function getConversationHistory(
  conversationId: string
): Promise<ConversationHistoryResponse> {
  return apiFetch<ConversationHistoryResponse>(
    `/api/conversations/${conversationId}/messages`
  );
}

export async function renameConversation(
  conversationId: string,
  title: string
): Promise<ConversationSummary> {
  return apiFetch<ConversationSummary>(`/api/conversations/${conversationId}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export async function deleteConversation(
  conversationId: string
): Promise<{ conversation_id: string; deleted: boolean }> {
  return apiFetch(`/api/conversations/${conversationId}`, { method: "DELETE" });
}
```

- [ ] **Step 4: Write `frontend-web/src/api/documents.ts`**

```ts
import { ApiError, getApiBaseUrl } from "./client";
import type { DocumentDeleteResponse, DocumentListResponse, UploadResponse } from "@/types";

export async function uploadPdfs(files: File[]): Promise<UploadResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const response = await fetch(`${getApiBaseUrl()}/api/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body?.detail || detail;
    } catch {
      // no JSON body
    }
    throw new ApiError(response.status, detail);
  }

  return (await response.json()) as UploadResponse;
}

export async function listDocuments(): Promise<DocumentListResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/documents/list`);
  if (!response.ok) {
    throw new ApiError(response.status, response.statusText);
  }
  return (await response.json()) as DocumentListResponse;
}

export async function deleteDocument(fileHash: string): Promise<DocumentDeleteResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/documents/${fileHash}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new ApiError(response.status, response.statusText);
  }
  return (await response.json()) as DocumentDeleteResponse;
}
```

`documents.ts` uses raw `fetch` instead of `apiFetch` because `/api/upload` sends `multipart/form-data` and must not carry the JSON `Content-Type` header `apiFetch` always sets; the two document GET/DELETE calls stay on raw `fetch` too for consistency within the file.

- [ ] **Step 5: Write `frontend-web/src/api/evaluation.ts`**

```ts
import { apiFetch } from "./client";
import type { EvaluationResponse } from "@/types";

export async function runEvaluation(): Promise<EvaluationResponse> {
  return apiFetch<EvaluationResponse>("/api/evaluate", { method: "POST" });
}
```

- [ ] **Step 6: Verify with a type-check and build**

Run: `cd frontend-web && npx tsc --noEmit && npm run build`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add frontend-web/src/api
git commit -m "Add frontend-web API client layer"
```

---

## Task 4: Zustand UI Store + App Shell Layout

**Files:**
- Create: `frontend-web/src/store/uiStore.ts`
- Create: `frontend-web/src/components/layout/AppShell.tsx`
- Create: `frontend-web/src/components/layout/Sidebar.tsx`
- Create: `frontend-web/src/components/layout/TopTabs.tsx`
- Modify: `frontend-web/src/App.tsx`
- Modify: `frontend-web/src/main.tsx`

**Interfaces:**
- Consumes: shadcn `Tabs` primitive from Task 1 (`@/components/ui/tabs`).
- Produces: `useUiStore` (state: `activeConversationId`, `sidebarOpen`, `theme`, `apiKey`; actions: `setActiveConversationId`, `setSidebarOpen`, `toggleSidebar`, `setTheme`, `setApiKey`) — every later task reads/writes this store. `AppShell` renders `Sidebar` + main content area + `TopTabs`; `ChatPage`/`EvaluationPage` (Tasks 6, 9) are its children, passed as props.

- [ ] **Step 1: Write `frontend-web/src/store/uiStore.ts`**

```ts
import { create } from "zustand";

export type Theme = "light" | "dark" | "system";

interface UiState {
  activeConversationId: string | null;
  sidebarOpen: boolean;
  theme: Theme;
  apiKey: string;
  setActiveConversationId: (id: string | null) => void;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
  setTheme: (theme: Theme) => void;
  setApiKey: (key: string) => void;
}

const THEME_STORAGE_KEY = "rag-chat-theme";
const API_KEY_SESSION_KEY = "rag-chat-api-key";

function readInitialTheme(): Theme {
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  return stored === "light" || stored === "dark" || stored === "system" ? stored : "system";
}

function readInitialApiKey(): string {
  return window.sessionStorage.getItem(API_KEY_SESSION_KEY) || "";
}

export const useUiStore = create<UiState>((set) => ({
  activeConversationId: null,
  sidebarOpen: true,
  theme: readInitialTheme(),
  apiKey: readInitialApiKey(),
  setActiveConversationId: (id) => set({ activeConversationId: id }),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setTheme: (theme) => {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    set({ theme });
  },
  setApiKey: (key) => {
    if (key) {
      window.sessionStorage.setItem(API_KEY_SESSION_KEY, key);
    } else {
      window.sessionStorage.removeItem(API_KEY_SESSION_KEY);
    }
    set({ apiKey: key });
  },
}));
```

- [ ] **Step 2: Wire `QueryClientProvider` in `frontend-web/src/main.tsx`**

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>
);
```

- [ ] **Step 3: Write `frontend-web/src/components/layout/TopTabs.tsx`**

```tsx
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

export type View = "chat" | "evaluation";

interface TopTabsProps {
  active: View;
  onChange: (view: View) => void;
}

export function TopTabs({ active, onChange }: TopTabsProps) {
  return (
    <Tabs value={active} onValueChange={(value) => onChange(value as View)}>
      <TabsList>
        <TabsTrigger value="chat">Chat</TabsTrigger>
        <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
      </TabsList>
    </Tabs>
  );
}
```

- [ ] **Step 4: Write `frontend-web/src/components/layout/Sidebar.tsx` (structural shell only — real content lands in Tasks 5 and 8)**

```tsx
import type { ReactNode } from "react";
import { useUiStore } from "@/store/uiStore";
import { cn } from "@/lib/utils";

interface SidebarProps {
  children: ReactNode;
}

export function Sidebar({ children }: SidebarProps) {
  const sidebarOpen = useUiStore((state) => state.sidebarOpen);

  return (
    <aside
      className={cn(
        "flex h-full w-80 shrink-0 flex-col gap-6 overflow-y-auto border-r border-border bg-card p-4 transition-transform md:static md:translate-x-0",
        sidebarOpen ? "translate-x-0" : "-translate-x-full",
        "fixed inset-y-0 left-0 z-40 md:relative"
      )}
    >
      {children}
    </aside>
  );
}
```

- [ ] **Step 5: Write `frontend-web/src/components/layout/AppShell.tsx`**

```tsx
import type { ReactNode } from "react";
import { Menu } from "lucide-react";
import { Sidebar } from "./Sidebar";
import { TopTabs, type View } from "./TopTabs";
import { Button } from "@/components/ui/button";
import { useUiStore } from "@/store/uiStore";

interface AppShellProps {
  sidebar: ReactNode;
  activeView: View;
  onViewChange: (view: View) => void;
  children: ReactNode;
}

export function AppShell({ sidebar, activeView, onViewChange, children }: AppShellProps) {
  const toggleSidebar = useUiStore((state) => state.toggleSidebar);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
      <Sidebar>{sidebar}</Sidebar>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-border p-4">
          <Button variant="ghost" size="icon" className="md:hidden" onClick={toggleSidebar}>
            <Menu className="h-5 w-5" />
          </Button>
          <h1 className="text-lg font-semibold">RAG Document Chat</h1>
          <div className="ml-auto">
            <TopTabs active={activeView} onChange={onViewChange} />
          </div>
        </header>
        <main className="min-h-0 flex-1 overflow-hidden">{children}</main>
      </div>
    </div>
  );
}
```

- [ ] **Step 6: Wire `frontend-web/src/App.tsx` with placeholder views**

```tsx
import { useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import type { View } from "@/components/layout/TopTabs";

export default function App() {
  const [view, setView] = useState<View>("chat");

  return (
    <AppShell
      sidebar={<div className="text-sm text-muted-foreground">Sidebar content lands in later tasks.</div>}
      activeView={view}
      onViewChange={setView}
    >
      {view === "chat" ? (
        <div className="p-6 text-sm text-muted-foreground">Chat view lands in Task 6.</div>
      ) : (
        <div className="p-6 text-sm text-muted-foreground">Evaluation view lands in Task 9.</div>
      )}
    </AppShell>
  );
}
```

- [ ] **Step 7: Verify manually**

Run: `cd frontend-web && npm run dev`
Expected: browser shows a header with "RAG Document Chat" title and Chat/Evaluation tabs, a left sidebar placeholder, and switching tabs swaps the main-area placeholder text. No console errors.

- [ ] **Step 8: Commit**

```bash
git add frontend-web/src
git commit -m "Add frontend-web app shell: sidebar, top tabs, UI store"
```

---

## Task 5: Conversation List and Switching

**Files:**
- Create: `frontend-web/src/hooks/useConversations.ts`
- Create: `frontend-web/src/components/conversations/ConversationList.tsx`
- Create: `frontend-web/src/components/conversations/ConversationItem.tsx`
- Create: `frontend-web/src/components/conversations/RenameDialog.tsx`
- Modify: `frontend-web/src/App.tsx`

**Interfaces:**
- Consumes: `listConversations`, `renameConversation`, `deleteConversation`, `getConversationHistory` (Task 3 `@/api/conversations`); `useUiStore` (Task 4).
- Produces: `useConversations()`, `useConversationHistory(id)`, `useRenameConversation()`, `useDeleteConversation()` — Task 6's `ChatPage` consumes `useConversationHistory` and the same query key `["conversations"]` this task defines, so streaming completion (Task 6) can invalidate it correctly.

- [ ] **Step 1: Write `frontend-web/src/hooks/useConversations.ts`**

```ts
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  deleteConversation,
  getConversationHistory,
  listConversations,
  renameConversation,
} from "@/api/conversations";

export const conversationsQueryKey = ["conversations"] as const;
export const conversationHistoryQueryKey = (id: string) => ["conversation", id] as const;

export function useConversations() {
  return useQuery({
    queryKey: conversationsQueryKey,
    queryFn: listConversations,
  });
}

export function useConversationHistory(conversationId: string | null) {
  return useQuery({
    queryKey: conversationId ? conversationHistoryQueryKey(conversationId) : ["conversation", "none"],
    queryFn: () => getConversationHistory(conversationId as string),
    enabled: Boolean(conversationId),
  });
}

export function useRenameConversation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) => renameConversation(id, title),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: conversationsQueryKey });
    },
  });
}

export function useDeleteConversation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => deleteConversation(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: conversationsQueryKey });
    },
  });
}
```

- [ ] **Step 2: Write `frontend-web/src/components/conversations/RenameDialog.tsx`**

```tsx
import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface RenameDialogProps {
  currentTitle: string;
  onRename: (title: string) => void;
  trigger: React.ReactNode;
}

export function RenameDialog({ currentTitle, onRename, trigger }: RenameDialogProps) {
  const [title, setTitle] = useState(currentTitle);
  const [open, setOpen] = useState(false);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Rename conversation</DialogTitle>
        </DialogHeader>
        <Input value={title} onChange={(event) => setTitle(event.target.value)} maxLength={200} />
        <DialogFooter>
          <Button
            onClick={() => {
              if (title.trim()) {
                onRename(title.trim());
                setOpen(false);
              }
            }}
          >
            Save title
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
```

- [ ] **Step 3: Write `frontend-web/src/components/conversations/ConversationItem.tsx`**

```tsx
import { MoreVertical, Trash2, Pencil } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { RenameDialog } from "./RenameDialog";
import { cn } from "@/lib/utils";
import type { ConversationSummary } from "@/types";

interface ConversationItemProps {
  conversation: ConversationSummary;
  active: boolean;
  onSelect: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
}

export function ConversationItem({
  conversation,
  active,
  onSelect,
  onRename,
  onDelete,
}: ConversationItemProps) {
  return (
    <div
      className={cn(
        "group flex items-center justify-between gap-2 rounded-md px-3 py-2 text-sm hover:bg-muted",
        active && "bg-muted font-medium"
      )}
    >
      <button
        type="button"
        onClick={onSelect}
        className="min-w-0 flex-1 truncate text-left"
        title={conversation.title}
      >
        {conversation.title || "Untitled conversation"}
      </button>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" className="h-7 w-7 opacity-0 group-hover:opacity-100">
            <MoreVertical className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <RenameDialog
            currentTitle={conversation.title}
            onRename={onRename}
            trigger={
              <DropdownMenuItem onSelect={(event) => event.preventDefault()}>
                <Pencil className="mr-2 h-4 w-4" /> Rename
              </DropdownMenuItem>
            }
          />
          <DropdownMenuItem onSelect={onDelete} className="text-destructive">
            <Trash2 className="mr-2 h-4 w-4" /> Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
```

- [ ] **Step 4: Write `frontend-web/src/components/conversations/ConversationList.tsx`**

```tsx
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";
import { ConversationItem } from "./ConversationItem";
import {
  useConversations,
  useDeleteConversation,
  useRenameConversation,
} from "@/hooks/useConversations";
import { useUiStore } from "@/store/uiStore";

interface ConversationListProps {
  onNewChat: () => void;
}

export function ConversationList({ onNewChat }: ConversationListProps) {
  const { data: conversations = [], isLoading } = useConversations();
  const activeConversationId = useUiStore((state) => state.activeConversationId);
  const setActiveConversationId = useUiStore((state) => state.setActiveConversationId);
  const renameMutation = useRenameConversation();
  const deleteMutation = useDeleteConversation();

  return (
    <div className="flex flex-col gap-2">
      <Button onClick={onNewChat} className="w-full justify-start gap-2">
        <Plus className="h-4 w-4" /> New chat
      </Button>

      {isLoading && <p className="text-xs text-muted-foreground">Loading conversations…</p>}
      {!isLoading && conversations.length === 0 && (
        <p className="text-xs text-muted-foreground">
          Start a new chat to create your first saved conversation.
        </p>
      )}

      <div className="flex flex-col gap-1">
        {conversations.map((conversation) => (
          <ConversationItem
            key={conversation.id}
            conversation={conversation}
            active={conversation.id === activeConversationId}
            onSelect={() => setActiveConversationId(conversation.id)}
            onRename={(title) => renameMutation.mutate({ id: conversation.id, title })}
            onDelete={() => {
              deleteMutation.mutate(conversation.id, {
                onSuccess: () => {
                  if (conversation.id === activeConversationId) {
                    const remaining = conversations.filter((item) => item.id !== conversation.id);
                    setActiveConversationId(remaining[0]?.id ?? crypto.randomUUID());
                  }
                },
              });
            }}
          />
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Wire conversation list into `frontend-web/src/App.tsx`, with a new-conversation bootstrap on first load**

```tsx
import { useEffect, useState } from "react";
import { AppShell } from "@/components/layout/AppShell";
import type { View } from "@/components/layout/TopTabs";
import { ConversationList } from "@/components/conversations/ConversationList";
import { useConversations } from "@/hooks/useConversations";
import { useUiStore } from "@/store/uiStore";

export default function App() {
  const [view, setView] = useState<View>("chat");
  const { data: conversations } = useConversations();
  const activeConversationId = useUiStore((state) => state.activeConversationId);
  const setActiveConversationId = useUiStore((state) => state.setActiveConversationId);

  useEffect(() => {
    if (activeConversationId || !conversations) return;
    setActiveConversationId(conversations[0]?.id ?? crypto.randomUUID());
  }, [conversations, activeConversationId, setActiveConversationId]);

  return (
    <AppShell
      sidebar={
        <ConversationList onNewChat={() => setActiveConversationId(crypto.randomUUID())} />
      }
      activeView={view}
      onViewChange={setView}
    >
      {view === "chat" ? (
        <div className="p-6 text-sm text-muted-foreground">
          Selected conversation: {activeConversationId ?? "none"}. Message list lands in Task 6.
        </div>
      ) : (
        <div className="p-6 text-sm text-muted-foreground">Evaluation view lands in Task 9.</div>
      )}
    </AppShell>
  );
}
```

- [ ] **Step 6: Verify manually against the running backend**

Run backend: `cd backend && uvicorn app.main:app --reload` (or however it's normally started; see `backend/README`/`docker-compose.yml` for the exact command).
Run frontend: `cd frontend-web && npm run dev`

Expected: sidebar lists existing conversations (if any exist in `backend/data/chat_history.sqlite`); clicking "New chat" generates a fresh UUID (visible in the placeholder text) without creating a premature sidebar entry; clicking a conversation row selects it (label updates); the overflow menu can rename and delete a conversation and the list updates.

- [ ] **Step 7: Commit**

```bash
git add frontend-web/src
git commit -m "Add frontend-web conversation list, switching, rename, delete"
```

---

## Task 6: Streaming Chat (`/api/chat/stream` NDJSON)

**Files:**
- Create: `frontend-web/src/hooks/useChatStream.ts`
- Create: `frontend-web/src/components/chat/ChatInput.tsx`
- Create: `frontend-web/src/components/chat/MessageBubble.tsx`
- Create: `frontend-web/src/components/chat/MessageList.tsx`
- Create: `frontend-web/src/pages/ChatPage.tsx`
- Modify: `frontend-web/src/App.tsx`

**Interfaces:**
- Consumes: `streamChat` (Task 3 `@/api/chat`); `useConversationHistory` and `conversationsQueryKey` (Task 5 `@/hooks/useConversations`); `useUiStore.apiKey` (Task 4).
- Produces: `useChatStream(onDone)` returning `{ isStreaming, draft, send, cancel }` where `draft: { toolQuery: string | null; sources: Source[]; text: string; notice: string | null; error: string | null }`. Task 7 extends `MessageBubble`'s rendering of `sources`/`toolQuery` with real `CitationCard`/`ToolStatusPill` components — it does not change this hook's shape.

- [ ] **Step 1: Write `frontend-web/src/hooks/useChatStream.ts`**

```ts
import { useCallback, useRef, useState } from "react";
import { streamChat } from "@/api/chat";
import type { ChatRequest, ChatStreamEvent, Source } from "@/types";

export interface StreamingDraft {
  toolQuery: string | null;
  sources: Source[];
  text: string;
  notice: string | null;
  error: string | null;
}

export interface ChatStreamResult {
  text: string;
  sources: Source[];
}

const EMPTY_DRAFT: StreamingDraft = {
  toolQuery: null,
  sources: [],
  text: "",
  notice: null,
  error: null,
};

export function useChatStream(onDone: (result: ChatStreamResult) => void) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [draft, setDraft] = useState<StreamingDraft>(EMPTY_DRAFT);
  const controllerRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setIsStreaming(false);
  }, []);

  const send = useCallback(
    (question: string, conversationId: string, apiKey?: string) => {
      const controller = new AbortController();
      controllerRef.current = controller;
      setDraft(EMPTY_DRAFT);
      setIsStreaming(true);

      const request: ChatRequest = {
        question,
        conversation_id: conversationId,
        openai_api_key: apiKey || undefined,
      };

      let text = "";
      let sources: Source[] = [];

      streamChat(
        request,
        (event: ChatStreamEvent) => {
          if (event.type === "tool_start") {
            setDraft((prev) => ({ ...prev, toolQuery: event.query }));
          } else if (event.type === "sources") {
            sources = event.data;
            setDraft((prev) => ({ ...prev, sources: event.data, toolQuery: null }));
          } else if (event.type === "token") {
            text += event.data;
            setDraft((prev) => ({ ...prev, text }));
          } else if (event.type === "notice") {
            setDraft((prev) => ({ ...prev, notice: event.data }));
          } else if (event.type === "error") {
            setDraft((prev) => ({ ...prev, error: event.data }));
          } else if (event.type === "done") {
            setIsStreaming(false);
            controllerRef.current = null;
            onDone({ text, sources });
          }
        },
        controller.signal
      ).catch((error: unknown) => {
        if (controller.signal.aborted) return;
        const message = error instanceof Error ? error.message : "Stream failed";
        setDraft((prev) => ({ ...prev, error: message }));
        setIsStreaming(false);
      });
    },
    [onDone]
  );

  return { isStreaming, draft, send, cancel };
}
```

- [ ] **Step 2: Write `frontend-web/src/components/chat/ChatInput.tsx`**

```tsx
import { useState, type KeyboardEvent } from "react";
import { Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface ChatInputProps {
  disabled: boolean;
  onSend: (question: string) => void;
}

export function ChatInput({ disabled, onSend }: ChatInputProps) {
  const [value, setValue] = useState("");

  const submit = () => {
    const question = value.trim();
    if (!question || disabled) return;
    onSend(question);
    setValue("");
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <div className="flex items-end gap-2 border-t border-border p-4">
      <Textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Enter your question…"
        disabled={disabled}
        rows={1}
        className="min-h-[44px] flex-1 resize-none"
      />
      <Button onClick={submit} disabled={disabled || !value.trim()} size="icon">
        <Send className="h-4 w-4" />
      </Button>
    </div>
  );
}
```

- [ ] **Step 3: Write `frontend-web/src/components/chat/MessageBubble.tsx` (plain rendering; Task 7 adds `ToolStatusPill`/`CitationCard`)**

```tsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import type { Source } from "@/types";

export interface MessageBubbleProps {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  toolQuery?: string | null;
  notice?: string | null;
  error?: string | null;
}

export function MessageBubble({ role, content, error }: MessageBubbleProps) {
  const isUser = role === "user";

  return (
    <div className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[75%] rounded-lg px-4 py-3 text-sm",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted text-foreground",
          error && "border border-destructive text-destructive"
        )}
      >
        <div className="prose prose-sm max-w-none dark:prose-invert">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content || " "}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
```

Install the markdown-typography plugin used above:

```bash
npm install -D @tailwindcss/typography
```

Add it to `frontend-web/tailwind.config.ts`'s `plugins` array: `plugins: [require("@tailwindcss/typography")]`.

- [ ] **Step 4: Write `frontend-web/src/components/chat/MessageList.tsx`**

```tsx
import { useEffect, useRef } from "react";
import { MessageBubble } from "./MessageBubble";
import type { StreamingDraft } from "@/hooks/useChatStream";
import type { ConversationMessage } from "@/types";

interface MessageListProps {
  messages: ConversationMessage[];
  isStreaming: boolean;
  draft: StreamingDraft;
}

export function MessageList({ messages, isStreaming, draft }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, draft.text]);

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-6">
      {messages.map((message, index) => (
        <MessageBubble
          key={index}
          role={message.role}
          content={message.content}
          sources={message.sources}
        />
      ))}
      {isStreaming && (
        <MessageBubble
          role="assistant"
          content={draft.text}
          sources={draft.sources}
          toolQuery={draft.toolQuery}
          notice={draft.notice}
          error={draft.error}
        />
      )}
      <div ref={bottomRef} />
    </div>
  );
}
```

- [ ] **Step 5: Write `frontend-web/src/pages/ChatPage.tsx`**

```tsx
import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { MessageList } from "@/components/chat/MessageList";
import { ChatInput } from "@/components/chat/ChatInput";
import { useChatStream } from "@/hooks/useChatStream";
import { conversationsQueryKey, useConversationHistory } from "@/hooks/useConversations";
import { useUiStore } from "@/store/uiStore";
import type { ConversationMessage } from "@/types";

export function ChatPage() {
  const activeConversationId = useUiStore((state) => state.activeConversationId);
  const apiKey = useUiStore((state) => state.apiKey);
  const queryClient = useQueryClient();
  const { data: history } = useConversationHistory(activeConversationId);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);

  useEffect(() => {
    setMessages(history?.messages ?? []);
  }, [activeConversationId, history]);

  const { isStreaming, draft, send, cancel } = useChatStream(({ text, sources }) => {
    setMessages((prev) => [...prev, { role: "assistant", content: text, sources }]);
    queryClient.invalidateQueries({ queryKey: conversationsQueryKey });
  });

  useEffect(() => {
    return () => cancel();
  }, [activeConversationId, cancel]);

  const handleSend = (question: string) => {
    if (!activeConversationId) return;
    setMessages((prev) => [...prev, { role: "user", content: question, sources: [] }]);
    send(question, activeConversationId, apiKey);
  };

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1">
        <MessageList messages={messages} isStreaming={isStreaming} draft={draft} />
      </div>
      <ChatInput disabled={isStreaming || !activeConversationId} onSend={handleSend} />
    </div>
  );
}
```

- [ ] **Step 6: Wire `ChatPage` into `frontend-web/src/App.tsx`, replacing the chat placeholder**

```tsx
// in App.tsx, replace the chat-view branch:
import { ChatPage } from "@/pages/ChatPage";
// ...
{view === "chat" ? <ChatPage /> : (
  <div className="p-6 text-sm text-muted-foreground">Evaluation view lands in Task 9.</div>
)}
```

- [ ] **Step 7: Verify manually against the running backend**

With backend running and an OpenAI key configured (either server-side `OPENAI_API_KEY` env var or typed into Settings once Task 8 lands — for this task's verification, set `OPENAI_API_KEY` in `backend/.env` so no UI key input is required yet):

1. Select or create a conversation, type a question, press Enter.
2. Expected: user bubble appears immediately; assistant bubble streams in token-by-token; after `done`, the assistant message is part of `messages` and the sidebar conversation list re-sorts/updates its title (confirms the `conversationsQueryKey` invalidation works).
3. Reload the page, reselect the same conversation: history is restored from the backend (confirms `useConversationHistory` works independent of local state).

- [ ] **Step 8: Commit**

```bash
git add frontend-web/src frontend-web/tailwind.config.ts frontend-web/package.json frontend-web/package-lock.json
git commit -m "Add frontend-web streaming chat via NDJSON"
```

---

## Task 7: Tool Status and Citation Cards

**Files:**
- Create: `frontend-web/src/components/chat/ToolStatusPill.tsx`
- Create: `frontend-web/src/components/chat/CitationCard.tsx`
- Modify: `frontend-web/src/components/chat/MessageBubble.tsx`

**Interfaces:**
- Consumes: `Source` type (Task 2); `MessageBubbleProps` shape already defined in Task 6 (`sources`, `toolQuery`, `notice`, `error` — this task only changes what the component renders, not its props).
- Produces: `ToolStatusPill({ query })`, `CitationCard({ source, index })` — used only inside `MessageBubble`; no other task consumes them directly.

- [ ] **Step 1: Write `frontend-web/src/components/chat/ToolStatusPill.tsx`**

```tsx
import { Search } from "lucide-react";

interface ToolStatusPillProps {
  query: string;
}

export function ToolStatusPill({ query }: ToolStatusPillProps) {
  return (
    <div className="mb-2 flex items-center gap-2 rounded-full bg-primary/10 px-3 py-1 text-xs text-primary">
      <Search className="h-3 w-3 animate-pulse" />
      <span>Searching uploaded documents: “{query}”</span>
    </div>
  );
}
```

- [ ] **Step 2: Write `frontend-web/src/components/chat/CitationCard.tsx`**

```tsx
import { ExternalLink } from "lucide-react";
import type { Source } from "@/types";

interface CitationCardProps {
  source: Source;
  index: number;
}

export function CitationCard({ source, index }: CitationCardProps) {
  const pageLabel =
    source.pages && source.pages.length > 1
      ? `${source.pages[0]}-${source.pages[source.pages.length - 1]}`
      : String(source.page);
  const excerpt = source.text.length > 400 ? `${source.text.slice(0, 400)}…` : source.text;

  return (
    <div className="rounded-md border border-border bg-card p-3 text-xs">
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="font-medium">
          Source {index}: {source.document} · Page {pageLabel} · Chunk {source.chunk}
        </span>
        {source.score !== null && (
          <span className="text-muted-foreground">score {source.score.toFixed(4)}</span>
        )}
      </div>
      <blockquote className="border-l-2 border-border pl-2 text-muted-foreground">{excerpt}</blockquote>
      {source.link && (
        <a
          href={source.link}
          target="_blank"
          rel="noreferrer"
          className="mt-1 flex items-center gap-1 text-primary hover:underline"
        >
          <ExternalLink className="h-3 w-3" /> {source.link}
        </a>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Update `frontend-web/src/components/chat/MessageBubble.tsx` to render tool status, citations, and notices**

Replace the file with:

```tsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import { ToolStatusPill } from "./ToolStatusPill";
import { CitationCard } from "./CitationCard";
import type { Source } from "@/types";

export interface MessageBubbleProps {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  toolQuery?: string | null;
  notice?: string | null;
  error?: string | null;
}

export function MessageBubble({ role, content, sources = [], toolQuery, notice, error }: MessageBubbleProps) {
  const isUser = role === "user";

  return (
    <div className={cn("flex w-full flex-col", isUser ? "items-end" : "items-start")}>
      {toolQuery && <ToolStatusPill query={toolQuery} />}
      <div
        className={cn(
          "max-w-[75%] rounded-lg px-4 py-3 text-sm",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted text-foreground",
          error && "border border-destructive text-destructive"
        )}
      >
        {notice && <p className="mb-2 text-xs text-amber-600 dark:text-amber-400">{notice}</p>}
        <div className="prose prose-sm max-w-none dark:prose-invert">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{error ? `❌ ${error}` : content || " "}</ReactMarkdown>
        </div>
      </div>
      {sources.length > 0 && (
        <details className="mt-2 w-full max-w-[75%]" open={false}>
          <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
            View sources ({sources.length})
          </summary>
          <div className="mt-2 flex flex-col gap-2">
            {sources.map((source, index) => (
              <CitationCard key={`${source.chunk}-${index}`} source={source} index={index + 1} />
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Verify manually**

Ask a question that requires document lookup. Expected: a "Searching uploaded documents: …" pill appears briefly before/while tokens stream in; once the answer finishes, a collapsible "View sources (N)" section appears below the bubble with one `CitationCard` per source, each showing document/page/chunk, an optional score, an excerpt, and an optional link.

- [ ] **Step 5: Commit**

```bash
git add frontend-web/src/components/chat
git commit -m "Add tool status pill and citation cards to chat messages"
```

---

## Task 8: PDF Upload and Document Management

**Files:**
- Create: `frontend-web/src/hooks/useDocuments.ts`
- Create: `frontend-web/src/components/documents/UploadDropzone.tsx`
- Create: `frontend-web/src/components/documents/DocumentList.tsx`
- Create: `frontend-web/src/components/documents/DocumentPanel.tsx`
- Modify: `frontend-web/src/App.tsx`

**Interfaces:**
- Consumes: `uploadPdfs`, `listDocuments`, `deleteDocument` (Task 3 `@/api/documents`).
- Produces: `useDocuments()`, `useUploadPdfs()`, `useDeleteDocument()`, and `DocumentPanel` (a self-contained sidebar section) — Task 9's `EvaluationPage` also calls `useDocuments()` for the indexed-document list, so its query key (`["documents"]`) must stay as defined here.

- [ ] **Step 1: Write `frontend-web/src/hooks/useDocuments.ts`**

```ts
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { deleteDocument, listDocuments, uploadPdfs } from "@/api/documents";

export const documentsQueryKey = ["documents"] as const;

export function useDocuments() {
  return useQuery({
    queryKey: documentsQueryKey,
    queryFn: listDocuments,
  });
}

export function useUploadPdfs() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (files: File[]) => uploadPdfs(files),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: documentsQueryKey });
    },
  });
}

export function useDeleteDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (fileHash: string) => deleteDocument(fileHash),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: documentsQueryKey });
    },
  });
}
```

- [ ] **Step 2: Write `frontend-web/src/components/documents/UploadDropzone.tsx`**

```tsx
import { useRef, useState, type DragEvent } from "react";
import { UploadCloud } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface UploadDropzoneProps {
  isUploading: boolean;
  onFilesSelected: (files: File[]) => void;
}

export function UploadDropzone({ isUploading, onFilesSelected }: UploadDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const files = Array.from(event.dataTransfer.files).filter((file) => file.type === "application/pdf");
    if (files.length) onFilesSelected(files);
  };

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={cn(
        "flex flex-col items-center gap-2 rounded-md border-2 border-dashed border-border p-4 text-center text-xs text-muted-foreground",
        isDragging && "border-primary bg-primary/5"
      )}
    >
      <UploadCloud className="h-6 w-6" />
      <p>Drag PDF files here, or</p>
      <Button
        type="button"
        variant="secondary"
        size="sm"
        disabled={isUploading}
        onClick={() => inputRef.current?.click()}
      >
        {isUploading ? "Uploading…" : "Choose files"}
      </Button>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        multiple
        className="hidden"
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          if (files.length) onFilesSelected(files);
          event.target.value = "";
        }}
      />
    </div>
  );
}
```

- [ ] **Step 3: Write `frontend-web/src/components/documents/DocumentList.tsx`**

```tsx
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { DocumentRecord } from "@/types";

interface DocumentListProps {
  documents: DocumentRecord[];
  onDelete: (fileHash: string) => void;
  deletingHash: string | null;
}

export function DocumentList({ documents, onDelete, deletingHash }: DocumentListProps) {
  if (documents.length === 0) {
    return <p className="text-xs text-muted-foreground">No documents indexed yet.</p>;
  }

  return (
    <div className="flex flex-col gap-1">
      {documents.map((document) => (
        <div
          key={document.file_hash}
          className="flex items-center justify-between gap-2 rounded-md px-2 py-1 text-xs hover:bg-muted"
        >
          <span className="min-w-0 truncate" title={document.document_name}>
            📄 {document.document_name} · Pages {document.pages} · Chunks {document.chunks}
          </span>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 shrink-0"
            disabled={deletingHash === document.file_hash}
            onClick={() => onDelete(document.file_hash)}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Confirm where shadcn generated the toast hook, then write `frontend-web/src/components/documents/DocumentPanel.tsx`**

The shadcn CLI's `toast` add (run in Task 1 Step 5) generates a `useToast` hook whose file path depends on the installed CLI version. Check which one exists before importing:

```bash
ls frontend-web/src/hooks/use-toast.ts frontend-web/src/components/ui/use-toast.ts 2>/dev/null
```

Use whichever path exists as the import source below (current shadcn CLI versions generate `@/hooks/use-toast`; older ones generated `@/components/ui/use-toast`).

```tsx
import { useState } from "react";
import { useToast } from "@/hooks/use-toast"; // adjust to @/components/ui/use-toast if that's the one that exists
import { UploadDropzone } from "./UploadDropzone";
import { DocumentList } from "./DocumentList";
import { useDocuments, useDeleteDocument, useUploadPdfs } from "@/hooks/useDocuments";

export function DocumentPanel() {
  const { data, isLoading } = useDocuments();
  const uploadMutation = useUploadPdfs();
  const deleteMutation = useDeleteDocument();
  const [deletingHash, setDeletingHash] = useState<string | null>(null);
  const { toast } = useToast();

  const handleUpload = (files: File[]) => {
    uploadMutation.mutate(files, {
      onSuccess: (result) => {
        result.messages.forEach((message) => toast({ description: message }));
      },
      onError: (error) => {
        toast({ description: `Indexing failed: ${error.message}`, variant: "destructive" });
      },
    });
  };

  const handleDelete = (fileHash: string) => {
    setDeletingHash(fileHash);
    deleteMutation.mutate(fileHash, {
      onSettled: () => setDeletingHash(null),
      onError: (error) => {
        toast({ description: `Delete failed: ${error.message}`, variant: "destructive" });
      },
    });
  };

  return (
    <div className="flex flex-col gap-3">
      <h2 className="text-sm font-semibold">📁 Document Management</h2>
      <UploadDropzone isUploading={uploadMutation.isPending} onFilesSelected={handleUpload} />

      {isLoading && <p className="text-xs text-muted-foreground">Loading documents…</p>}
      {data && (
        <>
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Total chunks: {data.total_chunks}</span>
            <span>Documents: {data.total_documents}</span>
          </div>
          <DocumentList documents={data.documents} onDelete={handleDelete} deletingHash={deletingHash} />
        </>
      )}
    </div>
  );
}
```

`useToast` is generated by shadcn's `toast` component from Task 1 (`npx shadcn add toast`), which also requires mounting `<Toaster />` once — add it in `frontend-web/src/App.tsx`'s top-level return, alongside `AppShell`, e.g. `<>{/* AppShell */}<Toaster /></>`.

- [ ] **Step 5: Add `DocumentPanel` (and a `Toaster` mount) to the sidebar in `frontend-web/src/App.tsx`**

```tsx
// in App.tsx
import { DocumentPanel } from "@/components/documents/DocumentPanel";
import { Toaster } from "@/components/ui/toaster";
import { Separator } from "@/components/ui/separator";
// ...
return (
  <>
    <AppShell
      sidebar={
        <div className="flex flex-col gap-4">
          <ConversationList onNewChat={() => setActiveConversationId(crypto.randomUUID())} />
          <Separator />
          <DocumentPanel />
        </div>
      }
      activeView={view}
      onViewChange={setView}
    >
      {view === "chat" ? <ChatPage /> : (
        <div className="p-6 text-sm text-muted-foreground">Evaluation view lands in Task 9.</div>
      )}
    </AppShell>
    <Toaster />
  </>
);
```

- [ ] **Step 6: Verify manually against the running backend**

1. Drag a PDF onto the dropzone (or use "Choose files"). Expected: a toast per file with the same status text the backend returns (`✅ … Indexed N text chunks`, `📄 … Already indexed`, or `⚠️ … Failed to extract text content`), and the document list below updates without a manual refresh.
2. Click delete on a listed document. Expected: it disappears from the list and stats update.

- [ ] **Step 7: Commit**

```bash
git add frontend-web/src
git commit -m "Add frontend-web PDF upload and document management"
```

---

## Task 9: Evaluation Page

**Files:**
- Create: `frontend-web/src/hooks/useEvaluation.ts`
- Create: `frontend-web/src/components/evaluation/EvaluationResultCard.tsx`
- Create: `frontend-web/src/pages/EvaluationPage.tsx`
- Modify: `frontend-web/src/App.tsx`

**Interfaces:**
- Consumes: `runEvaluation` (Task 3 `@/api/evaluation`); `useDocuments` (Task 8 `@/hooks/useDocuments`); `CitationCard` (Task 7 `@/components/chat/CitationCard`).
- Produces: `useRunEvaluation()`, `EvaluationPage` — mounted as the `"evaluation"` branch of `App.tsx`'s view switch; no other task consumes these.

- [ ] **Step 1: Write `frontend-web/src/hooks/useEvaluation.ts`**

```ts
import { useMutation } from "@tanstack/react-query";
import { runEvaluation } from "@/api/evaluation";

export function useRunEvaluation() {
  return useMutation({
    mutationFn: runEvaluation,
  });
}
```

- [ ] **Step 2: Write `frontend-web/src/components/evaluation/EvaluationResultCard.tsx`**

```tsx
import { CitationCard } from "@/components/chat/CitationCard";
import type { EvaluationResult } from "@/types";

interface EvaluationResultCardProps {
  result: EvaluationResult;
}

export function EvaluationResultCard({ result }: EvaluationResultCardProps) {
  return (
    <details className="rounded-md border border-border p-3 text-sm">
      <summary className="cursor-pointer font-medium">
        {result.id} · Final {result.final_score.toFixed(2)} · Document{" "}
        {result.document_hit_score.toFixed(2)} · Page {result.page_hit_score.toFixed(2)} · Keywords{" "}
        {result.keyword_score.toFixed(2)}
      </summary>
      <div className="mt-3 flex flex-col gap-2 text-xs">
        <p>
          <span className="font-medium">Question:</span> {result.question}
        </p>
        <p>
          <span className="font-medium">Expected Answer:</span> {result.expected_answer}
        </p>
        <p>
          <span className="font-medium">Expected Source:</span> {result.expected_document} page{" "}
          {result.expected_page}
        </p>
        <p>
          <span className="font-medium">Expected Keywords:</span> {result.expected_keywords.join(", ")}
        </p>
        <div className="grid grid-cols-4 gap-2 rounded-md bg-muted p-2 text-center">
          <div>
            <div className="font-semibold">{result.document_hit_score.toFixed(2)}</div>
            <div className="text-muted-foreground">Document Hit</div>
          </div>
          <div>
            <div className="font-semibold">{result.page_hit_score.toFixed(2)}</div>
            <div className="text-muted-foreground">Page Hit</div>
          </div>
          <div>
            <div className="font-semibold">{result.keyword_score.toFixed(2)}</div>
            <div className="text-muted-foreground">Keyword Score</div>
          </div>
          <div>
            <div className="font-semibold">{result.final_score.toFixed(2)}</div>
            <div className="text-muted-foreground">Final Score</div>
          </div>
        </div>
        {result.retrieved_sources.length > 0 ? (
          <div className="flex flex-col gap-2">
            {result.retrieved_sources.map((source, index) => (
              <CitationCard key={`${source.chunk}-${index}`} source={source} index={index + 1} />
            ))}
          </div>
        ) : (
          <p className="text-amber-600 dark:text-amber-400">No sources retrieved.</p>
        )}
      </div>
    </details>
  );
}
```

- [ ] **Step 3: Write `frontend-web/src/pages/EvaluationPage.tsx`**

```tsx
import { Button } from "@/components/ui/button";
import { EvaluationResultCard } from "@/components/evaluation/EvaluationResultCard";
import { useRunEvaluation } from "@/hooks/useEvaluation";
import { useDocuments } from "@/hooks/useDocuments";

const SAMPLE_DOCUMENTS = [
  "employee_handbook_en.pdf",
  "product_manual_en.pdf",
  "service_agreement_en.pdf",
];

export function EvaluationPage() {
  const { data: documents } = useDocuments();
  const evaluationMutation = useRunEvaluation();
  const result = evaluationMutation.data;

  const indexedNames = (documents?.documents ?? []).map((doc) => doc.document_name);
  const requiredDocuments = result?.required_documents ?? SAMPLE_DOCUMENTS;
  const missingDocuments =
    result?.missing_documents ?? requiredDocuments.filter((name) => !indexedNames.includes(name));

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-6">
      <div>
        <h2 className="text-base font-semibold">Evaluation Panel</h2>
        <p className="text-xs text-muted-foreground">
          Evaluation uses 5 predefined questions for the sample PDFs.
        </p>
      </div>

      <div className="text-xs">
        <p className="font-medium">Required sample documents:</p>
        {requiredDocuments.map((name) => (
          <p key={name} className="font-mono text-muted-foreground">
            sample_docs/{name}
          </p>
        ))}
      </div>

      {missingDocuments.length > 0 && (
        <div className="rounded-md border border-amber-400 bg-amber-50 p-3 text-xs text-amber-800 dark:bg-amber-950 dark:text-amber-300">
          Some required sample documents are missing. Please upload them before running evaluation.
          Scores may be low.
          <ul className="mt-1 list-disc pl-4">
            {missingDocuments.map((name) => (
              <li key={name} className="font-mono">
                sample_docs/{name}
              </li>
            ))}
          </ul>
        </div>
      )}

      <Button
        onClick={() => evaluationMutation.mutate()}
        disabled={evaluationMutation.isPending}
        className="w-fit"
      >
        {evaluationMutation.isPending ? "Running evaluation…" : "Run Evaluation"}
      </Button>

      {evaluationMutation.isError && (
        <p className="text-sm text-destructive">
          Evaluation failed: {(evaluationMutation.error as Error).message}
        </p>
      )}

      {!result && !evaluationMutation.isPending && (
        <p className="text-sm text-muted-foreground">Run evaluation to see retrieval scores.</p>
      )}

      {result && (
        <>
          <div className="flex gap-6 text-sm">
            <div>
              <div className="text-lg font-semibold">{result.tests_run}</div>
              <div className="text-muted-foreground">Tests Run</div>
            </div>
            <div>
              <div className="text-lg font-semibold">{result.average_final_score.toFixed(2)}</div>
              <div className="text-muted-foreground">Average Final Score</div>
            </div>
          </div>
          <div className="flex flex-col gap-2">
            {result.results.map((item) => (
              <EvaluationResultCard key={item.id} result={item} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Wire `EvaluationPage` into `frontend-web/src/App.tsx`, replacing the evaluation placeholder**

```tsx
// in App.tsx
import { EvaluationPage } from "@/pages/EvaluationPage";
// ...
{view === "chat" ? <ChatPage /> : <EvaluationPage />}
```

- [ ] **Step 5: Verify manually against the running backend**

Switch to the Evaluation tab. Expected: required sample-document list shown; a missing-documents warning if `sample_docs/*_en.pdf` aren't uploaded; clicking "Run Evaluation" shows a loading state, then summary metrics and one expandable card per test question with the four scores and retrieved-source citation cards (reusing `CitationCard` from Task 7).

- [ ] **Step 6: Commit**

```bash
git add frontend-web/src
git commit -m "Add frontend-web evaluation page"
```

---

## Task 10: Responsive Layout, Dark Mode, Error States

**Files:**
- Create: `frontend-web/src/hooks/useTheme.ts`
- Create: `frontend-web/src/components/settings/SettingsPanel.tsx`
- Create: `frontend-web/src/components/settings/ThemeToggle.tsx`
- Modify: `frontend-web/src/App.tsx`
- Modify: `frontend-web/src/components/layout/Sidebar.tsx`
- Modify: `frontend-web/src/components/layout/AppShell.tsx`
- Modify: `frontend-web/src/main.tsx`

**Interfaces:**
- Consumes: `useUiStore` (Task 4, `theme`/`setTheme`/`apiKey`/`setApiKey`/`sidebarOpen`/`setSidebarOpen`).
- Produces: `useTheme()` returning `{ theme, setTheme, resolvedTheme }`; `SettingsPanel` (API key input + `ThemeToggle`), mounted once in the sidebar — no other task depends on these.

- [ ] **Step 1: Write `frontend-web/src/hooks/useTheme.ts`**

```ts
import { useEffect, useState } from "react";
import { useUiStore, type Theme } from "@/store/uiStore";

function systemPrefersDark(): boolean {
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export function useTheme() {
  const theme = useUiStore((state) => state.theme);
  const setTheme = useUiStore((state) => state.setTheme);
  const [resolvedTheme, setResolvedTheme] = useState<"light" | "dark">(
    theme === "system" ? (systemPrefersDark() ? "dark" : "light") : theme
  );

  useEffect(() => {
    const resolve = () => (theme === "system" ? (systemPrefersDark() ? "dark" : "light") : theme);
    setResolvedTheme(resolve());

    if (theme !== "system") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const listener = () => setResolvedTheme(resolve());
    media.addEventListener("change", listener);
    return () => media.removeEventListener("change", listener);
  }, [theme]);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", resolvedTheme === "dark");
  }, [resolvedTheme]);

  return { theme, setTheme, resolvedTheme } as { theme: Theme; setTheme: (t: Theme) => void; resolvedTheme: "light" | "dark" };
}
```

- [ ] **Step 2: Write `frontend-web/src/components/settings/ThemeToggle.tsx`**

```tsx
import { Moon, Sun, Monitor } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";
import type { Theme } from "@/store/uiStore";

const OPTIONS: { value: Theme; icon: typeof Sun; label: string }[] = [
  { value: "light", icon: Sun, label: "Light" },
  { value: "dark", icon: Moon, label: "Dark" },
  { value: "system", icon: Monitor, label: "System" },
];

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  return (
    <div className="flex gap-1 rounded-md border border-border p-1">
      {OPTIONS.map(({ value, icon: Icon, label }) => (
        <Button
          key={value}
          type="button"
          variant="ghost"
          size="sm"
          className={cn("h-7 flex-1 gap-1 px-2", theme === value && "bg-muted")}
          onClick={() => setTheme(value)}
          title={label}
        >
          <Icon className="h-3.5 w-3.5" />
        </Button>
      ))}
    </div>
  );
}
```

- [ ] **Step 3: Write `frontend-web/src/components/settings/SettingsPanel.tsx`**

```tsx
import { Input } from "@/components/ui/input";
import { ThemeToggle } from "./ThemeToggle";
import { useUiStore } from "@/store/uiStore";
import { getApiBaseUrl } from "@/api/client";

export function SettingsPanel() {
  const apiKey = useUiStore((state) => state.apiKey);
  const setApiKey = useUiStore((state) => state.setApiKey);

  return (
    <div className="flex flex-col gap-3">
      <h2 className="text-sm font-semibold">⚙️ Settings</h2>
      <p className="text-xs text-muted-foreground">
        Backend: <code>{getApiBaseUrl()}</code>
      </p>
      <div className="flex flex-col gap-1">
        <label className="text-xs font-medium">OpenAI API Key (optional)</label>
        <Input
          type="password"
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value.trim())}
          placeholder="sk-…"
        />
        <p className="text-xs text-muted-foreground">
          Used only for this session. If empty, the app falls back to the backend's configured key or
          extractive answers.
        </p>
      </div>
      <div className="flex flex-col gap-1">
        <label className="text-xs font-medium">Theme</label>
        <ThemeToggle />
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Apply the resolved theme on startup in `frontend-web/src/main.tsx`, before React mounts, to avoid a flash of the wrong theme**

Add at the top of `main.tsx`, before `createRoot(...)`:

```ts
const storedTheme = window.localStorage.getItem("rag-chat-theme");
const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
const shouldUseDark = storedTheme === "dark" || (storedTheme !== "light" && prefersDark);
document.documentElement.classList.toggle("dark", shouldUseDark);
```

- [ ] **Step 5: Mount `SettingsPanel` in the sidebar and call `useTheme()` once at the app root, in `frontend-web/src/App.tsx`**

```tsx
// in App.tsx
import { SettingsPanel } from "@/components/settings/SettingsPanel";
import { useTheme } from "@/hooks/useTheme";
import { Separator } from "@/components/ui/separator";
// inside App():
useTheme(); // keeps <html class="dark"> in sync with store changes after mount
// ... inside the sidebar prop:
<div className="flex flex-col gap-4">
  <ConversationList onNewChat={() => setActiveConversationId(crypto.randomUUID())} />
  <Separator />
  <DocumentPanel />
  <Separator />
  <SettingsPanel />
</div>
```

- [ ] **Step 6: Make the sidebar responsive — close it by default on small screens and add a scrim overlay**

Modify `frontend-web/src/components/layout/Sidebar.tsx` to accept an `onClose` for the mobile scrim:

```tsx
import type { ReactNode } from "react";
import { useUiStore } from "@/store/uiStore";
import { cn } from "@/lib/utils";

interface SidebarProps {
  children: ReactNode;
}

export function Sidebar({ children }: SidebarProps) {
  const sidebarOpen = useUiStore((state) => state.sidebarOpen);
  const setSidebarOpen = useUiStore((state) => state.setSidebarOpen);

  return (
    <>
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex h-full w-80 shrink-0 flex-col gap-6 overflow-y-auto border-r border-border bg-card p-4 transition-transform md:static md:z-auto md:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        {children}
      </aside>
    </>
  );
}
```

Modify `frontend-web/src/components/layout/AppShell.tsx`'s default `sidebarOpen` behavior by initializing it based on viewport width — update `frontend-web/src/store/uiStore.ts`'s initial value:

```ts
// in uiStore.ts, replace: sidebarOpen: true,
sidebarOpen: window.matchMedia("(min-width: 768px)").matches,
```

- [ ] **Step 7: Verify manually**

1. Resize the browser below 768px width: sidebar should hide by default, the header's menu button should toggle it as an overlay with a dismissible scrim; above 768px it should sit inline and always be visible (no scrim).
2. Toggle theme to Dark: background/text colors flip immediately across sidebar, chat bubbles, cards. Reload the page: dark mode persists (no flash of light theme first). Set to System and toggle the OS theme (or devtools emulation): the app follows it live.
3. Trigger an error path (e.g., stop the backend and try sending a message): the assistant bubble shows the error state styling from `MessageBubble`, and document/evaluation actions show destructive-styled toasts instead of failing silently.

- [ ] **Step 8: Commit**

```bash
git add frontend-web/src
git commit -m "Add frontend-web responsive layout, dark mode, and error-state polish"
```

---

## Task 11: Feature Parity Verification Against Streamlit

**Files:**
- Create: `docs/superpowers/plans/2026-08-18-react-frontend-parity-checklist.md` (the completed checklist, kept as a record of this verification pass)

**Interfaces:**
- Consumes: both running frontends (`frontend/` on port 8501, `frontend-web/` dev server on port 5173) against the same backend and the same `backend/data/chat_history.sqlite` / Milvus data.
- Produces: a filled-in checklist file documenting pass/fail per item — this is the deliverable Task 12's Docker changes assume is complete before switching `frontend-web` into the compose stack for real use.

- [ ] **Step 1: Start the shared backend once**

```bash
cd backend && uvicorn app.main:app --reload --port 8000
```

- [ ] **Step 2: Start both frontends against the same backend**

```bash
cd frontend && streamlit run app.py   # http://localhost:8501
cd frontend-web && npm run dev        # http://localhost:5173
```

- [ ] **Step 3: Create the checklist file and work through it side by side, one row at a time, marking `✅`/`❌` with a one-line note for any failure**

Create `docs/superpowers/plans/2026-08-18-react-frontend-parity-checklist.md`:

```markdown
# frontend-web ↔ Streamlit Parity Checklist

Run both apps against the same backend/data. For each row, perform the action in both UIs and confirm equivalent behavior.

- [ ] New chat creates a fresh UUID thread; no sidebar entry appears until the first successful turn.
- [ ] Conversation list: select switches the main panel; rename updates the title in place; delete removes it and falls back to the next conversation (or a fresh chat if none remain).
- [ ] Reloading the page restores the previously selected conversation's history.
- [ ] Uploading multiple PDFs at once shows one status line per file (indexed / already indexed / unreadable), and a re-upload of the same file is reported as already indexed.
- [ ] Document stats (total chunks/documents) and the per-document list (name, pages, chunks) match between both UIs after the same uploads.
- [ ] Deleting a document removes it from both the list and the stats.
- [ ] Typing an OpenAI API key in Settings is used for the next request; clearing it falls back to the backend's configured key (or extractive fallback if neither exists) — confirm via the `notice` banner in the no-key case.
- [ ] Sending a question that needs document lookup shows a searching indicator, then streams the answer, then shows citation cards matching the sources Streamlit displays for the same question.
- [ ] Sending small talk / arithmetic does not trigger a tool call in either UI (no citation cards, no searching indicator).
- [ ] Sending an empty question is rejected client-side (or shows the backend's 400) in both UIs.
- [ ] Asking a multi-part question that forces several tool calls in a row eventually hits the recursion limit (or completes within it) with the same clear limit message/behavior in both UIs — no infinite spinner in either.
- [ ] Loading `/conversations/{id}/messages` for a conversation that was just deleted (e.g. delete it in one tab, still-open in another) surfaces a clear "not found" state in both UIs rather than a blank screen or unhandled crash.
- [ ] Evaluation tab: required/missing sample-document lists match; running evaluation produces the same `average_final_score` and per-question scores (allow for LLM non-determinism in retrieval-independent fields only — retrieval scores must match exactly since retrieval is deterministic).
- [ ] Stopping the backend mid-session and sending a question shows a clear error in both UIs instead of an unhandled crash.

## Result

Summarize any `❌` rows here with what's different and whether it blocks deleting `frontend/`.
```

Work through every row against both running apps, checking boxes and filling in the `## Result` section truthfully — if the React app doesn't match on some row, note it in the `## Result` section rather than checking the box; do not mark parity that wasn't actually observed.

- [ ] **Step 4: Fix any parity gaps found**

For each unchecked row, identify which `frontend-web` component/hook from Tasks 5–10 is responsible, fix it, re-verify that specific row, and check it off.

- [ ] **Step 5: Commit the completed checklist**

```bash
git add docs/superpowers/plans/2026-08-18-react-frontend-parity-checklist.md
git commit -m "Complete frontend-web vs Streamlit parity verification"
```

---

## Task 12: Docker Compose — Add `frontend-web` Alongside `frontend`

**Files:**
- Create: `frontend-web/Dockerfile`
- Create: `frontend-web/nginx.conf`
- Create: `frontend-web/.dockerignore`
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: `frontend-web/dist` (the Vite build output from Task 1's `npm run build`), `VITE_API_BASE_URL` (baked in at image build time via Docker build args).
- Produces: a `frontend-web` service in `docker-compose.yml` reachable on its own host port, running alongside the untouched `frontend`/`backend` services. Nothing here removes or renames the existing `frontend` service — that happens later, by the user, after Task 11's checklist is fully green.

- [ ] **Step 1: Write `frontend-web/.dockerignore`**

```text
node_modules
dist
.env
.env.local
```

- [ ] **Step 2: Write `frontend-web/nginx.conf`**

```nginx
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

- [ ] **Step 3: Write `frontend-web/Dockerfile`**

```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_BASE_URL=http://127.0.0.1:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

- [ ] **Step 4: Add the `frontend-web` service to `docker-compose.yml`**

Insert this new service after the existing `frontend` service in `docker-compose.yml` (the `frontend` service block itself is unchanged):

```yaml
  frontend-web:
    build:
      context: ./frontend-web
      dockerfile: Dockerfile
      args:
        VITE_API_BASE_URL: http://localhost:8000
    ports:
      - "5173:80"
    depends_on:
      backend:
        condition: service_healthy
    restart: unless-stopped
```

`VITE_API_BASE_URL` is set to `http://localhost:8000` (not `http://backend:8000`) because it's baked into the *browser-served* JS bundle at build time — the browser, not the container, makes the API calls, so it must resolve against the host machine's exposed backend port, same as how Streamlit's `frontend` service can reach the backend container-to-container (`http://backend:8000`) but a browser hitting the React app cannot.

- [ ] **Step 5: Verify the container builds and serves the app**

```bash
docker compose build frontend-web
docker compose up -d backend frontend-web
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5173
```

Expected: build succeeds; `curl` prints `200`; visiting `http://localhost:5173` in a browser shows the same working app as `npm run dev`, now able to reach the backend at `http://localhost:8000`.

```bash
docker compose down
```

- [ ] **Step 6: Commit**

```bash
git add frontend-web/Dockerfile frontend-web/nginx.conf frontend-web/.dockerignore docker-compose.yml
git commit -m "Add frontend-web Docker build and compose service alongside Streamlit"
```

---

## Explicit Non-Goals (carried from the spec)

- No changes to `backend/`, its API contracts, or its docs.
- No removal of `frontend/` — deletion happens later, by the user, after Task 11 is fully green.
- No authentication/user accounts.
- No automated frontend test suite in this pass.