# LangChain Agent Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a LangChain tool-calling agent that replaces the direct retrieval+generation call in `POST /api/chat`, with one tool (`search_uploaded_docs`) wrapping the existing Milvus-backed `vector_store.py` — per `docs/superpowers/specs/2026-08-06-langchain-agent-layer-design.md`.

**Architecture:** `backend/app/agent/tools.py` holds the tool's core logic and a per-request factory (`make_search_tool`) that binds the real `user_context` via closure so it's never LLM-fillable — only `query` is exposed to the model. `backend/app/agent/chat_agent.py` builds a `ChatOpenAI` bound to that one tool and runs a **bounded loop** (max 3 iterations): ask the model, and if it requests tool call(s), execute them, feed results back, and ask again — allowing genuine multi-step tool use (e.g., search once, then search again with a refined query) rather than a single fixed round trip. Every step (question received, each tool call + its args, each tool result's status, final answer) is printed to stdout as a server-side trace, mirroring `ai_orchestrator.py`'s existing trace style. The tool's citations are mapped onto the existing `Source` schema. `backend/app/api/chat.py`'s `POST /api/chat` calls this instead of `search_sources()`/`answer_question()`; `POST /api/chat/stream` is untouched.

**Tech Stack:** `langchain==1.3.14`, `langchain-core==1.5.3`, `langchain-openai==1.4.1` (exact versions verified installed and API-checked against this session — `@tool`-decorated closures, `ChatOpenAI.bind_tools`, `AIMessage.tool_calls`, `ToolMessage` all confirmed working as described below before writing this plan).

## Global Constraints

- Only realtime PDF (`vector_store.py`) is in scope. Offline docs, numeric data, and `/api/chat/stream` are untouched.
- `user_context` (`user_id`/`department_id`/`session_id`) is accepted by the tool's core function but **never exposed to the LLM's tool schema** — it's injected server-side via closure. Confirmed via live introspection this session: exposing it as an LLM-fillable field means the model would have to guess/hallucinate the values, which is wrong even though nothing uses them yet.
- `user_context` values are **not used to filter search results** — search still queries the single shared `realtime_pdf_collection`, per the earlier storage-layer decision.
- No frontend changes — `ChatResponse(answer, sources)` shape is unchanged; `chat.py`'s request/response contract is identical from the frontend's point of view.
- **Before running Task 3's test:** `00_RAG_Document_Chat/.env` needs a valid `OPENAI_API_KEY` (Task 2's test does not need one — it calls the tool function directly, never the LLM). Milvus must be reachable (same as the earlier DB-restructure plan's Global Constraints).

---

## File Structure

```
00_RAG_Document_Chat/
├── backend/
│   ├── requirements.txt                        # MODIFY: +langchain +langchain-core +langchain-openai
│   └── app/
│       ├── agent/                               # NEW package
│       │   ├── __init__.py
│       │   ├── tools.py                         # NEW: UserContext, core search fn, per-request tool factory
│       │   └── chat_agent.py                    # NEW: run_agent_chat()
│       └── api/
│           └── chat.py                          # MODIFY: POST /api/chat -> run_agent_chat(); /stream untouched
└── scripts/
    ├── test_search_uploaded_docs_tool.py        # NEW
    └── test_chat_agent.py                       # NEW
```

---

### Task 1: Add LangChain dependencies

**Files:**
- Modify: `backend/requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: `langchain`, `langchain-core`, `langchain-openai` importable from `backend/.venv`.

- [ ] **Step 1: Append to `backend/requirements.txt`**

```
langchain==1.3.14
langchain-core==1.5.3
langchain-openai==1.4.1
```

- [ ] **Step 2: Install**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 -m pip install -r backend/requirements.txt
```
Expected: succeeds (already verified installed in this environment during design).

- [ ] **Step 3: Verify**

```bash
backend/.venv/bin/python3 -c "
import langchain, langchain_core, langchain_openai
print(langchain.__version__, langchain_core.__version__, langchain_openai.__version__)
"
```
Expected: `1.3.14 1.5.3 1.4.1`.

- [ ] **Step 4: Commit**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
git add backend/requirements.txt
git commit -m "Add LangChain dependencies for the agent layer"
```

---

### Task 2: Build `agent/tools.py` (the `search_uploaded_docs` tool)

**Files:**
- Create: `backend/app/agent/__init__.py`
- Create: `backend/app/agent/tools.py`
- Test: `scripts/test_search_uploaded_docs_tool.py` (new)

**Interfaces:**
- Consumes: `app.rag.vector_store.query_chunks` (existing, unchanged — already embeds the query internally).
- Produces: `UserContext` (pydantic: `user_id`, `department_id`, `session_id`, all `str | None = None`), `make_search_tool(user_context: UserContext, top_k: int = 5) -> BaseTool` — the LLM-bindable tool whose schema exposes **only** `query: str`. Invoking it returns the fixed JSON shape: `{"status", "content", "citations": [{"source_id", "source_name", "page", "chunk_id", "text"}], "metadata": {"tool", "result_count"}, "error"}`.

- [ ] **Step 1: Write the failing test — `scripts/test_search_uploaded_docs_tool.py`**

```python
#!/usr/bin/env python
"""Test agent/tools.py: search_uploaded_docs tool in isolation, no LLM involved."""
from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_TEST_COLLECTION = "test_agent_tools_collection"
os.environ["REALTIME_PDF_COLLECTION_NAME"] = _TEST_COLLECTION

from app.rag import embeddings as embedding_module


def _test_embed(texts: list[str], dimensions: int = 384) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dimensions
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


embedding_module.embed_texts = _test_embed
embedding_module.embed_query = lambda q: _test_embed([q])[0]

from app.rag.types import Chunk
from app.rag.vector_store import add_chunks, get_collection
from app.agent.tools import UserContext, _search_uploaded_docs_impl, make_search_tool


def test_search_uploaded_docs_tool() -> bool:
    print("=" * 70)
    print("SEARCH_UPLOADED_DOCS TOOL TEST")
    print("=" * 70)

    chunks = [
        Chunk(
            id="hash1:p1:c1",
            text="The office WiFi password is SkyBlue42.",
            metadata={"document_name": "office_handbook.pdf", "file_hash": "hash1", "page": 3, "chunk_index": 1},
        ),
    ]

    try:
        print("\n[1/3] Seeding one chunk...")
        add_chunks(chunks)
        print("   OK")

        print("\n[2/3] Invoking the LLM-bindable tool (only `query` in its schema)...")
        user_context = UserContext(user_id="u1", department_id=None, session_id="s1")
        search_tool = make_search_tool(user_context, top_k=3)
        assert list(search_tool.args.keys()) == ["query"], f"expected only 'query' exposed, got {search_tool.args}"

        result = search_tool.invoke({"query": "What is the WiFi password?"})
        assert result["status"] == "success", f"expected success, got {result}"
        assert "SkyBlue42" in result["content"], f"expected password in content: {result}"
        assert len(result["citations"]) > 0, f"expected citations: {result}"
        citation = result["citations"][0]
        for key in ("source_id", "source_name", "page", "chunk_id", "text"):
            assert key in citation, f"citation missing '{key}': {citation}"
        assert citation["source_name"] == "office_handbook.pdf"
        assert citation["source_id"] == "hash1"
        assert result["metadata"] == {"tool": "search_uploaded_docs", "result_count": len(result["citations"])}
        assert result["error"] is None
        print(f"   OK: {result}")

        print("\n[3/3] Error path: empty query...")
        error_result = _search_uploaded_docs_impl("", user_context, top_k=3)
        assert error_result["status"] == "error", f"expected error status: {error_result}"
        assert error_result["error"], f"expected a non-null error message: {error_result}"
        assert error_result["content"] == ""
        assert error_result["citations"] == []
        print(f"   OK: {error_result}")

        print("\nPASSED: search_uploaded_docs tool works correctly, in isolation.")
        return True

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return False

    finally:
        client = get_collection()
        if client.has_collection(_TEST_COLLECTION):
            client.drop_collection(_TEST_COLLECTION)
            print(f"\n[cleanup] dropped {_TEST_COLLECTION}")


if __name__ == "__main__":
    success = test_search_uploaded_docs_tool()
    sys.exit(0 if success else 1)
```

- [ ] **Step 2: Run it, confirm it fails**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_search_uploaded_docs_tool.py
```
Expected: `ModuleNotFoundError: No module named 'app.agent'`.

- [ ] **Step 3: Create `backend/app/agent/__init__.py`** (empty)

- [ ] **Step 4: Create `backend/app/agent/tools.py`**

```python
from __future__ import annotations

from pydantic import BaseModel

from langchain_core.tools import BaseTool, tool

from app.rag.vector_store import query_chunks


class UserContext(BaseModel):
    user_id: str | None = None
    department_id: str | None = None
    session_id: str | None = None


def _search_uploaded_docs_impl(query: str, user_context: UserContext, top_k: int = 5) -> dict:
    """Core search logic. user_context is accepted for a future permission/scoping
    layer but is NOT used to filter results yet — search always queries the single
    shared realtime_pdf_collection, per the current storage-layer design."""
    try:
        results = query_chunks(query, top_k=top_k)
    except Exception as e:
        return {
            "status": "error",
            "content": "",
            "citations": [],
            "metadata": {"tool": "search_uploaded_docs", "result_count": 0},
            "error": str(e),
        }

    citations = []
    content_parts = []
    for hit in results:
        metadata = hit.get("metadata", {})
        file_hash = metadata.get("file_hash")
        page = metadata.get("page")
        chunk_index = metadata.get("chunk_index")
        text = hit.get("text", "")

        citations.append(
            {
                "source_id": file_hash,
                "source_name": metadata.get("document_name"),
                "page": page,
                "chunk_id": f"{file_hash}:p{page}:c{chunk_index}",
                "text": text,
            }
        )
        content_parts.append(text)

    return {
        "status": "success",
        "content": "\n\n---\n\n".join(content_parts),
        "citations": citations,
        "metadata": {"tool": "search_uploaded_docs", "result_count": len(citations)},
        "error": None,
    }


def make_search_tool(user_context: UserContext, top_k: int = 5) -> BaseTool:
    """Build a search_uploaded_docs tool bound to this request's real user_context.

    Only `query` is exposed in the tool's LLM-visible schema — user_context is
    captured via closure, never something the model fills in itself.
    """

    @tool
    def search_uploaded_docs(query: str) -> dict:
        """Search the user's uploaded PDF documents for passages relevant to the query.
        Use this whenever the question could be answered from documents the user has uploaded."""
        return _search_uploaded_docs_impl(query, user_context, top_k=top_k)

    return search_uploaded_docs
```

- [ ] **Step 5: Run the test, confirm it passes**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_search_uploaded_docs_tool.py
```
Expected: `PASSED: search_uploaded_docs tool works correctly, in isolation.`, exit 0. Requires Milvus
reachable; does **not** require `OPENAI_API_KEY` (no LLM involved in this test).

- [ ] **Step 6: Commit**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
git add backend/app/agent/__init__.py backend/app/agent/tools.py scripts/test_search_uploaded_docs_tool.py
git commit -m "Add search_uploaded_docs LangChain tool (query-only LLM schema, user_context injected server-side)"
```

---

### Task 3: Build `agent/chat_agent.py` and wire it into `chat.py`

**Files:**
- Create: `backend/app/agent/chat_agent.py`
- Modify: `backend/app/api/chat.py`
- Test: `scripts/test_chat_agent.py` (new)

**Interfaces:**
- Consumes: `app.agent.tools.UserContext`/`make_search_tool` (Task 2), `app.config.OPENAI_API_KEY`/`OPENAI_MODEL` (existing), `app.schemas.Source` (existing, unchanged shape).
- Produces: `run_agent_chat(question: str, user_context: dict | None = None, api_key: str | None = None, top_k: int = 5) -> dict` returning `{"answer": str, "sources": list[Source]}`. Internally loops up to `_MAX_TOOL_ITERATIONS = 3` rounds of tool calling before forcing a final answer, and prints a step-by-step trace to stdout on every call. `chat.py`'s `POST /api/chat` consumes this; `POST /api/chat/stream` is untouched and keeps using `search_sources`/`answer_question_stream` exactly as before.

- [ ] **Step 1: Write the failing test — `scripts/test_chat_agent.py`**

```python
#!/usr/bin/env python
"""Test agent/chat_agent.py end-to-end: tool routing, skipping, and multi-call use."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_TEST_COLLECTION = "test_chat_agent_collection"
os.environ["REALTIME_PDF_COLLECTION_NAME"] = _TEST_COLLECTION

from app.rag.types import Chunk
from app.rag.vector_store import add_chunks, get_collection
from app.agent.chat_agent import run_agent_chat


def test_chat_agent() -> bool:
    print("=" * 70)
    print("CHAT AGENT TEST")
    print("=" * 70)

    chunks = [
        Chunk(
            id="hash1:p1:c1",
            text="The office WiFi password is SkyBlue42.",
            metadata={"document_name": "office_handbook.pdf", "file_hash": "hash1", "page": 3, "chunk_index": 1},
        ),
        Chunk(
            id="hash2:p1:c1",
            text="The office manager's name is Priya Nair.",
            metadata={"document_name": "office_handbook.pdf", "file_hash": "hash2", "page": 7, "chunk_index": 1},
        ),
    ]

    try:
        print("\n[1/3] Seeding one chunk, asking a question that needs it...")
        add_chunks(chunks[:1])

        result = run_agent_chat("What is the office WiFi password?", user_context={})
        print(f"   Answer: {result['answer']}")
        print(f"   Sources: {result['sources']}")
        assert len(result["sources"]) > 0, f"expected the tool to be called: {result}"
        assert "SkyBlue42" in result["answer"], f"expected the password in the answer: {result}"
        print("   OK: tool was used, answer is grounded in the document")

        print("\n[2/3] Asking an unrelated question...")
        result2 = run_agent_chat("What is 2 + 2?", user_context={})
        print(f"   Answer: {result2['answer']}")
        print(f"   Sources: {result2['sources']}")
        assert result2["sources"] == [], f"expected the tool NOT to be called: {result2}"
        print("   OK: tool was correctly skipped for an unrelated question")

        print("\n[3/3] Asking a two-part question needing two distinct facts...")
        add_chunks(chunks[1:])
        result3 = run_agent_chat(
            "What is the office WiFi password, and what is the office manager's name?",
            user_context={},
        )
        print(f"   Answer: {result3['answer']}")
        print(f"   Sources: {result3['sources']}")
        assert len(result3["sources"]) >= 2, f"expected multiple tool results (parallel or looped): {result3}"
        assert "SkyBlue42" in result3["answer"], f"expected the WiFi password in the answer: {result3}"
        assert "Priya Nair" in result3["answer"], f"expected the manager's name in the answer: {result3}"
        print("   OK: agent gathered both facts, whether via one multi-hit search, parallel calls, or a loop")

        print("\nPASSED: chat_agent.py routes to the tool only when needed, and can use it more than once.")
        return True

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return False

    finally:
        client = get_collection()
        if client.has_collection(_TEST_COLLECTION):
            client.drop_collection(_TEST_COLLECTION)
            print(f"\n[cleanup] dropped {_TEST_COLLECTION}")


if __name__ == "__main__":
    success = test_chat_agent()
    sys.exit(0 if success else 1)
```

- [ ] **Step 2: Run it, confirm it fails**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_chat_agent.py
```
Expected: `ModuleNotFoundError: No module named 'app.agent.chat_agent'`.

- [ ] **Step 3: Create `backend/app/agent/chat_agent.py`**

```python
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.agent.tools import UserContext, make_search_tool
from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.schemas import Source

_SYSTEM_PROMPT = (
    "You are a helpful assistant. You have a tool, search_uploaded_docs, that searches "
    "documents the user has uploaded. Use it when the question could be answered from "
    "those documents. You may call it more than once in the same turn if the first "
    "search doesn't give you enough information — for example, to look up two distinct "
    "facts needed to answer a multi-part question. If the question is unrelated to any "
    "uploaded document (general knowledge, small talk, math, etc.), answer directly "
    "without using the tool."
)

_MAX_TOOL_ITERATIONS = 3


def _citation_to_source(citation: dict) -> Source:
    return Source(
        text=citation.get("text", ""),
        document=citation.get("source_name") or "Unknown",
        page=citation.get("page") if citation.get("page") is not None else "?",
        chunk=citation.get("chunk_id") or "?",
        score=None,
    )


def run_agent_chat(
    question: str,
    user_context: dict | None = None,
    api_key: str | None = None,
    top_k: int = 5,
) -> dict:
    print(f"🧑 [agent] question: {question!r}")

    ctx = UserContext(**(user_context or {}))
    search_tool = make_search_tool(ctx, top_k=top_k)

    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=api_key or OPENAI_API_KEY, temperature=0)
    llm_with_tools = llm.bind_tools([search_tool])

    messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=question)]
    sources: list[Source] = []

    for iteration in range(1, _MAX_TOOL_ITERATIONS + 1):
        ai_message = llm_with_tools.invoke(messages)

        if not ai_message.tool_calls:
            print(f"🤖 [agent] iteration {iteration}: no tool call, final answer")
            return {"answer": ai_message.content, "sources": sources}

        messages.append(ai_message)

        for tool_call in ai_message.tool_calls:
            print(f"🔧 [agent] iteration {iteration}: calling {tool_call['name']}({tool_call['args']})")
            tool_result = search_tool.invoke(tool_call["args"])
            print(
                f"   ↳ status={tool_result.get('status')} "
                f"result_count={tool_result.get('metadata', {}).get('result_count')}"
            )
            messages.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    tool_call_id=tool_call["id"],
                )
            )
            if tool_result.get("status") == "success":
                sources.extend(_citation_to_source(c) for c in tool_result.get("citations", []))

    # Hit the iteration cap: force a plain-text answer, without tools bound,
    # so the model can't request yet another call and stall the response.
    print(f"⚠️ [agent] hit max iterations ({_MAX_TOOL_ITERATIONS}), forcing final answer")
    final_message = llm.invoke(messages)
    return {"answer": final_message.content, "sources": sources}
```

- [ ] **Step 4: Rewrite `backend/app/api/chat.py`**

Replace the `chat()` function's body (keep `chat_stream()` and its imports untouched):

```python
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json

from app.agent.chat_agent import run_agent_chat
from app.config import TOP_K
from app.rag.generator import answer_question_stream
from app.rag.retriever import search_sources
from app.schemas import ChatRequest, ChatResponse, Source

router = APIRouter()


def _json_event(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _source_to_dict(source: Source) -> dict:
    if hasattr(source, "model_dump"):
        return source.model_dump()
    return source.dict()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Agent-backed chat endpoint - the agent decides whether to search uploaded
    documents (search_uploaded_docs tool) or answer directly.

    Args:
        request: ChatRequest with question and optional top_k, openai_api_key

    Returns:
        ChatResponse with answer and sources (empty if the tool wasn't used)

    Raises:
        HTTPException: If question is empty or other errors occur
    """
    try:
        if not request.question or not request.question.strip():
            raise ValueError("Question cannot be empty")

        result = run_agent_chat(
            question=request.question,
            user_context={},
            api_key=request.openai_api_key,
            top_k=request.top_k or TOP_K,
        )

        return ChatResponse(answer=result["answer"], sources=result["sources"])

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}") from e


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint - unchanged, still uses direct retrieval + generation
    (not the agent). Streams sources first, then answer tokens.

    Returns:
        NDJSON stream with sources, token chunks, and a done marker.
    """
    try:
        if not request.question or not request.question.strip():
            raise ValueError("Question cannot be empty")

        sources = search_sources(
            question=request.question,
            top_k=request.top_k or TOP_K
        )

        def generate():
            yield _json_event({
                "type": "sources",
                "data": [_source_to_dict(source) for source in sources],
            })

            for text_chunk in answer_question_stream(
                question=request.question,
                sources=sources,
                api_key=request.openai_api_key,
            ):
                yield _json_event({"type": "token", "data": text_chunk})

            yield _json_event({"type": "done"})

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stream processing failed: {str(e)}") from e
```

- [ ] **Step 5: Run the test, confirm it passes**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
backend/.venv/bin/python3 scripts/test_chat_agent.py
```
Expected: `PASSED: chat_agent.py routes to the tool only when needed, and can use it more than once.`,
exit 0. **Requires a valid `OPENAI_API_KEY` in `.env`** — this is the test that actually calls the
LLM. Watch the console output during this run — you should see the `🧑`/`🤖`/`🔧`/`↳` trace lines
for each of the three questions, confirming the tracing works end to end, not just that assertions
pass.

Note: steps 2 and 3 of the test depend on the model's judgment (whether/how many times it calls the
tool) — step 2 is expected to reliably skip the tool for an obviously unrelated question, and step 3
is expected to surface both seeded facts one way or another (a single search returning both hits,
parallel tool calls, or a second loop iteration all satisfy the assertion). If either ever flakes,
it's a prompt-tuning issue in `_SYSTEM_PROMPT`, not a sign the wiring or the loop itself is broken;
re-run once before investigating further.

- [ ] **Step 6: Commit**

```bash
cd /Users/baoshuangzhang/Desktop/Agentic_RAG/agentic_rag/00_RAG_Document_Chat
git add backend/app/agent/chat_agent.py backend/app/api/chat.py scripts/test_chat_agent.py
git commit -m "Add chat_agent.py and wire POST /api/chat to the LangChain agent"
```

---

## Self-Review

**Spec coverage:**
- `search_uploaded_docs` tool, query embedded inside the tool, fixed JSON response shape → Task 2, verified field-by-field in the test.
- Agent decides whether to use the tool → Task 3's test explicitly checks both branches (tool used / tool skipped).
- Agent can call the tool multiple times, not just once → `run_agent_chat()` is a bounded loop (`_MAX_TOOL_ITERATIONS = 3`), not a single fixed round trip; Task 3's third test case exercises a two-part question needing both seeded facts.
- Server-side tracing of the agent's process → every iteration, tool call (with args), tool result status, and the final "no tool call" / "max iterations hit" branch is printed in `run_agent_chat()`; Task 3's verification step explicitly calls out watching for these trace lines during the real run.
- `user_context` accepted but not used for filtering, and — the correctness fix agreed on — never LLM-fillable (only `query` in the bound tool's schema, asserted directly in Task 2's test: `assert list(search_tool.args.keys()) == ["query"]`).
- Document embedding automatic on upload → untouched, `upload.py` not modified by this plan.
- `/api/chat` replaced, `/api/chat/stream` untouched → Task 3, `chat_stream()` copied verbatim into the rewritten file.
- Offline docs / numeric data out of scope → no file in this plan touches `offline_docs_store.py` or `numeric_data_store.py`.

**Placeholder scan:** every step has complete, runnable code — the tool/agent code shown here was
validated against the actual installed `langchain`/`langchain-core`/`langchain-openai` APIs during
design (confirmed `@tool` schema inference, `bind_tools`, `tool_calls` shape, `ToolMessage` fields)
rather than guessed.

**Type/naming consistency:** `Source` fields (`text`, `document`, `page`, `chunk`, `score`) match
`app/schemas.py` exactly; `ChatResponse(answer, sources)` is unchanged; `UserContext` field names
match the user-specified schema (`user_id`, `department_id`, `session_id`) exactly.

---

**Plan complete and saved to `00_RAG_Document_Chat/docs/superpowers/plans/2026-08-06-langchain-agent-layer-implementation.md`.** Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — I execute the 3 tasks in this session using `executing-plans`, with checkpoints for you to review.

Which approach?