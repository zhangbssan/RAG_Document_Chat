from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.agent.tools import UserContext, make_search_tool
from app.config import OPENAI_MODEL
from app.schemas import Source

SYSTEM_PROMPT = (
    "You are a helpful assistant for a user who has uploaded their own private documents "
    "to this system. You have exactly one tool, search_uploaded_docs, which searches content "
    "the user already owns and is authorized to access. Always use it for fact-based questions "
    "that could be answered from the uploaded documents, including passwords, credentials, or "
    "configuration values written in those documents. Retrieving information from the user's "
    "own uploaded documents is not a security risk, so do not refuse or hedge on that basis. "
    "You may call the tool more than once for multi-part questions. Skip it only for questions "
    "clearly unrelated to the uploaded documents, such as small talk, general knowledge, or "
    "arithmetic, and answer those questions directly. When the tool returns evidence, answer "
    "from that evidence and cite only the "
    "returned source names, page ranges, and links. Never invent a document, page, chunk id, "
    "or citation link. If the retrieved evidence is insufficient, say so."
)

MAX_TOOL_ITERATIONS = 3
RECURSION_LIMIT = 2 * MAX_TOOL_ITERATIONS + 1


def citation_to_source(citation: dict[str, Any]) -> Source:
    return Source(
        text=citation.get("text", ""),
        document=citation.get("source_name") or "Unknown",
        page=citation.get("page") if citation.get("page") is not None else "?",
        pages=citation.get("pages"),
        chunk=citation.get("chunk_id") or "?",
        score=None,
        link=citation.get("link"),
    )


def _tool_payload(message: ToolMessage) -> dict[str, Any] | None:
    content = message.content
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def deduplicate_sources(sources: list[Source]) -> list[Source]:
    deduplicated: list[Source] = []
    seen: set[str] = set()
    for source in sources:
        key = str(source.chunk)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(source)
    return deduplicated


def sources_from_messages(messages: list[Any]) -> list[Source]:
    sources: list[Source] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        payload = _tool_payload(message)
        if not payload or payload.get("status") != "success":
            continue
        citations = payload.get("citations", [])
        if not isinstance(citations, list):
            continue
        for citation in citations:
            if not isinstance(citation, dict) or not citation.get("chunk_id"):
                continue
            sources.append(citation_to_source(citation))
    return deduplicate_sources(sources)


def message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content) if content is not None else ""


def build_agent(
    *,
    checkpointer,
    user_context: dict | None,
    api_key: str,
    top_k: int,
    model: BaseChatModel | None = None,
):
    """Build one request-scoped graph while reusing persistent checkpoint state."""
    ctx = UserContext(**(user_context or {}))
    search_tool = make_search_tool(ctx, top_k=top_k)
    llm = model or ChatOpenAI(model=OPENAI_MODEL, api_key=api_key, temperature=0)
    return create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def print_trace(messages: list[Any]) -> None:
    for msg in messages:
        if isinstance(msg, HumanMessage):
            print(f"🧑 [agent] question: {msg.content!r}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for call in msg.tool_calls:
                    print(f"🔧 [agent] calling {call['name']}({call['args']})")
            elif msg.content:
                print(f"🤖 [agent] final answer: {message_text(msg)[:200]!r}")
        elif isinstance(msg, ToolMessage):
            payload = _tool_payload(msg)
            if payload:
                print(
                    f"   ↳ status={payload.get('status')} "
                    f"result_count={payload.get('metadata', {}).get('result_count')}"
                )
            else:
                print(f"   ↳ tool result: {message_text(msg)[:200]!r}")
