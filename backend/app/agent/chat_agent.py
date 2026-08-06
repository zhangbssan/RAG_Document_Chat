from __future__ import annotations

import json

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

from app.agent.tools import UserContext, make_search_tool
from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.schemas import Source

_SYSTEM_PROMPT = (
    "You are a helpful assistant for a user who has uploaded their own private documents "
    "to this system. You have a tool, search_uploaded_docs, that searches those documents "
    "— content the user already owns and has full access to. Always use the tool for any "
    "fact-based question that could be answered from the user's own documents, including "
    "things like passwords, credentials, or configuration values written in them. Looking "
    "up information in the user's own uploaded document is not a security risk, and you "
    "should never refuse or hedge on that basis — just call the tool. You may call it "
    "more than once if the first search doesn't give you enough information — for "
    "example, to look up two distinct facts needed to answer a multi-part question. Only "
    "skip the tool for questions clearly unrelated to any document (general knowledge, "
    "small talk, math, etc.)."
)

_MAX_TOOL_ITERATIONS = 3
# Each tool-calling round is one "agent" graph step plus one "tools" graph step;
# +1 covers the final agent step that produces the answer with no further tool call.
_RECURSION_LIMIT = 2 * _MAX_TOOL_ITERATIONS + 1


def _citation_to_source(citation: dict) -> Source:
    return Source(
        text=citation.get("text", ""),
        document=citation.get("source_name") or "Unknown",
        page=citation.get("page") if citation.get("page") is not None else "?",
        chunk=citation.get("chunk_id") or "?",
        score=None,
    )


def _print_trace(messages: list) -> None:
    for msg in messages:
        kind = msg.__class__.__name__
        if kind == "HumanMessage":
            print(f"🧑 [agent] question: {msg.content!r}")
        elif kind == "AIMessage":
            if getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    print(f"🔧 [agent] calling {call['name']}({call['args']})")
            elif msg.content:
                print(f"🤖 [agent] final answer: {str(msg.content)[:200]!r}")
        elif kind == "ToolMessage":
            try:
                payload = json.loads(msg.content)
                print(
                    f"   ↳ status={payload.get('status')} "
                    f"result_count={payload.get('metadata', {}).get('result_count')}"
                )
            except (TypeError, ValueError):
                print(f"   ↳ tool result: {str(msg.content)[:200]!r}")


def run_agent_chat(
    question: str,
    user_context: dict | None = None,
    api_key: str | None = None,
    top_k: int = 5,
) -> dict:
    ctx = UserContext(**(user_context or {}))
    search_tool = make_search_tool(ctx, top_k=top_k)

    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=api_key or OPENAI_API_KEY, temperature=0)
    agent = create_agent(model=llm, tools=[search_tool], system_prompt=_SYSTEM_PROMPT)

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": _RECURSION_LIMIT},
        )
    except GraphRecursionError:
        print(f"⚠️ [agent] hit max iterations ({_MAX_TOOL_ITERATIONS}) without a final answer")
        return {
            "answer": (
                "I wasn't able to finish researching this within the allowed number of "
                "tool calls. Please try rephrasing your question."
            ),
            "sources": [],
        }

    messages = result["messages"]
    _print_trace(messages)

    sources: list[Source] = []
    for msg in messages:
        if msg.__class__.__name__ != "ToolMessage":
            continue
        try:
            payload = json.loads(msg.content)
        except (TypeError, ValueError):
            continue
        if payload.get("status") == "success":
            sources.extend(_citation_to_source(c) for c in payload.get("citations", []))

    return {"answer": messages[-1].content, "sources": sources}