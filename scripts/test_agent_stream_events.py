#!/usr/bin/env python
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from app.agent import chat_agent
from app.agent.runtime import AgentRuntime


@tool("search_uploaded_docs")
def fake_search_uploaded_docs(query: str) -> dict:
    """Search deterministic test documents."""
    return {
        "status": "success",
        "content": "The trial period is three months.",
        "citations": [
            {
                "source_id": "hash",
                "source_name": "handbook.pdf",
                "page": 4,
                "pages": [4],
                "chunk_id": "hash:seq4",
                "link": "doc:hash#p4",
                "text": "The trial period is three months.",
            }
        ],
        "metadata": {"tool": "search_uploaded_docs", "result_count": 1},
        "error": None,
    }


class ToolCallingFakeModel(BaseChatModel):
    _calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "tool-calling-stream-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        self._calls += 1
        if self._calls == 1:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_uploaded_docs",
                        "args": {"query": "trial period"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            message = AIMessage(content="The trial period is three months.")
        return ChatResult(generations=[ChatGeneration(message=message)])


class DirectFakeModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "direct-stream-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Four."))])


def main() -> bool:
    original_tool_factory = chat_agent.make_search_tool
    chat_agent.make_search_tool = lambda _ctx, top_k=5: fake_search_uploaded_docs
    try:
        with tempfile.TemporaryDirectory(prefix="agent-stream-") as tmp:
            runtime = AgentRuntime(
                Path(tmp) / "chat.sqlite",
                model_factory=lambda _key: ToolCallingFakeModel(),
            )
            runtime.start()
            try:
                events = list(
                    runtime.stream_chat(
                        question="What is the trial period?",
                        conversation_id="tool-chat",
                        user_context={},
                        api_key="test",
                        top_k=5,
                    )
                )
                event_types = [event["type"] for event in events]
                assert event_types == ["tool_start", "sources", "token", "done"], events
                assert events[0]["query"] == "trial period"
                assert events[1]["data"][0].chunk == "hash:seq4"
                assert events[2]["data"] == "The trial period is three months."

                runtime.model_factory = lambda _key: DirectFakeModel()
                direct_events = list(
                    runtime.stream_chat(
                        question="What is 2 + 2?",
                        conversation_id="direct-chat",
                        user_context={},
                        api_key="test",
                        top_k=5,
                    )
                )
                assert [event["type"] for event in direct_events] == ["token", "done"]
                assert direct_events[0]["data"] == "Four."
            finally:
                runtime.close()
    finally:
        chat_agent.make_search_tool = original_tool_factory

    print("PASSED: native LangGraph stream events cover Tool and no-Tool turns.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
