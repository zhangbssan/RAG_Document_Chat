#!/usr/bin/env python
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent.runtime import AgentRuntime


class HistoryAwareFakeModel(BaseChatModel):
    observations: list[list[str]]

    @property
    def _llm_type(self) -> str:
        return "history-aware-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        human_messages = [str(message.content) for message in messages if isinstance(message, HumanMessage)]
        self.observations.append(human_messages)
        answer = "seen:" + "|".join(human_messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])


def main() -> bool:
    observations: list[list[str]] = []

    def model_factory(_api_key: str) -> BaseChatModel:
        return HistoryAwareFakeModel(observations=observations)

    with tempfile.TemporaryDirectory(prefix="agent-memory-") as tmp:
        db_path = Path(tmp) / "chat.sqlite"
        runtime = AgentRuntime(db_path, model_factory=model_factory)
        runtime.start()
        try:
            first = runtime.run_chat(
                question="first question",
                conversation_id="conversation-a",
                user_context={},
                api_key="test-key",
                top_k=5,
            )
            assert first["answer"] == "seen:first question"

            follow_up = runtime.run_chat(
                question="follow up",
                conversation_id="conversation-a",
                user_context={},
                api_key="test-key",
                top_k=5,
            )
            assert follow_up["answer"] == "seen:first question|follow up"

            isolated = runtime.run_chat(
                question="separate question",
                conversation_id="conversation-b",
                user_context={},
                api_key="test-key",
                top_k=5,
            )
            assert isolated["answer"] == "seen:separate question"
        finally:
            runtime.close()

        restarted = AgentRuntime(db_path, model_factory=model_factory)
        restarted.start()
        try:
            resumed = restarted.run_chat(
                question="after restart",
                conversation_id="conversation-a",
                user_context={},
                api_key="test-key",
                top_k=5,
            )
            assert resumed["answer"] == "seen:first question|follow up|after restart"
            history = restarted.get_display_history("conversation-a")
            assert [message["role"] for message in history] == [
                "user",
                "assistant",
                "user",
                "assistant",
                "user",
                "assistant",
            ]
            assert history[-1]["content"].endswith("after restart")

            restarted.delete_conversation_state("conversation-a")
            assert restarted.checkpoint_messages("conversation-a") == []
        finally:
            restarted.close()

    print("PASSED: LangGraph SQLite memory persists, isolates threads, and survives restart.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
