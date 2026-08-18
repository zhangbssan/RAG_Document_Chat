from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable

from fastapi import Request
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError

from app.agent.chat_agent import (
    RECURSION_LIMIT,
    build_agent,
    deduplicate_sources,
    message_text,
    print_trace,
    sources_from_messages,
)
from app.agent.conversation_store import ConversationStore

ModelFactory = Callable[[str], BaseChatModel]


class AgentRuntime:
    """Lifecycle wrapper around LangChain's framework-owned agent graph."""

    def __init__(self, db_path: Path, model_factory: ModelFactory | None = None) -> None:
        self.db_path = Path(db_path)
        self.model_factory = model_factory
        self.conversations = ConversationStore(self.db_path)
        self._conn: sqlite3.Connection | None = None
        self._checkpointer: SqliteSaver | None = None
        self._lifecycle_lock = threading.RLock()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._checkpointer is not None:
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            checkpointer = SqliteSaver(conn)
            checkpointer.setup()

            self._conn = conn
            self._checkpointer = checkpointer
            self.conversations.start()

    def close(self) -> None:
        with self._lifecycle_lock:
            self.conversations.close()
            if self._conn is not None:
                self._conn.close()
            self._conn = None
            self._checkpointer = None

    @property
    def checkpointer(self) -> SqliteSaver:
        if self._checkpointer is None:
            raise RuntimeError("AgentRuntime has not been started")
        return self._checkpointer

    @staticmethod
    def _config(conversation_id: str) -> dict:
        return {
            "configurable": {"thread_id": conversation_id},
            "recursion_limit": RECURSION_LIMIT,
        }

    def checkpoint_messages(self, conversation_id: str) -> list[Any]:
        checkpoint = self.checkpointer.get(self._config(conversation_id))
        if not checkpoint:
            return []
        channel_values = checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        return list(messages) if isinstance(messages, (list, tuple)) else []

    def _model(self, api_key: str) -> BaseChatModel | None:
        return self.model_factory(api_key) if self.model_factory else None

    def run_chat(
        self,
        *,
        question: str,
        conversation_id: str,
        user_context: dict | None,
        api_key: str,
        top_k: int,
    ) -> dict:
        config = self._config(conversation_id)
        previous_message_count = len(self.checkpoint_messages(conversation_id))
        agent = build_agent(
            checkpointer=self.checkpointer,
            user_context=user_context,
            api_key=api_key,
            top_k=top_k,
            model=self._model(api_key),
        )

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=question)]},
                config=config,
            )
        except GraphRecursionError:
            return {
                "answer": (
                    "I wasn't able to finish researching this within the allowed number of "
                    "tool calls. Please try rephrasing your question."
                ),
                "sources": [],
            }

        messages = list(result.get("messages", []))
        new_messages = messages[previous_message_count:]
        print_trace(new_messages)
        final_message = next(
            (
                message
                for message in reversed(new_messages)
                if isinstance(message, AIMessage) and not message.tool_calls and message_text(message)
            ),
            None,
        )
        if final_message is None:
            raise RuntimeError("Agent completed without a final answer")

        return {
            "answer": message_text(final_message),
            "sources": sources_from_messages(new_messages),
        }

    def get_display_history(self, conversation_id: str) -> list[dict]:
        display: list[dict] = []
        pending_tool_messages: list[Any] = []

        for message in self.checkpoint_messages(conversation_id):
            if isinstance(message, HumanMessage):
                display.append({"role": "user", "content": message_text(message), "sources": []})
                pending_tool_messages = []
            elif isinstance(message, ToolMessage):
                pending_tool_messages.append(message)
            elif isinstance(message, AIMessage) and not message.tool_calls and message_text(message):
                display.append(
                    {
                        "role": "assistant",
                        "content": message_text(message),
                        "sources": sources_from_messages(pending_tool_messages),
                    }
                )
                pending_tool_messages = []

        return display

    def stream_chat(
        self,
        *,
        question: str,
        conversation_id: str,
        user_context: dict | None,
        api_key: str,
        top_k: int,
    ):
        """Stream framework events without implementing a custom Agent loop."""
        config = self._config(conversation_id)
        agent = build_agent(
            checkpointer=self.checkpointer,
            user_context=user_context,
            api_key=api_key,
            top_k=top_k,
            model=self._model(api_key),
        )
        buffered_model_text: list[str] = []
        current_sources = []

        try:
            for mode, data in agent.stream(
                {"messages": [HumanMessage(content=question)]},
                config=config,
                stream_mode=["messages", "updates"],
                durability="sync",
            ):
                if mode == "messages":
                    message, metadata = data
                    if metadata.get("langgraph_node") == "model" and isinstance(message, AIMessage):
                        text = message_text(message)
                        if text:
                            buffered_model_text.append(text)
                    continue

                if mode != "updates" or not isinstance(data, dict):
                    continue

                model_update = data.get("model")
                if isinstance(model_update, dict):
                    model_messages = model_update.get("messages", [])
                    for message in model_messages:
                        if not isinstance(message, AIMessage):
                            continue
                        if message.tool_calls:
                            buffered_model_text = []
                            for call in message.tool_calls:
                                yield {
                                    "type": "tool_start",
                                    "tool": call.get("name", "search_uploaded_docs"),
                                    "query": call.get("args", {}).get("query", ""),
                                }
                        else:
                            pieces = buffered_model_text or [message_text(message)]
                            for piece in pieces:
                                if piece:
                                    yield {"type": "token", "data": piece}
                            buffered_model_text = []

                tools_update = data.get("tools")
                if isinstance(tools_update, dict):
                    tool_messages = tools_update.get("messages", [])
                    new_sources = sources_from_messages(list(tool_messages))
                    if new_sources:
                        current_sources = deduplicate_sources(current_sources + new_sources)
                        yield {"type": "sources", "data": current_sources}

            yield {"type": "done"}
        except GraphRecursionError:
            yield {
                "type": "error",
                "data": "The Agent reached its Tool-call limit. Please rephrase the question.",
            }
        except Exception as exc:
            yield {"type": "error", "data": f"Agent stream failed: {str(exc)}"}

    def delete_conversation_state(self, conversation_id: str) -> None:
        self.checkpointer.delete_thread(conversation_id)


def get_agent_runtime(request: Request) -> AgentRuntime:
    runtime = getattr(request.app.state, "agent_runtime", None)
    if not isinstance(runtime, AgentRuntime):
        raise RuntimeError("Agent runtime is not available")
    return runtime
