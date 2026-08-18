#!/usr/bin/env python
from __future__ import annotations

import os
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_tmp = tempfile.TemporaryDirectory(prefix="conversation-api-")
os.environ["CHAT_DB_PATH"] = str(Path(_tmp.name) / "chat.sqlite")

from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.main import app

A_ID = "11111111-1111-4111-8111-111111111111"
B_ID = "22222222-2222-4222-8222-222222222222"
C_ID = "33333333-3333-4333-8333-333333333333"


class HistoryAwareFakeModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "api-history-aware-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        humans = [str(message.content) for message in messages if isinstance(message, HumanMessage)]
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="seen:" + "|".join(humans)))]
        )


def main() -> bool:
    try:
        with TestClient(app) as client:
            app.state.agent_runtime.model_factory = lambda _key: HistoryAwareFakeModel()

            first = client.post(
                "/api/chat",
                json={"question": "first", "conversation_id": A_ID, "openai_api_key": "test"},
            )
            assert first.status_code == 200, first.text
            assert first.json()["answer"] == "seen:first"

            follow_up = client.post(
                "/api/chat",
                json={"question": "follow up", "conversation_id": A_ID, "openai_api_key": "test"},
            )
            assert follow_up.json()["answer"] == "seen:first|follow up"

            separate = client.post(
                "/api/chat",
                json={"question": "separate", "conversation_id": B_ID, "openai_api_key": "test"},
            )
            assert separate.json()["answer"] == "seen:separate"

            streamed = client.post(
                "/api/chat/stream",
                json={"question": "streamed", "conversation_id": C_ID, "openai_api_key": "test"},
            )
            assert streamed.status_code == 200, streamed.text
            stream_events = [json.loads(line) for line in streamed.text.splitlines() if line]
            assert [event["type"] for event in stream_events] == ["token", "done"]
            assert stream_events[0]["data"] == "seen:streamed"

            listed = client.get("/api/conversations")
            assert listed.status_code == 200
            assert {item["id"] for item in listed.json()["conversations"]} == {A_ID, B_ID, C_ID}

            history = client.get(f"/api/conversations/{A_ID}/messages")
            assert history.status_code == 200
            assert [item["role"] for item in history.json()["messages"]] == [
                "user",
                "assistant",
                "user",
                "assistant",
            ]

            renamed = client.patch(f"/api/conversations/{A_ID}", json={"title": "Renamed A"})
            assert renamed.status_code == 200
            assert renamed.json()["title"] == "Renamed A"

            deleted = client.delete(f"/api/conversations/{B_ID}")
            assert deleted.status_code == 200 and deleted.json()["deleted"] is True
            deleted_again = client.delete(f"/api/conversations/{B_ID}")
            assert deleted_again.status_code == 200 and deleted_again.json()["deleted"] is False
            assert client.get(f"/api/conversations/{B_ID}/messages").status_code == 404
    finally:
        _tmp.cleanup()

    print("PASSED: conversation APIs persist, isolate, list, load, rename, and delete chats.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
