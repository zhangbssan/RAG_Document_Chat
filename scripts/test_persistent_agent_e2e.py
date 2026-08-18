#!/usr/bin/env python
"""End-to-end upload → Tool → Agent memory test with live Milvus and a fake LLM."""
from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

_repo_dir = Path(__file__).resolve().parents[1]
_backend_dir = _repo_dir / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_temp_dir = tempfile.TemporaryDirectory(prefix="persistent-agent-e2e-")
_TEST_COLLECTION = "test_persistent_agent_e2e_collection"
os.environ["REALTIME_PDF_COLLECTION_NAME"] = _TEST_COLLECTION
os.environ["CHAT_DB_PATH"] = str(Path(_temp_dir.name) / "chat.sqlite")
os.environ["UPLOAD_DIR"] = str(Path(_temp_dir.name) / "uploads")

from app.rag import embeddings as embedding_module


def _test_embed(texts: list[str], dimensions: int = 384) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dimensions
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        vectors.append([value / norm for value in vector])
    return vectors


embedding_module.embed_texts = _test_embed
embedding_module.embed_query = lambda query: _test_embed([query])[0]

from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.main import app
from app.rag.vector_store import get_collection

CONVERSATION_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
CONVERSATION_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


class RetrievalAwareFakeModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "retrieval-aware-e2e-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        if messages and isinstance(messages[-1], ToolMessage):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Grounded in the uploaded PDF."))]
            )

        humans = [message for message in messages if isinstance(message, HumanMessage)]
        question = str(humans[-1].content) if humans else ""
        prior_tool_results = [message for message in messages if isinstance(message, ToolMessage)]

        if "follow-up" in question.lower() or "remember" in question.lower():
            answer = "I remember the earlier PDF evidence." if prior_tool_results else "No prior PDF evidence."
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_uploaded_docs",
                                "args": {"query": question},
                                "id": "e2e-tool-call",
                                "type": "tool_call",
                            }
                        ],
                    )
                )
            ]
        )


def _configure_fake_model() -> None:
    app.state.agent_runtime.model_factory = lambda _key: RetrievalAwareFakeModel()


def main() -> bool:
    sample_pdf = _repo_dir / "sample_docs" / "employee_handbook_en.pdf"
    try:
        with TestClient(app) as client:
            _configure_fake_model()
            with sample_pdf.open("rb") as pdf_file:
                uploaded = client.post(
                    "/api/upload",
                    files={"files": (sample_pdf.name, pdf_file, "application/pdf")},
                )
            assert uploaded.status_code == 200, uploaded.text
            assert uploaded.json()["added_chunks"] > 0

            first = client.post(
                "/api/chat",
                json={
                    "question": "What is the annual leave entitlement?",
                    "conversation_id": CONVERSATION_A,
                    "openai_api_key": "test",
                },
            )
            assert first.status_code == 200, first.text
            assert first.json()["sources"], first.json()
            assert first.json()["sources"][0]["document"] == sample_pdf.name

            follow_up = client.post(
                "/api/chat",
                json={
                    "question": "This is a follow-up; do you remember that evidence?",
                    "conversation_id": CONVERSATION_A,
                    "openai_api_key": "test",
                },
            )
            assert "remember" in follow_up.json()["answer"].lower()
            assert follow_up.json()["sources"] == []

            isolated = client.post(
                "/api/chat",
                json={
                    "question": "This is a follow-up; do you remember that evidence?",
                    "conversation_id": CONVERSATION_B,
                    "openai_api_key": "test",
                },
            )
            assert isolated.json()["answer"] == "No prior PDF evidence."

        # A new lifespan creates a new runtime on the same SQLite file.
        with TestClient(app) as restarted_client:
            _configure_fake_model()
            resumed = restarted_client.post(
                "/api/chat",
                json={
                    "question": "After restart, do you remember the earlier result?",
                    "conversation_id": CONVERSATION_A,
                    "openai_api_key": "test",
                },
            )
            assert resumed.status_code == 200, resumed.text
            assert "remember" in resumed.json()["answer"].lower()

            history = restarted_client.get(f"/api/conversations/{CONVERSATION_A}/messages")
            assert history.status_code == 200
            assistant_messages = [
                message for message in history.json()["messages"] if message["role"] == "assistant"
            ]
            assert assistant_messages[0]["sources"], assistant_messages
            assert assistant_messages[1]["sources"] == []

        print("PASSED: upload, retrieval Tool, memory, isolation, restart, and citations work end to end.")
        return True
    finally:
        client = get_collection()
        if client.has_collection(_TEST_COLLECTION):
            client.drop_collection(_TEST_COLLECTION)
        _temp_dir.cleanup()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
