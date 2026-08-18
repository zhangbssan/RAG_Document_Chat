#!/usr/bin/env python
"""Test agent/chat_agent.py end-to-end: tool routing, skipping, and multi-call use."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_TEST_COLLECTION = "test_chat_agent_collection"
os.environ["REALTIME_PDF_COLLECTION_NAME"] = _TEST_COLLECTION

from app.rag.types import Chunk
from app.rag.vector_store import add_chunks, get_collection
from app.agent.runtime import AgentRuntime
from app.config import OPENAI_API_KEY


def test_chat_agent() -> bool:
    print("=" * 70)
    print("CHAT AGENT TEST")
    print("=" * 70)

    if not OPENAI_API_KEY:
        print("SKIPPED: OPENAI_API_KEY is not configured.")
        return True

    chunks = [
        Chunk(
            id="hash1:p1:c1",
            text="The office WiFi password is SkyBlue42.",
            metadata={"document_name": "office_handbook.pdf", "file_hash": "hash1", "page": 3, "chunk_index": 1, "chunk_seq": 1},
        ),
        Chunk(
            id="hash2:p1:c1",
            text="The office manager's name is Priya Nair.",
            metadata={"document_name": "office_handbook.pdf", "file_hash": "hash2", "page": 7, "chunk_index": 1, "chunk_seq": 1},
        ),
    ]

    temp_dir = tempfile.TemporaryDirectory(prefix="live-chat-agent-")
    runtime = AgentRuntime(Path(temp_dir.name) / "chat.sqlite")
    runtime.start()

    try:
        print("\n[1/3] Seeding one chunk, asking a question that needs it...")
        add_chunks(chunks[:1])

        result = runtime.run_chat(
            question="What is the office WiFi password?",
            conversation_id="live-tool-question",
            user_context={},
            api_key=OPENAI_API_KEY,
            top_k=5,
        )
        print(f"   Answer: {result['answer']}")
        print(f"   Sources: {result['sources']}")
        assert len(result["sources"]) > 0, f"expected the tool to be called: {result}"
        assert "SkyBlue42" in result["answer"], f"expected the password in the answer: {result}"
        print("   OK: tool was used, answer is grounded in the document")

        print("\n[2/3] Asking an unrelated question...")
        result2 = runtime.run_chat(
            question="What is 2 + 2?",
            conversation_id="live-general-question",
            user_context={},
            api_key=OPENAI_API_KEY,
            top_k=5,
        )
        print(f"   Answer: {result2['answer']}")
        print(f"   Sources: {result2['sources']}")
        assert result2["sources"] == [], f"expected the tool NOT to be called: {result2}"
        print("   OK: tool was correctly skipped for an unrelated question")

        print("\n[3/3] Asking a two-part question needing two distinct facts...")
        add_chunks(chunks[1:])
        result3 = runtime.run_chat(
            question="What is the office WiFi password, and what is the office manager's name?",
            conversation_id="live-multipart-question",
            user_context={},
            api_key=OPENAI_API_KEY,
            top_k=5,
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
        runtime.close()
        temp_dir.cleanup()
        client = get_collection()
        if client.has_collection(_TEST_COLLECTION):
            client.drop_collection(_TEST_COLLECTION)
            print(f"\n[cleanup] dropped {_TEST_COLLECTION}")


if __name__ == "__main__":
    success = test_chat_agent()
    sys.exit(0 if success else 1)
