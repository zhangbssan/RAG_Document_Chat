#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.schemas import ChatRequest, ConversationHistoryResponse, ConversationMessage, ConversationSummary, Source


def main() -> bool:
    conversation_id = "11111111-1111-4111-8111-111111111111"
    request = ChatRequest(question="hello", conversation_id=f" {conversation_id} ")
    assert request.conversation_id == conversation_id

    try:
        ChatRequest(question="hello", conversation_id="   ")
        raise AssertionError("blank conversation_id should fail validation")
    except ValidationError:
        pass

    try:
        ChatRequest(question="hello", conversation_id="not-a-uuid")
        raise AssertionError("non-UUID conversation_id should fail validation")
    except ValidationError:
        pass

    source = Source(
        text="Evidence",
        document="doc.pdf",
        page=2,
        pages=[2, 3],
        chunk="hash:seq2-3",
        link="doc:hash#p2-3",
    )
    history = ConversationHistoryResponse(
        conversation=ConversationSummary(
            id=conversation_id,
            title="A title",
            created_at="2026-08-17T00:00:00+00:00",
            updated_at="2026-08-17T00:00:00+00:00",
        ),
        messages=[ConversationMessage(role="assistant", content="Answer", sources=[source])],
    )
    payload = history.model_dump()
    assert payload["messages"][0]["sources"][0]["pages"] == [2, 3]
    assert payload["messages"][0]["sources"][0]["link"] == "doc:hash#p2-3"
    print("PASSED: conversation schemas validate ids and preserve source ranges/links.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
