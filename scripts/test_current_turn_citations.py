#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.chat_agent import sources_from_messages


def _tool_message(chunk_id: str, source_name: str = "doc.pdf") -> ToolMessage:
    return ToolMessage(
        content=json.dumps(
            {
                "status": "success",
                "content": "evidence",
                "citations": [
                    {
                        "source_id": "hash",
                        "source_name": source_name,
                        "page": 2,
                        "pages": [2],
                        "chunk_id": chunk_id,
                        "link": "doc:hash#p2",
                        "text": "evidence",
                    }
                ],
                "metadata": {"tool": "search_uploaded_docs", "result_count": 1},
                "error": None,
            }
        ),
        tool_call_id=f"call-{chunk_id}",
        name="search_uploaded_docs",
    )


def main() -> bool:
    old_messages = [
        HumanMessage(content="old question"),
        _tool_message("old-chunk"),
        AIMessage(content="old answer"),
    ]
    new_messages = [
        HumanMessage(content="new question"),
        _tool_message("new-chunk"),
        _tool_message("new-chunk"),
        ToolMessage(content="not json", tool_call_id="broken", name="search_uploaded_docs"),
        AIMessage(content="new answer"),
    ]

    current_turn = (old_messages + new_messages)[len(old_messages):]
    sources = sources_from_messages(current_turn)
    assert [source.chunk for source in sources] == ["new-chunk"]
    assert sources[0].document == "doc.pdf"
    assert sources[0].link == "doc:hash#p2"

    no_tool_follow_up = [HumanMessage(content="thanks"), AIMessage(content="you are welcome")]
    assert sources_from_messages(no_tool_follow_up) == []

    print("PASSED: citations are current-turn only, validated, and deduplicated.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
