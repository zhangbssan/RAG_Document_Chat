#!/usr/bin/env python
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.agent.conversation_store import ConversationStore


def main() -> bool:
    with tempfile.TemporaryDirectory(prefix="conversation-store-") as tmp:
        store = ConversationStore(Path(tmp) / "chat.sqlite")
        store.start()
        try:
            first = store.ensure("a", "  What   does the handbook say about annual leave?  ")
            assert first["title"] == "What does the handbook say about annual leave?"

            repeated = store.ensure("a", "This must not replace the title")
            assert repeated["title"] == first["title"]
            assert repeated["created_at"] == first["created_at"]

            time.sleep(0.002)
            store.ensure("b", "Second conversation")
            assert [item["id"] for item in store.list()] == ["b", "a"]

            renamed = store.rename("a", "  Renamed   conversation ")
            assert renamed is not None and renamed["title"] == "Renamed conversation"
            assert store.get("missing") is None
            assert store.delete("a") is True
            assert store.delete("a") is False
        finally:
            store.close()

    print("PASSED: conversation catalog CRUD is persistent and metadata-only.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
