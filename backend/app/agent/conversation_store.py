from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _initial_title(question: str, max_length: int = 60) -> str:
    normalized = " ".join(question.split())
    return (normalized or "New conversation")[:max_length]


class ConversationStore:
    """Metadata-only catalog for discovering LangGraph conversation threads."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            if self._conn is not None:
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
            self._conn = conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ConversationStore has not been started")
        return self._conn

    @staticmethod
    def _validate_id(conversation_id: str) -> str:
        normalized = conversation_id.strip()
        if not normalized:
            raise ValueError("conversation_id cannot be empty")
        return normalized

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def ensure(self, conversation_id: str, first_question: str) -> dict:
        conversation_id = self._validate_id(conversation_id)
        now = _utc_now()
        with self._lock:
            conn = self._connection()
            conn.execute(
                """
                INSERT OR IGNORE INTO app_conversations (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, _initial_title(first_question), now, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM app_conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to create conversation: {conversation_id}")
        return self._row_to_dict(row)

    def list(self) -> list[dict]:
        with self._lock:
            rows = self._connection().execute(
                """
                SELECT id, title, created_at, updated_at
                FROM app_conversations
                ORDER BY updated_at DESC, created_at DESC
                """
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get(self, conversation_id: str) -> dict | None:
        conversation_id = self._validate_id(conversation_id)
        with self._lock:
            row = self._connection().execute(
                "SELECT id, title, created_at, updated_at FROM app_conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def touch(self, conversation_id: str) -> None:
        conversation_id = self._validate_id(conversation_id)
        with self._lock:
            conn = self._connection()
            conn.execute(
                "UPDATE app_conversations SET updated_at = ? WHERE id = ?",
                (_utc_now(), conversation_id),
            )
            conn.commit()

    def rename(self, conversation_id: str, title: str) -> dict | None:
        conversation_id = self._validate_id(conversation_id)
        normalized_title = " ".join(title.split())
        if not normalized_title:
            raise ValueError("title cannot be empty")

        with self._lock:
            conn = self._connection()
            cursor = conn.execute(
                "UPDATE app_conversations SET title = ?, updated_at = ? WHERE id = ?",
                (normalized_title[:200], _utc_now(), conversation_id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                return None
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM app_conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def delete(self, conversation_id: str) -> bool:
        conversation_id = self._validate_id(conversation_id)
        with self._lock:
            conn = self._connection()
            cursor = conn.execute(
                "DELETE FROM app_conversations WHERE id = ?",
                (conversation_id,),
            )
            conn.commit()
        return cursor.rowcount > 0
