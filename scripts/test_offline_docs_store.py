#!/usr/bin/env python
"""Test offline_docs_store.py: ingest markdown -> MariaDB + Milvus, then search + verify back-reference."""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_TEST_COLLECTION = "test_offline_docs_collection"
_TEST_DEPARTMENT = "test-restructure-dept"
os.environ["OFFLINE_DOCS_COLLECTION_NAME"] = _TEST_COLLECTION

from app.rag import embeddings as embedding_module


def _test_embed(texts: list[str], dimensions: int = 384) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dimensions
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


embedding_module.embed_texts = _test_embed
embedding_module.embed_query = lambda q: _test_embed([q])[0]

from app.orchestrator.offline_docs_store import (
    _get_db_connection,
    _get_milvus_client,
    ingest_offline_docs,
    search_offline_docs,
)


def test_offline_docs_store() -> bool:
    print("=" * 70)
    print("OFFLINE DOCS STORE TEST")
    print("=" * 70)

    temp_dir = Path(tempfile.mkdtemp(prefix="offline-docs-test-"))
    dept_dir = temp_dir / _TEST_DEPARTMENT
    dept_dir.mkdir(parents=True)

    (dept_dir / "onboarding.md").write_text(
        "# Onboarding\n\nNew hires get a laptop and a company badge in week one.",
        encoding="utf-8",
    )
    (dept_dir / "security.md").write_text(
        "# Security\n\nAll staff must enable two-factor authentication within 24 hours of joining.",
        encoding="utf-8",
    )

    conn = None
    try:
        print("\n[1/3] Ingesting 2 markdown files...")
        stats = ingest_offline_docs(temp_dir)
        assert stats["articles_inserted"] == 2, f"expected 2 articles, got {stats}"
        assert stats["chunks_indexed"] >= 2, f"expected >=2 chunks, got {stats}"
        print(f"   OK: {stats}")

        print("\n[2/3] Verifying MariaDB ground truth...")
        conn = _get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT a.id, a.title, a.breadcrumbs, d.name AS department "
                "FROM articles a JOIN departments d ON a.dept_id = d.id "
                "WHERE d.name = %s",
                (_TEST_DEPARTMENT,),
            )
            rows = cursor.fetchall()
        assert len(rows) == 2, f"expected 2 MariaDB rows, got {rows}"
        titles = {row["title"] for row in rows}
        assert titles == {"onboarding", "security"}, f"unexpected titles: {titles}"
        print(f"   OK: {rows}")

        print("\n[3/3] Searching Milvus + verifying article_id back-reference...")
        hits = search_offline_docs("How do new hires get set up with equipment?", top_k=3)
        assert len(hits) > 0, "expected at least one hit"
        top_hit = hits[0]
        assert top_hit["department"] == _TEST_DEPARTMENT, f"unexpected department: {top_hit}"
        matching_row = next((r for r in rows if r["id"] == top_hit["article_id"]), None)
        assert matching_row is not None, f"article_id {top_hit['article_id']} not found in {rows}"
        print(f"   OK: top hit article_id={top_hit['article_id']} -> MariaDB title='{matching_row['title']}'")

        print("\nPASSED: offline_docs_store.py works end to end.")
        return True

    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return False

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

        client = _get_milvus_client()
        if client.has_collection(_TEST_COLLECTION):
            client.drop_collection(_TEST_COLLECTION)
            print(f"[cleanup] dropped Milvus collection {_TEST_COLLECTION}")

        if conn is None:
            conn = _get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE a FROM articles a JOIN departments d ON a.dept_id = d.id WHERE d.name = %s",
                (_TEST_DEPARTMENT,),
            )
            cursor.execute("DELETE FROM departments WHERE name = %s", (_TEST_DEPARTMENT,))
        conn.commit()
        conn.close()
        print(f"[cleanup] removed MariaDB test rows for department '{_TEST_DEPARTMENT}'")


if __name__ == "__main__":
    success = test_offline_docs_store()
    sys.exit(0 if success else 1)