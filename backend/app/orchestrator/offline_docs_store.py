from __future__ import annotations

import os
import re
from pathlib import Path

import pymysql
from pymilvus import DataType, MilvusClient

from app.config import (
    MARIADB_DATABASE,
    MARIADB_HOST,
    MARIADB_PASSWORD,
    MARIADB_PORT,
    MARIADB_USER,
    MILVUS_HOST,
    MILVUS_PORT,
    OFFLINE_DOCS_COLLECTION_NAME,
)
from app.rag import embeddings
from app.rag.chunker import chunk_text

_milvus_client: MilvusClient | None = None


def _get_db_connection():
    return pymysql.connect(
        host=MARIADB_HOST,
        port=MARIADB_PORT,
        user=MARIADB_USER,
        password=MARIADB_PASSWORD,
        database=MARIADB_DATABASE,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def _get_milvus_client() -> MilvusClient:
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

    if not _milvus_client.has_collection(OFFLINE_DOCS_COLLECTION_NAME):
        dim = len(embeddings.embed_query("dimension probe"))

        schema = _milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
        _milvus_client.create_collection(collection_name=OFFLINE_DOCS_COLLECTION_NAME, schema=schema)

        index_params = _milvus_client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        _milvus_client.create_index(OFFLINE_DOCS_COLLECTION_NAME, index_params)
        _milvus_client.load_collection(OFFLINE_DOCS_COLLECTION_NAME)

    return _milvus_client


def clean_markdown_content(raw_text: str) -> str:
    cleaned_text = re.sub(r"!\[.*?\]\(.*?\)", "", raw_text)
    cleaned_text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", cleaned_text)
    return cleaned_text.strip()


def ensure_tables() -> None:
    conn = _get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS departments (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(50) UNIQUE NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    content MEDIUMTEXT NOT NULL,
                    breadcrumbs VARCHAR(500) NOT NULL,
                    dept_id INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dept_id) REFERENCES departments(id)
                )
                """
            )
        conn.commit()
    finally:
        conn.close()


def ingest_offline_docs(base_folder: Path) -> dict:
    """Ingest every .md file under base_folder: MariaDB ground truth + Milvus chunks."""
    ensure_tables()
    milvus_client = _get_milvus_client()

    conn = _get_db_connection()
    articles_inserted = 0
    chunks_indexed = 0

    try:
        with conn.cursor() as cursor:
            for root, _dirs, files in os.walk(base_folder):
                for file_name in files:
                    if not file_name.endswith(".md"):
                        continue

                    full_path = Path(root) / file_name
                    rel_path = Path(root).relative_to(base_folder)

                    if str(rel_path) == ".":
                        department = "General"
                        breadcrumbs = "General"
                    else:
                        parts = rel_path.parts
                        department = parts[0]
                        breadcrumbs = " > ".join(parts)

                    cursor.execute("INSERT IGNORE INTO departments (name) VALUES (%s)", (department,))
                    cursor.execute("SELECT id FROM departments WHERE name = %s", (department,))
                    dept_id = cursor.fetchone()["id"]

                    raw_content = full_path.read_text(encoding="utf-8")
                    cleaned_content = clean_markdown_content(raw_content)
                    title = full_path.stem.replace("_", " ")

                    cursor.execute(
                        "INSERT INTO articles (title, content, breadcrumbs, dept_id) VALUES (%s, %s, %s, %s)",
                        (title, cleaned_content, breadcrumbs, dept_id),
                    )
                    article_id = cursor.lastrowid
                    articles_inserted += 1

                    chunk_texts = chunk_text(cleaned_content)
                    if not chunk_texts:
                        continue

                    vectors = embeddings.embed_texts(chunk_texts)
                    data = [
                        {"text": chunk, "embedding": vector, "article_id": article_id, "department": department}
                        for chunk, vector in zip(chunk_texts, vectors)
                    ]
                    milvus_client.insert(collection_name=OFFLINE_DOCS_COLLECTION_NAME, data=data)
                    chunks_indexed += len(data)

        conn.commit()
    finally:
        conn.close()

    milvus_client.flush(OFFLINE_DOCS_COLLECTION_NAME)
    return {"articles_inserted": articles_inserted, "chunks_indexed": chunks_indexed}


def search_offline_docs(query: str, top_k: int = 5) -> list[dict]:
    if not query.strip():
        raise ValueError("Query must not be empty.")

    client = _get_milvus_client()
    stats = client.get_collection_stats(OFFLINE_DOCS_COLLECTION_NAME)
    if int(stats.get("row_count", 0)) == 0:
        return []

    query_vector = embeddings.embed_query(query)
    results = client.search(
        collection_name=OFFLINE_DOCS_COLLECTION_NAME,
        data=[query_vector],
        anns_field="embedding",
        limit=top_k,
        output_fields=["text", "article_id", "department"],
    )

    hits: list[dict] = []
    for result_group in results:
        for hit in result_group:
            entity = hit.get("entity", {})
            distance = hit.get("distance")
            hits.append(
                {
                    "text": entity.get("text"),
                    "article_id": entity.get("article_id"),
                    "department": entity.get("department"),
                    "score": float(distance) if distance is not None else 0.0,
                }
            )

    return hits