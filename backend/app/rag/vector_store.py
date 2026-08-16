from __future__ import annotations

from pymilvus import MilvusClient

from app.config import MILVUS_HOST, MILVUS_PORT, REALTIME_PDF_COLLECTION_NAME
from app.rag import embeddings
from app.rag.hybrid_schema import build_realtime_pdf_index_params, build_realtime_pdf_schema
from app.rag.types import Chunk

_client: MilvusClient | None = None


def get_collection() -> MilvusClient:
    """Return the shared MilvusClient, creating the realtime PDF collection if needed."""
    global _client
    if _client is None:
        _client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

    if not _client.has_collection(REALTIME_PDF_COLLECTION_NAME):
        dim = len(embeddings.embed_query("dimension probe"))

        schema = build_realtime_pdf_schema(_client, dim=dim)
        _client.create_collection(collection_name=REALTIME_PDF_COLLECTION_NAME, schema=schema)

        index_params = build_realtime_pdf_index_params(_client)
        _client.create_index(REALTIME_PDF_COLLECTION_NAME, index_params)
        _client.load_collection(REALTIME_PDF_COLLECTION_NAME)

    return _client


def indexed_file_hashes() -> set[str]:
    client = get_collection()
    rows = client.query(
        collection_name=REALTIME_PDF_COLLECTION_NAME,
        filter="",
        output_fields=["file_hash"],
        limit=16384,
    )
    return {row["file_hash"] for row in rows if row.get("file_hash")}


def add_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0

    client = get_collection()
    texts = [chunk.text for chunk in chunks]
    vectors = embeddings.embed_texts(texts)

    data = [
        {"text": chunk.text, "embedding": vector, **chunk.metadata}
        for chunk, vector in zip(chunks, vectors)
    ]
    client.insert(collection_name=REALTIME_PDF_COLLECTION_NAME, data=data)
    client.flush(REALTIME_PDF_COLLECTION_NAME)
    return len(chunks)


def delete_document(file_hash: str) -> int:
    client = get_collection()
    matches = client.query(
        collection_name=REALTIME_PDF_COLLECTION_NAME,
        filter=f'file_hash == "{file_hash}"',
        output_fields=["id"],
    )
    ids = [row["id"] for row in matches]

    if not ids:
        return 0

    client.delete(collection_name=REALTIME_PDF_COLLECTION_NAME, ids=ids)
    client.flush(REALTIME_PDF_COLLECTION_NAME)
    return len(ids)


_SEARCH_OUTPUT_FIELDS = ["text", "document_name", "file_hash", "page", "chunk_index", "chunk_seq"]


def _hits_from_search_results(results) -> list[dict]:
    """Flatten a Milvus search() response (list[list[hit]]) into the shared hit
    shape used by both dense (query_chunks) and sparse (sparse_search) search."""
    retrieved: list[dict] = []
    for hits in results:
        for hit in hits:
            entity = hit.get("entity", {})
            distance = hit.get("distance")
            retrieved.append(
                {
                    "id": hit.get("id"),
                    "text": entity.get("text"),
                    "metadata": {
                        "document_name": entity.get("document_name"),
                        "file_hash": entity.get("file_hash"),
                        "page": entity.get("page"),
                        "chunk_index": entity.get("chunk_index"),
                        "chunk_seq": entity.get("chunk_seq"),
                    },
                    "distance": float(distance) if distance is not None else None,
                    "score": float(distance) if distance is not None else 0.0,
                }
            )
    return retrieved


def query_chunks(query: str, top_k: int = 5) -> list[dict]:
    """Dense (embedding) search."""
    if not query.strip():
        raise ValueError("Query must not be empty.")

    client = get_collection()
    stats = client.get_collection_stats(REALTIME_PDF_COLLECTION_NAME)
    if int(stats.get("row_count", 0)) == 0:
        return []

    query_vector = embeddings.embed_query(query)
    results = client.search(
        collection_name=REALTIME_PDF_COLLECTION_NAME,
        data=[query_vector],
        anns_field="embedding",
        limit=top_k,
        output_fields=_SEARCH_OUTPUT_FIELDS,
    )
    return _hits_from_search_results(results)


def sparse_search(query: str, top_k: int = 5) -> list[dict]:
    """BM25 full-text search via Milvus's native `sparse_vector` Function field
    (see hybrid_schema.py). Milvus tokenizes and BM25-scores `query` itself —
    unlike query_chunks(), no local embedding call is made here."""
    if not query.strip():
        raise ValueError("Query must not be empty.")

    client = get_collection()
    stats = client.get_collection_stats(REALTIME_PDF_COLLECTION_NAME)
    if int(stats.get("row_count", 0)) == 0:
        return []

    results = client.search(
        collection_name=REALTIME_PDF_COLLECTION_NAME,
        data=[query],
        anns_field="sparse_vector",
        limit=top_k,
        output_fields=_SEARCH_OUTPUT_FIELDS,
    )
    return _hits_from_search_results(results)


def get_chunks_by_seq(file_hash: str, chunk_seqs: list[int]) -> list[dict]:
    """Fetch specific chunks of one document by chunk_seq — used to build an
    anchor's ±window context. Returns flat rows (same convention as
    indexed_file_hashes()/list_documents()), not the nested search() shape."""
    if not chunk_seqs:
        return []

    client = get_collection()
    seq_list = ",".join(str(seq) for seq in chunk_seqs)
    return client.query(
        collection_name=REALTIME_PDF_COLLECTION_NAME,
        filter=f'file_hash == "{file_hash}" && chunk_seq in [{seq_list}]',
        output_fields=["id", "text", "document_name", "file_hash", "page", "chunk_index", "chunk_seq"],
        limit=len(chunk_seqs),
        consistency_level="Strong",
    )


def list_documents() -> list[dict]:
    client = get_collection()
    rows = client.query(
        collection_name=REALTIME_PDF_COLLECTION_NAME,
        filter="",
        output_fields=["document_name", "file_hash", "page"],
        limit=16384,
    )

    documents: dict[str, dict] = {}
    for row in rows:
        document_name = row.get("document_name")
        if not document_name:
            continue

        if document_name not in documents:
            documents[document_name] = {
                "document_name": document_name,
                "file_hash": row.get("file_hash"),
                "pages": set(),
                "chunks": 0,
            }

        page = row.get("page")
        if page is not None:
            documents[document_name]["pages"].add(page)
        documents[document_name]["chunks"] += 1

    return [
        {
            "document_name": doc["document_name"],
            "file_hash": doc["file_hash"],
            "pages": len(doc["pages"]),
            "chunks": doc["chunks"],
        }
        for doc in documents.values()
    ]