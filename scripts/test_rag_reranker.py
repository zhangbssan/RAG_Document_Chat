#!/usr/bin/env python3
"""Smoke test RAG retrieval before and after lexical RRF reranking."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import logging
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
SAMPLE_DOCS_DIR = REPO_ROOT / "sample_docs"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="rag-reranker-test-"))
atexit.register(shutil.rmtree, TEST_DATA_DIR, ignore_errors=True)

os.environ["CHROMA_DIR"] = str(TEST_DATA_DIR / "chroma")
os.environ["UPLOAD_DIR"] = str(TEST_DATA_DIR / "uploads")

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

try:
    import posthog

    posthog.disabled = True
    posthog.capture = lambda *args, **kwargs: None
except ImportError:
    pass


from app.config import RERANK_TOP_K, TOP_K
from app.rag import embeddings as embedding_module


def _test_embed_texts(texts: list[str], dimensions: int = 128) -> list[list[float]]:
    """Deterministic embeddings for repeatable local smoke tests."""
    vectors: list[list[float]] = []

    for text in texts:
        vector = [0.0] * dimensions
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        vectors.append([value / norm for value in vector])

    return vectors


embedding_module.embed_texts = _test_embed_texts

from app.rag.chunker import build_chunks_from_pages
from app.rag.pdf_loader import pdf_extraction
from app.rag.retriever import retrieve_chunks
from app.rag.vector_store import add_chunks, query_chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index sample_docs and compare dense retrieval vs reranked retrieval."
    )
    parser.add_argument(
        "--query",
        default="annual leave social insurance employee benefits",
        help="Query to run against the sample document index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Dense retrieval top-k before reranking.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=RERANK_TOP_K,
        help="Final top-k after reranking.",
    )
    args = parser.parse_args()

    if not SAMPLE_DOCS_DIR.exists():
        print(f"sample_docs not found: {SAMPLE_DOCS_DIR}")
        return 1

    pdf_files = sorted(SAMPLE_DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {SAMPLE_DOCS_DIR}")
        return 1

    print("Indexing sample_docs")
    print(f"Temporary Chroma directory: {os.environ['CHROMA_DIR']}")

    all_chunks = []
    for pdf_path in pdf_files:
        pages = pdf_extraction(file_name=pdf_path.name, pdf_path=pdf_path)
        chunks = build_chunks_from_pages(pages)
        all_chunks.extend(chunks)
        print(f"- {pdf_path.name}: {len(pages)} pages, {len(chunks)} chunks")

    added = add_chunks(all_chunks)
    print(f"\nIndexed chunks: {added}")
    print(f"Query: {args.query!r}")
    print(f"Dense top_k: {args.top_k}")
    print(f"Rerank top_k: {args.rerank_top_k}")

    dense_results = query_chunks(query=args.query, top_k=args.top_k)
    reranked_results = retrieve_chunks(
        query=args.query,
        top_k=args.top_k,
        rerank_top_k=args.rerank_top_k,
    )

    _print_results(
        title="Results without rerank",
        results=dense_results,
        show_rerank_fields=False,
    )
    _print_results(
        title="Results with rerank",
        results=reranked_results,
        show_rerank_fields=True,
    )

    return 0


def _print_results(
    title: str,
    results: list[dict[str, Any]],
    show_rerank_fields: bool,
) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not results:
        print("No results.")
        return

    for index, chunk in enumerate(results, start=1):
        metadata = chunk.get("metadata") or {}
        document = metadata.get("document_name", "?")
        page = metadata.get("page", "?")
        chunk_index = metadata.get("chunk_index", "?")
        text = " ".join(str(chunk.get("text", "")).split())
        preview = text[:180] + ("..." if len(text) > 180 else "")

        print(f"\n[{index}] {document} | page={page} | chunk={chunk_index}")
        print(f"    id: {chunk.get('id', '?')}")
        print(f"    score: {_format_score(chunk.get('score'))}")

        if show_rerank_fields:
            print(
                "    ranks: "
                f"dense={chunk.get('dense_rank', '?')} "
                f"lexical={chunk.get('lexical_rank', '?')}"
            )
            print(
                "    rerank: "
                f"rrf={_format_score(chunk.get('rerank_score'))} "
                f"vector={_format_score(chunk.get('vector_score'))} "
                f"overlap={chunk.get('keyword_overlap', '?')}"
            )
            print(f"    matched_keywords: {chunk.get('matched_keywords', [])}")

        print(f"    text: {preview}")


def _format_score(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{value:.6f}"
    return "?"


if __name__ == "__main__":
    raise SystemExit(main())
