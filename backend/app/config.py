from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", DATA_DIR / "uploads"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", DATA_DIR / "chroma"))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_chunks")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TOP_K = int(os.getenv("TOP_K", "5"))
SPARSE_TOP_K = int(os.getenv("SPARSE_TOP_K", str(TOP_K)))
ANCHOR_TOP_N = int(os.getenv("ANCHOR_TOP_N", "2"))
ANCHOR_WINDOW = int(os.getenv("ANCHOR_WINDOW", "1"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "950"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))

MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
REALTIME_PDF_COLLECTION_NAME = os.getenv("REALTIME_PDF_COLLECTION_NAME", "realtime_pdf_collection")
OFFLINE_DOCS_COLLECTION_NAME = os.getenv("OFFLINE_DOCS_COLLECTION_NAME", "offline_docs_collection")

MARIADB_HOST = os.getenv("MARIADB_HOST", "127.0.0.1")
MARIADB_PORT = int(os.getenv("MARIADB_PORT", "3307"))
MARIADB_USER = os.getenv("MARIADB_USER", "root")
MARIADB_PASSWORD = os.getenv("MARIADB_PASSWORD", "offerishere")
MARIADB_DATABASE = os.getenv("MARIADB_DATABASE", "wiki_db")
MARIADB_SQLALCHEMY_URI = (
    f"mysql+pymysql://{MARIADB_USER}:{MARIADB_PASSWORD}@{MARIADB_HOST}:{MARIADB_PORT}/{MARIADB_DATABASE}"
)
