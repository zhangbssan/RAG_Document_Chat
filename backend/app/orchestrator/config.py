import os
from dotenv import load_dotenv
import pymysql
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FILE_NAME = "mock_inventory.json"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)
LEAVE_POLICY_FILE_NAME = "leave_policy.pdf"
LEAVE_POLICY_FILE_PATH = os.path.join(DATA_DIR, LEAVE_POLICY_FILE_NAME)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "rag_json_demo_collection"
LEAVE_POLICY_COLLECTION_NAME = "rag_leave_policy_collection"

# Random JSON demo (sparse + dense hybrid ingest smoke test)
RANDOM_JSON_DEMO_FILE_NAME = "random_inventory_demo.json"
RANDOM_JSON_DEMO_PATH = os.path.join(DATA_DIR, RANDOM_JSON_DEMO_FILE_NAME)
RANDOM_JSON_COLLECTION_NAME = "rag_random_json_hybrid_collection"

def get_db_connection():
    return pymysql.connect(
        host='127.0.0.1',
        port=3307,
        user='root',
        password='offerishere',  
        database='wiki_db',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
