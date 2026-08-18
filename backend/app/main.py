from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.runtime import AgentRuntime
from app.api import chat, conversations, documents, evaluation, upload
from app.config import CHAT_DB_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = AgentRuntime(CHAT_DB_PATH)
    runtime.start()
    app.state.agent_runtime = runtime
    try:
        yield
    finally:
        runtime.close()


app = FastAPI(title="RAG Document Chat API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(conversations.router, prefix="/api", tags=["conversations"])
app.include_router(documents.router, prefix="/api", tags=["documents"])
app.include_router(evaluation.router, prefix="/api", tags=["evaluation"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
