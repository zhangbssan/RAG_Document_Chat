from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

from app.agent.runtime import get_agent_runtime
from app.schemas import (
    ConversationDeleteResponse,
    ConversationHistoryResponse,
    ConversationListResponse,
    ConversationRenameRequest,
    ConversationSummary,
)

router = APIRouter()


@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations(request: Request) -> ConversationListResponse:
    runtime = get_agent_runtime(request)
    return ConversationListResponse(conversations=runtime.conversations.list())


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=ConversationHistoryResponse,
)
def conversation_history(conversation_id: UUID, request: Request) -> ConversationHistoryResponse:
    runtime = get_agent_runtime(request)
    conversation_id_str = str(conversation_id)
    conversation = runtime.conversations.get(conversation_id_str)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationHistoryResponse(
        conversation=ConversationSummary(**conversation),
        messages=runtime.get_display_history(conversation_id_str),
    )


@router.patch("/conversations/{conversation_id}", response_model=ConversationSummary)
def rename_conversation(
    conversation_id: UUID,
    payload: ConversationRenameRequest,
    request: Request,
) -> ConversationSummary:
    runtime = get_agent_runtime(request)
    conversation = runtime.conversations.rename(str(conversation_id), payload.title)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationSummary(**conversation)


@router.delete(
    "/conversations/{conversation_id}",
    response_model=ConversationDeleteResponse,
)
def delete_conversation(conversation_id: UUID, request: Request) -> ConversationDeleteResponse:
    runtime = get_agent_runtime(request)
    conversation_id_str = str(conversation_id)
    deleted = runtime.conversations.delete(conversation_id_str)
    runtime.delete_conversation_state(conversation_id_str)
    return ConversationDeleteResponse(conversation_id=conversation_id_str, deleted=deleted)
