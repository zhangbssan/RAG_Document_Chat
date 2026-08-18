import { apiFetch } from "./client";
import type {
  ConversationHistoryResponse,
  ConversationListResponse,
  ConversationSummary,
} from "@/types";

export async function listConversations(): Promise<ConversationSummary[]> {
  const result = await apiFetch<ConversationListResponse>("/api/conversations");
  return result.conversations;
}

export async function getConversationHistory(
  conversationId: string
): Promise<ConversationHistoryResponse> {
  return apiFetch<ConversationHistoryResponse>(
    `/api/conversations/${conversationId}/messages`
  );
}

export async function renameConversation(
  conversationId: string,
  title: string
): Promise<ConversationSummary> {
  return apiFetch<ConversationSummary>(`/api/conversations/${conversationId}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export async function deleteConversation(
  conversationId: string
): Promise<{ conversation_id: string; deleted: boolean }> {
  return apiFetch(`/api/conversations/${conversationId}`, { method: "DELETE" });
}
