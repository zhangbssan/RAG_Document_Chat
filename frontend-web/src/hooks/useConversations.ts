import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  deleteConversation,
  getConversationHistory,
  listConversations,
  renameConversation,
} from "@/api/conversations";

export const conversationsQueryKey = ["conversations"] as const;
export const conversationHistoryQueryKey = (id: string) => ["conversation", id] as const;

export function useConversations() {
  return useQuery({
    queryKey: conversationsQueryKey,
    queryFn: listConversations,
  });
}

export function useConversationHistory(conversationId: string | null) {
  return useQuery({
    queryKey: conversationId ? conversationHistoryQueryKey(conversationId) : ["conversation", "none"],
    queryFn: () => getConversationHistory(conversationId as string),
    enabled: Boolean(conversationId),
  });
}

export function useRenameConversation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) => renameConversation(id, title),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: conversationsQueryKey });
    },
  });
}

export function useDeleteConversation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => deleteConversation(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: conversationsQueryKey });
    },
  });
}
