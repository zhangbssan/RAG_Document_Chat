import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";
import { ConversationItem } from "./ConversationItem";
import {
  useConversations,
  useDeleteConversation,
  useRenameConversation,
} from "@/hooks/useConversations";
import { useUiStore } from "@/store/uiStore";

interface ConversationListProps {
  onNewChat: () => void;
}

export function ConversationList({ onNewChat }: ConversationListProps) {
  const { data: conversations = [], isLoading } = useConversations();
  const activeConversationId = useUiStore((state) => state.activeConversationId);
  const setActiveConversationId = useUiStore((state) => state.setActiveConversationId);
  const renameMutation = useRenameConversation();
  const deleteMutation = useDeleteConversation();

  return (
    <div className="flex flex-col gap-2">
      <Button onClick={onNewChat} className="w-full justify-start gap-2">
        <Plus className="h-4 w-4" /> New chat
      </Button>

      {isLoading && <p className="text-xs text-muted-foreground">Loading conversations…</p>}
      {!isLoading && conversations.length === 0 && (
        <p className="text-xs text-muted-foreground">
          Start a new chat to create your first saved conversation.
        </p>
      )}

      <div className="flex flex-col gap-1">
        {conversations.map((conversation) => (
          <ConversationItem
            key={conversation.id}
            conversation={conversation}
            active={conversation.id === activeConversationId}
            onSelect={() => setActiveConversationId(conversation.id)}
            onRename={(title) => renameMutation.mutate({ id: conversation.id, title })}
            onDelete={() => {
              deleteMutation.mutate(conversation.id, {
                onSuccess: () => {
                  if (conversation.id === activeConversationId) {
                    const remaining = conversations.filter((item) => item.id !== conversation.id);
                    setActiveConversationId(remaining[0]?.id ?? crypto.randomUUID());
                  }
                },
              });
            }}
          />
        ))}
      </div>
    </div>
  );
}
