import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { MessageList } from "@/components/chat/MessageList";
import { ChatInput } from "@/components/chat/ChatInput";
import { useChatStream } from "@/hooks/useChatStream";
import { conversationsQueryKey, useConversationHistory } from "@/hooks/useConversations";
import { useUiStore } from "@/store/uiStore";
import type { ConversationMessage } from "@/types";

export function ChatPage() {
  const activeConversationId = useUiStore((state) => state.activeConversationId);
  const apiKey = useUiStore((state) => state.apiKey);
  const queryClient = useQueryClient();
  const { data: history } = useConversationHistory(activeConversationId);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);

  useEffect(() => {
    setMessages(history?.messages ?? []);
  }, [activeConversationId, history]);

  const { isStreaming, draft, send, cancel } = useChatStream(({ text, sources }) => {
    setMessages((prev) => [...prev, { role: "assistant", content: text, sources }]);
    queryClient.invalidateQueries({ queryKey: conversationsQueryKey });
  });

  useEffect(() => {
    return () => cancel();
  }, [activeConversationId, cancel]);

  const handleSend = (question: string) => {
    if (!activeConversationId) return;
    setMessages((prev) => [...prev, { role: "user", content: question, sources: [] }]);
    send(question, activeConversationId, apiKey);
  };

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1">
        <MessageList messages={messages} isStreaming={isStreaming} draft={draft} />
      </div>
      <ChatInput disabled={isStreaming || !activeConversationId} onSend={handleSend} />
    </div>
  );
}
