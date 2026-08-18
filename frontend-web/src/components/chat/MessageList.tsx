import { useEffect, useRef } from "react";
import { MessageBubble } from "./MessageBubble";
import type { StreamingDraft } from "@/hooks/useChatStream";
import type { ConversationMessage } from "@/types";

interface MessageListProps {
  messages: ConversationMessage[];
  isStreaming: boolean;
  draft: StreamingDraft;
}

export function MessageList({ messages, isStreaming, draft }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, draft.text]);

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-6">
      {messages.map((message, index) => (
        <MessageBubble
          key={index}
          role={message.role}
          content={message.content}
          sources={message.sources}
        />
      ))}
      {isStreaming && (
        <MessageBubble
          role="assistant"
          content={draft.text}
          sources={draft.sources}
          toolQuery={draft.toolQuery}
          notice={draft.notice}
          error={draft.error}
        />
      )}
      <div ref={bottomRef} />
    </div>
  );
}
