import { useCallback, useRef, useState } from "react";
import { streamChat } from "@/api/chat";
import type { ChatRequest, ChatStreamEvent, Source } from "@/types";

export interface StreamingDraft {
  toolQuery: string | null;
  sources: Source[];
  text: string;
  notice: string | null;
  error: string | null;
}

export interface ChatStreamResult {
  text: string;
  sources: Source[];
}

const EMPTY_DRAFT: StreamingDraft = {
  toolQuery: null,
  sources: [],
  text: "",
  notice: null,
  error: null,
};

export function useChatStream(onDone: (result: ChatStreamResult) => void) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [draft, setDraft] = useState<StreamingDraft>(EMPTY_DRAFT);
  const controllerRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setIsStreaming(false);
  }, []);

  const send = useCallback(
    (question: string, conversationId: string, apiKey?: string) => {
      const controller = new AbortController();
      controllerRef.current = controller;
      setDraft(EMPTY_DRAFT);
      setIsStreaming(true);

      const request: ChatRequest = {
        question,
        conversation_id: conversationId,
        openai_api_key: apiKey || undefined,
      };

      let text = "";
      let sources: Source[] = [];

      streamChat(
        request,
        (event: ChatStreamEvent) => {
          if (event.type === "tool_start") {
            setDraft((prev) => ({ ...prev, toolQuery: event.query }));
          } else if (event.type === "sources") {
            sources = event.data;
            setDraft((prev) => ({ ...prev, sources: event.data, toolQuery: null }));
          } else if (event.type === "token") {
            text += event.data;
            setDraft((prev) => ({ ...prev, text }));
          } else if (event.type === "notice") {
            setDraft((prev) => ({ ...prev, notice: event.data }));
          } else if (event.type === "error") {
            setDraft((prev) => ({ ...prev, error: event.data }));
          } else if (event.type === "done") {
            setIsStreaming(false);
            controllerRef.current = null;
            onDone({ text, sources });
          }
        },
        controller.signal
      ).catch((error: unknown) => {
        if (controller.signal.aborted) return;
        const message = error instanceof Error ? error.message : "Stream failed";
        setDraft((prev) => ({ ...prev, error: message }));
        setIsStreaming(false);
      });
    },
    [onDone]
  );

  return { isStreaming, draft, send, cancel };
}
