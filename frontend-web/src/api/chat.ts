import { ApiError, getApiBaseUrl } from "./client";
import type { ChatRequest, ChatResponse, ChatStreamEvent } from "@/types";

export async function postChat(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw new ApiError(response.status, response.statusText);
  }
  return (await response.json()) as ChatResponse;
}

export async function streamChat(
  request: ChatRequest,
  onEvent: (event: ChatStreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${getApiBaseUrl()}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new ApiError(response.status, response.statusText || "Stream request failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        onEvent(JSON.parse(trimmed) as ChatStreamEvent);
      } catch {
        // ignore malformed NDJSON lines
      }
    }
  }

  const remaining = buffer.trim();
  if (remaining) {
    try {
      onEvent(JSON.parse(remaining) as ChatStreamEvent);
    } catch {
      // ignore
    }
  }
}
