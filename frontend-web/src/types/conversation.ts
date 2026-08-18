import type { Source } from "./chat";

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
  sources: Source[];
}

export interface ConversationHistoryResponse {
  conversation: ConversationSummary;
  messages: ConversationMessage[];
}

export interface ConversationListResponse {
  conversations: ConversationSummary[];
}
