export interface Source {
  text: string;
  document: string;
  page: number | string;
  chunk: number | string;
  score: number | null;
  pages: number[] | null;
  link: string | null;
}

export interface ChatRequest {
  question: string;
  conversation_id: string;
  top_k?: number;
  openai_api_key?: string;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
}

export type ChatStreamEvent =
  | { type: "tool_start"; tool: string; query: string }
  | { type: "sources"; data: Source[] }
  | { type: "token"; data: string }
  | { type: "notice"; data: string }
  | { type: "error"; data: string }
  | { type: "done" };
