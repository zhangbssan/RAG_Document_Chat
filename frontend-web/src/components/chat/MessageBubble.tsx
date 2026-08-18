import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import { ToolStatusPill } from "./ToolStatusPill";
import { CitationCard } from "./CitationCard";
import type { Source } from "@/types";

export interface MessageBubbleProps {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  toolQuery?: string | null;
  notice?: string | null;
  error?: string | null;
}

export function MessageBubble({ role, content, sources = [], toolQuery, notice, error }: MessageBubbleProps) {
  const isUser = role === "user";

  return (
    <div className={cn("flex w-full flex-col", isUser ? "items-end" : "items-start")}>
      {toolQuery && <ToolStatusPill query={toolQuery} />}
      <div
        className={cn(
          "max-w-[75%] rounded-lg px-4 py-3 text-sm",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted text-foreground",
          error && "border border-destructive text-destructive"
        )}
      >
        {notice && <p className="mb-2 text-xs text-amber-600 dark:text-amber-400">{notice}</p>}
        <div className="prose prose-sm max-w-none dark:prose-invert">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{error ? `❌ ${error}` : content || " "}</ReactMarkdown>
        </div>
      </div>
      {sources.length > 0 && (
        <details className="mt-2 w-full max-w-[75%]">
          <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
            View sources ({sources.length})
          </summary>
          <div className="mt-2 flex flex-col gap-2">
            {sources.map((source, index) => (
              <CitationCard key={`${source.chunk}-${index}`} source={source} index={index + 1} />
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
