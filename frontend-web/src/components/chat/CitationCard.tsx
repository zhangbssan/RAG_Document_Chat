import { ExternalLink } from "lucide-react";
import type { Source } from "@/types";

interface CitationCardProps {
  source: Source;
  index: number;
}

export function CitationCard({ source, index }: CitationCardProps) {
  const pageLabel =
    source.pages && source.pages.length > 1
      ? `${source.pages[0]}-${source.pages[source.pages.length - 1]}`
      : String(source.page);
  const excerpt = source.text.length > 400 ? `${source.text.slice(0, 400)}…` : source.text;

  return (
    <div className="rounded-md border border-border bg-card p-3 text-xs">
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="font-medium">
          Source {index}: {source.document} · Page {pageLabel} · Chunk {source.chunk}
        </span>
        {source.score !== null && (
          <span className="text-muted-foreground">score {source.score.toFixed(4)}</span>
        )}
      </div>
      <blockquote className="border-l-2 border-border pl-2 text-muted-foreground">{excerpt}</blockquote>
      {source.link && (
        <a
          href={source.link}
          target="_blank"
          rel="noreferrer"
          className="mt-1 flex items-center gap-1 text-primary hover:underline"
        >
          <ExternalLink className="h-3 w-3" /> {source.link}
        </a>
      )}
    </div>
  );
}
