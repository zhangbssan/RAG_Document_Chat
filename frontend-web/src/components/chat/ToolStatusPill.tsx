import { Search } from "lucide-react";

interface ToolStatusPillProps {
  query: string;
}

export function ToolStatusPill({ query }: ToolStatusPillProps) {
  return (
    <div className="mb-2 flex items-center gap-2 rounded-full bg-primary/10 px-3 py-1 text-xs text-primary">
      <Search className="h-3 w-3 animate-pulse" />
      <span>Searching uploaded documents: “{query}”</span>
    </div>
  );
}
