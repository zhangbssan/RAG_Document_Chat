import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { DocumentRecord } from "@/types";

interface DocumentListProps {
  documents: DocumentRecord[];
  onDelete: (fileHash: string) => void;
  deletingHash: string | null;
}

export function DocumentList({ documents, onDelete, deletingHash }: DocumentListProps) {
  if (documents.length === 0) {
    return <p className="text-xs text-muted-foreground">No documents indexed yet.</p>;
  }

  return (
    <div className="flex flex-col gap-1">
      {documents.map((document) => (
        <div
          key={document.file_hash}
          className="flex items-center justify-between gap-2 rounded-md px-2 py-1 text-xs hover:bg-muted"
        >
          <span className="min-w-0 truncate" title={document.document_name}>
            📄 {document.document_name} · Pages {document.pages} · Chunks {document.chunks}
          </span>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 shrink-0"
            disabled={deletingHash === document.file_hash}
            onClick={() => onDelete(document.file_hash)}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      ))}
    </div>
  );
}
