import { useRef, useState, type DragEvent } from "react";
import { UploadCloud } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface UploadDropzoneProps {
  isUploading: boolean;
  onFilesSelected: (files: File[]) => void;
}

export function UploadDropzone({ isUploading, onFilesSelected }: UploadDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const files = Array.from(event.dataTransfer.files).filter((file) => file.type === "application/pdf");
    if (files.length) onFilesSelected(files);
  };

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={cn(
        "flex flex-col items-center gap-2 rounded-md border-2 border-dashed border-border p-4 text-center text-xs text-muted-foreground",
        isDragging && "border-primary bg-primary/5"
      )}
    >
      <UploadCloud className="h-6 w-6" />
      <p>Drag PDF files here, or</p>
      <Button
        type="button"
        variant="secondary"
        size="sm"
        disabled={isUploading}
        onClick={() => inputRef.current?.click()}
      >
        {isUploading ? "Uploading…" : "Choose files"}
      </Button>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        multiple
        className="hidden"
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          if (files.length) onFilesSelected(files);
          event.target.value = "";
        }}
      />
    </div>
  );
}
