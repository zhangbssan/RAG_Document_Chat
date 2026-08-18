import { useState } from "react";
import { toast } from "@/components/ui/toast";
import { UploadDropzone } from "./UploadDropzone";
import { DocumentList } from "./DocumentList";
import { useDocuments, useDeleteDocument, useUploadPdfs } from "@/hooks/useDocuments";

export function DocumentPanel() {
  const { data, isLoading } = useDocuments();
  const uploadMutation = useUploadPdfs();
  const deleteMutation = useDeleteDocument();
  const [deletingHash, setDeletingHash] = useState<string | null>(null);

  const handleUpload = (files: File[]) => {
    uploadMutation.mutate(files, {
      onSuccess: (result) => {
        result.messages.forEach((message) => toast.add({ description: message }));
      },
      onError: (error) => {
        toast.add({ description: `Indexing failed: ${error.message}`, type: "error" });
      },
    });
  };

  const handleDelete = (fileHash: string) => {
    setDeletingHash(fileHash);
    deleteMutation.mutate(fileHash, {
      onSettled: () => setDeletingHash(null),
      onError: (error) => {
        toast.add({ description: `Delete failed: ${error.message}`, type: "error" });
      },
    });
  };

  return (
    <div className="flex flex-col gap-3">
      <h2 className="text-sm font-semibold">📁 Document Management</h2>
      <UploadDropzone isUploading={uploadMutation.isPending} onFilesSelected={handleUpload} />

      {isLoading && <p className="text-xs text-muted-foreground">Loading documents…</p>}
      {data && (
        <>
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Total chunks: {data.total_chunks}</span>
            <span>Documents: {data.total_documents}</span>
          </div>
          <DocumentList documents={data.documents} onDelete={handleDelete} deletingHash={deletingHash} />
        </>
      )}
    </div>
  );
}
