import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { deleteDocument, listDocuments, uploadPdfs } from "@/api/documents";

export const documentsQueryKey = ["documents"] as const;

export function useDocuments() {
  return useQuery({
    queryKey: documentsQueryKey,
    queryFn: listDocuments,
  });
}

export function useUploadPdfs() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (files: File[]) => uploadPdfs(files),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: documentsQueryKey });
    },
  });
}

export function useDeleteDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (fileHash: string) => deleteDocument(fileHash),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: documentsQueryKey });
    },
  });
}
