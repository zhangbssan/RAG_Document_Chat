import { ApiError, getApiBaseUrl } from "./client";
import type { DocumentDeleteResponse, DocumentListResponse, UploadResponse } from "@/types";

export async function uploadPdfs(files: File[]): Promise<UploadResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const response = await fetch(`${getApiBaseUrl()}/api/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body?.detail || detail;
    } catch {
      // no JSON body
    }
    throw new ApiError(response.status, detail);
  }

  return (await response.json()) as UploadResponse;
}

export async function listDocuments(): Promise<DocumentListResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/documents/list`);
  if (!response.ok) {
    throw new ApiError(response.status, response.statusText);
  }
  return (await response.json()) as DocumentListResponse;
}

export async function deleteDocument(fileHash: string): Promise<DocumentDeleteResponse> {
  const response = await fetch(`${getApiBaseUrl()}/api/documents/${fileHash}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new ApiError(response.status, response.statusText);
  }
  return (await response.json()) as DocumentDeleteResponse;
}
