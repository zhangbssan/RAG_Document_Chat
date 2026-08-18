export interface DocumentRecord {
  document_name: string;
  file_hash: string;
  pages: number;
  chunks: number;
}

export interface DocumentListResponse {
  total_chunks: number;
  total_documents: number;
  documents: DocumentRecord[];
}

export interface UploadResponse {
  added_chunks: number;
  messages: string[];
}

export interface DocumentDeleteResponse {
  message: string;
  document_name: string;
  file_hash: string;
  deleted_chunks: number;
  deleted_file: boolean;
}
