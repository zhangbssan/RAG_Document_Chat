import type { Source } from "./chat";

export interface EvaluationResult {
  id: string;
  question: string;
  expected_answer: string;
  expected_document: string;
  expected_page: number;
  expected_keywords: string[];
  retrieved_sources: Source[];
  document_hit_score: number;
  page_hit_score: number;
  keyword_score: number;
  final_score: number;
}

export interface EvaluationResponse {
  status: string;
  tests_run: number;
  average_final_score: number;
  required_documents: string[];
  indexed_documents: string[];
  missing_documents: string[];
  results: EvaluationResult[];
}
