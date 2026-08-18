import { apiFetch } from "./client";
import type { EvaluationResponse } from "@/types";

export async function runEvaluation(): Promise<EvaluationResponse> {
  return apiFetch<EvaluationResponse>("/api/evaluate", { method: "POST" });
}
