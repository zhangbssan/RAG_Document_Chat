import { useMutation } from "@tanstack/react-query";
import { runEvaluation } from "@/api/evaluation";

export function useRunEvaluation() {
  return useMutation({
    mutationFn: runEvaluation,
  });
}
