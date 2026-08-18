import { Button } from "@/components/ui/button";
import { EvaluationResultCard } from "@/components/evaluation/EvaluationResultCard";
import { useRunEvaluation } from "@/hooks/useEvaluation";
import { useDocuments } from "@/hooks/useDocuments";

const SAMPLE_DOCUMENTS = [
  "employee_handbook_en.pdf",
  "product_manual_en.pdf",
  "service_agreement_en.pdf",
];

export function EvaluationPage() {
  const { data: documents } = useDocuments();
  const evaluationMutation = useRunEvaluation();
  const result = evaluationMutation.data;

  const indexedNames = (documents?.documents ?? []).map((doc) => doc.document_name);
  const requiredDocuments = result?.required_documents ?? SAMPLE_DOCUMENTS;
  const missingDocuments =
    result?.missing_documents ?? requiredDocuments.filter((name) => !indexedNames.includes(name));

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-6">
      <div>
        <h2 className="text-base font-semibold">Evaluation Panel</h2>
        <p className="text-xs text-muted-foreground">
          Evaluation uses 5 predefined questions for the sample PDFs.
        </p>
      </div>

      <div className="text-xs">
        <p className="font-medium">Required sample documents:</p>
        {requiredDocuments.map((name) => (
          <p key={name} className="font-mono text-muted-foreground">
            sample_docs/{name}
          </p>
        ))}
      </div>

      {missingDocuments.length > 0 && (
        <div className="rounded-md border border-amber-400 bg-amber-50 p-3 text-xs text-amber-800 dark:bg-amber-950 dark:text-amber-300">
          Some required sample documents are missing. Please upload them before running evaluation.
          Scores may be low.
          <ul className="mt-1 list-disc pl-4">
            {missingDocuments.map((name) => (
              <li key={name} className="font-mono">
                sample_docs/{name}
              </li>
            ))}
          </ul>
        </div>
      )}

      <Button
        onClick={() => evaluationMutation.mutate()}
        disabled={evaluationMutation.isPending}
        className="w-fit"
      >
        {evaluationMutation.isPending ? "Running evaluation…" : "Run Evaluation"}
      </Button>

      {evaluationMutation.isError && (
        <p className="text-sm text-destructive">
          Evaluation failed: {(evaluationMutation.error as Error).message}
        </p>
      )}

      {!result && !evaluationMutation.isPending && (
        <p className="text-sm text-muted-foreground">Run evaluation to see retrieval scores.</p>
      )}

      {result && (
        <>
          <div className="flex gap-6 text-sm">
            <div>
              <div className="text-lg font-semibold">{result.tests_run}</div>
              <div className="text-muted-foreground">Tests Run</div>
            </div>
            <div>
              <div className="text-lg font-semibold">{result.average_final_score.toFixed(2)}</div>
              <div className="text-muted-foreground">Average Final Score</div>
            </div>
          </div>
          <div className="flex flex-col gap-2">
            {result.results.map((item) => (
              <EvaluationResultCard key={item.id} result={item} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
