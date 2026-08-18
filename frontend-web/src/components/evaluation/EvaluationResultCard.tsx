import { CitationCard } from "@/components/chat/CitationCard";
import type { EvaluationResult } from "@/types";

interface EvaluationResultCardProps {
  result: EvaluationResult;
}

export function EvaluationResultCard({ result }: EvaluationResultCardProps) {
  return (
    <details className="rounded-md border border-border p-3 text-sm">
      <summary className="cursor-pointer font-medium">
        {result.id} · Final {result.final_score.toFixed(2)} · Document{" "}
        {result.document_hit_score.toFixed(2)} · Page {result.page_hit_score.toFixed(2)} · Keywords{" "}
        {result.keyword_score.toFixed(2)}
      </summary>
      <div className="mt-3 flex flex-col gap-2 text-xs">
        <p>
          <span className="font-medium">Question:</span> {result.question}
        </p>
        <p>
          <span className="font-medium">Expected Answer:</span> {result.expected_answer}
        </p>
        <p>
          <span className="font-medium">Expected Source:</span> {result.expected_document} page{" "}
          {result.expected_page}
        </p>
        <p>
          <span className="font-medium">Expected Keywords:</span> {result.expected_keywords.join(", ")}
        </p>
        <div className="grid grid-cols-4 gap-2 rounded-md bg-muted p-2 text-center">
          <div>
            <div className="font-semibold">{result.document_hit_score.toFixed(2)}</div>
            <div className="text-muted-foreground">Document Hit</div>
          </div>
          <div>
            <div className="font-semibold">{result.page_hit_score.toFixed(2)}</div>
            <div className="text-muted-foreground">Page Hit</div>
          </div>
          <div>
            <div className="font-semibold">{result.keyword_score.toFixed(2)}</div>
            <div className="text-muted-foreground">Keyword Score</div>
          </div>
          <div>
            <div className="font-semibold">{result.final_score.toFixed(2)}</div>
            <div className="text-muted-foreground">Final Score</div>
          </div>
        </div>
        {result.retrieved_sources.length > 0 ? (
          <div className="flex flex-col gap-2">
            {result.retrieved_sources.map((source, index) => (
              <CitationCard key={`${source.chunk}-${index}`} source={source} index={index + 1} />
            ))}
          </div>
        ) : (
          <p className="text-amber-600 dark:text-amber-400">No sources retrieved.</p>
        )}
      </div>
    </details>
  );
}
