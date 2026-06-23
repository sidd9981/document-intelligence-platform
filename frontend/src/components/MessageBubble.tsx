import { Message } from "../types";
import { CitationCard } from "./CitationCard";
import { FaithfulnessBadge } from "./FaithfulnessBadge";
import { AlertTriangle, Zap } from "lucide-react";

interface Props {
  message: Message;
}

export function MessageBubble({ message }: Props) {
  const { query, response, error } = message;

  return (
    <div className="flex flex-col gap-4">
      {/* Query */}
      <div className="flex justify-end">
        <div className="max-w-2xl px-4 py-3 rounded-2xl rounded-tr-sm bg-blue-600 text-white text-sm leading-relaxed">
          {query}
        </div>
      </div>

      {/* Response */}
      {error && (
        <div className="flex gap-3 items-start max-w-2xl px-4 py-3 rounded-2xl rounded-tl-sm bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
          <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          {error}
        </div>
      )}

      {response && (
        <div className="flex flex-col gap-3 max-w-3xl">
          {/* Answer */}
          <div className="px-4 py-3 rounded-2xl rounded-tl-sm bg-white/5 border border-white/10 text-gray-200 text-sm leading-relaxed">
            {response.answer}
          </div>

          {/* Meta row */}
          <div className="flex flex-wrap items-center gap-2 px-1">
            <FaithfulnessBadge score={response.faithfulness_score} />

            {response.cache_hit && (
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-purple-500/20 text-purple-400 border border-purple-500/30">
                <Zap className="w-3 h-3" />
                Cached
              </span>
            )}

            <span className="text-xs text-gray-500">
              {Math.round(response.latency_ms / 1000)}s
            </span>

            <span className="text-xs text-gray-500">
              {response.model_used}
            </span>
          </div>

          {/* Warning */}
          {response.warning && (
            <div className="flex gap-2 items-start px-3 py-2 rounded-lg bg-yellow-500/10 border border-yellow-500/20 text-yellow-400 text-xs">
              <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
              {response.warning}
            </div>
          )}

          {/* Citations */}
          {response.citations.length > 0 && (
            <div className="flex flex-col gap-2">
              <p className="text-xs text-gray-500 px-1">
                {response.citations.length} citation{response.citations.length > 1 ? "s" : ""}
              </p>
              {response.citations.map((c, i) => (
                <CitationCard key={c.source_chunk_id} citation={c} index={i} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}