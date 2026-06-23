import { Citation } from "../types";

interface Props {
  citation: Citation;
  index: number;
}

export function CitationCard({ citation, index }: Props) {
  return (
    <div className="flex gap-3 p-3 rounded-lg bg-white/5 border border-white/10 text-sm">
      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-500/20 text-blue-400 text-xs flex items-center justify-center font-medium">
        {index + 1}
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-gray-300 leading-relaxed">{citation.claim}</p>
        <p className="mt-1 text-xs text-gray-500 font-mono truncate">
          {citation.source_doc_id.slice(0, 16)}...
        </p>
      </div>
      <span className="flex-shrink-0 text-xs text-gray-500">
        {Math.round(citation.confidence * 100)}%
      </span>
    </div>
  );
}