interface Props {
    score: number;
  }
  
  export function FaithfulnessBadge({ score }: Props) {
    const pct = Math.round(score * 100);
  
    const color =
      score >= 0.85
        ? "bg-green-500/20 text-green-400 border-green-500/30"
        : score >= 0.70
        ? "bg-yellow-500/20 text-yellow-400 border-yellow-500/30"
        : "bg-red-500/20 text-red-400 border-red-500/30";
  
    const label =
      score >= 0.85 ? "High confidence" : score >= 0.70 ? "Medium confidence" : "Low confidence";
  
    return (
      <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${color}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-current" />
        {label} {pct}%
      </span>
    );
  }