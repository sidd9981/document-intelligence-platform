import { useState } from "react";
import { Team } from "../types";
import { Shield, BarChart3, Settings, Loader2 } from "lucide-react";

interface Props {
  onLogin: (team: Team) => Promise<void>;
}

const TEAMS: { id: Team; label: string; description: string; icon: React.ReactNode; color: string; border: string; hover: string }[] = [
  {
    id: "analysis",
    label: "Analysis",
    description: "Multi-document synthesis, trend analysis, 64k context, 2M tokens/day",
    icon: <BarChart3 className="w-5 h-5" />,
    color: "text-blue-400",
    border: "border-blue-500/30",
    hover: "hover:border-blue-500/60 hover:bg-blue-500/5",
  },
  {
    id: "risk",
    label: "Risk",
    description: "Compliance checks, risk extraction, 32k context, 800K tokens/day",
    icon: <Shield className="w-5 h-5" />,
    color: "text-yellow-400",
    border: "border-yellow-500/30",
    hover: "hover:border-yellow-500/60 hover:bg-yellow-500/5",
  },
  {
    id: "ops",
    label: "Operations",
    description: "Fast factual lookups, 8k context, 200K tokens/day",
    icon: <Settings className="w-5 h-5" />,
    color: "text-green-400",
    border: "border-green-500/30",
    hover: "hover:border-green-500/60 hover:bg-green-500/5",
  },
];

export function LoginScreen({ onLogin }: Props) {
  const [loading, setLoading] = useState<Team | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSelect = async (team: Team) => {
    setLoading(team);
    setError(null);
    try {
      await onLogin(team);
    } catch {
      setError("Failed to authenticate. Is the gateway running?");
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-6">
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-3xl font-semibold text-white tracking-tight">
            Fin<span className="text-blue-400">Sight</span>
          </h1>
          <p className="text-gray-400 mt-2 text-sm">
            Enterprise document intelligence for financial institutions
          </p>
        </div>

        {/* Team picker */}
        <div className="flex flex-col gap-3">
          <p className="text-xs text-gray-500 uppercase tracking-wider text-center mb-1">
            Select your team to continue
          </p>

          {TEAMS.map((team) => (
            <button
              key={team.id}
              onClick={() => handleSelect(team.id)}
              disabled={loading !== null}
              className={`flex items-center gap-4 px-4 py-4 rounded-xl border bg-white/3 transition-all text-left disabled:opacity-50 ${team.border} ${team.hover}`}
            >
              <span className={`flex-shrink-0 ${team.color}`}>{team.icon}</span>
              <div className="flex-1 min-w-0">
                <p className={`text-sm font-medium ${team.color}`}>{team.label}</p>
                <p className="text-xs text-gray-500 mt-0.5 leading-relaxed">{team.description}</p>
              </div>
              {loading === team.id && (
                <Loader2 className="w-4 h-4 text-gray-400 animate-spin flex-shrink-0" />
              )}
            </button>
          ))}
        </div>

        {error && (
          <p className="mt-4 text-center text-sm text-red-400">{error}</p>
        )}

        {/* Footer */}
        <p className="mt-8 text-center text-xs text-gray-600">
          AAPL · MSFT · TSLA · 10-K filings 2022–2026
        </p>
      </div>
    </div>
  );
}