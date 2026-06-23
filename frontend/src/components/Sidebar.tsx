import { AuthState, Team } from "../types";
import { LogOut, Shield, BarChart3, Settings, Plus, Trash2, MessageSquare } from "lucide-react";
import { Conversation } from "../hooks/useChat";

interface Props {
  auth: AuthState;
  conversations: Conversation[];
  activeId: string;
  lastModelUsed: string | null;
  budget: { used: number; limit: number; pct: number } | null;
  onNewChat: () => void;
  onSwitchConvo: (id: string) => void;
  onDeleteConvo: (id: string) => void;
  onLogout: () => void;
}

const TEAM_CONFIG: Record<Team, {
  label: string;
  color: string;
  bg: string;
  border: string;
  icon: React.ReactNode;
  budget: string;
  model: string;
  description: string;
}> = {
  analysis: {
    label: "Analysis",
    color: "text-blue-400",
    bg: "bg-blue-500/10",
    border: "border-blue-500/20",
    icon: <BarChart3 className="w-4 h-4" />,
    budget: "2M tokens/day",
    model: "llama3.1:8b",
    description: "Multi-document synthesis, trend analysis",
  },
  risk: {
    label: "Risk",
    color: "text-yellow-400",
    bg: "bg-yellow-500/10",
    border: "border-yellow-500/20",
    icon: <Shield className="w-4 h-4" />,
    budget: "800K tokens/day",
    model: "llama3.1:8b",
    description: "Compliance checks, risk extraction",
  },
  ops: {
    label: "Operations",
    color: "text-green-400",
    bg: "bg-green-500/10",
    border: "border-green-500/20",
    icon: <Settings className="w-4 h-4" />,
    budget: "200K tokens/day",
    model: "llama3.2:3b",
    description: "Fast factual lookups",
  },
};

export function Sidebar({
  auth,
  conversations,
  activeId,
  lastModelUsed,
  budget,
  onNewChat,
  onSwitchConvo,
  onDeleteConvo,
  onLogout,
}: Props) {
  const config = TEAM_CONFIG[auth.team];

  return (
    <div className="w-64 flex-shrink-0 flex flex-col bg-gray-900 border-r border-white/10">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-white/10">
        <h1 className="text-lg font-semibold text-white tracking-tight">
          Fin<span className="text-blue-400">Sight</span>
        </h1>
        <p className="text-xs text-gray-500 mt-0.5">Document Intelligence</p>
      </div>

      {/* Team badge */}
      <div className="px-4 py-4 border-b border-white/10">
        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Active team</p>
        <div className={`flex items-center gap-2.5 px-3 py-2.5 rounded-lg ${config.bg} border ${config.border}`}>
          <span className={config.color}>{config.icon}</span>
          <div>
            <p className={`text-sm font-medium ${config.color}`}>{config.label}</p>
            <p className="text-xs text-gray-500">{config.description}</p>
          </div>
        </div>
      </div>

      {/* New chat button */}
      <div className="px-4 pt-4">
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-white/10 bg-white/3 hover:bg-white/8 text-sm text-gray-300 hover:text-white transition-colors"
        >
          <Plus className="w-4 h-4" />
          New chat
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto px-4 py-3 flex flex-col gap-1">
        {conversations.map((convo) => (
          <div
            key={convo.id}
            className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
              convo.id === activeId
                ? "bg-white/10 text-white"
                : "text-gray-400 hover:bg-white/5 hover:text-white"
            }`}
            onClick={() => onSwitchConvo(convo.id)}
          >
            <MessageSquare className="w-3.5 h-3.5 flex-shrink-0" />
            <span className="text-xs flex-1 truncate">{convo.title}</span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDeleteConvo(convo.id);
              }}
              className="opacity-0 group-hover:opacity-100 flex-shrink-0 text-gray-500 hover:text-red-400 transition-all"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          </div>
        ))}
      </div>

      {/* Stats */}
      <div className="px-4 py-4 flex flex-col gap-3 border-t border-white/10">
        <div className="flex flex-col gap-1.5">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Budget</span>
            <span className="text-white font-medium text-xs">
              {budget ? `${budget.pct}% used` : config.budget}
            </span>
          </div>
          {budget && (
            <div className="w-full h-1.5 rounded-full bg-white/10">
              <div
                className={`h-1.5 rounded-full transition-all ${
                  budget.pct > 80
                    ? "bg-red-500"
                    : budget.pct > 50
                    ? "bg-yellow-500"
                    : "bg-green-500"
                }`}
                style={{ width: `${Math.min(budget.pct, 100)}%` }}
              />
            </div>
          )}
        </div>

        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Model</span>
          <span className="text-white font-medium font-mono text-xs">
            {lastModelUsed ?? config.model}
          </span>
        </div>

        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Chunks</span>
          <span className="text-white font-medium">1,939</span>
        </div>
      </div>

      {/* Logout */}
      <div className="px-4 py-4 border-t border-white/10">
        <button
          onClick={onLogout}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-gray-400 hover:text-white hover:bg-white/5 transition-colors"
        >
          <LogOut className="w-4 h-4" />
          Switch team
        </button>
      </div>
    </div>
  );
}