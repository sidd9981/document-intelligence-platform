import { useState, useRef, useEffect, useCallback } from "react";
import { AuthState, Team } from "./types";
import { getToken, getBudget } from "./api/client";
import { LoginScreen } from "./components/LoginScreen";
import { Sidebar } from "./components/Sidebar";
import { ChatInput } from "./components/ChatInput";
import { MessageBubble } from "./components/MessageBubble";
import { useChat } from "./hooks/useChat";

const AUTH_KEY = "finsight_auth";

function loadAuth(): AuthState | null {
  try {
    const raw = localStorage.getItem(AUTH_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed.expiresAt < Date.now()) {
      localStorage.removeItem(AUTH_KEY);
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export default function App() {
  const [auth, setAuth] = useState<AuthState | null>(() => loadAuth());

  const handleLogin = async (team: Team) => {
    const authState = await getToken(team);
    localStorage.setItem(AUTH_KEY, JSON.stringify(authState));
    setAuth(authState);
  };

  const handleLogout = () => {
    localStorage.removeItem(AUTH_KEY);
    setAuth(null);
  };

  if (!auth) {
    return <LoginScreen onLogin={handleLogin} />;
  }

  return <ChatView auth={auth} onLogout={handleLogout} />;
}

function ChatView({ auth, onLogout }: { auth: AuthState; onLogout: () => void }) {
  const { messages, conversations, activeId, loading, submit, newChat, switchConvo, deleteConvo } = useChat(auth);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [budget, setBudget] = useState<{ used: number; limit: number; pct: number } | null>(null);

  const lastModel = messages.length > 0
    ? messages[messages.length - 1].response?.model_used ?? null
    : null;

  const fetchBudget = useCallback(async () => {
    try {
      const b = await getBudget(auth.team, auth.token);
      setBudget(b);
    } catch {}
  }, [auth.team, auth.token]);

  useEffect(() => {
    fetchBudget();
  }, [messages.length, fetchBudget]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex h-screen bg-gray-950 text-white overflow-hidden">
      <Sidebar
        auth={auth}
        conversations={conversations}
        activeId={activeId}
        lastModelUsed={lastModel}
        budget={budget}
        onNewChat={newChat}
        onSwitchConvo={switchConvo}
        onDeleteConvo={deleteConvo}
        onLogout={onLogout}
      />
      <div className="flex-1 flex flex-col min-w-0">
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
          <div>
            <h2 className="text-sm font-medium text-white">Financial Document Q&A</h2>
            <p className="text-xs text-gray-500 mt-0.5">
              Hybrid retrieval · Graph RAG · LLM-as-judge faithfulness
            </p>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-6 flex flex-col gap-8">
          {messages.length === 0 && (
            <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center">
              <div className="w-12 h-12 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
                <span className="text-blue-400 text-xl">F</span>
              </div>
              <div>
                <p className="text-white font-medium">Ask anything about the corpus</p>
                <p className="text-gray-500 text-sm mt-1">
                  AAPL, MSFT, TSLA — 10-K filings from 2022 to 2026
                </p>
              </div>
              <div className="flex flex-col gap-2 mt-2 w-full max-w-sm">
                {EXAMPLE_QUERIES.map((q) => (
                  <button
                    key={q}
                    onClick={() => submit(q)}
                    className="px-4 py-2.5 rounded-xl border border-white/10 bg-white/3 hover:bg-white/5 text-sm text-gray-400 hover:text-white transition-colors text-left"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}

          <div ref={bottomRef} />
        </div>

        <ChatInput onSubmit={submit} loading={loading} />
      </div>
    </div>
  );
}

const EXAMPLE_QUERIES = [
  "What are Apple main supply chain risks?",
  "How does Microsoft describe its cloud competition?",
  "What manufacturing risks does Tesla disclose?",
];