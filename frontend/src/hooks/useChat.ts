import { useState, useCallback, useEffect } from "react";
import { Message, AuthState } from "../types";
import { sendQuery } from "../api/client";

const CONVOS_KEY = (team: string) => `finsight_convos_${team}`;
const ACTIVE_KEY = (team: string) => `finsight_active_${team}`;

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
}

function loadConversations(team: string): Conversation[] {
  try {
    const raw = localStorage.getItem(CONVOS_KEY(team));
    if (!raw) return [];
    return JSON.parse(raw).map((c: Conversation) => ({
      ...c,
      messages: c.messages.map((m: Message) => ({
        ...m,
        timestamp: new Date(m.timestamp),
      })),
    }));
  } catch {
    return [];
  }
}

function saveConversations(team: string, convos: Conversation[]) {
  try {
    localStorage.setItem(CONVOS_KEY(team), JSON.stringify(convos));
  } catch {}
}

function newConversation(): Conversation {
  return {
    id: crypto.randomUUID(),
    title: "New conversation",
    messages: [],
    createdAt: new Date().toISOString(),
  };
}

function deriveTitle(query: string): string {
  return query.length > 40 ? query.slice(0, 40) + "..." : query;
}

export function useChat(auth: AuthState) {
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    const saved = loadConversations(auth.team);
    return saved.length > 0 ? saved : [newConversation()];
  });

  const [activeId, setActiveId] = useState<string>(() => {
    const saved = loadConversations(auth.team);
    if (saved.length > 0) {
      return localStorage.getItem(ACTIVE_KEY(auth.team)) ?? saved[0].id;
    }
    return conversations[0]?.id ?? "";
  });

  const [loading, setLoading] = useState(false);

  const activeConvo = conversations.find((c) => c.id === activeId) ?? conversations[0];
  const messages = activeConvo?.messages ?? [];

  useEffect(() => {
    saveConversations(auth.team, conversations);
  }, [conversations, auth.team]);

  useEffect(() => {
    localStorage.setItem(ACTIVE_KEY(auth.team), activeId);
  }, [activeId, auth.team]);

  const updateActive = useCallback(
    (updater: (c: Conversation) => Conversation) => {
      setConversations((prev) =>
        prev.map((c) => (c.id === activeId ? updater(c) : c))
      );
    },
    [activeId]
  );

  const submit = useCallback(
    async (query: string) => {
      if (!query.trim() || loading) return;

      setLoading(true);
      const msgId = crypto.randomUUID();

      updateActive((c) => ({
        ...c,
        title: c.messages.length === 0 ? deriveTitle(query) : c.title,
        messages: [
          ...c.messages,
          {
            id: msgId,
            query,
            response: null,
            streaming: false,
            streamedAnswer: "",
            error: null,
            timestamp: new Date(),
          },
        ],
      }));

      try {
        const currentConvo = conversations.find((c) => c.id === activeId);
        const history = (currentConvo?.messages ?? [])
          .filter((m) => m.response?.answer)
          .slice(-1)
          .map((m) => ({
            query: m.query,
            answer: m.response!.answer,
          }));

        const response = await sendQuery(query, auth.token, history);

        setConversations((prev) =>
          prev.map((c) =>
            c.id === activeId
              ? {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === msgId ? { ...m, response } : m
                  ),
                }
              : c
          )
        );
      } catch (err) {
        setConversations((prev) =>
          prev.map((c) =>
            c.id === activeId
              ? {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === msgId
                      ? {
                          ...m,
                          error:
                            err instanceof Error
                              ? err.message
                              : "Something went wrong",
                        }
                      : m
                  ),
                }
              : c
          )
        );
      } finally {
        setLoading(false);
      }
    },
    [auth.token, loading, activeId, updateActive, conversations]
  );

  const newChat = useCallback(() => {
    const convo = newConversation();
    setConversations((prev) => [convo, ...prev]);
    setActiveId(convo.id);
  }, []);

  const switchConvo = useCallback((id: string) => {
    setActiveId(id);
  }, []);

  const deleteConvo = useCallback(
    (id: string) => {
      setConversations((prev) => {
        const filtered = prev.filter((c) => c.id !== id);
        if (filtered.length === 0) {
          const fresh = newConversation();
          setActiveId(fresh.id);
          return [fresh];
        }
        if (id === activeId) {
          setActiveId(filtered[0].id);
        }
        return filtered;
      });
    },
    [activeId]
  );

  return {
    messages,
    conversations,
    activeId,
    loading,
    submit,
    newChat,
    switchConvo,
    deleteConvo,
  };
}