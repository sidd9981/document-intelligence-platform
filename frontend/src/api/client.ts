import { Team, QueryResponse, AuthState } from "../types";

const BASE = "/api";

export async function getToken(team: Team): Promise<AuthState> {
  const body = new URLSearchParams({
    grant_type: "client_credentials",
    client_id: team,
    client_secret: `dev-secret-${team}`,
  });

  const res = await fetch(`${BASE}/oauth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString(),
  });

  if (!res.ok) throw new Error("Authentication failed");

  const data = await res.json();
  return {
    team,
    token: data.access_token,
    expiresAt: Date.now() + data.expires_in * 1000,
  };
}

export async function sendQuery(
    query: string,
    token: string,
    history: { query: string; answer: string }[] = []
  ): Promise<QueryResponse> {
    const res = await fetch(`${BASE}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ query, history }),
    });
  
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail?.message || err.detail || "Query failed");
    }
  
    return res.json();
  }

export async function getBudget(
team: string,
token: string
): Promise<{ used: number; limit: number; pct: number }> {
const res = await fetch(`${BASE}/budget/${team}`, {
    headers: { Authorization: `Bearer ${token}` },
});
if (!res.ok) throw new Error("Failed to fetch budget");
return res.json();
}

export async function* streamQuery(
  query: string,
  token: string
): AsyncGenerator<string> {
  const res = await fetch(`${BASE}/query/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ query }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail?.message || err.detail || "Stream failed");
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split("\n");

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const token = line.slice(6);
        if (token === "[DONE]" || token === "[ERROR]") return;
        if (token.trim()) yield token;
      }
    }
  }
}