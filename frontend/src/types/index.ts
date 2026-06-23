export type Team = "analysis" | "risk" | "ops";

export interface Citation {
  claim: string;
  source_chunk_id: string;
  source_doc_id: string;
  confidence: number;
}

export interface QueryResponse {
  trace_id: string;
  answer: string;
  citations: Citation[];
  faithfulness_score: number;
  model_used: string;
  latency_ms: number;
  cache_hit: boolean;
  warning: string | null;
}

export interface TokenInfo {
  team_id: Team;
  daily_budget: number;
  used_today: number;
}

export interface Message {
  id: string;
  query: string;
  response: QueryResponse | null;
  streaming: boolean;
  streamedAnswer: string;
  error: string | null;
  timestamp: Date;
}

export interface AuthState {
  team: Team;
  token: string;
  expiresAt: number;
}