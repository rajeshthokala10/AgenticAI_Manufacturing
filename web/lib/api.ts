// Thin client for the FastAPI backend. All requests go through Next.js's
// rewrite rule (see next.config.js) which proxies /api/* to NEXT_PUBLIC_API_ORIGIN.

export type TurnKind =
  | "text"
  | "correction"
  | "clarify"
  | "answer"
  | "system";

export type EvidenceItem = {
  source: string;
  doc_type?: string;
  page?: number | null;
  sheet?: string | null;
  section?: string | null;
  score?: number;
  text: string;
};

export type TurnMeta = {
  intent?: string;
  intent_confidence?: number;
  entities?: Array<{ type: string; value: string }>;
  is_complete?: boolean;
  metrics?: {
    latency_ms?: number;
    tokens?: number;
    cost_usd?: number;
  };
  critic?: {
    verdict?: string;
    confidence?: number;
    rationale?: string;
  };
  evidence?: EvidenceItem[];
  evidence_count?: number;
  graph_context?: {
    node_count: number;
    edge_count: number;
    nodes: Array<{ id: string; type: string; chunks: number }>;
  };
  mode?: string;
  corrected_query?: string;
  spelling_fixes?: string[];
  acronym_fixes?: string[];
  slot?: string;
  slot_required?: boolean;
  slot_prompt?: string;
};

export type ChatTurn = {
  role: "user" | "assistant" | "system";
  content: string;
  kind: TurnKind;
  meta: TurnMeta;
};

export type ChatState = {
  turns: ChatTurn[];
  awaiting_slot: string | null;
  awaiting_prompt: string | null;
};

export type HealthResponse = {
  status: string;
  ready: boolean;
  error: string | null;
  llm_enabled: boolean;
  llm_model: string;
  embedding_model: string;
  version: string;
};

export type StatsResponse = {
  documents: number;
  vectors: number;
  embedding_dim: number;
  kg_nodes: number;
  kg_edges: number;
  kg_entity_types: Record<string, number>;
  kg_relation_types: Record<string, number>;
  llm_enabled: boolean;
  input_dirs: string[];
};

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => jsonFetch<HealthResponse>("/api/health"),
  stats: () => jsonFetch<StatsResponse>("/api/stats"),
  chat: (sessionId: string, message: string) =>
    jsonFetch<{ session_id: string; new_turns: ChatTurn[]; state: ChatState }>(
      "/api/chat",
      {
        method: "POST",
        body: JSON.stringify({ session_id: sessionId, message }),
      },
    ),
  reset: (sessionId: string) =>
    jsonFetch<{ ok: boolean }>("/api/reset", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    }),
  getSession: (sessionId: string) =>
    jsonFetch<{ session_id: string; state: ChatState }>(
      `/api/sessions/${sessionId}`,
    ),
};
