// Thin client for the FastAPI backend. All requests go through Next.js's
// rewrite rule (see next.config.js) which proxies /api/* to NEXT_PUBLIC_API_ORIGIN.
//
// Auth note: the FastAPI server issues a bearer token via /api/auth/login.
// We persist it in localStorage so a refresh keeps the user signed in, and
// every fetch automatically attaches `Authorization: Bearer <token>`.

const TOKEN_KEY = "mfg-graphrag-auth-token";

function readToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setAuthToken(token: string | null): void {
  if (typeof window === "undefined") return;
  if (token) window.localStorage.setItem(TOKEN_KEY, token);
  else window.localStorage.removeItem(TOKEN_KEY);
}

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
  pending_approval_thread_id?: string | null;
};

export type ApprovalSnapshot = {
  thread_id: string;
  session_id?: string | null;
  raw_query?: string;
  answer?: string;
  risk?: {
    score: number;
    drivers: string[];
    needs_human?: boolean;
  };
  purchase_request?: Record<string, unknown> | null;
  required_roles?: string[];
  maker_user_id?: string | null;
  can_current_user_approve?: boolean;
};

export type Role = {
  id: string;
  label: string;
  description: string;
  is_maker: boolean;
  is_checker: boolean;
};

export type AuthUser = {
  user_id: string;
  role: string;
  display_name: string;
  created_at: number;
};

export type TokenResponse = {
  token: string;
  expires_at: number;
  user: AuthUser;
};

export type HealthResponse = {
  status: string;
  ready: boolean;
  error: string | null;
  llm_enabled: boolean;
  llm_model: string;
  embedding_model: string;
  version: string;
  use_hitl?: boolean;
  use_langgraph?: boolean;
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
  const token = readToken();
  const baseHeaders: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) baseHeaders.Authorization = `Bearer ${token}`;

  const res = await fetch(path, {
    ...init,
    headers: {
      ...baseHeaders,
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    // Auto-clear a stale token so the login screen takes over on the next render.
    if (res.status === 401) setAuthToken(null);
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
  getApproval: (threadId: string) =>
    jsonFetch<ApprovalSnapshot>(`/api/approvals/${threadId}`),
  resumeApproval: (
    threadId: string,
    body: {
      approved: boolean;
      approver?: string;
      comments?: string;
      edited_answer?: string | null;
    },
  ) =>
    jsonFetch<{ session_id: string | null; thread_id: string }>(
      `/api/approvals/${threadId}/resume`,
      { method: "POST", body: JSON.stringify(body) },
    ),
  listRoles: () => jsonFetch<{ roles: Role[] }>("/api/auth/roles"),
  signup: (body: { user_id: string; password: string; role: string; display_name?: string }) =>
    jsonFetch<TokenResponse>("/api/auth/signup", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  login: (body: { user_id: string; password: string }) =>
    jsonFetch<TokenResponse>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  logout: () =>
    jsonFetch<{ ok: boolean }>("/api/auth/logout", { method: "POST" }),
  me: () => jsonFetch<AuthUser>("/api/auth/me"),
};
