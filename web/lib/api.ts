// Thin client for the FastAPI backend. All requests go through Next.js's
// rewrite rule (see next.config.js) which proxies /api/* to NEXT_PUBLIC_API_ORIGIN.
//
// Auth note: the FastAPI server issues a bearer token via /api/auth/login.
// We persist it in localStorage so a refresh keeps the user signed in, and
// every fetch automatically attaches `Authorization: Bearer <token>`.

const TOKEN_KEY = "mfg-graphrag-auth-token";

export function readAuthToken(): string | null {
  return readToken();
}

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

export type PurchaseRequestPayload = {
  part_id?: string | null;
  quantity?: number | null;
  total_usd?: number | null;
  vendor?: string | null;
  urgent?: boolean;
  used_by_equipment?: string[];
  equipment_criticality?: string | null;
  single_source?: boolean | null;
  lead_time_days?: number | null;
  last_known_unit_price?: number | null;
  notes?: string[];
};

export type ApprovalSnapshot = {
  thread_id: string;
  session_id?: string | null;
  ts?: number;
  raw_query?: string;
  answer?: string;
  risk?: {
    score: number;
    drivers: string[];
    needs_human?: boolean;
  };
  purchase_request?: PurchaseRequestPayload | null;
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

export type AuditEntry = {
  id: number;
  ts: number;
  ts_iso: string;
  thread_id: string;
  decision: "approved" | "rejected";
  approver: string;
  approver_user_id?: string | null;
  approver_role?: string | null;
  maker_user_id?: string | null;
  risk_score: number;
  drivers: string[];
  required_roles: string[];
  domain: string;
  query: string;
  proposed_answer: string;
  edited_answer?: string | null;
  comments?: string | null;
};

export type AccessPolicyResponse = {
  role: string | null;
  max_tier: "public" | "restricted" | "confidential";
  allowed_classifications: Array<"public" | "restricted" | "confidential">;
  classifications_catalogue: Array<{
    id: "public" | "restricted" | "confidential";
    label: string;
    description: string;
  }>;
  user_id?: string;
  display_name?: string;
};

export type MyApprovalsResponse = {
  user: AuthUser;
  stats: {
    total: number;
    pending: number;
    approved: number;
    rejected: number;
    approval_rate: number;
    pending_for_me: number;
  };
  pending: ApprovalSnapshot[];
  pending_for_me: ApprovalSnapshot[];
  decisions: AuditEntry[];
  actioned: AuditEntry[];
};

// Domain registry is sourced from /api/domains at runtime — see
// ``api.domains()`` below. ``Domain`` is just ``string`` because adding a
// new domain on the backend (drop a schema YAML) should not require a
// TypeScript edit on the frontend.
export type Domain = string;
export const DEFAULT_DOMAIN: Domain = "manufacturing";

export type DomainCatalogEntry = {
  id: Domain;
  label: string;
  emoji: string;
  color: string;
  // UX copy authored in the schema YAML — keeps the Next.js bundle free of
  // domain literals.
  placeholder?: string;
  empty_state?: { heading?: string; blurb?: string };
  examples?: string[];
  loaded?: boolean;
};

export type DomainCatalog = {
  default: Domain;
  domains: DomainCatalogEntry[];
};

// Sensible fallback when /api/domains hasn't responded yet (first paint).
// Keeps the UI from flashing empty during the round-trip.
export const FALLBACK_CATALOG: DomainCatalog = {
  default: "manufacturing",
  domains: [
    {
      id: "manufacturing",
      label: "Manufacturing",
      emoji: "🏭",
      color: "#475569",
      empty_state: { heading: "Manufacturing Copilot" },
      examples: [],
    },
    {
      id: "aviation",
      label: "Aviation",
      emoji: "✈️",
      color: "#B45309",
      empty_state: { heading: "Aviation Copilot" },
      examples: [],
    },
  ],
};

// Shape returned by /api/diagnostic — mirrors PipelineResult.to_dict() in
// pipeline/unified_pipeline.py. Fields beyond ``answer`` / ``evidence`` are
// optional because not every pipeline mode populates them.
export type DiagnosticResponse = {
  mode: string;
  query: string;
  answer: string;
  evidence: Array<{
    text?: string;
    metadata?: Record<string, unknown>;
    vector_score?: number;
    rrf_score?: number;
  }>;
  graph_context?: {
    nodes?: Array<{ id: string; entity_type?: string }>;
    edges?: Array<{ source: string; target: string; relation?: string }>;
  } | null;
  metrics?: Record<string, unknown>;
  procedure?: Record<string, unknown> | null;
  cause_ranking?: Record<string, unknown> | null;
  rejected?: boolean;
  pipeline_status?: string;
};

export type HealthResponse = {
  status: string;
  ready: boolean;
  error: string | null;
  // Newer multi-domain shape: `domains` lists all configured domains and
  // `domain_status` maps each one to its readiness flags. The legacy
  // single-domain fields (`llm_enabled` etc.) are no longer set by the
  // FastAPI layer.
  domains?: Domain[];
  default_domain?: Domain;
  domain_status?: Record<Domain, { loaded: boolean; llm_enabled: boolean }>;
  llm_enabled?: boolean;
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
  domains: () => jsonFetch<DomainCatalog>("/api/domains"),
  stats: (domain: Domain = DEFAULT_DOMAIN) =>
    jsonFetch<StatsResponse>(`/api/stats?domain=${domain}`),
  chat: (sessionId: string, message: string, domain: Domain = DEFAULT_DOMAIN) =>
    jsonFetch<{
      session_id: string;
      domain: Domain;
      new_turns: ChatTurn[];
      state: ChatState;
    }>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, message, domain }),
    }),
  diagnostic: (message: string, domain: Domain = DEFAULT_DOMAIN) =>
    jsonFetch<DiagnosticResponse>("/api/diagnostic", {
      method: "POST",
      body: JSON.stringify({ message, domain }),
    }),
  reset: (sessionId: string, domain: Domain = DEFAULT_DOMAIN) =>
    jsonFetch<{ ok: boolean }>("/api/reset", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, domain }),
    }),
  getSession: (sessionId: string, domain: Domain = DEFAULT_DOMAIN) =>
    jsonFetch<{ session_id: string; domain: Domain; state: ChatState }>(
      `/api/sessions/${sessionId}?domain=${domain}`,
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
  myApprovals: () => jsonFetch<MyApprovalsResponse>("/api/approvals/my"),
  accessPolicy: () => jsonFetch<AccessPolicyResponse>("/api/access/policy"),
};
