"use client";

import type {
  AccessPolicyResponse,
  AuthUser,
  HealthResponse,
  StatsResponse,
} from "@/lib/api";

const SUGGESTIONS = [
  "What is the OEE target for Plant A in Q1 2026?",
  "Pump P-203 has high vibration alarm ALM-P001. Cause and fix?",
  "Why did CNC Line 4 shut down in February?",
  "Compare Nippon Steel vs ArcelorMittal",
  "Belt tracking deviation on conveyor CV-301",
  "maintanance schedul for spindle bearings",
  "PLC fault code FC-003 on conveyor CV-302",
];

// One-click queries that exercise the HITL approval gate. Each is designed
// to trip a specific deterministic risk driver in `core/criticality_classifier.py`
// so reviewers can demo / smoke-test the Approvals queue without typing.
const HITL_TRIGGERS: Array<{
  label: string;
  query: string;
  badge: string;
  tone: "safety" | "hot" | "purchase";
}> = [
  {
    label: "Lockout/tagout (safety pause)",
    query: "What is the lockout/tagout procedure for pump P-203?",
    badge: "safety",
    tone: "safety",
  },
  {
    label: "Hot work permit (high risk)",
    query: "Hot work permit for tank T-9 — emergency shutdown.",
    badge: "hot work",
    tone: "hot",
  },
  {
    label: "$5,000 spare-part PO (over threshold)",
    query:
      "Please raise a PO for 5 BRG-7203 spare bearings at $5000 from Vendor SKF urgent.",
    badge: "≥ $2k",
    tone: "purchase",
  },
];

const BADGE_TONE: Record<"safety" | "hot" | "purchase", string> = {
  safety: "bg-rose-50 text-rose-700 border-rose-200",
  hot: "bg-amber-50 text-amber-800 border-amber-200",
  purchase: "bg-copper-500/10 text-copper-600 border-copper-500/30",
};

type Props = {
  health: HealthResponse | null;
  stats: StatsResponse | null;
  user?: AuthUser | null;
  accessPolicy?: AccessPolicyResponse | null;
  onPickSuggestion: (s: string) => void;
  onNewChat: () => void;
  onSignOut?: () => void;
};

// Visual tone per document-classification tier. The tone scales with the
// "sensitivity" of the tier so a plant manager's confidential badge looks
// distinctly different from an operator's public badge.
const TIER_TONE: Record<
  "public" | "restricted" | "confidential",
  { pill: string; label: string; icon: string }
> = {
  public: {
    pill: "bg-emerald-50 text-emerald-800 border-emerald-200",
    label: "Public",
    icon: "🟢",
  },
  restricted: {
    pill: "bg-amber-50 text-amber-800 border-amber-200",
    label: "Restricted",
    icon: "🟡",
  },
  confidential: {
    pill: "bg-rose-50 text-rose-800 border-rose-200",
    label: "Confidential",
    icon: "🔴",
  },
};

// Visual tone per role family so the badge is glanceable in the sidebar.
const ROLE_TONE: Record<string, string> = {
  operator: "bg-slate-100 text-slate-700 border-slate-300",
  shift_supervisor: "bg-sky-50 text-sky-800 border-sky-200",
  maintenance_planner: "bg-sky-50 text-sky-800 border-sky-200",
  maintenance_engineer: "bg-indigo-50 text-indigo-800 border-indigo-200",
  ehs_officer: "bg-rose-50 text-rose-800 border-rose-200",
  quality_engineer: "bg-violet-50 text-violet-800 border-violet-200",
  buyer: "bg-amber-50 text-amber-800 border-amber-200",
  procurement_manager: "bg-copper-500/10 text-copper-600 border-copper-500/30",
  plant_manager: "bg-emerald-50 text-emerald-800 border-emerald-200",
};

export function Sidebar({
  health,
  stats,
  user,
  accessPolicy,
  onPickSuggestion,
  onNewChat,
  onSignOut,
}: Props) {
  return (
    <aside className="hidden w-72 shrink-0 flex-col border-r border-ink-900/8 bg-cream-100/60 px-5 py-6 md:flex">
      <div className="mb-6 flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-copper-500 text-base font-bold text-cream-50 shadow-soft">
          🏭
        </div>
        <div>
          <div className="text-sm font-semibold text-ink-900">
            Manufacturing
          </div>
          <div className="text-xs text-ink-500">Hybrid GraphRAG · v1.0</div>
        </div>
      </div>

      {user ? (
        <div className="mb-4 rounded-xl border border-ink-900/8 bg-white/80 p-3 shadow-soft">
          <div className="flex items-center justify-between">
            <div className="text-[10.5px] font-semibold uppercase tracking-wide text-ink-500">
              Signed in
            </div>
            {onSignOut ? (
              <button
                onClick={onSignOut}
                className="text-[10.5px] font-semibold text-copper-600 hover:underline"
              >
                Sign out
              </button>
            ) : null}
          </div>
          <div className="mt-1 truncate text-[13px] font-semibold text-ink-900">
            {user.display_name || user.user_id}
          </div>
          <div className="mt-0.5 truncate text-[11px] text-ink-500">
            {user.user_id}
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <span
              className={`inline-block rounded-full border px-2 py-0.5 text-[10.5px] font-semibold uppercase tracking-wide ${
                ROLE_TONE[user.role] ?? "bg-cream-100 text-ink-700 border-ink-900/10"
              }`}
            >
              {user.role}
            </span>
            {accessPolicy ? (
              <span
                title={`Knowledge-base access tier — this user can read ${accessPolicy.allowed_classifications.join(", ")} content.`}
                className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10.5px] font-semibold uppercase tracking-wide ${TIER_TONE[accessPolicy.max_tier].pill}`}
              >
                <span aria-hidden>{TIER_TONE[accessPolicy.max_tier].icon}</span>
                <span>KB · {TIER_TONE[accessPolicy.max_tier].label}</span>
              </span>
            ) : null}
          </div>
          {accessPolicy ? (
            <p className="mt-2 text-[11px] leading-snug text-ink-400">
              Knowledge base filtered to{" "}
              <span className="font-medium text-ink-600">
                {accessPolicy.allowed_classifications.join(" + ")}
              </span>{" "}
              content for this role.
            </p>
          ) : null}
        </div>
      ) : null}

      <button
        onClick={onNewChat}
        className="mb-5 flex items-center justify-center gap-2 rounded-full border border-ink-900/10 bg-white px-4 py-2 text-sm font-medium text-ink-700 shadow-soft hover:bg-cream-50"
      >
        <span>＋</span>
        <span>New chat</span>
      </button>

      <div className="mb-6">
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-ink-500">
          Try a question
        </div>
        <div className="flex flex-col gap-1.5">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => onPickSuggestion(s)}
              className="rounded-lg border border-transparent px-2.5 py-1.5 text-left text-[12.5px] leading-snug text-ink-700 hover:border-ink-900/8 hover:bg-white"
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {health?.use_hitl ? (
        <div className="mb-6">
          <div className="mb-2 flex items-center justify-between">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-ink-500">
              Test approvals (HITL)
            </div>
            <span className="rounded-full bg-emerald-50 px-1.5 py-0.5 text-[9.5px] font-semibold uppercase tracking-wide text-emerald-700 ring-1 ring-emerald-200">
              gate on
            </span>
          </div>
          <div className="flex flex-col gap-1.5">
            {HITL_TRIGGERS.map((t) => (
              <button
                key={t.query}
                onClick={() => onPickSuggestion(t.query)}
                title={t.query}
                className="group flex items-start gap-2 rounded-lg border border-transparent px-2.5 py-1.5 text-left text-[12.5px] leading-snug text-ink-700 hover:border-ink-900/8 hover:bg-white"
              >
                <span
                  className={`mt-0.5 shrink-0 rounded-full border px-1.5 py-0.5 text-[9.5px] font-semibold uppercase tracking-wide ${BADGE_TONE[t.tone]}`}
                >
                  {t.badge}
                </span>
                <span className="flex-1">{t.label}</span>
              </button>
            ))}
          </div>
          <p className="mt-2 text-[11px] leading-snug text-ink-400">
            Each click pauses the workflow at the approval gate. Resolve it in
            the Streamlit{" "}
            <code className="text-ink-500">📋 Approvals</code> tab.
          </p>
        </div>
      ) : null}

      <div className="mt-auto space-y-3 text-[12.5px] text-ink-600">
        <div>
          <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-ink-500">
            Status
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`h-2 w-2 rounded-full ${
                health?.ready ? "bg-emerald-500" : "bg-amber-500"
              }`}
            />
            <span>
              {health?.ready
                ? health.llm_enabled
                  ? "Ready · LLM connected"
                  : "Ready · retrieval only"
                : "Booting…"}
            </span>
          </div>
        </div>

        {stats ? (
          <div className="grid grid-cols-2 gap-2 rounded-xl border border-ink-900/8 bg-white/70 p-3">
            <Stat label="Chunks" value={stats.documents} />
            <Stat label="Vectors" value={stats.vectors} />
            <Stat label="KG Nodes" value={stats.kg_nodes} />
            <Stat label="KG Edges" value={stats.kg_edges} />
          </div>
        ) : null}

        <div className="text-[11px] text-ink-400">
          {health?.llm_model ? (
            <>LLM: <code className="text-ink-500">{health.llm_model}</code></>
          ) : null}
        </div>
      </div>
    </aside>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="text-base font-semibold text-ink-900">
        {value.toLocaleString()}
      </div>
      <div className="text-[10.5px] uppercase tracking-wide text-ink-500">
        {label}
      </div>
    </div>
  );
}
