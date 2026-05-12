"use client";

import type { HealthResponse, StatsResponse } from "@/lib/api";

const SUGGESTIONS = [
  "What is the OEE target for Plant A in Q1 2026?",
  "Pump P-203 has high vibration alarm ALM-P001. Cause and fix?",
  "Why did CNC Line 4 shut down in February?",
  "Compare Nippon Steel vs ArcelorMittal",
  "Belt tracking deviation on conveyor CV-301",
  "maintanance schedul for spindle bearings",
  "PLC fault code FC-003 on conveyor CV-302",
];

type Props = {
  health: HealthResponse | null;
  stats: StatsResponse | null;
  onPickSuggestion: (s: string) => void;
  onNewChat: () => void;
};

export function Sidebar({ health, stats, onPickSuggestion, onNewChat }: Props) {
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
