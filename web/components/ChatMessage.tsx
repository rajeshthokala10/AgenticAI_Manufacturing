"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatTurn } from "@/lib/api";

function MetaChips({ meta }: { meta: ChatTurn["meta"] }) {
  const chips: React.ReactNode[] = [];

  if (meta.intent) {
    chips.push(
      <span
        key="intent"
        className="rounded-full bg-ink-900 px-2.5 py-0.5 text-[10.5px] font-semibold uppercase tracking-wide text-cream-50"
      >
        {meta.intent}
      </span>,
    );
  }
  if (typeof meta.intent_confidence === "number") {
    chips.push(
      <span
        key="conf"
        className="rounded-full bg-cream-200/70 px-2 py-0.5 text-[10.5px] text-ink-600"
      >
        {Math.round(meta.intent_confidence * 100)}% conf.
      </span>,
    );
  }
  meta.entities?.slice(0, 4).forEach((e, i) =>
    chips.push(
      <span
        key={`ent-${i}`}
        className="rounded-full bg-cream-200/70 px-2 py-0.5 text-[10.5px] text-ink-700"
      >
        {e.type}: {e.value}
      </span>,
    ),
  );
  if (meta.metrics?.latency_ms !== undefined && meta.metrics.latency_ms > 0) {
    chips.push(
      <span
        key="lat"
        className="rounded-full bg-cream-200/70 px-2 py-0.5 text-[10.5px] text-ink-600"
      >
        ⏱ {(meta.metrics.latency_ms / 1000).toFixed(2)}s
      </span>,
    );
  }
  if (meta.metrics?.tokens) {
    chips.push(
      <span
        key="tok"
        className="rounded-full bg-cream-200/70 px-2 py-0.5 text-[10.5px] text-ink-600"
      >
        🧩 {meta.metrics.tokens.toLocaleString()}
      </span>,
    );
  }
  if (meta.metrics?.cost_usd !== undefined && meta.metrics.cost_usd > 0) {
    chips.push(
      <span
        key="cost"
        className="rounded-full bg-cream-200/70 px-2 py-0.5 text-[10.5px] text-ink-600"
      >
        💲 ${meta.metrics.cost_usd.toFixed(4)}
      </span>,
    );
  }
  if (meta.critic?.verdict) {
    const v = meta.critic.verdict;
    const cls =
      v === "PASS"
        ? "bg-emerald-100 text-emerald-800"
        : v === "FAIL"
          ? "bg-rose-100 text-rose-800"
          : "bg-amber-100 text-amber-800";
    chips.push(
      <span
        key="crit"
        className={`rounded-full px-2 py-0.5 text-[10.5px] font-semibold ${cls}`}
      >
        Critic: {v}
      </span>,
    );
  }

  if (chips.length === 0) return null;
  return <div className="mt-3 flex flex-wrap gap-1.5">{chips}</div>;
}

function EvidencePanel({ meta }: { meta: ChatTurn["meta"] }) {
  const [open, setOpen] = useState(false);
  const evidence = meta.evidence || [];
  if (evidence.length === 0) return null;

  return (
    <div className="mt-3 rounded-xl border border-ink-900/8 bg-cream-100/60">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-4 py-2.5 text-left text-sm font-medium text-ink-700 hover:text-ink-900"
      >
        <span>
          📎 Evidence —{" "}
          <span className="font-normal text-ink-500">
            {meta.evidence_count ?? evidence.length} chunks
          </span>
        </span>
        <span className="text-ink-400">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="space-y-3 px-4 pb-4">
          {evidence.map((ev, i) => (
            <div
              key={i}
              className="rounded-lg border border-ink-900/8 bg-white/70 p-3"
            >
              <div className="mb-1 flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-wide">
                <span className="rounded bg-copper-500 px-1.5 py-0.5 font-semibold text-cream-50">
                  {ev.doc_type || "DOC"}
                </span>
                <code className="font-mono text-ink-700">{ev.source}</code>
                {ev.page !== null && ev.page !== undefined ? (
                  <span className="text-ink-500">p.{ev.page}</span>
                ) : null}
                {ev.sheet ? (
                  <span className="text-ink-500">sheet {ev.sheet}</span>
                ) : null}
                {ev.section ? (
                  <span className="text-ink-500">{ev.section}</span>
                ) : null}
                {typeof ev.score === "number" && ev.score > 0 ? (
                  <span className="ml-auto font-mono text-ink-500">
                    {ev.score.toFixed(3)}
                  </span>
                ) : null}
              </div>
              <p className="whitespace-pre-wrap text-[13px] leading-snug text-ink-700">
                {ev.text.length > 600
                  ? ev.text.slice(0, 600).trimEnd() + " …"
                  : ev.text}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function GraphPanel({ meta }: { meta: ChatTurn["meta"] }) {
  const [open, setOpen] = useState(false);
  const ctx = meta.graph_context;
  if (!ctx || ctx.node_count === 0) return null;

  return (
    <div className="mt-2 rounded-xl border border-ink-900/8 bg-cream-100/60">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-4 py-2.5 text-left text-sm font-medium text-ink-700 hover:text-ink-900"
      >
        <span>
          🔗 Knowledge graph —{" "}
          <span className="font-normal text-ink-500">
            {ctx.node_count} nodes · {ctx.edge_count} edges
          </span>
        </span>
        <span className="text-ink-400">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="flex flex-wrap gap-1.5 px-4 pb-4">
          {ctx.nodes.map((n, i) => (
            <span
              key={i}
              className="rounded-full border border-ink-900/8 bg-white/70 px-2.5 py-1 text-[12px] text-ink-700"
              title={`type: ${n.type} · chunks: ${n.chunks}`}
            >
              <span className="font-mono">{n.id}</span>
              <span className="ml-1.5 text-[10.5px] uppercase tracking-wide text-ink-500">
                {n.type}
              </span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export function ChatMessage({ turn }: { turn: ChatTurn }) {
  const isUser = turn.role === "user";

  // System messages (reset confirmations, errors) → centered, muted
  if (turn.kind === "system") {
    return (
      <div className="my-2 text-center text-xs text-ink-400">{turn.content}</div>
    );
  }

  // Correction bubble (auto spelling/acronym) — small inline assistant note
  if (turn.kind === "correction") {
    return (
      <div className="mx-auto max-w-2xl px-3">
        <div className="my-2 rounded-bubble border border-amber-200 bg-amber-50/80 px-4 py-2.5 text-[13.5px] leading-relaxed text-amber-900">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.content}</ReactMarkdown>
        </div>
      </div>
    );
  }

  // Clarifier follow-up question — soft blue bubble
  if (turn.kind === "clarify") {
    return (
      <div className="mx-auto max-w-2xl px-3">
        <div className="my-2 rounded-bubble border border-sky-200 bg-sky-50/80 px-4 py-3 text-[14px] leading-relaxed text-sky-900">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.content}</ReactMarkdown>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`mx-auto flex w-full max-w-3xl px-3 ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div
        className={
          isUser
            ? "max-w-[78%] rounded-bubble bg-ink-900 px-4 py-3 text-cream-50 shadow-soft"
            : "w-full max-w-[88%] rounded-bubble bg-white/85 px-5 py-4 shadow-soft ring-1 ring-ink-900/5"
        }
      >
        {isUser ? (
          <p className="whitespace-pre-wrap text-[15px] leading-relaxed">
            {turn.content}
          </p>
        ) : (
          <>
            <div className="prose-msg">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {turn.content}
              </ReactMarkdown>
            </div>
            <MetaChips meta={turn.meta} />
            <EvidencePanel meta={turn.meta} />
            <GraphPanel meta={turn.meta} />
          </>
        )}
      </div>
    </div>
  );
}

export function TypingBubble() {
  return (
    <div className="mx-auto flex w-full max-w-3xl px-3">
      <div className="rounded-bubble bg-white/85 px-5 py-4 shadow-soft ring-1 ring-ink-900/5">
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="typing-dot" />
      </div>
    </div>
  );
}
