"use client";

// Shared atoms used by both `MyRequestsDashboard` (maker view) and
// `ApprovalsTab` (checker view). Keep this file dumb: no data-fetching, just
// stateless UI primitives + tiny formatting helpers. The two dashboards
// import everything they need from here so the visual language stays
// consistent and we don't duplicate the table styling.

import { useState } from "react";
import {
  api,
  type ApprovalSnapshot,
  type AuditEntry,
} from "@/lib/api";

// ── KPI card ────────────────────────────────────────────────────────────

export const KPI_TONE: Record<
  "neutral" | "amber" | "emerald" | "rose" | "copper",
  string
> = {
  neutral: "bg-white border-ink-900/8 text-ink-900",
  amber: "bg-amber-50 border-amber-200 text-amber-900",
  emerald: "bg-emerald-50 border-emerald-200 text-emerald-900",
  rose: "bg-rose-50 border-rose-200 text-rose-900",
  copper: "bg-copper-500/10 border-copper-500/30 text-copper-600",
};

export function Kpi({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: keyof typeof KPI_TONE;
}) {
  return (
    <div
      className={`rounded-2xl border px-4 py-3 shadow-soft ${KPI_TONE[tone]}`}
    >
      <div className="text-[10.5px] font-semibold uppercase tracking-wide opacity-80">
        {label}
      </div>
      <div className="mt-1 text-2xl font-bold">{value.toLocaleString()}</div>
    </div>
  );
}

// ── Section header / empty state ────────────────────────────────────────

export function SectionHeader({
  title,
  count,
  hint,
}: {
  title: string;
  count: number;
  hint?: string;
}) {
  return (
    <div className="mb-2 flex items-baseline justify-between">
      <h3 className="font-serif text-[15px] font-semibold text-ink-900">
        {title}{" "}
        <span className="ml-1 rounded-full bg-cream-200 px-2 py-0.5 text-[11px] font-semibold text-ink-700">
          {count}
        </span>
      </h3>
      {hint ? <span className="text-[11.5px] text-ink-500">{hint}</span> : null}
    </div>
  );
}

export function EmptyState({ text }: { text: string }) {
  return (
    <div className="rounded-2xl border border-dashed border-ink-900/10 bg-white/60 px-4 py-6 text-center text-[12.5px] text-ink-500">
      {text}
    </div>
  );
}

// ── Table cells ─────────────────────────────────────────────────────────

export function Th({ children }: { children: React.ReactNode }) {
  return <th className="px-3 py-2 font-semibold">{children}</th>;
}

export function Td({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <td className={`border-t border-ink-900/5 px-3 py-2 align-top ${className}`}>
      {children}
    </td>
  );
}

// ── Tiny formatting helpers ─────────────────────────────────────────────

export function shortenQuery(q: string, max = 90): string {
  const cleaned = (q || "").replace(/\s+/g, " ").trim();
  return cleaned.length > max ? cleaned.slice(0, max - 1) + "…" : cleaned;
}

export function relativeAgo(ts: number): string {
  const sec = Math.max(0, Date.now() / 1000 - ts);
  if (sec < 60) return `${Math.round(sec)}s ago`;
  if (sec < 3600) return `${Math.round(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.round(sec / 3600)}h ago`;
  return `${Math.round(sec / 86400)}d ago`;
}

// ── Rich card: a pending approval that the current user can action ──────
//
// Used by the Approvals tab. Renders a full snapshot of the request
// (submitter, drivers, purchase-line, proposed answer) plus an inline
// approve/reject controller so the checker never has to leave the page.

export function PendingForMeCard({
  item,
  onResolved,
}: {
  item: ApprovalSnapshot;
  onResolved: () => void;
}) {
  const ts = item.ts;
  const drivers = item.risk?.drivers ?? [];
  const required = item.required_roles ?? [];
  const purchase = item.purchase_request;
  const proposed = item.answer ?? "";

  const [comments, setComments] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (approved: boolean) => {
    if (submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await api.resumeApproval(item.thread_id, {
        approved,
        comments: comments || undefined,
      });
      onResolved();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="rounded-2xl border border-amber-300 bg-white shadow-soft">
      <div className="border-b border-amber-200 bg-amber-50/60 px-4 py-2.5">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div className="flex items-center gap-2 text-[12px] text-amber-900">
            <span className="font-semibold">
              From {item.maker_user_id ?? "unknown"}
            </span>
            <span className="opacity-60">·</span>
            <span>risk {item.risk?.score?.toFixed(2) ?? "—"}</span>
            <span className="opacity-60">·</span>
            <span>{ts ? relativeAgo(ts) : "—"}</span>
          </div>
          <code className="text-[11px] font-mono text-amber-800">
            {item.thread_id}
          </code>
        </div>
      </div>

      <div className="space-y-3 px-4 py-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-ink-500">
            Request
          </div>
          <div className="mt-1 text-[14px] text-ink-800">{item.raw_query}</div>
        </div>

        {purchase ? (
          <div className="rounded-xl border border-ink-900/8 bg-cream-50 px-3 py-2 text-[12.5px] text-ink-800">
            <div className="font-semibold text-ink-900">Purchase request</div>
            <div className="mt-0.5">
              {purchase.quantity ? <>qty {purchase.quantity} · </> : null}
              {purchase.part_id ? <code>{purchase.part_id}</code> : null}
              {purchase.total_usd ? (
                <>
                  {" "}
                  · total{" "}
                  <span className="font-semibold">
                    ${purchase.total_usd.toLocaleString()}
                  </span>
                </>
              ) : null}
              {purchase.vendor ? <> · vendor {purchase.vendor}</> : null}
              {purchase.urgent ? (
                <span className="ml-1 rounded-full bg-rose-100 px-1.5 py-0.5 text-[10px] font-semibold text-rose-800">
                  URGENT
                </span>
              ) : null}
            </div>
          </div>
        ) : null}

        {drivers.length > 0 ? (
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-wide text-ink-500">
              Drivers
            </div>
            <div className="mt-1 flex flex-wrap gap-1">
              {drivers.map((d) => (
                <span
                  key={d}
                  className="rounded-full border border-amber-300 bg-amber-50 px-2 py-0.5 text-[11px] font-medium text-amber-900"
                >
                  {d}
                </span>
              ))}
            </div>
          </div>
        ) : null}

        {required.length > 0 ? (
          <div className="text-[12px] text-ink-600">
            Required role(s):{" "}
            {required.map((r, i) => (
              <span key={r}>
                {i > 0 ? " · " : ""}
                <code className="rounded bg-cream-100 px-1.5 py-0.5 text-[11px] font-semibold text-ink-700">
                  {r}
                </code>
              </span>
            ))}
          </div>
        ) : null}

        {proposed ? (
          <details className="rounded-xl border border-ink-900/8 bg-white">
            <summary className="cursor-pointer select-none px-3 py-2 text-[12px] font-semibold text-ink-700">
              Proposed answer
            </summary>
            <div className="border-t border-ink-900/5 px-3 py-2 text-[12.5px] leading-relaxed text-ink-700">
              {proposed.length > 1200 ? proposed.slice(0, 1200) + "…" : proposed}
            </div>
          </details>
        ) : null}

        <div>
          <label className="text-[11px] font-semibold uppercase tracking-wide text-ink-500">
            Comments (audit log)
          </label>
          <textarea
            value={comments}
            onChange={(e) => setComments(e.target.value)}
            placeholder="Optional context for the decision…"
            rows={2}
            className="mt-1 block w-full resize-none rounded-lg border border-ink-900/10 bg-white px-3 py-2 text-[13px] text-ink-800 placeholder-ink-400 focus:border-copper-500/60 focus:outline-none focus:ring-2 focus:ring-copper-500/20"
          />
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => submit(true)}
            disabled={submitting}
            className="rounded-full bg-emerald-600 px-4 py-1.5 text-[13px] font-semibold text-white hover:bg-emerald-700 disabled:bg-ink-300"
          >
            {submitting ? "Submitting…" : "✓ Approve"}
          </button>
          <button
            type="button"
            onClick={() => submit(false)}
            disabled={submitting}
            className="rounded-full bg-rose-600 px-4 py-1.5 text-[13px] font-semibold text-white hover:bg-rose-700 disabled:bg-ink-300"
          >
            ✗ Reject
          </button>
          {error ? (
            <span className="text-[12px] text-rose-900">{error}</span>
          ) : null}
        </div>
      </div>
    </div>
  );
}

// ── Compact table row: pending submitted by this user ───────────────────

export function PendingRow({ item }: { item: ApprovalSnapshot }) {
  const ts = item.ts;
  const drivers = item.risk?.drivers ?? [];
  const required = item.required_roles ?? [];
  return (
    <tr>
      <Td className="whitespace-nowrap text-ink-500">
        {ts ? relativeAgo(ts) : "—"}
      </Td>
      <Td>
        <div className="text-ink-800">
          {shortenQuery(item.raw_query || "(no query)")}
        </div>
        {drivers.length > 0 ? (
          <div className="mt-1 flex flex-wrap gap-1">
            {drivers.slice(0, 4).map((d) => (
              <span
                key={d}
                className="rounded-full border border-amber-300 bg-amber-50 px-1.5 py-0.5 text-[10px] text-amber-900"
              >
                {d}
              </span>
            ))}
          </div>
        ) : null}
      </Td>
      <Td className="whitespace-nowrap text-amber-800">
        {item.risk?.score !== undefined ? item.risk.score.toFixed(2) : "—"}
      </Td>
      <Td>
        <div className="flex flex-wrap gap-1">
          {required.map((r) => (
            <code
              key={r}
              className="rounded bg-cream-100 px-1.5 py-0.5 text-[10.5px] font-semibold text-ink-700"
            >
              {r}
            </code>
          ))}
        </div>
      </Td>
      <Td>
        <code className="text-[11px] font-mono text-ink-500">
          {item.thread_id}
        </code>
      </Td>
    </tr>
  );
}

// ── Resolved decision row (maker view vs checker view) ──────────────────

export function DecisionRow({
  entry,
  viewerIsMaker,
}: {
  entry: AuditEntry;
  viewerIsMaker: boolean;
}) {
  const approved = entry.decision === "approved";
  return (
    <tr>
      <Td className="whitespace-nowrap text-ink-500">{entry.ts_iso}</Td>
      <Td>
        <div className="text-ink-800">{shortenQuery(entry.query)}</div>
        {entry.drivers && entry.drivers.length > 0 ? (
          <div className="mt-1 flex flex-wrap gap-1">
            {entry.drivers.slice(0, 3).map((d) => (
              <span
                key={d}
                className="rounded-full border border-ink-900/10 bg-cream-100 px-1.5 py-0.5 text-[10px] text-ink-600"
              >
                {d}
              </span>
            ))}
          </div>
        ) : null}
      </Td>
      {viewerIsMaker ? (
        <>
          <Td>
            <span
              className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${
                approved
                  ? "bg-emerald-50 text-emerald-800"
                  : "bg-rose-50 text-rose-800"
              }`}
            >
              {approved ? "✓ approved" : "✗ rejected"}
            </span>
          </Td>
          <Td>
            <div className="font-semibold text-ink-800">{entry.approver}</div>
            <div className="text-[11px] text-ink-500">
              {entry.approver_user_id ?? "—"}
              {entry.approver_role ? (
                <>
                  {" · "}
                  <code className="rounded bg-cream-100 px-1 py-0.5 text-[10px] font-semibold text-ink-700">
                    {entry.approver_role}
                  </code>
                </>
              ) : null}
            </div>
          </Td>
        </>
      ) : (
        <>
          <Td>
            <div className="font-semibold text-ink-800">
              {entry.maker_user_id ?? "—"}
            </div>
          </Td>
          <Td>
            <span
              className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${
                approved
                  ? "bg-emerald-50 text-emerald-800"
                  : "bg-rose-50 text-rose-800"
              }`}
            >
              {approved ? "✓ approved" : "✗ rejected"}
            </span>
          </Td>
        </>
      )}
      <Td className="text-[12px] text-ink-600">
        {entry.comments ? entry.comments : <span className="text-ink-400">—</span>}
      </Td>
    </tr>
  );
}

// ── Role helper used by the tab-gating logic ────────────────────────────
//
// Stays in lock-step with `core/rbac.py::ROLES`. Operators are the only
// makers in the current catalogue, so the simplest correct check is
// "anyone whose role isn't operator". If new pure-maker roles get added,
// extend this set rather than scattering string checks across components.

const PURE_MAKER_ROLES = new Set<string>(["operator"]);

export function isCheckerRole(role: string | null | undefined): boolean {
  if (!role) return false;
  return !PURE_MAKER_ROLES.has(role);
}
