"use client";

// Maker-side dashboard: "what did *I* submit, and where is it?".
//
// Everything checker-related (approvals waiting on me, decisions I made)
// lives in `ApprovalsTab.tsx` so each role gets a focused workspace. The
// only role-conditional bit here is hiding the dashboard for ops who
// haven't escalated anything yet — but we render the empty state so they
// can see the contract.

import { useCallback, useEffect, useState } from "react";
import {
  api,
  type AuthUser,
  type MyApprovalsResponse,
} from "@/lib/api";
import {
  DecisionRow,
  EmptyState,
  Kpi,
  PendingRow,
  SectionHeader,
  Th,
} from "./dashboard-atoms";

type Props = {
  user: AuthUser;
  // Bumps to force a refetch (used when the user resolves an approval from
  // the chat banner without leaving the page).
  refreshKey?: number;
};

export function MyRequestsDashboard({ user, refreshKey = 0 }: Props) {
  const [data, setData] = useState<MyApprovalsResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await api.myApprovals();
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh, refreshKey]);

  if (!data && busy) {
    return (
      <div className="mx-auto max-w-5xl px-6 py-10 text-center text-ink-500">
        Loading your dashboard…
      </div>
    );
  }
  if (error && !data) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-10">
        <div className="rounded-bubble border border-rose-200 bg-rose-50 px-4 py-3 text-[13.5px] text-rose-900">
          Couldn't load the dashboard: {error}
        </div>
      </div>
    );
  }
  if (!data) return null;

  const stats = data.stats;

  return (
    <div className="mx-auto w-full max-w-5xl px-6 py-6">
      {/* ── Header ────────────────────────────────────────────────── */}
      <div className="mb-5 flex items-end justify-between">
        <div>
          <h2 className="font-serif text-2xl text-ink-900">My Requests</h2>
          <p className="mt-0.5 text-[13px] text-ink-500">
            Approval history for{" "}
            <span className="font-semibold text-ink-700">
              {user.display_name || user.user_id}
            </span>{" "}
            ({user.user_id}) · role{" "}
            <code className="rounded bg-cream-100 px-1.5 py-0.5 text-[11px] font-semibold text-ink-700">
              {user.role}
            </code>
          </p>
        </div>
        <button
          onClick={refresh}
          disabled={busy}
          className="rounded-full border border-ink-900/10 bg-white px-3.5 py-1.5 text-xs font-medium text-ink-700 shadow-soft hover:bg-cream-50 disabled:opacity-50"
        >
          {busy ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {/* ── KPI cards (maker view: 4-up) ─────────────────────────── */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Kpi label="Total submitted" value={stats.total} tone="neutral" />
        <Kpi label="Pending approval" value={stats.pending} tone="amber" />
        <Kpi label="Approved" value={stats.approved} tone="emerald" />
        <Kpi label="Rejected" value={stats.rejected} tone="rose" />
      </div>
      {stats.total > 0 ? (
        <div className="mt-3 text-[12px] text-ink-500">
          Approval rate:{" "}
          <span className="font-semibold text-ink-700">
            {(stats.approval_rate * 100).toFixed(0)}%
          </span>{" "}
          of resolved requests
        </div>
      ) : null}

      {/* ── Pending submissions (items I submitted) ──────────────── */}
      <section className="mt-7">
        <SectionHeader
          title="My pending submissions"
          count={data.pending.length}
          hint="Requests you submitted, waiting on a checker"
        />
        {data.pending.length === 0 ? (
          <EmptyState text="🎉 Nothing pending — every request you submitted has been resolved." />
        ) : (
          <div className="overflow-hidden rounded-2xl border border-ink-900/8 bg-white shadow-soft">
            <table className="w-full text-left text-[13px]">
              <thead className="bg-cream-100/70 text-[11px] uppercase tracking-wide text-ink-500">
                <tr>
                  <Th>When</Th>
                  <Th>Request</Th>
                  <Th>Risk</Th>
                  <Th>Waiting on</Th>
                  <Th>Thread</Th>
                </tr>
              </thead>
              <tbody>
                {data.pending.map((p) => (
                  <PendingRow key={p.thread_id} item={p} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* ── Decisions on my requests ──────────────────────────────── */}
      <section className="mt-7">
        <SectionHeader
          title="Resolved requests"
          count={data.decisions.length}
          hint="Decisions taken on requests you submitted"
        />
        {data.decisions.length === 0 ? (
          <EmptyState text="No decisions yet on your submissions." />
        ) : (
          <div className="overflow-hidden rounded-2xl border border-ink-900/8 bg-white shadow-soft">
            <table className="w-full text-left text-[13px]">
              <thead className="bg-cream-100/70 text-[11px] uppercase tracking-wide text-ink-500">
                <tr>
                  <Th>When</Th>
                  <Th>Request</Th>
                  <Th>Decision</Th>
                  <Th>Approver</Th>
                  <Th>Comments</Th>
                </tr>
              </thead>
              <tbody>
                {data.decisions.map((d) => (
                  <DecisionRow key={d.id} entry={d} viewerIsMaker />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
