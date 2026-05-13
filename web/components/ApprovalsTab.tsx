"use client";

// Checker-side workspace: "what's waiting on me to action, and what have
// I already cleared?".
//
// This tab is mounted only for users whose role is a checker (see
// `isCheckerRole` in `dashboard-atoms.tsx` and the gate in `app/page.tsx`).
// We deliberately keep it separate from `MyRequestsDashboard` so the
// signed-in user always sees the right surface for their next action:
//
//   * Operators see "My Requests" only (they're never on the hook to approve).
//   * Checkers get both tabs — "My Requests" for anything they happened to
//     submit, and "Approvals" for the queue waiting on them.
//
// The data comes from `GET /api/approvals/my`, which already buckets the
// live queue into ``pending_for_me`` (items where the user is authorised
// AND not the maker) and ``actioned`` (decisions they already took). The
// stats counters mirror the audit log so the KPI cards stay consistent
// even after restarts.

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  api,
  type AuditEntry,
  type AuthUser,
  type MyApprovalsResponse,
} from "@/lib/api";
import {
  DecisionRow,
  EmptyState,
  Kpi,
  PendingForMeCard,
  SectionHeader,
  Th,
} from "./dashboard-atoms";

type Props = {
  user: AuthUser;
  // Bumped by the parent when an approval is resolved elsewhere (e.g. from
  // the chat banner) so the queue refetches without a manual reload.
  refreshKey?: number;
};

export function ApprovalsTab({ user, refreshKey = 0 }: Props) {
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

  // Derive checker-side counters from the actioned history. We could have
  // the API return these pre-computed, but a single pass over an already-
  // returned array keeps the contract small and avoids another endpoint.
  const checkerStats = useMemo(() => {
    const actioned = data?.actioned ?? [];
    let approved = 0;
    let rejected = 0;
    for (const a of actioned) {
      if (a.decision === "approved") approved += 1;
      else if (a.decision === "rejected") rejected += 1;
    }
    return { approved, rejected, total: actioned.length };
  }, [data?.actioned]);

  if (!data && busy) {
    return (
      <div className="mx-auto max-w-5xl px-6 py-10 text-center text-ink-500">
        Loading the approvals queue…
      </div>
    );
  }
  if (error && !data) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-10">
        <div className="rounded-bubble border border-rose-200 bg-rose-50 px-4 py-3 text-[13.5px] text-rose-900">
          Couldn't load the approvals queue: {error}
        </div>
      </div>
    );
  }
  if (!data) return null;

  const pendingForMe = data.pending_for_me ?? [];

  return (
    <div className="mx-auto w-full max-w-5xl px-6 py-6">
      {/* ── Header ────────────────────────────────────────────────── */}
      <div className="mb-5 flex items-end justify-between">
        <div>
          <h2 className="font-serif text-2xl text-ink-900">Approvals</h2>
          <p className="mt-0.5 text-[13px] text-ink-500">
            Queue for{" "}
            <span className="font-semibold text-ink-700">
              {user.display_name || user.user_id}
            </span>{" "}
            · role{" "}
            <code className="rounded bg-cream-100 px-1.5 py-0.5 text-[11px] font-semibold text-ink-700">
              {user.role}
            </code>{" "}
            · only requests routed to your role appear here
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

      {/* ── KPI cards ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Kpi
          label="Pending your approval"
          value={pendingForMe.length}
          tone="copper"
        />
        <Kpi
          label="Approved by you"
          value={checkerStats.approved}
          tone="emerald"
        />
        <Kpi
          label="Rejected by you"
          value={checkerStats.rejected}
          tone="rose"
        />
        <Kpi
          label="Total actioned"
          value={checkerStats.total}
          tone="neutral"
        />
      </div>

      {/* ── Pending queue ─────────────────────────────────────────── */}
      <section className="mt-7">
        <SectionHeader
          title="Pending your approval"
          count={pendingForMe.length}
          hint={`Requests routed to ${user.role}`}
        />
        {pendingForMe.length === 0 ? (
          <EmptyState text="🎉 Nothing waiting on you — the queue is clear." />
        ) : (
          <div className="space-y-3">
            {pendingForMe.map((p) => (
              <PendingForMeCard
                key={p.thread_id}
                item={p}
                onResolved={refresh}
              />
            ))}
          </div>
        )}
      </section>

      {/* ── Action history ────────────────────────────────────────── */}
      <section className="mt-7">
        <SectionHeader
          title="Approvals I actioned"
          count={data.actioned.length}
          hint={`Decisions you took as ${user.role}`}
        />
        {data.actioned.length === 0 ? (
          <EmptyState text="You haven't approved or rejected anything yet." />
        ) : (
          <div className="overflow-hidden rounded-2xl border border-ink-900/8 bg-white shadow-soft">
            <table className="w-full text-left text-[13px]">
              <thead className="bg-cream-100/70 text-[11px] uppercase tracking-wide text-ink-500">
                <tr>
                  <Th>When</Th>
                  <Th>Request</Th>
                  <Th>Submitted by</Th>
                  <Th>Decision</Th>
                  <Th>Comments</Th>
                </tr>
              </thead>
              <tbody>
                {data.actioned.map((d: AuditEntry) => (
                  <DecisionRow key={d.id} entry={d} viewerIsMaker={false} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
