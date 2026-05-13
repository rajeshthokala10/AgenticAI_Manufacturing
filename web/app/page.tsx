"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ChatMessage, TypingBubble } from "@/components/ChatMessage";
import { ChatComposer } from "@/components/ChatComposer";
import { Sidebar } from "@/components/Sidebar";
import { AuthGate } from "@/components/AuthGate";
import { MyRequestsDashboard } from "@/components/MyRequestsDashboard";
import { ApprovalsTab } from "@/components/ApprovalsTab";
import { isCheckerRole } from "@/components/dashboard-atoms";
import {
  api,
  setAuthToken,
  type ApprovalSnapshot,
  type AuthUser,
  type ChatTurn,
  type HealthResponse,
  type StatsResponse,
} from "@/lib/api";

type ActiveTab = "chat" | "requests" | "approvals";

// Default Streamlit port from run.sh. Override at build time with
// NEXT_PUBLIC_STREAMLIT_ORIGIN if your deployment uses a non-default host.
const STREAMLIT_ORIGIN =
  process.env.NEXT_PUBLIC_STREAMLIT_ORIGIN ?? "http://localhost:8501";

const SESSION_KEY = "mfg-graphrag-session-id";

function makeSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID().replace(/-/g, "");
  }
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export default function ChatPage() {
  const [sessionId, setSessionId] = useState<string>("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [awaitingPrompt, setAwaitingPrompt] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pendingThreadId, setPendingThreadId] = useState<string | null>(null);
  const [pendingDetail, setPendingDetail] = useState<ApprovalSnapshot | null>(
    null,
  );
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [activeTab, setActiveTab] = useState<ActiveTab>("chat");
  // Bumped after every approval action so the dashboard refetches without a
  // manual reload.
  const [dashboardRefreshKey, setDashboardRefreshKey] = useState(0);

  const scrollerRef = useRef<HTMLDivElement>(null);

  // Resolve the signed-in user on first load. Stays null if no token is
  // present or the token is invalid — `AuthGate` then handles login/signup.
  useEffect(() => {
    let cancelled = false;
    api
      .me()
      .then((u) => {
        if (!cancelled) setAuthUser(u);
      })
      .catch(() => {
        if (!cancelled) setAuthUser(null);
      })
      .finally(() => {
        if (!cancelled) setAuthChecked(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleSignOut = useCallback(async () => {
    try {
      await api.logout();
    } catch {
      /* ignore */
    }
    setAuthToken(null);
    setAuthUser(null);
    setTurns([]);
    setAwaitingPrompt(null);
    setPendingThreadId(null);
    setPendingDetail(null);
  }, []);

  // Bootstrap session id from localStorage (or create one).
  useEffect(() => {
    let id =
      typeof window !== "undefined" ? localStorage.getItem(SESSION_KEY) : null;
    if (!id) {
      id = makeSessionId();
      if (typeof window !== "undefined") localStorage.setItem(SESSION_KEY, id);
    }
    setSessionId(id);
  }, []);

  // Health + stats poll (and load any existing transcript).
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;

    const refresh = async () => {
      try {
        const h = await api.health();
        if (!cancelled) setHealth(h);
        if (h.ready) {
          try {
            const s = await api.stats();
            if (!cancelled) setStats(s);
          } catch {
            /* ignore */
          }
          try {
            const sess = await api.getSession(sessionId);
            if (!cancelled) {
              setTurns(sess.state.turns);
              setAwaitingPrompt(sess.state.awaiting_prompt);
              setPendingThreadId(
                sess.state.pending_approval_thread_id ?? null,
              );
            }
          } catch {
            /* fresh session */
          }
        }
      } catch (e) {
        if (!cancelled) setHealth(null);
      }
    };
    refresh();
    const t = setInterval(refresh, 8000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [sessionId]);

  // Auto-scroll to the bottom when content changes.
  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [turns.length, busy]);

  const send = useCallback(
    async (msg: string) => {
      const trimmed = msg.trim();
      if (!trimmed || !sessionId || busy) return;
      setInput("");
      setError(null);
      // Optimistic user bubble
      setTurns((cur) => [
        ...cur,
        { role: "user", content: trimmed, kind: "text", meta: {} },
      ]);
      setBusy(true);
      try {
        const res = await api.chat(sessionId, trimmed);
        setTurns(res.state.turns);
        setAwaitingPrompt(res.state.awaiting_prompt);
        const nextThread = res.state.pending_approval_thread_id ?? null;
        setPendingThreadId(nextThread);
        // A new request entered the approval queue — make sure the dashboard
        // shows it next time the user switches tabs.
        if (nextThread) setDashboardRefreshKey((k) => k + 1);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
      } finally {
        setBusy(false);
      }
    },
    [sessionId, busy],
  );

  const handleSubmit = () => send(input);
  const handlePickSuggestion = (s: string) => {
    setInput(s);
    setTimeout(() => send(s), 0);
  };
  const handleNewChat = async () => {
    if (!sessionId) return;
    setError(null);
    try {
      await api.reset(sessionId);
    } catch {
      /* ignore */
    }
    setTurns([]);
    setAwaitingPrompt(null);
    setPendingThreadId(null);
    setPendingDetail(null);
    setInput("");
  };

  // When a workflow pauses, fetch the approval payload (drivers, risk, etc.)
  // so the banner can explain *why* it paused without a manual lookup.
  useEffect(() => {
    if (!pendingThreadId) {
      setPendingDetail(null);
      return;
    }
    let cancelled = false;
    api
      .getApproval(pendingThreadId)
      .then((snap) => {
        if (!cancelled) setPendingDetail(snap);
      })
      .catch(() => {
        if (!cancelled) setPendingDetail(null);
      });
    return () => {
      cancelled = true;
    };
  }, [pendingThreadId]);

  const placeholder = pendingThreadId
    ? "Workflow paused for approval — resolve it in the Streamlit Approvals tab"
    : awaitingPrompt
      ? `Answer: ${awaitingPrompt}  (or type 'skip')`
      : "Ask anything about your manufacturing operations…";

  // Auth gate. While `me()` is in-flight we render nothing to avoid a flash
  // of the chat UI for unauthenticated users.
  if (!authChecked) {
    return <div className="h-screen w-full bg-cream-50" />;
  }
  if (!authUser) {
    return <AuthGate onSignIn={(u) => setAuthUser(u)} />;
  }

  // Approvals tab is only mounted for checker roles (operators never approve).
  const showApprovalsTab = isCheckerRole(authUser.role);
  // Defensive: if a tab is hidden after a sign-in/out it should fall back to
  // chat instead of leaving the user staring at a blank pane.
  const effectiveTab: ActiveTab =
    activeTab === "approvals" && !showApprovalsTab ? "chat" : activeTab;

  return (
    <div className="flex h-screen w-full overflow-hidden">
      <Sidebar
        health={health}
        stats={stats}
        user={authUser}
        onPickSuggestion={handlePickSuggestion}
        onNewChat={handleNewChat}
        onSignOut={handleSignOut}
      />

      <main className="relative flex h-full flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-ink-900/8 px-6 py-4">
          <div className="flex items-center gap-4">
            <div>
              <h1 className="font-serif text-xl font-semibold text-ink-900">
                Manufacturing Copilot
              </h1>
              <p className="text-xs text-ink-500">
                Auto-correct · interactive clarifications · grounded in your
                documents + knowledge graph
              </p>
            </div>
            <nav className="ml-2 hidden gap-1 rounded-full border border-ink-900/10 bg-white p-1 shadow-soft md:flex">
              <TabButton
                active={effectiveTab === "chat"}
                onClick={() => setActiveTab("chat")}
              >
                💬 Chat
              </TabButton>
              <TabButton
                active={effectiveTab === "requests"}
                onClick={() => setActiveTab("requests")}
              >
                📊 My Requests
              </TabButton>
              {showApprovalsTab ? (
                <TabButton
                  active={effectiveTab === "approvals"}
                  onClick={() => setActiveTab("approvals")}
                >
                  🛡️ Approvals
                </TabButton>
              ) : null}
            </nav>
          </div>
          <button
            onClick={handleNewChat}
            className="rounded-full border border-ink-900/10 bg-white px-3.5 py-1.5 text-xs font-medium text-ink-700 shadow-soft hover:bg-cream-50 md:hidden"
          >
            New chat
          </button>
        </header>

        {/* Mobile-friendly tabs (hidden on md+ where they live in the header). */}
        <nav className="flex gap-1 border-b border-ink-900/8 bg-white px-4 py-2 md:hidden">
          <TabButton
            active={effectiveTab === "chat"}
            onClick={() => setActiveTab("chat")}
          >
            💬 Chat
          </TabButton>
          <TabButton
            active={effectiveTab === "requests"}
            onClick={() => setActiveTab("requests")}
          >
            📊 My Requests
          </TabButton>
          {showApprovalsTab ? (
            <TabButton
              active={effectiveTab === "approvals"}
              onClick={() => setActiveTab("approvals")}
            >
              🛡️ Approvals
            </TabButton>
          ) : null}
        </nav>

        {effectiveTab === "chat" ? (
          <>
            <div ref={scrollerRef} className="flex-1 overflow-y-auto">
              <div className="mx-auto flex w-full flex-col gap-3 py-6">
                {turns.length === 0 ? (
                  <EmptyState onPick={handlePickSuggestion} />
                ) : (
                  turns.map((t, i) => <ChatMessage key={i} turn={t} />)
                )}
                {busy ? <TypingBubble /> : null}
                {error ? (
                  <div className="mx-auto max-w-2xl px-3">
                    <div className="my-2 rounded-bubble border border-rose-200 bg-rose-50/80 px-4 py-2.5 text-[13.5px] text-rose-900">
                      {error}
                    </div>
                  </div>
                ) : null}
                {pendingThreadId ? (
                  <ApprovalBanner
                    threadId={pendingThreadId}
                    detail={pendingDetail}
                    currentUser={authUser}
                    onResolved={() => {
                      setPendingThreadId(null);
                      setPendingDetail(null);
                      setDashboardRefreshKey((k) => k + 1);
                      api
                        .getSession(sessionId)
                        .then((s) => {
                          setTurns(s.state.turns);
                          setAwaitingPrompt(s.state.awaiting_prompt);
                        })
                        .catch(() => {});
                    }}
                  />
                ) : null}
                <div className="h-24" />
              </div>
            </div>
            <ChatComposer
              value={input}
              onChange={setInput}
              onSubmit={handleSubmit}
              placeholder={placeholder}
              disabled={busy || !health?.ready || Boolean(pendingThreadId)}
            />
          </>
        ) : effectiveTab === "approvals" && showApprovalsTab ? (
          <div className="flex-1 overflow-y-auto">
            <ApprovalsTab user={authUser} refreshKey={dashboardRefreshKey} />
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto">
            <MyRequestsDashboard
              user={authUser}
              refreshKey={dashboardRefreshKey}
            />
          </div>
        )}
      </main>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-full px-3 py-1.5 text-[12.5px] font-semibold transition ${
        active
          ? "bg-copper-500 text-cream-50 shadow-soft"
          : "text-ink-600 hover:bg-cream-50"
      }`}
    >
      {children}
    </button>
  );
}

function ApprovalBanner({
  threadId,
  detail,
  currentUser,
  onResolved,
}: {
  threadId: string;
  detail: ApprovalSnapshot | null;
  currentUser: AuthUser | null;
  onResolved: () => void;
}) {
  const drivers = detail?.risk?.drivers ?? [];
  const score = detail?.risk?.score;
  const requiredRoles = detail?.required_roles ?? [];
  const makerUserId = detail?.maker_user_id ?? null;
  const isMaker =
    !!currentUser &&
    !!makerUserId &&
    currentUser.user_id.toLowerCase() === makerUserId.toLowerCase();
  const roleAllowed =
    !!currentUser && requiredRoles.includes(currentUser.role);
  const canApprove =
    detail?.can_current_user_approve ?? (roleAllowed && !isMaker);

  const [comments, setComments] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const submit = async (approved: boolean) => {
    if (submitting) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      await api.resumeApproval(threadId, {
        approved,
        comments: comments || undefined,
      });
      onResolved();
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  const approvalsUrl = `${STREAMLIT_ORIGIN}/?tab=approvals`;

  return (
    <div className="mx-auto max-w-2xl px-3">
      <div className="my-2 rounded-bubble border border-amber-300 bg-amber-50/80 px-4 py-3 text-[13.5px] text-amber-900 shadow-soft">
        <div className="flex items-start gap-3">
          <span className="mt-0.5 text-lg" aria-hidden>
            ⏸️
          </span>
          <div className="flex-1">
            <div className="font-semibold">
              Workflow paused for human approval
            </div>
            <div className="mt-0.5 text-[12.5px] text-amber-800/90">
              Thread <code className="font-mono">{threadId}</code>
              {typeof score === "number" ? (
                <> · risk score {score.toFixed(2)}</>
              ) : null}
            </div>
            {drivers.length > 0 ? (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {drivers.map((d) => (
                  <span
                    key={d}
                    className="rounded-full border border-amber-300 bg-white/70 px-2 py-0.5 text-[11px] font-medium text-amber-900"
                  >
                    {d}
                  </span>
                ))}
              </div>
            ) : null}
            {requiredRoles.length > 0 ? (
              <div className="mt-2 text-[12px] text-amber-800/90">
                Required role(s):{" "}
                {requiredRoles.map((r, i) => (
                  <span key={r}>
                    {i > 0 ? " · " : ""}
                    <code className="rounded bg-white/70 px-1.5 py-0.5 text-[11px] font-semibold">
                      {r}
                    </code>
                  </span>
                ))}
              </div>
            ) : null}
            {makerUserId ? (
              <div className="mt-1 text-[11.5px] text-amber-800/80">
                Submitted by <code>{makerUserId}</code>
                {isMaker ? " (you — cannot self-approve)" : ""}
              </div>
            ) : null}

            {/* ── Decision controls ─────────────────────────────────── */}
            {canApprove ? (
              <div className="mt-3 rounded-lg border border-amber-300 bg-white/70 p-2.5">
                <textarea
                  value={comments}
                  onChange={(e) => setComments(e.target.value)}
                  placeholder="Optional comments for the audit log…"
                  rows={2}
                  className="block w-full resize-none rounded border border-amber-200 bg-white px-2 py-1 text-[12.5px] text-ink-800 placeholder-ink-400 focus:border-amber-500 focus:outline-none"
                />
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    disabled={submitting}
                    onClick={() => submit(true)}
                    className="rounded-full bg-emerald-600 px-3 py-1 text-[12px] font-semibold text-white hover:bg-emerald-700 disabled:bg-ink-300"
                  >
                    {submitting ? "Submitting…" : "Approve"}
                  </button>
                  <button
                    type="button"
                    disabled={submitting}
                    onClick={() => submit(false)}
                    className="rounded-full bg-rose-600 px-3 py-1 text-[12px] font-semibold text-white hover:bg-rose-700 disabled:bg-ink-300"
                  >
                    Reject
                  </button>
                  <a
                    href={approvalsUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="text-[11.5px] text-amber-800 underline-offset-2 hover:underline"
                  >
                    Or use the Streamlit console →
                  </a>
                </div>
                {submitError ? (
                  <div className="mt-2 rounded border border-rose-200 bg-rose-50 px-2 py-1 text-[12px] text-rose-900">
                    {submitError}
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="mt-2.5 rounded-lg border border-amber-200 bg-white/60 px-3 py-2 text-[12.5px] text-amber-800/90">
                {isMaker ? (
                  <>
                    You submitted this request — segregation of duties prevents
                    you from approving it. Hand it to a colleague with one of
                    the required roles above.
                  </>
                ) : currentUser ? (
                  <>
                    Your role{" "}
                    <code className="rounded bg-amber-50 px-1.5 py-0.5 text-[11px] font-semibold">
                      {currentUser.role}
                    </code>{" "}
                    is not on this approval's allow-list. Ask a holder of one
                    of the required roles above to resolve it.
                  </>
                ) : (
                  <>Sign in to action this approval.</>
                )}
                <div className="mt-1.5">
                  <a
                    href={approvalsUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="text-[12px] font-semibold text-amber-700 hover:underline"
                  >
                    Open Approvals tab →
                  </a>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (s: string) => void }) {
  const examples = [
    "What is the OEE target for Plant A in Q1 2026?",
    "Pump P-203 has high vibration alarm ALM-P001. Cause and fix?",
    "maintanance schedul for spindle bearings",
    "Why did CNC Line 4 shut down in February?",
  ];
  return (
    <div className="mx-auto mt-10 max-w-2xl px-6 text-center">
      <div className="mb-4 text-4xl">🏭</div>
      <h2 className="font-serif text-2xl text-ink-900">
        How can I help with the line today?
      </h2>
      <p className="mt-2 text-sm text-ink-500">
        I can troubleshoot equipment, look up KPIs, compare suppliers, and
        explain procedures — all grounded in your manufacturing documents.
      </p>
      <div className="mt-7 grid grid-cols-1 gap-2 sm:grid-cols-2">
        {examples.map((s) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="rounded-2xl border border-ink-900/8 bg-white/70 px-4 py-3 text-left text-[13px] leading-snug text-ink-700 shadow-soft transition hover:border-copper-500/40 hover:bg-white"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
