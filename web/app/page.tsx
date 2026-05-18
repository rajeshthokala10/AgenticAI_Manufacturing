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
  DEFAULT_DOMAIN,
  FALLBACK_CATALOG,
  type AccessPolicyResponse,
  type ApprovalSnapshot,
  type AuthUser,
  type ChatTurn,
  type DiagnosticResponse,
  type Domain,
  type DomainCatalog,
  type DomainCatalogEntry,
  type HealthResponse,
  type LlmBackendStatus,
  type StatsResponse,
} from "@/lib/api";

type ActiveTab = "chat" | "troubleshoot" | "requests" | "approvals";

// Default Streamlit port from run.sh. Override at build time with
// NEXT_PUBLIC_STREAMLIT_ORIGIN if your deployment uses a non-default host.
const STREAMLIT_ORIGIN =
  process.env.NEXT_PUBLIC_STREAMLIT_ORIGIN ?? "http://localhost:8501";

const SESSION_KEY = "mfg-graphrag-session-id";
const DOMAIN_KEY = "mfg-graphrag-active-domain";

function makeSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID().replace(/-/g, "");
  }
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export default function ChatPage() {
  const [sessionId, setSessionId] = useState<string>("");
  const [activeDomain, setActiveDomain] = useState<Domain>(DEFAULT_DOMAIN);
  const [domainCatalog, setDomainCatalog] =
    useState<DomainCatalog>(FALLBACK_CATALOG);
  const [llmBackend, setLlmBackend] = useState<LlmBackendStatus | null>(null);
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
  const [accessPolicy, setAccessPolicy] =
    useState<AccessPolicyResponse | null>(null);
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
    setAccessPolicy(null);
    setTurns([]);
    setAwaitingPrompt(null);
    setPendingThreadId(null);
    setPendingDetail(null);
  }, []);

  // Pull the document-ACL view whenever the signed-in user changes so the
  // sidebar can render the "Knowledge access tier" badge. The endpoint is
  // tolerant of anonymous callers, but we only show the badge once we
  // actually have a user.
  useEffect(() => {
    let cancelled = false;
    if (!authUser) {
      setAccessPolicy(null);
      return () => {
        cancelled = true;
      };
    }
    api
      .accessPolicy()
      .then((p) => {
        if (!cancelled) setAccessPolicy(p);
      })
      .catch(() => {
        if (!cancelled) setAccessPolicy(null);
      });
    return () => {
      cancelled = true;
    };
  }, [authUser]);

  // Bootstrap session id from localStorage (or create one).
  useEffect(() => {
    let id =
      typeof window !== "undefined" ? localStorage.getItem(SESSION_KEY) : null;
    if (!id) {
      id = makeSessionId();
      if (typeof window !== "undefined") localStorage.setItem(SESSION_KEY, id);
    }
    setSessionId(id);

    // Restore last-selected domain. The catalog effect below will trim this
    // back to a valid id if the saved value is from a removed domain.
    if (typeof window !== "undefined") {
      const d = localStorage.getItem(DOMAIN_KEY);
      if (d) setActiveDomain(d);
    }
  }, []);

  // Fetch the domain catalog once on mount. Any new domain dropped into
  // ``schemas/`` shows up here without a frontend rebuild.
  useEffect(() => {
    let cancelled = false;
    api
      .domains()
      .then((cat) => {
        if (cancelled) return;
        setDomainCatalog(cat);
        const valid = cat.domains.some((d) => d.id === activeDomain);
        if (!valid) setActiveDomain(cat.default);
      })
      .catch(() => {
        /* keep FALLBACK_CATALOG */
      });
    // Same one-shot pattern for the LLM backend snapshot.
    api.llmBackendStatus().then((s) => {
      if (!cancelled) setLlmBackend(s);
    }).catch(() => {
      /* keep null — pill renders a stub */
    });
    return () => {
      cancelled = true;
    };
    // Run once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Flip the LLM backend at runtime — POSTs to FastAPI, which mutates the
  // process-wide state. Next chat / diagnostic / onboarding call uses the
  // new backend automatically; no page reload needed.
  const handleLlmBackendChange = useCallback(
    async (next: "auto" | "local" | "cloud") => {
      try {
        const updated = await api.setLlmBackend(next);
        setLlmBackend(updated);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [],
  );

  // Persist domain selection.
  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(DOMAIN_KEY, activeDomain);
    }
  }, [activeDomain]);

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
            const s = await api.stats(activeDomain);
            if (!cancelled) setStats(s);
          } catch {
            /* ignore */
          }
          try {
            const sess = await api.getSession(sessionId, activeDomain);
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
  }, [sessionId, activeDomain]);

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
        const res = await api.chat(sessionId, trimmed, activeDomain);
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
    [sessionId, busy, activeDomain],
  );

  const handleSubmit = () => send(input);

  // Switching domains clears the visible transcript so the user is never
  // confused about which corpus they're chatting against. The server-side
  // session for the new domain is fetched by the health/stats effect.
  const handleDomainSwitch = useCallback((next: Domain) => {
    if (next === activeDomain) return;
    setActiveDomain(next);
    setTurns([]);
    setAwaitingPrompt(null);
    setPendingThreadId(null);
    setPendingDetail(null);
    setError(null);
  }, [activeDomain]);
  const handlePickSuggestion = (s: string) => {
    setInput(s);
    setTimeout(() => send(s), 0);
  };
  const handleNewChat = async () => {
    if (!sessionId) return;
    setError(null);
    try {
      await api.reset(sessionId, activeDomain);
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

  // Resolve the active domain's catalog entry so the header + tabs render
  // labels / emojis / copy from schemas/<domain>.yaml. Falls back to a
  // generic copilot label when the catalog hasn't loaded yet.
  const activeEntry: DomainCatalogEntry | undefined =
    domainCatalog.domains.find((d) => d.id === activeDomain);
  const activeLabel = activeEntry?.label ?? "Copilot";
  const activeEmoji = activeEntry?.emoji ?? "🏭";

  return (
    <div className="flex h-screen w-full overflow-hidden">
      <Sidebar
        health={health}
        stats={stats}
        user={authUser}
        accessPolicy={accessPolicy}
        onPickSuggestion={handlePickSuggestion}
        onNewChat={handleNewChat}
        onSignOut={handleSignOut}
        activeDomain={activeDomain}
        catalog={domainCatalog}
      />

      <main className="relative flex h-full flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-ink-900/8 px-6 py-4">
          <div className="flex items-center gap-4">
            <div>
              <h1 className="font-serif text-xl font-semibold text-ink-900">
                <span className="mr-1.5">{activeEmoji}</span>
                {activeLabel} Copilot
              </h1>
              <p className="text-xs text-ink-500">
                Auto-correct · interactive clarifications · grounded in the
                {" "}{activeLabel.toLowerCase()} corpus and knowledge graph
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
                active={effectiveTab === "troubleshoot"}
                onClick={() => setActiveTab("troubleshoot")}
              >
                🔧 Troubleshooting Copilot
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
          <div className="flex items-center gap-3">
            <LlmBackendPill
              status={llmBackend}
              onChange={handleLlmBackendChange}
            />
            <DomainSwitcher
              active={activeDomain}
              catalog={domainCatalog}
              onChange={handleDomainSwitch}
            />
            <button
              onClick={handleNewChat}
              className="rounded-full border border-ink-900/10 bg-white px-3.5 py-1.5 text-xs font-medium text-ink-700 shadow-soft hover:bg-cream-50 md:hidden"
            >
              New chat
            </button>
          </div>
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
            active={effectiveTab === "troubleshoot"}
            onClick={() => setActiveTab("troubleshoot")}
          >
            🔧 Troubleshoot
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
                  <EmptyState
                    onPick={handlePickSuggestion}
                    domain={activeDomain}
                    catalog={domainCatalog}
                  />
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
        ) : effectiveTab === "troubleshoot" ? (
          <div className="flex-1 overflow-y-auto">
            <TroubleshootingPanel
              domain={activeDomain}
              entry={activeEntry}
              sessionId={sessionId}
            />
          </div>
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

// Empty state copy is pulled from the catalog entry the backend returned
// (sourced from schemas/<domain>.yaml). No domain literals here, so adding
// a new domain doesn't require editing this file.
function EmptyState({
  onPick,
  domain,
  catalog,
}: {
  onPick: (s: string) => void;
  domain: Domain;
  catalog: DomainCatalog;
}) {
  const entry: DomainCatalogEntry | undefined = catalog.domains.find(
    (d) => d.id === domain,
  );
  const emoji = entry?.emoji ?? "📁";
  const heading =
    entry?.empty_state?.heading ?? `How can I help with ${entry?.label ?? domain} today?`;
  const blurb =
    entry?.empty_state?.blurb ??
    "Ask anything grounded in this domain's documents — retrieval is restricted to this domain's corpus and knowledge graph.";
  const examples = (entry?.examples ?? []).slice(0, 4);
  return (
    <div className="mx-auto mt-10 max-w-2xl px-6 text-center">
      <div className="mb-4 text-4xl">{emoji}</div>
      <h2 className="font-serif text-2xl text-ink-900">{heading}</h2>
      <p className="mt-2 text-sm text-ink-500">{blurb}</p>
      {examples.length > 0 ? (
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
      ) : null}
    </div>
  );
}

// ── Domain affordances ────────────────────────────────────────────────────
function DomainSwitcher({
  active,
  catalog,
  onChange,
}: {
  active: Domain;
  catalog: DomainCatalog;
  onChange: (next: Domain) => void;
}) {
  if (catalog.domains.length <= 1) return null;
  return (
    <div className="hidden items-center gap-1 rounded-full border border-ink-900/10 bg-white p-1 shadow-soft md:flex">
      {catalog.domains.map((d) => {
        const isActive = d.id === active;
        return (
          <button
            key={d.id}
            onClick={() => onChange(d.id)}
            className="rounded-full px-3 py-1 text-[12px] font-semibold transition"
            style={{
              background: isActive ? `${d.color}22` : "transparent",
              color: isActive ? d.color : "#475569",
              border: isActive ? `1px solid ${d.color}66` : "1px solid transparent",
            }}
            aria-pressed={isActive}
            aria-label={`Switch to ${d.label} domain`}
          >
            <span className="mr-1">{d.emoji}</span>
            {d.label}
          </button>
        );
      })}
    </div>
  );
}

// `DomainPill` is currently unused inside this file — kept as a local
// component for future use (e.g. per-evidence badge). Not exported because
// Next.js page files only allow `default` + a fixed set of metadata exports.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function DomainPill({
  domain,
  catalog,
}: {
  domain: Domain;
  catalog: DomainCatalog;
}) {
  const entry = catalog.domains.find((d) => d.id === domain);
  if (!entry) return null;
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10.5px] font-semibold uppercase tracking-wide"
      style={{
        background: `${entry.color}1A`,
        color: entry.color,
        border: `1px solid ${entry.color}44`,
      }}
    >
      <span>{entry.emoji}</span>
      {entry.label}
    </span>
  );
}

// ── Troubleshooting Copilot ────────────────────────────────────────────────
// Mirrors Streamlit's Diagnostic tab: one POST → /api/diagnostic, blocking
// while the LangGraph pipeline runs, then renders the resulting answer +
// evidence + (when present) procedure / cause ranking. No SSE, same backend
// path (``pipe.diagnostic()``).
function TroubleshootingPanel({
  domain,
  entry,
  sessionId: _sessionId, // currently unused; kept for parity with the Chat tab
}: {
  domain: Domain;
  entry: DomainCatalogEntry | undefined;
  sessionId: string;
}) {
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<DiagnosticResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsedSec, setElapsedSec] = useState(0);

  const examples = (entry?.examples ?? []).slice(0, 4);
  const placeholder =
    entry?.placeholder ||
    "Describe the issue you want to troubleshoot — equipment, symptom, anything you've already tried.";

  // Keep a live "elapsed: Xs" counter ticking so the user knows the
  // request is still in flight (Ollama diagnostic runs can take 5–6 min).
  useEffect(() => {
    if (!busy) return;
    setElapsedSec(0);
    const t = setInterval(() => setElapsedSec((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [busy]);

  const submit = useCallback(
    async (query: string) => {
      const trimmed = query.trim();
      if (!trimmed || busy) return;
      setBusy(true);
      setResult(null);
      setError(null);
      try {
        const r = await api.diagnostic(trimmed, domain);
        setResult(r);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [busy, domain],
  );

  const evidence = result?.evidence ?? [];

  return (
    <div className="mx-auto max-w-3xl px-6 py-8">
      <header className="mb-4">
        <h2 className="font-serif text-2xl text-ink-900">
          🔧 Troubleshooting Copilot
          <span
            className="ml-2 align-middle text-[11px] font-semibold uppercase"
            style={{ color: entry?.color ?? "#475569" }}
          >
            {entry?.label ?? domain}
          </span>
        </h2>
        <p className="mt-1 text-sm text-ink-500">
          One-shot diagnostic run against the {entry?.label?.toLowerCase() ?? domain} corpus.
        </p>
      </header>

      <div className="rounded-2xl border border-ink-900/8 bg-white p-4 shadow-soft">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          rows={4}
          placeholder={placeholder}
          className="w-full resize-none rounded-xl border border-ink-900/10 bg-cream-50/40 p-3 text-[14px] text-ink-900 outline-none focus:border-copper-500/40"
          disabled={busy}
        />
        <div className="mt-3 flex items-center justify-between">
          <div className="flex flex-wrap gap-1.5">
            {examples.map((s) => (
              <button
                key={s}
                onClick={() => {
                  setInput(s);
                  setTimeout(() => submit(s), 0);
                }}
                disabled={busy}
                className="rounded-full border border-ink-900/8 bg-white px-3 py-1 text-[11.5px] text-ink-700 hover:bg-cream-50 disabled:opacity-50"
              >
                {s}
              </button>
            ))}
          </div>
          <button
            onClick={() => submit(input)}
            disabled={busy || !input.trim()}
            className="rounded-full px-4 py-1.5 text-[13px] font-semibold text-cream-50 shadow-soft disabled:opacity-50"
            style={{ background: entry?.color ?? "#B45309" }}
          >
            {busy ? `Diagnosing… ${elapsedSec}s` : "Diagnose"}
          </button>
        </div>
      </div>

      {busy ? (
        <div className="mt-5 flex items-center gap-3 rounded-2xl border border-ink-900/8 bg-white p-4 shadow-soft">
          <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-copper-500 border-t-transparent" />
          <div className="text-[13px] text-ink-700">
            Running the diagnostic pipeline (clarify → retrieve → cause-rank →
            answer → critic). On local Ollama this takes 5–6 minutes.
          </div>
        </div>
      ) : null}

      {error ? (
        <div className="mt-5 rounded-xl border border-rose-200 bg-rose-50/80 px-4 py-2.5 text-[13.5px] text-rose-900">
          {error}
        </div>
      ) : null}

      {result?.answer ? (
        <div className="mt-5 rounded-2xl border border-ink-900/8 bg-white p-5 shadow-soft">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-ink-500">
            Diagnostic answer
          </div>
          <div className="prose prose-sm max-w-none whitespace-pre-wrap text-ink-900">
            {result.answer}
          </div>
          {evidence.length > 0 ? (
            <div className="mt-5">
              <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-ink-500">
                Evidence ({evidence.length})
              </div>
              <ol className="flex flex-col gap-2">
                {evidence.slice(0, 5).map((ev, i) => {
                  const meta = (ev.metadata ?? {}) as Record<string, unknown>;
                  const src = String(meta.source ?? meta.source_file ?? "unknown");
                  const page = meta.page ? ` · page ${String(meta.page)}` : "";
                  const score = ev.vector_score ?? ev.rrf_score ?? 0;
                  return (
                    <li
                      key={i}
                      className="rounded-lg border border-ink-900/8 bg-cream-50/30 px-3 py-2 text-[12.5px]"
                      style={{ borderLeftWidth: 3, borderLeftColor: entry?.color ?? "#B45309" }}
                    >
                      <div className="flex items-center justify-between text-[11px] text-ink-500">
                        <span className="truncate">{src.split("/").pop()}{page}</span>
                        <span className="font-semibold text-ink-700">{Number(score).toFixed(3)}</span>
                      </div>
                      <div className="mt-1 line-clamp-3 text-ink-800">
                        {(ev.text ?? "").slice(0, 360)}
                      </div>
                    </li>
                  );
                })}
              </ol>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

// ── LLM backend pill ──────────────────────────────────────────────────────
// Small chip in the header that shows the active LLM backend (☁️ cloud /
// 💻 local) with the answer-task model under it. Click cycles auto →
// local → cloud → auto. Disabled with a tooltip if the cloud key is
// missing.
function LlmBackendPill({
  status,
  onChange,
}: {
  status: LlmBackendStatus | null;
  onChange: (next: "auto" | "local" | "cloud") => void;
}) {
  if (!status) {
    return (
      <div className="hidden items-center gap-2 rounded-full border border-ink-900/10 bg-white px-3 py-1 text-[11px] text-ink-500 shadow-soft md:flex">
        <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-amber-400" />
        LLM…
      </div>
    );
  }
  const isCloud = status.active === "cloud";
  const color = isCloud ? "#0EA5E9" : "#10B981";
  const emoji = isCloud ? "☁️" : "💻";
  const label = isCloud ? "Cloud" : "Local";
  const cycle: Record<"auto" | "local" | "cloud", "auto" | "local" | "cloud"> = {
    auto: "local",
    local: status.openai_key_valid ? "cloud" : "auto",
    cloud: "auto",
  };
  const next = cycle[status.raw];
  const disabled = !status.openai_key_valid && next === "cloud";
  return (
    <button
      onClick={() => !disabled && onChange(next)}
      disabled={disabled}
      className="hidden items-center gap-1.5 rounded-full border bg-white px-3 py-1 text-[11px] font-semibold shadow-soft transition hover:bg-cream-50 disabled:cursor-not-allowed disabled:opacity-60 md:flex"
      style={{ color, borderColor: `${color}55` }}
      title={
        disabled
          ? "Cloud requires a valid OPENAI_API_KEY"
          : `LLM: ${status.raw} (resolves to ${status.active}). ` +
            `Click to switch to ${next}. ` +
            `Answer model: ${status.per_task.answer}.`
      }
      aria-label={`LLM backend: ${status.active}, click to cycle`}
    >
      <span>{emoji}</span>
      <span>{label}</span>
      <span className="hidden text-ink-500 lg:inline">
        ·{" "}<code className="text-[10px]">{status.per_task.answer}</code>
      </span>
    </button>
  );
}
