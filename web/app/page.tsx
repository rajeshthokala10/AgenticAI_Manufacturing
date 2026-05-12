"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ChatMessage, TypingBubble } from "@/components/ChatMessage";
import { ChatComposer } from "@/components/ChatComposer";
import { Sidebar } from "@/components/Sidebar";
import {
  api,
  type ChatTurn,
  type HealthResponse,
  type StatsResponse,
} from "@/lib/api";

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

  const scrollerRef = useRef<HTMLDivElement>(null);

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
    setInput("");
  };

  const placeholder = awaitingPrompt
    ? `Answer: ${awaitingPrompt}  (or type 'skip')`
    : "Ask anything about your manufacturing operations…";

  return (
    <div className="flex h-screen w-full overflow-hidden">
      <Sidebar
        health={health}
        stats={stats}
        onPickSuggestion={handlePickSuggestion}
        onNewChat={handleNewChat}
      />

      <main className="relative flex h-full flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-ink-900/8 px-6 py-4">
          <div>
            <h1 className="font-serif text-xl font-semibold text-ink-900">
              Manufacturing Copilot
            </h1>
            <p className="text-xs text-ink-500">
              Auto-correct · interactive clarifications · grounded in your
              documents + knowledge graph
            </p>
          </div>
          <button
            onClick={handleNewChat}
            className="rounded-full border border-ink-900/10 bg-white px-3.5 py-1.5 text-xs font-medium text-ink-700 shadow-soft hover:bg-cream-50 md:hidden"
          >
            New chat
          </button>
        </header>

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
            <div className="h-24" />
          </div>
        </div>

        <ChatComposer
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          placeholder={placeholder}
          disabled={busy || !health?.ready}
        />
      </main>
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
