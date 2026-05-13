"use client";

import { useEffect, useState } from "react";
import { api, setAuthToken, type AuthUser, type Role } from "@/lib/api";

type Props = {
  onSignIn: (user: AuthUser) => void;
};

// Demo accounts seeded by `core/auth_store.py`. Surfacing them in the UI
// makes the role-routing demo immediate; remove this block before any real
// deployment.
const DEMO_CREDS: Array<{ user_id: string; password: string; tagline: string }> = [
  { user_id: "alice@plant.local",         password: "operator123",   tagline: "Operator — submits queries, can never approve" },
  { user_id: "bob.supervisor@plant.local", password: "supervisor123", tagline: "Shift Supervisor — routine PMs, low-confidence answers" },
  { user_id: "carol.eng@plant.local",     password: "engineer123",   tagline: "Maintenance Engineer — LOTO, Class-A equipment" },
  { user_id: "dave.ehs@plant.local",      password: "ehs123",        tagline: "EHS Officer — all safety / permit-to-work" },
  { user_id: "eve.buyer@plant.local",     password: "buyer123",      tagline: "Buyer — POs ≤ $10k" },
  { user_id: "frank.proc@plant.local",    password: "procurement123", tagline: "Procurement Manager — POs > $10k, single-source" },
  { user_id: "henry.pm@plant.local",      password: "plant123",      tagline: "Plant Manager — capex, fatality, regulatory" },
];

export function AuthGate({ onSignIn }: Props) {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [userId, setUserId] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("operator");
  const [displayName, setDisplayName] = useState("");
  const [roles, setRoles] = useState<Role[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listRoles()
      .then((r) => setRoles(r.roles))
      .catch(() => setRoles([]));
  }, []);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy) return;
    setError(null);
    setBusy(true);
    try {
      const fn = mode === "login" ? api.login : api.signup;
      const body =
        mode === "login"
          ? { user_id: userId, password }
          : { user_id: userId, password, role, display_name: displayName };
      const res = await fn(body as Parameters<typeof fn>[0]);
      setAuthToken(res.token);
      onSignIn(res.user);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const pickDemo = (cred: (typeof DEMO_CREDS)[number]) => {
    setMode("login");
    setUserId(cred.user_id);
    setPassword(cred.password);
    setError(null);
  };

  return (
    <div className="flex h-screen w-full items-center justify-center bg-cream-50 px-4">
      <div className="grid w-full max-w-5xl grid-cols-1 gap-6 md:grid-cols-2">
        {/* ── Form column ──────────────────────────────────────────── */}
        <form
          onSubmit={submit}
          className="rounded-2xl border border-ink-900/8 bg-white p-6 shadow-soft"
        >
          <div className="mb-1 flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-copper-500 text-base font-bold text-cream-50">
              🏭
            </div>
            <span className="font-serif text-lg text-ink-900">
              Manufacturing Copilot
            </span>
          </div>
          <h2 className="mt-3 font-serif text-2xl text-ink-900">
            {mode === "login" ? "Sign in to continue" : "Create an account"}
          </h2>
          <p className="mt-1 text-[13px] text-ink-500">
            Approvals are role-gated. Pick the role that matches your
            responsibility on the line.
          </p>

          <div className="mt-5 space-y-3">
            <label className="block text-[12px] font-medium text-ink-700">
              User ID (email)
              <input
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="alice@plant.local"
                autoComplete="username"
                required
                className="mt-1 block w-full rounded-lg border border-ink-900/10 bg-white px-3 py-2 text-[14px] text-ink-800 placeholder-ink-400 focus:border-copper-500/60 focus:outline-none focus:ring-2 focus:ring-copper-500/20"
              />
            </label>
            <label className="block text-[12px] font-medium text-ink-700">
              Password
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                required
                minLength={6}
                className="mt-1 block w-full rounded-lg border border-ink-900/10 bg-white px-3 py-2 text-[14px] text-ink-800 placeholder-ink-400 focus:border-copper-500/60 focus:outline-none focus:ring-2 focus:ring-copper-500/20"
              />
            </label>

            {mode === "signup" ? (
              <>
                <label className="block text-[12px] font-medium text-ink-700">
                  Display name
                  <input
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    placeholder="Optional"
                    className="mt-1 block w-full rounded-lg border border-ink-900/10 bg-white px-3 py-2 text-[14px] text-ink-800 placeholder-ink-400 focus:border-copper-500/60 focus:outline-none focus:ring-2 focus:ring-copper-500/20"
                  />
                </label>
                <label className="block text-[12px] font-medium text-ink-700">
                  Role
                  <select
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                    className="mt-1 block w-full rounded-lg border border-ink-900/10 bg-white px-3 py-2 text-[14px] text-ink-800 focus:border-copper-500/60 focus:outline-none focus:ring-2 focus:ring-copper-500/20"
                  >
                    {roles.map((r) => (
                      <option key={r.id} value={r.id}>
                        {r.label} — {r.is_maker ? "maker" : "checker"}
                      </option>
                    ))}
                  </select>
                  <span className="mt-1 block text-[11px] text-ink-500">
                    {roles.find((r) => r.id === role)?.description}
                  </span>
                </label>
              </>
            ) : null}

            {error ? (
              <div className="rounded-lg border border-rose-200 bg-rose-50/80 px-3 py-2 text-[12.5px] text-rose-900">
                {error}
              </div>
            ) : null}

            <button
              type="submit"
              disabled={busy || !userId || !password}
              className="w-full rounded-full bg-copper-500 px-4 py-2.5 text-[14px] font-semibold text-cream-50 transition hover:bg-copper-600 disabled:cursor-not-allowed disabled:bg-ink-300"
            >
              {busy ? "Working…" : mode === "login" ? "Sign in" : "Create account"}
            </button>

            <div className="text-center text-[12px] text-ink-500">
              {mode === "login" ? (
                <>
                  Need an account?{" "}
                  <button
                    type="button"
                    className="font-semibold text-copper-600 hover:underline"
                    onClick={() => {
                      setMode("signup");
                      setError(null);
                    }}
                  >
                    Sign up
                  </button>
                </>
              ) : (
                <>
                  Have an account?{" "}
                  <button
                    type="button"
                    className="font-semibold text-copper-600 hover:underline"
                    onClick={() => {
                      setMode("login");
                      setError(null);
                    }}
                  >
                    Sign in
                  </button>
                </>
              )}
            </div>
          </div>
        </form>

        {/* ── Demo-credentials column ──────────────────────────────── */}
        <div className="rounded-2xl border border-ink-900/8 bg-cream-100/50 p-6 shadow-soft">
          <h3 className="font-serif text-lg text-ink-900">
            Demo accounts (seeded)
          </h3>
          <p className="mt-1 text-[12.5px] text-ink-500">
            Click any card to fill the form. Each role demonstrates a different
            slice of the maker / checker policy.
          </p>
          <div className="mt-4 grid grid-cols-1 gap-2">
            {DEMO_CREDS.map((c) => (
              <button
                key={c.user_id}
                type="button"
                onClick={() => pickDemo(c)}
                className="rounded-xl border border-ink-900/8 bg-white px-3 py-2 text-left text-[12.5px] leading-snug text-ink-700 transition hover:border-copper-500/30 hover:bg-cream-50"
              >
                <div className="flex items-center justify-between">
                  <code className="text-[12px] font-mono text-ink-800">
                    {c.user_id}
                  </code>
                  <span className="text-[10.5px] text-ink-400">
                    pw: <code>{c.password}</code>
                  </span>
                </div>
                <div className="mt-0.5 text-[11.5px] text-ink-500">{c.tagline}</div>
              </button>
            ))}
          </div>
          <p className="mt-3 text-[11px] text-ink-400">
            Seeded passwords are intentionally weak. Change them or disable
            seeding via <code>core/auth_store.py</code> before any real
            deployment.
          </p>
        </div>
      </div>
    </div>
  );
}
