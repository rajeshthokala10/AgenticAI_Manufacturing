"use client";

import { useEffect, useRef } from "react";

type Props = {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  placeholder?: string;
  disabled?: boolean;
};

export function ChatComposer({
  value,
  onChange,
  onSubmit,
  placeholder = "Ask anything about your manufacturing operations…",
  disabled = false,
}: Props) {
  const taRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow up to ~6 lines
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 220)}px`;
  }, [value]);

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && value.trim()) onSubmit();
    }
  };

  return (
    <div className="pointer-events-none sticky bottom-0 z-10 w-full bg-gradient-to-t from-cream-50 via-cream-50/95 to-transparent pb-6 pt-6">
      <div className="pointer-events-auto mx-auto w-full max-w-3xl px-3">
        <div className="flex items-end gap-2 rounded-3xl border border-ink-900/10 bg-white px-4 py-3 shadow-soft focus-within:border-copper-500/60 focus-within:ring-2 focus-within:ring-copper-500/20">
          <textarea
            ref={taRef}
            value={value}
            disabled={disabled}
            placeholder={placeholder}
            rows={1}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKey}
            className="block w-full resize-none border-0 bg-transparent text-[15px] leading-relaxed text-ink-800 placeholder-ink-400 focus:outline-none focus:ring-0 disabled:opacity-50"
          />
          <button
            type="button"
            onClick={onSubmit}
            disabled={disabled || !value.trim()}
            className="shrink-0 rounded-full bg-copper-500 px-4 py-2 text-sm font-semibold text-cream-50 transition hover:bg-copper-600 disabled:cursor-not-allowed disabled:bg-ink-300"
            title="Send (Enter)"
          >
            Send
          </button>
        </div>
        <div className="mt-2 text-center text-[11px] text-ink-400">
          Press <kbd className="rounded bg-cream-200 px-1.5 py-0.5">Enter</kbd> to
          send · <kbd className="rounded bg-cream-200 px-1.5 py-0.5">Shift</kbd>+
          <kbd className="rounded bg-cream-200 px-1.5 py-0.5">Enter</kbd> for new
          line · type <code className="text-ink-500">skip</code> to bypass a
          clarifying question
        </div>
      </div>
    </div>
  );
}
