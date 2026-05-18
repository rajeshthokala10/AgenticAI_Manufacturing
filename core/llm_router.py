"""Single-file LLM routing layer — **the** source of truth for which model
each task uses under each backend.

Design intent
-------------

  * ``.env`` carries **only secrets** — namely ``OPENAI_API_KEY``.
    No model names, no backend choice, no per-task knobs live there.

  * Everything else lives in this file:

      - ``PROFILES``         which model serves each task role under
                              each backend (``local`` vs ``cloud``)
      - ``DEFAULT_BACKEND``  the boot-time default
      - ``TASKS``            the six task roles every call site uses

  * Runtime switching happens through the UI (Streamlit sidebar dropdown
    / Next.js header pill) — which calls ``set_active_backend()``.
    Persistence across restarts: edit ``DEFAULT_BACKEND`` below.

Public surface
--------------

    task_model(task: str) -> str          # resolved model for this role
    get_active_backend() -> str           # "local" | "cloud" (never "auto")
    set_active_backend(name: str) -> str  # runtime flip; returns name set
    list_profiles() -> dict               # backend → (task → model) snapshot
    call(task, system, user, **kw) -> str # convenience over llm_client.call_llm
    call_with_metrics(task, …) -> dict    # same + usage counters
    with_backend(name) -> ctx mgr         # request-scoped override
"""

from __future__ import annotations

import contextvars
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("core.llm_router")


# ─── Task taxonomy ──────────────────────────────────────────────────────────
# Six roles cover every LLM call site in the codebase today. Keep this list
# short — additional fine-grained tuning belongs in env-var overrides, not
# in the taxonomy itself.

TASKS = (
    "answer",       # user-facing final answer (chat / troubleshoot / diagnostic)
    "procedure",    # structured procedure-drafting JSON
    "critic",       # answer quality check
    "analyze",      # short classification / cause-ranking / direct baselines
    "tool",         # tool-call planning
    "onboarding",   # one-shot schema authoring (biggest model)
)


# ─── Profiles ───────────────────────────────────────────────────────────────
# Out-of-the-box defaults for the two ends of the cost/quality spectrum.
# Override per-task by setting LLM_<TASK>_MODEL — e.g. ``LLM_CRITIC_MODEL=gpt-4o``
# pins the critic to a specific model regardless of the active backend.

# ──────────────────────────────────────────────────────────────────────────
# 🎛  EDIT THIS TABLE  ◀ single source of truth for which model serves
#                       which task under which backend.
#
# Rows are task roles, columns are backends. Change a cell, restart the
# server, done — no .env edit needed. For runtime switching between
# backends without a restart, use the UI sidebar/header toggle (which
# calls ``set_active_backend`` below); the model assignments themselves
# always come from this table.
# ──────────────────────────────────────────────────────────────────────────
PROFILES: Dict[str, Dict[str, str]] = {
    "local": {
        # Ollama. Make sure each model below is `ollama pull`'d.
        "answer":     "qwen2.5:3b",
        "procedure":  "qwen2.5:3b",
        "critic":     "qwen2.5:3b",
        "analyze":    "qwen2.5:3b",
        "tool":       "qwen2.5:3b",
        # ``onboarding`` (schema authoring) is regex + structured-JSON
        # heavy; bigger model recommended. qwen2.5:14b is the smallest
        # reasonable choice — bump to llama3:70b / qwen2.5:32b if you have
        # the RAM.
        "onboarding": "qwen2.5:14b",
    },
    "cloud": {
        # OpenAI (or any OpenAI-compatible host configured at startup).
        "answer":     "gpt-4o",
        "procedure":  "gpt-4o",
        "critic":     "gpt-4o-mini",
        "analyze":    "gpt-4o-mini",
        "tool":       "gpt-4o-mini",
        "onboarding": "gpt-4o",
    },
}

# Boot-time default. ``auto`` picks ``cloud`` when ``OPENAI_API_KEY`` is
# valid, else falls back to ``local``. Change here to permanently bias the
# default (UI flips still override at runtime).
DEFAULT_BACKEND: str = "cloud"

BACKENDS = ("local", "cloud", "auto")


# ─── Active-backend state ───────────────────────────────────────────────────
# Three layers in priority order: per-request ContextVar → process state →
# env default.

_request_backend: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "llm_request_backend", default=None,
)

_process_backend: Optional[str] = None  # mutated by ``set_active_backend``


def _resolve_auto() -> str:
    """``auto`` → ``cloud`` when OPENAI_API_KEY looks valid, else ``local``.

    We deliberately don't probe Ollama liveness here — that's
    ``llm_available()``'s job and it's relatively expensive. ``auto``
    only chooses based on what's *configured*, not what's reachable.
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key and key.startswith(("sk-", "sk_")) and len(key) > 16:
        return "cloud"
    return "local"


def _raw_backend() -> str:
    """Internal — returns the unresolved backend (may be ``auto``)."""
    ctx = _request_backend.get()
    if ctx:
        return ctx
    if _process_backend:
        return _process_backend
    return DEFAULT_BACKEND


def get_active_backend() -> str:
    """Return the *resolved* active backend — always ``local`` or ``cloud``."""
    b = _raw_backend()
    if b == "auto":
        return _resolve_auto()
    return b


def set_active_backend(name: str) -> str:
    """Flip the process-wide backend. Validates the name; returns the value set.

    Setting ``auto`` is allowed and just delegates back to auto-resolution
    on every subsequent call. To clear the process override entirely so
    ``DEFAULT_BACKEND`` wins again, pass ``""``.
    """
    global _process_backend
    if name == "":
        _process_backend = None
        logger.info("llm_router: cleared process backend override")
        return DEFAULT_BACKEND
    if name not in BACKENDS:
        raise ValueError(f"unknown LLM backend {name!r}; expected one of {BACKENDS}")
    _process_backend = name
    logger.info("llm_router: process backend set to %s", name)
    return name


class with_backend:
    """Context manager / decorator that overrides the backend for one block.

    Useful in API request handlers: a request can carry
    ``?llm_backend=local`` and the handler does

        with with_backend("local"):
            agent.handle(...)

    Backed by ContextVar so async-safe.
    """

    def __init__(self, name: str):
        if name not in BACKENDS:
            raise ValueError(f"unknown LLM backend {name!r}; expected one of {BACKENDS}")
        self._name = name
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> "with_backend":
        self._token = _request_backend.set(self._name)
        return self

    def __exit__(self, *exc_info) -> None:
        if self._token is not None:
            _request_backend.reset(self._token)


# ─── Task → model resolution ────────────────────────────────────────────────


def task_model(task: str) -> str:
    """Resolve the model string for a task role under the active backend.

    Pure ``PROFILES[backend][task]`` lookup — no env vars, no overrides.
    The only knobs are this file's ``PROFILES`` table and the runtime
    backend switch (UI / ``set_active_backend``).

    Raises ``KeyError`` if ``task`` is not one of the six declared roles.
    """
    if task not in TASKS:
        raise KeyError(f"unknown task role {task!r}; expected one of {TASKS}")
    backend = get_active_backend()
    profile = PROFILES.get(backend) or PROFILES["cloud"]
    return profile[task]


def list_profiles() -> Dict[str, Dict[str, str]]:
    """Snapshot of (backend → task → model) for every backend — drives
    the Streamlit sidebar dropdown + Next.js header pill so the user can
    see exactly which model each task will route to under each backend.
    """
    return {b: dict(PROFILES[b]) for b in ("local", "cloud")}


# ─── LLM call dispatch ──────────────────────────────────────────────────────


def call(
    task: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> str:
    """Convenience: resolve task → model → call llm_client.call_llm.

    Existing call sites that pass an explicit ``model=`` argument continue
    to work via ``core.llm_client.call_llm`` directly. New code should
    prefer this entry point so backend switches retarget automatically.
    """
    # Local import to avoid a startup cycle (llm_client imports config which
    # imports this module via the constants shim).
    from core.llm_client import call_llm

    return call_llm(
        system_prompt,
        user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model=task_model(task),
    )


def call_with_metrics(
    task: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> Dict[str, Any]:
    """Same as ``call`` but returns the metrics dict from ``llm_client``."""
    from core.llm_client import call_llm_with_metrics

    return call_llm_with_metrics(
        system_prompt,
        user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model=task_model(task),
    )


# ─── Diagnostics ────────────────────────────────────────────────────────────


def status() -> Dict[str, Any]:
    """Snapshot for ``/api/llm/backend`` and the Streamlit sidebar.

    Returns:
        {
          "active":             "local" | "cloud",   (resolved)
          "raw":                "local" | "cloud" | "auto",  (unresolved)
          "openai_key_valid":   bool,
          "ollama_reachable":   bool,
          "per_task":           {task: model, …}     (under the active backend)
          "profiles":           {backend: {task: model, …}}
        }
    """
    # Local imports so this module stays importable without pulling config.
    from config import _openai_key_valid, _ollama_reachable
    return {
        "active": get_active_backend(),
        "raw": _raw_backend(),
        "openai_key_valid": _openai_key_valid(),
        "ollama_reachable": _ollama_reachable(),
        "per_task": {t: task_model(t) for t in TASKS},
        "profiles": list_profiles(),
        "tasks": list(TASKS),
        "backends": list(BACKENDS),
    }
