"""FastAPI backend for the Manufacturing Hybrid GraphRAG conversational pipeline.

Endpoints
---------
GET  /api/health                         — liveness + LLM availability + flags.
GET  /api/stats                          — document / vector / graph counts.
POST /api/chat                           — submit a user turn, get the updated transcript.
POST /api/reset                          — clear a session's conversation.
GET  /api/sessions/{id}                  — fetch the current transcript for a session.

GET  /api/approvals/pending              — list paused HITL workflows (Phase A).
GET  /api/approvals/{thread_id}          — full snapshot of one pending approval.
POST /api/approvals/{thread_id}/resume   — approve or reject a paused workflow.
GET  /api/audit                          — recent approval decisions (Phase B).

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from api.auth import (  # noqa: E402
    get_optional_user,
    require_user,
    router as auth_router,
)
from api.serializers import serialize_state, serialize_turn  # noqa: E402
from config import (  # noqa: E402
    DEFAULT_DOMAIN,
    DOMAIN_DISPLAY,
    DOMAIN_EMPTY_STATE,
    DOMAIN_EXAMPLES,
    DOMAIN_PLACEHOLDER,
    DOMAINS,
    EMBEDDING_MODEL,
    LLM_MODEL,
    USE_HITL,
    USE_LANGGRAPH,
    llm_available,
    normalize_domain,
)
from core.audit_log import get_default_log  # noqa: E402
from core.auth_store import UserRecord  # noqa: E402
from core.domain_prompts import all_schema_statuses, reload_schemas, schema_status  # noqa: E402
from core.document_acl import (  # noqa: E402
    policy_snapshot,
    with_user_classifications,
)
from core.rbac import can_approve, is_maker_locked, required_roles_for  # noqa: E402
from pipeline import ChatAgent, ChatState, ManufacturingPipeline  # noqa: E402

logger = logging.getLogger("api.server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────────── app + state ────────────────────────────────────

app = FastAPI(
    title="Manufacturing Hybrid GraphRAG API",
    description=(
        "Chat API for the unified Hybrid GraphRAG pipeline — "
        "auto-corrects domain jargon, asks clarifying questions, "
        "and returns grounded answers with evidence + KG context."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)
app.include_router(auth_router)


class _Singleton:
    # Per-domain registries. ``pipe`` / ``agent`` (the legacy attributes)
    # always point at the DEFAULT_DOMAIN entry for back-compat with code
    # paths that still access them directly.
    pipes: Dict[str, ManufacturingPipeline] = {}
    agents: Dict[str, ChatAgent] = {}
    # Sessions are keyed by ``(domain, session_id)`` so the same client
    # token can hold two independent conversations.
    sessions: Dict[tuple, ChatState] = {}
    session_locks: Dict[tuple, threading.Lock] = {}
    # Map paused approval thread_id → (domain, session_id) so the resume
    # endpoint can append the resumed answer to the right ChatState.
    thread_to_session: Dict[str, tuple] = {}
    ready: bool = False
    error: str | None = None


def _default_pipe() -> ManufacturingPipeline | None:
    """Back-compat helper for endpoints that still operate on a single
    domain — primarily the HITL approval flow, which is cross-domain."""
    return _Singleton.pipes.get(DEFAULT_DOMAIN)


def _pipe_for(domain: str | None) -> ManufacturingPipeline:
    d = normalize_domain(domain)
    pipe = _Singleton.pipes.get(d)
    if pipe is None:
        raise HTTPException(503, detail=f"Pipeline for domain {d!r} not ready")
    return pipe


def _agent_for(domain: str | None) -> ChatAgent:
    d = normalize_domain(domain)
    agent = _Singleton.agents.get(d)
    if agent is None:
        raise HTTPException(503, detail=f"Agent for domain {d!r} not ready")
    return agent


def _get_session_lock(domain: str, session_id: str) -> threading.Lock:
    key = (domain, session_id)
    lock = _Singleton.session_locks.get(key)
    if lock is None:
        lock = threading.Lock()
        _Singleton.session_locks[key] = lock
    return lock


@app.on_event("startup")
def _bootstrap() -> None:
    # Validate every domain's schema first — surfaces typos and bad
    # YAML loud and early so they don't appear later as silent wrong
    # behaviour. Validation failure does NOT block startup; the affected
    # domain still loads with whatever fallback the loader resolves to,
    # and the error is exposed via /api/domains so the UI can flag it.
    statuses = all_schema_statuses(tuple(DOMAINS))
    for d, st in statuses.items():
        if st.errors:
            logger.error(
                "schema validation FAILED for domain=%r: %d error(s); %d warning(s). "
                "Errors: %s",
                d, len(st.errors), len(st.warnings), "; ".join(st.errors),
            )
        elif st.warnings:
            logger.warning(
                "schema validation OK for domain=%r with %d warning(s): %s",
                d, len(st.warnings), "; ".join(st.warnings),
            )
        else:
            logger.info("schema validation OK for domain=%r", d)

    try:
        for domain in DOMAINS:
            logger.info("Building / loading ManufacturingPipeline [%s]…", domain)
            pipe = ManufacturingPipeline(domain=domain)
            pipe.build_or_load(enable_llm=llm_available())
            _Singleton.pipes[domain] = pipe
            _Singleton.agents[domain] = ChatAgent(pipe, max_optional_asks=1)
            logger.info("Pipeline ready [%s]: %s", domain, pipe.stats)
        _Singleton.ready = True
    except Exception as exc:  # pragma: no cover — startup failure is fatal but logged
        _Singleton.error = repr(exc)
        logger.exception("Failed to bootstrap pipelines: %s", exc)


# ────────────────────────── request / response ──────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's latest chat message.")
    session_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Stable session identifier (UUID). Reuse to keep history.",
    )
    domain: str = Field(
        default=DEFAULT_DOMAIN,
        description=f"Which domain to query: one of {list(DOMAINS)}.",
    )


class ResetRequest(BaseModel):
    session_id: str
    domain: str = Field(default=DEFAULT_DOMAIN)


# ────────────────────────────── routes ──────────────────────────────────────

@app.get("/api/health")
def health() -> Dict:
    return {
        "status": "ok" if _Singleton.ready else "starting",
        "ready": _Singleton.ready,
        "error": _Singleton.error,
        "domains": list(DOMAINS),
        "default_domain": DEFAULT_DOMAIN,
        "domain_status": {
            d: {
                "loaded": d in _Singleton.pipes,
                "llm_enabled": (
                    _Singleton.pipes[d].llm_enabled if d in _Singleton.pipes else False
                ),
            }
            for d in DOMAINS
        },
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "version": "1.0.0",
        "use_langgraph": USE_LANGGRAPH,
        "use_hitl": USE_HITL,
    }


class LlmBackendRequest(BaseModel):
    backend: str = Field(..., description="One of 'local' | 'cloud' | 'auto' (or '' to clear).")


@app.get("/api/llm/backend")
def llm_backend_status() -> Dict:
    """Snapshot of the active LLM backend + per-task model resolution.

    Drives the Streamlit sidebar dropdown and the Next.js header pill.
    """
    from core.llm_router import status
    return status()


@app.post("/api/llm/backend")
def llm_backend_set(req: LlmBackendRequest) -> Dict:
    """Flip the process-wide LLM backend at runtime. Returns the new status."""
    from core.llm_router import set_active_backend, status
    try:
        set_active_backend(req.backend)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    return status()


@app.get("/api/domains")
def domains() -> Dict:
    """Auto-discovered domain catalog.

    Returns the same registry the Streamlit sidebar selector and the
    Next.js header switcher use. Source of truth: ``config.DOMAIN_DISPLAY``,
    which itself is derived from ``schemas/*.yaml`` at startup.
    """
    def _entry(d: str) -> Dict[str, Any]:
        st = schema_status(d)
        return {
            "id": d,
            "label": DOMAIN_DISPLAY[d]["label"],
            "emoji": DOMAIN_DISPLAY[d]["emoji"],
            "color": DOMAIN_DISPLAY[d]["color"],
            "placeholder": DOMAIN_PLACEHOLDER.get(d, ""),
            "empty_state": DOMAIN_EMPTY_STATE.get(d, {}),
            "examples": DOMAIN_EXAMPLES.get(d, []),
            "loaded": d in _Singleton.pipes,
            "schema_status": {
                "ok": st.ok,
                "errors": list(st.errors),
                "warnings": list(st.warnings),
            },
        }

    return {
        "default": DEFAULT_DOMAIN,
        "domains": [_entry(d) for d in DOMAINS],
    }


@app.post("/api/domains/reload")
def domains_reload() -> Dict:
    """Drop the schema cache and re-validate every domain on disk.

    Use after editing a ``schemas/*.yaml`` so prompts / safety keywords /
    clarifier overrides / procedure-gate config take effect without a
    full API restart. The KG snapshots and vector stores are NOT
    rebuilt — for that, restart the API.
    """
    statuses = reload_schemas(tuple(DOMAINS))
    return {
        "reloaded": list(DOMAINS),
        "statuses": {d: st.to_dict() for d, st in statuses.items()},
    }


@app.get("/api/stats")
def stats(domain: str = DEFAULT_DOMAIN) -> Dict:
    if not _Singleton.ready:
        raise HTTPException(503, detail="Pipelines not ready")
    pipe = _pipe_for(domain)
    out = dict(pipe.stats)
    out["domain"] = normalize_domain(domain)
    return out


@app.post("/api/chat")
def chat(
    req: ChatRequest,
    user: Optional[UserRecord] = Depends(get_optional_user),
) -> Dict:
    if not _Singleton.ready:
        raise HTTPException(503, detail="Pipelines still bootstrapping — try again in a moment.")
    if not req.message.strip():
        raise HTTPException(400, detail="Empty message.")

    domain = normalize_domain(req.domain)
    agent = _agent_for(domain)
    pipe = _pipe_for(domain)

    lock = _get_session_lock(domain, req.session_id)
    with lock:
        state = _Singleton.sessions.setdefault((domain, req.session_id), ChatState())
        # Stamp the request's document-ACL view so every retriever called
        # under ``agent.handle`` filters chunks against the signed-in user's
        # role. Anonymous chats fall through to ``public``-only via the
        # default ContextVar value.
        user_role = user.role if user is not None else None
        try:
            with with_user_classifications(user_role):
                agent.handle(state, req.message)
        except Exception as exc:
            logger.exception("Agent failure on session %s/%s", domain, req.session_id)
            raise HTTPException(500, detail=f"Agent error: {exc!r}") from exc

        # Phase A: track pending approvals so /api/approvals/{id}/resume can
        # find the originating (domain, session).
        if state.pending_approval_thread_id:
            _Singleton.thread_to_session[state.pending_approval_thread_id] = (
                domain, req.session_id,
            )
            _annotate_new_pending(state.pending_approval_thread_id, user, pipe=pipe)

        new_turns = state.turns[-6:]
        body = {
            "session_id": req.session_id,
            "domain": domain,
            "new_turns": [serialize_turn(t) for t in new_turns],
            "state": serialize_state(state),
            "awaiting_approval": bool(state.pending_approval_thread_id),
        }
        if state.pending_approval_thread_id:
            body["approval_thread_id"] = state.pending_approval_thread_id
        return body


def _annotate_new_pending(
    thread_id: str,
    user: Optional[UserRecord],
    pipe: ManufacturingPipeline | None = None,
) -> None:
    """Attach maker_user_id + computed required_roles to a newly paused thread."""
    if pipe is None:
        pipe = _Singleton.pipes.get(DEFAULT_DOMAIN)
    if pipe is None or not hasattr(pipe, "annotate_pending"):
        return
    pending = pipe.get_pending_approval(thread_id) or {}
    drivers = (pending.get("risk") or {}).get("drivers", []) or []
    required = required_roles_for(drivers, pending.get("purchase_request"))
    pipe.annotate_pending(
        thread_id,
        maker_user_id=(user.user_id if user is not None else None),
        required_roles=required,
    )


class OnboardAnalyzeRequest(BaseModel):
    domain_id: str = Field(..., description="Lowercase a-z/0-9/_ identifier for the new domain.")
    docs: List[str] = Field(..., description="Plaintext sample documents.")
    domain_hint: str = Field(default="", description="Optional human label.")
    user_prefs: Dict[str, Any] = Field(default_factory=dict)
    prior_qa: List[Dict[str, str]] = Field(default_factory=list)
    force_generate: bool = Field(
        default=False,
        description="When true, instruct the agent to produce a YAML even when ambiguity remains.",
    )


class OnboardSaveRequest(BaseModel):
    domain_id: str
    yaml: str


@app.post("/api/onboard/analyze")
def onboard_analyze(req: OnboardAnalyzeRequest) -> Dict:
    """Drive one round of the schema-authoring agent. Returns either a
    list of follow-up questions or a validated YAML blob.
    """
    from core.onboarding_agent import analyze
    try:
        response = analyze(
            req.domain_id,
            req.docs,
            domain_hint=req.domain_hint,
            user_prefs=req.user_prefs,
            prior_qa=req.prior_qa,
            force_generate=req.force_generate,
        )
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        logger.exception("onboarding analyze failed for %s", req.domain_id)
        raise HTTPException(500, detail=f"Onboarding error: {e!r}") from e
    return response.to_dict()


@app.post("/api/onboard/save")
def onboard_save(req: OnboardSaveRequest) -> Dict:
    """Persist the agent-authored YAML to ``schemas/<domain>.yaml``."""
    from core.onboarding_agent import save_schema
    try:
        dest = save_schema(req.domain_id, req.yaml)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        logger.exception("save_schema failed for %s", req.domain_id)
        raise HTTPException(500, detail=f"Save error: {e!r}") from e
    return {"saved_to": str(dest), "domain_id": req.domain_id}


class DiagnosticRequest(BaseModel):
    message: str = Field(..., description="The diagnostic query.")
    domain: str = Field(
        default=DEFAULT_DOMAIN,
        description=f"Which domain to run against: one of {list(DOMAINS)}.",
    )


@app.post("/api/diagnostic")
def diagnostic(
    req: DiagnosticRequest,
    user: Optional[UserRecord] = Depends(get_optional_user),
) -> Dict:
    """Single-shot blocking diagnostic run — mirrors Streamlit's Diagnostic
    tab. Calls ``pipe.diagnostic(query)`` and returns the full
    ``PipelineResult`` dict (answer + evidence + graph_context + metrics).
    """
    if not _Singleton.ready:
        raise HTTPException(503, detail="Pipelines still bootstrapping — try again in a moment.")
    if not req.message.strip():
        raise HTTPException(400, detail="Empty message.")

    pipe = _pipe_for(req.domain)
    user_role = user.role if user is not None else None
    try:
        with with_user_classifications(user_role):
            result = pipe.diagnostic(req.message)
    except Exception as exc:
        logger.exception("Diagnostic failure on domain %s", req.domain)
        raise HTTPException(500, detail=f"Diagnostic error: {exc!r}") from exc
    return result.to_dict()


@app.post("/api/chat/stream")
def chat_stream(
    req: ChatRequest,
    user: Optional[UserRecord] = Depends(get_optional_user),
) -> StreamingResponse:
    """Stream the diagnostic pipeline as Server-Sent Events.

    Each line emitted on the wire is a JSON-encoded event:

    * ``{"event": "node_update", "node": "...", "update": {...}}``
    * ``{"event": "complete", "response": {...}}`` (terminal, success)
    * ``{"event": "interrupted", "response": {...}}`` (HITL pause)
    * ``{"event": "error", "message": "..."}``

    Requires ``USE_LANGGRAPH=true``. Falls back to a single ``complete``
    event for the procedural orchestrator (which has no intermediate state).
    """
    import json as _json

    if not _Singleton.ready:
        raise HTTPException(503, detail="Pipelines still bootstrapping — try again in a moment.")
    if not req.message.strip():
        raise HTTPException(400, detail="Empty message.")

    pipe = _pipe_for(req.domain)
    user_role = user.role if user is not None else None

    def _event_stream():
        try:
            with with_user_classifications(user_role):
                if hasattr(pipe, "diagnostic_stream") and pipe._orchestrator_engine == "langgraph":
                    for event in pipe.diagnostic_stream(req.message):
                        yield f"data: {_json.dumps(event, default=str)}\n\n"
                else:
                    result = pipe.diagnostic(req.message)
                    yield f"data: {_json.dumps({'event': 'complete', 'response': result.to_dict()}, default=str)}\n\n"
        except Exception as exc:  # pragma: no cover - error surface
            logger.exception("chat_stream failure")
            err = {"event": "error", "message": f"{exc!r}"}
            yield f"data: {_json.dumps(err)}\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@app.post("/api/reset")
def reset(req: ResetRequest) -> Dict:
    domain = normalize_domain(req.domain)
    key = (domain, req.session_id)
    if key in _Singleton.sessions:
        _Singleton.thread_to_session = {
            tid: ds for tid, ds in _Singleton.thread_to_session.items()
            if ds != key
        }
        _Singleton.sessions[key].reset()
    return {"ok": True, "session_id": req.session_id, "domain": domain}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, domain: str = DEFAULT_DOMAIN) -> Dict:
    d = normalize_domain(domain)
    state = _Singleton.sessions.get((d, session_id))
    if state is None:
        return {"session_id": session_id, "domain": d,
                "state": {"turns": [], "awaiting_slot": None, "awaiting_prompt": None}}
    return {"session_id": session_id, "state": serialize_state(state)}


# ─────────────────── HITL: approvals + audit log ────────────────────────────

class ApprovalDecision(BaseModel):
    approved: bool
    approver: str = Field(default="unknown", description="Username / email of the approver.")
    comments: Optional[str] = Field(default=None)
    edited_answer: Optional[str] = Field(
        default=None,
        description="Optional rewrite — replaces the proposed answer if provided.",
    )


def _require_hitl() -> None:
    if not USE_HITL:
        raise HTTPException(404, detail="HITL is disabled. Set USE_HITL=true and restart.")
    if not _Singleton.ready or _default_pipe() is None:
        raise HTTPException(503, detail="Pipeline not ready")


def _enrich_pending(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Decorate a raw pending entry with session id + RBAC fields the UI needs."""
    thread_id = entry.get("thread_id")
    drivers = (entry.get("risk") or {}).get("drivers", []) or []
    purchase = entry.get("purchase_request")
    # The annotate_pending() hook usually fills required_roles already, but
    # recompute as a defence-in-depth fallback (e.g. pre-Phase-D paused
    # threads recovered from SQLite checkpointer).
    required = entry.get("required_roles") or required_roles_for(drivers, purchase)
    session_key = _Singleton.thread_to_session.get(thread_id)
    session_id = session_key[1] if session_key else None
    domain = session_key[0] if session_key else None
    return {
        **entry,
        "session_id": session_id,
        "domain": domain,
        "required_roles": list(required),
        "maker_user_id": entry.get("maker_user_id"),
    }


@app.get("/api/approvals/my")
def my_approvals(
    user: UserRecord = Depends(require_user),
    limit: int = 50,
) -> Dict[str, Any]:
    """Maker + checker dashboard for the signed-in user.

    Returns:
      * ``stats``      — total / pending / approved / rejected for *requests
                          this user has submitted* (rolled up from the audit
                          log + the live pending queue).
      * ``pending``    — requests this user submitted that are still waiting
                          for approval, including who is allowed to approve.
      * ``decisions``  — last N resolved requests (approved or rejected) the
                          user submitted, including approver identity + role.
      * ``actioned``   — last N decisions the user *took* as an approver
                          (only populated for checker roles).
    """
    _require_hitl()
    log = get_default_log()

    # Walk the live queue once and bucket each entry into two views:
    #   * my_pending      — items the user *submitted* (maker side)
    #   * pending_for_me  — items the user is *authorised to action* (checker
    #                       side: role ∈ required_roles AND maker-lock clear)
    all_pending = _default_pipe().pending_approvals()
    my_pending: List[Dict[str, Any]] = []
    pending_for_me: List[Dict[str, Any]] = []
    for entry in all_pending:
        enriched = _enrich_pending(entry)
        # Trim the heavy fields the dashboard doesn't need.
        enriched.pop("evidence", None)
        enriched.pop("interrupt_payload", None)

        maker = (enriched.get("maker_user_id") or "").lower()
        if maker == user.user_id.lower():
            my_pending.append(enriched)
            continue

        if can_approve(user.role, enriched.get("required_roles", [])) and not is_maker_locked(
            user.user_id, enriched.get("maker_user_id")
        ):
            enriched_for_me = {**enriched, "can_current_user_approve": True}
            pending_for_me.append(enriched_for_me)

    # Resolved decisions on my submissions.
    decisions = [e.to_dict() for e in log.for_maker(user.user_id, limit=limit)]

    # Decisions I took (only meaningful for checkers).
    actioned: List[Dict[str, Any]] = []
    if user.role != "operator":
        actioned = [e.to_dict() for e in log.for_approver(user.user_id, limit=limit)]

    audit_stats = log.stats_for_maker(user.user_id)
    stats = {
        "total": audit_stats["total"] + len(my_pending),
        "pending": len(my_pending),
        "approved": audit_stats["approved"],
        "rejected": audit_stats["rejected"],
        "approval_rate": audit_stats["approval_rate"],
        "pending_for_me": len(pending_for_me),
    }

    return {
        "user": {
            "user_id": user.user_id,
            "role": user.role,
            "display_name": user.display_name,
        },
        "stats": stats,
        "pending": my_pending,
        "pending_for_me": pending_for_me,
        "decisions": decisions,
        "actioned": actioned,
    }


@app.get("/api/access/policy")
def get_access_policy(
    user: Optional[UserRecord] = Depends(get_optional_user),
) -> Dict[str, Any]:
    """Document-ACL view for the signed-in user.

    The Next.js UI calls this on login to render the *access-tier badge*
    next to the chat composer. Anonymous callers get the safe default
    (``public`` only) so the UI can still display the badge for guests.
    """
    role = user.role if user is not None else None
    snap = policy_snapshot(role)
    if user is not None:
        snap["user_id"] = user.user_id
        snap["display_name"] = user.display_name
    return snap


@app.get("/api/approvals/pending")
def list_pending_approvals(
    user: Optional[UserRecord] = Depends(get_optional_user),
) -> Dict[str, Any]:
    _require_hitl()
    pending = _default_pipe().pending_approvals()
    enriched = [_enrich_pending(entry) for entry in pending]
    # If the caller is authenticated, surface a per-item "can_i_approve" flag
    # so the UI can grey out items they can't action.
    if user is not None:
        for item in enriched:
            item["can_current_user_approve"] = (
                can_approve(user.role, item["required_roles"])
                and not is_maker_locked(user.user_id, item.get("maker_user_id"))
            )
    return {"pending": enriched, "count": len(enriched)}


@app.get("/api/approvals/{thread_id}")
def get_approval(
    thread_id: str,
    user: Optional[UserRecord] = Depends(get_optional_user),
) -> Dict[str, Any]:
    _require_hitl()
    entry = _default_pipe().get_pending_approval(thread_id)
    if entry is None:
        raise HTTPException(404, detail=f"No pending approval for thread {thread_id}")
    enriched = _enrich_pending(entry)
    if user is not None:
        enriched["can_current_user_approve"] = (
            can_approve(user.role, enriched["required_roles"])
            and not is_maker_locked(user.user_id, enriched.get("maker_user_id"))
        )
    return enriched


@app.post("/api/approvals/{thread_id}/resume")
def resume_approval(
    thread_id: str,
    decision: ApprovalDecision,
    user: UserRecord = Depends(require_user),
) -> Dict[str, Any]:
    """Approve or reject a paused workflow.

    Authentication is **required**. The caller's role must intersect the
    pending item's ``required_roles``, and the caller's user-id must NOT
    match the maker's (segregation of duties).
    """
    _require_hitl()
    pending = _default_pipe().get_pending_approval(thread_id)
    if pending is None:
        raise HTTPException(404, detail=f"No pending approval for thread {thread_id}")

    # ── RBAC: only role-holders listed in required_roles may resolve ─────
    drivers = (pending.get("risk") or {}).get("drivers", []) or []
    required = pending.get("required_roles") or required_roles_for(
        drivers, pending.get("purchase_request")
    )
    if not can_approve(user.role, required):
        raise HTTPException(
            403,
            detail=(
                f"Your role '{user.role}' is not authorised for this approval. "
                f"Required role(s): {', '.join(required)}."
            ),
        )

    # ── Maker-lock: the request submitter cannot self-approve ────────────
    maker_user_id = pending.get("maker_user_id")
    if is_maker_locked(user.user_id, maker_user_id):
        raise HTTPException(
            409,
            detail=(
                "Segregation of duties: you are the request submitter "
                f"({maker_user_id!r}) and cannot approve your own escalation. "
                "Have a different role-holder review it."
            ),
        )

    # The audit log records the *real* approver, ignoring whatever the
    # client sent in the (legacy) ``approver`` field — the token wins.
    effective_decision = decision.copy(update={
        "approver": user.display_name or user.user_id,
    })

    session_key = _Singleton.thread_to_session.get(thread_id)
    if session_key is None or session_key not in _Singleton.sessions:
        # Resume detached from any session — still works, just no chat-thread update.
        try:
            result = _default_pipe().resume_diagnostic(thread_id, effective_decision.dict())
        except Exception as exc:
            logger.exception("Detached resume failure on thread %s", thread_id)
            raise HTTPException(500, detail=f"Resume error: {exc!r}") from exc
        _record_audit(thread_id, effective_decision, pending, user, required)
        return {
            "thread_id": thread_id,
            "answer": result.answer,
            "rejected": result.rejected,
            "pipeline_status": result.pipeline_status,
            "session_id": None,
            "approver_user_id": user.user_id,
            "approver_role": user.role,
        }

    domain, session_id = session_key
    lock = _get_session_lock(domain, session_id)
    with lock:
        state = _Singleton.sessions[session_key]
        try:
            _Singleton.agents[domain].apply_resolution(state, thread_id, effective_decision.dict())
        except Exception as exc:
            logger.exception("Resume failure on thread %s", thread_id)
            raise HTTPException(500, detail=f"Resume error: {exc!r}") from exc
        _record_audit(thread_id, effective_decision, pending, user, required)
        # Clear the pending mapping if the agent finished resolving.
        if not state.pending_approval_thread_id:
            _Singleton.thread_to_session.pop(thread_id, None)
        new_turns = state.turns[-6:]
        return {
            "session_id": session_id,
            "domain": domain,
            "thread_id": thread_id,
            "new_turns": [serialize_turn(t) for t in new_turns],
            "state": serialize_state(state),
            "approver_user_id": user.user_id,
            "approver_role": user.role,
        }


def _record_audit(
    thread_id: str,
    decision: ApprovalDecision,
    pending: Dict[str, Any],
    user: Optional[UserRecord] = None,
    required_roles: Optional[List[str]] = None,
) -> None:
    try:
        risk = pending.get("risk", {}) or {}
        purchase = pending.get("purchase_request")
        get_default_log().record(
            thread_id=thread_id,
            decision="approved" if decision.approved else "rejected",
            approver=decision.approver,
            risk_score=float(risk.get("score", 0.0)),
            drivers=risk.get("drivers", []),
            domain="purchase_request" if purchase else "diagnostic",
            query=pending.get("raw_query", ""),
            proposed_answer=pending.get("answer", ""),
            edited_answer=decision.edited_answer,
            comments=decision.comments,
            maker_user_id=pending.get("maker_user_id"),
            approver_user_id=(user.user_id if user is not None else None),
            approver_role=(user.role if user is not None else None),
            required_roles=required_roles or pending.get("required_roles") or [],
        )
    except Exception:  # pragma: no cover — audit must never break the request
        logger.exception("Failed to write audit log entry for thread %s", thread_id)


@app.get("/api/audit")
def get_audit(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    _require_hitl()
    log = get_default_log()
    rows = [e.to_dict() for e in log.recent(limit=limit, offset=offset)]
    return {"entries": rows, "stats": log.stats(), "limit": limit, "offset": offset}


@app.get("/")
def root() -> Dict:
    return {
        "service": "Manufacturing Hybrid GraphRAG API",
        "docs": "/docs",
        "health": "/api/health",
        "approvals": "/api/approvals/pending",
        "auth": "/api/auth/login",
        "roles": "/api/auth/roles",
    }
