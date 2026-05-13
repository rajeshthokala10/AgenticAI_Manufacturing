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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from api.serializers import serialize_state, serialize_turn  # noqa: E402
from config import (  # noqa: E402
    EMBEDDING_MODEL,
    LLM_MODEL,
    USE_HITL,
    USE_LANGGRAPH,
    llm_available,
)
from core.audit_log import get_default_log  # noqa: E402
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


class _Singleton:
    pipe: ManufacturingPipeline | None = None
    agent: ChatAgent | None = None
    sessions: Dict[str, ChatState] = {}
    session_locks: Dict[str, threading.Lock] = {}
    # Map paused approval thread_id → session_id so the resume endpoint can
    # append the resumed answer to the right ChatState.
    thread_to_session: Dict[str, str] = {}
    ready: bool = False
    error: str | None = None


def _get_session_lock(session_id: str) -> threading.Lock:
    lock = _Singleton.session_locks.get(session_id)
    if lock is None:
        lock = threading.Lock()
        _Singleton.session_locks[session_id] = lock
    return lock


@app.on_event("startup")
def _bootstrap() -> None:
    try:
        logger.info("Building / loading ManufacturingPipeline…")
        pipe = ManufacturingPipeline()
        pipe.build_or_load(enable_llm=llm_available())
        _Singleton.pipe = pipe
        _Singleton.agent = ChatAgent(pipe, max_optional_asks=1)
        _Singleton.ready = True
        logger.info("Pipeline ready: %s", pipe.stats)
    except Exception as exc:  # pragma: no cover — startup failure is fatal but logged
        _Singleton.error = repr(exc)
        logger.exception("Failed to bootstrap pipeline: %s", exc)


# ────────────────────────── request / response ──────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's latest chat message.")
    session_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Stable session identifier (UUID). Reuse to keep history.",
    )


class ResetRequest(BaseModel):
    session_id: str


# ────────────────────────────── routes ──────────────────────────────────────

@app.get("/api/health")
def health() -> Dict:
    return {
        "status": "ok" if _Singleton.ready else "starting",
        "ready": _Singleton.ready,
        "error": _Singleton.error,
        "llm_enabled": (_Singleton.pipe.llm_enabled if _Singleton.pipe else False),
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "version": "1.0.0",
        "use_langgraph": USE_LANGGRAPH,
        "use_hitl": USE_HITL,
    }


@app.get("/api/stats")
def stats() -> Dict:
    if not _Singleton.ready or _Singleton.pipe is None:
        raise HTTPException(503, detail="Pipeline not ready")
    return _Singleton.pipe.stats


@app.post("/api/chat")
def chat(req: ChatRequest) -> Dict:
    if not _Singleton.ready or _Singleton.agent is None:
        raise HTTPException(503, detail="Pipeline still bootstrapping — try again in a moment.")
    if not req.message.strip():
        raise HTTPException(400, detail="Empty message.")

    lock = _get_session_lock(req.session_id)
    with lock:
        state = _Singleton.sessions.setdefault(req.session_id, ChatState())
        try:
            _Singleton.agent.handle(state, req.message)
        except Exception as exc:
            logger.exception("Agent failure on session %s", req.session_id)
            raise HTTPException(500, detail=f"Agent error: {exc!r}") from exc

        # Phase A: track pending approvals so /api/approvals/{id}/resume can
        # find the originating session.
        if state.pending_approval_thread_id:
            _Singleton.thread_to_session[state.pending_approval_thread_id] = req.session_id

        new_turns = state.turns[-6:]
        body = {
            "session_id": req.session_id,
            "new_turns": [serialize_turn(t) for t in new_turns],
            "state": serialize_state(state),
            "awaiting_approval": bool(state.pending_approval_thread_id),
        }
        if state.pending_approval_thread_id:
            body["approval_thread_id"] = state.pending_approval_thread_id
        return body


@app.post("/api/reset")
def reset(req: ResetRequest) -> Dict:
    if req.session_id in _Singleton.sessions:
        # Drop any thread→session mappings owned by this session.
        _Singleton.thread_to_session = {
            tid: sid for tid, sid in _Singleton.thread_to_session.items()
            if sid != req.session_id
        }
        _Singleton.sessions[req.session_id].reset()
    return {"ok": True, "session_id": req.session_id}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> Dict:
    state = _Singleton.sessions.get(session_id)
    if state is None:
        return {"session_id": session_id, "state": {"turns": [], "awaiting_slot": None, "awaiting_prompt": None}}
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
    if not _Singleton.ready or _Singleton.pipe is None:
        raise HTTPException(503, detail="Pipeline not ready")


@app.get("/api/approvals/pending")
def list_pending_approvals() -> Dict[str, Any]:
    _require_hitl()
    pending = _Singleton.pipe.pending_approvals()
    enriched: List[Dict[str, Any]] = []
    for entry in pending:
        thread_id = entry.get("thread_id")
        enriched.append({
            **entry,
            "session_id": _Singleton.thread_to_session.get(thread_id),
        })
    return {"pending": enriched, "count": len(enriched)}


@app.get("/api/approvals/{thread_id}")
def get_approval(thread_id: str) -> Dict[str, Any]:
    _require_hitl()
    entry = _Singleton.pipe.get_pending_approval(thread_id)
    if entry is None:
        raise HTTPException(404, detail=f"No pending approval for thread {thread_id}")
    return {
        **entry,
        "session_id": _Singleton.thread_to_session.get(thread_id),
    }


@app.post("/api/approvals/{thread_id}/resume")
def resume_approval(thread_id: str, decision: ApprovalDecision) -> Dict[str, Any]:
    _require_hitl()
    pending = _Singleton.pipe.get_pending_approval(thread_id)
    if pending is None:
        raise HTTPException(404, detail=f"No pending approval for thread {thread_id}")

    session_id = _Singleton.thread_to_session.get(thread_id)
    if session_id is None or session_id not in _Singleton.sessions:
        # Resume detached from any session — still works, just no chat-thread update.
        try:
            result = _Singleton.pipe.resume_diagnostic(thread_id, decision.dict())
        except Exception as exc:
            logger.exception("Detached resume failure on thread %s", thread_id)
            raise HTTPException(500, detail=f"Resume error: {exc!r}") from exc
        _record_audit(thread_id, decision, pending)
        return {
            "thread_id": thread_id,
            "answer": result.answer,
            "rejected": result.rejected,
            "pipeline_status": result.pipeline_status,
            "session_id": None,
        }

    lock = _get_session_lock(session_id)
    with lock:
        state = _Singleton.sessions[session_id]
        try:
            _Singleton.agent.apply_resolution(state, thread_id, decision.dict())
        except Exception as exc:
            logger.exception("Resume failure on thread %s", thread_id)
            raise HTTPException(500, detail=f"Resume error: {exc!r}") from exc
        _record_audit(thread_id, decision, pending)
        # Clear the pending mapping if the agent finished resolving.
        if not state.pending_approval_thread_id:
            _Singleton.thread_to_session.pop(thread_id, None)
        new_turns = state.turns[-6:]
        return {
            "session_id": session_id,
            "thread_id": thread_id,
            "new_turns": [serialize_turn(t) for t in new_turns],
            "state": serialize_state(state),
        }


def _record_audit(thread_id: str, decision: ApprovalDecision, pending: Dict[str, Any]) -> None:
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
    }
