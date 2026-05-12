"""FastAPI backend for the Manufacturing Hybrid GraphRAG conversational pipeline.

Endpoints
---------
GET  /api/health       — liveness + LLM availability + pipeline-ready flag.
GET  /api/stats        — document / vector / graph counts.
POST /api/chat         — submit a user turn, get the updated transcript back.
POST /api/reset        — clear a session's conversation.
GET  /api/sessions/{id} — fetch the current transcript for a session.

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import sys
import threading
import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from api.serializers import serialize_state, serialize_turn  # noqa: E402
from config import LLM_MODEL, EMBEDDING_MODEL, llm_available  # noqa: E402
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

        new_turns = state.turns[-6:]
        return {
            "session_id": req.session_id,
            "new_turns": [serialize_turn(t) for t in new_turns],
            "state": serialize_state(state),
        }


@app.post("/api/reset")
def reset(req: ResetRequest) -> Dict:
    if req.session_id in _Singleton.sessions:
        _Singleton.sessions[req.session_id].reset()
    return {"ok": True, "session_id": req.session_id}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> Dict:
    state = _Singleton.sessions.get(session_id)
    if state is None:
        return {"session_id": session_id, "state": {"turns": [], "awaiting_slot": None, "awaiting_prompt": None}}
    return {"session_id": session_id, "state": serialize_state(state)}


@app.get("/")
def root() -> Dict:
    return {
        "service": "Manufacturing Hybrid GraphRAG API",
        "docs": "/docs",
        "health": "/api/health",
    }
