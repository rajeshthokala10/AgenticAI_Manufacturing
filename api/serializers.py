"""JSON serializers for ChatAgent state — used by the FastAPI backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pipeline import ChatState, ChatTurn


def _serialize_evidence(evidence: List[Dict] | None, limit: int = 8) -> List[Dict]:
    if not evidence:
        return []
    out: List[Dict] = []
    for ev in evidence[:limit]:
        meta = ev.get("metadata", {}) or {}
        src = meta.get("source", meta.get("source_file", "unknown"))
        out.append({
            "source": Path(str(src)).name,
            "doc_type": str(meta.get("doc_type", "")).upper(),
            "page": meta.get("page"),
            "sheet": meta.get("sheet_name"),
            "section": meta.get("section_title"),
            "score": float(ev.get("vector_score", ev.get("rrf_score", 0.0)) or 0.0),
            "text": (ev.get("text") or "")[:1200],
        })
    return out


def _serialize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Strip dataclasses / enums into plain JSON-safe primitives."""
    out: Dict[str, Any] = {}

    clar = meta.get("clarification")
    if clar is not None:
        out["intent"] = clar.intent.value
        out["intent_confidence"] = float(clar.intent_confidence)
        out["entities"] = [
            {"type": e.entity_type, "value": e.normalized}
            for e in clar.entities
        ]
        out["is_complete"] = bool(clar.is_complete)

    metrics = meta.get("metrics") or {}
    if metrics:
        out["metrics"] = {
            "latency_ms": float(metrics.get("total_latency_ms", 0.0)),
            "tokens": int(metrics.get("total_tokens", 0)),
            "cost_usd": float(metrics.get("cost_estimate_usd", 0.0)),
        }

    critic = (meta.get("critic") or {}).get("final_verdict", {}) or {}
    if critic:
        out["critic"] = {
            "verdict": critic.get("verdict"),
            "confidence": float(critic.get("confidence", 0.0)),
            "rationale": critic.get("rationale", ""),
        }

    if meta.get("evidence"):
        out["evidence"] = _serialize_evidence(meta["evidence"])
        out["evidence_count"] = len(meta["evidence"])

    graph_ctx = meta.get("graph_context")
    if graph_ctx and graph_ctx.get("nodes"):
        out["graph_context"] = {
            "node_count": len(graph_ctx.get("nodes", [])),
            "edge_count": len(graph_ctx.get("edges", [])),
            "nodes": [
                {"id": n.get("id"), "type": n.get("type"), "chunks": n.get("chunks", 0)}
                for n in graph_ctx.get("nodes", [])[:12]
            ],
        }

    if meta.get("mode"):
        out["mode"] = meta["mode"]
    if meta.get("corrected_query"):
        out["corrected_query"] = meta["corrected_query"]
    if meta.get("spelling_fixes"):
        out["spelling_fixes"] = meta["spelling_fixes"]
    if meta.get("acronym_fixes"):
        out["acronym_fixes"] = meta["acronym_fixes"]
    if meta.get("slot"):
        out["slot"] = meta["slot"]
        out["slot_required"] = bool(meta.get("required", False))
        out["slot_prompt"] = meta.get("prompt", "")

    # HITL-related fields
    if meta.get("risk"):
        risk = meta["risk"]
        out["risk"] = {
            "score": float(risk.get("score", 0.0)),
            "needs_human": bool(risk.get("needs_human", False)),
            "drivers": list(risk.get("drivers", []) or [])[:8],
            "summary": risk.get("summary", ""),
        }
    if meta.get("thread_id"):
        out["thread_id"] = meta["thread_id"]
    if meta.get("domain"):
        out["domain"] = meta["domain"]
    if meta.get("purchase_request"):
        out["purchase_request"] = meta["purchase_request"]
    if meta.get("human_decision"):
        out["human_decision"] = meta["human_decision"]
    if meta.get("rejected"):
        out["rejected"] = bool(meta["rejected"])
    return out


def serialize_turn(turn: ChatTurn) -> Dict[str, Any]:
    return {
        "role": turn.role,
        "content": turn.content,
        "kind": turn.kind,
        "meta": _serialize_meta(turn.meta or {}),
    }


def serialize_state(state: ChatState) -> Dict[str, Any]:
    return {
        "turns": [serialize_turn(t) for t in state.turns],
        "awaiting_slot": (state.awaiting_slot.name if state.awaiting_slot else None),
        "awaiting_prompt": (state.awaiting_slot.prompt if state.awaiting_slot else None),
        "pending_approval_thread_id": state.pending_approval_thread_id,
    }
