"""Cause-ranking LLM stage.

Optional pipeline stage that turns retrieval evidence + KG cause/failure-mode
entities into a *ranked* list of likely root causes for a troubleshooting
query. Enable by setting ``USE_CAUSE_RANKING=true`` in ``.env``.

The output is consumed by the answer LLM (via :func:`format_for_prompt`) so
the final diagnostic answer is anchored on explicitly scored candidates rather
than mining them implicitly from the chunks.

The ranker is *intent-gated*: it short-circuits to an empty result for
non-troubleshooting queries (e.g. plain lookups) so it never adds cost or
hallucination surface to queries that don't need it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import CAUSE_RANK_MODEL, CAUSE_RANK_TOP_K
from core.llm_client import call_llm_with_metrics

logger = logging.getLogger("core.cause_ranker")


CAUSE_RANK_SYSTEM_PROMPT = """You are a manufacturing root-cause analyst. Given a problem query and a set of
evidence chunks, identify and rank the most likely ROOT CAUSES.

RULES:
1. Every cause MUST be grounded in the evidence chunks provided. Do not invent causes.
2. Output STRICT JSON ONLY — a list of objects with keys: "cause", "score", "rationale", "evidence_chunk_ids".
3. "score" is a float in [0.0, 1.0] representing likelihood given the evidence. Use distinct scores.
4. "evidence_chunk_ids" is a list of chunk_id strings copied verbatim from the evidence headers.
5. Rank from MOST LIKELY (top) to LEAST LIKELY (bottom).
6. Return AT MOST {top_k} causes.
7. If the query is not about troubleshooting / failure analysis, return an empty list [].
8. No prose before or after the JSON. No markdown fences."""


# Intent strings (from core/query_formatter or doc_pipeline/clarifier_agent) that
# benefit from the cause-ranking stage.
_TROUBLESHOOTING_TRIGGERS = (
    "troubleshoot", "diagnos", "root_cause", "root-cause", "rootcause",
    "failure", "fault", "repair", "fix", "incident", "broken", "alarm",
)


@dataclass
class CauseCandidate:
    """A single ranked root-cause candidate."""

    cause: str
    score: float
    rationale: str = ""
    evidence_chunk_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cause": self.cause,
            "score": round(float(self.score), 4),
            "rationale": self.rationale,
            "evidence_chunk_ids": self.evidence_chunk_ids,
        }


def _intent_is_troubleshooting(intent: Optional[str]) -> bool:
    if not intent:
        return False
    lower = str(intent).lower()
    return any(t in lower for t in _TROUBLESHOOTING_TRIGGERS)


def _format_evidence_for_ranking(chunks: List[Dict[str, Any]], limit: int = 8) -> str:
    parts: List[str] = []
    for chunk in chunks[:limit]:
        cid = chunk.get("chunk_id", "?")
        text = (chunk.get("text", "") or "").strip()
        if not text:
            continue
        if len(text) > 800:
            text = text[:800] + "..."
        parts.append(f"[chunk_id={cid}]\n{text}")
    return "\n\n".join(parts)


def _format_kg_causes(graph_context: Optional[Dict[str, Any]]) -> str:
    if not graph_context:
        return ""
    nodes = graph_context.get("nodes") or []
    cause_like: List[str] = []
    for n in nodes:
        etype = str(n.get("entity_type") or n.get("type") or "").lower()
        if etype in ("cause", "failuremode", "failure_mode"):
            label = n.get("label") or n.get("name") or n.get("id") or ""
            if label:
                cause_like.append(f"- {label} ({etype})")
    if not cause_like:
        return ""
    return "KG CAUSE / FAILURE-MODE ENTITIES (from the knowledge graph subgraph):\n" + "\n".join(
        cause_like[:15]
    )


def _parse_response(text: str, top_k: int) -> List[CauseCandidate]:
    """Robustly parse the LLM's JSON output even if it's wrapped in fences/prose."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if not match:
        return []

    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    candidates: List[CauseCandidate] = []
    if not isinstance(items, list):
        return []

    for item in items:
        if not isinstance(item, dict):
            continue
        cause = str(item.get("cause", "")).strip()
        if not cause:
            continue
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))

        rationale = str(item.get("rationale", "")).strip()
        ev = item.get("evidence_chunk_ids", []) or []
        if not isinstance(ev, list):
            ev = [str(ev)]
        ev = [str(x) for x in ev if x]

        candidates.append(
            CauseCandidate(
                cause=cause,
                score=score,
                rationale=rationale,
                evidence_chunk_ids=ev,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


def rank_causes(
    query: str,
    intent: Optional[str],
    evidence_chunks: List[Dict[str, Any]],
    graph_context: Optional[Dict[str, Any]] = None,
    top_k: int = CAUSE_RANK_TOP_K,
    model: str = CAUSE_RANK_MODEL,
) -> Dict[str, Any]:
    """Run the cause-ranking LLM stage.

    Returns a dict with the ranked ``candidates`` plus LLM usage metrics, so
    the caller can fold token counts and cost into the overall response.

    The function is *cheap* by default — ``CAUSE_RANK_MODEL`` defaults to
    ``qwen2.5:3b`` (free, local via Ollama). Override with any OpenAI model
    in ``.env`` if you prefer cloud quality.
    """
    base_metrics: Dict[str, Any] = {
        "candidates": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_estimate": 0.0,
        "model": model,
    }

    if not _intent_is_troubleshooting(intent):
        return {**base_metrics, "skipped": "intent_not_troubleshooting"}

    if not evidence_chunks:
        return {**base_metrics, "skipped": "no_evidence"}

    evidence_block = _format_evidence_for_ranking(evidence_chunks)
    kg_block = _format_kg_causes(graph_context)
    user_prompt = (
        f"QUERY: {query}\n\n"
        + (kg_block + "\n\n" if kg_block else "")
        + "EVIDENCE CHUNKS:\n"
        + evidence_block
        + "\n\n"
        + f"Identify and rank up to {top_k} root causes as STRICT JSON."
    )

    try:
        llm_result = call_llm_with_metrics(
            system_prompt=CAUSE_RANK_SYSTEM_PROMPT.format(top_k=top_k),
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=600,
            model=model,
        )
    except Exception as exc:  # pragma: no cover - network / API failures
        logger.warning("Cause ranking LLM call failed: %s", exc)
        return {**base_metrics, "error": str(exc)}

    raw = llm_result.get("response", "") or ""
    candidates = _parse_response(raw, top_k)

    return {
        "candidates": [c.to_dict() for c in candidates],
        "raw_response": raw,
        "prompt_tokens": llm_result.get("prompt_tokens", 0),
        "completion_tokens": llm_result.get("completion_tokens", 0),
        "total_tokens": llm_result.get("total_tokens", 0),
        "cost_estimate": llm_result.get("cost_estimate", 0.0),
        "model": llm_result.get("model", model),
    }


def format_for_prompt(candidates: List[Dict[str, Any]]) -> str:
    """Render ranked candidates as a prompt block for the answer LLM.

    Returns an empty string when there are no candidates, so callers can
    safely concatenate the result into a larger prompt.
    """
    if not candidates:
        return ""
    lines = ["LIKELY ROOT CAUSES (pre-ranked by a dedicated cause-ranking model, highest probability first):"]
    for i, c in enumerate(candidates, 1):
        score = float(c.get("score", 0.0) or 0.0)
        cause = c.get("cause", "")
        ev = c.get("evidence_chunk_ids") or []
        ev_str = f"   (evidence: {', '.join(ev)})" if ev else ""
        lines.append(f"  {i}. [score {score:.2f}] {cause}{ev_str}")
        rationale = c.get("rationale", "")
        if rationale:
            lines.append(f"        rationale: {rationale}")
    return "\n".join(lines)
