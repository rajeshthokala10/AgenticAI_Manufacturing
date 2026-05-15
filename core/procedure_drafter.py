"""Two-stage generation — structured procedure drafting.

Ported from ``piston-engine-copilot/src/generation/procedure_drafting.py``
and adapted for the manufacturing domain.

After cause-ranking has produced a list of probable root causes, this stage
asks the LLM to emit a STRUCTURED diagnostic procedure as JSON:

    {
      "steps": [
        {"step": 1, "action": "...", "citations": ["chunk_id_a", "chunk_id_b"]},
        ...
      ]
    }

Each step's ``citations`` are validated against the retrieved chunk ids —
any chunk_id the LLM invents is silently dropped. The dataclass output can
be rendered as Markdown for legacy free-form answer surfaces (Streamlit,
Next.js) or consumed directly by downstream UIs that want the structure.

Why this matters vs. the single-blob answer the legacy orchestrator
produces: the procedure layout makes citations per-step, simplifies the
critic check (every step must cite at least one valid chunk), and gives
the Streamlit / Next.js front-ends a stable structure to render
incrementally during streaming.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import PROCEDURE_MODEL
from core.llm_client import call_llm_with_metrics

logger = logging.getLogger("core.procedure_drafter")


PROCEDURE_SYSTEM_PROMPT = """You are a manufacturing diagnostic copilot drafting a step-by-step
procedure for the user's troubleshooting query.

RULES:
1. The output MUST be STRICT JSON ONLY — a single object with key "steps".
2. Each step is an object: {"step": <int>, "action": "<imperative sentence>",
   "citations": ["chunk_id_a", ...]}.
3. Every ``citations`` entry MUST be copied verbatim from the retrieved
   evidence headers. Do NOT invent chunk ids.
4. Sequence the steps so safety preconditions (LOTO, de-energise, vent
   pressure) come BEFORE any inspection or component handling.
5. Address the highest-ranked causes first. Reference equipment IDs and
   alarm codes by name.
6. 5-10 steps total. Keep each action short (one sentence, imperative).
7. No prose before or after the JSON. No markdown fences."""


@dataclass
class ProcedureStep:
    step: int
    action: str
    citations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "action": self.action, "citations": list(self.citations)}


@dataclass
class Procedure:
    steps: List[ProcedureStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps]}

    def __bool__(self) -> bool:
        return bool(self.steps)


# ─── Formatting helpers ─────────────────────────────────────────────────────


def _format_causes(causes: List[Dict[str, Any]]) -> str:
    if not causes:
        return "(no ranked causes)"
    lines: List[str] = []
    for i, c in enumerate(causes, 1):
        score = float(c.get("score", 0.0) or 0.0)
        rationale = (c.get("rationale") or "").strip()
        line = f"{i}. {c.get('cause', '')} (score {score:.2f})"
        if rationale:
            line += f" — {rationale}"
        lines.append(line)
    return "\n".join(lines)


def _format_evidence(chunks: List[Dict[str, Any]], doc_types: Optional[set] = None) -> str:
    rows: List[str] = []
    for c in chunks:
        meta = c.get("metadata") or {}
        doc_type = str(meta.get("doc_type") or "").lower()
        if doc_types and doc_type not in doc_types:
            continue
        cid = c.get("chunk_id", "?")
        text = (c.get("text") or "").strip().replace("\n", " ")
        if len(text) > 600:
            text = text[:600] + "…"
        source = meta.get("source", "unknown")
        rows.append(f"[chunk_id={cid} | {source}] {text}")
    return "\n".join(rows) if rows else "(none)"


def _build_user_prompt(
    query: str,
    cause_candidates: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    feedback: Optional[str],
) -> str:
    manual_block = _format_evidence(
        evidence, doc_types={"manual", "pdf", "sop", "procedure", "specification"}
    )
    other_block = _format_evidence(evidence, doc_types=None)
    feedback_block = (
        f"\nCritic feedback to address in this revision:\n{feedback}\n"
        if feedback else ""
    )
    return (
        f"QUERY: {query}\n\n"
        f"RANKED PROBABLE CAUSES:\n{_format_causes(cause_candidates)}\n\n"
        f"RETRIEVED EVIDENCE (cite chunk_ids verbatim):\n{other_block}\n"
        f"{feedback_block}\n"
        "Return JSON only with the shape "
        "{\"steps\":[{\"step\":1,\"action\":\"…\",\"citations\":[\"…\"]}]}."
    )


def _parse_response(text: str, retrieved_ids: set) -> List[ProcedureStep]:
    """Robustly parse the LLM's JSON output even if it's wrapped in fences."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Find the outer JSON object — be permissive about leading/trailing prose.
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return []

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    raw_steps = data.get("steps") or data.get("procedure") or []
    if not isinstance(raw_steps, list):
        return []

    steps: List[ProcedureStep] = []
    for i, item in enumerate(raw_steps, 1):
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        if not action:
            continue
        cites = item.get("citations") or item.get("evidence_chunk_ids") or []
        if not isinstance(cites, list):
            cites = [str(cites)]
        # Drop citations that don't resolve to a real chunk (anti-hallucination).
        # If ``retrieved_ids`` is empty (eg. unit test) we accept everything.
        if retrieved_ids:
            cites = [str(c).strip() for c in cites if str(c).strip() in retrieved_ids]
        else:
            cites = [str(c).strip() for c in cites if str(c).strip()]
        try:
            step_num = int(item.get("step") or i)
        except (TypeError, ValueError):
            step_num = i
        steps.append(ProcedureStep(step=step_num, action=action, citations=cites))

    # Renumber so the output is always contiguous 1..N regardless of LLM quirks.
    for i, s in enumerate(steps, 1):
        s.step = i
    return steps


# ─── Public entry point ────────────────────────────────────────────────────


def draft_procedure(
    query: str,
    cause_candidates: List[Dict[str, Any]],
    evidence_chunks: List[Dict[str, Any]],
    *,
    feedback: Optional[str] = None,
    model: str = PROCEDURE_MODEL,
) -> Dict[str, Any]:
    """Generate a structured procedure for the user's troubleshooting query.

    Returns a dict with the parsed ``Procedure`` plus LLM usage metrics so
    the caller can fold token counts and cost into the overall response.
    """
    base_metrics: Dict[str, Any] = {
        "procedure": Procedure().to_dict(),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_estimate": 0.0,
        "model": model,
    }

    if not evidence_chunks:
        return {**base_metrics, "skipped": "no_evidence"}

    retrieved_ids = {str(c.get("chunk_id", "")).strip() for c in evidence_chunks}
    retrieved_ids.discard("")

    user_prompt = _build_user_prompt(query, cause_candidates, evidence_chunks, feedback)

    try:
        llm_result = call_llm_with_metrics(
            system_prompt=PROCEDURE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=1200,
            model=model,
        )
    except Exception as exc:  # pragma: no cover - network / API failure
        logger.warning("procedure drafting LLM call failed: %s", exc)
        return {**base_metrics, "error": str(exc)}

    raw = llm_result.get("response", "") or ""
    steps = _parse_response(raw, retrieved_ids)
    procedure = Procedure(steps=steps)

    return {
        "procedure": procedure.to_dict(),
        "raw_response": raw,
        "prompt_tokens": llm_result.get("prompt_tokens", 0),
        "completion_tokens": llm_result.get("completion_tokens", 0),
        "total_tokens": llm_result.get("total_tokens", 0),
        "cost_estimate": llm_result.get("cost_estimate", 0.0),
        "model": llm_result.get("model", model),
    }


def render_as_markdown(procedure_dict: Dict[str, Any]) -> str:
    """Render a procedure dict (from :func:`draft_procedure`) as Markdown.

    The output is suitable for surfacing through the legacy ``answer`` field
    when callers want the structured procedure shown to humans (Streamlit,
    Next.js chat, FastAPI clients that haven't been updated to consume the
    structured payload).
    """
    steps = (procedure_dict or {}).get("steps") or []
    if not steps:
        return ""
    lines: List[str] = ["## Diagnostic Procedure"]
    for s in steps:
        action = (s.get("action") or "").strip()
        if not action:
            continue
        cites = s.get("citations") or []
        cite_str = " " + " ".join(f"`[{c}]`" for c in cites) if cites else ""
        lines.append(f"{s.get('step', '?')}. {action}{cite_str}")
    return "\n".join(lines)
