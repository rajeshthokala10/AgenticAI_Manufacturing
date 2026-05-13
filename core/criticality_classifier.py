"""Criticality classifier — decides per-query whether a human should approve.

The classifier is the gateway that turns the LangGraph orchestrator into a
production-grade workflow engine: routine queries continue uninterrupted,
high-risk ones pause at a `human_approval` interrupt.

Design notes
------------
* **Rules first** (cheap and explainable). A single keyword hit is enough to
  escalate; we don't want a surprising LLM call to *under*-grade an obviously
  dangerous procedure. Drivers are returned verbatim so the approver UI can
  show *why* the approval was triggered.
* **LLM grader only for the inconclusive band** (``0.3 < score < 0.7``), and
  only when ``OPENAI_API_KEY`` (or Ollama) is reachable. The grader uses the
  same free local model as the cause-ranker by default — no additional cloud
  spend.
* **Pure**: no I/O outside the (optional) LLM call. Easy to unit-test.

The output is consumed by ``pipeline/langgraph_orchestrator.py`` to decide
whether to short-circuit straight to the critic or detour through the
``human_approval`` interrupt.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from config import (
    CAUSE_RANK_MODEL,
    HITL_AUTO_APPROVE_BELOW_USD,
    HITL_HIGH_RISK_KEYWORDS,
    HITL_RISK_THRESHOLD,
)

logger = logging.getLogger("core.criticality")


@dataclass
class Risk:
    """Risk assessment for a proposed answer or action."""

    score: float = 0.0  # 0.0 (safe to auto-approve) → 1.0 (definitely escalate)
    needs_human: bool = False
    drivers: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["score"] = round(float(self.score), 4)
        return d


# Patterns that bump the score by a fixed amount.
_KEYWORD_WEIGHT = 0.55
_DOLLAR_WEIGHT = 0.4
_LOW_CONFIDENCE_WEIGHT = 0.3
_INTENT_BUMP = 0.4

# Critical-equipment intents (extracted from the clarifier intent vocabulary).
_HIGH_RISK_INTENTS = {
    "shutdown", "emergency", "permit_to_work", "lockout_tagout",
}


def _hits(text: str, keywords) -> List[str]:
    """Return the list of keywords present in ``text`` (case-insensitive)."""
    if not text:
        return []
    lower = text.lower()
    return [kw for kw in keywords if kw and kw in lower]


_DOLLAR_RE = re.compile(r"\$\s*([0-9][0-9,]*(?:\.\d+)?)")


def _extract_dollar_amount(text: str) -> Optional[float]:
    """Return the largest ``$nnn`` mentioned in ``text``, or None."""
    if not text:
        return None
    amounts: List[float] = []
    for raw in _DOLLAR_RE.findall(text):
        try:
            amounts.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return max(amounts) if amounts else None


def _llm_grade(query: str, proposed_answer: str) -> Optional[float]:
    """Optional tier-2 LLM grader. Returns ``None`` on any failure."""
    try:
        from core.llm_client import call_llm  # local import to keep startup snappy
    except Exception:  # pragma: no cover
        return None

    prompt = (
        "On a scale of 0.0 (clearly safe to auto-approve) to 1.0 (must be "
        "reviewed by a human supervisor), score the operational risk of "
        "executing the following proposed action in a manufacturing plant. "
        "Reply with ONE number only, no prose.\n\n"
        f"USER QUERY:\n{query}\n\nPROPOSED ANSWER:\n{proposed_answer[:1500]}"
    )
    try:
        out = call_llm(
            system_prompt="You are a plant-safety risk grader. Return one float in [0,1].",
            user_prompt=prompt,
            model=CAUSE_RANK_MODEL,  # reuse the cheap free local model
            temperature=0.0,
            max_tokens=8,
        )
    except Exception as exc:  # pragma: no cover - network / API
        logger.warning("LLM grader failed, falling back to rules: %s", exc)
        return None

    match = re.search(r"[01]?\.\d+|\d", str(out))
    if not match:
        return None
    try:
        val = float(match.group(0))
    except ValueError:
        return None
    return max(0.0, min(1.0, val))


def classify(
    query: str,
    intent: Optional[str] = None,
    proposed_answer: str = "",
    critic_confidence: Optional[float] = None,
    purchase_request: Optional[Dict[str, Any]] = None,
    threshold: float = HITL_RISK_THRESHOLD,
    keywords=HITL_HIGH_RISK_KEYWORDS,
    enable_llm_grader: bool = True,
) -> Risk:
    """Score the proposed answer / action and decide whether to escalate.

    Returns a :class:`Risk` instance. The orchestrator route depends only on
    ``Risk.needs_human``.
    """

    score = 0.0
    drivers: List[str] = []
    haystack = f"{query}\n{proposed_answer}".lower()

    # ── Rule 1: high-risk safety / regulatory keywords ────────────────────
    keyword_hits = _hits(haystack, keywords)
    if keyword_hits:
        score = max(score, _KEYWORD_WEIGHT + 0.05 * (len(keyword_hits) - 1))
        drivers.extend(f"safety_keyword:{kw}" for kw in keyword_hits[:5])

    # ── Rule 2: high-risk intents ─────────────────────────────────────────
    if intent and any(t in intent.lower() for t in _HIGH_RISK_INTENTS):
        score = max(score, _INTENT_BUMP + 0.5)
        drivers.append(f"high_risk_intent:{intent}")

    # ── Rule 3: low critic confidence ─────────────────────────────────────
    if critic_confidence is not None and critic_confidence < 0.5:
        score = max(score, _LOW_CONFIDENCE_WEIGHT)
        drivers.append(f"low_critic_confidence:{critic_confidence:.2f}")

    # ── Rule 4: purchase-request dollar threshold (Phase C) ───────────────
    pr = purchase_request or {}
    total_usd = pr.get("total_usd")
    if total_usd is None:
        # Fall back to scraping a $amount out of the query / answer.
        total_usd = _extract_dollar_amount(haystack)

    if total_usd is not None and total_usd >= HITL_AUTO_APPROVE_BELOW_USD:
        score = max(score, _DOLLAR_WEIGHT + 0.3)
        drivers.append(
            f"purchase_value=${total_usd:,.0f}>=${HITL_AUTO_APPROVE_BELOW_USD:,.0f}"
        )

    if pr.get("single_source"):
        score = max(score, 0.65)
        drivers.append("single_source_vendor")

    if pr.get("lead_time_days") is not None and pr["lead_time_days"] > 7:
        score = max(score, 0.55)
        drivers.append(f"long_lead_time:{pr['lead_time_days']}d")

    if (pr.get("equipment_criticality") or "").upper() == "A":
        score = max(score, 0.7)
        drivers.append("class_A_equipment")

    # ── Rule 5: tier-2 LLM grader for the inconclusive band ───────────────
    if enable_llm_grader and 0.3 < score < 0.7:
        graded = _llm_grade(query, proposed_answer)
        if graded is not None:
            score = max(score, graded)
            drivers.append(f"llm_grader:{graded:.2f}")

    score = min(score, 1.0)
    needs_human = score >= threshold

    summary = (
        f"score={score:.2f} (threshold {threshold:.2f}) "
        + ("→ escalate" if needs_human else "→ auto-approve")
    )

    return Risk(
        score=score,
        needs_human=needs_human,
        drivers=drivers,
        summary=summary,
    )
