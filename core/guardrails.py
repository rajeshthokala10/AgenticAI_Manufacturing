"""Guardrails post-processor — deterministic safety + citation checks.

Sits between the answer LLM and the LLM critic. The critic is *judgmental*
(it scores grounding via another LLM), the guardrails are *deterministic*
(regex + structured rules). Running both in series gives us:

* Defence in depth — a single LLM critic can be fooled by a confident
  hallucination; a regex insisting on at least one ``[source, chunk_id]``
  citation cannot.
* Explainable refusals — every flag has a stable id (e.g.
  ``no_citation``, ``unsafe_loto_bypass``) the UI can render.

Verdicts:

* ``"PASS"``                 — answer is safe + grounded; pipeline proceeds.
* ``"FAIL_REWRITE"``         — recoverable; feed the violations back to the
                               critic/retry loop as additional issues.
* ``"FAIL_BLOCK"``           — never deliver; escalate to HITL or refuse.

The output dict is *additive* — orchestrators merge ``violations`` into the
critic's ``issues`` list so the existing retry loop can fix them on the
next pass without any other plumbing changes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("core.guardrails")


# ─── Citation patterns ──────────────────────────────────────────────────
#
# The answer prompt instructs the LLM to use ``[source_name, chunk_id]``;
# we accept any bracketed reference that contains a chunk-id-looking
# substring (chunk-… / c-… / chk-…) or a path-like source token.
_CITATION_BRACKET = re.compile(r"\[([^\[\]]{3,}?)\]")
_CHUNK_ID_TOKEN = re.compile(r"\b(?:chunk[-_:]\S+|c[-_]\S+|chk[-_]\S+|c\d{2,})\b", re.IGNORECASE)


# ─── Safety patterns ────────────────────────────────────────────────────
#
# Tuples of ``(rule_id, regex, severity, message)``. ``severity`` is
# "block" for hard violations and "rewrite" for soft ones (e.g. missing
# PPE callout) that the retry loop can address.

# Each pattern is intentionally narrow — we want zero false positives on
# legitimate safety advice (e.g. "perform lockout/tagout before…").
_UNSAFE_PATTERNS: List[Tuple[str, re.Pattern, str, str]] = [
    (
        "loto_bypass",
        re.compile(r"\b(?:bypass|skip|ignore|defeat|override)\b[^.]{0,40}\b(?:lockout|tagout|loto)\b", re.IGNORECASE),
        "block",
        "Answer suggests bypassing lockout/tagout — never permitted.",
    ),
    (
        "interlock_bypass",
        re.compile(r"\b(?:bypass|defeat|jumper|disable)\b[^.]{0,40}\b(?:interlock|safety[- ]?switch|e-?stop|emergency[- ]?stop)\b", re.IGNORECASE),
        "block",
        "Answer suggests bypassing a safety interlock or e-stop.",
    ),
    (
        "hot_work_no_permit",
        re.compile(r"\b(?:hot work|welding|cutting|grinding)\b(?![^.]{0,80}\bpermit\b)", re.IGNORECASE),
        "rewrite",
        "Hot work mentioned without referencing a hot-work permit.",
    ),
    (
        "confined_space_no_permit",
        re.compile(r"\bconfined space\b(?![^.]{0,80}\bpermit\b)", re.IGNORECASE),
        "rewrite",
        "Confined-space work mentioned without a permit reference.",
    ),
    (
        "live_electrical",
        re.compile(r"\b(?:work on|service|repair|open)\b[^.]{0,40}\b(?:live|energi[sz]ed|powered)\b[^.]{0,40}\b(?:panel|cabinet|bus|conductor|circuit)\b", re.IGNORECASE),
        "block",
        "Answer recommends working on energised electrical equipment.",
    ),
    (
        "chemical_no_ppe",
        re.compile(r"\b(?:acid|caustic|solvent|chemical|coolant|hydraulic fluid)\b(?![^.]{0,120}\b(?:ppe|gloves|goggles|respirator|face[- ]?shield)\b)", re.IGNORECASE),
        "rewrite",
        "Chemical handling mentioned without PPE / SDS reference.",
    ),
    (
        "no_action_for_fire",
        re.compile(r"\b(?:fire|smoke|smouldering)\b", re.IGNORECASE),
        "rewrite",
        "Fire/smoke mentioned — the answer must reference emergency response and evacuation.",
    ),
]


# ─── Refusal patterns (uncertain LLM output) ────────────────────────────
_REFUSAL_PHRASES = (
    "i cannot",
    "i can't",
    "i'm sorry",
    "i am unable",
    "as an ai",
    "i don't have access",
    "no information",
)


@dataclass
class Violation:
    rule_id: str
    severity: str  # "block" | "rewrite"
    message: str
    snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "snippet": self.snippet,
        }


@dataclass
class GuardrailReport:
    verdict: str = "PASS"  # PASS | FAIL_REWRITE | FAIL_BLOCK
    violations: List[Violation] = field(default_factory=list)
    citation_count: int = 0
    cited_chunk_ids: List[str] = field(default_factory=list)
    referenced_sources: List[str] = field(default_factory=list)
    has_refusal: bool = False
    risk_boost: float = 0.0  # added to the HITL classifier when present

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "violations": [v.to_dict() for v in self.violations],
            "citation_count": self.citation_count,
            "cited_chunk_ids": self.cited_chunk_ids,
            "referenced_sources": self.referenced_sources,
            "has_refusal": self.has_refusal,
            "risk_boost": round(self.risk_boost, 3),
        }

    def as_critic_issues(self) -> List[str]:
        """Format violations so they can be appended to the critic feedback."""
        return [f"[guardrail:{v.rule_id}/{v.severity}] {v.message}" for v in self.violations]


def evaluate(
    answer: str,
    evidence_chunks: Optional[List[Dict[str, Any]]] = None,
    *,
    require_citations: bool = True,
    min_citations: int = 1,
    block_on_unsafe: bool = True,
) -> GuardrailReport:
    """Run the guardrails over a freshly generated answer.

    Parameters
    ----------
    answer
        The LLM-generated answer text.
    evidence_chunks
        The retrieved evidence the answer should reference. Used to verify
        cited chunk ids actually exist (helps catch fabricated citations).
    require_citations
        When True, an answer without any citation fails with ``FAIL_REWRITE``.
    min_citations
        Minimum number of citations required when ``require_citations``.
    block_on_unsafe
        When True, hard-block on unsafe patterns. Set False during eval.
    """
    report = GuardrailReport()
    text = answer or ""

    # ── Citation check ────────────────────────────────────────────────
    valid_chunk_ids: Set[str] = set()
    valid_sources: Set[str] = set()
    if evidence_chunks:
        for chunk in evidence_chunks:
            cid = str(chunk.get("chunk_id") or "")
            if cid:
                valid_chunk_ids.add(cid.lower())
            meta = chunk.get("metadata") or {}
            src = str(meta.get("source") or "")
            if src:
                valid_sources.add(src.lower())

    cited_chunk_ids: List[str] = []
    cited_sources: List[str] = []
    for match in _CITATION_BRACKET.findall(text):
        inner = match.strip()
        # Look for explicit chunk id tokens first…
        chunk_tokens = _CHUNK_ID_TOKEN.findall(inner)
        if chunk_tokens:
            cited_chunk_ids.extend(t.lower() for t in chunk_tokens)
        # …and then for source-name hits.
        for src in valid_sources:
            if src and src in inner.lower():
                cited_sources.append(src)
    report.cited_chunk_ids = sorted(set(cited_chunk_ids))
    report.referenced_sources = sorted(set(cited_sources))
    report.citation_count = len(report.cited_chunk_ids) + len(report.referenced_sources)

    # Fabricated chunk-id detection: any cited chunk-id not in the evidence
    # set is a strong hallucination signal even when count >= min_citations.
    fabricated: List[str] = []
    if valid_chunk_ids:
        for cid in report.cited_chunk_ids:
            if cid not in valid_chunk_ids:
                fabricated.append(cid)
    if fabricated:
        report.violations.append(
            Violation(
                rule_id="fabricated_citation",
                severity="rewrite",
                message=(
                    "Citations reference chunks that are not in the evidence: "
                    + ", ".join(fabricated[:5])
                ),
                snippet=", ".join(fabricated[:5]),
            )
        )

    if require_citations and report.citation_count < max(min_citations, 1):
        report.violations.append(
            Violation(
                rule_id="no_citation",
                severity="rewrite",
                message=(
                    f"Answer must include at least {min_citations} "
                    "[source, chunk_id] citation(s)."
                ),
            )
        )

    # ── Safety patterns ───────────────────────────────────────────────
    for rule_id, pattern, severity, message in _UNSAFE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        snippet = text[max(0, match.start() - 30) : match.end() + 30].strip()
        report.violations.append(
            Violation(
                rule_id=rule_id,
                severity=severity,
                message=message,
                snippet=snippet,
            )
        )

    # ── Refusal detection (low-information answer) ────────────────────
    lower = text.lower()
    if any(phrase in lower for phrase in _REFUSAL_PHRASES) and len(text) < 400:
        report.has_refusal = True
        report.violations.append(
            Violation(
                rule_id="refusal_or_low_information",
                severity="rewrite",
                message="Answer is an apology / refusal with no actionable content.",
            )
        )

    # ── Derive verdict + HITL risk boost ──────────────────────────────
    severities = {v.severity for v in report.violations}
    if "block" in severities and block_on_unsafe:
        report.verdict = "FAIL_BLOCK"
        report.risk_boost = 0.5
    elif severities:
        report.verdict = "FAIL_REWRITE"
        report.risk_boost = 0.2 if "rewrite" in severities else 0.0
    else:
        report.verdict = "PASS"

    return report


def merge_into_critic(
    critic_result: Dict[str, Any],
    report: GuardrailReport,
) -> Dict[str, Any]:
    """Fold guardrail violations into a critic_result so the retry loop sees them.

    Returns a new dict (does not mutate inputs). When the guardrail blocks,
    we force the critic verdict to FAIL regardless of what the LLM critic said,
    because the regex evidence is authoritative for the safety rules.
    """
    merged = dict(critic_result or {})
    merged.setdefault("issues", [])
    merged.setdefault("attempt", critic_result.get("attempt", 1))

    if report.violations:
        merged["issues"] = list(merged.get("issues") or []) + report.as_critic_issues()
        merged["guardrails"] = report.to_dict()
        if report.verdict in ("FAIL_REWRITE", "FAIL_BLOCK"):
            merged["verdict"] = "FAIL"
            merged.setdefault("suggestion", "")
            sugg = "; ".join(v.message for v in report.violations[:5])
            merged["suggestion"] = (
                f"{merged['suggestion']}\nGuardrails require: {sugg}".strip()
            )
            merged["guardrails_blocked"] = report.verdict == "FAIL_BLOCK"
    else:
        merged["guardrails"] = report.to_dict()

    return merged
