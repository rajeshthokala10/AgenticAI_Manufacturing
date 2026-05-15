"""RAGAS-style metrics for offline evaluation.

We implement *cheap, deterministic* versions of the classic RAGAS metrics
so the harness can run without paying for an external LLM judge on every
PR. An optional LLM judge can be plugged in later (see ``llm_judge`` hook
in :mod:`harness`).

Metric definitions:

* ``faithfulness``
    Bag-of-words overlap of answer tokens with the union of evidence text.
    Bounded in [0, 1]. Penalises claims that cite no supporting evidence.

* ``answer_relevancy``
    Token-set similarity between the answer and the (question + ground truth).
    Captures "did the answer actually address what was asked?".

* ``context_precision``
    Fraction of retrieved evidence chunks whose source name or text
    matches one of the ``expected_sources`` substrings.

* ``citation_accuracy``
    Fraction of citations in the answer that resolve to a real chunk in
    the evidence pack. Re-uses the guardrails citation parser so this
    metric stays consistent with the runtime guardrail.

* ``must_mention_coverage`` / ``forbidden_violations``
    Hard checks; we compute them per-item so the report can surface them.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

from comparison.eval.golden import GoldenItem
from core.guardrails import evaluate as guardrail_evaluate


_TOKEN_RE = re.compile(r"\b[\w\-]+\b")


def _tokens(text: str) -> Set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def faithfulness(answer: str, evidence_chunks: Iterable[Dict[str, Any]]) -> float:
    """Token-level grounding score in [0,1]."""
    ans_tokens = _tokens(answer)
    if not ans_tokens:
        return 0.0
    evidence_tokens: Set[str] = set()
    for chunk in evidence_chunks or []:
        evidence_tokens |= _tokens(str(chunk.get("text") or ""))
    if not evidence_tokens:
        return 0.0
    grounded = len(ans_tokens & evidence_tokens)
    return round(grounded / len(ans_tokens), 4)


def answer_relevancy(answer: str, item: GoldenItem) -> float:
    """How well the answer addresses the question + ground truth."""
    target = f"{item.question} {item.ground_truth}"
    return round(_jaccard(_tokens(answer), _tokens(target)), 4)


def context_precision(item: GoldenItem, evidence_chunks: Iterable[Dict[str, Any]]) -> float:
    """Fraction of retrieved chunks that look on-topic per ``expected_sources``."""
    expected = [s.lower() for s in item.expected_sources if s.strip()]
    chunks = list(evidence_chunks or [])
    if not expected or not chunks:
        return 0.0
    hits = 0
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        haystack = " ".join([
            str(meta.get("source") or ""),
            str(meta.get("doc_type") or ""),
            str(chunk.get("text") or "")[:500],
        ]).lower()
        if any(token in haystack for token in expected):
            hits += 1
    return round(hits / len(chunks), 4)


def citation_accuracy(answer: str, evidence_chunks: Iterable[Dict[str, Any]]) -> float:
    """Fraction of citations that resolve to a real chunk id in the evidence."""
    report = guardrail_evaluate(answer, list(evidence_chunks or []), require_citations=False, block_on_unsafe=False)
    if report.citation_count == 0:
        return 0.0
    valid_ids = {str(c.get("chunk_id") or "").lower() for c in evidence_chunks or []}
    valid_ids.discard("")
    matched = sum(1 for cid in report.cited_chunk_ids if cid in valid_ids)
    if not report.cited_chunk_ids:
        # we still have source-name citations
        return 1.0 if report.referenced_sources else 0.0
    return round(matched / len(report.cited_chunk_ids), 4)


def must_mention_coverage(answer: str, item: GoldenItem) -> float:
    if not item.must_mention:
        return 1.0
    lower = (answer or "").lower()
    hits = sum(1 for term in item.must_mention if term.lower() in lower)
    return round(hits / len(item.must_mention), 4)


def forbidden_violations(answer: str, item: GoldenItem) -> List[str]:
    if not item.forbidden:
        return []
    lower = (answer or "").lower()
    return [term for term in item.forbidden if term.lower() in lower]


def guardrail_pass(answer: str, evidence_chunks: Iterable[Dict[str, Any]]) -> bool:
    report = guardrail_evaluate(answer, list(evidence_chunks or []))
    return report.verdict == "PASS"


# ─── Hard targets (piston-style) ────────────────────────────────────────────


def top_cause_match(
    item: GoldenItem,
    cause_ranking: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Boolean check that the top cause from the cause-ranker equals
    ``item.expected_top_cause``. Returns a dict that includes the predicted
    value so the report can surface false positives.
    """
    expected = (item.expected_top_cause or "").strip()
    if not expected:
        return {"applicable": False}
    candidates = (cause_ranking or {}).get("candidates") or []
    predicted = ""
    if candidates:
        predicted = str(candidates[0].get("cause", "")).strip()
    return {
        "applicable": True,
        "expected": expected,
        "predicted": predicted,
        "match": bool(predicted) and predicted.lower() == expected.lower(),
    }


def subsystem_match(
    item: GoldenItem,
    clarification: Dict[str, Any] | None,
    evidence_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Boolean check that the inferred subsystem matches the gold value.

    The subsystem can come from two surfaces: the clarifier's structured
    output (``clarification['entities']``) or — as a fallback — the
    top-ranked evidence chunk's ``metadata.doc_type``. Either match counts.
    """
    expected = (item.expected_subsystem or "").strip().lower()
    if not expected:
        return {"applicable": False}

    candidates: List[str] = []
    for ent in (clarification or {}).get("entities") or []:
        # entities may arrive as tuples (kind, value) or dicts.
        if isinstance(ent, (list, tuple)) and len(ent) >= 2:
            candidates.append(str(ent[1]))
        elif isinstance(ent, dict):
            candidates.extend(str(v) for v in ent.values())
    for chunk in evidence_chunks[:3]:
        meta = chunk.get("metadata") or {}
        candidates.append(str(meta.get("doc_type") or ""))
        candidates.append(str(meta.get("source") or ""))

    haystack = " ".join(candidates).lower()
    return {
        "applicable": True,
        "expected": expected,
        "match": expected in haystack,
    }


def score_record(
    answer: str,
    evidence_chunks: List[Dict[str, Any]],
    item: GoldenItem,
    *,
    cause_ranking: Dict[str, Any] | None = None,
    clarification: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Aggregate every metric for a single pipeline × item result.

    Soft metrics always run. Hard targets (``top_cause_match`` /
    ``subsystem_match``) only emit numbers when the golden item declared
    an expectation.
    """
    out: Dict[str, Any] = {
        "faithfulness": faithfulness(answer, evidence_chunks),
        "answer_relevancy": answer_relevancy(answer, item),
        "context_precision": context_precision(item, evidence_chunks),
        "citation_accuracy": citation_accuracy(answer, evidence_chunks),
        "must_mention_coverage": must_mention_coverage(answer, item),
        "forbidden_violations": forbidden_violations(answer, item),
        "guardrail_pass": guardrail_pass(answer, evidence_chunks),
    }
    cause_check = top_cause_match(item, cause_ranking)
    if cause_check.get("applicable"):
        out["top_cause_match"] = bool(cause_check["match"])
        out["top_cause_predicted"] = cause_check.get("predicted")
        out["top_cause_expected"] = cause_check.get("expected")
    sub_check = subsystem_match(item, clarification, evidence_chunks)
    if sub_check.get("applicable"):
        out["subsystem_match"] = bool(sub_check["match"])
        out["subsystem_expected"] = sub_check.get("expected")
    return out


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average the numeric metrics + compute pass-rates.

    Hard-target metrics (``top_cause_match`` / ``subsystem_match``) are
    aggregated only over records where the corresponding key is present
    (i.e. the golden item declared an expectation); the denominator reflects
    just those records so the rate is meaningful.
    """
    if not records:
        return {}
    keys_numeric = ["faithfulness", "answer_relevancy", "context_precision",
                    "citation_accuracy", "must_mention_coverage"]
    out: Dict[str, Any] = {}
    for k in keys_numeric:
        vals = [float(r.get(k, 0.0) or 0.0) for r in records]
        out[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
    out["guardrail_pass_rate"] = round(
        sum(1 for r in records if r.get("guardrail_pass")) / len(records), 4
    )
    out["forbidden_violation_rate"] = round(
        sum(1 for r in records if r.get("forbidden_violations")) / len(records), 4
    )

    cause_recs = [r for r in records if "top_cause_match" in r]
    if cause_recs:
        out["top_cause_match_rate"] = round(
            sum(1 for r in cause_recs if r["top_cause_match"]) / len(cause_recs), 4
        )
        out["top_cause_match_n"] = len(cause_recs)
    sub_recs = [r for r in records if "subsystem_match" in r]
    if sub_recs:
        out["subsystem_match_rate"] = round(
            sum(1 for r in sub_recs if r["subsystem_match"]) / len(sub_recs), 4
        )
        out["subsystem_match_n"] = len(sub_recs)

    out["n"] = len(records)
    return out
