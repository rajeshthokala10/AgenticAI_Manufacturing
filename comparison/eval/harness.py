"""Offline evaluation harness — runs all 3 pipelines on the golden set.

Usage::

    from comparison.eval import EvalHarness
    from pipeline.unified_pipeline import ManufacturingPipeline

    pipe = ManufacturingPipeline()
    pipe.build_or_load()

    harness = EvalHarness(pipe)
    report = harness.run()
    harness.write_markdown_report(report, "comparison/eval/report.md")

CLI::

    python -m comparison.eval.run --output comparison/eval/report.md

The harness is designed for *fast iteration*: it caches results on disk
per ``(pipeline, golden_id)`` so a partial run can be resumed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from comparison.eval.golden import GoldenItem, load_golden_set
from comparison.eval.metrics import aggregate, score_record

logger = logging.getLogger("comparison.eval.harness")


PIPELINE_KINDS = ("direct_llm", "classical_rag", "hybrid_graphrag")


@dataclass
class EvalResult:
    item_id: str
    pipeline: str
    question: str
    answer: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tokens: int = 0
    citations: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "pipeline": self.pipeline,
            "question": self.question,
            "answer": self.answer,
            "metrics": self.metrics,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "tokens": self.tokens,
            "citations": self.citations,
            "error": self.error,
        }


class EvalHarness:
    """Runs the offline eval. Supports caching + selective pipelines."""

    def __init__(
        self,
        pipeline: Any,
        *,
        cache_dir: Optional[Path] = None,
        pipelines: List[str] = list(PIPELINE_KINDS),
    ):
        self.pipeline = pipeline
        self.pipelines = list(pipelines)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────

    def run(
        self,
        golden: Optional[List[GoldenItem]] = None,
        *,
        progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        items = golden or load_golden_set()
        records: List[EvalResult] = []

        for item in items:
            for kind in self.pipelines:
                if progress:
                    progress(f"{kind} :: {item.id}")
                rec = self._maybe_cached(item, kind)
                if rec is None:
                    rec = self._run_one(item, kind)
                    self._save_cache(item, kind, rec)
                records.append(rec)

        return self._build_report(items, records)

    # ── Per-record execution ──────────────────────────────────────────

    def _run_one(self, item: GoldenItem, kind: str) -> EvalResult:
        question = item.question
        rec = EvalResult(item_id=item.id, pipeline=kind, question=question, answer="")

        try:
            t0 = time.time()
            if kind == "direct_llm":
                result = self.pipeline.direct(question)
            elif kind == "classical_rag":
                result = self.pipeline.classical(question)
            elif kind == "hybrid_graphrag":
                result = self.pipeline.diagnostic(question)
            else:
                raise ValueError(f"unknown pipeline kind: {kind}")
            elapsed = (time.time() - t0) * 1000

            result_dict = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            answer = str(result_dict.get("answer") or "")
            evidence = list(result_dict.get("evidence") or [])
            metrics = dict(result_dict.get("metrics") or {})

            rec.answer = answer
            rec.latency_ms = float(metrics.get("total_latency_ms", elapsed))
            rec.cost_usd = float(metrics.get("cost_estimate_usd", 0.0) or 0.0)
            rec.tokens = int(metrics.get("total_tokens", 0) or 0)
            scored = score_record(answer, evidence, item)
            rec.metrics = scored
            rec.citations = scored.get("citation_accuracy", 0.0) and 1 or 0
        except Exception as exc:  # pragma: no cover - eval should not crash
            logger.exception("eval failed for %s/%s", kind, item.id)
            rec.error = str(exc)
        return rec

    # ── Cache helpers ────────────────────────────────────────────────

    def _cache_path(self, item: GoldenItem, kind: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{kind}__{item.id}.json"

    def _maybe_cached(self, item: GoldenItem, kind: str) -> Optional[EvalResult]:
        path = self._cache_path(item, kind)
        if not path or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return EvalResult(
            item_id=data.get("item_id", item.id),
            pipeline=data.get("pipeline", kind),
            question=data.get("question", item.question),
            answer=data.get("answer", ""),
            metrics=data.get("metrics", {}),
            latency_ms=float(data.get("latency_ms", 0.0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
            tokens=int(data.get("tokens", 0)),
            citations=int(data.get("citations", 0)),
            error=data.get("error"),
        )

    def _save_cache(self, item: GoldenItem, kind: str, rec: EvalResult) -> None:
        path = self._cache_path(item, kind)
        if not path:
            return
        path.write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")

    # ── Report ───────────────────────────────────────────────────────

    @staticmethod
    def _build_report(
        items: List[GoldenItem],
        records: List[EvalResult],
    ) -> Dict[str, Any]:
        per_pipeline_records: Dict[str, List[Dict[str, Any]]] = {p: [] for p in PIPELINE_KINDS}
        for rec in records:
            per_pipeline_records.setdefault(rec.pipeline, []).append(rec.metrics)

        summary = {p: aggregate(records_list) for p, records_list in per_pipeline_records.items()}

        per_pipeline_latency = {p: [] for p in PIPELINE_KINDS}
        per_pipeline_cost = {p: [] for p in PIPELINE_KINDS}
        per_pipeline_tokens = {p: [] for p in PIPELINE_KINDS}
        for rec in records:
            per_pipeline_latency.setdefault(rec.pipeline, []).append(rec.latency_ms)
            per_pipeline_cost.setdefault(rec.pipeline, []).append(rec.cost_usd)
            per_pipeline_tokens.setdefault(rec.pipeline, []).append(rec.tokens)

        for p in summary:
            lats = per_pipeline_latency.get(p, []) or [0.0]
            costs = per_pipeline_cost.get(p, []) or [0.0]
            toks = per_pipeline_tokens.get(p, []) or [0]
            summary[p]["avg_latency_ms"] = round(sum(lats) / len(lats), 2)
            summary[p]["total_cost_usd"] = round(sum(costs), 6)
            summary[p]["avg_tokens"] = round(sum(toks) / len(toks), 1)

        return {
            "n_items": len(items),
            "n_records": len(records),
            "items": [
                {"id": it.id, "category": it.category, "difficulty": it.difficulty}
                for it in items
            ],
            "summary": summary,
            "records": [r.to_dict() for r in records],
        }

    # ── Reporting helpers ────────────────────────────────────────────

    @staticmethod
    def write_markdown_report(report: Dict[str, Any], path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        lines.append("# Hybrid GraphRAG — Offline Eval Report")
        lines.append("")
        lines.append(f"- Golden items: **{report['n_items']}**")
        lines.append(f"- Total records: **{report['n_records']}**")
        lines.append("")
        lines.append("## Aggregate metrics by pipeline")
        lines.append("")
        cols = [
            "pipeline", "n", "faithfulness", "answer_relevancy",
            "context_precision", "citation_accuracy",
            "must_mention_coverage", "guardrail_pass_rate",
            "forbidden_violation_rate", "avg_latency_ms",
            "total_cost_usd", "avg_tokens",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for p, m in report["summary"].items():
            if not m:
                continue
            row = [p] + [str(m.get(c, "")) for c in cols[1:]]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("## Per-record breakdown")
        lines.append("")
        lines.append("| item | pipeline | faithfulness | relevancy | citation_acc | guardrail | latency_ms |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in report["records"]:
            m = r.get("metrics", {})
            lines.append(
                "| "
                + " | ".join([
                    r["item_id"], r["pipeline"],
                    str(m.get("faithfulness", "")),
                    str(m.get("answer_relevancy", "")),
                    str(m.get("citation_accuracy", "")),
                    "PASS" if m.get("guardrail_pass") else "FAIL",
                    f"{r.get('latency_ms', 0):.0f}",
                ])
                + " |"
            )
        out.write_text("\n".join(lines), encoding="utf-8")
        return out
