"""
Unified CLI entry point for the Manufacturing Hybrid GraphRAG pipeline.

Usage:
    python main.py                 # build/load indexes and run a small demo
    python main.py --rebuild       # force a full rebuild of FAISS + KG
    python main.py --query "..."   # run a single query (quick-search mode)
    python main.py --diagnostic "..." # run the full LLM + critic pipeline
    python main.py --compare "..." # run all three pipelines side by side
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import ManufacturingPipeline


def format_evidence(evidence: list, limit: int = 3) -> str:
    if not evidence:
        return "  (no evidence)"
    lines = []
    for i, ev in enumerate(evidence[:limit], 1):
        meta = ev.get("metadata", {})
        source = Path(str(meta.get("source", meta.get("source_file", "?")))).name
        page = meta.get("page", "")
        section = meta.get("section_title", "")
        score = ev.get("vector_score", ev.get("rrf_score", 0.0))
        snippet = (ev.get("text", "")[:200] + "…").replace("\n", " ")
        loc = f", page {page}" if page else (f", section {section}" if section else "")
        lines.append(f"  {i}. [{source}{loc}] (score={score:.3f}) {snippet}")
    return "\n".join(lines)


def run_demo(pipe: ManufacturingPipeline) -> None:
    queries = [
        "What is the OEE target for Q2 2026?",
        "Pump P-203 has high vibration alarm ALM-P001. What's the likely cause?",
        "How do I perform a tool change on the Mori Seiki NHX5000?",
        "maintanance schedul for spindle bearings",
    ]
    for q in queries:
        print("\n" + "=" * 72)
        print(f"QUERY: {q}")
        t = time.time()
        result = pipe.quick_search(q, top_k=3)
        elapsed = (time.time() - t) * 1000
        c = result.clarification
        print(f"  Intent: {c.intent.value} ({c.intent_confidence:.0%})")
        print(f"  Entities: {[(e.entity_type, e.normalized) for e in c.entities]}")
        if result.correction.corrections_applied:
            print(f"  Corrections: {result.correction.corrections_applied}")
        print(f"  Results: {len(result.evidence)} (latency {elapsed:.0f}ms)")
        print(format_evidence(result.evidence, limit=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manufacturing Hybrid GraphRAG CLI")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force a full rebuild of FAISS + KG.")
    parser.add_argument("--no-llm", action="store_true",
                        help="Do not initialise LLM components even if API key is set.")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a quick-search query.")
    parser.add_argument("--diagnostic", type=str, default=None,
                        help="Run the full diagnostic pipeline (LLM + critic).")
    parser.add_argument("--compare", type=str, default=None,
                        help="Run all 3 pipelines side by side.")
    parser.add_argument("--json", action="store_true",
                        help="Print the result as JSON.")
    args = parser.parse_args()

    print("Building / loading unified pipeline...")
    pipe = ManufacturingPipeline()
    stats = pipe.build_or_load(rebuild=args.rebuild, enable_llm=not args.no_llm)

    print("\n=== Pipeline ready ===")
    for k, v in stats.items():
        if isinstance(v, (str, int, float, bool)):
            print(f"  {k}: {v}")

    if args.compare:
        results = pipe.compare(args.compare)
        if args.json:
            print(json.dumps({k: v.to_dict() for k, v in results.items()}, indent=2, default=str))
        else:
            for name, r in results.items():
                print(f"\n--- {name} ---")
                print(f"Answer: {r.answer[:500]}")
                print(f"Metrics: {r.metrics}")
        return

    if args.diagnostic:
        r = pipe.diagnostic(args.diagnostic)
        if args.json:
            print(json.dumps(r.to_dict(), indent=2, default=str))
        else:
            print("\nAnswer:\n" + r.answer)
            print("\nMetrics:", r.metrics)
            verdict = (r.critic or {}).get("final_verdict", {})
            print("Critic:", verdict.get("verdict"), verdict.get("confidence"))
        return

    if args.query:
        r = pipe.quick_search(args.query)
        if args.json:
            print(json.dumps(r.to_dict(), indent=2, default=str))
        else:
            c = r.clarification
            print(f"\nIntent: {c.intent.value} ({c.intent_confidence:.0%})")
            print(f"Entities: {[(e.entity_type, e.normalized) for e in c.entities]}")
            print(f"\nTop {len(r.evidence)} results:")
            print(format_evidence(r.evidence, limit=5))
        return

    run_demo(pipe)


if __name__ == "__main__":
    main()
