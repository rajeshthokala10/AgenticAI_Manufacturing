"""CLI entry point for the offline eval harness.

::

    python -m comparison.eval.run --output comparison/eval/report.md
    python -m comparison.eval.run --pipelines hybrid_graphrag --cache-dir .eval_cache

Exit code is non-zero when any pipeline's ``faithfulness`` drops below
the floor specified by ``--min-faithfulness`` (defaults to 0.0). This
makes the harness CI-friendly without forcing strict numbers on day one —
tighten the floor as the eval set matures.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Hybrid GraphRAG offline eval.")
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=["direct_llm", "classical_rag", "hybrid_graphrag"],
        choices=["direct_llm", "classical_rag", "hybrid_graphrag"],
        help="Which pipelines to grade (default: all three).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("comparison/eval/report.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to also write the raw JSON report.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache per-record results here to support resumes.",
    )
    parser.add_argument(
        "--min-faithfulness",
        type=float,
        default=0.0,
        help="Fail the run if hybrid_graphrag.faithfulness < this floor.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from pipeline.unified_pipeline import ManufacturingPipeline
    from comparison.eval import EvalHarness

    pipe = ManufacturingPipeline()
    pipe.build_or_load()

    if not pipe.llm_enabled and ("classical_rag" in args.pipelines or "direct_llm" in args.pipelines or "hybrid_graphrag" in args.pipelines):
        print(
            "[eval] OPENAI_API_KEY not set — only quick-search-like checks will run."
            " Configure .env to grade direct/classical/hybrid pipelines.",
            file=sys.stderr,
        )

    harness = EvalHarness(pipe, cache_dir=args.cache_dir, pipelines=list(args.pipelines))
    report = harness.run(progress=lambda label: print(f"[eval] {label}", file=sys.stderr))

    out_md = harness.write_markdown_report(report, args.output)
    print(f"[eval] Wrote markdown report: {out_md}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[eval] Wrote JSON report: {args.json_output}")

    hybrid_summary = report["summary"].get("hybrid_graphrag", {})
    faithfulness = float(hybrid_summary.get("faithfulness", 0.0) or 0.0)
    if faithfulness < args.min_faithfulness:
        print(
            f"[eval] FAIL — hybrid_graphrag.faithfulness={faithfulness:.4f} "
            f"< floor={args.min_faithfulness:.4f}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
