"""Convert piston-engine-copilot work-order JSONL into ingestable .txt files.

The sibling piston repo at
``/home/anil-y/app_ideas/manufacture/piston-engine-copilot/`` ships 500
ASRS-derived work orders as a single JSONL. The manufacturing
``doc_pipeline`` only reads ``.pdf / .txt / .xlsx`` from ``input_docs/``,
so we materialise each record as its own ``.txt`` under
``doc_pipeline/input_docs/aviation/work_orders/``.

The body deliberately repeats the canonical ``wo:asrs:NNN`` id in the
text — ``pipeline.adapter`` regex-extracts equipment ids from chunk text
(not from frontmatter), so writing the id into the body is what causes
KeywordExtractor edges to fire.

Usage::

    python scripts/ingest_piston_work_orders.py            # all 500 rows
    python scripts/ingest_piston_work_orders.py --limit 50 # first 50 rows
    python scripts/ingest_piston_work_orders.py --clean    # remove existing .txt first

The script is idempotent: rerunning with the same JSONL overwrites the
.txt files in place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PISTON_JSONL = Path(
    "/home/anil-y/app_ideas/manufacture/piston-engine-copilot/data/processed/work_orders.jsonl"
)
DEST_DIR = ROOT / "doc_pipeline" / "input_docs" / "aviation" / "work_orders"


def render(record: dict) -> str:
    wo_id = record["wo_id"]
    subsystem = record.get("subsystem", "unknown")
    complaint = (record.get("complaint_text_normalized") or record.get("complaint_text") or "").strip()
    resolution = (record.get("resolution_text") or "").strip()

    lines = [
        f"Work Order: {wo_id}",
        f"Subsystem: {subsystem}",
        f"Source: ASRS (Aviation Safety Reporting System)",
        "",
        "## Complaint",
        complaint or "(no complaint text)",
        "",
        "## Resolution",
        resolution or "(no resolution text)",
    ]
    return "\n".join(lines) + "\n"


def slug(wo_id: str) -> str:
    return wo_id.replace(":", "_") + ".txt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Convert only the first N rows (default: all).")
    parser.add_argument("--clean", action="store_true",
                        help="Delete existing .txt files in DEST_DIR before writing.")
    parser.add_argument("--source", type=Path, default=PISTON_JSONL,
                        help="Path to work_orders.jsonl (default: sibling piston repo).")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"ERROR: source JSONL not found: {args.source}", file=sys.stderr)
        return 1

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for p in DEST_DIR.glob("*.txt"):
            p.unlink()

    written = 0
    with args.source.open() as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            (DEST_DIR / slug(record["wo_id"])).write_text(render(record))
            written += 1

    print(f"Wrote {written} work-order .txt files → {DEST_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
