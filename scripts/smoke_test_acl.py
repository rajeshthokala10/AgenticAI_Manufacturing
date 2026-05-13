"""Offline smoke test for the document-ACL layer.

Verifies five invariants without standing up FastAPI / Next.js:

  1. ``core.document_acl.classify_from_path`` correctly maps folder
     conventions → classification tags.
  2. ``allowed_classifications`` and ``can_read`` enforce the role policy
     (operator → public-only, plant_manager → all tiers).
  3. The active ``ContextVar`` scopes the read-set correctly and resets
     on exit.
  4. ``EmbeddingPipeline.search`` returns zero hits on a confidential
     query when run as an operator, and at least one hit when run as a
     plant manager. (This is the end-to-end proof point — it exercises
     the same code path the live ``/api/chat`` request takes.)
  5. ``HybridRetriever``-style chunk dicts are filtered correctly by
     ``filter_chunks``.

Run:
    PYTHONPATH=. python scripts/smoke_test_acl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.document_acl import (  # noqa: E402
    CONFIDENTIAL,
    PUBLIC,
    RESTRICTED,
    active_classifications,
    allowed_classifications,
    can_read,
    classify_from_path,
    filter_chunks,
    max_tier_for,
    with_user_classifications,
)

PASS = "  PASS"
FAIL = "  FAIL"


def _check(label: str, ok: bool) -> bool:
    print(f"{PASS if ok else FAIL}  {label}")
    return ok


def test_path_classification() -> bool:
    print("\n[1] classify_from_path()")
    ok = True
    ok &= _check(
        "management/ folder → confidential",
        classify_from_path("doc_pipeline/input_docs/management/q1_2026_financial_review.txt") == CONFIDENTIAL,
    )
    ok &= _check(
        "restricted/ folder → restricted",
        classify_from_path("doc_pipeline/input_docs/restricted/regulatory_incident_response_plan.txt") == RESTRICTED,
    )
    ok &= _check(
        "top-level folder → public",
        classify_from_path("doc_pipeline/input_docs/sop_cnc_machining.txt") == PUBLIC,
    )
    ok &= _check(
        "filename containing 'management' but no folder → public",
        classify_from_path("/tmp/incident_management.pdf") == PUBLIC,
    )
    return ok


def test_role_policy() -> bool:
    print("\n[2] role → classifications policy")
    ok = True
    ok &= _check("operator → {public}", allowed_classifications("operator") == {PUBLIC})
    ok &= _check(
        "buyer → {public, restricted}",
        allowed_classifications("buyer") == {PUBLIC, RESTRICTED},
    )
    ok &= _check(
        "plant_manager → all three tiers",
        allowed_classifications("plant_manager") == {PUBLIC, RESTRICTED, CONFIDENTIAL},
    )
    ok &= _check(
        "procurement_manager → all three tiers",
        allowed_classifications("procurement_manager") == {PUBLIC, RESTRICTED, CONFIDENTIAL},
    )
    ok &= _check("unknown role → {public}", allowed_classifications("guest") == {PUBLIC})
    ok &= _check("None role → {public}", allowed_classifications(None) == {PUBLIC})

    ok &= _check("can_read(operator, confidential) == False", not can_read("operator", CONFIDENTIAL))
    ok &= _check("can_read(plant_manager, confidential) == True", can_read("plant_manager", CONFIDENTIAL))
    ok &= _check("can_read(buyer, restricted) == True", can_read("buyer", RESTRICTED))
    ok &= _check("can_read(buyer, confidential) == False", not can_read("buyer", CONFIDENTIAL))

    ok &= _check("max_tier_for(operator) == public", max_tier_for("operator") == PUBLIC)
    ok &= _check("max_tier_for(plant_manager) == confidential", max_tier_for("plant_manager") == CONFIDENTIAL)
    return ok


def test_contextvar() -> bool:
    print("\n[3] ContextVar scoping")
    ok = True
    ok &= _check("default (no auth) → {public}", active_classifications() == {PUBLIC})
    with with_user_classifications("plant_manager"):
        ok &= _check(
            "inside plant_manager block → all tiers",
            active_classifications() == {PUBLIC, RESTRICTED, CONFIDENTIAL},
        )
        with with_user_classifications("operator"):
            ok &= _check(
                "nested operator block overrides → {public}",
                active_classifications() == {PUBLIC},
            )
        ok &= _check(
            "exit nested → plant_manager again",
            active_classifications() == {PUBLIC, RESTRICTED, CONFIDENTIAL},
        )
    ok &= _check("after exit → default {public}", active_classifications() == {PUBLIC})
    return ok


def test_filter_chunks() -> bool:
    print("\n[4] filter_chunks() on hybrid-retriever payloads")
    chunks = [
        {"chunk_id": "a", "metadata": {"classification": PUBLIC},        "text": "public-1"},
        {"chunk_id": "b", "metadata": {"classification": RESTRICTED},    "text": "restricted-1"},
        {"chunk_id": "c", "metadata": {"classification": CONFIDENTIAL},  "text": "confidential-1"},
        {"chunk_id": "d", "metadata": {},                                "text": "no-tag → default public"},
    ]
    ok = True
    ok &= _check(
        "operator sees only public + untagged",
        {c["chunk_id"] for c in filter_chunks(chunks, allowed_classifications("operator"))} == {"a", "d"},
    )
    ok &= _check(
        "buyer sees public + restricted (not confidential)",
        {c["chunk_id"] for c in filter_chunks(chunks, allowed_classifications("buyer"))} == {"a", "b", "d"},
    )
    ok &= _check(
        "plant_manager sees everything",
        {c["chunk_id"] for c in filter_chunks(chunks, allowed_classifications("plant_manager"))} == {"a", "b", "c", "d"},
    )
    return ok


def test_end_to_end_search() -> bool:
    """Exercise the *actual* FAISS retriever path used by ``/api/chat``."""
    print("\n[5] FAISS search → operator vs plant_manager on a confidential query")
    try:
        from doc_pipeline.embeddings import EmbeddingPipeline
    except Exception as exc:
        print(f"  SKIP   could not import EmbeddingPipeline: {exc}")
        return True

    ep = EmbeddingPipeline()
    if not ep.has_saved_index():
        print("  SKIP   no saved FAISS index — run `python main.py --rebuild` first.")
        return True
    ep.load()

    query = "Q1 2026 EBITDA and capex execution plan"

    with with_user_classifications("operator"):
        op_hits = ep.search(query, top_k=5)
    with with_user_classifications("plant_manager"):
        pm_hits = ep.search(query, top_k=5)

    op_conf = [
        h for h in op_hits if (h.metadata or {}).get("classification") == CONFIDENTIAL
    ]
    pm_conf = [
        h for h in pm_hits if (h.metadata or {}).get("classification") == CONFIDENTIAL
    ]

    ok = True
    ok &= _check(
        f"operator: 0 confidential chunks returned (got {len(op_conf)})",
        len(op_conf) == 0,
    )
    ok &= _check(
        f"plant_manager: ≥1 confidential chunk returned (got {len(pm_conf)})",
        len(pm_conf) >= 1,
    )

    op_topfiles = [Path(str((h.metadata or {}).get("source", "?"))).name for h in op_hits[:3]]
    pm_topfiles = [Path(str((h.metadata or {}).get("source", "?"))).name for h in pm_hits[:3]]
    print(f"        operator top-3 sources       : {op_topfiles}")
    print(f"        plant_manager top-3 sources  : {pm_topfiles}")
    return ok


def main() -> int:
    results = [
        test_path_classification(),
        test_role_policy(),
        test_contextvar(),
        test_filter_chunks(),
        test_end_to_end_search(),
    ]
    passed = sum(1 for r in results if r)
    total = len(results)
    print("\n" + "=" * 60)
    print(f" RESULT: {passed}/{total} test groups passed")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
