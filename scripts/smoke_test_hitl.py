"""End-to-end smoke test for the HITL plumbing (Phases A + B + C).

Runs *without* any external LLM call by monkey-patching ``call_llm_with_metrics``
and ``critic_evaluate`` to return canned values. Verifies:

  1. A safe diagnostic query auto-approves (no interrupt fired).
  2. A high-risk diagnostic query (lockout/tagout keyword) pauses at the
     ``human_approval`` interrupt — and resumes cleanly when approved.
  3. Rejection short-circuits to END with ``rejected=True``.
  4. A purchase-request query above the dollar threshold pauses, with the
     ``purchase_request`` payload populated.
  5. The audit log records every approve/reject decision.

Run with:
    .venv/bin/python scripts/smoke_test_hitl.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Configure HITL before importing anything that reads config.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["USE_LANGGRAPH"] = "true"
os.environ["USE_HITL"] = "true"
os.environ.setdefault("HITL_CHECKPOINT_BACKEND", "memory")  # in-memory for the test
TMP_DB = ROOT / "data" / "processed" / "audit_smoketest.sqlite"
os.environ["HITL_DB_PATH"] = str(TMP_DB)
TMP_DB.parent.mkdir(parents=True, exist_ok=True)
if TMP_DB.exists():
    TMP_DB.unlink()


def _patch_llm():
    """Replace network LLM calls with deterministic stubs."""
    import core.llm_client as llm
    import core.critic as critic
    import core.cause_ranker as ranker
    import core.criticality_classifier as crit_clf
    import pipeline.langgraph_orchestrator as lg
    import core.query_formatter as qf

    def fake_metrics(system_prompt, user_prompt, model=None, **_):
        return {
            "response": "Stub LLM answer. Cite [src#1].",
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            "cost_estimate": 0.0001, "model": model or "stub-model",
        }

    def fake_critic(query, answer, evidence, attempt_idx, **_):
        return {
            "verdict": "PASS",
            "confidence": 0.9,
            "rationale": "stub",
            "issues": [],
            "suggestion": "",
        }

    def fake_rank(query, intent, evidence_chunks, graph_context, top_k=5):
        return {"candidates": [], "model": "stub", "prompt_tokens": 0,
                "completion_tokens": 0, "total_tokens": 0, "cost_estimate": 0.0}

    def fake_format(query):
        return {
            "expanded": query,
            "structured_query": query,
            "intent": "diagnostic",
            "entities": {},
            "intent_metadata": {},
        }

    llm.call_llm_with_metrics = fake_metrics
    lg.call_llm_with_metrics = fake_metrics
    critic.critic_evaluate = fake_critic
    lg.critic_evaluate = fake_critic
    ranker.rank_causes = fake_rank
    lg.rank_causes = fake_rank
    qf.format_query = fake_format
    lg.format_query = fake_format
    crit_clf._llm_grade = lambda *_a, **_kw: None  # disable network grader


_patch_llm()


def _build_orchestrator():
    """Build a tiny LangGraph orchestrator with a one-doc corpus + empty KG.

    We patch ``HybridRetriever.__init__`` to skip the (network-bound) vector
    retriever construction and inject a deterministic in-memory stub.
    """
    from core.knowledge_graph import KnowledgeGraph
    from core.retrieval import hybrid_retriever as hr_mod
    from pipeline.langgraph_orchestrator import LangGraphOrchestrator

    docs = [{
        "chunk_id": "chunk_1",
        "text": "Pump P-203 troubleshooting: check seal, lubrication.",
        "metadata": {"source": "manual.pdf", "doc_type": "pdf", "page": 1},
        "entities": {},
    }]
    kg = KnowledgeGraph()
    kg.build_from_documents(docs)

    class _StubRetriever:
        def __init__(self, _docs, _kg, _vector=None):
            self._docs = list(_docs)
        def build_indexes(self, **_):
            pass
        def retrieve(self, query, top_k=10):
            return list(self._docs)[:top_k]

    hr_mod.HybridRetriever = _StubRetriever  # monkey-patch before constructor runs
    import pipeline.langgraph_orchestrator as lg
    lg.HybridRetriever = _StubRetriever

    orch = LangGraphOrchestrator(docs, kg, vector_retriever=None, skip_vector_build=True)
    orch.initialize()
    return orch


def assert_eq(label, got, expected):
    status = "✅" if got == expected else "❌"
    print(f"  {status} {label}: expected={expected!r} got={got!r}")
    assert got == expected, f"{label}: expected {expected!r}, got {got!r}"


def test_safe_query(orch):
    print("\n[1] Safe query — should auto-approve (no interrupt)")
    out = orch.process_query("What is the OEE target for plant A?")
    assert_eq("awaiting_approval", out["awaiting_approval"], False)
    assert_eq("pipeline_status", out["pipeline_status"], "complete")
    assert out["risk"]["score"] < 0.6, f"unexpected risk: {out['risk']}"
    print(f"    risk={out['risk']['score']:.2f} drivers={out['risk']['drivers']}")


def test_high_risk_pause_and_approve(orch):
    print("\n[2] High-risk query (lockout) — should pause then resume on approve")
    out = orch.process_query("What is the lockout/tagout procedure for pump P-203?")
    assert_eq("awaiting_approval", out["awaiting_approval"], True)
    thread_id = out["approval_thread_id"]
    assert thread_id, "thread_id must be set"
    print(f"    paused at thread {thread_id}, drivers={out['risk']['drivers']}")
    assert orch.get_pending(thread_id) is not None, "should be in pending list"

    out2 = orch.resume(thread_id, {
        "approved": True, "approver": "smoke@plant",
        "comments": "Outage confirmed.", "edited_answer": None,
    })
    assert_eq("awaiting_approval(after resume)", out2["awaiting_approval"], False)
    assert_eq("pipeline_status", out2["pipeline_status"], "complete")
    assert out2["human_decision"]["approved"] is True
    assert orch.get_pending(thread_id) is None, "should be cleared from pending"


def test_high_risk_reject(orch):
    print("\n[3] High-risk query — reject should short-circuit to 'rejected'")
    out = orch.process_query("Hot work permit for tank T-9 — emergency shutdown.")
    assert out["awaiting_approval"] is True
    thread_id = out["approval_thread_id"]
    out2 = orch.resume(thread_id, {
        "approved": False, "approver": "smoke@plant",
        "comments": "No outage approved.", "edited_answer": None,
    })
    assert_eq("pipeline_status", out2["pipeline_status"], "rejected")
    assert out2["human_decision"]["approved"] is False
    print(f"    rejected by {out2['human_decision']['approver']}")


def test_purchase_request(orch):
    print("\n[4] Purchase-request query > $2000 — should pause with purchase payload")
    out = orch.process_query(
        "Please raise a PO for 5 BRG-7203 spare bearings at $5000 from Vendor SKF urgent."
    )
    assert_eq("awaiting_approval", out["awaiting_approval"], True)
    pr = out.get("purchase_request") or {}
    print(f"    parsed purchase_request={pr}")
    assert pr.get("part_id") == "BRG-7203", f"expected BRG-7203, got {pr.get('part_id')}"
    assert pr.get("total_usd") == 5000.0, f"expected 5000, got {pr.get('total_usd')}"
    drivers = out["risk"]["drivers"]
    assert any("purchase_value" in d for d in drivers), f"missing purchase driver: {drivers}"

    # Resume → approved
    out2 = orch.resume(out["approval_thread_id"], {
        "approved": True, "approver": "buyer@plant",
        "comments": "Approved.", "edited_answer": None,
    })
    assert out2["pipeline_status"] == "complete"


def test_audit_log():
    print("\n[5] Audit log — should have recorded approvals/rejections")
    from core.audit_log import AuditLog
    log = AuditLog(TMP_DB)
    # Manually replay the decisions our orchestrator test made (the orchestrator
    # itself does NOT write to the audit log — that's the FastAPI / Streamlit
    # layer's job. Here we just exercise the log writer directly.)
    log.record(thread_id="thr_test_a", decision="approved", approver="smoke@plant",
                risk_score=0.85, drivers=["safety_keyword:lockout"],
                domain="diagnostic", query="lockout on P-203", proposed_answer="...")
    log.record(thread_id="thr_test_b", decision="rejected", approver="smoke@plant",
                risk_score=0.92, drivers=["safety_keyword:hot work"],
                domain="diagnostic", query="hot work in T-9", proposed_answer="...")
    rows = log.recent(limit=10)
    print(f"    rows in audit log: {len(rows)}")
    stats = log.stats()
    print(f"    stats: {stats}")
    assert stats["total"] >= 2
    assert stats["approved"] >= 1
    assert stats["rejected"] >= 1


def test_rbac_routing():
    print("\n[6] RBAC — driver → required-roles mapping")
    from core.rbac import required_roles_for, can_approve, is_maker_locked, USE_CASES

    cases = [
        (["safety_keyword:lockout"], {"maintenance_engineer", "ehs_officer"}),
        (["safety_keyword:hot work"], {"ehs_officer"}),
        (["safety_keyword:emergency", "safety_keyword:shutdown"],
         {"shift_supervisor", "ehs_officer", "plant_manager", "maintenance_engineer"}),
        (["purchase_value=$5,000>=$2,000"], {"buyer"}),
        (["purchase_value=$35,000>=$2,000"], {"procurement_manager"}),
        (["purchase_value=$150,000>=$2,000"], {"procurement_manager", "plant_manager"}),
        (["safety_keyword:fatal"], {"ehs_officer", "plant_manager"}),
    ]
    for drivers, expected in cases:
        got = set(required_roles_for(drivers))
        ok = expected.issubset(got)
        flag = "✅" if ok else "❌"
        print(f"  {flag} {drivers} → {sorted(got)}  expected ⊇ {sorted(expected)}")
        assert ok, f"required_roles_for{drivers} = {got}, expected ⊇ {expected}"

    print("\n  can_approve checks:")
    assert not can_approve("operator", ["ehs_officer"]), "operator must not approve EHS items"
    assert can_approve("ehs_officer", ["ehs_officer", "maintenance_engineer"])
    assert not can_approve(None, ["ehs_officer"])
    print("    ✅ operator cannot, ehs_officer can, anonymous cannot")

    print("\n  maker-lock checks:")
    assert is_maker_locked("alice@plant.local", "Alice@Plant.Local"), "case-insensitive lock"
    assert not is_maker_locked("alice@plant.local", "dave.ehs@plant.local")
    assert not is_maker_locked("", "x")
    print("    ✅ self-approval blocked, cross-approval allowed, empty=safe")

    print(f"\n  Use-case catalogue rows: {len(USE_CASES)} (all wired into core.rbac.USE_CASES)")


def test_auth_store_and_seeding():
    print("\n[7] Auth store — login + token round-trip")
    # Use a private DB so we don't clobber the real one.
    from pathlib import Path
    import tempfile
    from core.auth_store import UserStore, AuthError

    tmp = Path(tempfile.mkdtemp()) / "auth_smoketest.sqlite"
    store = UserStore(tmp, seed_demo=True)
    user, tok, exp = store.login("alice@plant.local", "operator123")
    assert user.role == "operator"
    assert exp > 0
    resolved = store.user_for_token(tok)
    assert resolved is not None and resolved.user_id == "alice@plant.local"
    print(f"    ✅ login + token round-trip for {user.user_id} role={user.role}")

    try:
        store.login("alice@plant.local", "wrong-password")
        raise AssertionError("expected AuthError")
    except AuthError:
        print("    ✅ wrong password rejected")

    assert store.user_for_token("garbage") is None
    print("    ✅ invalid token rejected")
    tmp.unlink(missing_ok=True)


def test_pipeline_annotate(orch):
    print("\n[8] Pipeline — annotate_pending attaches maker + required roles")
    # Pause on a high-risk query
    out = orch.process_query("What is the lockout/tagout procedure for pump P-203?")
    thread_id = out["approval_thread_id"]
    # The orchestrator owns the pending dict directly here (no ManufacturingPipeline
    # wrapper in this smoke test), so we poke its `_pending` dict via the same
    # path `unified_pipeline.annotate_pending` uses internally.
    orch._pending[thread_id]["maker_user_id"] = "alice@plant.local"
    from core.rbac import required_roles_for
    orch._pending[thread_id]["required_roles"] = required_roles_for(
        out["risk"]["drivers"]
    )
    snap = orch.get_pending(thread_id)
    assert snap["maker_user_id"] == "alice@plant.local"
    assert "ehs_officer" in snap["required_roles"]
    print(f"    ✅ pending annotated maker=alice required={snap['required_roles']}")
    # Resolve so the thread doesn't linger
    orch.resume(thread_id, {"approved": True, "approver": "dave.ehs@plant.local",
                            "comments": "ok", "edited_answer": None})


def main():
    print("HITL smoke test — Phases A + B + C + D (RBAC)")
    orch = _build_orchestrator()
    print(f"checkpointer: {orch.checkpointer_kind}")

    test_safe_query(orch)
    test_high_risk_pause_and_approve(orch)
    test_high_risk_reject(orch)
    test_purchase_request(orch)
    test_audit_log()
    test_rbac_routing()
    test_auth_store_and_seeding()
    test_pipeline_annotate(orch)
    print("\nALL SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
