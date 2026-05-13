# Human-in-the-Loop (HITL) Design — Hybrid GraphRAG Manufacturing

**Status:** Implemented (Phases A + B + C, opt-in via `USE_HITL=true`)
**Owners:** Manufacturing Hybrid GraphRAG team
**Last reviewed:** 2026-05-13
**Related code:** `core/criticality_classifier.py`, `core/audit_log.py`, `core/purchase_request.py`,
`pipeline/langgraph_orchestrator.py`, `api/server.py`, `app.py`

---

## 1. Goal & non-goals

### Goal
Make the pipeline **production-grade** by adding a deterministic, auditable
"auto-route small stuff, escalate big stuff" loop on top of the existing
LangGraph diagnostic engine. A configurable **criticality classifier** decides
per query whether the proposed answer (or proposed action) requires a human
sign-off before it is delivered. Rejected / paused workflows are durably
checkpointed and resumable across restarts.

### Non-goals (this iteration)
- **Authn / authz**: any caller can approve. Wiring up a real identity provider
  (OIDC, SAML) is left to the deployment.
- **Notification delivery** (Slack, email, PagerDuty): the audit log records
  events; webhooks are a future extension.
- **Multi-step approval chains**: a single approver suffices for now. Multi-stage
  workflows (e.g. supervisor → safety officer) are a Phase 4 follow-up.
- **Rich approver UX in Next.js**: Streamlit "📋 Approvals" is the canonical
  console for this iteration. Next.js integration is a Phase 4 follow-up.

---

## 2. Use cases in scope

| # | Workflow | Auto-approve when … | Escalates when … |
| - | -------- | ------------------- | ---------------- |
| 1 | **Diagnostic / repair recommendation** | Routine PM (filter swap, recalibration), critic PASSes, no safety triggers | Lockout/tagout · hot work · Class-A equipment · safety procedure cited · low critic confidence |
| 2 | **Spare-part / purchase request** _(supply-chain)_ | PO total < `HITL_AUTO_APPROVE_BELOW_USD` (default $2 000) and no rush flag | Total ≥ threshold, single-source vendor, lead time > 7 days, criticality `A` equipment |
| 3 | **Document-review approval** _(future hook in this PR)_ | Typo / metadata fix on non-controlled doc | Edits to SOPs / safety procedures / regulatory documents |
| 4 | **Knowledge-graph mutation** _(future hook)_ | Additive ingestion of new chunks | Editing / deleting `FailureMode` / `Procedure` entities or removing `RESOLVED_BY` edges |

This document covers cases 1 and 2 in code; cases 3 and 4 are designed-for via
the same `HITL Action` envelope and can be added as new intents without
changing the orchestrator.

---

## 3. Phase scope

| Phase | Ships | Ships file(s) |
| ----- | ----- | ------------- |
| **A — minimal HITL** | criticality classifier, interrupt node, in-memory checkpointer, FastAPI approval endpoints, Streamlit "📋 Approvals" tab, `USE_HITL=true` flag | `core/criticality_classifier.py`, `pipeline/langgraph_orchestrator.py`, `api/server.py`, `app.py` |
| **B — production durability** | SQLite checkpointer (`langgraph-checkpoint-sqlite`), audit log table, recent-decisions UI | `core/audit_log.py`, `data/processed/audit.sqlite` |
| **C — purchase-request domain** | `purchase_request` intent, KG vendor/part lookup, dollar-threshold rule shared with classifier | `core/purchase_request.py`, `core/criticality_classifier.py` |

---

## 4. Architecture changes

### 4.1 LangGraph topology (after Phase A)

```
START → format → retrieve → [rank_causes] → generate → criticality_check
                                                              │
                                                ┌─────────────┴─────────────┐
                                                ▼                           ▼
                                        human_approval                    critic
                                          (interrupt)                       │
                                                │             ┌─────PASS──┘
                                                ▼             ▼
                                            critic           END
                                                │
                                          PASS / max
                                                ▼
                                              END
```

The new nodes:

* **`criticality_check`** — pure function: rules first (cheap, explainable),
  optional LLM grader for the inconclusive band (`0.3 < score < 0.7`).
* **`human_approval`** — calls `interrupt(payload)`, persisted by the
  checkpointer. When resumed via `Command(resume=...)`, the returned `decision`
  flows into the rest of the graph. If `approved=False`, the graph short-circuits
  to END with a rejection annotation.

### 4.2 State additions

```python
class GraphState(TypedDict, total=False):
    # … existing fields …
    risk: Dict[str, Any]                # Risk(score, drivers, needs_human).to_dict()
    purchase_request: Dict[str, Any]    # parsed PO request (Phase C)
    human_decision: Dict[str, Any]      # {"approved", "approver", "comments", "edited_answer"}
    pipeline_status: str                # "complete" | "awaiting_approval" | "rejected"
```

### 4.3 ManufacturingPipeline contract

`PipelineResult` gains four fields:

```python
@dataclass
class PipelineResult:
    # … existing fields …
    risk: Optional[Dict] = None
    requires_approval: bool = False
    approval_thread_id: Optional[str] = None
    rejected: bool = False
```

`diagnostic(query)` returns a `PipelineResult` whose `requires_approval=True`
when the graph paused at an interrupt. The new method
`resume_diagnostic(thread_id, decision)` returns the final `PipelineResult`
after the human responds.

### 4.4 FastAPI surface

```
GET  /api/approvals/pending              → [{thread_id, ts, summary, drivers, …}]
GET  /api/approvals/{thread_id}          → full snapshot (state + risk + payload)
POST /api/approvals/{thread_id}/resume   → {approved, approver, comments, edited_answer?}
GET  /api/audit                          → recent decisions (paginated)
```

POST `/api/chat` is unchanged for the happy path. When an approval is required
the response body now includes `awaiting_approval: true` plus the `thread_id`,
and the ChatAgent emits a turn of `kind="approval_pending"` so the UI can
render a clear banner.

### 4.5 Streamlit "📋 Approvals" tab

| Column         | Source                              |
| -------------- | ----------------------------------- |
| Pending queue  | `GET /api/approvals/pending`        |
| Risk drivers   | `risk["drivers"]` from the pause    |
| Proposed answer| `state["answer"]`                   |
| Evidence       | `state["evidence"]`                 |
| Approve / Reject + free-text comment | `POST /api/approvals/{id}/resume` |
| Audit log (last N) | `GET /api/audit`               |

The tab is visible only when `USE_HITL=true` (server returns the flag in
`/api/health`).

---

## 5. State machine

```
                  ┌─────────────────┐
                  │   awaiting_     │  POST /api/approvals/{id}/resume
                  │   approval      │ ─────────────────────────────────► approved? ─┐
                  └────────▲────────┘                                                │
                           │ interrupt()                                             │
                  ┌────────┴────────┐                                                │
   user query →   │  in_progress    │                                                │
                  └────────┬────────┘                                                │
                           │  no escalation                                          │
                           ▼                                                         │
                  ┌─────────────────┐                                                │
                  │   complete      │ ◄──────────────────────────────────────────────┘
                  └─────────────────┘   (approved=true → critic + END)
                                                                                     │
                                                                                     ▼
                                                                            ┌──────────────────┐
                                                                            │   rejected       │
                                                                            └──────────────────┘
                                                                            (approved=false
                                                                             → END with reason)
```

Terminal states: `complete`, `rejected`.

---

## 6. Configuration

| Env var                           | Default                | Notes |
| --------------------------------- | ---------------------- | ----- |
| `USE_HITL`                        | `false`                | Master switch for all of Phase A/B/C. |
| `HITL_RISK_THRESHOLD`             | `0.6`                  | `Risk.score >= this` ⇒ human approval. |
| `HITL_AUTO_APPROVE_BELOW_USD`     | `2000`                 | Used by the purchase-request classifier (Phase C). |
| `HITL_HIGH_RISK_KEYWORDS`         | `lockout,tagout,hot work,fire,explosion,h2s,arc flash,confined space,fatal,injury,death,toxic` | Comma-separated. Substring match (case-insensitive) on the proposed answer + the user query. |
| `HITL_DB_PATH`                    | `data/processed/audit.sqlite` | Used for the audit log + SQLite checkpointer. |
| `HITL_CHECKPOINT_BACKEND`         | `sqlite`               | `memory` (Phase A) or `sqlite` (Phase B). Auto-falls back to `memory` if the SQLite library isn't installed. |

All flags follow the same opt-in convention as `USE_LANGGRAPH` / `USE_CAUSE_RANKING`.

---

## 7. Phase B — durability

* SQLite checkpointer (`langgraph-checkpoint-sqlite>=2.0`) — persists the
  `GraphState` per `thread_id` to `HITL_DB_PATH`. Survives process restarts.
* `core/audit_log.py` — single SQLite DB (separate table from the checkpointer)
  capturing `(ts, thread_id, decision, approver, drivers, comments, query,
  proposed_answer)`. Append-only. Write happens at every approve / reject.
* `GET /api/audit` returns the most recent `N` rows for the Streamlit tab.

If SQLite is unavailable for any reason (file-system permission, missing
package), the server logs a warning and silently falls back to in-memory
checkpointing — failing-open is preferable to losing the conversation.

---

## 8. Phase C — purchase-request domain

* `core/purchase_request.py` parses a structured `PurchaseRequest` from the
  user query when intent looks like a PO request (`buy`, `purchase`, `order`,
  `request part`, `PO for`, `replace`, etc.).
* The KG is queried for the part / vendor:
  `Equipment ─REQUIRES_PART→ SparePart ─SUPPLIED_BY→ Vendor`. If the part
  exists, the parsed request is enriched with KG facts (lead time, single-
  source flag, last-known price).
* The classifier picks up the `PurchaseRequest` from `state["purchase_request"]`
  and applies the dollar-threshold rule:
  * Auto-approve when `total_usd < HITL_AUTO_APPROVE_BELOW_USD` AND not
    single-source AND `lead_time_days <= 7`.
  * Otherwise escalate.

The HITL plumbing is identical — the only addition is a richer payload that
the Streamlit tab displays as a vendor card alongside the proposed message.

---

## 9. API examples

### Pause

`POST /api/chat`

```json
{
  "session_id": "abc",
  "new_turns": [
    { "role": "assistant", "kind": "approval_pending",
      "content": "This action needs supervisor approval before I proceed.",
      "meta": { "thread_id": "thr_…", "drivers": ["safety_keyword:lockout"], "score": 0.85 }
    }
  ],
  "awaiting_approval": true,
  "approval_thread_id": "thr_…"
}
```

### Resume

`POST /api/approvals/thr_…/resume`

```json
{
  "approved": true,
  "approver": "rajesh@plant",
  "comments": "Confirmed scheduled outage tonight 22:00–02:00."
}
```

Response is the final answer turn, identical in shape to a normal `/api/chat`
response.

---

## 10. Out-of-scope (deliberately)

| Item                                       | Why deferred                                  |
| ------------------------------------------ | --------------------------------------------- |
| OIDC / SAML auth                           | Belongs in the deployment infra, not the app. |
| Multi-stage approval chains                 | Domain-specific; design depends on the customer. |
| Slack / email / PagerDuty webhooks         | Easy to add via an `audit_log` listener; not needed for the demo. |
| RBAC on `/api/approvals/*`                 | Deployment concern (e.g. Caddy / nginx auth). |
| Approver UX in Next.js                     | Streamlit covers the operations console for this PR. |
| ERP integration for `purchase_request`     | Stub is enough to demo the agentic pattern; real ERPs are bespoke. |

---

## 11. Rollout

1. Merge with `USE_HITL=false` (default). Zero behavioural change.
2. Internal demo with `USE_HITL=true` and the in-memory checkpointer (Phase A only).
3. Flip `HITL_CHECKPOINT_BACKEND=sqlite` for an environment that needs durability (Phase B).
4. Pilot the supply-chain hook with a small parts catalogue (Phase C).
5. Add OIDC + Slack notifications (Phase 4, post-merge).

---

## 12. Test plan

* Unit tests on `criticality_classifier` rules (keywords, dollar threshold,
  intent gating).
* End-to-end smoke test on the LangGraph orchestrator with `interrupt` →
  resume cycle using a mocked LLM (no external API calls).
* SQLite round-trip test: write checkpoint, read it back from a fresh
  process, resume.
* `purchase_request` parser: verify regex extraction, KG lookup happy path,
  graceful fallback when the KG doesn't know the part.

---

## 13. Open questions / future work

* Should the critic loop run **before** or **after** the human approval gate?
  Current design: **after** (human approval → critic → END), so the critic can
  validate any inline edits the approver made. We can revisit if it confuses
  approvers.
* Should `purchase_request` write to an actual ERP after approval, or just
  emit a webhook? Out of scope for this PR.
* Multi-tenant audit log partitioning (per-plant / per-business-unit) — punt
  until we have a real customer asking for it.
