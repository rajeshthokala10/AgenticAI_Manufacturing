# Architecture Decisions

A running log of the non-obvious choices in this repo and the reasoning
behind them. New decisions go at the top; date format is `YYYY-MM-DD`.

---

## 2026-05-14 — Apply kgrag's three-tier KG model

**What:** `core/knowledge_graph.py` was refactored from a single-tier
extraction dump into kgrag's three-tier model:

1. **Schema** (`schemas/manufacturing.yaml` + `core/kg/schema.py`) — the
   hand-curated ontology. Entity types declare `id_pattern` or
   `vocabulary` for closed validation; edge types declare allowed
   `source`/`target` types and `min_cardinality`/`max_cardinality`.
   Every node and edge is schema-validated at build time; out-of-schema
   candidates land in `kg.rejected()` instead of polluting the graph.

2. **Instances** (`core/kg/extractors/`) — three extractors with
   descending confidence:
   - `CodeExtractor` — deterministic regex over chunk text (codes / IDs),
     `confidence=1.0`, author `system:code`.
   - `MetadataExtractor` — pre-parsed entity lists from chunk metadata,
     `confidence=0.95`, author `system:metadata`. Emits the cross-product
     co-occurrence edges (Equipment→Alarm, etc.) that the legacy KG
     derived inline.
   - `NarrativeExtractor` — keyword + regex heuristics over prose
     (Symptoms, Procedures, Components), `confidence=0.5-0.7`, author
     `system:llm_extract`. These are *candidates* — the gap detector
     surfaces them for HITL review by default.

3. **Provenance** (`core/kg/provenance.py`) — every node and edge carries
   `{author, confidence, source_chunk_id, timestamp, supersedes}`.
   Authors are namespaced (`system:`, `import:`, `user:`) so retrieval
   can prefer high-trust sources and the gap detector can identify
   human-resolved edges.

**Why:** the single-tier KG conflated assertion strength. A symptom
extracted from prose was indistinguishable from an equipment ID extracted
from a structured PDF table. Hallucination risk in the retrieval allow-
list scaled with that conflation. Splitting source-confidence into a
first-class field lets the retrieval floor be tuned (`KG_RETRIEVAL_MIN_CONFIDENCE`)
without retraining anything, and gives the HITL UI a principled
"which edges need review" signal.

**Specifics added:**
- `core/kg/gap_detector.py` walks the graph against the schema and
  surfaces `MISSING_EDGE` / `CONFLICTING_EDGES` / `LOW_CONFIDENCE_EDGE`.
- `KnowledgeGraph.record_human_edge()` is the HITL writeback. It stamps
  `user:<id>` provenance with `supersedes` set to the original edge id;
  the gap detector then ignores the now-resolved edge.
- `KnowledgeGraph.get_allow_list(query, min_confidence=N)` is the
  provenance-aware retrieval filter, threaded through `HybridRetriever`
  via the new `KG_RETRIEVAL_MIN_CONFIDENCE` env var (default 0.0 = legacy).

**Migration:** previously-saved graph JSON files still load — the
`Provenance.from_dict` handler returns `None` for legacy entries, and
all provenance-aware code paths degrade gracefully when provenance is
missing. A `python main.py --rebuild` repopulates with full provenance.

**What's deferred:**
- A FastAPI route for HITL gap-review (the orchestration plumbing exists;
  the UI endpoint is a small additional surface).
- An LLM-backed `NarrativeExtractor` replacement (the current keyword
  version is a placeholder — the interface accepts any `Extractor`
  subclass, so swapping it is a one-class change).
- CMMS / FMEA system-of-record importers (would be a new
  `extractors/import_cmms.py` with author `import:cmms`).

The pattern is the kgrag L1–L7 contract applied to a single domain;
when kgrag itself stabilises its public API, this implementation can
delegate to it instead.

---

## 2026-05-14 — Port piston-engine-copilot patterns into the manufacturing stack

Five distinct changes landed together as part of "implement what piston is
having into manufacturing." Each is independent and reversible by flipping
the relevant env flag back to the prior default.

### 1. Vector store: FAISS → Qdrant

**What:** The FAISS-backed `EmbeddingPipeline` was rewritten to use
`qdrant-client` (embedded on-disk by default at
`doc_pipeline/vector_store/qdrant/`; remote via `QDRANT_URL=http://host:6333`).
The public API of `EmbeddingPipeline` is unchanged — `build_index`, `search`,
`save`, `load`, `has_saved_index`, `.dimension`, `.chunks`, and the
`.index.ntotal` shim are preserved so every caller continues to work.

**Why:** Qdrant gives us (a) first-class payload filtering, which the
document-ACL layer already needs and the legacy FAISS code had to
re-implement above the result list; (b) a clean upgrade path from embedded
on-disk to a remote server by changing one env var; (c) point-level
upserts which simplify incremental ingestion. FAISS is faster on pure
similarity search but the ACL filter + metadata join we need on every
query mean the win was already eroded.

**Migration:** Saved FAISS indexes (`*.faiss`, `*_embeddings.npy`) are
ignored by the new loader. Run `python main.py --rebuild` once to
regenerate. The chunk index JSON (`*_chunks.json`) is still loaded.

### 2. Embedding model: `all-MiniLM-L6-v2` → `BAAI/bge-small-en-v1.5`

**What:** Default `EMBEDDING_MODEL` flipped. Same 384-dim, drop-in.

**Why:** bge-small consistently outperforms MiniLM on technical / industrial
retrieval (MTEB shows ~3-5 points lift on Trec-COVID, NF-Corpus, FiQA-style
benchmarks that resemble manufacturing manuals + work orders). Existing
`.env` files keep MiniLM unless explicitly updated, so this is a
no-impact-on-existing-deployments change.

### 3. Cross-encoder reranker: optional → on by default

**What:** `USE_RERANKER` default flipped from `false` to `true`. Reranker
model already defaulted to `BAAI/bge-reranker-base`; that did not change.

**Why:** The reranker reads `(query, chunk)` jointly rather than matching
independent embeddings — 5-15% answer-quality lift on noisy corpora — and
the cost is ~50ms per query after warm-up. Already pinned as a transitive
dependency of sentence-transformers, so no new packages. Existing
deployments that wanted the old behaviour can set `USE_RERANKER=false`
explicitly.

### 4. Configuration: `os.getenv` → `pydantic-settings`

**What:** `config.py` rewritten on top of `pydantic_settings.BaseSettings`.
Module-level constants (`EMBEDDING_MODEL`, `USE_RERANKER`, …) are still
exported for back-compat with the dozens of `from config import X`
call-sites across `core/`, `pipeline/`, `doc_pipeline/`, `comparison/`,
and the front-ends. The canonical instance is `config.settings`.

**Why:** Typed validation at startup catches `.env` typos that previously
silently turned features off (e.g. `USE_RERANKER=on` was accepted but
`use_reranker=TRUE` is now coerced correctly). Easier to introspect from
the FastAPI `/api/health` endpoint and to override in tests.

### 5. Two-stage generation (piston pattern)

**What:** New module `core/procedure_drafter.py`. When
`USE_PROCEDURE_DRAFTING=true` and the query intent is troubleshooting,
the diagnostic graph replaces the free-form answer LLM call with a
structured procedure call returning JSON of the shape
`{"steps": [{"step": int, "action": str, "citations": [chunk_id, ...]}]}`.

The rendered Markdown of that procedure becomes the legacy `answer`
string so the critic + guardrails operate on the same surface. The
structured object is additionally surfaced at `result.procedure` and
through `/api/chat`, `/api/chat/stream`.

**Why:** Piston ships exactly this two-stage pattern (cause-ranking →
procedure-drafting) and its outputs are noticeably more navigable than
a single answer blob — citations are per-step, sequencing is explicit
(safety preconditions before component handling), and Streamlit / Next.js
can render the structure incrementally during streaming.

### 6. Fixed cause taxonomy enforcement (piston pattern)

**What:** `core/cause_ranker.py` now accepts an optional
`CAUSE_TAXONOMY` env var (comma-separated). When set, the LLM is told
to constrain its cause names to that allow-list, and the parser drops
anything outside it (anti-hallucination guarantee). When empty (default),
the prior free-form behaviour is preserved.

**Why:** Piston gets a measurable lift on top-cause-match accuracy from
this constraint because the LLM cannot drift to synonyms. We don't ship
a default taxonomy because manufacturing is multi-domain — the operator
configures one to match their fault dictionary.

### 7. Streaming pipeline render

**What:** `LangGraphOrchestrator.stream_query()` yields per-node updates
via `graph.stream(stream_mode="updates")`. Exposed through
`ManufacturingPipeline.diagnostic_stream()`, the `/api/chat/stream`
FastAPI SSE endpoint, and a "Stream pipeline stages" toggle in the
Streamlit Diagnostic tab.

**Why:** The diagnostic graph runs ~4-15s end-to-end (longer with the
cross-encoder reranker cold-start). A blocking UI loses the user during
that window. Piston's incremental render is the right pattern.

### 8. Eval harness — hard-target metrics

**What:** `GoldenItem` gains optional `expected_top_cause` /
`expected_subsystem` fields. When present, the per-record metrics include
boolean `top_cause_match` / `subsystem_match`, and the aggregate report
adds `top_cause_match_rate` / `subsystem_match_rate` (denominator is just
the records that declared an expectation).

**Why:** Piston's eval is brutal-but-honest: 4/10 top-cause accuracy is
visible at a glance. Soft RAGAS-style metrics (faithfulness, relevancy)
can drift up while real diagnostic accuracy stays low — these hard
targets keep the harness grounded.

### Deliberately NOT ported

- **Modular YAML/Jinja domain abstraction** (piston's `modular/`). That
  would have meant moving every manufacturing-specific assumption
  (ontology types, alarm-code regexes, safety keyword list) into config
  and threading a domain parameter through every call. It's a real
  generalization — but at the cost of a ~1000 LOC refactor that touches
  every module. Out of scope for this pass.
- **Anthropic LLM backend.** Already supported indirectly through any
  OpenAI-compatible proxy; adding a first-class `anthropic` SDK path
  duplicated logic for no concrete deployment that needed it.
- **Per-step retry feedback in the procedure drafter.** Piston re-runs
  procedure_drafting with the critic's `suggested_fixes`. We left the
  retry path as the legacy free-form answer LLM — surgical change,
  preserves existing semantics, and the critic rarely rejects a
  structured procedure that already has per-step citations.

---

## Pre-existing decisions (inferred from commit history)

- **HITL = LangGraph only.** The procedural orchestrator cannot pause
  for human approval because there is no durable checkpoint to resume
  from. `USE_HITL=true` implicitly requires `USE_LANGGRAPH=true`; the
  Streamlit / FastAPI surfaces enforce this.
- **Tiered model routing.** Expensive answer models are reserved for
  user-facing output (`ANSWER_MODEL`, `RETRY_MODEL`, `PROCEDURE_MODEL`
  default to `gpt-4o`); cheap local Qwen handles classification, critic,
  and tool-planner roles. Override per-task via env.
- **Reranker uses normalised blending.** RRF and cross-encoder scores
  are min-max normalised before being blended with weight
  `RERANK_BLEND_WEIGHT=0.7`, so a single retriever with a runaway score
  cannot dominate the final ranking.
