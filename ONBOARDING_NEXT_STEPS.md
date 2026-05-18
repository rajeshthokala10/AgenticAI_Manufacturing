# Onboarding Agent — Next Steps

**Status as of 2026-05-17.** The four-stage pipeline (`core/onboarding_agent.py`) plus deterministic vocab miner (`core/vocab_miner.py`) ships an EV-manufacturing schema that passes all 7 validation gates with 0 repair attempts. The remaining gap is **coverage** — the LLM under-uses the ~40 mined clusters and emits only 2-3 vocab terms per entity type. Below: prioritised follow-ups.

---

## P0 — Highest leverage, smallest changes

### 1. Switch onboarding to a reasoning model

**File:** `core/llm_router.py` line 87-95 (`PROFILES["cloud"]`)

**Change:** `"onboarding": "gpt-4o"` → `"onboarding": "o4-mini"` (or `"o3"`).

**Why:** Stage A's job is now "for each of these 40 clusters, decide signal-vs-noise and assign an entity type." That's a thorough labelling task — exactly what reasoning models do well. gpt-4o does it in a single forward pass and produces conservative output (2-3 terms per type, ignoring the rest). A reasoning model with chain-of-thought consistently exercises broader coverage on labelling tasks.

**Cost:** onboarding is once-per-domain. Cost is irrelevant.

**Risk:** o-series models occasionally have different output-format conformance. Verify the `STRICT JSON` instruction still produces parseable output. If not, see P0-2 (Structured Outputs) which makes this moot.

**Estimated effort:** 1-line config change + smoke test.

---

### 2. Adopt OpenAI Structured Outputs / JSON Schema mode

**Files:** `core/llm_client.py` (the `call_llm()` wrapper); `core/onboarding_agent.py` (`_parse_json_response`, `_characterize_corpus`, `_repair_schema`).

**Change:** thread a `response_format={"type": "json_schema", "json_schema": {...}}` argument through `call_llm`. Define Pydantic models for Stage A / Stage B / repair outputs and pass their `model_json_schema()` to the API.

**Why:**
- Eliminates the "model wraps output in ```json fences" failure mode — the existing fence-stripping logic in `_parse_json_response` becomes unnecessary.
- The API *guarantees* valid JSON conforming to the schema. No more parse-failure fallbacks.
- Moves the response shape contract out of the prompt (where it consumes tokens) and into the API.

**Estimated effort:** ~50 lines of changes; 2-3 Pydantic models.

---

### 3. Fix the silent YAML quoting bug

**File:** wherever the schema gets persisted (currently `save_schema` in `core/onboarding_agent.py:728`).

**Change:** before writing, re-serialise the YAML through `yaml.safe_dump(..., default_flow_style=False)`. Manually quote `display.color` and `display.emoji` if the model emits them unquoted.

**Why:** Every recent EV draft has `color: #0EA5E9` (unquoted) which YAML parses as `color: None` (`#` starts a comment). The UI silently falls back to a default color. The schema loader doesn't fail because color is informational.

**Estimated effort:** 5-line post-process.

---

## P1 — Coverage improvements

### 4. Add a vocab-coverage gate (Gate 8)

**File:** `core/onboarding_agent.py:_validate_schema`

**Change:** after Gate 5, add a warning (not error) if any declared closed-vocab type has fewer than 5 terms when the miner had ≥ 10 high-coverage clusters available. Surface in the UI so users see "you might be under-using available vocab."

**Why:** Gate 5 verifies that what was emitted is grounded, but doesn't catch the symmetric problem — too few terms emitted relative to what the corpus contained. Combined with P0-1 (reasoning model) this gate would warn instead of silently shipping a thin schema.

**Estimated effort:** 20 lines.

---

### 5. Add an adversarial critic stage (Stage B½)

**File:** new section in `core/onboarding_agent.py` between Stage B and Stage C.

**Change:** a second LLM call with a different persona ("skeptical domain expert; find every block that doesn't match the corpus; be harsh") reviewing the Stage B draft. Output is fed to Stage D-style repair before Stage C runs.

**Why:** Stage C catches mechanical errors (vocab not in corpus, fabricated IDs). The critic would catch semantic drift the mechanical gates miss — wrong intents in `classify_system`, mismatched `procedure_system` safety preconditions, persona-archetype mismatch, etc.

**Estimated effort:** ~80 lines + new system prompt.

---

### 6. Cache Stage A characterizations + miner output by corpus hash

**Files:** `core/onboarding_agent.py:analyze`, `app.py:1430-1448` (Streamlit wizard flow).

**Change:** key a process-level LRU cache on `sha1(domain_id + concat(docs))`. Stage A and the miner are deterministic-or-near-deterministic — re-running them on every Q&A turn wastes tokens and CPU.

**Why:** the multi-turn Q&A flow currently re-mines and re-characterizes on every `analyze()` call (unless the caller passes `corpus_characterization=` back, which `app.py` doesn't).

**Estimated effort:** 20 lines.

---

## P2 — Polish & operational

### 7. Persist Stage A characterizations alongside the schema

**Change:** write each Stage A output to `schemas/<domain>.characterization.json` next to the YAML. Enables audit trail, schema-drift detection, re-onboarding diffs.

---

### 8. Add tests for the new gates

**File:** new `tests/test_onboarding_gates.py`.

**Change:** unit test each of Gates 5/6/7 with a fixture YAML + fixture corpus. Smoke test that runs `analyze()` against a known-good docset and asserts archetype + gates pass.

**Why:** no tests exist for the onboarding agent at all. If the validator is ever simplified or refactored, these gates will silently regress.

---

### 9. Multilingual support in the miner

**File:** `core/vocab_miner.py:_load_spacy`

**Change:** detect language with fasttext or langdetect; switch to `xx_ent_wiki_sm` (multilingual) when non-English content is detected.

**Why:** the EV battery cell PDF is German-original-translated; if the user uploads non-English docs, the current `en_core_web_sm` model misses noun phrases.

---

### 10. Telemetry on `repair_attempts` + gate failures

**Change:** log `OnboardingResponse.validation` and `repair_attempts` to a JSONL file per session. Over time, surface "which gate fails most often" and "which archetype needs prompt tightening."

---

## P3 — Backlog ideas

- **Re-onboarding diff command:** re-run agent against existing docs, surface diff vs. current schema. Catches drift.
- **Few-shot negative examples in `SYSTEM_PROMPT_DRAFT`** — one anti-pattern per archetype.
- **Cluster confidence in Stage A's UI:** show users which clusters the agent assigned vs. dropped, let them re-label.
- **Increase `MAX_DOC_CHARS_PER_SAMPLE` from 12k → 30k** when the model has the context window for it (reasoning models do).
- **Form-classifier tuning in advanced parser:** currently over-triggers on technical reports (66/216 chunks classified `form` for the EV docs — many are body text).

---

## Reference — files touched this session

- `core/onboarding_agent.py` — 4-stage pipeline, Gates 5-7, repair loop
- `core/vocab_miner.py` — NEW deterministic miner
- `requirements.txt` — `img2table>=2.0.0` pin
- `.env` — `USE_ADVANCED_PARSER=true` (user-side)

Existing flags still in play:
- `USE_ADVANCED_PARSER` (config flag) — needed for table content
- `MIN_VOCAB_GROUNDING_FRACTION = 0.40` (in `onboarding_agent.py`)
- `MAX_REPAIR_ATTEMPTS = 2`
- `TARGET_CLUSTER_COUNT = 50` (in `vocab_miner.py`)
- `CLUSTER_DISTANCE_THRESHOLD = 0.30`
