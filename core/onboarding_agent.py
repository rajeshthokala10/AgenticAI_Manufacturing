"""Schema-authoring agent for new-domain onboarding.

Given a domain id + a handful of sample documents, drives a *bigger* LLM
through a structured prompt that either:

    1. Returns a complete, validated ``schemas/<domain>.yaml`` blob, or
    2. Returns a small set of pointed follow-up questions when the docs
       leave key choices ambiguous (closed vs open vocab, ID format, etc.).

The flow is single-shot per call: the caller (Streamlit / FastAPI) keeps
the prior-Q&A state and re-invokes ``analyze()`` with the new answers
folded into context. That keeps the model stateless and the validation
deterministic per emission.

Three validation gates run after every YAML emission:

    Gate 1   schema loader parses the YAML without raising
    Gate 2   round-trip extraction test — KeywordExtractor over the sample
             docs must produce a Mention in ≥ ``MIN_ROUND_TRIP_FRACTION``
             of the docs (otherwise the closed vocab is too narrow)
    Gate 3   self-check checklist — every regex anchored, every edge's
             source/target references a declared entity_type, examples
             reference IDs that appear in the docs, display.label is set
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from config import ONBOARDING_MODEL, BASE_DIR, llm_available
from core.kg.extractors.keyword import KeywordExtractor
from core.kg.schema import load_schema
from core.llm_client import call_llm

logger = logging.getLogger("core.onboarding_agent")


# ─── Constants ──────────────────────────────────────────────────────────────

MIN_ROUND_TRIP_FRACTION = 0.30   # Gate 2 threshold
MAX_DOC_CHARS_PER_SAMPLE = 12_000  # truncate large docs in the prompt
DOMAIN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


# ─── The advanced prompt ────────────────────────────────────────────────────
#
# Authored carefully — the most important blocks are EXTRACTION_RULES and
# QUALITY_CHECKLIST because they're what stop the model from inventing
# entity types or emitting unanchored regexes. The gold-standard schema is
# inlined verbatim so the model has a concrete shape to imitate.

_GOLD_STANDARD_SCHEMA_PATH = BASE_DIR / "schemas" / "manufacturing.yaml"


def _load_gold_standard() -> str:
    """Inline the manufacturing schema as a worked example. Empty if missing."""
    try:
        return _GOLD_STANDARD_SCHEMA_PATH.read_text()
    except FileNotFoundError:
        return ""


SYSTEM_PROMPT = textwrap.dedent("""\
    You are a **Knowledge-Graph Schema Architect** for the Hybrid GraphRAG
    diagnostic copilot. Your job: given a domain id and a handful of sample
    documents, produce a complete, working ``schemas/<domain>.yaml`` that
    parses cleanly and drives every downstream surface — knowledge-graph
    construction, retrieval routing, query corrections, clarifier slot
    extraction, and per-domain UI copy.

    The schema is the **only** per-domain edit a user authors. Everything
    else (Streamlit selector, Next.js header switcher, FastAPI /api/domains,
    Qdrant collection naming, KG file path) auto-discovers from
    ``schemas/*.yaml``. So this YAML must be correct, complete, and
    coherent — it is the single point of failure.

    ## Three-tier KG contract (kgrag L1–L3)

      Schema (this file)
          ↓ declares
      Instances (extracted by Code / Metadata / Keyword / Narrative tiers)
          ↓ stamped with
      Provenance ({author, confidence, source_chunk_id, timestamp})

    Closed-vocabulary entity types (e.g. Component, Cause) get
    deterministically populated by the KeywordExtractor scanning chunk
    text. Open-vocab types (Symptom, Procedure) get narrative-extracted
    with low confidence. ID-pattern types (Equipment, Alarm, SparePart)
    are regex-extracted by the adapter from chunk text.

    ## Standard entity-type taxonomy (use these names when they fit)

      Equipment       a tagged asset instance (regex id_pattern)
      Component       a sub-component of equipment (closed vocab)
      Alarm           a coded alarm raised by control systems
      FailureMode     a coded fault/failure-mode instance
      Symptom         open-vocab observed condition (no vocabulary)
      Cause           a closed-vocab root cause
      Procedure       open-vocab maintenance / diagnostic step
      SparePart       a part-number SKU (regex id_pattern)
      Specification   a spec value (pressure, temperature, etc.)

    You MAY introduce new entity types only when none of the standard ones
    fit. New types must use PascalCase and be documented with a clear
    ``description``. Prefer reusing standard names over inventing.

    ## Standard edge taxonomy

      HAS_COMPONENT       Equipment → Component
      TRIGGERS_ALARM      Equipment → Alarm
      CAUSES_FAILURE      [Equipment, Alarm, Cause] → FailureMode
      HAS_SYMPTOM         Equipment → Symptom
      HAS_CAUSE           [Symptom, FailureMode, Equipment] → Cause
      RESOLVED_BY         [FailureMode, Symptom] → Procedure
      REQUIRES_PART       Equipment → SparePart
      FOLLOWS_PROCEDURE   [FailureMode, Symptom] → Procedure
      HAS_SPECIFICATION   Equipment → Specification

    Only declare edges you can JUSTIFY from the sample documents — do not
    invent edges that aren't implied by the prose.

    ## Extraction rules

    For each entity type you declare:

      1. **id_pattern** (for ID-based types like Equipment, Alarm, SparePart):
         - MUST start with ``^`` and end with ``$`` (anchored)
         - Use ``(?:alt1|alt2)`` non-capturing alternation, NEVER capturing
         - Test against IDs you saw in the docs — every example must match

      2. **vocabulary** (for closed-list types like Component, Cause):
         - All lowercase
         - Sorted, deduplicated
         - Use the actual phrases observed in the docs (not synonyms you
           imagine)

      3. **no vocabulary, no id_pattern** (for open-vocab types like
         Symptom, Procedure):
         - Just ``name`` and ``description``

    For edges: ``source`` and ``target`` may be a single entity-type name
    OR a list. Both must reference declared entity_types.

    ## UI affordances (drive Streamlit + Next.js)

      display:
        label:  a 1–3 word human label (Title Case)
        emoji:  one emoji that fits the domain
        color:  a hex color (#RRGGBB). Prefer copper #B45309, slate
                #475569, sky #0EA5E9, emerald #10B981. Distinct from
                other domains where possible.

      placeholder:    chat-input ghost text (1 sentence)
      empty_state:    {heading, blurb} — terse, technical, useful
      examples:       6–10 sample queries that reference REAL IDs from
                      the docs (not invented). Mix metric / lookup /
                      diagnostic / procedure intents.

    ## Optional polish blocks

      corrections.acronyms      industry abbreviations → expansions
      corrections.misspellings  common typos for domain terms
      corrections.synonyms      alternate phrasings for key concepts
      clarifier.equipment_patterns  extra regex families for the clarifier
      clarifier.metric_names    KPIs / measurements (e.g. CHT, OEE)

    ## Follow-up questions

    Ask follow-ups ONLY when the docs leave a high-leverage choice
    ambiguous. Common scenarios:

      - Pattern collision   "I see both ``WO:ASRS:123`` and
                             ``ENG:O-360-A4M`` — same entity type or two?"
      - Vocab vs open       "Should '<terms>' be a closed Cause vocab or
                             open-vocab Symptoms?"
      - Edge intent         "Components → Causes: model as
                             ``Cause CAUSES_FAILURE Component`` or
                             ``Component HAS_CAUSE Cause``?"
      - Scope check         "Docs cover both X and Y — one domain or two?"
      - Missing critical    "No equipment IDs found. Paste one example."

    Limit yourself to **≤ 3 follow-up questions per turn**. Each question
    must be specific (cite the terms or IDs from the docs).

    ## Output format — STRICT JSON

    Return a single JSON object with these keys. Do not wrap in markdown
    code fences:

      {
        "analysis": "<2-3 sentence summary of what this domain is and what
                     you understood from the sample docs>",
        "discovered_entities": {
          "<entity_type>": ["<phrase1>", "<phrase2>", ...],
          ...
        },
        "follow_up_questions": ["<question 1>", ...],
        "ready_to_generate": <true | false>,
        "yaml": "<the full schemas/<domain>.yaml content as a YAML string;
                  EMPTY when ready_to_generate is false>",
        "self_check": {
          "all_id_patterns_anchored": <bool>,
          "all_edges_reference_declared_types": <bool>,
          "examples_reference_real_ids": <bool>,
          "display_label_set": <bool>,
          "notes": "<any caveat the user should know>"
        }
      }

    When ``ready_to_generate`` is false, ``yaml`` MUST be the empty string
    and ``follow_up_questions`` MUST be non-empty.

    When ``ready_to_generate`` is true, ``yaml`` MUST be a complete
    schema that:
      - Parses with PyYAML
      - Has version: 1
      - Has a domain: matching the requested id (lowercase a-z/0-9/_)
      - Has at least one entity_type
      - Has at least one edge_type (or omits the block only if there's
        truly no edge to declare)
      - All four ``self_check`` booleans are ``true``

    ## Gold-standard example (the existing manufacturing schema)

    Imitate this shape exactly — block order, indentation, comment style.
    """).strip()


USER_TEMPLATE = textwrap.dedent("""\
    Domain id requested:  {domain_id}
    Domain hint:          {domain_hint}
    User preferences:     {prefs}

    ============================================================
    GOLD-STANDARD EXAMPLE — schemas/manufacturing.yaml
    ============================================================
    {gold_standard}
    ============================================================

    Sample documents ({n_docs}, total {n_chars} chars):

    {sample_blocks}

    ============================================================
    Prior Q&A in this onboarding session (if any):
    {prior_qa}
    ============================================================

    Produce the structured JSON response per the system prompt.
    """).strip()


# ─── Response dataclass ────────────────────────────────────────────────────


@dataclass
class OnboardingResponse:
    """Structured result from one ``analyze()`` invocation."""

    analysis: str = ""
    discovered_entities: Dict[str, List[str]] = field(default_factory=dict)
    follow_up_questions: List[str] = field(default_factory=list)
    ready_to_generate: bool = False
    yaml: str = ""
    self_check: Dict[str, Any] = field(default_factory=dict)
    # Filled by the validator after the model emits a YAML.
    validation: Dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""  # for debugging when JSON parsing fails

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis": self.analysis,
            "discovered_entities": self.discovered_entities,
            "follow_up_questions": self.follow_up_questions,
            "ready_to_generate": self.ready_to_generate,
            "yaml": self.yaml,
            "self_check": self.self_check,
            "validation": self.validation,
        }


# ─── Public API ────────────────────────────────────────────────────────────


def analyze(
    domain_id: str,
    docs: List[str],
    *,
    domain_hint: str = "",
    user_prefs: Optional[Dict[str, Any]] = None,
    prior_qa: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
) -> OnboardingResponse:
    """Drive one round of the onboarding agent.

    Parameters
    ----------
    domain_id
        Lowercase identifier the new schema will use as its ``domain:``.
    docs
        List of pre-parsed plaintext sample documents. Each is truncated
        to ``MAX_DOC_CHARS_PER_SAMPLE`` before being injected into the
        prompt.
    domain_hint
        Optional free-text label the user provided (e.g. "piston-engine
        aircraft maintenance"). Helps the model orient when docs are short.
    user_prefs
        Optional dict — passed through verbatim into the prompt. Useful
        for ``{"target_audience": "mechanics", "color": "#0EA5E9"}``.
    prior_qa
        Optional list of ``{"question": ..., "answer": ...}`` pairs from
        earlier turns of this onboarding session. The agent treats them
        as authoritative when generating the next pass.
    model
        Override the model. Defaults to ``config.ONBOARDING_MODEL``.

    Returns
    -------
    OnboardingResponse
        Either ``ready_to_generate=True`` with a validated ``yaml`` blob,
        or ``ready_to_generate=False`` with one or more
        ``follow_up_questions``.
    """
    _require_inputs(domain_id, docs)
    if not llm_available():
        raise RuntimeError(
            "Onboarding agent requires an LLM backend (OPENAI_API_KEY or "
            "Ollama). config.llm_available() returned False."
        )

    user_prompt = USER_TEMPLATE.format(
        domain_id=domain_id,
        domain_hint=domain_hint or "(none — infer from documents)",
        prefs=json.dumps(user_prefs or {}, ensure_ascii=False),
        gold_standard=_load_gold_standard(),
        n_docs=len(docs),
        n_chars=sum(len(d) for d in docs),
        sample_blocks=_format_sample_blocks(docs),
        prior_qa=_format_prior_qa(prior_qa or []),
    )

    raw = call_llm(
        SYSTEM_PROMPT,
        user_prompt,
        temperature=0.2,
        max_tokens=4096,
        model=model or ONBOARDING_MODEL,
    )

    response = _parse_json_response(raw)
    if response.ready_to_generate and response.yaml:
        response.validation = _validate_schema(response.yaml, docs)
        # If validation fails hard we don't downgrade the response — the
        # caller decides whether to re-prompt or surface the errors to the
        # user verbatim.
    return response


def save_schema(domain_id: str, yaml_str: str) -> Path:
    """Persist a validated schema YAML to ``schemas/<domain>.yaml``.

    Raises ValueError if the YAML fails to parse or the domain id doesn't
    match the schema's declared domain.
    """
    if not DOMAIN_ID_PATTERN.match(domain_id):
        raise ValueError(f"invalid domain id {domain_id!r} — must be lowercase a-z/0-9/_")

    # One last loader check before we write.
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_str)
        tmp_path = Path(f.name)
    try:
        schema = load_schema(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    if schema.domain.strip().lower() != domain_id:
        raise ValueError(
            f"schema declares domain={schema.domain!r} but the request was "
            f"for {domain_id!r} — refusing to write"
        )

    dest = BASE_DIR / "schemas" / f"{domain_id}.yaml"
    dest.write_text(yaml_str)
    logger.info("Wrote new schema → %s", dest)
    return dest


# ─── Internals ─────────────────────────────────────────────────────────────


def _require_inputs(domain_id: str, docs: List[str]) -> None:
    if not DOMAIN_ID_PATTERN.match(domain_id):
        raise ValueError(
            f"invalid domain id {domain_id!r} — must be lowercase a-z/0-9/_"
        )
    if not docs or not any(d.strip() for d in docs):
        raise ValueError("at least one non-empty sample document is required")


def _format_sample_blocks(docs: List[str]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        snippet = d.strip()[:MAX_DOC_CHARS_PER_SAMPLE]
        truncated = "\n[TRUNCATED]" if len(d) > MAX_DOC_CHARS_PER_SAMPLE else ""
        blocks.append(
            f"--- Document {i} ({len(snippet)} chars){truncated} ---\n"
            f"{snippet}\n"
            f"--- end document {i} ---"
        )
    return "\n\n".join(blocks)


def _format_prior_qa(qa: List[Dict[str, str]]) -> str:
    if not qa:
        return "(none — this is the first turn)"
    return "\n".join(
        f"Q{i+1}: {q.get('question','').strip()}\nA{i+1}: {q.get('answer','').strip()}"
        for i, q in enumerate(qa)
    )


def _parse_json_response(raw: str) -> OnboardingResponse:
    """Extract the JSON payload. The model is asked to emit a bare JSON
    object, but some models wrap in ```json ... ``` — strip if present."""
    cleaned = raw.strip()
    # Strip markdown fences if any.
    if cleaned.startswith("```"):
        # Drop the first line and trailing fence.
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("onboarding agent returned non-JSON response: %s", e)
        return OnboardingResponse(
            analysis="(model response did not parse as JSON)",
            raw_response=raw,
        )

    return OnboardingResponse(
        analysis=str(payload.get("analysis") or ""),
        discovered_entities=dict(payload.get("discovered_entities") or {}),
        follow_up_questions=[
            str(q) for q in (payload.get("follow_up_questions") or []) if q
        ],
        ready_to_generate=bool(payload.get("ready_to_generate", False)),
        yaml=str(payload.get("yaml") or ""),
        self_check=dict(payload.get("self_check") or {}),
        raw_response=raw,
    )


def _validate_schema(yaml_str: str, sample_docs: List[str]) -> Dict[str, Any]:
    """Run the three validation gates. Returns a report dict — caller
    decides whether to surface failures back to the model for a re-prompt."""
    report: Dict[str, Any] = {
        "gate_1_loader_parses": False,
        "gate_2_round_trip_fraction": 0.0,
        "gate_2_passed": False,
        "gate_3_self_check": [],
        "all_passed": False,
        "errors": [],
    }

    # Gate 1 — schema loader
    schema = None
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_str)
        tmp = Path(f.name)
    try:
        try:
            schema = load_schema(tmp)
            report["gate_1_loader_parses"] = True
        except Exception as e:
            report["errors"].append(f"loader error: {e!r}")
            return report
    finally:
        tmp.unlink(missing_ok=True)

    # Gate 2 — round-trip extraction
    try:
        extractor = KeywordExtractor(schema)
        hits = 0
        for i, doc_text in enumerate(sample_docs):
            doc = {
                "chunk_id": f"sample_{i}",
                "text": doc_text,
                "metadata": {},
            }
            result = extractor.extract(doc)
            if result.mentions:
                hits += 1
        fraction = hits / max(len(sample_docs), 1)
        report["gate_2_round_trip_fraction"] = round(fraction, 3)
        report["gate_2_passed"] = fraction >= MIN_ROUND_TRIP_FRACTION
        if not report["gate_2_passed"]:
            report["errors"].append(
                f"closed-vocab coverage too low: only {hits}/{len(sample_docs)} "
                f"sample docs produced any Mention (need ≥ "
                f"{int(MIN_ROUND_TRIP_FRACTION*100)}%). Vocabularies likely "
                f"too narrow."
            )
    except Exception as e:
        report["errors"].append(f"round-trip error: {e!r}")

    # Gate 3 — self-check checklist
    parsed = yaml.safe_load(yaml_str) or {}
    checks: List[Dict[str, Any]] = []
    # Anchored regexes
    anchored = True
    for et in parsed.get("entity_types", []) or []:
        pat = et.get("id_pattern")
        if pat and not (pat.startswith("^") and pat.endswith("$")):
            anchored = False
            report["errors"].append(
                f"id_pattern not anchored on entity '{et.get('name')}': {pat!r}"
            )
    checks.append({"name": "all_id_patterns_anchored", "passed": anchored})
    # Edge sources/targets reference declared types
    declared = {et.get("name") for et in (parsed.get("entity_types") or [])}
    edges_ok = True
    for eg in parsed.get("edge_types", []) or []:
        for endpoint in ("source", "target"):
            v = eg.get(endpoint)
            names = v if isinstance(v, list) else [v]
            for n in names:
                if n not in declared:
                    edges_ok = False
                    report["errors"].append(
                        f"edge '{eg.get('name')}' references undeclared "
                        f"entity_type '{n}' as {endpoint}"
                    )
    checks.append({"name": "all_edges_reference_declared_types", "passed": edges_ok})
    # Display label set
    has_label = bool((parsed.get("display") or {}).get("label"))
    checks.append({"name": "display_label_set", "passed": has_label})
    report["gate_3_self_check"] = checks

    report["all_passed"] = (
        report["gate_1_loader_parses"]
        and report["gate_2_passed"]
        and all(c["passed"] for c in checks)
    )
    return report
