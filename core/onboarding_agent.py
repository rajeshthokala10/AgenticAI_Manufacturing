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

from config import BASE_DIR, llm_available
from core.kg.extractors.keyword import KeywordExtractor
from core.kg.schema import load_schema
from core.llm_client import call_llm
from core.llm_router import task_model
from core.schema_validator import validate_schema as validate_new_blocks

logger = logging.getLogger("core.onboarding_agent")


# ─── Constants ──────────────────────────────────────────────────────────────

MIN_ROUND_TRIP_FRACTION = 0.30   # Gate 2 threshold
MIN_VOCAB_GROUNDING_FRACTION = 0.40  # Gate 5: closed-vocab coverage in corpus
MAX_DOC_CHARS_PER_SAMPLE = 12_000  # truncate large docs in the prompt
MAX_REPAIR_ATTEMPTS = 2
DOMAIN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


# ─── Two-stage pipeline ─────────────────────────────────────────────────────
#
# Single-shot drafting + a 300-line inlined gold-standard caused the agent
# to mechanically copy ``schemas/manufacturing.yaml`` regardless of corpus
# shape — the EV-manufacturing trial shipped fabricated equipment IDs and
# a Cause vocabulary lifted verbatim from the maintenance domain.
#
# The agent now runs:
#
#   Stage A   _characterize_corpus()  →  archetype + candidate terms with
#                                         evidence quotes from the docs
#   Stage B   _draft_schema()         →  YAML adapted to the archetype, with
#                                         only grounded vocab/IDs
#   Stage C   _validate_schema()      →  Gates 1-7, including programmatic
#                                         grounding checks against the corpus
#   Stage D   _repair_schema()        →  Re-prompt with the specific gate
#                                         errors when validation fails
#                                         (≤ MAX_REPAIR_ATTEMPTS)
#
# Stage A keeps Stage B from defaulting to the manufacturing skeleton when
# the corpus is, say, a clinical reference or a process tutorial.

_GOLD_STANDARD_SCHEMA_PATH = BASE_DIR / "schemas" / "manufacturing.yaml"


def _load_gold_standard() -> str:
    """Inline the manufacturing schema. Used as a *reference* (not a
    template to imitate) and ONLY when archetype == maintenance_manual."""
    try:
        return _GOLD_STANDARD_SCHEMA_PATH.read_text()
    except FileNotFoundError:
        return ""


# Archetype labels Stage A can emit. Stage B uses these to pick a skeleton
# and to bias the entity-type taxonomy. ``general_technical`` is the safe
# fallback when none of the others clearly fit.
ARCHETYPES = (
    "maintenance_manual",      # asset-centric: equipment + alarms + spare parts
    "process_tutorial",        # workflow-centric: steps + materials + defects
    "regulatory_compliance",   # rule-centric: requirements + obligations + audits
    "clinical_reference",      # patient-centric: conditions + treatments + dosages
    "financial_report",        # metric-centric: KPIs + accounts + periods
    "legal_document",          # clause-centric: parties + obligations + jurisdictions
    "design_specification",    # spec-centric: components + requirements + tolerances
    "general_technical",       # fallback when none of the above clearly fit
)

# Compressed skeletons (block names + intent, NOT full YAML) — short enough
# that the model isn't tempted to copy them verbatim. The drafter receives
# ONE skeleton, picked by Stage A's archetype label.
ARCHETYPE_SKELETONS: Dict[str, str] = {
    "maintenance_manual": textwrap.dedent("""\
        Typical entity types: Equipment (id_pattern), Component (closed vocab),
        Alarm (id_pattern), FailureMode (id_pattern), Symptom (open),
        Cause (closed vocab), Procedure (open), SparePart (id_pattern),
        Specification (open).
        Typical edges: HAS_COMPONENT, TRIGGERS_ALARM, CAUSES_FAILURE,
        HAS_SYMPTOM, HAS_CAUSE, RESOLVED_BY, REQUIRES_PART.
        Typical intents: troubleshoot, procedure, specification, alarm, inventory.
        Safety hazards usually include LOTO, arc flash, confined space.
        """).strip(),

    "process_tutorial": textwrap.dedent("""\
        Typical entity types: ProcessStep (closed vocab of named stages),
        Material (closed vocab of inputs), Equipment (process equipment,
        often no asset-ID), Defect (closed vocab of quality failures),
        Parameter (open — viscosity, peel strength, etc.), Procedure (open).
        Typical edges: STEP_FOLLOWS (ProcessStep → ProcessStep),
        USES_MATERIAL (ProcessStep → Material), CAUSES_DEFECT
        (ProcessStep → Defect), MEASURES_PARAMETER (ProcessStep → Parameter).
        Typical intents: process_lookup, defect_diagnosis, parameter_lookup,
        material_specification, general.
        DO NOT declare Equipment with an asset-tag id_pattern unless the
        docs actually contain plant-asset IDs — process tutorials usually
        don't. DO NOT copy maintenance-domain Cause vocabularies (bearing
        wear, cavitation, etc.) — they will not match the corpus.
        """).strip(),

    "regulatory_compliance": textwrap.dedent("""\
        Typical entity types: Regulation (id_pattern for clause refs like
        "21 CFR 820.30"), Requirement (open), Obligation (open), Party
        (closed vocab of regulated entities), Audit (open), Penalty (open).
        Typical edges: REQUIRES, APPLIES_TO, ENFORCED_BY, VIOLATES, AUDITED_BY.
        DO NOT carry Equipment / Alarm / SparePart from the maintenance template.
        """).strip(),

    "clinical_reference": textwrap.dedent("""\
        Typical entity types: Condition (closed vocab of disease/syndrome
        names), Symptom (open), Treatment (open), Medication (closed vocab),
        Dosage (open with units), Contraindication (open), Procedure (open).
        Typical edges: PRESENTS_WITH, TREATED_BY, CONTRAINDICATED_WITH,
        DOSED_AT, FOLLOWS_PROCEDURE.
        Safety hazards: anaphylaxis, overdose, drug-interaction, sepsis.
        DO NOT carry Equipment / Alarm / SparePart from the maintenance template.
        """).strip(),

    "financial_report": textwrap.dedent("""\
        Typical entity types: Metric (closed vocab — revenue, EBITDA, OEE,
        churn, etc.), Account (id_pattern for ledger codes), Period
        (id_pattern like FY2025Q1), Segment (closed vocab), Trend (open).
        Typical edges: REPORTED_BY, OVER_PERIOD, BELONGS_TO_SEGMENT,
        COMPARED_TO.
        Typical intents: lookup, trend, comparison, period_over_period.
        DO NOT carry safety keywords (LOTO, arc flash) — irrelevant.
        """).strip(),

    "legal_document": textwrap.dedent("""\
        Typical entity types: Clause (id_pattern), Party (closed vocab),
        Obligation (open), Jurisdiction (closed vocab), Definition (open),
        Remedy (open).
        Typical edges: BINDS, DEFINES, GOVERNED_BY, AMENDS, VIOLATES.
        DO NOT carry Equipment / Alarm / Cause from the maintenance template.
        """).strip(),

    "design_specification": textwrap.dedent("""\
        Typical entity types: Component (closed vocab), Requirement (open
        or id_pattern for REQ-### style refs), Tolerance (open with units),
        Interface (open), Standard (id_pattern for ISO/IEEE refs).
        Typical edges: HAS_COMPONENT, SATISFIES_REQUIREMENT, CONFORMS_TO,
        INTERFACES_WITH.
        Use Procedure only if the docs contain step-by-step instructions.
        """).strip(),

    "general_technical": textwrap.dedent("""\
        No fixed skeleton — propose entity types that match the actual
        nouns and id-shaped tokens in the corpus. AVOID adopting the
        maintenance template (Equipment / Symptom / Cause / RESOLVED_BY)
        unless the docs are obviously about maintenance / troubleshooting.
        """).strip(),
}


# ─── Stage A — corpus characterizer ─────────────────────────────────────────

SYSTEM_PROMPT_CHARACTERIZE = textwrap.dedent("""\
    You are a **Corpus Analyst** preparing a knowledge-graph schema brief.

    Given a handful of sample documents from a single domain — plus a
    pre-mined list of candidate noun-phrase clusters extracted
    deterministically from the FULL corpus — your job is to:

      (1) determine WHAT KIND of corpus this is (the archetype), and
      (2) LABEL each mined cluster: decide if it is signal or noise and,
          if signal, which entity type it belongs to.

    You are NOT extracting vocabulary from raw text; the deterministic
    miner has already done that. You are also NOT writing a schema; a
    downstream agent will draft it using your brief.

    ## Archetypes (pick exactly one)

      maintenance_manual       asset-centric — plant-floor equipment,
                                troubleshooting, alarms, spare parts
      process_tutorial         workflow-centric — how something is
                                manufactured / produced step by step
      regulatory_compliance    rule-centric — regulations, requirements,
                                audits, obligations, penalties
      clinical_reference       patient-centric — conditions, symptoms,
                                treatments, medications, dosages
      financial_report         metric-centric — KPIs, accounts, periods,
                                segments
      legal_document           clause-centric — parties, clauses,
                                obligations, jurisdictions
      design_specification     spec-centric — components, requirements,
                                tolerances, standards
      general_technical        fallback when none of the above clearly fit

    Bias rule: when in doubt between maintenance_manual and process_tutorial,
    pick process_tutorial — process tutorials look superficially like
    maintenance manuals (they mention equipment) but the focus is on HOW
    a product is built, not on diagnosing or fixing assets. Maintenance
    manuals contain asset tags, alarm codes, fault codes, work orders;
    process tutorials contain step sequences, materials, parameters, defects.

    ## Grounding discipline

    Every term you assign to ``candidate_vocabulary`` MUST be a cluster
    label or a member from the pre-mined ``candidate_clusters`` list in
    the user message. You may DROP clusters you judge to be noise
    (table-header residue, author names, generic connectors) but you
    may NOT ADD terms not in the list.

    For each cluster the miner provides:

      - ``label``         the representative phrase
      - ``members``       near-synonyms (singular/plural, hyphen variants)
      - ``frequency``     total mention count across the corpus
      - ``doc_coverage``  how many docs mention it (higher = stronger signal)
      - ``sample_quote``  one sentence from the docs showing usage

    Heuristics for labelling:

      - High doc_coverage (≥ all docs) AND high frequency → almost
        always signal; assign to an entity type.
      - Single-doc clusters with low frequency → usually noise unless
        the surface form is clearly domain-specific.
      - Phrases like "form data", "table on", "excerpt", "quality
        impacts" → noise; the miner couldn't strip every table-header
        artefact.
      - Author / affiliation phrases → noise.
      - Phrases that are too generic on their own ("material", "process",
        "quality", "design") → either drop, or only keep if they're
        modified ("temperature profile", "process step", "winding scheme").

    ## Coverage target — IMPORTANT

    Aim to assign **at least 5–10 clusters per entity type** when the
    miner provides them. Conservative pruning (only labelling the top 2
    obvious ones) is the WRONG default — vocabulary breadth is what
    drives downstream extraction recall. If you can plausibly assign a
    cluster to an entity type, DO assign it. The downstream gates and
    HITL review will catch borderline calls. Err on inclusion.

    A good Stage A response uses **most** of the high-doc-coverage
    clusters (≥ 2 docs) and only drops the obvious table-residue or
    author boilerplate. A bad Stage A response cherry-picks 2 terms
    per type and discards everything else.

    ## Output — STRICT JSON ONLY, no markdown fences

    {
      "archetype": "<one of the labels above>",
      "archetype_confidence": <float 0.0-1.0>,
      "archetype_rationale": "<1-2 sentence justification citing terms
                              from the docs>",
      "summary": "<2-3 sentence corpus description>",
      "primary_actors": ["<noun>", ...],
      "primary_workflows": ["<verb-phrase>", ...],
      "recommended_entity_types": ["<PascalCase>", ...],
      "candidate_vocabulary": {
        "<EntityType>": [
          {"term": "<lowercase phrase>",
           "evidence_quote": "<verbatim from doc>",
           "doc_index": <0-based int>}
        ]
      },
      "representative_ids": [
        {"id": "<verbatim id>",
         "evidence_quote": "<verbatim from doc>",
         "doc_index": <int>,
         "suggested_id_pattern": "<anchored regex>"}
      ],
      "domain_hazards": ["<short safety phrase>", ...],
      "domain_intents": ["<intent label>", ...]
    }

    representative_ids should be EMPTY when the corpus contains no
    asset-tag-style identifiers (common for process tutorials, regulatory
    docs, clinical references). DO NOT invent placeholder IDs.
    """).strip()


# ─── Stage B — schema drafter ───────────────────────────────────────────────

SYSTEM_PROMPT_DRAFT = textwrap.dedent("""\
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

    ## Reference taxonomy (maintenance-domain — use ONLY when archetype
    ## is ``maintenance_manual``; otherwise use the archetype skeleton)

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

    ## Per-domain runtime overrides (HIGH IMPACT — strongly preferred)

    A schema that omits these blocks falls back to the manufacturing
    defaults. That is benign for manufacturing-adjacent domains but
    nonsensical for medical / legal / aviation / financial domains where
    the manufacturing persona, HITL keywords, and intent vocabulary
    would confabulate. AUTHOR these unless the new domain is itself
    plant-floor / industrial.

    prompts:
      persona:           one-line label (e.g. "aviation maintenance copilot")
      answer_system:     system prompt for the free-form answer LLM
      retry_system:      MUST contain the literal "{critic_feedback}"
                         placeholder (it's substituted at runtime)
      critic_rules:      criteria the quality critic uses to grade answers
      procedure_system:  drafter persona — name the safety preconditions
                         appropriate to the domain (e.g. LOTO for plants,
                         mag-ground/prop-clear for aviation, sterile-field
                         for medical)
      cause_rank_system: MUST contain "{top_k}" and "{taxonomy_clause}"
                         placeholders
      classify_system:   intent classifier prompt — list the categories
                         relevant to the domain
      risk_grader_system / risk_grader_user: tier-2 HITL risk grader

    safety:
      high_risk_keywords:  the words/phrases that escalate a query or
                           answer to a human supervisor. STRONGLY drop
                           manufacturing-specific terms (LOTO, H2S, arc
                           flash) if they don't apply, and add the
                           domain's real hazards (e.g. mayday / engine
                           fire / airworthiness / fatal dose / patient
                           safety event / sanctions violation). An
                           explicit empty list is a valid opt-out.

    clarifier.intent_patterns:
      List of {intent, patterns, boost} entries that layer on top of
      the manufacturing-default regexes. Each ``intent`` MUST be one of
      the canonical names: LOOKUP / COMPARISON / TROUBLESHOOTING /
      COMPLIANCE / METRIC_QUERY / PROCEDURE / TREND / STATUS /
      ROOT_CAUSE / UNKNOWN. ``patterns`` is a non-empty list of regex
      strings. ``boost`` is the confidence (0.0–0.99) when any pattern
      hits; default 0.85.

    clarifier.slot_templates:
      Per-intent slot replacements so the clarifier asks domain-natural
      follow-ups. Example: aviation TROUBLESHOOTING uses
      "Which engine or aircraft?" not "Which CNC line?". Each slot is
      {name, entity_types, required, prompt}.

    procedure:
      enabled: <true|false>      true → run the structured procedure
                                  drafter on trigger intents. false →
                                  skip entirely (correct for purely
                                  expository domains: legal lookup,
                                  market research, medical reference).
      trigger_intents: [list]    substring matchers against the
                                  classified intent. Use prefix forms
                                  like "diagnos" to catch
                                  diagnose/diagnosis/diagnostic.

    Validation is STRICT on all four blocks above — typos like
    ``saftey`` or ``high_risk_keyword`` (singular) are rejected, and
    YAML strings like ``enabled: "false"`` (quoted) raise. Always use
    literal ``true`` / ``false`` for booleans.

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

    ## Archetype-driven authoring (CRITICAL)

    The corpus has been pre-analysed by an upstream agent. You will
    receive an ``archetype`` label, a per-archetype skeleton, and a
    grounded list of candidate vocabulary terms WITH evidence quotes.

    Authoring rules:

      1. The archetype's skeleton is the STARTING POINT for your entity
         and edge taxonomy. DO NOT carry entity types from a different
         archetype's template. In particular: if archetype is NOT
         ``maintenance_manual``, do NOT default to Equipment/Symptom/
         Cause/SparePart/Alarm/FailureMode — that is the maintenance
         template, not a universal default.

      2. Every closed-vocabulary term you emit MUST appear in the
         candidate_vocabulary list provided to you. You may DROP terms
         (if they don't fit) but you may NOT ADD terms not in the list.
         If you think a term is missing, leave a note in ``analysis``
         instead of inventing it.

      3. Every example query you emit MUST reference either (a) a
         representative_id provided to you, or (b) a generic noun
         phrase from the corpus — NEVER a fabricated ID like
         ``EV-MTR-001`` that doesn't appear in the docs.

      4. Every id_pattern you declare MUST be derived from a real ID
         pattern in representative_ids. If representative_ids is empty,
         DO NOT declare any entity type with an id_pattern.

      5. The persona, safety.high_risk_keywords, classify_system
         intents, and procedure_system safety preconditions MUST match
         the archetype. Don't ship LOTO / arc-flash keywords for a
         clinical or financial domain. Don't ship "diagnose the
         alarm" intents for a process_tutorial.

    Reference: when archetype == ``maintenance_manual`` you may also
    consult the ``manufacturing.yaml`` reference block at the bottom of
    the user message. For other archetypes that block will not appear —
    use the skeleton and the candidate_vocabulary as your only guide.
    """).strip()

# Backwards-compat alias — older callers and tests still import SYSTEM_PROMPT.
SYSTEM_PROMPT = SYSTEM_PROMPT_DRAFT


USER_TEMPLATE = textwrap.dedent("""\
    Domain id requested:  {domain_id}
    Domain hint:          {domain_hint}
    User preferences:     {prefs}

    ============================================================
    STAGE A — corpus characterization (use these as ground truth)
    ============================================================
    {characterization_block}
    ============================================================

    ============================================================
    Archetype skeleton — {archetype}
    ============================================================
    {archetype_skeleton}
    ============================================================

    Sample documents ({n_docs}, total {n_chars} chars):

    {sample_blocks}

    ============================================================
    Prior Q&A in this onboarding session (if any):
    {prior_qa}
    ============================================================
    {gold_standard_block}
    Produce the structured JSON response per the system prompt.
    """).strip()


# Re-prompt used by the repair loop when validation gates fail.
SYSTEM_PROMPT_REPAIR = textwrap.dedent("""\
    You previously authored a YAML schema for the Hybrid GraphRAG copilot.
    Programmatic validation gates flagged the issues listed in the user
    message. Your job: emit a CORRECTED schema that fixes every flagged
    issue, while keeping unrelated blocks identical.

    Hard rules:

      1. Do not invent vocabulary or example IDs that are not in the
         candidate lists. If a term has no grounding, drop it.
      2. Anchor every id_pattern with ^...$ and verify it matches at
         least one representative_id provided to you.
      3. Keep the {critic_feedback} placeholder in retry_system and the
         {top_k} + {taxonomy_clause} placeholders in cause_rank_system.
      4. Do not change ``domain:`` or ``version:`` unless that is the
         specific gate failure.

    Output STRICT JSON ONLY, no markdown fences:

      {
        "yaml": "<the full corrected schema YAML>",
        "fix_summary": "<one short paragraph: what you changed and why>"
      }
    """).strip()


# ─── Response dataclass ────────────────────────────────────────────────────


@dataclass
class CorpusCharacterization:
    """Output of Stage A — feeds Stage B as ground truth."""

    archetype: str = "general_technical"
    archetype_confidence: float = 0.0
    archetype_rationale: str = ""
    summary: str = ""
    primary_actors: List[str] = field(default_factory=list)
    primary_workflows: List[str] = field(default_factory=list)
    recommended_entity_types: List[str] = field(default_factory=list)
    candidate_vocabulary: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    representative_ids: List[Dict[str, Any]] = field(default_factory=list)
    domain_hazards: List[str] = field(default_factory=list)
    domain_intents: List[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "archetype": self.archetype,
            "archetype_confidence": self.archetype_confidence,
            "archetype_rationale": self.archetype_rationale,
            "summary": self.summary,
            "primary_actors": self.primary_actors,
            "primary_workflows": self.primary_workflows,
            "recommended_entity_types": self.recommended_entity_types,
            "candidate_vocabulary": self.candidate_vocabulary,
            "representative_ids": self.representative_ids,
            "domain_hazards": self.domain_hazards,
            "domain_intents": self.domain_intents,
        }


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
    # Filled by Stage A; reported back so callers (and tests) can inspect.
    corpus_characterization: Optional[CorpusCharacterization] = None
    # How many repair attempts the loop made, for telemetry.
    repair_attempts: int = 0
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
            "corpus_characterization": (
                self.corpus_characterization.to_dict()
                if self.corpus_characterization else None
            ),
            "repair_attempts": self.repair_attempts,
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
    force_generate: bool = False,
    corpus_characterization: Optional[CorpusCharacterization] = None,
) -> OnboardingResponse:
    """Drive one round of the onboarding agent.

    Pipeline: Stage A (corpus characterization) → Stage B (schema draft)
    → Stage C (validation gates) → Stage D (repair loop, ≤
    ``MAX_REPAIR_ATTEMPTS`` retries).

    Stage A is run lazily on the first call of a multi-turn session;
    callers can pass ``corpus_characterization=`` on subsequent turns to
    skip the re-analysis.
    """
    _require_inputs(domain_id, docs)
    if not llm_available():
        raise RuntimeError(
            "Onboarding agent requires an LLM backend (OPENAI_API_KEY or "
            "Ollama). config.llm_available() returned False."
        )

    resolved_model = model or task_model("onboarding")

    # ── Stage A — corpus characterization (cached across turns) ─────
    if corpus_characterization is None:
        try:
            corpus_characterization = _characterize_corpus(
                docs, domain_hint=domain_hint, model=resolved_model,
            )
        except Exception as e:
            logger.warning("Stage A characterization failed: %r — falling back to general_technical", e)
            corpus_characterization = CorpusCharacterization(
                archetype="general_technical",
                archetype_rationale=f"fallback: {e!r}",
            )

    # ── Stage B — schema drafter ───────────────────────────────────
    user_prompt = _build_drafter_prompt(
        domain_id=domain_id,
        domain_hint=domain_hint,
        user_prefs=user_prefs,
        prior_qa=prior_qa,
        docs=docs,
        characterization=corpus_characterization,
        force_generate=force_generate,
    )

    raw = call_llm(
        SYSTEM_PROMPT_DRAFT,
        user_prompt,
        temperature=0.2,
        max_tokens=4096,
        model=resolved_model,
    )

    response = _parse_json_response(raw)
    response.corpus_characterization = corpus_characterization

    if response.ready_to_generate and response.yaml:
        # ── Stage C — validation (loader + grounding gates) ────────
        report = _validate_schema(response.yaml, docs, corpus_characterization)
        response.validation = report

        # ── Stage D — repair loop ──────────────────────────────────
        attempts = 0
        while not report.get("all_passed") and attempts < MAX_REPAIR_ATTEMPTS:
            attempts += 1
            logger.info(
                "Repair attempt %d/%d for domain %s — errors: %s",
                attempts, MAX_REPAIR_ATTEMPTS, domain_id, report.get("errors"),
            )
            repaired_yaml = _repair_schema(
                yaml_str=response.yaml,
                report=report,
                characterization=corpus_characterization,
                docs=docs,
                domain_id=domain_id,
                model=resolved_model,
            )
            if not repaired_yaml or repaired_yaml.strip() == response.yaml.strip():
                # Model couldn't improve; bail.
                break
            response.yaml = repaired_yaml
            report = _validate_schema(repaired_yaml, docs, corpus_characterization)
            response.validation = report
        response.repair_attempts = attempts

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


def _format_clusters_block(mining: Optional[Any]) -> str:
    """Render the vocab miner's clusters into a compact LLM-friendly
    table. One cluster per line: label, doc-coverage, frequency, sample
    quote (truncated). Stage A reads this and assigns each cluster to an
    entity_type or drops it as noise."""
    if mining is None or not getattr(mining, "clusters", None):
        return "(miner unavailable or produced 0 clusters — extract directly from the sample docs above)"
    lines = [
        f"{'#':>3}  {'label':<35} cov/freq  members → sample_quote",
        "    " + "-" * 100,
    ]
    for i, c in enumerate(mining.clusters, 1):
        members_str = ", ".join(c.members[:3])
        if len(c.members) > 3:
            members_str += f" (+{len(c.members) - 3})"
        quote = c.sample_quote.replace("\n", " ")[:120]
        lines.append(
            f"{i:>3}. {c.label[:35]:<35} {c.doc_coverage}d/{c.frequency:<3}f "
            f"[{members_str}] → \"{quote}\""
        )
    return "\n".join(lines)


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


def _validate_schema(
    yaml_str: str,
    sample_docs: List[str],
    characterization: Optional[CorpusCharacterization] = None,
) -> Dict[str, Any]:
    """Run validation gates 1-7. Returns a report dict — caller decides
    whether to surface failures back to the model for a re-prompt.

    Gates 5/6/7 are corpus-grounding gates: they require the vocabulary,
    example IDs, and id_patterns the model emits to actually appear in
    the sample documents. They catch the most common Stage B failure mode
    — copy-pasting vocabulary or fabricating IDs that don't exist."""
    report: Dict[str, Any] = {
        "gate_1_loader_parses": False,
        "gate_2_round_trip_fraction": 0.0,
        "gate_2_passed": False,
        "gate_3_self_check": [],
        "gate_5_vocab_grounding": [],
        "gate_5_passed": True,
        "gate_6_example_ids": [],
        "gate_6_passed": True,
        "gate_7_id_pattern_matches": [],
        "gate_7_passed": True,
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

    # Gate 4 — strict pydantic validation of the four NEW blocks
    # (prompts / safety / clarifier / procedure). Catches typos like
    # ``saftey:`` / ``high_risk_keyword:``, string-bool footguns like
    # ``enabled: "false"``, and bad {placeholders} in retry_system /
    # cause_rank_system that would explode at .format() time. Warnings
    # don't fail the gate; errors do.
    domain_id = str(parsed.get("domain") or "(unknown)").strip().lower()
    new_block_result = validate_new_blocks(domain_id, parsed)
    report["gate_4_new_blocks"] = {
        "ok": new_block_result.ok,
        "errors": list(new_block_result.errors),
        "warnings": list(new_block_result.warnings),
    }
    if not new_block_result.ok:
        report["errors"].extend(
            f"new-blocks: {e}" for e in new_block_result.errors
        )

    # ── Gates 5, 6, 7 — corpus grounding ─────────────────────────────
    # These prevent the most common Stage B drift: copy-pasted vocab,
    # fabricated example IDs, id_patterns that match nothing in the docs.
    corpus_text = " \n ".join(sample_docs).lower()

    # Gate 5 — closed-vocab term grounding
    vocab_report: List[Dict[str, Any]] = []
    for et in parsed.get("entity_types", []) or []:
        vocab = et.get("vocabulary") or []
        if not vocab:
            continue
        present = [t for t in vocab if str(t).lower() in corpus_text]
        absent = [t for t in vocab if str(t).lower() not in corpus_text]
        frac = len(present) / max(len(vocab), 1)
        passed = frac >= MIN_VOCAB_GROUNDING_FRACTION
        vocab_report.append({
            "entity_type": et.get("name"),
            "fraction_present": round(frac, 3),
            "absent": absent,
            "passed": passed,
        })
        if not passed:
            report["gate_5_passed"] = False
            report["errors"].append(
                f"vocab grounding: entity_type '{et.get('name')}' has "
                f"{len(present)}/{len(vocab)} terms in the corpus "
                f"(need ≥ {int(MIN_VOCAB_GROUNDING_FRACTION*100)}%). "
                f"Absent terms: {absent[:8]}. "
                f"These were likely copied from a template — drop them or "
                f"replace with terms actually present in the docs."
            )
    report["gate_5_vocab_grounding"] = vocab_report

    # Gate 6 — example queries must reference real IDs (when they look
    # like IDs). We extract ID-shaped tokens (anything that matches a
    # declared id_pattern) and check each one is present in the corpus.
    # Strip ^...$ anchors so the pattern can match as a substring inside
    # example queries and inside doc text. Authors are required to anchor
    # the YAML pattern (Gate 3 checks), but for corpus grounding we need
    # to search anywhere in the text.
    def _unanchor(p: str) -> str:
        s = p
        if s.startswith("^"):
            s = s[1:]
        if s.endswith("$"):
            s = s[:-1]
        return s

    declared_patterns = []
    for et in parsed.get("entity_types", []) or []:
        pat = et.get("id_pattern")
        if pat:
            try:
                declared_patterns.append(
                    (et.get("name"), re.compile(_unanchor(pat), re.IGNORECASE))
                )
            except re.error:
                pass

    examples_report: List[Dict[str, Any]] = []
    for ex in parsed.get("examples", []) or []:
        ex_str = str(ex)
        for et_name, pat in declared_patterns:
            for match in pat.findall(ex_str):
                found = match if isinstance(match, str) else "".join(match)
                if not found:
                    continue
                in_corpus = found.lower() in corpus_text
                examples_report.append({
                    "example": ex_str,
                    "id": found,
                    "entity_type": et_name,
                    "in_corpus": in_corpus,
                })
                if not in_corpus:
                    report["gate_6_passed"] = False
                    report["errors"].append(
                        f"example id grounding: example query {ex_str!r} "
                        f"references id {found!r} which does not appear in "
                        f"the corpus. Replace with a real id from the docs "
                        f"or rephrase without an id."
                    )
    report["gate_6_example_ids"] = examples_report

    # Gate 7 — every declared id_pattern must match at least one substring
    # in the corpus. A pattern that matches nothing is dead weight and a
    # strong signal the model invented an asset-tag convention that
    # doesn't exist in this corpus.
    id_pat_report: List[Dict[str, Any]] = []
    for et_name, pat in declared_patterns:
        # findall over corpus text — use the original case-insensitive
        # compiled pattern. We search a window of each doc with word
        # boundaries to keep this fast.
        matches: List[str] = []
        for doc_text in sample_docs:
            for m in pat.finditer(doc_text):
                matches.append(m.group(0))
                if len(matches) >= 3:
                    break
            if len(matches) >= 3:
                break
        id_pat_report.append({
            "entity_type": et_name,
            "match_count": len(matches),
            "sample_matches": matches[:3],
        })
        if not matches:
            report["gate_7_passed"] = False
            report["errors"].append(
                f"id_pattern grounding: entity_type '{et_name}' declares "
                f"an id_pattern but it matches 0 strings in the corpus. "
                f"Either drop the id_pattern, drop the entity_type, or "
                f"rewrite the pattern to match real ids in the docs."
            )
    report["gate_7_id_pattern_matches"] = id_pat_report

    report["all_passed"] = (
        report["gate_1_loader_parses"]
        and report["gate_2_passed"]
        and all(c["passed"] for c in checks)
        and new_block_result.ok
        and report["gate_5_passed"]
        and report["gate_6_passed"]
        and report["gate_7_passed"]
    )
    return report


# ─── Stage A — corpus characterizer ─────────────────────────────────────────


def _characterize_corpus(
    docs: List[str],
    *,
    domain_hint: str = "",
    model: Optional[str] = None,
) -> CorpusCharacterization:
    """Stage A. Deterministic miner runs first over the full corpus,
    then a focused LLM call (a) picks the archetype and (b) labels each
    mined cluster with an entity type (or drops it as noise).

    Decoupling extraction (deterministic) from labelling (LLM) is what
    fixes the 'gpt-4o only emits 2 vocab terms per type' bottleneck —
    the miner produces ~40 candidate clusters from the full doc, and
    the LLM only has to assign them.
    """
    # Mine candidate vocabulary clusters from the full corpus. Truncation
    # affects only the LLM input, not the miner — so the miner sees 100%
    # of the parsed text while the LLM sees a small clusters summary.
    try:
        from core.vocab_miner import mine_candidates  # noqa: PLC0415
        mining = mine_candidates(docs, target_clusters=40)
        logger.info(
            "Vocab miner: %d clusters from %d docs (%d phrases kept, embeddings=%s)",
            mining.cluster_count, mining.docs_seen,
            mining.kept_phrase_count, mining.used_embeddings,
        )
    except Exception as exc:
        logger.warning("Vocab miner failed: %r — Stage A falls back to LLM extraction", exc)
        mining = None

    sample_blocks = _format_sample_blocks(docs)
    clusters_block = _format_clusters_block(mining)
    user_prompt = textwrap.dedent(f"""\
        Domain hint:  {domain_hint or "(none — infer from documents)"}
        Sample documents ({len(docs)}, total {sum(len(d) for d in docs)} chars):

        {sample_blocks}

        ============================================================
        Pre-mined candidate clusters (deterministic, full-corpus)
        ============================================================
        {clusters_block}
        ============================================================

        Produce the structured JSON brief per the system prompt. STRICT JSON.
        Assign each mined cluster to an entity_type OR drop it as noise.
        Use cluster labels verbatim as term names; copy the sample_quote
        verbatim into evidence_quote.
        """).strip()

    raw = call_llm(
        SYSTEM_PROMPT_CHARACTERIZE,
        user_prompt,
        temperature=0.1,
        max_tokens=3072,
        model=model or task_model("onboarding"),
    )

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("Stage A returned non-JSON: %s — using fallback", e)
        return CorpusCharacterization(
            archetype="general_technical",
            archetype_rationale=f"fallback: JSON parse failed ({e})",
            raw_response=raw,
        )

    archetype = str(payload.get("archetype") or "general_technical").strip()
    if archetype not in ARCHETYPES:
        logger.warning("Stage A produced unknown archetype %r — coercing to general_technical", archetype)
        archetype = "general_technical"

    return CorpusCharacterization(
        archetype=archetype,
        archetype_confidence=float(payload.get("archetype_confidence") or 0.0),
        archetype_rationale=str(payload.get("archetype_rationale") or ""),
        summary=str(payload.get("summary") or ""),
        primary_actors=[str(x) for x in (payload.get("primary_actors") or [])],
        primary_workflows=[str(x) for x in (payload.get("primary_workflows") or [])],
        recommended_entity_types=[str(x) for x in (payload.get("recommended_entity_types") or [])],
        candidate_vocabulary=dict(payload.get("candidate_vocabulary") or {}),
        representative_ids=list(payload.get("representative_ids") or []),
        domain_hazards=[str(x) for x in (payload.get("domain_hazards") or [])],
        domain_intents=[str(x) for x in (payload.get("domain_intents") or [])],
        raw_response=raw,
    )


# ─── Stage B — drafter prompt builder ───────────────────────────────────────


def _format_characterization_block(c: CorpusCharacterization) -> str:
    """Human-readable rendering of Stage A's output, embedded in the
    drafter user message. We keep it short and prescriptive: archetype,
    summary, candidate vocab with citations, representative IDs."""
    parts = [
        f"Archetype: {c.archetype}  (confidence {c.archetype_confidence:.2f})",
        f"Rationale: {c.archetype_rationale}",
        f"Summary:   {c.summary}",
    ]
    if c.primary_actors:
        parts.append(f"Primary actors:    {', '.join(c.primary_actors)}")
    if c.primary_workflows:
        parts.append(f"Primary workflows: {', '.join(c.primary_workflows)}")
    if c.recommended_entity_types:
        parts.append(f"Recommended entity types: {', '.join(c.recommended_entity_types)}")
    if c.domain_intents:
        parts.append(f"Domain intents:    {', '.join(c.domain_intents)}")
    if c.domain_hazards:
        parts.append(f"Domain hazards:    {', '.join(c.domain_hazards)}")

    if c.candidate_vocabulary:
        parts.append("")
        parts.append("Candidate vocabulary (use ONLY these terms — drop, don't add):")
        for et_name, terms in c.candidate_vocabulary.items():
            terms_list = []
            for t in terms:
                if isinstance(t, dict):
                    term = t.get("term", "")
                    quote = t.get("evidence_quote", "")
                    snip = (quote[:60] + "…") if len(quote) > 60 else quote
                    terms_list.append(f"      - {term!r}   ← {snip!r}")
                else:
                    terms_list.append(f"      - {str(t)!r}")
            parts.append(f"   {et_name}:")
            parts.extend(terms_list)

    if c.representative_ids:
        parts.append("")
        parts.append("Representative IDs found in corpus (use ONLY these in examples / id_patterns):")
        for rid in c.representative_ids:
            if isinstance(rid, dict):
                parts.append(
                    f"   - {rid.get('id', '')!r}  pattern={rid.get('suggested_id_pattern', '')!r}"
                )
            else:
                parts.append(f"   - {str(rid)!r}")
    else:
        parts.append("")
        parts.append(
            "Representative IDs: NONE — the corpus contains no asset-tag-style "
            "identifiers. DO NOT declare any entity_type with an id_pattern, "
            "and DO NOT reference invented IDs in examples."
        )

    return "\n".join(parts)


def _build_drafter_prompt(
    *,
    domain_id: str,
    domain_hint: str,
    user_prefs: Optional[Dict[str, Any]],
    prior_qa: Optional[List[Dict[str, str]]],
    docs: List[str],
    characterization: CorpusCharacterization,
    force_generate: bool,
) -> str:
    """Build the Stage B user prompt. The gold-standard manufacturing
    schema is inlined ONLY when archetype == maintenance_manual; for any
    other archetype the model gets the (much shorter) archetype skeleton
    plus Stage A's grounded candidates."""
    archetype = characterization.archetype
    skeleton = ARCHETYPE_SKELETONS.get(archetype, ARCHETYPE_SKELETONS["general_technical"])

    if archetype == "maintenance_manual":
        gold = _load_gold_standard()
        gold_block = (
            "============================================================\n"
            "Maintenance-domain reference — schemas/manufacturing.yaml\n"
            "(consult for shape only; do NOT carry over EV / aviation / etc.\n"
            " specifics — your corpus is its own domain)\n"
            "============================================================\n"
            f"{gold}\n"
            "============================================================\n"
        )
    else:
        gold_block = (
            "(no reference schema inlined — archetype is not maintenance_manual; "
            "use the skeleton above and the candidate_vocabulary as your only guide)\n"
        )

    user_prompt = USER_TEMPLATE.format(
        domain_id=domain_id,
        domain_hint=domain_hint or "(none — infer from documents)",
        prefs=json.dumps(user_prefs or {}, ensure_ascii=False),
        characterization_block=_format_characterization_block(characterization),
        archetype=archetype,
        archetype_skeleton=skeleton,
        n_docs=len(docs),
        n_chars=sum(len(d) for d in docs),
        sample_blocks=_format_sample_blocks(docs),
        prior_qa=_format_prior_qa(prior_qa or []),
        gold_standard_block=gold_block,
    )

    if force_generate or (prior_qa and len(prior_qa) >= 1):
        user_prompt += textwrap.dedent("""

            ============================================================
            FORCE-GENERATE MODE
            ============================================================
            The user has already provided answers OR explicitly asked to
            skip further questions. You MUST now emit a complete schema:

              - ``ready_to_generate`` MUST be ``true``
              - ``follow_up_questions`` MUST be an empty list ``[]``
              - ``yaml`` MUST be a complete, valid schema YAML

            For any remaining ambiguity, make a reasonable guess based on
            the evidence in the documents and surface it in ``analysis``
            as "Assumed X because Y". Do not refuse. Do not ask another
            question. Ship a working schema; the user can iterate later.
            """).strip()

    return user_prompt


# ─── Stage D — repair loop ──────────────────────────────────────────────────


def _repair_schema(
    *,
    yaml_str: str,
    report: Dict[str, Any],
    characterization: CorpusCharacterization,
    docs: List[str],
    domain_id: str,
    model: Optional[str] = None,
) -> Optional[str]:
    """Single repair pass. Given the failing report and the previous
    YAML, ask the model for a corrected version targeted at the specific
    gate failures. Returns the repaired YAML string, or None on parse
    failure (caller bails out of the loop)."""
    errors = report.get("errors") or []
    if not errors:
        return None

    # Compact, actionable error list — keep it short so the model focuses.
    error_block = "\n".join(f"  - {e}" for e in errors[:20])

    user_prompt = textwrap.dedent(f"""\
        Domain id:   {domain_id}
        Archetype:   {characterization.archetype}

        ## Failing validation gates

        {error_block}

        ## Candidate grounding (Stage A, ground truth)

        {_format_characterization_block(characterization)}

        ## Previous YAML (correct only the failing blocks; keep the rest)

        ```yaml
        {yaml_str}
        ```

        Emit STRICT JSON only:

          {{"yaml": "<full corrected schema>", "fix_summary": "<one paragraph>"}}
        """).strip()

    raw = call_llm(
        SYSTEM_PROMPT_REPAIR,
        user_prompt,
        temperature=0.1,
        max_tokens=4096,
        model=model or task_model("onboarding"),
    )

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("Stage D repair returned non-JSON: %s", e)
        return None

    new_yaml = str(payload.get("yaml") or "").strip()
    if not new_yaml:
        return None
    summary = payload.get("fix_summary", "")
    if summary:
        logger.info("Repair pass applied: %s", summary)
    return new_yaml
