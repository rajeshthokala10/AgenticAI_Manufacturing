"""Schema validator for the multi-domain contract.

Validates the four NEW top-level blocks introduced for schema-driven
domain onboarding: ``prompts:``, ``safety:``, ``clarifier:``,
``procedure:``. Catches:

* typos at the top level (``prommpts:`` / ``saftey:``) via fuzzy match warnings
* typos inside the new blocks (``high_risk_keyword:``) via pydantic
  ``extra='forbid'``
* string-coerced booleans like ``enabled: "false"`` via pydantic ``StrictBool``
* malformed format placeholders in ``retry_system`` / ``cause_rank_system``
  that would explode at ``.format(...)`` time
* empty / non-list / non-dict values where structure is required

Existing schema blocks (``entity_types``, ``edge_types``,
``traversal_routes``, ``corrections``, ``display``, ``placeholder``,
``empty_state``, ``examples``, ``gap_thresholds``, ``version``,
``domain``) are intentionally NOT re-validated here — they're already
parsed by ``core/kg/schema.py`` (KG ontology) or ``config.py`` (UI copy
discovery) and re-validating them would duplicate that contract.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from difflib import get_close_matches
from string import Formatter
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, field_validator

logger = logging.getLogger("core.schema_validator")


# ─── Known surfaces ───────────────────────────────────────────────────────

# Top-level keys the system will recognise. Anything else triggers a
# warning (not an error — third-party tooling may legitimately stash
# extra metadata in the YAML).
_KNOWN_TOP_KEYS = {
    "version", "domain", "display",
    "entity_types", "edge_types", "traversal_routes",
    "gap_thresholds",
    "prompts", "safety", "clarifier", "procedure", "corrections",
    "placeholder", "empty_state", "examples",
}

# Format-string placeholders the system passes to specific prompts.
# Anything else in those prompts will raise ``KeyError`` at .format(...)
# time — we catch it here instead.
_ALLOWED_PLACEHOLDERS: Dict[str, set] = {
    "retry_system": {"critic_feedback"},
    "cause_rank_system": {"top_k", "taxonomy_clause"},
    # Other prompts (answer_system, critic_rules, procedure_system,
    # classify_system, risk_grader_*) are NOT format()-called — they're
    # passed directly as system_prompt strings, so braces are fine.
}


# ─── Pydantic models for the new blocks ───────────────────────────────────


class _PromptsBlock(BaseModel):
    """``prompts:`` block — all keys optional, none may be empty strings."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    persona: Optional[str] = None
    answer_system: Optional[str] = None
    retry_system: Optional[str] = None
    critic_rules: Optional[str] = None
    procedure_system: Optional[str] = None
    cause_rank_system: Optional[str] = None
    classify_system: Optional[str] = None
    risk_grader_system: Optional[str] = None
    risk_grader_user: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def _reject_blank(cls, v):
        # Treat the literal empty string as "missing" so authors don't
        # accidentally suppress the fallback by writing ``answer_system: ""``.
        if isinstance(v, str) and not v.strip():
            return None
        return v


class _SafetyBlock(BaseModel):
    """``safety:`` block — HITL escalation keywords."""

    model_config = ConfigDict(extra="forbid")

    high_risk_keywords: Optional[List[str]] = None

    @field_validator("high_risk_keywords")
    @classmethod
    def _clean_keywords(cls, v):
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("must be a YAML list")
        cleaned = [str(x).strip() for x in v if str(x).strip()]
        # An explicitly empty list is a *valid* opt-out (the domain
        # genuinely wants zero keyword-based escalation). We return [] so
        # the loader can distinguish empty (opt-out) from None (use the
        # global fallback).
        return cleaned


class _ClarifierEquipmentPattern(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pattern: str
    type: str = "equipment_id"


class _ClarifierIntentPattern(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: str
    patterns: List[str]
    boost: float = 0.85

    @field_validator("patterns")
    @classmethod
    def _non_empty(cls, v):
        if not v:
            raise ValueError("patterns must be a non-empty list")
        return [str(p) for p in v]


class _SlotDef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    entity_types: List[str] = Field(default_factory=list)
    required: bool = False
    prompt: str


class _ClarifierBlock(BaseModel):
    """``clarifier:`` block — entity hints, intent regexes, slot templates."""

    model_config = ConfigDict(extra="forbid")

    equipment_patterns: List[_ClarifierEquipmentPattern] = Field(default_factory=list)
    part_number_patterns: List[_ClarifierEquipmentPattern] = Field(default_factory=list)
    supplier_names: Dict[str, str] = Field(default_factory=dict)
    metric_names: Dict[str, str] = Field(default_factory=dict)
    department_names: Dict[str, str] = Field(default_factory=dict)
    intent_patterns: List[_ClarifierIntentPattern] = Field(default_factory=list)
    slot_templates: Dict[str, List[_SlotDef]] = Field(default_factory=dict)


class _ProcedureBlock(BaseModel):
    """``procedure:`` block — drafter opt-in + trigger control.

    ``enabled`` is ``StrictBool`` so YAML strings like ``"false"`` or
    ``"no"`` are rejected with a clear message instead of being silently
    coerced to ``True`` by Python's truthiness rules.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: StrictBool = True
    trigger_intents: List[str] = Field(default_factory=list)

    @field_validator("trigger_intents")
    @classmethod
    def _clean(cls, v):
        return [str(t).strip().lower() for t in v if str(t).strip()]


# ─── Validation result type ───────────────────────────────────────────────


@dataclass
class SchemaValidationResult:
    """Outcome of validating a single domain's schema YAML.

    ``ok`` is true iff there are no ``errors`` — warnings don't fail
    validation. The loader keeps a cached copy per domain so the API
    can report it via /api/domains without re-parsing.
    """

    domain: str
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


# ─── Public API ───────────────────────────────────────────────────────────


def _check_placeholders(block: Dict[str, Any], result: SchemaValidationResult) -> None:
    """Walk known format-string fields; reject unexpected placeholders."""
    for key, allowed in _ALLOWED_PLACEHOLDERS.items():
        text = block.get(key)
        if not isinstance(text, str) or not text:
            continue
        try:
            found = {fname for _, fname, _, _ in Formatter().parse(text) if fname}
        except Exception as exc:
            result.errors.append(
                f"prompts.{key}: cannot parse format placeholders ({exc})"
            )
            continue
        unexpected = found - allowed
        if unexpected:
            result.errors.append(
                f"prompts.{key}: contains unexpected placeholders "
                f"{sorted(unexpected)} — only {sorted(allowed)} are substituted "
                f"at runtime; others will raise KeyError at .format() time."
            )
        # cause_rank_system MUST mention {taxonomy_clause} or the taxonomy
        # discipline silently breaks. {top_k} is also load-bearing.
        if key == "cause_rank_system":
            missing = allowed - found
            if missing:
                result.warnings.append(
                    f"prompts.{key}: missing required placeholders {sorted(missing)} — "
                    f"the cause ranker will not enforce taxonomy / top_k limits."
                )


def _check_unknown_top_keys(raw: Dict[str, Any], result: SchemaValidationResult) -> None:
    """Warn on top-level keys that look like a typo of a known one."""
    for key in raw:
        if key in _KNOWN_TOP_KEYS:
            continue
        close = get_close_matches(key, _KNOWN_TOP_KEYS, n=1, cutoff=0.75)
        if close:
            result.warnings.append(
                f"unknown top-level key {key!r} — did you mean {close[0]!r}? "
                f"(falling back to defaults for that block)"
            )
        else:
            result.warnings.append(
                f"unknown top-level key {key!r} — ignored."
            )


def validate_schema(domain: str, raw: Dict[str, Any]) -> SchemaValidationResult:
    """Validate the four NEW blocks of a domain schema.

    ``raw`` is the already-parsed YAML dict (we don't reparse so callers
    can validate in-memory test schemas too). Returns a result you can
    log or surface via the API.
    """
    result = SchemaValidationResult(domain=domain)

    if not isinstance(raw, dict):
        result.ok = False
        result.errors.append("schema root must be a YAML mapping")
        return result

    _check_unknown_top_keys(raw, result)

    block_to_model = [
        ("prompts", _PromptsBlock),
        ("safety", _SafetyBlock),
        ("clarifier", _ClarifierBlock),
        ("procedure", _ProcedureBlock),
    ]
    for name, model in block_to_model:
        block = raw.get(name)
        if block is None:
            continue
        if not isinstance(block, dict):
            result.errors.append(f"{name!r}: must be a YAML mapping, got {type(block).__name__}")
            continue
        try:
            model.model_validate(block)
        except ValidationError as exc:
            for e in exc.errors():
                loc = ".".join(str(p) for p in e["loc"]) or "(root)"
                result.errors.append(f"{name}.{loc}: {e['msg']}")

    # Cross-block check: format-string placeholders.
    prompts = raw.get("prompts") or {}
    if isinstance(prompts, dict):
        _check_placeholders(prompts, result)

    result.ok = not result.errors
    return result
