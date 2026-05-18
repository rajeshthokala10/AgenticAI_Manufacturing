"""Per-domain LLM prompt + safety-keyword loader.

Reads ``prompts:`` and ``safety:`` blocks from ``schemas/<domain>.yaml`` and
returns them to the orchestrator / critic / drafter / cause-ranker /
classifier nodes. When a block (or the whole schema) is silent on a given
key, callers receive the manufacturing-flavoured default they passed in —
nothing regresses.

The schema YAML is the single source of truth for any user-authored
overrides. New domains author overrides at file-edit time; no Python
changes required.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

from core.schema_validator import SchemaValidationResult, validate_schema

logger = logging.getLogger("core.domain_prompts")

# Per-domain validation results. Populated as a side-effect of
# ``_load_schema`` so callers (API, CLI) can surface schema status
# without re-parsing the YAML.
_VALIDATION_CACHE: Dict[str, SchemaValidationResult] = {}


@lru_cache(maxsize=None)
def _load_schema(domain: str) -> Dict[str, Any]:
    """Load + validate the schema YAML for ``domain``.

    Cached for the process lifetime. Validation runs once per domain on
    first load; results are cached separately in ``_VALIDATION_CACHE``
    and can be retrieved via :func:`schema_status`. Errors and warnings
    are logged at load time so they show up in the bootstrap log even
    if no caller asks for them.
    """
    try:
        import yaml as _yaml
        from config import schema_path
        path = schema_path(domain)
        raw = _yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        logger.error("schema for domain=%r not found: %s", domain, exc)
        result = SchemaValidationResult(
            domain=domain, ok=False,
            errors=[f"schema file not found: {exc}"],
        )
        _VALIDATION_CACHE[domain] = result
        return {}
    except Exception as exc:
        logger.error(
            "schema for domain=%r could not be parsed: %s", domain, exc,
        )
        result = SchemaValidationResult(
            domain=domain, ok=False,
            errors=[f"YAML parse error: {exc}"],
        )
        _VALIDATION_CACHE[domain] = result
        return {}

    if not isinstance(raw, dict):
        logger.error("schema for domain=%r is not a YAML mapping", domain)
        _VALIDATION_CACHE[domain] = SchemaValidationResult(
            domain=domain, ok=False,
            errors=["schema root must be a YAML mapping"],
        )
        return {}

    result = validate_schema(domain, raw)
    _VALIDATION_CACHE[domain] = result
    if result.errors:
        logger.error(
            "schema for domain=%r has %d error(s): %s",
            domain, len(result.errors), "; ".join(result.errors),
        )
    if result.warnings:
        logger.warning(
            "schema for domain=%r has %d warning(s): %s",
            domain, len(result.warnings), "; ".join(result.warnings),
        )
    return raw


def schema_status(domain: str) -> SchemaValidationResult:
    """Return the cached validation result for ``domain``.

    Lazily triggers a load if the schema hasn't been read yet — guaranteed
    to populate the cache. Always returns a result; never raises.
    """
    if domain not in _VALIDATION_CACHE:
        _load_schema(domain)
    return _VALIDATION_CACHE.get(
        domain,
        SchemaValidationResult(
            domain=domain, ok=False,
            errors=["schema not loaded (no cache entry)"],
        ),
    )


def all_schema_statuses(domains: Tuple[str, ...]) -> Dict[str, SchemaValidationResult]:
    """Force-load + validate every domain. Useful at API startup."""
    return {d: schema_status(d) for d in domains}


def reload_schemas(
    domains: Optional[Tuple[str, ...]] = None,
) -> Dict[str, SchemaValidationResult]:
    """Drop the schema + validation caches and re-validate.

    Call after editing a ``schemas/*.yaml`` so the next ``get_prompt`` /
    ``get_high_risk_keywords`` / ``procedure_should_run`` call picks up
    the change without an API restart. Returns the fresh per-domain
    validation results so callers can surface them immediately.

    Note: this only invalidates this module's caches. The orchestrators
    and per-domain ``ManufacturingPipeline`` instances built at startup
    keep their references — for a *full* domain rebuild (KG, vector
    store), the API still needs to rebuild pipelines explicitly. This is
    fine for the common "tweak a prompt, see the change" loop.
    """
    _load_schema.cache_clear()
    _VALIDATION_CACHE.clear()
    if domains is None:
        from config import DOMAINS as _DOMAINS  # local import to avoid cycle
        domains = tuple(_DOMAINS)
    return all_schema_statuses(domains)


def get_prompt(domain: Optional[str], key: str, default: str) -> str:
    """Return the schema-defined prompt ``key`` for ``domain``, else ``default``.

    ``default`` is the module-level constant the caller already has, so this
    is a strict overlay — silent schemas leave behaviour unchanged.
    """
    if not domain:
        return default
    block = _load_schema(domain).get("prompts") or {}
    value = block.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return default


def get_procedure_config(domain: Optional[str]) -> Dict[str, Any]:
    """Return the schema's ``procedure:`` block for ``domain``.

    Shape::

        {"enabled": bool, "trigger_intents": tuple[str, ...]}

    When the schema is silent on a key, callers get sensible defaults:
    ``enabled=True`` (matches legacy behaviour where the drafter was always
    on for troubleshooting intents) and ``trigger_intents=()`` (caller's
    fallback list — typically the legacy ``_TROUBLESHOOTING_TRIGGERS`` —
    is then used).
    """
    if not domain:
        return {"enabled": True, "trigger_intents": ()}
    block = _load_schema(domain).get("procedure") or {}
    enabled = block.get("enabled", True)
    raw = block.get("trigger_intents") or []
    triggers = tuple(str(t).strip().lower() for t in raw if str(t).strip())
    return {"enabled": bool(enabled), "trigger_intents": triggers}


def procedure_should_run(
    domain: Optional[str],
    intent: Optional[str],
    fallback_triggers: Tuple[str, ...],
) -> bool:
    """Decide whether the structured procedure drafter should fire.

    Honours the schema's ``procedure.enabled`` flag (a domain can opt out
    entirely). When ``procedure.trigger_intents`` is set on the schema we
    use that list; otherwise we fall back to ``fallback_triggers``
    (typically the legacy substring triggers in core/cause_ranker.py).
    """
    cfg = get_procedure_config(domain)
    if not cfg["enabled"]:
        return False
    if not intent:
        return False
    needle = str(intent).lower()
    triggers = cfg["trigger_intents"] or fallback_triggers
    return any(t in needle for t in triggers)


def get_high_risk_keywords(
    domain: Optional[str],
    default: Tuple[str, ...],
) -> Tuple[str, ...]:
    """Return the per-domain HITL escalation keywords or ``default``.

    Lower-cased, stripped, and de-duplicated, matching the contract
    ``config.HITL_HIGH_RISK_KEYWORDS`` already satisfies for the global list.
    """
    if not domain:
        return default
    safety = _load_schema(domain).get("safety") or {}
    raw = safety.get("high_risk_keywords")
    if not raw:
        return default
    seen: Dict[str, None] = {}
    for item in raw:
        kw = str(item).strip().lower()
        if kw:
            seen.setdefault(kw, None)
    return tuple(seen) or default
