"""Knowledge-graph schema — kgrag L1 contract.

A `Schema` is the hand-curated ontology: which entity types exist, which
edges connect them, what identifiers are valid, and what the cardinality
constraints are. Loaded once at startup from a YAML file (defaults to
``schemas/manufacturing.yaml``); used by:

* `KnowledgeGraph` to validate every node and edge at construction time.
* The gap detector to spot MISSING_EDGE / CONFLICTING_EDGES / OUT_OF_VOCAB.
* The retrieval allow-list to traverse only declared edges.

The schema separates *what is allowed to exist* (this file) from *what
actually exists* (the graph instance). That separation is the whole point
of the three-tier model — humans curate the schema, extractors populate
the instances, provenance distinguishes who said what.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("core.kg.schema")


@dataclass(frozen=True)
class EntityType:
    name: str
    description: str = ""
    vocabulary: Tuple[str, ...] = ()
    id_pattern: Optional[re.Pattern] = None
    case_sensitive: bool = True

    def accepts(self, identifier: str) -> bool:
        """Return True iff ``identifier`` is a valid instance of this type."""
        if not identifier:
            return False
        if self.id_pattern is not None:
            ident = identifier if self.case_sensitive else identifier
            return bool(self.id_pattern.match(ident))
        if self.vocabulary:
            cmp = identifier if self.case_sensitive else identifier.lower()
            vocab = self.vocabulary if self.case_sensitive else tuple(
                v.lower() for v in self.vocabulary
            )
            return cmp in vocab
        # No constraint declared — accept anything.
        return True


@dataclass(frozen=True)
class EdgeType:
    name: str
    source: Tuple[str, ...]
    target: Tuple[str, ...]
    min_cardinality: int = 0
    max_cardinality: Optional[int] = None
    description: str = ""

    def accepts(self, source_type: str, target_type: str) -> bool:
        return source_type in self.source and target_type in self.target


@dataclass
class Schema:
    """Top-level container declaring the ontology of the KG."""

    version: int
    domain: str
    entity_types: Dict[str, EntityType]
    edge_types: Dict[str, EdgeType]
    traversal_routes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    gap_thresholds: Dict[str, float] = field(default_factory=dict)
    # Optional display metadata — drives the per-domain UI affordances
    # (sidebar selector, chat-header pill, evidence-card border).
    display: Dict[str, str] = field(default_factory=dict)
    # Optional UX copy. ``examples`` is a list of suggested queries shown
    # in the Streamlit sidebar + Next.js empty state; ``empty_state`` holds
    # the heading/blurb shown when the chat has no turns; ``placeholder``
    # is the chat-input ghost text. All optional — sensible fallbacks apply.
    examples: List[str] = field(default_factory=list)
    empty_state: Dict[str, str] = field(default_factory=dict)
    placeholder: str = ""

    # ── Lookups ──────────────────────────────────────────────────────────

    def entity(self, name: str) -> Optional[EntityType]:
        return self.entity_types.get(name)

    def edge(self, name: str) -> Optional[EdgeType]:
        return self.edge_types.get(name)

    # ── Validation ───────────────────────────────────────────────────────

    def validate_entity(self, identifier: str, type_name: str) -> bool:
        """Return True iff ``identifier`` may be inserted as ``type_name``.

        Mismatches log at DEBUG level (extraction is expected to produce
        rejects) and return False so the caller can drop the candidate.
        """
        et = self.entity(type_name)
        if et is None:
            logger.debug("schema reject: unknown entity type %r", type_name)
            return False
        if not et.accepts(identifier):
            logger.debug(
                "schema reject: identifier %r is not a valid %s", identifier, type_name,
            )
            return False
        return True

    def validate_edge(
        self,
        relation: str,
        source_type: str,
        target_type: str,
    ) -> bool:
        eg = self.edge(relation)
        if eg is None:
            logger.debug("schema reject: unknown edge type %r", relation)
            return False
        if not eg.accepts(source_type, target_type):
            logger.debug(
                "schema reject: %s does not connect %s → %s",
                relation, source_type, target_type,
            )
            return False
        return True

    # ── Iteration helpers ────────────────────────────────────────────────

    def entity_names(self) -> List[str]:
        return list(self.entity_types)

    def edge_names(self) -> List[str]:
        return list(self.edge_types)


# ─── Loading ────────────────────────────────────────────────────────────────


def _as_tuple(value: Union[str, Sequence[str], None]) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def load_schema(path: Union[str, Path]) -> Schema:
    """Load a Schema from a YAML file. Raises FileNotFoundError or ValueError."""
    import yaml  # local import; PyYAML is optional in some envs

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"schema file not found: {path}")

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"schema file must be a mapping at top level: {path}")

    entity_types: Dict[str, EntityType] = {}
    for et_raw in raw.get("entity_types") or []:
        name = et_raw["name"]
        vocab = tuple(et_raw.get("vocabulary") or [])
        pattern_str = et_raw.get("id_pattern")
        case_sensitive = bool(et_raw.get("case_sensitive", True))
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(pattern_str, flags) if pattern_str else None
        entity_types[name] = EntityType(
            name=name,
            description=et_raw.get("description", ""),
            vocabulary=vocab,
            id_pattern=pattern,
            case_sensitive=case_sensitive,
        )

    edge_types: Dict[str, EdgeType] = {}
    for eg_raw in raw.get("edge_types") or []:
        edge_types[eg_raw["name"]] = EdgeType(
            name=eg_raw["name"],
            source=_as_tuple(eg_raw.get("source")),
            target=_as_tuple(eg_raw.get("target")),
            min_cardinality=int(eg_raw.get("min_cardinality", 0)),
            max_cardinality=(
                int(eg_raw["max_cardinality"]) if eg_raw.get("max_cardinality") is not None else None
            ),
            description=eg_raw.get("description", ""),
        )

    return Schema(
        version=int(raw.get("version", 1)),
        domain=str(raw.get("domain", "unknown")),
        entity_types=entity_types,
        edge_types=edge_types,
        traversal_routes=dict(raw.get("traversal_routes") or {}),
        gap_thresholds={
            k: float(v) for k, v in (raw.get("gap_thresholds") or {}).items()
        },
        display={
            str(k): str(v) for k, v in (raw.get("display") or {}).items()
        },
        examples=[str(x) for x in (raw.get("examples") or []) if x],
        empty_state={
            str(k): str(v).strip()
            for k, v in (raw.get("empty_state") or {}).items()
        },
        placeholder=str(raw.get("placeholder") or "").strip(),
    )


def load_default_schema(domain: str | None = None) -> Schema:
    """Load the schema for ``domain`` (one of ``config.DOMAINS``).

    Resolution order:

    1. ``KG_SCHEMA_PATH`` env var, if set — overrides everything.
    2. ``config.schema_path(domain)`` for the requested domain.
    3. ``config.DEFAULT_DOMAIN`` if ``domain`` is None.
    """
    import os

    explicit = os.getenv("KG_SCHEMA_PATH", "").strip()
    if explicit:
        return load_schema(explicit)

    from config import schema_path  # local import to avoid a startup cycle
    return load_schema(schema_path(domain))
