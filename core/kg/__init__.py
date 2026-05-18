"""Knowledge-graph layer with kgrag's three-tier model.

Public surface:

    from core.kg import Schema, Provenance, load_default_schema
    from core.kg.gap_detector import detect_gaps, Gap

See ``schemas/manufacturing.yaml`` for the declarative schema and
``DECISIONS.md`` for the design rationale.
"""

from core.kg.provenance import Provenance, ProvenanceAuthor
from core.kg.schema import EdgeType, EntityType, Schema, load_default_schema

__all__ = [
    "EdgeType",
    "EntityType",
    "Provenance",
    "ProvenanceAuthor",
    "Schema",
    "load_default_schema",
]
