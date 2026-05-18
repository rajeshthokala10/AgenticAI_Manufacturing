"""Tiered extraction — kgrag L3.

Four extractors with descending confidence. Each emits the same
`(Mention, EdgeCandidate)` shapes so the KG builder can stamp consistent
provenance and the schema validator can reject out-of-schema candidates
uniformly.

* :class:`CodeExtractor`      — deterministic regex (codes, IDs); 1.0 confidence.
* :class:`MetadataExtractor`  — chunk metadata fields already parsed by the
  ingestion layer; 0.95 confidence.
* :class:`KeywordExtractor`   — generic, schema-vocab-driven substring
  matcher (kgrag's L3 'config not code' tier); 0.95 confidence. Adding a
  new domain is a schema YAML edit — no Python.
* :class:`NarrativeExtractor` — open-vocab regex heuristics over prose
  (Symptom / Procedure phrases the schema deliberately leaves
  unconstrained); 0.5 confidence (HITL-reviewed candidates).
"""

from core.kg.extractors.base import EdgeCandidate, ExtractionResult, Extractor, Mention
from core.kg.extractors.code import CodeExtractor
from core.kg.extractors.keyword import KeywordExtractor
from core.kg.extractors.metadata import MetadataExtractor
from core.kg.extractors.narrative import NarrativeExtractor

__all__ = [
    "CodeExtractor",
    "EdgeCandidate",
    "ExtractionResult",
    "Extractor",
    "KeywordExtractor",
    "MetadataExtractor",
    "Mention",
    "NarrativeExtractor",
]
