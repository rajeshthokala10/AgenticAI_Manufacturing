"""KeywordExtractor — generic, schema-driven vocabulary matcher.

Ported from kgrag's L3 keyword extractor. The idea: any entity type that
declares a closed ``vocabulary`` in the schema can be lifted out of chunk
text by a single, domain-agnostic substring scan — no Python edits when a
new domain (piston engines, medical devices, semiconductor tools, …) is
added to the schema.

For every chunk this extractor:

1. Iterates every ``EntityType`` in the schema that declares a non-empty
   ``vocabulary``.
2. Lowercases the chunk text once and scans for each vocab phrase as a
   word-boundaried substring. Each hit emits a :class:`Mention` whose
   ``identifier`` is the vocab phrase itself (matching the schema's
   ``accepts()`` check).
3. For each matched entity, walks the schema's ``edge_types`` looking for
   an edge whose ``target`` includes this entity type AND whose ``source``
   contains ``Equipment``. If chunk metadata supplied ``equipment_ids``,
   emit one :class:`EdgeCandidate` per (equipment, vocab) pair.

Confidence is 0.95 — high, because matches are deterministic against a
hand-curated vocab, but not 1.0 (CodeExtractor-grade) because substring
matching can still surface false positives in long prose.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from core.kg.extractors.base import ExtractionResult, Extractor
from core.kg.provenance import ProvenanceAuthor
from core.kg.schema import Schema


def _compile_vocab(
    schema: Schema,
) -> List[Tuple[str, str, re.Pattern]]:
    """Pre-compute (entity_type_name, vocab_phrase, regex) for every vocab term.

    Word-boundaried (``\\b``) so "seal" doesn't match "sealed" and "valve"
    doesn't match "valved". Phrases containing spaces still work because
    ``\\b`` matches at non-word transitions.
    """
    compiled: List[Tuple[str, str, re.Pattern]] = []
    for ent_name, ent in schema.entity_types.items():
        if not ent.vocabulary:
            continue
        for phrase in ent.vocabulary:
            pat = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
            compiled.append((ent_name, phrase, pat))
    return compiled


def _edges_for_target(schema: Schema, target_type: str, source_type: str = "Equipment") -> List[str]:
    """Return edge_type names where source_type→target_type is valid."""
    out: List[str] = []
    for name, edge in schema.edge_types.items():
        if target_type in edge.target and source_type in edge.source:
            out.append(name)
    return out


class KeywordExtractor(Extractor):
    """Schema-vocab-driven extractor — the L3 'config not code' tier."""

    author = ProvenanceAuthor.KEYWORD
    default_confidence = 0.95

    def __init__(self, schema: Schema):
        self.schema = schema
        self._vocab = _compile_vocab(schema)
        # edge_type name → {target_type: True} cache so we don't rescan
        # the schema for every chunk.
        self._target_edges: Dict[str, List[str]] = {}
        for target_type in {ent_name for ent_name, _, _ in self._vocab}:
            self._target_edges[target_type] = _edges_for_target(schema, target_type)

    def extract(self, document: Dict[str, Any]) -> ExtractionResult:
        result = ExtractionResult()
        text = (document.get("text") or "")
        if not text:
            return result

        chunk_id = document["chunk_id"]
        meta = document.get("metadata") or {}
        equipment_ids = [str(x) for x in (meta.get("equipment_ids") or []) if x]

        # De-dup mentions per (entity_type, phrase) per chunk: a single
        # chunk shouldn't emit the same node ten times because the phrase
        # appears ten times.
        seen_mentions: set[Tuple[str, str]] = set()

        for ent_name, phrase, pat in self._vocab:
            if not pat.search(text):
                continue

            key = (ent_name, phrase)
            if key in seen_mentions:
                continue
            seen_mentions.add(key)

            result.mentions.append(
                self._stamp_mention(phrase, ent_name, chunk_id)
            )

            # Emit edges from any Equipment id present in metadata to this
            # vocab mention, restricted to schema-declared edge types.
            for edge_name in self._target_edges.get(ent_name, ()):
                for eq in equipment_ids:
                    result.edges.append(
                        self._stamp_edge(eq, phrase, edge_name, chunk_id)
                    )

        return result
