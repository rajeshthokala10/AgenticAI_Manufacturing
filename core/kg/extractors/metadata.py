"""MetadataExtractor — pulls pre-parsed entity lists from chunk metadata.

Second tier of the extraction stack (confidence = 0.95). The doc_pipeline
ingestion layer already runs the structured parsers (PDF tables, Excel
columns) and writes ``equipment_ids``, ``alarm_codes``, ``part_numbers``,
``fault_codes`` into the chunk metadata. We trust those — they came from
structured sources — but slightly less than a regex over the chunk's own
text because the ingestion parser is upstream and harder to audit.

Also emits the cross-product edges (Equipment→Alarm, Equipment→SparePart,
Alarm→FailureMode, …) that the legacy KnowledgeGraph used to derive
inline. Moving them here keeps all "what does this chunk's metadata
imply for the graph" logic in one place.
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.kg.extractors.base import ExtractionResult, Extractor
from core.kg.provenance import ProvenanceAuthor
from core.kg.schema import Schema


class MetadataExtractor(Extractor):
    author = ProvenanceAuthor.METADATA
    default_confidence = 0.95

    # Mapping from metadata key → entity type.
    _META_TO_TYPE = {
        "equipment_ids": "Equipment",
        "alarm_codes": "Alarm",
        "part_numbers": "SparePart",
        "fault_codes": "FailureMode",
    }

    def __init__(self, schema: Schema):
        self.schema = schema

    def extract(self, document: Dict[str, Any]) -> ExtractionResult:
        result = ExtractionResult()
        meta = document.get("metadata", {}) or {}
        chunk_id = document["chunk_id"]

        # ── Pull entity mentions out of structured metadata ────────────
        per_type: Dict[str, List[str]] = {}
        for meta_key, type_name in self._META_TO_TYPE.items():
            ids = meta.get(meta_key) or []
            if not isinstance(ids, list):
                continue
            per_type[type_name] = [str(x) for x in ids if x]
            for ident in per_type[type_name]:
                result.mentions.append(self._stamp_mention(ident, type_name, chunk_id))

        # ── Co-occurrence edges within the same chunk ──────────────────
        # We only emit edges the schema declares; the KG builder will
        # validate again but rejecting up-front saves churn.
        equipment = per_type.get("Equipment", [])
        alarms = per_type.get("Alarm", [])
        parts = per_type.get("SparePart", [])
        failures = per_type.get("FailureMode", [])

        for eq in equipment:
            for al in alarms:
                if self.schema.edge("TRIGGERS_ALARM"):
                    result.edges.append(self._stamp_edge(eq, al, "TRIGGERS_ALARM", chunk_id))
            for sp in parts:
                if self.schema.edge("REQUIRES_PART"):
                    result.edges.append(self._stamp_edge(eq, sp, "REQUIRES_PART", chunk_id))
            for fm in failures:
                if self.schema.edge("CAUSES_FAILURE"):
                    result.edges.append(self._stamp_edge(eq, fm, "CAUSES_FAILURE", chunk_id))

        for al in alarms:
            for fm in failures:
                if self.schema.edge("CAUSES_FAILURE"):
                    result.edges.append(self._stamp_edge(al, fm, "CAUSES_FAILURE", chunk_id))

        return result
