"""NarrativeExtractor — keyword-list heuristics over chunk prose.

Bottom tier of the extraction stack (confidence = 0.5). These are
*candidates*, not assertions — the gap detector will likely flag low-
confidence symptom / procedure / component edges for HITL review, and
that's the intended workflow:

1. Narrative extraction surfaces candidates from prose.
2. Gap detector flags them as `LOW_CONFIDENCE_EDGE`.
3. Operator reviews + confirms (or rejects); HITL writes back as
   `user:<id>` provenance which supersedes the system edge.

The patterns here are intentionally cheap — regex + keyword lookup.
Plug a NarrativeExtractor backed by an LLM in here when you want fuzzier
relation extraction; the interface stays the same so callers don't change.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from core.kg.extractors.base import ExtractionResult, Extractor
from core.kg.provenance import ProvenanceAuthor
from core.kg.schema import Schema


_SYMPTOM_PATTERNS = (
    re.compile(r"(?:symptom|indication|sign|observed|detected|reported)[:\s]+([^.]+)", re.IGNORECASE),
    re.compile(r"(?:high|low|excessive|abnormal|unexpected)\s+\w+(?:\s+\w+){0,3}", re.IGNORECASE),
)

_PROCEDURE_PATTERNS = (
    re.compile(r"(?:procedure|step|action|resolution)[:\s]+([^.]+)", re.IGNORECASE),
    re.compile(
        r"(?:replace|inspect|check|verify|adjust|clean|lubricate|tighten)\s+(?:the\s+)?(\w+(?:\s+\w+){0,4})",
        re.IGNORECASE,
    ),
)


class NarrativeExtractor(Extractor):
    author = ProvenanceAuthor.LLM_EXTRACT
    default_confidence = 0.5

    def __init__(self, schema: Schema):
        self.schema = schema
        # Closed-vocab extraction (Component, Cause, …) is handled by the
        # KeywordExtractor — this extractor focuses on the open-vocab tiers
        # (Symptom phrases, Procedure phrases) the schema deliberately
        # leaves unconstrained.

    def extract(self, document: Dict[str, Any]) -> ExtractionResult:
        result = ExtractionResult()
        text = (document.get("text") or "").strip()
        chunk_id = document["chunk_id"]
        if not text:
            return result

        meta = document.get("metadata") or {}
        equipment_ids = [str(x) for x in (meta.get("equipment_ids") or []) if x]

        # ── Symptoms ────────────────────────────────────────────────────
        for pattern in _SYMPTOM_PATTERNS:
            for match in pattern.findall(text)[:3]:
                phrase = (match if isinstance(match, str) else match[0]).strip()
                if not phrase:
                    continue
                sym_id = f"SYM:{phrase[:50]}"
                result.mentions.append(self._stamp_mention(sym_id, "Symptom", chunk_id))
                for eq in equipment_ids:
                    if self.schema.edge("HAS_SYMPTOM"):
                        result.edges.append(
                            self._stamp_edge(eq, sym_id, "HAS_SYMPTOM", chunk_id)
                        )

        # ── Procedures ──────────────────────────────────────────────────
        for pattern in _PROCEDURE_PATTERNS:
            for match in pattern.findall(text)[:3]:
                phrase = (match if isinstance(match, str) else match[0]).strip()
                if not phrase:
                    continue
                proc_id = f"PROC:{phrase[:50]}"
                result.mentions.append(self._stamp_mention(proc_id, "Procedure", chunk_id))

        return result
