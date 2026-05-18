"""CodeExtractor — deterministic regex over chunk text.

Top tier of the three-tier extraction stack (confidence = 1.0). Matches the
canonical identifier patterns declared in the schema (equipment IDs,
alarm codes, part numbers, fault codes). Anything matched here is
treated as authoritative; the LLM never sees this data with discretion.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Tuple

from core.kg.extractors.base import EdgeCandidate, ExtractionResult, Extractor, Mention
from core.kg.provenance import ProvenanceAuthor
from core.kg.schema import Schema


class CodeExtractor(Extractor):
    """Pull identifiers out of chunk text using the schema's id_patterns."""

    author = ProvenanceAuthor.CODE
    default_confidence = 1.0

    def __init__(self, schema: Schema):
        self.schema = schema
        # Pre-compile only entity types that declared an id_pattern.
        self._patterns: Tuple[Tuple[str, re.Pattern], ...] = tuple(
            (name, et.id_pattern)
            for name, et in schema.entity_types.items()
            if et.id_pattern is not None
        )

    def extract(self, document: Dict[str, Any]) -> ExtractionResult:
        result = ExtractionResult()
        text = document.get("text", "") or ""
        chunk_id = document["chunk_id"]
        if not text or not self._patterns:
            return result

        # Use `findall` per pattern — patterns are anchored (`^...$`) but
        # we want to scan inline, so wrap each one in a positive lookbehind
        # for non-word boundary. Simpler approach: strip anchors and use
        # `re.findall` against the bare body.
        for type_name, pattern in self._patterns:
            body = pattern.pattern.strip("^$")
            flags = pattern.flags
            try:
                # Strip existing capture groups by re-wrapping in a non-
                # capturing group; otherwise ``finditer.group(0)`` returns
                # the full match while findall would return groups only.
                inline = re.compile(rf"\b(?:{body})\b", flags)
            except re.error:
                # If the original pattern is too complex to inline-wrap,
                # skip this type — better to miss a few than to crash.
                continue
            for match in inline.finditer(text):
                identifier = match.group(0)
                if flags & re.IGNORECASE:
                    identifier = identifier.upper()
                result.mentions.append(
                    self._stamp_mention(identifier, type_name, chunk_id)
                )

        return result
