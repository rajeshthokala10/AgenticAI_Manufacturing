"""
Adapter — converts doc_pipeline `Chunk` objects into the
`{chunk_id, text, metadata}` dict format consumed by core/ retrievers and
core/knowledge_graph.

Also enriches metadata with entity fields (equipment_ids, alarm_codes,
part_numbers, fault_codes, section_title) that the KG builder relies on.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("pipeline.adapter")


def _schema_equipment_pattern(domain: str | None) -> re.Pattern | None:
    """Pull the Equipment ``id_pattern`` out of ``schemas/<domain>.yaml``.

    The schema is the *only* source of truth for what an Equipment id
    looks like in this domain. New domains added via
    ``scripts/onboard_domain.sh`` get their ids picked up without
    touching this module.
    """
    if not domain:
        return None
    try:
        from core.kg.schema import load_default_schema
        schema = load_default_schema(domain)
    except Exception:  # pragma: no cover — defensive: skip if schema fails to load
        return None
    eq = schema.entity("Equipment")
    if eq is None or eq.id_pattern is None:
        return None
    # Schema id_patterns are anchored (``^...$``) for validation; the
    # adapter scans free chunk text, where anchors block substring
    # matches. Strip the outer anchors and wrap in word boundaries so we
    # still require a token boundary (otherwise ``SP-1234`` would partially
    # match ``P-1234`` from a wider mfg pattern).
    src = eq.id_pattern.pattern
    if src.startswith("^"):
        src = src[1:]
    if src.endswith("$"):
        src = src[:-1]
    src = rf"(?:^|\b|(?<=[\s,.;:]))(?:{src})(?=\b|$|[\s,.;:?!])"
    try:
        return re.compile(src, eq.id_pattern.flags)
    except re.error:  # pragma: no cover — malformed regex shouldn't kill ingestion
        return None
ALARM_RE = re.compile(r'ALM-[A-Z]\d{3}')
PART_RE = re.compile(r'(?:SP-\d{4}|TH-\d{4}|BRK-\d{4}|SFT-\d{4}|HSG-\d{4}|GR-\d{4})')
FAULT_RE = re.compile(r'FC-\d{3}')


def stable_chunk_id(source: str, index: int) -> str:
    """Deterministic short hash so repeated indexing produces stable IDs."""
    raw = f"{source}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _extract_entity_metadata(
    text: str,
    equipment_pattern: re.Pattern | None = None,
) -> Dict[str, List[str]]:
    """Regex-extract equipment / alarm / part / fault ids from chunk text.

    ``equipment_pattern`` comes from the active domain's schema (see
    ``_schema_equipment_pattern``). When None, no equipment ids are
    lifted — which is fine: domains without an Equipment ``id_pattern``
    simply don't surface them in chunk metadata.
    """
    out: Dict[str, List[str]] = {}
    if equipment_pattern is not None:
        # Use finditer().group(0) so author-side capturing groups in the
        # schema's id_pattern don't change what we extract — we always
        # want the full match.
        matches = [m.group(0) for m in equipment_pattern.finditer(text)]
        if matches:
            out["equipment_ids"] = sorted({m.upper() for m in matches})

    for key, pat in [
        ("alarm_codes", ALARM_RE),
        ("part_numbers", PART_RE),
        ("fault_codes", FAULT_RE),
    ]:
        matches = pat.findall(text)
        if matches:
            out[key] = sorted({m.upper() for m in matches})
    return out


def chunks_to_core_docs(chunks: List, domain: str | None = None) -> List[Dict]:
    """Convert doc_pipeline `Chunk` objects to core-style dicts.

    Each output element is:
        {
            "chunk_id": "<stable hash>",
            "text": "<chunk text>",
            "metadata": { ...existing chunk metadata, plus equipment_ids etc. },
        }
    """
    equipment_pattern = _schema_equipment_pattern(domain)
    docs: List[Dict] = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        source_name = Path(source).stem if source != "unknown" else "unknown"
        cid = stable_chunk_id(source_name, chunk.chunk_id if hasattr(chunk, "chunk_id") else i)

        metadata = dict(chunk.metadata)
        metadata.setdefault("source", source_name)
        metadata.setdefault("source_file", Path(source).name if source != "unknown" else "unknown")
        metadata.setdefault("doc_type", getattr(chunk, "strategy", "unknown"))
        metadata.setdefault("chunk_index", i)

        for k, v in _extract_entity_metadata(chunk.text, equipment_pattern).items():
            metadata.setdefault(k, v)

        docs.append({
            "chunk_id": cid,
            "text": chunk.text,
            "metadata": metadata,
        })
    return docs
