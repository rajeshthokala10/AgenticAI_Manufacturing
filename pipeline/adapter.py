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


EQUIPMENT_RE = re.compile(r'(?:P-\d{3}|CV-\d{3}|HP-\d{3}|CNC-[A-Z]-\d{3}|'
                          r'STAMP-[A-Z]-\d{3}|WELD-[A-Z]-\d{3}|HT-[A-Z]-\d{3}|'
                          r'COAT-[A-Z]-\d{3})')
ALARM_RE = re.compile(r'ALM-[A-Z]\d{3}')
PART_RE = re.compile(r'(?:SP-\d{4}|TH-\d{4}|BRK-\d{4}|SFT-\d{4}|HSG-\d{4}|GR-\d{4})')
FAULT_RE = re.compile(r'FC-\d{3}')


def stable_chunk_id(source: str, index: int) -> str:
    """Deterministic short hash so repeated indexing produces stable IDs."""
    raw = f"{source}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _extract_entity_metadata(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key, pat in [
        ("equipment_ids", EQUIPMENT_RE),
        ("alarm_codes", ALARM_RE),
        ("part_numbers", PART_RE),
        ("fault_codes", FAULT_RE),
    ]:
        matches = pat.findall(text)
        if matches:
            out[key] = sorted({m.upper() for m in matches})
    return out


def chunks_to_core_docs(chunks: List) -> List[Dict]:
    """Convert doc_pipeline `Chunk` objects to core-style dicts.

    Each output element is:
        {
            "chunk_id": "<stable hash>",
            "text": "<chunk text>",
            "metadata": { ...existing chunk metadata, plus equipment_ids etc. },
        }
    """
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

        for k, v in _extract_entity_metadata(chunk.text).items():
            metadata.setdefault(k, v)

        docs.append({
            "chunk_id": cid,
            "text": chunk.text,
            "metadata": metadata,
        })
    return docs
