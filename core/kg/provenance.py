"""Provenance — kgrag's first-class source tag on every node and edge.

Every KG mutation stamps a `Provenance` so retrieval can later prefer
authoritative sources, the audit log can attribute every decision, and the
gap detector can identify low-confidence edges for HITL review.

The ``author`` field uses a namespaced convention:

* ``system:code``        — deterministic regex extraction (highest trust)
* ``system:metadata``    — chunk-metadata extraction (already-structured)
* ``system:llm_extract`` — LLM-extracted narrative relation (lowest trust)
* ``user:<user_id>``     — human-authored / HITL-confirmed
* ``import:<system>``    — bulk-imported from a system of record (CMMS, FMEA)

A human-authored edge can ``supersedes`` a system edge, which is the
mechanism that closes a HITL gap: the gap detector ignores nodes / edges
that have been superseded by a ``user:`` provenance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class ProvenanceAuthor:
    """Canonical author strings — avoid magic literals across the codebase."""

    CODE = "system:code"
    METADATA = "system:metadata"
    KEYWORD = "system:keyword"
    LLM_EXTRACT = "system:llm_extract"
    IMPORT_CMMS = "import:cmms"
    IMPORT_FMEA = "import:fmea"

    @staticmethod
    def user(user_id: str) -> str:
        return f"user:{user_id}"

    @staticmethod
    def is_user(author: str) -> bool:
        return bool(author) and author.startswith("user:")

    @staticmethod
    def is_system(author: str) -> bool:
        return bool(author) and author.startswith("system:")

    @staticmethod
    def is_import(author: str) -> bool:
        return bool(author) and author.startswith("import:")


@dataclass
class Provenance:
    """Per-node / per-edge metadata stamped at creation time."""

    author: str
    confidence: float = 1.0
    source_chunk_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    supersedes: Optional[str] = None  # id of a node/edge this one replaces
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "author": self.author,
            "confidence": round(float(self.confidence), 4),
            "source_chunk_id": self.source_chunk_id,
            "timestamp": self.timestamp,
            "supersedes": self.supersedes,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["Provenance"]:
        if not data:
            return None
        return cls(
            author=str(data.get("author", "")),
            confidence=float(data.get("confidence", 1.0) or 0.0),
            source_chunk_id=data.get("source_chunk_id"),
            timestamp=float(data.get("timestamp", time.time())),
            supersedes=data.get("supersedes"),
            notes=str(data.get("notes", "") or ""),
        )

    @property
    def is_user_authored(self) -> bool:
        return ProvenanceAuthor.is_user(self.author)

    @property
    def is_high_trust(self) -> bool:
        """High-trust = deterministic / structured / human-authored."""
        return (
            ProvenanceAuthor.is_user(self.author)
            or ProvenanceAuthor.is_import(self.author)
            or self.author in (ProvenanceAuthor.CODE, ProvenanceAuthor.METADATA)
        )
