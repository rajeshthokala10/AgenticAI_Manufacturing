"""Base types for KG extractors.

Every extractor walks a chunk (text + metadata) and emits two streams:

* :class:`Mention` — an entity assertion (identifier + type).
* :class:`EdgeCandidate` — a directed relation between two mentioned entities.

The KG builder validates each item against the schema, dedupes,
and stamps :class:`Provenance` with the extractor's author + confidence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Mention:
    identifier: str
    entity_type: str
    source_chunk_id: str
    confidence: float = 1.0
    author: str = ""  # filled by the extractor base class on emit

    def __post_init__(self):
        # Normalise so the rest of the codebase can do set-membership /
        # exact match without re-lowering.
        object.__setattr__(self, "identifier", str(self.identifier))


@dataclass(frozen=True)
class EdgeCandidate:
    source_id: str
    target_id: str
    relation: str
    source_chunk_id: str
    confidence: float = 1.0
    author: str = ""

    def __post_init__(self):
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "target_id", str(self.target_id))


@dataclass
class ExtractionResult:
    mentions: List[Mention] = field(default_factory=list)
    edges: List[EdgeCandidate] = field(default_factory=list)

    def extend(self, other: "ExtractionResult") -> None:
        self.mentions.extend(other.mentions)
        self.edges.extend(other.edges)


class Extractor(ABC):
    """Abstract base class for the three-tier extractors.

    Subclasses declare their ``author`` (a `ProvenanceAuthor` constant) and
    ``default_confidence``; the base class stamps both onto every emitted
    `Mention` / `EdgeCandidate` so call-sites don't need to duplicate
    the bookkeeping.
    """

    author: str = ""
    default_confidence: float = 1.0

    @abstractmethod
    def extract(self, document: Dict[str, Any]) -> ExtractionResult:
        """Walk ``document`` (a `{chunk_id, text, metadata}` dict) and
        return mentions + edge candidates. Subclasses should call
        :meth:`_stamp` on each emitted item.
        """
        raise NotImplementedError

    def _stamp_mention(
        self,
        identifier: str,
        entity_type: str,
        source_chunk_id: str,
        *,
        confidence: float | None = None,
    ) -> Mention:
        return Mention(
            identifier=identifier,
            entity_type=entity_type,
            source_chunk_id=source_chunk_id,
            confidence=self.default_confidence if confidence is None else confidence,
            author=self.author,
        )

    def _stamp_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        source_chunk_id: str,
        *,
        confidence: float | None = None,
    ) -> EdgeCandidate:
        return EdgeCandidate(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            source_chunk_id=source_chunk_id,
            confidence=self.default_confidence if confidence is None else confidence,
            author=self.author,
        )
