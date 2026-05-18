"""KnowledgeGraph — three-tier model (schema + instances + provenance).

This module is a thin orchestrator on top of:

* :mod:`core.kg.schema` (L1) — declarative ontology loaded from
  ``schemas/manufacturing.yaml``. Validates every node + edge at build time.
* :mod:`core.kg.extractors` (L3) — three extractors with descending
  confidence: ``CodeExtractor`` (1.0), ``MetadataExtractor`` (0.95),
  ``NarrativeExtractor`` (0.5). Each one stamps its emissions with the
  matching :class:`Provenance`.
* :mod:`core.kg.provenance` — first-class source-tag on every node and
  edge. Lets retrieval prefer high-trust sources and the gap detector
  flag low-confidence edges for HITL review.

Public API is preserved from the legacy single-tier implementation:

* ``build_from_documents(documents)``    — build from ingested chunks
* ``get_allow_list(query, min_confidence=0.0)`` — retrieval pre-filter
* ``get_edge_priors(chunk_ids)``         — RRF score boost
* ``get_subgraph_for_query(query)``      — graph context for the LLM
* ``get_stats()``                        — entity / edge counts
* ``save() / load()``                    — JSON persistence

New surface for the three-tier model:

* ``schema`` attribute → :class:`Schema` in use
* ``node_provenance(node_id)`` / ``edge_provenance(u, v)``
* ``nodes_by_author(...)`` / ``edges_by_author(...)``
* ``record_human_edge(...)``  — HITL writeback (supersedes a system edge)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from config import GRAPH_PATH
from core.kg.extractors import (
    CodeExtractor,
    EdgeCandidate,
    ExtractionResult,
    Extractor,
    KeywordExtractor,
    MetadataExtractor,
    Mention,
    NarrativeExtractor,
)
from core.kg.extractors.base import Mention as _Mention  # for type checker
from core.kg.provenance import Provenance, ProvenanceAuthor
from core.kg.schema import Schema, load_default_schema

logger = logging.getLogger("core.knowledge_graph")


class KnowledgeGraph:
    """Three-tier KG: hand-curated schema, mixed-source instances, provenance-tagged."""

    def __init__(
        self,
        schema: Optional[Schema] = None,
        domain: Optional[str] = None,
    ):
        self.graph = nx.DiGraph()
        self._entity_index: Dict[str, Set[str]] = {}

        # The hand-curated ontology. Anything that doesn't conform is dropped
        # at build time. Override with a custom schema for other domains.
        # ``domain`` selects ``schemas/<domain>.yaml`` when ``schema`` is None.
        self.domain: str = (domain or (schema.domain if schema else None) or "manufacturing")
        self.schema: Schema = schema or load_default_schema(self.domain)

        # Configured extractors, in priority order. Earlier extractors run
        # first so their high-confidence findings win on ties. KeywordExtractor
        # is the generic, schema-vocab-driven tier (kgrag's "config not code"
        # promise) — adding a new domain to ``schemas/manufacturing.yaml`` is
        # all it takes to start populating the graph from new text.
        self.extractors: List[Extractor] = [
            CodeExtractor(self.schema),
            MetadataExtractor(self.schema),
            KeywordExtractor(self.schema),
            NarrativeExtractor(self.schema),
        ]

        # Track rejected items so callers (and tests) can audit what the
        # schema discarded.
        self._rejected: List[Dict[str, Any]] = []

    # ─── Build ───────────────────────────────────────────────────────────

    def build_from_documents(self, documents: List[Dict]) -> None:
        for doc in documents:
            self._ingest_document(doc)
        self._build_entity_index()
        self._compute_edge_priors(documents)
        self.save()

    def _ingest_document(self, doc: Dict) -> None:
        """Run every extractor on ``doc`` and merge the results."""
        merged = ExtractionResult()
        for extractor in self.extractors:
            try:
                merged.extend(extractor.extract(doc))
            except Exception as exc:  # pragma: no cover - extractor bug
                logger.warning(
                    "Extractor %s failed on chunk %s: %s",
                    extractor.__class__.__name__, doc.get("chunk_id"), exc,
                )

        # First add all mentions so edges can find their endpoints.
        accepted_ids: Dict[str, str] = {}  # identifier → entity_type
        for m in merged.mentions:
            if not self.schema.validate_entity(m.identifier, m.entity_type):
                self._rejected.append({
                    "kind": "mention",
                    "identifier": m.identifier,
                    "entity_type": m.entity_type,
                    "author": m.author,
                    "chunk_id": m.source_chunk_id,
                })
                continue
            self._add_entity(m, existing_type=accepted_ids.get(m.identifier))
            accepted_ids[m.identifier] = m.entity_type

        # Then edges, validated against accepted endpoints.
        for e in merged.edges:
            src_type = accepted_ids.get(e.source_id) or self._node_type(e.source_id)
            tgt_type = accepted_ids.get(e.target_id) or self._node_type(e.target_id)
            if src_type is None or tgt_type is None:
                self._rejected.append({
                    "kind": "edge_missing_endpoint",
                    "edge": (e.source_id, e.relation, e.target_id),
                    "author": e.author,
                })
                continue
            if not self.schema.validate_edge(e.relation, src_type, tgt_type):
                self._rejected.append({
                    "kind": "edge",
                    "edge": (e.source_id, e.relation, e.target_id),
                    "source_type": src_type, "target_type": tgt_type,
                    "author": e.author,
                })
                continue
            self._add_relation(e)

    def _add_entity(self, m: Mention, *, existing_type: Optional[str] = None) -> None:
        if self.graph.has_node(m.identifier):
            data = self.graph.nodes[m.identifier]
            data["chunk_ids"].add(m.source_chunk_id)
            # Upgrade provenance if the new mention is more authoritative.
            current = data.get("provenance")
            new_prov = self._mention_provenance(m)
            if current is None or new_prov.confidence > current.confidence:
                data["provenance"] = new_prov
            return

        self.graph.add_node(
            m.identifier,
            entity_type=m.entity_type,
            chunk_ids={m.source_chunk_id},
            provenance=self._mention_provenance(m),
        )

    def _add_relation(self, e: EdgeCandidate) -> None:
        if self.graph.has_edge(e.source_id, e.target_id):
            data = self.graph[e.source_id][e.target_id]
            data["chunk_ids"].add(e.source_chunk_id)
            data["weight"] += 1
            current: Optional[Provenance] = data.get("provenance")
            new_prov = self._edge_provenance(e)
            # Upgrade if higher confidence or if a human edge supersedes a
            # system one.
            if current is None or new_prov.confidence > current.confidence:
                data["provenance"] = new_prov
            return

        self.graph.add_edge(
            e.source_id, e.target_id,
            relation=e.relation,
            chunk_ids={e.source_chunk_id},
            weight=1,
            provenance=self._edge_provenance(e),
        )

    @staticmethod
    def _mention_provenance(m: Mention) -> Provenance:
        return Provenance(
            author=m.author or ProvenanceAuthor.METADATA,
            confidence=float(m.confidence),
            source_chunk_id=m.source_chunk_id,
        )

    @staticmethod
    def _edge_provenance(e: EdgeCandidate) -> Provenance:
        return Provenance(
            author=e.author or ProvenanceAuthor.METADATA,
            confidence=float(e.confidence),
            source_chunk_id=e.source_chunk_id,
        )

    def _node_type(self, node_id: str) -> Optional[str]:
        if not self.graph.has_node(node_id):
            return None
        return self.graph.nodes[node_id].get("entity_type")

    def _build_entity_index(self) -> None:
        self._entity_index = {}
        for node, data in self.graph.nodes(data=True):
            etype = data.get("entity_type", "Unknown")
            self._entity_index.setdefault(etype, set()).add(node)

    def _compute_edge_priors(self, documents: List[Dict]) -> None:
        total_docs = len(documents)
        for u, v, data in self.graph.edges(data=True):
            co_occurrence = len(data.get("chunk_ids", set()))
            data["prior"] = min(co_occurrence / max(total_docs * 0.01, 1), 1.0)

    # ─── Retrieval-facing API (preserved + extended) ─────────────────────

    def get_allow_list(self, query: str, *, min_confidence: float = 0.0) -> Set[str]:
        """Return the chunk-id allow-list for ``query``, optionally gated
        by the provenance confidence of the visited nodes.

        ``min_confidence=0.0`` (default) matches the legacy behaviour:
        every entity match contributes. Raise it to e.g. 0.95 to restrict
        to authoritative / structured sources.
        """
        query_entities = self._match_query_to_entities(query)
        allowed_chunks: Set[str] = set()

        def _contributes(node_id: str) -> bool:
            prov: Optional[Provenance] = self.graph.nodes[node_id].get("provenance")
            if prov is None:
                return min_confidence <= 0.0
            return prov.confidence >= min_confidence

        for entity_id in query_entities:
            if entity_id not in self.graph:
                continue
            if _contributes(entity_id):
                allowed_chunks.update(self.graph.nodes[entity_id].get("chunk_ids", set()))
            for neighbor in self.graph.neighbors(entity_id):
                if _contributes(neighbor):
                    allowed_chunks.update(self.graph.nodes[neighbor].get("chunk_ids", set()))
                    for second_hop in self.graph.neighbors(neighbor):
                        if _contributes(second_hop):
                            allowed_chunks.update(
                                self.graph.nodes[second_hop].get("chunk_ids", set())
                            )

        for entity_id in query_entities:
            if entity_id in self.graph:
                for predecessor in self.graph.predecessors(entity_id):
                    if _contributes(predecessor):
                        allowed_chunks.update(
                            self.graph.nodes[predecessor].get("chunk_ids", set())
                        )

        return allowed_chunks

    def get_edge_priors(self, chunk_ids: Set[str]) -> Dict[str, float]:
        priors: Dict[str, float] = {}
        for u, v, data in self.graph.edges(data=True):
            edge_chunks = data.get("chunk_ids", set())
            overlap = edge_chunks & chunk_ids
            if overlap:
                prior = data.get("prior", 0.0)
                for cid in overlap:
                    priors[cid] = max(priors.get(cid, 0.0), prior)
        return priors

    def get_subgraph_for_query(self, query: str) -> Dict:
        entities = self._match_query_to_entities(query)
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        visited: Set[str] = set()

        for eid in entities:
            if eid not in self.graph or eid in visited:
                continue
            visited.add(eid)
            nodes.append(self._node_summary(eid))
            for neighbor in list(self.graph.neighbors(eid)) + list(self.graph.predecessors(eid)):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                nodes.append(self._node_summary(neighbor))

            for neighbor in self.graph.neighbors(eid):
                edges.append(self._edge_summary(eid, neighbor))
            for predecessor in self.graph.predecessors(eid):
                edges.append(self._edge_summary(predecessor, eid))

        return {"nodes": nodes, "edges": edges}

    def _node_summary(self, node_id: str) -> Dict[str, Any]:
        data = self.graph.nodes[node_id]
        prov: Optional[Provenance] = data.get("provenance")
        return {
            "id": node_id,
            "type": data.get("entity_type", "Unknown"),
            "chunks": len(data.get("chunk_ids", set())),
            "confidence": prov.confidence if prov else None,
            "author": prov.author if prov else None,
        }

    def _edge_summary(self, u: str, v: str) -> Dict[str, Any]:
        data = self.graph[u][v]
        prov: Optional[Provenance] = data.get("provenance")
        return {
            "source": u,
            "target": v,
            "relation": data.get("relation", "RELATED"),
            "weight": data.get("weight", 1),
            "confidence": prov.confidence if prov else None,
            "author": prov.author if prov else None,
        }

    # ─── Query-to-entity matching ────────────────────────────────────────

    def _match_query_to_entities(self, query: str) -> Set[str]:
        matched: Set[str] = set()

        # Code-pattern lookups — re-use the schema's id_patterns so this
        # is automatically domain-agnostic.
        for type_name, et in self.schema.entity_types.items():
            if et.id_pattern is None:
                continue
            for m in et.id_pattern.finditer(query):
                ident = m.group(0)
                if not et.case_sensitive:
                    ident = ident.upper()
                matched.add(ident)
            # Inline-wrap so we also catch IDs not anchored at line boundaries.
            try:
                body = et.id_pattern.pattern.strip("^$")
                inline = re.compile(rf"\b(?:{body})\b", et.id_pattern.flags)
                for m in inline.finditer(query):
                    ident = m.group(0)
                    if not et.case_sensitive:
                        ident = ident.upper()
                    matched.add(ident)
            except re.error:
                pass

        # Closed-vocabulary lookups (Component, Cause, …) — the identifier
        # IS the vocabulary term, so a direct node-id check works.
        query_lower = query.lower()
        for type_name, et in self.schema.entity_types.items():
            if not et.vocabulary:
                continue
            for term in et.vocabulary:
                if term.lower() not in query_lower:
                    continue
                if term in self.graph:
                    matched.add(term)

        # Symptom keyword → existing Symptom node (kept from the legacy code
        # for the common case "vibration"/"leak"/"overheat" etc.).
        symptom_keywords = {
            "vibration", "leak", "overheat", "cavitation", "noise",
            "pressure", "tracking", "speed", "temperature", "flow",
        }
        for kw in symptom_keywords:
            if kw not in query_lower:
                continue
            for node in self.graph.nodes():
                if node.startswith("SYM:") and kw in node.lower():
                    matched.add(node)

        return matched

    # ─── Stats / introspection ───────────────────────────────────────────

    def get_stats(self) -> Dict:
        type_counts: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            etype = data.get("entity_type", "Unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1

        rel_counts: Dict[str, int] = {}
        author_counts: Dict[str, int] = {}
        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation", "RELATED")
            rel_counts[rel] = rel_counts.get(rel, 0) + 1
            prov: Optional[Provenance] = data.get("provenance")
            if prov is not None:
                author_counts[prov.author] = author_counts.get(prov.author, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "entity_types": type_counts,
            "relation_types": rel_counts,
            "edge_authors": author_counts,
            "rejected_count": len(self._rejected),
        }

    def rejected(self) -> List[Dict[str, Any]]:
        """Return the list of mentions / edges the schema dropped at
        build time. Useful for diagnosing extractor / schema drift.
        """
        return list(self._rejected)

    # ─── Provenance-aware lookup helpers ─────────────────────────────────

    def node_provenance(self, node_id: str) -> Optional[Provenance]:
        if not self.graph.has_node(node_id):
            return None
        return self.graph.nodes[node_id].get("provenance")

    def edge_provenance(self, u: str, v: str) -> Optional[Provenance]:
        if not self.graph.has_edge(u, v):
            return None
        return self.graph[u][v].get("provenance")

    def nodes_by_author(self, *authors: str) -> List[str]:
        wanted = set(authors)
        out: List[str] = []
        for node, data in self.graph.nodes(data=True):
            prov: Optional[Provenance] = data.get("provenance")
            if prov is not None and prov.author in wanted:
                out.append(node)
        return out

    def edges_by_author(self, *authors: str) -> List[Tuple[str, str]]:
        wanted = set(authors)
        out: List[Tuple[str, str]] = []
        for u, v, data in self.graph.edges(data=True):
            prov: Optional[Provenance] = data.get("provenance")
            if prov is not None and prov.author in wanted:
                out.append((u, v))
        return out

    def record_human_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        *,
        user_id: str,
        confidence: float = 1.0,
        notes: str = "",
    ) -> bool:
        """HITL writeback — stamp a human-authored edge that supersedes
        any existing low-confidence edge between the same endpoints.

        Returns True on success, False if the schema rejects the edge.
        Designed to back ``POST /api/kg/edges`` once the FastAPI route
        for HITL gap-resolution lands.
        """
        src_type = self._node_type(source_id)
        tgt_type = self._node_type(target_id)
        if not src_type or not tgt_type:
            return False
        if not self.schema.validate_edge(relation, src_type, tgt_type):
            return False

        existing_id: Optional[str] = None
        if self.graph.has_edge(source_id, target_id):
            existing_id = f"{source_id}->{target_id}#{relation}"

        prov = Provenance(
            author=ProvenanceAuthor.user(user_id),
            confidence=float(confidence),
            supersedes=existing_id,
            notes=notes,
        )
        if self.graph.has_edge(source_id, target_id):
            self.graph[source_id][target_id]["relation"] = relation
            self.graph[source_id][target_id]["provenance"] = prov
            self.graph[source_id][target_id]["weight"] = max(
                self.graph[source_id][target_id].get("weight", 1), 1
            )
        else:
            self.graph.add_edge(
                source_id, target_id,
                relation=relation,
                chunk_ids=set(),
                weight=1,
                prior=0.0,
                provenance=prov,
            )
        return True

    # ─── Persistence ─────────────────────────────────────────────────────

    def _graph_path(self) -> Path:
        """Per-domain KG file. Falls back to the legacy single-file path
        when the domain is unset (back-compat)."""
        from config import kg_path
        if self.domain:
            return Path(kg_path(self.domain))
        return Path(GRAPH_PATH)

    def save(self) -> None:
        graph_path = self._graph_path()
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        save_data: Dict[str, Any] = {"nodes": {}, "edges": []}
        for node, data in self.graph.nodes(data=True):
            prov: Optional[Provenance] = data.get("provenance")
            save_data["nodes"][node] = {
                "entity_type": data.get("entity_type", "Unknown"),
                "chunk_ids": list(data.get("chunk_ids", set())),
                "provenance": prov.to_dict() if prov else None,
            }
        for u, v, data in self.graph.edges(data=True):
            prov = data.get("provenance")
            save_data["edges"].append({
                "source": u,
                "target": v,
                "relation": data.get("relation", "RELATED"),
                "weight": data.get("weight", 1),
                "prior": data.get("prior", 0.0),
                "chunk_ids": list(data.get("chunk_ids", set())),
                "provenance": prov.to_dict() if prov else None,
            })
        graph_path.write_text(json.dumps(save_data, indent=2))

    def load(self) -> bool:
        graph_path = self._graph_path()
        if not graph_path.exists():
            return False
        save_data = json.loads(graph_path.read_text())
        self.graph = nx.DiGraph()
        for node_id, data in save_data["nodes"].items():
            self.graph.add_node(
                node_id,
                entity_type=data["entity_type"],
                chunk_ids=set(data.get("chunk_ids", [])),
                provenance=Provenance.from_dict(data.get("provenance")),
            )
        for edge in save_data["edges"]:
            self.graph.add_edge(
                edge["source"], edge["target"],
                relation=edge["relation"],
                weight=edge["weight"],
                prior=edge.get("prior", 0.0),
                chunk_ids=set(edge.get("chunk_ids", [])),
                provenance=Provenance.from_dict(edge.get("provenance")),
            )
        self._build_entity_index()
        return True
