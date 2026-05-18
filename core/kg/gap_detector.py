"""Schema-driven gap detector — kgrag L7.

Walks the constructed graph against the declared schema and surfaces
gaps that need human review. Designed to back the HITL review flow:

* ``MISSING_EDGE``        — an Equipment with no HAS_COMPONENT (or any
  edge whose source declares ``min_cardinality > 0`` but has zero edges).
* ``CONFLICTING_EDGES``   — more edges than the schema's ``max_cardinality``.
* ``LOW_CONFIDENCE_EDGE`` — an edge whose provenance confidence is below
  ``schema.gap_thresholds['low_confidence']`` (default 0.7) AND that has
  not already been superseded by a human-authored edge.

Each gap carries enough context (source node, target candidates, current
state) for the FastAPI HITL endpoint to render a review card.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx

from core.kg.provenance import Provenance, ProvenanceAuthor
from core.kg.schema import Schema


GAP_MISSING_EDGE = "MISSING_EDGE"
GAP_CONFLICTING_EDGES = "CONFLICTING_EDGES"
GAP_LOW_CONFIDENCE_EDGE = "LOW_CONFIDENCE_EDGE"


@dataclass
class Gap:
    kind: str
    node_id: str
    node_type: str
    relation: str = ""
    summary: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "node_id": self.node_id,
            "node_type": self.node_type,
            "relation": self.relation,
            "summary": self.summary,
            "context": self.context,
        }


def detect_gaps(graph: nx.DiGraph, schema: Schema) -> List[Gap]:
    """Walk ``graph`` against ``schema`` and return every detected gap.

    The output ordering is stable: MISSING_EDGE → CONFLICTING_EDGES →
    LOW_CONFIDENCE_EDGE so a UI can render them in priority order.
    """
    threshold = float(schema.gap_thresholds.get("low_confidence", 0.7))
    out: List[Gap] = []

    # ── Pass 1: cardinality checks per (node, edge type) ────────────────
    for node, data in graph.nodes(data=True):
        node_type = data.get("entity_type", "Unknown")
        out.extend(_check_cardinality(graph, schema, node, node_type))

    # ── Pass 2: low-confidence edges (after cardinality so a missing edge
    # never overlaps with a low-confidence edge on the same triple) ──────
    for u, v, edge_data in graph.edges(data=True):
        prov: Optional[Provenance] = edge_data.get("provenance")
        if prov is None:
            continue
        if prov.is_user_authored:
            continue  # human-resolved; never flag
        if prov.confidence >= threshold:
            continue
        # Skip if a higher-confidence edge (e.g. from a deterministic
        # extractor) already exists on the same triple — the upgrade path
        # in ``_add_relation`` should have replaced this, but if a stale
        # load surfaces both, we don't double-flag.
        out.append(Gap(
            kind=GAP_LOW_CONFIDENCE_EDGE,
            node_id=u,
            node_type=graph.nodes[u].get("entity_type", "Unknown"),
            relation=edge_data.get("relation", ""),
            summary=(
                f"Edge {u} --{edge_data.get('relation')}--> {v} has "
                f"confidence {prov.confidence:.2f} (< {threshold:.2f}). "
                f"Confirm or reject in HITL review."
            ),
            context={
                "target": v,
                "target_type": graph.nodes[v].get("entity_type", "Unknown"),
                "author": prov.author,
                "confidence": prov.confidence,
                "source_chunk_id": prov.source_chunk_id,
            },
        ))

    return out


def _check_cardinality(
    graph: nx.DiGraph,
    schema: Schema,
    node: str,
    node_type: str,
) -> List[Gap]:
    """Per-node cardinality check across every edge type the schema
    declares with this node-type as a source.
    """
    out: List[Gap] = []
    for edge_name, edge_type in schema.edge_types.items():
        if node_type not in edge_type.source:
            continue
        outgoing = [
            (v, data) for _, v, data in graph.out_edges(node, data=True)
            if data.get("relation") == edge_name
        ]
        n = len(outgoing)

        if edge_type.min_cardinality and n < edge_type.min_cardinality:
            out.append(Gap(
                kind=GAP_MISSING_EDGE,
                node_id=node,
                node_type=node_type,
                relation=edge_name,
                summary=(
                    f"{node_type} {node!r} has {n} {edge_name} edge(s); "
                    f"schema requires at least {edge_type.min_cardinality}."
                ),
                context={
                    "missing": edge_type.min_cardinality - n,
                    "expected_target_types": list(edge_type.target),
                },
            ))

        if edge_type.max_cardinality is not None and n > edge_type.max_cardinality:
            out.append(Gap(
                kind=GAP_CONFLICTING_EDGES,
                node_id=node,
                node_type=node_type,
                relation=edge_name,
                summary=(
                    f"{node_type} {node!r} has {n} {edge_name} edge(s); "
                    f"schema allows at most {edge_type.max_cardinality}."
                ),
                context={
                    "extra": n - edge_type.max_cardinality,
                    "edges": [
                        {
                            "target": v,
                            "author": (
                                (data.get("provenance") or Provenance(author="")).author
                            ),
                        }
                        for v, data in outgoing
                    ],
                },
            ))

    return out
