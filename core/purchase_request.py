"""Purchase-request domain (Phase C of HITL).

Detects when a user query is asking the assistant to **place a spare-part
purchase order** rather than ask a diagnostic question, parses out the
relevant fields (part, qty, vendor, $ amount, urgency), and enriches the
request with whatever the knowledge graph already knows about the part
(equipment that uses it, single-source flag, last-known lead time).

The output is a dict that flows through the LangGraph state and is consumed
by ``core.criticality_classifier`` to decide whether the PO is small enough
for auto-approval or needs to escalate to a buyer.

This module is **deliberately self-contained**: it knows nothing about
LangGraph or FastAPI, so it can be reused (and unit-tested) standalone.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("core.purchase")


# A small set of phrases that strongly signal a purchase intent. Conservative
# on purpose — false positives would route normal diagnostic queries through
# an unnecessary approval gate.
_PURCHASE_TRIGGERS = (
    "purchase",
    "purchase order",
    "po for",
    "raise a po",
    "create a po",
    "buy ",
    "order ",
    "request part",
    "request a part",
    "request spare",
    "procure",
    "stock up",
    "replace ",
)

# Match a part identifier like BRG-7203, P-203-IMP, M8x40-BOLT, etc.
_PART_RE = re.compile(r"\b([A-Z]{2,5}[-_/][A-Z0-9-_/]{2,12})\b")
_QTY_RE = re.compile(r"\b(\d{1,4})\s*(?:x|×|units?|pcs?|each)?\b", re.IGNORECASE)
_DOLLAR_RE = re.compile(r"\$\s*([0-9][0-9,]*(?:\.\d+)?)")
_URGENCY_RE = re.compile(r"\b(urgent|asap|emergency|rush|expedite|priority)\b", re.IGNORECASE)


@dataclass
class PurchaseRequest:
    raw_query: str
    part_id: Optional[str] = None
    quantity: int = 1
    total_usd: Optional[float] = None
    vendor: Optional[str] = None
    urgent: bool = False
    # Enriched-from-KG fields (None when unknown):
    used_by_equipment: List[str] = field(default_factory=list)
    equipment_criticality: Optional[str] = None
    single_source: Optional[bool] = None
    lead_time_days: Optional[int] = None
    last_known_unit_price: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def is_purchase_intent(query: str) -> bool:
    """Cheap, dependency-free check for purchase-request intent."""
    if not query:
        return False
    lower = query.lower()
    return any(t in lower for t in _PURCHASE_TRIGGERS)


def parse(query: str) -> Optional[PurchaseRequest]:
    """Extract a structured ``PurchaseRequest`` from a free-text query.

    Returns ``None`` when no purchase intent is detected.
    """
    if not is_purchase_intent(query):
        return None

    pr = PurchaseRequest(raw_query=query.strip())

    if (m := _PART_RE.search(query)):
        pr.part_id = m.group(1)
    if (m := _QTY_RE.search(query)):
        try:
            pr.quantity = max(1, int(m.group(1)))
        except ValueError:
            pass
    if (m := _DOLLAR_RE.search(query)):
        try:
            pr.total_usd = float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    if _URGENCY_RE.search(query):
        pr.urgent = True

    # Cheap vendor extraction: "from <Vendor>" or "vendor <Name>".
    vendor_match = re.search(
        r"\b(?:from|vendor|supplier)\s+([A-Z][A-Za-z0-9 &\.\-]{2,40})",
        query,
    )
    if vendor_match:
        pr.vendor = vendor_match.group(1).strip(". ,")

    return pr


def enrich_from_kg(pr: PurchaseRequest, knowledge_graph: Any) -> PurchaseRequest:
    """Walk the KG to fill in equipment / vendor / lead-time facts.

    The KG is the project's NetworkX-backed ``KnowledgeGraph`` (we don't
    import the class to keep this module duck-typed). We look for the part
    node by its id or label, then read attributes off it and any
    ``REQUIRES_PART`` / ``SUPPLIED_BY`` neighbours.
    """
    if not pr.part_id or knowledge_graph is None:
        return pr

    graph = getattr(knowledge_graph, "graph", None)
    if graph is None:
        return pr

    candidates: List[str] = []
    for node, data in graph.nodes(data=True):
        label = (data.get("label") or data.get("name") or str(node)).lower()
        if pr.part_id.lower() in label or pr.part_id.lower() in str(node).lower():
            entity_type = (data.get("entity_type") or "").lower()
            if entity_type in ("sparepart", "spare_part", "component"):
                candidates.append(node)

    if not candidates:
        pr.notes.append(f"part_id {pr.part_id} not found in KG")
        return pr

    part_node = candidates[0]
    data = graph.nodes[part_node]

    if "single_source" in data:
        pr.single_source = bool(data["single_source"])
    if "lead_time_days" in data:
        try:
            pr.lead_time_days = int(data["lead_time_days"])
        except (TypeError, ValueError):
            pass
    if "unit_price_usd" in data:
        try:
            pr.last_known_unit_price = float(data["unit_price_usd"])
        except (TypeError, ValueError):
            pass

    # Walk neighbours: SparePart ←REQUIRES_PART— Equipment, SparePart —SUPPLIED_BY→ Vendor
    try:
        neighbours = list(graph.predecessors(part_node)) + list(graph.successors(part_node))
    except Exception:
        neighbours = []

    for nb in neighbours:
        nb_data = graph.nodes.get(nb, {})
        nb_type = (nb_data.get("entity_type") or "").lower()
        if nb_type == "equipment":
            pr.used_by_equipment.append(nb)
            crit = nb_data.get("criticality") or nb_data.get("class")
            if crit and not pr.equipment_criticality:
                pr.equipment_criticality = str(crit).upper()
        elif nb_type == "vendor" and not pr.vendor:
            pr.vendor = nb_data.get("label") or nb_data.get("name") or str(nb)

    # Compute total if we now know unit price
    if pr.total_usd is None and pr.last_known_unit_price is not None:
        pr.total_usd = round(pr.last_known_unit_price * pr.quantity, 2)
        pr.notes.append("total estimated from KG unit price")

    return pr


def detect_and_enrich(query: str, knowledge_graph: Any = None) -> Optional[Dict[str, Any]]:
    """Convenience wrapper used by the orchestrator.

    Returns a serialisable dict (the ``GraphState`` only stores plain types)
    or ``None`` when the query isn't a purchase request.
    """
    pr = parse(query)
    if pr is None:
        return None
    if knowledge_graph is not None:
        pr = enrich_from_kg(pr, knowledge_graph)
    return pr.to_dict()


def format_for_review(pr: Dict[str, Any]) -> str:
    """Render a purchase request as a human-friendly markdown card."""
    lines = ["**Purchase request**"]
    lines.append(f"  • Part: `{pr.get('part_id') or 'unknown'}`")
    lines.append(f"  • Qty: {pr.get('quantity', 1)}")
    if pr.get("total_usd") is not None:
        lines.append(f"  • Total: **${pr['total_usd']:,.2f}**")
    if pr.get("vendor"):
        lines.append(f"  • Vendor: {pr['vendor']}")
    if pr.get("single_source"):
        lines.append(f"  • ⚠ Single-source vendor")
    if pr.get("lead_time_days") is not None:
        lines.append(f"  • Lead time: {pr['lead_time_days']} days")
    if pr.get("equipment_criticality"):
        lines.append(f"  • Equipment criticality: {pr['equipment_criticality']}")
    if pr.get("used_by_equipment"):
        eq = ", ".join(pr["used_by_equipment"][:5])
        lines.append(f"  • Used by: {eq}")
    if pr.get("urgent"):
        lines.append("  • 🚨 Marked urgent")
    if pr.get("notes"):
        lines.append("  • Notes: " + "; ".join(pr["notes"]))
    return "\n".join(lines)
