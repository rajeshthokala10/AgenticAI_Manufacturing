"""Tool registry — declarative ERP/MES tools with HITL-aware dispatch.

A *tool* is a named callable with:

* ``name``                — stable identifier the LLM emits in its JSON output.
* ``description``         — human/LLM-readable spec.
* ``parameters``          — JSON-schema-ish dict (compatible with OpenAI
                            ``tools`` function calling).
* ``side_effect``         — one of ``"none"`` (read-only) or ``"write"``.
* ``requires_approval``   — when True, the tool envelope is paused and only
                            executed after a human approves via the existing
                            HITL workflow.
* ``risk_score``          — baseline risk used by the criticality classifier
                            in absence of a query-derived score.

The registry separates *envelope creation* (``prepare``) from
*execution* (``execute``). The orchestrator uses ``prepare`` to assemble a
:class:`ToolCall` it can route through HITL; once approved, it calls
``execute`` to actually hit the backend.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("core.tools.registry")


# ─── Data classes ────────────────────────────────────────────────────────


@dataclass
class ToolDefinition:
    """Static metadata + handler for a single tool."""

    name: str
    description: str
    parameters: Dict[str, Any]
    side_effect: str = "none"  # "none" | "write"
    requires_approval: bool = False
    risk_score: float = 0.0
    handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    def to_openai_schema(self) -> Dict[str, Any]:
        """Return an OpenAI ``tools`` function-calling schema entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    """A pending tool invocation. Stored on the LangGraph state."""

    name: str
    arguments: Dict[str, Any]
    side_effect: str = "none"
    requires_approval: bool = False
    risk_score: float = 0.0
    rationale: str = ""
    call_id: str = field(default_factory=lambda: f"tc_{uuid.uuid4().hex[:10]}")
    issued_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    """Execution outcome — returned to the orchestrator / audit log."""

    call_id: str
    name: str
    status: str  # "ok" | "error" | "rejected"
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─── Backend protocol ────────────────────────────────────────────────────


class ToolBackend:
    """Pluggable backend that can fulfil a tool call.

    Subclass to wire in a real ERP/MES adapter. The default
    :class:`MockToolBackend` provides deterministic stub data so the demo
    UI / eval harness work without external connectivity.
    """

    def execute(self, call: ToolCall) -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError


# ─── Mock backend (default) ──────────────────────────────────────────────


class MockToolBackend(ToolBackend):
    """In-memory stub for ERP / MES / SAP / Database side effects.

    The mock keeps a list of "created" objects in process memory so a
    demo can issue a PO, see it in the audit log, and re-read it via the
    ``list_purchase_orders`` tool.
    """

    def __init__(self) -> None:
        self._inventory: Dict[str, Dict[str, Any]] = {
            "BRG-7203": {"part_id": "BRG-7203", "on_hand": 4, "reorder_point": 6, "unit_price_usd": 180.0},
            "SEAL-22": {"part_id": "SEAL-22", "on_hand": 12, "reorder_point": 4, "unit_price_usd": 45.0},
            "FLT-P3": {"part_id": "FLT-P3", "on_hand": 0, "reorder_point": 2, "unit_price_usd": 60.0},
        }
        self._work_orders: Dict[str, Dict[str, Any]] = {
            "WO-1042": {"work_order_id": "WO-1042", "equipment": "P-203", "status": "scheduled", "owner": "tech.maria"},
        }
        self._purchase_orders: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def execute(self, call: ToolCall) -> ToolResult:
        t0 = time.time()
        try:
            handler = getattr(self, f"_tool_{call.name}", None)
            if handler is None:
                return ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    status="error",
                    error=f"unknown tool: {call.name}",
                    elapsed_ms=(time.time() - t0) * 1000,
                )
            output = handler(call.arguments or {})
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="ok",
                output=output,
                elapsed_ms=(time.time() - t0) * 1000,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Mock tool %s failed", call.name)
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="error",
                error=str(exc),
                elapsed_ms=(time.time() - t0) * 1000,
            )

    # ── Read-only tools ──────────────────────────────────────────────
    def _tool_get_inventory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        part_id = str(args.get("part_id", "")).strip()
        with self._lock:
            if part_id and part_id in self._inventory:
                return {"item": self._inventory[part_id]}
            return {"items": list(self._inventory.values())}

    def _tool_get_work_order_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        wo_id = str(args.get("work_order_id", "")).strip()
        with self._lock:
            wo = self._work_orders.get(wo_id)
            if wo is None:
                return {"found": False, "work_order_id": wo_id}
            return {"found": True, "work_order": wo}

    def _tool_list_purchase_orders(self, _args: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            return {"purchase_orders": list(self._purchase_orders)}

    # ── Write tools ──────────────────────────────────────────────────
    def _tool_create_purchase_order(self, args: Dict[str, Any]) -> Dict[str, Any]:
        po_id = f"PO-{int(time.time() * 1000) % 1_000_000:06d}"
        record = {
            "po_id": po_id,
            "part_id": args.get("part_id"),
            "quantity": int(args.get("quantity", 1) or 1),
            "vendor": args.get("vendor"),
            "total_usd": float(args.get("total_usd", 0.0) or 0.0),
            "urgent": bool(args.get("urgent", False)),
            "created_at": time.time(),
        }
        with self._lock:
            self._purchase_orders.append(record)
        return {"created": record}

    def _tool_create_work_order(self, args: Dict[str, Any]) -> Dict[str, Any]:
        wo_id = f"WO-{int(time.time() * 1000) % 10_000:04d}"
        record = {
            "work_order_id": wo_id,
            "equipment": args.get("equipment"),
            "task": args.get("task"),
            "priority": args.get("priority", "normal"),
            "owner": args.get("owner"),
            "status": "scheduled",
        }
        with self._lock:
            self._work_orders[wo_id] = record
        return {"created": record}


# ─── Registry ────────────────────────────────────────────────────────────


_DEFAULT_TOOLS: List[ToolDefinition] = [
    ToolDefinition(
        name="get_inventory",
        description=(
            "Look up current on-hand inventory for a spare part in the ERP/MES."
            " Returns the matching item when ``part_id`` is provided, or the full list otherwise."
        ),
        parameters={
            "type": "object",
            "properties": {
                "part_id": {"type": "string", "description": "Part identifier, e.g. BRG-7203"},
            },
            "required": [],
        },
        side_effect="none",
        requires_approval=False,
        risk_score=0.0,
    ),
    ToolDefinition(
        name="get_work_order_status",
        description="Fetch the status of an existing MES work order by id.",
        parameters={
            "type": "object",
            "properties": {
                "work_order_id": {"type": "string", "description": "Work order id, e.g. WO-1042"},
            },
            "required": ["work_order_id"],
        },
        side_effect="none",
        requires_approval=False,
        risk_score=0.0,
    ),
    ToolDefinition(
        name="list_purchase_orders",
        description="List all purchase orders created in this session.",
        parameters={"type": "object", "properties": {}, "required": []},
        side_effect="none",
        requires_approval=False,
    ),
    ToolDefinition(
        name="create_purchase_order",
        description=(
            "Create a purchase order in ERP for a spare part. Requires human "
            "approval — the call is routed through the HITL workflow."
        ),
        parameters={
            "type": "object",
            "properties": {
                "part_id": {"type": "string"},
                "quantity": {"type": "integer", "minimum": 1},
                "vendor": {"type": "string"},
                "total_usd": {"type": "number"},
                "urgent": {"type": "boolean"},
            },
            "required": ["part_id", "quantity"],
        },
        side_effect="write",
        requires_approval=True,
        risk_score=0.6,
    ),
    ToolDefinition(
        name="create_work_order",
        description=(
            "Create an MES work order. Requires human approval — the call is "
            "routed through the HITL workflow."
        ),
        parameters={
            "type": "object",
            "properties": {
                "equipment": {"type": "string"},
                "task": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
                "owner": {"type": "string"},
            },
            "required": ["equipment", "task"],
        },
        side_effect="write",
        requires_approval=True,
        risk_score=0.55,
    ),
]


class ToolRegistry:
    """Singleton-style registry that holds tool definitions + backend."""

    def __init__(self, backend: Optional[ToolBackend] = None):
        self._backend: ToolBackend = backend or MockToolBackend()
        self._tools: Dict[str, ToolDefinition] = {t.name: t for t in _DEFAULT_TOOLS}

    # ── Discovery ────────────────────────────────────────────────────
    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def openai_schemas(self) -> List[Dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    # ── Mutation ─────────────────────────────────────────────────────
    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def set_backend(self, backend: ToolBackend) -> None:
        self._backend = backend

    # ── Dispatch ─────────────────────────────────────────────────────
    def prepare(
        self,
        name: str,
        arguments: Dict[str, Any],
        rationale: str = "",
    ) -> ToolCall:
        """Build a :class:`ToolCall` envelope without executing it."""
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        return ToolCall(
            name=tool.name,
            arguments=dict(arguments or {}),
            side_effect=tool.side_effect,
            requires_approval=tool.requires_approval,
            risk_score=tool.risk_score,
            rationale=rationale,
        )

    def execute(self, call: ToolCall) -> ToolResult:
        """Run a previously-prepared :class:`ToolCall` against the backend."""
        if self.get(call.name) is None:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status="error",
                error=f"unknown tool: {call.name}",
            )
        return self._backend.execute(call)


# ─── Module-level singleton ──────────────────────────────────────────────

_REGISTRY: Optional[ToolRegistry] = None
_LOCK = threading.Lock()


def get_registry() -> ToolRegistry:
    """Return the process-wide :class:`ToolRegistry` singleton."""
    global _REGISTRY
    with _LOCK:
        if _REGISTRY is None:
            _REGISTRY = ToolRegistry()
        return _REGISTRY


def set_backend(backend: ToolBackend) -> None:
    """Swap the backend on the global registry (e.g. real SAP adapter)."""
    get_registry().set_backend(backend)
