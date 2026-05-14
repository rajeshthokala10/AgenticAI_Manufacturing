"""Tool-calling agent for ERP / MES / SAP / Database integrations.

The :class:`ToolRegistry` provides a deterministic dispatch layer the
LangGraph orchestrator (or any caller) can use to invoke side-effectful
integrations while still gating them through the existing HITL approval
workflow.

* Read-only tools (``get_inventory``, ``get_work_order_status``) execute
  immediately and return a JSON-serialisable result.
* Write tools (``create_purchase_order``, ``create_work_order``) emit a
  :class:`ToolCall` envelope flagged ``requires_approval=True`` which the
  orchestrator hands to the same ``criticality_check`` →
  ``human_approval`` interrupt used elsewhere. Only after a human signs
  off is :meth:`ToolRegistry.execute` called to actually mutate the
  remote system.

Drop-in mock backends are bundled so the pipeline is demoable without an
ERP/MES connection. Swap them for real implementations by registering a
new ``ToolBackend`` via :func:`set_backend`.
"""

from core.tools.registry import (  # noqa: F401
    ToolBackend,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
    get_registry,
    set_backend,
)
