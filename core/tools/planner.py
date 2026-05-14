"""Tool-call planner — decide which ERP/MES tool (if any) the answer needs.

We treat this as a *small structured decision*, not a free-form agent
loop, so it stays cheap and predictable:

1. Local-first rule heuristics (regex over the query). For obvious cases
   like "what's the on-hand inventory of BRG-7203?" we skip the LLM
   entirely.
2. Fall back to a cheap LLM call (defaults to the local Qwen on Ollama)
   that returns a strict JSON envelope ``{tool, arguments, rationale}``.

The output is a list of :class:`ToolCall` envelopes (possibly empty). The
orchestrator then:

* Executes ``requires_approval=False`` tools immediately and folds the
  output into the evidence pack the answer LLM sees.
* Pauses on the existing ``human_approval`` interrupt for
  ``requires_approval=True`` tools and only executes them once approved.

The planner never executes tools itself — that is the orchestrator's job.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from core.tools.registry import ToolCall, ToolRegistry, get_registry

logger = logging.getLogger("core.tools.planner")


PLANNER_SYSTEM_PROMPT = """You are a tool-routing planner for a manufacturing copilot.
Given a user query and a list of available tools, decide which (if any) tools should be invoked.

RULES:
1. Return STRICT JSON. Top-level key "calls" is a list of {"tool": str, "arguments": object, "rationale": str}.
2. Only choose tools from the provided list. Never invent tool names.
3. For purely informational queries (e.g. "what causes vibration?") return {"calls": []}.
4. Use side-effect ("write") tools ONLY when the user clearly asks for an action — never speculatively.
5. Prefer at most ONE write tool per response.
6. No prose before or after the JSON. No markdown fences."""


# ── Rule heuristics that bypass the LLM ────────────────────────────────

_PART_ID_RE = re.compile(r"\b([A-Z]{2,5}[-_/][A-Z0-9-_/]{2,12})\b")
_WORK_ORDER_RE = re.compile(r"\b(WO-?\d{2,6})\b", re.IGNORECASE)

_INVENTORY_TRIGGERS = (
    "inventory", "on-hand", "on hand", "in stock", "stock level", "stock of",
    "do we have", "how many", "available units",
)
_WORK_ORDER_TRIGGERS = (
    "work order", "wo status", "status of wo", "status of work order",
)


def _rule_based_calls(query: str, registry: ToolRegistry) -> List[ToolCall]:
    """Cheap rule-based shortcut for the obvious cases."""
    if not query:
        return []
    lower = query.lower()
    out: List[ToolCall] = []

    if any(t in lower for t in _INVENTORY_TRIGGERS) and registry.get("get_inventory"):
        m = _PART_ID_RE.search(query)
        args: Dict[str, Any] = {}
        if m:
            args["part_id"] = m.group(1)
        out.append(
            registry.prepare(
                "get_inventory",
                args,
                rationale="rule:inventory-lookup",
            )
        )

    if any(t in lower for t in _WORK_ORDER_TRIGGERS) and registry.get("get_work_order_status"):
        m = _WORK_ORDER_RE.search(query)
        if m:
            out.append(
                registry.prepare(
                    "get_work_order_status",
                    {"work_order_id": m.group(1).upper()},
                    rationale="rule:work-order-status",
                )
            )

    return out


# ── LLM planner ────────────────────────────────────────────────────────


def _llm_plan(query: str, registry: ToolRegistry, model: Optional[str]) -> List[ToolCall]:
    """Ask a cheap LLM to pick tools. Returns [] on any failure."""
    try:
        from core.llm_client import call_llm  # local import to avoid cycle
    except Exception:  # pragma: no cover
        return []

    tools_descriptor = []
    for t in registry.list_tools():
        tools_descriptor.append(
            {
                "name": t.name,
                "description": t.description,
                "side_effect": t.side_effect,
                "requires_approval": t.requires_approval,
                "parameters": t.parameters,
            }
        )

    user_prompt = (
        f"USER QUERY:\n{query}\n\n"
        f"AVAILABLE TOOLS:\n{json.dumps(tools_descriptor, indent=2)}\n\n"
        "Respond with the JSON envelope described in the system prompt."
    )

    try:
        raw = call_llm(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=model,
            temperature=0.0,
            max_tokens=400,
        )
    except Exception as exc:  # pragma: no cover - network / API
        logger.warning("Tool planner LLM call failed: %s", exc)
        return []

    cleaned = _strip_fences(raw or "")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    calls_raw = data.get("calls") if isinstance(data, dict) else None
    if not isinstance(calls_raw, list):
        return []

    out: List[ToolCall] = []
    for item in calls_raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool") or "").strip()
        if not name or registry.get(name) is None:
            continue
        args = item.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        rationale = str(item.get("rationale") or "").strip()
        try:
            out.append(registry.prepare(name, args, rationale=f"llm:{rationale[:120]}"))
        except ValueError:
            continue
    return out


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


# ── Public entry point ────────────────────────────────────────────────


def plan_tool_calls(
    query: str,
    *,
    intent: Optional[str] = None,
    use_llm: bool = True,
    model: Optional[str] = None,
    registry: Optional[ToolRegistry] = None,
) -> List[ToolCall]:
    """Return the list of tool envelopes for the query (possibly empty)."""
    reg = registry or get_registry()

    calls = _rule_based_calls(query, reg)
    if calls:
        logger.info("Tool planner (rules): %d call(s)", len(calls))
        return calls

    if not use_llm:
        return []

    # Skip the LLM round-trip for clearly diagnostic intents.
    if intent and any(
        t in intent.lower() for t in ("troubleshoot", "diagnos", "lookup", "specification")
    ):
        # diagnostic queries rarely need ERP write actions — still allow
        # read-only inventory checks via the rule heuristic above
        return calls

    return _llm_plan(query, reg, model)


def split_pending_calls(calls: List[ToolCall]) -> Dict[str, List[ToolCall]]:
    """Bucket tool calls into ``read`` (run now) and ``write`` (HITL gate)."""
    return {
        "read": [c for c in calls if not c.requires_approval],
        "write": [c for c in calls if c.requires_approval],
    }
