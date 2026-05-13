"""
Conversational chat agent for the Manufacturing Hybrid GraphRAG pipeline.

Implements a ChatGPT-style multi-turn UX on top of `ManufacturingPipeline`:

  1. Auto-corrects the user's input using the manufacturing vocabulary
     (spelling fixes, acronym expansion, domain synonyms).
  2. Runs the ClarifierAgent (intent + entity + slot filling).
  3. If required slots are missing, asks ONE follow-up question per turn
     until enough context is gathered (or the user types `skip`).
  4. Optionally asks a single high-value optional slot for better grounding.
  5. Runs the pipeline:
        - `diagnostic()`     when the LLM stack is enabled
        - `quick_search()`   as a retrieval-only fallback
  6. Surfaces evidence, intent, entities, critic verdict, and metrics.

The agent is UI-agnostic — `app.py` just persists a `ChatState` in
`st.session_state` and feeds new user messages into `ChatAgent.handle`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from doc_pipeline.clarifier_agent import ClarifierResult, Slot
from doc_pipeline.query_correction import CorrectedQuery

from .unified_pipeline import ManufacturingPipeline, PipelineResult

logger = logging.getLogger("pipeline.chat")


# ── Conversation state ─────────────────────────────────────────────────────

Role = Literal["user", "assistant", "system"]
TurnKind = Literal["text", "correction", "clarify", "answer", "system", "approval_pending", "approval_resolved"]


@dataclass
class ChatTurn:
    """A single message in the conversation transcript."""
    role: Role
    content: str
    kind: TurnKind = "text"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatState:
    """Per-session conversation state held in `st.session_state`."""
    turns: List[ChatTurn] = field(default_factory=list)

    # Active multi-turn slot-filling context
    accumulated_query: str = ""
    pending_required: List[Slot] = field(default_factory=list)
    pending_optional: List[Slot] = field(default_factory=list)
    asked_optional: int = 0
    awaiting_slot: Optional[Slot] = None

    # Reference to the last completed pipeline result (for rendering panels)
    last_result: Optional[PipelineResult] = None

    # HITL: when the last query paused at an interrupt, the chat tab needs to
    # show a banner / disable input until the approval is resolved.
    pending_approval_thread_id: Optional[str] = None

    def reset(self) -> None:
        self.turns.clear()
        self.clear_inflight()
        self.last_result = None
        self.pending_approval_thread_id = None

    def clear_inflight(self) -> None:
        self.accumulated_query = ""
        self.pending_required = []
        self.pending_optional = []
        self.asked_optional = 0
        self.awaiting_slot = None


# ── Chat agent ─────────────────────────────────────────────────────────────

class ChatAgent:
    """Stateless coordinator that mutates a `ChatState` per user turn."""

    SKIP_TOKENS = {
        "skip", "n/a", "none", "no idea", "dunno", "don't know",
        "any", "anything", "not sure", "idk", "pass", "-",
    }
    RESET_TOKENS = {"/reset", "/new", "/clear", "new chat", "reset"}

    def __init__(self, pipe: ManufacturingPipeline, max_optional_asks: int = 1):
        self.pipe = pipe
        self.max_optional_asks = max_optional_asks

    # ── Public entry point ────────────────────────────────────────────────

    def handle(self, state: ChatState, message: str) -> ChatState:
        """Process a new user message and update the conversation state."""
        msg = (message or "").strip()
        if not msg:
            return state

        state.turns.append(ChatTurn(role="user", content=msg))

        if msg.lower() in self.RESET_TOKENS:
            state.reset()
            state.turns.append(ChatTurn(
                role="assistant",
                content="Started a new conversation. Ask me anything about your manufacturing operations.",
                kind="system",
            ))
            return state

        # Mid-clarification → treat reply as a slot answer.
        if state.awaiting_slot is not None:
            return self._answer_slot(state, msg)

        return self._new_question(state, msg)

    # ── Question handling ────────────────────────────────────────────────

    def _new_question(self, state: ChatState, msg: str) -> ChatState:
        correction = self.pipe.query_corrector.correct(msg)

        spelling_fixes = [c for c in correction.corrections_applied if c.startswith("spelling:")]
        acronym_fixes = [c for c in correction.corrections_applied if c.startswith("acronym:")]

        # Only surface visible corrections when spelling actually changed the text.
        if spelling_fixes:
            note = self._format_correction_note(correction, spelling_fixes, acronym_fixes)
            state.turns.append(ChatTurn(
                role="assistant", content=note, kind="correction",
                meta={"spelling_fixes": spelling_fixes, "acronym_fixes": acronym_fixes},
            ))
            corrected = correction.corrected
        else:
            # Even without spelling changes, the cleaned version (no bracketed expansions)
            # is what we treat as the canonical query.
            corrected = msg

        clar = self.pipe.clarifier.analyze(corrected)

        state.accumulated_query = corrected
        state.pending_required = list(clar.missing_required_slots)
        state.pending_optional = self._prioritise_optional(clar.missing_optional_slots)
        state.asked_optional = 0

        return self._ask_next_or_run(state, clar)

    def _answer_slot(self, state: ChatState, msg: str) -> ChatState:
        slot = state.awaiting_slot
        state.awaiting_slot = None
        is_skip = msg.lower().strip(".!? ") in self.SKIP_TOKENS

        if not is_skip and slot is not None:
            state.accumulated_query = f"{state.accumulated_query} {msg}".strip()

        clar = self.pipe.clarifier.analyze(state.accumulated_query)
        return self._ask_next_or_run(state, clar)

    def _ask_next_or_run(self, state: ChatState, clar: ClarifierResult) -> ChatState:
        # 1. Required slots first — only ask if still missing after re-analysis.
        for slot in list(state.pending_required):
            if self._slot_is_filled(clar, slot):
                state.pending_required.remove(slot)
                continue
            state.pending_required.remove(slot)
            return self._ask_slot(state, slot, required=True)

        # 2. One (or a few) optional slot(s) for stronger grounding.
        while (state.pending_optional and state.asked_optional < self.max_optional_asks):
            slot = state.pending_optional.pop(0)
            if self._slot_is_filled(clar, slot):
                continue
            state.asked_optional += 1
            return self._ask_slot(state, slot, required=False)

        # 3. Everything we need — run the pipeline.
        return self._finalize_and_run(state, clar)

    def _ask_slot(self, state: ChatState, slot: Slot, required: bool) -> ChatState:
        state.awaiting_slot = slot
        prefix = "I need one more detail" if required else "Quick optional detail"
        body = (
            f"{prefix} to give you a precise answer:\n\n"
            f"**{slot.prompt}**\n\n"
            f"_Type `skip` if you'd rather I proceed with what we have._"
        )
        state.turns.append(ChatTurn(
            role="assistant", content=body, kind="clarify",
            meta={"slot": slot.name, "required": required, "prompt": slot.prompt},
        ))
        return state

    def _finalize_and_run(self, state: ChatState, clar: ClarifierResult) -> ChatState:
        query = state.accumulated_query
        if not query:
            return state

        try:
            result = self._run(query)
        except Exception as exc:
            logger.exception("Pipeline failure on query %r", query)
            state.turns.append(ChatTurn(
                role="assistant",
                content=f"⚠️ I hit an error while answering: `{exc}`. Try rephrasing or "
                        f"check the pipeline logs.",
                kind="system",
            ))
            state.clear_inflight()
            return state

        state.last_result = result
        return self._render_result(state, query, result)

    def _render_result(self, state: ChatState, query: str, result: PipelineResult) -> ChatState:
        # HITL: graph paused at an interrupt — surface an approval banner and
        # remember the thread_id so the UI can disable input until resolved.
        if result.requires_approval:
            state.pending_approval_thread_id = result.approval_thread_id
            risk = result.risk or {}
            drivers = risk.get("drivers", []) or []
            score = risk.get("score", 0.0)
            payload = result.interrupt_payload or {}
            domain = payload.get("domain", "diagnostic")
            body_lines = [
                "🛑 **This action needs supervisor approval before I proceed.**",
                "",
                f"Risk score: **{score:.2f}** · domain: `{domain}`",
            ]
            if drivers:
                body_lines.append("Drivers: " + ", ".join(f"`{d}`" for d in drivers[:6]))
            if payload.get("purchase_request_card"):
                body_lines.append("")
                body_lines.append(payload["purchase_request_card"])
            body_lines.append("")
            body_lines.append(
                "_Open the **📋 Approvals** tab to approve, reject, or edit before I send the answer._"
            )
            state.turns.append(ChatTurn(
                role="assistant",
                content="\n".join(body_lines),
                kind="approval_pending",
                meta={
                    "thread_id": result.approval_thread_id,
                    "risk": risk,
                    "domain": domain,
                    "proposed_answer": result.answer,
                    "evidence": result.evidence,
                    "purchase_request": result.purchase_request,
                    "interrupt_payload": payload,
                    "corrected_query": query,
                },
            ))
            state.clear_inflight()
            return state

        answer = result.answer or self._synthesize_answer(result)
        if result.rejected:
            decision = result.human_decision or {}
            comments = decision.get("comments") or "_no reason provided_"
            answer = (
                "❌ **Action rejected by the human reviewer.** "
                f"_Reviewer:_ `{decision.get('approver', 'unknown')}`. "
                f"_Comments:_ {comments}\n\n"
                "Try rephrasing or escalate offline."
            )

        state.turns.append(ChatTurn(
            role="assistant",
            content=answer,
            kind="answer",
            meta={
                "evidence": result.evidence,
                "graph_context": result.graph_context,
                "metrics": result.metrics,
                "critic": result.critic,
                "clarification": result.clarification,
                "correction": result.correction,
                "mode": result.mode,
                "corrected_query": query,
                "risk": result.risk,
                "human_decision": result.human_decision,
                "purchase_request": result.purchase_request,
                "rejected": result.rejected,
            },
        ))

        state.pending_approval_thread_id = None
        state.clear_inflight()
        return state

    def apply_resolution(
        self,
        state: ChatState,
        thread_id: str,
        decision: Dict[str, Any],
    ) -> ChatState:
        """Resume a paused diagnostic graph and append the resolved answer.

        Called by the Streamlit "📋 Approvals" tab and by FastAPI's
        ``/api/approvals/{thread_id}/resume`` endpoint.
        """
        try:
            result = self.pipe.resume_diagnostic(thread_id, decision)
        except Exception as exc:
            logger.exception("Resume failure on thread %s", thread_id)
            state.turns.append(ChatTurn(
                role="assistant",
                content=f"⚠️ I couldn't resume the paused workflow: `{exc}`.",
                kind="system",
            ))
            state.pending_approval_thread_id = None
            return state
        state.last_result = result
        # Find the original query that was paused (for rendering meta).
        query = (result.query or "").strip()
        return self._render_result(state, query, result)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _run(self, query: str) -> PipelineResult:
        if self.pipe.llm_enabled:
            return self.pipe.diagnostic(query)
        return self.pipe.quick_search(query, top_k=5, use_context_window=True)

    @staticmethod
    def _slot_is_filled(clar: ClarifierResult, slot: Slot) -> bool:
        for s in clar.slots:
            if s.name == slot.name:
                return s.filled
        return False

    @staticmethod
    def _prioritise_optional(slots: List[Slot]) -> List[Slot]:
        """Reorder optional slots so high-value ones (time / equipment / plant) come first."""
        priority = {
            "time_period": 0, "time_range": 0, "equipment": 1,
            "plant": 2, "department": 3, "severity": 4, "metric": 5,
        }
        return sorted(slots, key=lambda s: priority.get(s.name, 9))

    @staticmethod
    def _format_correction_note(
        correction: CorrectedQuery,
        spelling_fixes: List[str],
        acronym_fixes: List[str],
    ) -> str:
        lines = [
            f"_I read that as:_ **{correction.corrected.split(' [')[0]}**",
        ]
        details = []
        for f in spelling_fixes[:4]:
            details.append("· " + f.replace("spelling:", "✏️").strip())
        if acronym_fixes:
            acronyms = ", ".join(f.split("=")[0].replace("acronym:", "").strip() for f in acronym_fixes[:5])
            details.append(f"· 📖 expanded acronyms: {acronyms}")
        if details:
            lines.append("\n" + "\n".join(details))
        return "\n".join(lines)

    @staticmethod
    def _synthesize_answer(result: PipelineResult) -> str:
        """Build a brief answer from top evidence when no LLM is available."""
        if not result.evidence:
            return (
                "_I couldn't find anything relevant in the indexed documents. "
                "Try rephrasing, mention a specific equipment ID (e.g. P-203, CNC-A-004), "
                "or check that the source files are loaded._"
            )
        lines = ["Here's what I found in the source documents:\n"]
        for i, ev in enumerate(result.evidence[:5], 1):
            meta = ev.get("metadata", {}) or {}
            src = Path(str(meta.get("source", meta.get("source_file", "?")))).name
            page = meta.get("page")
            sheet = meta.get("sheet_name")
            loc = []
            if page:  loc.append(f"p.{page}")
            if sheet: loc.append(f"sheet `{sheet}`")
            loc_str = f" ({', '.join(loc)})" if loc else ""

            text = (ev.get("text") or "").strip().replace("\n", " ")
            if len(text) > 320:
                cut = text[:320].rfind(". ")
                text = text[:cut + 1 if cut > 200 else 320] + " …"
            lines.append(f"**{i}. `{src}`{loc_str}**\n> {text}\n")
        lines.append("\n_(LLM disabled — set `OPENAI_API_KEY` for grounded, summarised answers.)_")
        return "\n".join(lines)
