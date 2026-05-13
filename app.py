"""
Manufacturing Hybrid GraphRAG — unified Streamlit application.

Single entry point exposing every stage of the pipeline:

    streamlit run app.py

Tabs:
    1. Quick Search        — Clarifier + FAISS retrieval (no LLM required)
    2. Diagnostic Copilot  — Hybrid retrieval + LLM answer + Critic loop
    3. Pipeline Comparison — Direct LLM vs Classical RAG vs Hybrid GraphRAG
    4. Knowledge Graph     — Explore entities & relations
    5. Pipeline Flow       — Architecture & cost analysis
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import LLM_MODEL, EMBEDDING_MODEL, USE_HITL, USE_LANGGRAPH, llm_available
from pipeline import ChatAgent, ChatState, ManufacturingPipeline
from utils.metrics import (
    compute_accuracy_estimates, compute_cost_projection,
    format_cost, format_latency,
)


# ── Page setup ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manufacturing Hybrid GraphRAG",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    .main-header {
        font-size: 2.3rem; font-weight: 700;
        background: linear-gradient(90deg, #1E2761, #408EC6, #7A2048);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header { color: #5c6bc0; font-size: 1.05rem; margin-top: 0; }

    .pass-badge { background:#d4edda; color:#155724; padding:4px 12px; border-radius:12px; font-weight:600; }
    .fail-badge { background:#f8d7da; color:#721c24; padding:4px 12px; border-radius:12px; font-weight:600; }
    .skip-badge { background:#fff3cd; color:#856404; padding:4px 12px; border-radius:12px; font-weight:600; }

    .pipeline-header { font-size:1.1rem; font-weight:600; padding:.5rem 1rem; border-radius:8px; margin-bottom:.5rem; }
    .direct-llm     { background:#f8d7da; color:#721c24; }
    .classical-rag  { background:#fff3cd; color:#856404; }
    .hybrid-graphrag{ background:#d4edda; color:#155724; }

    .entity-tag {
        display:inline-block; background:#e8eaf6; color:#283593;
        padding:2px 10px; border-radius:12px; margin:2px 4px; font-size:.85rem;
    }
    .intent-tag {
        display:inline-block; background:#283593; color:white;
        padding:4px 14px; border-radius:16px; font-size:.9rem; font-weight:600;
    }
    .slot-filled { color:#2e7d32; }
    .slot-missing { color:#c62828; }
    .result-card {
        background:#f5f5f5; border-left:4px solid #283593;
        padding:12px 16px; margin:8px 0; border-radius:4px;
    }
    .metric-box { background:#e8eaf6; border-radius:8px; padding:16px; text-align:center; }
    .metric-num { font-size:1.6rem; font-weight:700; color:#283593; }
    .metric-label { font-size:.85rem; color:#5c6bc0; }

    /* Chat */
    .chat-meta-chip {
        display:inline-block; background:#eef2ff; color:#3949ab;
        padding:2px 9px; border-radius:10px; font-size:.75rem;
        margin:1px 4px 1px 0;
    }
    .chat-intent {
        display:inline-block; background:#283593; color:#fff;
        padding:2px 10px; border-radius:10px; font-size:.75rem; font-weight:600;
        margin-right:6px;
    }
    .chat-correction {
        background:#fffde7; border-left:4px solid #f9a825;
        padding:8px 12px; border-radius:4px; font-size:.95rem;
    }
    .chat-clarify {
        background:#e3f2fd; border-left:4px solid #1565c0;
        padding:8px 12px; border-radius:4px; font-size:.95rem;
    }
    .chat-empty {
        text-align:center; color:#9fa8da; padding:42px 12px;
        border:1px dashed #c5cae9; border-radius:12px; margin:18px 0;
    }

    /* HITL approval panels */
    .hitl-banner {
        background:#fff3e0; border-left:5px solid #e65100;
        padding:14px 18px; border-radius:6px; margin:12px 0;
    }
    .hitl-banner.rejected { background:#ffebee; border-left-color:#c62828; }
    .hitl-banner.approved { background:#e8f5e9; border-left-color:#2e7d32; }
    .hitl-driver {
        display:inline-block; background:#fff8e1; color:#bf360c;
        padding:2px 9px; border-radius:10px; font-size:.75rem;
        margin:1px 4px 1px 0; border:1px solid #ffe0b2;
    }
    .hitl-empty {
        text-align:center; color:#9fa8da; padding:42px 12px;
        border:1px dashed #c5cae9; border-radius:12px; margin:18px 0;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── Pipeline bootstrap ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building / loading the unified pipeline (FAISS + Knowledge Graph)...")
def load_pipeline():
    pipe = ManufacturingPipeline()
    pipe.build_or_load(enable_llm=llm_available())
    return pipe


pipe = load_pipeline()
LLM_ON = pipe.llm_enabled


@st.cache_resource(show_spinner=False)
def get_chat_agent() -> ChatAgent:
    return ChatAgent(pipe, max_optional_asks=1)


chat_agent = get_chat_agent()


# ── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        opts = {
            "top_k": st.slider("Results to retrieve", 1, 10, 5),
            "use_context": st.checkbox("Context window (neighbouring chunks)", value=True),
            "show_clarifier": st.checkbox("Show clarifier analysis", value=True),
            "show_corrections": st.checkbox("Show query corrections", value=True),
        }

        st.markdown("---")
        if st.button("🔄 Rebuild Indexes", use_container_width=True):
            with st.spinner("Rebuilding FAISS + KG..."):
                load_pipeline.clear()
                st.rerun()

        st.markdown("---")
        st.markdown("### 📊 Pipeline Status")
        s = pipe.stats
        st.metric("Documents (chunks)", s["documents"])
        st.metric("FAISS vectors", s["vectors"])
        st.metric("Graph nodes", s["kg_nodes"])
        st.metric("Graph edges", s["kg_edges"])

        st.markdown("---")
        st.markdown("### 🔧 Configuration")
        st.caption(f"Embedding: `{EMBEDDING_MODEL}`")
        if LLM_ON:
            st.success(f"LLM connected: `{LLM_MODEL}`")
        else:
            st.warning("LLM disabled (no OPENAI_API_KEY)")

        st.markdown("---")
        st.markdown("### 💡 Try these queries")
        examples = [
            "What is the OEE target for Q2 2026?",
            "Pump P-203 has high vibration alarm ALM-P001. Cause and fix?",
            "Why did CNC Line 4 shut down in February?",
            "Compare Nippon Steel vs ArcelorMittal",
            "Belt tracking deviation on conveyor CV-301",
            "Hydraulic press HP-401 pressure loss — diagnose",
            "What is the CAPA process for critical NCR?",
            "maintanance schedul for spindle bearings",
            "PLC fault code FC-003 on conveyor CV-302",
            "scrap rate for welding Plant A vs Plant B",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state["query_input"] = ex
                st.rerun()

        return opts


# ── Reusable renderers ──────────────────────────────────────────────────────
def render_clarifier_analysis(clarification, correction, opts: dict) -> None:
    if not (opts["show_clarifier"] or opts["show_corrections"]):
        return
    with st.expander("🔍 Query Analysis", expanded=True):
        c1, c2 = st.columns(2)

        if opts["show_clarifier"] and clarification is not None:
            with c1:
                st.markdown("**Clarifier Agent**")
                st.markdown(
                    f'<span class="intent-tag">{clarification.intent.value.upper()}</span> '
                    f'confidence: {clarification.intent_confidence:.0%}',
                    unsafe_allow_html=True,
                )
                if clarification.entities:
                    st.markdown("**Entities:**")
                    tags = "".join(
                        f'<span class="entity-tag">{e.entity_type}: {e.normalized}</span>'
                        for e in clarification.entities
                    )
                    st.markdown(tags, unsafe_allow_html=True)
                st.markdown("**Slots:**")
                for s in clarification.slots:
                    if s.filled:
                        st.markdown(f'<span class="slot-filled">✅ {s.name}</span> = {s.value}',
                                    unsafe_allow_html=True)
                    else:
                        req = "required" if s.required else "optional"
                        st.markdown(f'<span class="slot-missing">❌ {s.name} ({req})</span>',
                                    unsafe_allow_html=True)
                if not clarification.is_complete and clarification.clarification_prompt:
                    st.warning(clarification.clarification_prompt)

        if opts["show_corrections"] and correction is not None:
            with c2:
                st.markdown("**Query Correction**")
                if correction.corrections_applied:
                    st.markdown(
                        f'<div class="result-card">'
                        f'<b>Original:</b> {correction.original}<br>'
                        f'<b>Corrected:</b> {correction.corrected}</div>',
                        unsafe_allow_html=True,
                    )
                    for fix in correction.corrections_applied:
                        st.markdown(f"  • {fix}")
                else:
                    st.markdown("✅ No corrections needed")


def render_evidence(evidence: list, limit: int | None = None) -> None:
    if not evidence:
        st.info("No evidence retrieved.")
        return
    items = evidence if limit is None else evidence[:limit]
    type_colors = {"PDF": "#c62828", "TXT": "#2e7d32", "EXCEL": "#1565c0"}

    for i, ev in enumerate(items, 1):
        meta = ev.get("metadata", {})
        source = Path(str(meta.get("source", meta.get("source_file", "unknown")))).name
        doc_type = str(meta.get("doc_type", "unknown")).upper()
        score = ev.get("vector_score", ev.get("rrf_score", 0.0))

        location = []
        if "page" in meta: location.append(f"Page {meta['page']}")
        if "sheet_name" in meta: location.append(f"Sheet: {meta['sheet_name']}")
        if "section_title" in meta: location.append(f"Section: {meta['section_title']}")

        color = type_colors.get(doc_type, "#555")
        header_col, score_col = st.columns([5, 1])
        with header_col:
            st.markdown(
                f'**Result {i}** · '
                f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:.8rem">{doc_type}</span> '
                f'`{source}`'
                + (f' · {" · ".join(location)}' if location else ""),
                unsafe_allow_html=True,
            )
        with score_col:
            st.markdown(
                f'<div style="text-align:right;font-weight:700">{score:.3f}</div>',
                unsafe_allow_html=True,
            )
        preview = ev.get("text", "")
        if len(preview) > 800:
            last_dot = preview[:800].rfind(".")
            preview = preview[:last_dot + 1 if last_dot > 500 else 800] + " ..."
        st.markdown(f'<div class="result-card">{preview}</div>', unsafe_allow_html=True)


def render_graph_view(graph_context: dict | None) -> None:
    if not graph_context or not graph_context.get("nodes"):
        st.info("No graph entities matched. Try including equipment IDs (P-203, CNC-A-004) "
                "or alarm codes (ALM-P001).")
        return

    nodes, edges = graph_context["nodes"], graph_context.get("edges", [])
    type_colors = {
        "Equipment":"#1E2761", "Component":"#408EC6", "Alarm":"#dc3545",
        "FailureMode":"#ffc107", "Symptom":"#ff6b6b", "Cause":"#e67e22",
        "Procedure":"#28a745", "SparePart":"#6c757d", "Specification":"#17a2b8",
    }
    n = len(nodes)
    node_x, node_y, node_text, node_color, node_size, positions = [], [], [], [], [], {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / max(n, 1)
        x, y = math.cos(angle) * 3, math.sin(angle) * 3
        node_x.append(x); node_y.append(y)
        node_text.append(f"{node['id']}<br>Type: {node['type']}<br>Chunks: {node['chunks']}")
        node_color.append(type_colors.get(node["type"], "#999"))
        node_size.append(max(15, node["chunks"] * 5))
        positions[node["id"]] = (x, y)

    edge_x, edge_y = [], []
    for edge in edges:
        if edge["source"] in positions and edge["target"] in positions:
            x0, y0 = positions[edge["source"]]
            x1, y1 = positions[edge["target"]]
            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                              line=dict(width=1, color="#ccc"), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color="white")),
        text=[n["id"][:20] for n in nodes], textposition="top center", textfont=dict(size=9),
        hovertext=node_text, hoverinfo="text",
    ))
    fig.update_layout(
        showlegend=False, height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white", margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tabs ────────────────────────────────────────────────────────────────────
def _ensure_chat_state() -> ChatState:
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = ChatState()
    return st.session_state.chat_state


def _render_answer_meta(meta: dict) -> None:
    """Compact metadata strip shown under each LLM/quick answer."""
    chips: list[str] = []

    clar = meta.get("clarification")
    if clar is not None:
        chips.append(
            f'<span class="chat-intent">{clar.intent.value.upper()}</span>'
            f'<span class="chat-meta-chip">confidence {clar.intent_confidence:.0%}</span>'
        )
        for e in clar.entities[:5]:
            chips.append(f'<span class="chat-meta-chip">{e.entity_type}: {e.normalized}</span>')

    metrics = meta.get("metrics") or {}
    if metrics.get("total_latency_ms"):
        chips.append(f'<span class="chat-meta-chip">⏱ {format_latency(metrics["total_latency_ms"])}</span>')
    if metrics.get("total_tokens"):
        chips.append(f'<span class="chat-meta-chip">🧩 {metrics["total_tokens"]:,} tokens</span>')
    if metrics.get("cost_estimate_usd") is not None:
        chips.append(f'<span class="chat-meta-chip">💲 {format_cost(metrics["cost_estimate_usd"])}</span>')

    critic = (meta.get("critic") or {}).get("final_verdict", {}) or {}
    verdict = critic.get("verdict")
    if verdict:
        cls = {"PASS": "pass-badge", "FAIL": "fail-badge"}.get(verdict, "skip-badge")
        chips.append(f'<span class="{cls}">Critic: {verdict}</span>')

    if chips:
        st.markdown(" ".join(chips), unsafe_allow_html=True)


def _render_chat_turn(turn) -> None:
    if turn.kind == "correction":
        with st.chat_message("assistant", avatar="🪄"):
            st.markdown(f'<div class="chat-correction">{turn.content}</div>', unsafe_allow_html=True)
        return

    if turn.kind == "clarify":
        with st.chat_message("assistant", avatar="❓"):
            st.markdown(f'<div class="chat-clarify">{turn.content}</div>', unsafe_allow_html=True)
        return

    if turn.kind == "system":
        with st.chat_message("assistant", avatar="ℹ️"):
            st.markdown(turn.content)
        return

    if turn.kind == "approval_pending":
        with st.chat_message("assistant", avatar="🛑"):
            st.markdown(
                f'<div class="hitl-banner">{turn.content}</div>',
                unsafe_allow_html=True,
            )
            risk = turn.meta.get("risk") or {}
            if risk.get("drivers"):
                chips = "".join(
                    f'<span class="hitl-driver">{d}</span>'
                    for d in risk["drivers"][:8]
                )
                st.markdown(chips, unsafe_allow_html=True)
        return

    if turn.role == "user":
        with st.chat_message("user"):
            st.markdown(turn.content)
        return

    with st.chat_message("assistant"):
        rejected = bool(turn.meta.get("rejected"))
        if rejected:
            st.markdown(f'<div class="hitl-banner rejected">{turn.content}</div>',
                        unsafe_allow_html=True)
        else:
            if turn.meta.get("human_decision"):
                st.markdown(
                    '<div class="hitl-banner approved">✅ <b>Approved by reviewer</b> — '
                    f'{turn.meta["human_decision"].get("approver", "unknown")}'
                    + (f" — {turn.meta['human_decision'].get('comments')}"
                       if turn.meta["human_decision"].get("comments") else "")
                    + '</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(turn.content)
        if turn.kind == "answer":
            _render_answer_meta(turn.meta)

            evidence = turn.meta.get("evidence") or []
            if evidence:
                with st.expander(f"📎 Evidence ({len(evidence)} chunks)"):
                    render_evidence(evidence)

            graph_ctx = turn.meta.get("graph_context")
            if graph_ctx and graph_ctx.get("nodes"):
                with st.expander(f"🔗 Knowledge graph context ({len(graph_ctx['nodes'])} nodes)"):
                    render_graph_view(graph_ctx)


def tab_chat(opts: dict) -> None:
    state = _ensure_chat_state()

    st.markdown("### 💬 Chat — Conversational Manufacturing Copilot")
    st.caption(
        "Ask anything about your operations. I auto-correct domain jargon, ask "
        "follow-up questions when details are missing, and ground answers in "
        "your documents + knowledge graph."
    )

    head_l, head_r = st.columns([6, 1])
    with head_r:
        if st.button("🆕 New chat", use_container_width=True, key="chat_reset"):
            state.reset()
            st.rerun()

    if not LLM_ON:
        st.info(
            "Running in **retrieval-only mode** (no `OPENAI_API_KEY` detected). "
            "Answers will summarise top evidence; for grounded LLM answers, set the key."
        )

    if not state.turns:
        st.markdown(
            '<div class="chat-empty">👋 Try something like<br>'
            "<i>“why did CNC Line 4 shut down in February?”</i> · "
            "<i>“maintanance schedul for spindle bearings”</i> · "
            "<i>“OEE target for Q2 2026?”</i></div>",
            unsafe_allow_html=True,
        )

    for turn in state.turns:
        _render_chat_turn(turn)

    if USE_HITL and state.pending_approval_thread_id:
        st.markdown(
            f'<div class="hitl-banner">⏸️ <b>Workflow paused for approval.</b> '
            f'Thread <code>{state.pending_approval_thread_id}</code> is waiting in the '
            f'<b>📋 Approvals</b> tab. Resolve it there before sending the next message.</div>',
            unsafe_allow_html=True,
        )
        return

    if state.awaiting_slot is not None:
        placeholder = f"Answer: {state.awaiting_slot.prompt}  (or type 'skip')"
    else:
        placeholder = "Ask anything about manufacturing operations…"

    message = st.chat_input(placeholder)
    if message:
        with st.spinner("Thinking…"):
            chat_agent.handle(state, message)
        st.rerun()


def tab_quick_search(opts: dict) -> None:
    st.markdown("### 🔍 Quick Search — Clarifier + FAISS Retrieval")
    st.caption("Fast semantic search. Does not call any LLM.")

    query = st.text_input(
        "Ask a question about manufacturing operations:",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g. What is the OEE for Plant A in Q1 2026?",
    )
    if not query:
        return

    start = time.time()
    result = pipe.quick_search(query, top_k=opts["top_k"], use_context_window=opts["use_context"])
    elapsed = time.time() - start

    c = result.clarification
    cols = st.columns(4)
    metrics = [
        (len(result.evidence), "Results"),
        (c.intent.value.upper(), f"Intent ({c.intent_confidence:.0%})"),
        (len(c.entities), "Entities"),
        (f"{elapsed:.2f}s", "Latency"),
    ]
    for col, (num, label) in zip(cols, metrics):
        col.markdown(
            f'<div class="metric-box"><div class="metric-num">{num}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    render_clarifier_analysis(result.clarification, result.correction, opts)
    st.markdown("---")
    st.markdown(f"### 📋 Retrieved Results ({len(result.evidence)})")
    render_evidence(result.evidence)


def tab_diagnostic(opts: dict) -> None:
    st.markdown("### 🩺 Diagnostic Copilot — Hybrid Retrieval + LLM + Critic")
    if not LLM_ON:
        st.warning("This tab requires an OPENAI_API_KEY in .env. "
                    "Use Quick Search for non-LLM exploration.")
        return

    st.caption(f"Hybrid retrieval (BM25 + FAISS + Graph + RRF) → LLM ({LLM_MODEL}) → Critic loop.")

    query = st.text_area(
        "Enter a manufacturing query:",
        value=st.session_state.get("query_input", ""),
        height=80,
        placeholder="e.g. Pump P-203 has high vibration alarm ALM-P001. What is the cause and fix?",
    )
    if st.button("🚀 Run Diagnostic", type="primary"):
        if not query.strip():
            return
        with st.spinner("Running clarifier → hybrid retrieval → LLM → critic..."):
            result = pipe.diagnostic(query)

        metrics = result.metrics
        critic = (result.critic or {}).get("final_verdict", {}) or {}
        verdict = critic.get("verdict", "N/A")

        cols = st.columns(4)
        cols[0].metric("Latency", format_latency(metrics.get("total_latency_ms", 0)))
        cols[1].metric("Tokens", f"{metrics.get('total_tokens', 0):,}")
        cols[2].metric("Cost", format_cost(metrics.get("cost_estimate_usd", 0)))
        badge = {"PASS":"pass-badge", "FAIL":"fail-badge"}.get(verdict, "skip-badge")
        cols[3].markdown(f'Critic: <span class="{badge}">{verdict}</span>',
                          unsafe_allow_html=True)

        render_clarifier_analysis(result.clarification, result.correction, opts)

        st.markdown("### Answer")
        st.markdown(result.answer or "_No answer returned._")

        with st.expander(f"📎 Evidence ({len(result.evidence)} chunks)"):
            render_evidence(result.evidence)

        with st.expander("🔗 Graph Context"):
            render_graph_view(result.graph_context)


def tab_comparison() -> None:
    st.markdown("### ⚖️ Pipeline Comparison — Direct LLM vs Classical RAG vs Hybrid GraphRAG")
    if not LLM_ON:
        st.warning("This tab requires an OPENAI_API_KEY.")
        return

    query = st.text_area(
        "Enter a manufacturing query:",
        value=st.session_state.get("query_input", ""),
        height=80,
        placeholder="e.g. Belt tracking deviation on conveyor CV-301...",
        key="cmp_query",
    )
    if not st.button("🚀 Run All Three Pipelines", type="primary"):
        return
    if not query.strip():
        return

    with st.spinner("Running all 3 pipelines..."):
        results = pipe.compare(query)

    cols = st.columns(3)
    cfg = [
        ("direct_llm", "Direct LLM", "direct-llm", "🔴"),
        ("classical_rag", "Classical RAG", "classical-rag", "🟡"),
        ("hybrid_graphrag", "Hybrid GraphRAG", "hybrid-graphrag", "🟢"),
    ]
    for col, (key, label, css, icon) in zip(cols, cfg):
        with col:
            r = results[key]
            metrics = r.metrics
            critic = (r.critic or {}).get("final_verdict", {}) or {}
            verdict = critic.get("verdict", "N/A")
            st.markdown(f'<div class="pipeline-header {css}">{icon} {label}</div>',
                         unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Latency", format_latency(metrics.get("total_latency_ms", 0)))
            m2.metric("Tokens", f"{metrics.get('total_tokens', 0):,}")
            m3, m4 = st.columns(2)
            m3.metric("Cost", format_cost(metrics.get("cost_estimate_usd", 0)))
            badge = {"PASS":"pass-badge", "FAIL":"fail-badge"}.get(verdict, "skip-badge")
            m4.markdown(f'<span class="{badge}">{verdict}</span>', unsafe_allow_html=True)
            st.caption(f"Evidence chunks: {len(r.evidence)}")
            with st.expander("View Answer", expanded=(key == "hybrid_graphrag")):
                st.markdown(r.answer or "_no response_")
            if r.evidence:
                with st.expander(f"Evidence ({len(r.evidence)})"):
                    render_evidence(r.evidence, limit=3)


def tab_knowledge_graph() -> None:
    st.markdown("### 🔗 Knowledge Graph Explorer")
    kg = pipe.kg
    if kg is None:
        st.info("Knowledge graph not built yet.")
        return
    stats = kg.get_stats()
    cols = st.columns(3)
    cols[0].metric("Total Nodes", stats["total_nodes"])
    cols[1].metric("Total Edges", stats["total_edges"])
    cols[2].metric("Entity Types", len(stats.get("entity_types", {})))

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Entity Types**")
        if stats.get("entity_types"):
            fig = px.pie(names=list(stats["entity_types"].keys()),
                          values=list(stats["entity_types"].values()))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    with col_right:
        st.markdown("**Relation Types**")
        if stats.get("relation_types"):
            fig = px.bar(x=list(stats["relation_types"].values()),
                          y=list(stats["relation_types"].keys()),
                          orientation="h")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    explore = st.text_input("Explore an entity (equipment ID, alarm code, etc.):",
                              placeholder="e.g. P-203, ALM-P001, CNC-A-004")
    if explore:
        render_graph_view(kg.get_subgraph_for_query(explore))


def tab_approvals() -> None:
    """HITL operations console — approve / reject paused workflows + audit log."""
    st.markdown("### 📋 Approvals — Human-in-the-Loop")
    if not USE_HITL:
        st.info(
            "HITL is **disabled** (`USE_HITL=false`). Set `USE_HITL=true` (and "
            "`USE_LANGGRAPH=true`) in `.env` and restart the stack to enable the "
            "approval gate."
        )
        return
    if not USE_LANGGRAPH:
        st.warning(
            "HITL requires the LangGraph orchestrator. Set `USE_LANGGRAPH=true` "
            "in `.env` and restart."
        )
        return

    pending = pipe.pending_approvals()
    state = _ensure_chat_state()

    cols = st.columns([1, 1, 1, 4])
    cols[0].metric("Pending", len(pending))
    try:
        from core.audit_log import get_default_log
        audit_stats = get_default_log().stats()
        cols[1].metric("Approved", audit_stats["approved"])
        cols[2].metric("Rejected", audit_stats["rejected"])
        cols[3].metric("Approval rate", f"{audit_stats['approval_rate'] * 100:.0f}%")
    except Exception as exc:
        cols[1].caption(f"Audit log unavailable: {exc}")

    st.markdown("---")
    if not pending:
        st.markdown(
            '<div class="hitl-empty">🎉 No pending approvals — the queue is clear.</div>',
            unsafe_allow_html=True,
        )
    else:
        for entry in pending:
            _render_pending_approval(entry, state)

    st.markdown("---")
    st.markdown("### 📜 Recent decisions")
    try:
        from core.audit_log import get_default_log
        recent = get_default_log().recent(limit=20)
        if not recent:
            st.caption("No decisions recorded yet.")
        else:
            df = pd.DataFrame([
                {
                    "When": e.to_dict()["ts_iso"],
                    "Decision": "✅ approved" if e.decision == "approved" else "❌ rejected",
                    "Domain": e.domain,
                    "Approver": e.approver,
                    "Risk": f"{e.risk_score:.2f}",
                    "Drivers": ", ".join(e.drivers[:3]),
                    "Query": (e.query or "")[:80],
                    "Comments": (e.comments or "")[:80],
                }
                for e in recent
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.caption(f"Audit log unavailable: {exc}")


def _render_pending_approval(entry: dict, state: ChatState) -> None:
    thread_id = entry.get("thread_id", "?")
    risk = entry.get("risk", {}) or {}
    purchase = entry.get("purchase_request")
    domain = "purchase_request" if purchase else "diagnostic"
    score = float(risk.get("score", 0.0))

    with st.container(border=True):
        head_l, head_r = st.columns([5, 1])
        head_l.markdown(
            f"**Thread `{thread_id}`** · domain: `{domain}` · "
            f"risk: **{score:.2f}** ({risk.get('summary', '')})"
        )
        head_r.caption(time.strftime("%H:%M:%S", time.localtime(entry.get("ts", time.time()))))

        if risk.get("drivers"):
            chips = "".join(f'<span class="hitl-driver">{d}</span>' for d in risk["drivers"][:8])
            st.markdown(chips, unsafe_allow_html=True)

        st.markdown("**User query**")
        st.markdown(f"> {entry.get('raw_query', '')}")

        if purchase:
            from core.purchase_request import format_for_review
            st.markdown(format_for_review(purchase))

        st.markdown("**Proposed answer**")
        proposed = entry.get("answer", "") or "_(empty)_"
        edited = st.text_area(
            "Edit before approving (optional)",
            value=proposed,
            height=180,
            key=f"hitl_edit_{thread_id}",
        )

        c1, c2, c3 = st.columns([2, 2, 3])
        approver = c3.text_input(
            "Approver",
            value=st.session_state.get(f"hitl_approver_{thread_id}", "demo@plant"),
            key=f"hitl_approver_{thread_id}",
        )
        comments = c3.text_input(
            "Comments (optional)",
            key=f"hitl_comments_{thread_id}",
        )

        approve_clicked = c1.button(
            "✅ Approve", key=f"hitl_approve_{thread_id}", type="primary",
            use_container_width=True,
        )
        reject_clicked = c2.button(
            "❌ Reject", key=f"hitl_reject_{thread_id}",
            use_container_width=True,
        )

        if approve_clicked or reject_clicked:
            decision = {
                "approved": bool(approve_clicked),
                "approver": approver or "unknown",
                "comments": comments or None,
                "edited_answer": (edited if edited.strip() and edited.strip() != proposed.strip() else None),
            }
            with st.spinner("Resuming workflow…"):
                try:
                    chat_agent.apply_resolution(state, thread_id, decision)
                    _record_audit_from_streamlit(thread_id, entry, decision)
                except Exception as exc:
                    st.error(f"Resume failed: {exc}")
                    return
            st.success(
                ("Approved" if approve_clicked else "Rejected") +
                f" · thread `{thread_id}`. The answer has been delivered to the chat tab."
            )
            st.rerun()


def _record_audit_from_streamlit(thread_id: str, entry: dict, decision: dict) -> None:
    """Mirror the FastAPI audit-write so the in-process Streamlit demo also logs."""
    try:
        from core.audit_log import get_default_log
        risk = entry.get("risk", {}) or {}
        purchase = entry.get("purchase_request")
        get_default_log().record(
            thread_id=thread_id,
            decision="approved" if decision.get("approved") else "rejected",
            approver=decision.get("approver") or "unknown",
            risk_score=float(risk.get("score", 0.0)),
            drivers=risk.get("drivers", []),
            domain="purchase_request" if purchase else "diagnostic",
            query=entry.get("raw_query", ""),
            proposed_answer=entry.get("answer", ""),
            edited_answer=decision.get("edited_answer"),
            comments=decision.get("comments"),
        )
    except Exception:
        pass


def tab_architecture() -> None:
    st.markdown("### 🔄 Unified Pipeline Flow")
    st.markdown("""
    ```
    PDFs / TXT / Excel  ──▶  doc_pipeline.parsers ──▶  HybridChunker
                                                              │
                                                              ▼
                                           ┌──────────────────────────────┐
                                           │  FAISS embeddings (canonical)│
                                           └──────────────────────────────┘
                                                              │
                                                              ▼
              ClarifierAgent + QueryCorrector ─────▶  ┌─────────────────┐
                                                      │ BM25  Vector KG │  RRF fusion + edge priors
                                                      └─────────────────┘
                                                              │
                                                              ▼
                                                     Hybrid evidence
                                                              │
                                                              ▼
                                                     LLM (gpt-4o-mini)
                                                              │
                                                              ▼
                                                      Critic ↺ retry
                                                              │
                                                              ▼
                                              Grounded answer + citations
    ```
    """)
    st.markdown("### Architecture Comparison")
    comp = {
        "Feature": [
            "Retrieval Method", "Knowledge Graph", "Query Understanding",
            "Evidence Grounding", "Self-Correction", "Citation Support",
            "ID/Jargon Handling", "Audit Trail",
        ],
        "Direct LLM": ["None", "No", "Basic", "None", "No", "No", "Poor", "No"],
        "Classical RAG": ["Vector only", "No", "Basic", "Partial", "No", "Partial", "Limited", "Limited"],
        "Hybrid GraphRAG": [
            "BM25 + FAISS + Graph", "Yes", "Clarifier + Intent + Entity",
            "Full — chunk-level", "Yes — critic loop", "Yes — with provenance",
            "Excellent — graph-aware", "Yes — full pipeline trace",
        ],
    }
    st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

    st.markdown("### 💰 Cost-Benefit Analysis")
    col1, col2 = st.columns(2)
    with col1:
        queries = st.number_input("Queries / month", value=100000, step=10000)
    with col2:
        error_cost = st.number_input("Cost per wrong answer ($)", value=300, step=50)

    proj = compute_cost_projection(queries, error_cost)
    cols = st.columns(3)
    for col, (key, label) in zip(cols, [
        ("direct_llm", "Direct LLM"),
        ("classical_rag", "Classical RAG"),
        ("hybrid_graphrag", "Hybrid GraphRAG"),
    ]):
        col.metric(
            f"{label} Total",
            f"${proj[key]['total_monthly_cost']:,.0f}/mo",
            f"{proj[key]['wrong_answers']:,} wrong answers",
            delta_color="inverse",
        )
    st.success(
        f"**Savings vs Classical RAG:** ${proj['savings_vs_classical']:,.0f}/month | "
        f"**Savings vs Direct LLM:** ${proj['savings_vs_direct']:,.0f}/month | "
        f"**ROI:** {proj['roi_vs_classical']:,.0f}x"
    )


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    st.markdown('<p class="main-header">🏭 Manufacturing Hybrid GraphRAG</p>',
                 unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Unified pipeline — PDF/TXT/Excel ingestion · smart chunking · '
        'FAISS + BM25 + Knowledge Graph · Clarifier · LLM with critic loop</p>',
        unsafe_allow_html=True,
    )

    opts = render_sidebar()

    tab_titles = [
        "💬 Chat",
        "🔍 Quick Search",
        "🩺 Diagnostic Copilot",
        "⚖️ Comparison",
        "🔗 Knowledge Graph",
        "📋 Approvals",
        "🔄 Architecture & Cost",
    ]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        tab_chat(opts)
    with tabs[1]:
        tab_quick_search(opts)
    with tabs[2]:
        tab_diagnostic(opts)
    with tabs[3]:
        tab_comparison()
    with tabs[4]:
        tab_knowledge_graph()
    with tabs[5]:
        tab_approvals()
    with tabs[6]:
        tab_architecture()


main()
