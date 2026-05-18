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
import tempfile
import time
from pathlib import Path
from typing import List

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


# ── Pipeline bootstrap (per domain) ─────────────────────────────────────────
from config import (  # noqa: E402
    DOMAINS,
    DEFAULT_DOMAIN,
    DOMAIN_DISPLAY,
    DOMAIN_EXAMPLES,
    DOMAIN_EMPTY_STATE,
    DOMAIN_PLACEHOLDER,
)

# Every domain string surfaced to the UI lives in ``schemas/<domain>.yaml``
# (display + examples + empty_state + placeholder blocks). The dicts below
# are thin views over ``config.DOMAIN_*`` so new domains land for free.
DOMAIN_LABELS: dict[str, str] = {d: DOMAIN_DISPLAY[d]["label"] for d in DOMAINS}
DOMAIN_COLORS: dict[str, str] = {d: DOMAIN_DISPLAY[d]["color"] for d in DOMAINS}
DOMAIN_EMOJI:  dict[str, str] = {d: DOMAIN_DISPLAY[d]["emoji"] for d in DOMAINS}

EXAMPLE_QUERIES:    dict[str, list[str]] = DOMAIN_EXAMPLES
EXAMPLE_PLACEHOLDER: dict[str, str]      = DOMAIN_PLACEHOLDER

# A short, italicised one-liner under the chat empty-state heading. Derived
# from the first three ``examples`` so adding a new domain doesn't require
# authoring a separate hint.
def _empty_state_hint(domain: str) -> str:
    samples = (DOMAIN_EXAMPLES.get(domain) or [])[:3]
    if not samples:
        return ""
    return " · ".join(f"<i>“{s}”</i>" for s in samples)

EMPTY_STATE_HINTS: dict[str, str] = {d: _empty_state_hint(d) for d in DOMAINS}
EMPTY_STATE_EMOJI: dict[str, str] = {d: DOMAIN_EMOJI[d] for d in DOMAINS}


@st.cache_resource(show_spinner="Building / loading pipelines for both domains...")
def load_pipelines() -> dict:
    out: dict = {}
    for d in DOMAINS:
        p = ManufacturingPipeline(domain=d)
        p.build_or_load(enable_llm=llm_available())
        out[d] = p
    return out


PIPES = load_pipelines()


# Active domain default — selector in the sidebar mutates this on every
# rerun, and the module-level ``pipe`` / ``chat_agent`` below resolve
# against whatever is in session_state at script-rerun time.
if "active_domain" not in st.session_state:
    st.session_state["active_domain"] = DEFAULT_DOMAIN


@st.cache_resource(show_spinner=False)
def _chat_agent_for(domain: str) -> ChatAgent:
    return ChatAgent(PIPES[domain], max_optional_asks=1)


# Resolved once per Streamlit rerun. Every tab function reads this global,
# so a sidebar selector change → rerun → ``pipe`` points at the new domain.
ACTIVE_DOMAIN: str = st.session_state["active_domain"]
pipe = PIPES[ACTIVE_DOMAIN]
LLM_ON = pipe.llm_enabled
chat_agent = _chat_agent_for(ACTIVE_DOMAIN)


# ── Domain affordances ──────────────────────────────────────────────────────
def domain_pill(domain: str | None) -> str:
    """Return a small HTML pill marking which domain a chunk/turn belongs to."""
    if not domain:
        return ""
    color = DOMAIN_COLORS.get(domain, "#64748B")
    label = DOMAIN_LABELS.get(domain, domain.title())
    emoji = DOMAIN_EMOJI.get(domain, "")
    return (
        f"<span style='display:inline-block;padding:1px 8px;border-radius:999px;"
        f"background:{color}22;color:{color};font-size:0.72rem;"
        f"font-weight:600;letter-spacing:.02em;border:1px solid {color}55;'>"
        f"{emoji} {label}</span>"
    )


def domain_border_style(domain: str | None) -> str:
    """4px left-border CSS, color-keyed to the chunk's domain."""
    color = DOMAIN_COLORS.get(domain or "", "#CBD5E1")
    return (
        f"border-left:4px solid {color};padding-left:10px;margin-bottom:8px;"
    )


# ── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    from core.llm_router import (
        get_active_backend,
        set_active_backend,
        status as llm_status,
    )

    with st.sidebar:
        # ── LLM backend selector ───────────────────────────────────────
        st.markdown("### 🤖 LLM backend")
        backend_status = llm_status()
        active_backend = backend_status["active"]
        # Each option label hints which models will be used.
        backend_labels = {
            "auto":  f"🪄 Auto · resolves to {backend_status['active'].title()}",
            "cloud": "☁️ Cloud · OpenAI (gpt-4o / gpt-4o-mini)",
            "local": "💻 Local · Ollama (qwen2.5:3b)",
        }
        # Disable cloud if the key is invalid.
        available = ["auto", "local"]
        if backend_status["openai_key_valid"]:
            available.append("cloud")
        else:
            backend_labels["cloud"] += " · ⚠ key missing"
        raw_backend = backend_status["raw"]
        if raw_backend not in available:
            raw_backend = "auto"
        backend_pick = st.selectbox(
            "LLM backend",
            options=["auto", "local", "cloud"],
            index=["auto", "local", "cloud"].index(raw_backend),
            format_func=lambda b: backend_labels[b],
            label_visibility="collapsed",
            key="llm_backend_selector",
        )
        if backend_pick == "cloud" and not backend_status["openai_key_valid"]:
            st.warning(
                "OpenAI key not detected — set OPENAI_API_KEY in `.env`. "
                "Staying on the previous backend."
            )
        elif backend_pick != raw_backend:
            try:
                set_active_backend(backend_pick)
                st.rerun()
            except ValueError as e:
                st.error(str(e))
        # Active-backend chip
        backend_color = "#10B981" if active_backend == "local" else "#0EA5E9"
        backend_emoji = "💻" if active_backend == "local" else "☁️"
        st.markdown(
            f"<div style='font-size:0.8rem;color:{backend_color};margin-bottom:8px;'>"
            f"{backend_emoji} <b>{active_backend}</b> &middot; "
            f"answer=<code>{backend_status['per_task']['answer']}</code></div>",
            unsafe_allow_html=True,
        )

        # ── Domain selector ────────────────────────────────────────────
        st.markdown("### 🗂️ Active domain")
        domain_choice = st.selectbox(
            "Active domain",
            options=list(DOMAINS),
            index=list(DOMAINS).index(st.session_state["active_domain"]),
            format_func=lambda d: f"{DOMAIN_EMOJI.get(d,'')} {DOMAIN_LABELS.get(d,d.title())}",
            label_visibility="collapsed",
            key="domain_selector",
        )
        if domain_choice != st.session_state["active_domain"]:
            st.session_state["active_domain"] = domain_choice
            st.rerun()
        st.markdown(
            f"<div style='font-size:0.8rem;color:{DOMAIN_COLORS[domain_choice]};margin-bottom:8px;'>"
            f"{DOMAIN_EMOJI[domain_choice]} {DOMAIN_LABELS[domain_choice]} corpus active "
            f"&middot; collection <code>{pipe.embedding_pipeline._collection}</code></div>",
            unsafe_allow_html=True,
        )

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
        examples = EXAMPLE_QUERIES.get(ACTIVE_DOMAIN, [])
        for ex in examples:
            if st.button(ex, key=f"ex_{ACTIVE_DOMAIN}_{ex}", use_container_width=True):
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
        # Domain inferred from the chunk's source path. Falls back to the
        # active domain so legacy/unknown chunks still get a coloured rail.
        src = str(meta.get("source", meta.get("source_file", ""))).lower()
        if "/aviation/" in src or src.startswith("aviation/"):
            chunk_domain = "aviation"
        elif "/manufacturing/" in src or src.startswith("manufacturing/"):
            chunk_domain = "manufacturing"
        else:
            chunk_domain = st.session_state.get("active_domain")

        location = []
        if "page" in meta: location.append(f"Page {meta['page']}")
        if "sheet_name" in meta: location.append(f"Sheet: {meta['sheet_name']}")
        if "section_title" in meta: location.append(f"Section: {meta['section_title']}")

        color = type_colors.get(doc_type, "#555")
        header_col, score_col = st.columns([5, 1])
        with header_col:
            st.markdown(
                f'**Result {i}** · '
                + domain_pill(chunk_domain) + ' '
                + f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:.8rem">{doc_type}</span> '
                + f'`{source}`'
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
        st.markdown(
            f'<div class="result-card" style="{domain_border_style(chunk_domain)}">{preview}</div>',
            unsafe_allow_html=True,
        )


def render_graph_view(graph_context: dict | None) -> None:
    if not graph_context or not graph_context.get("nodes"):
        # Surface a couple of real KG identifiers from the active domain so
        # the hint is always relevant, never hardcoded.
        domain = st.session_state.get("active_domain") or DEFAULT_DOMAIN
        kg = PIPES[domain].kg if PIPES.get(domain) is not None else None
        samples: list[str] = []
        if kg is not None:
            for n, d in kg.graph.nodes(data=True):
                if d.get("entity_type") == "Equipment" and len(samples) < 1:
                    samples.append(str(n))
                if d.get("entity_type") in ("Cause", "Component", "Alarm") and len(samples) < 3:
                    samples.append(str(n))
                if len(samples) >= 3:
                    break
        hint = (
            f"Try including identifiers like {', '.join(samples)}."
            if samples
            else "Try including an entity id or one of the schema's vocab terms."
        )
        st.info(f"No graph entities matched. {hint}")
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
    """One ChatState per active domain — switching domains gives the user a
    fresh transcript and stops cross-domain context bleed."""
    domain = st.session_state.get("active_domain", DEFAULT_DOMAIN)
    states = st.session_state.setdefault("chat_states", {})
    if domain not in states:
        states[domain] = ChatState()
    return states[domain]


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

    # Each turn carries the domain it was asked under. Falls back to the
    # currently-active domain for legacy turns saved before the split.
    turn_domain = (turn.meta or {}).get("domain") or st.session_state.get("active_domain")

    if turn.role == "user":
        with st.chat_message("user"):
            st.markdown(
                f"{domain_pill(turn_domain)} &nbsp;{turn.content}",
                unsafe_allow_html=True,
            )
        return

    with st.chat_message("assistant"):
        st.markdown(domain_pill(turn_domain), unsafe_allow_html=True)
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
        es = DOMAIN_EMPTY_STATE.get(ACTIVE_DOMAIN, {})
        heading = es.get("heading", f"{DOMAIN_LABELS[ACTIVE_DOMAIN]} Copilot")
        blurb = es.get("blurb", "")
        hint = EMPTY_STATE_HINTS[ACTIVE_DOMAIN]
        st.markdown(
            '<div class="chat-empty">'
            f'<div style="font-size:2rem">{EMPTY_STATE_EMOJI[ACTIVE_DOMAIN]}</div>'
            f'<div style="font-weight:600;font-size:1.1rem;margin-top:4px">{heading}</div>'
            + (f'<div style="color:#475569;font-size:.9rem;margin-top:6px">{blurb}</div>' if blurb else "")
            + (f'<div style="margin-top:10px">Try something like<br>{hint}</div>' if hint else "")
            + '</div>',
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
        placeholder = EXAMPLE_PLACEHOLDER.get(
            ACTIVE_DOMAIN, "Ask a question…"
        )

    message = st.chat_input(placeholder)
    if message:
        prior_turn_count = len(state.turns)
        with st.spinner(f"Thinking… [{DOMAIN_LABELS[ACTIVE_DOMAIN]}]"):
            chat_agent.handle(state, message)
        # Stamp the active domain on every turn this exchange produced so
        # the renderer can show the correct domain pill, even if the user
        # switches domains for the next message.
        for t in state.turns[prior_turn_count:]:
            t.meta.setdefault("domain", ACTIVE_DOMAIN)
        st.rerun()


def tab_quick_search(opts: dict) -> None:
    st.markdown("### 🔍 Quick Search — Clarifier + FAISS Retrieval")
    st.caption("Fast semantic search. Does not call any LLM.")

    # Both label and placeholder draw from the schema's UX block — adding a
    # new domain doesn't require any branching here.
    label = f"Ask a {DOMAIN_LABELS[ACTIVE_DOMAIN]} question:"
    examples_for_domain = EXAMPLE_QUERIES.get(ACTIVE_DOMAIN) or []
    placeholder = (
        f"e.g. {examples_for_domain[0]}"
        if examples_for_domain
        else EXAMPLE_PLACEHOLDER.get(ACTIVE_DOMAIN, "Ask a question…")
    )
    query = st.text_input(
        label,
        value=st.session_state.get("query_input", ""),
        placeholder=placeholder,
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

    st.caption(
        f"Hybrid retrieval (BM25 + Qdrant + Graph + RRF) → bge-reranker "
        f"→ LLM ({LLM_MODEL}) → Critic loop."
    )

    diag_label = f"Enter a {DOMAIN_LABELS[ACTIVE_DOMAIN]} diagnostic query:"
    examples_for_domain = EXAMPLE_QUERIES.get(ACTIVE_DOMAIN) or []
    diag_placeholder = (
        f"e.g. {examples_for_domain[0]}"
        if examples_for_domain
        else EXAMPLE_PLACEHOLDER.get(ACTIVE_DOMAIN, "Ask a question…")
    )
    query = st.text_area(
        diag_label,
        value=st.session_state.get("query_input", ""),
        height=80,
        placeholder=diag_placeholder,
    )

    can_stream = (
        getattr(pipe, "_orchestrator_engine", "") == "langgraph"
        and hasattr(pipe, "diagnostic_stream")
    )
    stream_mode = st.toggle(
        "Stream pipeline stages (LangGraph)",
        value=can_stream,
        disabled=not can_stream,
        help="Render each pipeline stage (retrieval, cause-ranking, "
             "procedure, critic) as soon as it finishes instead of "
             "waiting for the whole graph. Requires USE_LANGGRAPH=true.",
    )

    if st.button("🚀 Run Diagnostic", type="primary"):
        if not query.strip():
            return
        if stream_mode and can_stream:
            _run_diagnostic_streaming(query, opts)
        else:
            _run_diagnostic_blocking(query, opts)


def _run_diagnostic_blocking(query: str, opts: dict) -> None:
    with st.spinner("Running clarifier → hybrid retrieval → LLM → critic..."):
        result = pipe.diagnostic(query)
    _render_diagnostic_result(result, opts)


def _run_diagnostic_streaming(query: str, opts: dict) -> None:
    """Stream pipeline stages incrementally (piston-style)."""
    stage_labels = {
        "clarify": "Clarifier…",
        "correct": "Query correction…",
        "format": "Query understanding…",
        "detect_purchase": "Purchase-request detection…",
        "retrieve": "Hybrid retrieval (BM25 + Qdrant + Graph)…",
        "tools_read": "Tool calls (ERP / MES)…",
        "rank_causes": "Cause ranking…",
        "draft_procedure": "Drafting procedure…",
        "generate": "Generating answer…",
        "criticality_check": "Risk scoring…",
        "human_approval": "Awaiting human approval…",
        "critic": "Critic review…",
        "retry": "Retrying with critic feedback…",
    }

    evidence_ph = st.empty()
    causes_ph = st.empty()
    procedure_ph = st.empty()
    answer_ph = st.empty()
    critic_ph = st.empty()
    metrics_ph = st.empty()

    final_response: dict | None = None
    with st.status("Running pipeline…", expanded=False) as status:
        for event in pipe.diagnostic_stream(query):
            kind = event.get("event")
            if kind == "node_update":
                node = event.get("node", "")
                status.update(label=stage_labels.get(node, f"{node}…"))
                update = event.get("update") or {}
                if "evidence" in update:
                    with evidence_ph.container():
                        with st.expander(
                            f"📎 Retrieved Evidence — {len(update['evidence'])} chunks",
                            expanded=True,
                        ):
                            render_evidence(update["evidence"])
                if "cause_ranking" in update:
                    cands = (update.get("cause_ranking") or {}).get("candidates", [])
                    with causes_ph.container():
                        with st.expander(f"🎯 Ranked Causes — {len(cands)}", expanded=True):
                            for i, c in enumerate(cands, 1):
                                st.markdown(
                                    f"**{i}. {c.get('cause', '')}**  · score `{c.get('score', 0):.2f}`"
                                )
                                if c.get("rationale"):
                                    st.write(c["rationale"])
                if "procedure" in update:
                    steps = ((update.get("procedure") or {}).get("procedure") or {}).get("steps") or []
                    with procedure_ph.container():
                        with st.expander(
                            f"🛠️ Diagnostic Procedure — {len(steps)} steps", expanded=True,
                        ):
                            for s in steps:
                                cites = " ".join(f"`[{c}]`" for c in (s.get("citations") or []))
                                st.markdown(f"**Step {s.get('step', '?')}.** {s.get('action', '')} {cites}")
                if "answer" in update and update["answer"]:
                    with answer_ph.container():
                        st.markdown("### Answer")
                        st.markdown(update["answer"])
                if "attempts" in update:
                    attempts = update["attempts"]
                    last = attempts[-1] if attempts else {}
                    verdict = last.get("verdict", "N/A")
                    badge = {"PASS": "pass-badge", "FAIL": "fail-badge"}.get(verdict, "skip-badge")
                    with critic_ph.container():
                        st.markdown(
                            f'Critic: <span class="{badge}">{verdict}</span> '
                            f"(attempt {len(attempts)})",
                            unsafe_allow_html=True,
                        )
            elif kind in ("complete", "interrupted"):
                final_response = event.get("response") or {}
                status.update(
                    label="Awaiting approval" if kind == "interrupted" else "Pipeline complete",
                    state=("running" if kind == "interrupted" else "complete"),
                )
            elif kind == "error":
                st.error(event.get("message", "unknown error"))

    if final_response:
        metrics = final_response.get("metrics", {}) or {}
        with metrics_ph.container():
            cols = st.columns(4)
            cols[0].metric("Latency", format_latency(metrics.get("total_latency_ms", 0)))
            cols[1].metric("Tokens", f"{metrics.get('total_tokens', 0):,}")
            cols[2].metric("Cost", format_cost(metrics.get("cost_estimate_usd", 0)))
            cols[3].metric("Risk", f"{(final_response.get('risk') or {}).get('score', 0):.2f}")


def _render_diagnostic_result(result, opts: dict) -> None:
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

    # If a structured procedure was emitted, surface it as its own section
    # for easier scanning. The free-form answer is the markdown render so
    # this is purely additive.
    procedure = (result.procedure or {}).get("procedure") if result.procedure else None
    steps = (procedure or {}).get("steps") or []
    if steps:
        with st.expander(f"🛠️ Diagnostic Procedure — {len(steps)} steps", expanded=True):
            for s in steps:
                cites = " ".join(f"`[{c}]`" for c in (s.get("citations") or []))
                st.markdown(f"**Step {s.get('step', '?')}.** {s.get('action', '')} {cites}")

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

    # Pick a couple of real Equipment ids + one Cause/Component vocab term
    # from the active KG so the placeholder is domain-relevant without any
    # hardcoded strings.
    eg = []
    for n, d in kg.graph.nodes(data=True):
        if d.get("entity_type") == "Equipment":
            eg.append(str(n)); break
    for n, d in kg.graph.nodes(data=True):
        if d.get("entity_type") in ("Cause", "Component"):
            eg.append(str(n)); break
    explore_placeholder = f"e.g. {', '.join(eg)}" if eg else "type an entity id or vocab term"
    explore = st.text_input(
        "Explore an entity (equipment ID, alarm code, etc.):",
        placeholder=explore_placeholder,
    )
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

    # ── Sign-in (RBAC) ──────────────────────────────────────────────────
    # Approvals are role-gated and require a maker-lock check, so a signed-in
    # identity is mandatory. The same SQLite user store backs both the
    # FastAPI bearer-token flow and this in-process Streamlit form.
    if not _render_streamlit_signin():
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


def _render_streamlit_signin() -> bool:
    """Render an in-tab sign-in form. Returns True once a user is signed in.

    Uses the same SQLite store as the FastAPI auth layer so credentials are
    interchangeable across the two UIs (no separate Streamlit user table).
    """
    from core.auth_store import AuthError, get_default_store
    from core.rbac import ROLES_BY_ID

    user_dict = st.session_state.get("hitl_user")
    if user_dict:
        sub_l, sub_r = st.columns([5, 1])
        sub_l.markdown(
            f"👤 Signed in as **{user_dict['display_name'] or user_dict['user_id']}** "
            f"(`{user_dict['user_id']}`) · role: "
            f"`{user_dict['role']}`"
        )
        if sub_r.button("Sign out", key="hitl_signout"):
            st.session_state.pop("hitl_user", None)
            st.rerun()
        return True

    with st.expander("🔐 Sign in to approve", expanded=True):
        st.caption(
            "Approvals are role-gated and enforce maker-cannot-be-checker. "
            "Use one of the demo accounts (see README) or sign up."
        )
        with st.form("hitl_signin", clear_on_submit=False):
            email = st.text_input("User ID", placeholder="dave.ehs@plant.local")
            pw = st.text_input("Password", type="password")
            colA, colB = st.columns(2)
            do_login = colA.form_submit_button("Sign in", type="primary", use_container_width=True)
            do_signup_toggle = colB.form_submit_button("Sign up instead", use_container_width=True)
            if do_login and email and pw:
                try:
                    user, _token, _exp = get_default_store().login(email, pw)
                except AuthError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["hitl_user"] = {
                        "user_id": user.user_id,
                        "role": user.role,
                        "display_name": user.display_name,
                    }
                    st.rerun()
            elif do_signup_toggle:
                st.session_state["hitl_signup_mode"] = True
                st.rerun()

        if st.session_state.get("hitl_signup_mode"):
            with st.form("hitl_signup", clear_on_submit=False):
                em = st.text_input("New user ID (email)")
                pw2 = st.text_input("Password (≥ 6 chars)", type="password")
                role_options = list(ROLES_BY_ID.keys())
                role = st.selectbox(
                    "Role",
                    role_options,
                    format_func=lambda r: f"{ROLES_BY_ID[r].label} — "
                    f"{'maker' if ROLES_BY_ID[r].is_maker else 'checker'}",
                )
                name = st.text_input("Display name (optional)")
                submitted = st.form_submit_button("Create account", type="primary")
                if submitted and em and pw2:
                    try:
                        get_default_store().signup(em, pw2, role, name)
                        user, _t, _e = get_default_store().login(em, pw2)
                    except AuthError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["hitl_user"] = {
                            "user_id": user.user_id,
                            "role": user.role,
                            "display_name": user.display_name,
                        }
                        st.session_state.pop("hitl_signup_mode", None)
                        st.rerun()

    return False


def _render_pending_approval(entry: dict, state: ChatState) -> None:
    from core.rbac import can_approve, is_maker_locked, required_roles_for

    thread_id = entry.get("thread_id", "?")
    risk = entry.get("risk", {}) or {}
    purchase = entry.get("purchase_request")
    domain = "purchase_request" if purchase else "diagnostic"
    score = float(risk.get("score", 0.0))
    drivers = risk.get("drivers", []) or []
    required_roles = entry.get("required_roles") or required_roles_for(drivers, purchase)
    maker_user_id = entry.get("maker_user_id")

    cur_user = st.session_state.get("hitl_user") or {}
    role_ok = can_approve(cur_user.get("role"), required_roles)
    maker_locked = is_maker_locked(cur_user.get("user_id"), maker_user_id)
    user_can_approve = role_ok and not maker_locked

    with st.container(border=True):
        head_l, head_r = st.columns([5, 1])
        head_l.markdown(
            f"**Thread `{thread_id}`** · domain: `{domain}` · "
            f"risk: **{score:.2f}** ({risk.get('summary', '')})"
        )
        head_r.caption(time.strftime("%H:%M:%S", time.localtime(entry.get("ts", time.time()))))

        if drivers:
            chips = "".join(f'<span class="hitl-driver">{d}</span>' for d in drivers[:8])
            st.markdown(chips, unsafe_allow_html=True)

        if required_roles:
            badges = " ".join(f"`{r}`" for r in required_roles)
            st.caption(f"Required role(s): {badges}")
        if maker_user_id:
            st.caption(
                f"Submitted by `{maker_user_id}`"
                + (" (you — segregation of duties prevents self-approval)" if maker_locked else "")
            )

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
            disabled=not user_can_approve,
        )

        c1, c2, c3 = st.columns([2, 2, 3])
        approver_label = cur_user.get("display_name") or cur_user.get("user_id") or "unknown"
        c3.markdown(
            f"_Approver:_ **{approver_label}** · _role:_ `{cur_user.get('role', 'unknown')}`"
        )
        comments = c3.text_input(
            "Comments (optional)",
            key=f"hitl_comments_{thread_id}",
            disabled=not user_can_approve,
        )

        approve_clicked = c1.button(
            "✅ Approve",
            key=f"hitl_approve_{thread_id}",
            type="primary",
            use_container_width=True,
            disabled=not user_can_approve,
        )
        reject_clicked = c2.button(
            "❌ Reject",
            key=f"hitl_reject_{thread_id}",
            use_container_width=True,
            disabled=not user_can_approve,
        )

        if not user_can_approve:
            if maker_locked:
                st.warning(
                    "You submitted this request — segregation of duties prevents you "
                    "from approving it. Ask a different role-holder to action it."
                )
            elif not role_ok:
                st.info(
                    f"Your role `{cur_user.get('role', '?')}` is not on this "
                    f"approval's allow-list. Required: {', '.join(required_roles)}."
                )

        if approve_clicked or reject_clicked:
            decision = {
                "approved": bool(approve_clicked),
                "approver": approver_label,
                "comments": comments or None,
                "edited_answer": (edited if edited.strip() and edited.strip() != proposed.strip() else None),
            }
            with st.spinner("Resuming workflow…"):
                try:
                    chat_agent.apply_resolution(state, thread_id, decision)
                    _record_audit_from_streamlit(
                        thread_id, entry, decision, cur_user, required_roles
                    )
                except Exception as exc:
                    st.error(f"Resume failed: {exc}")
                    return
            st.success(
                ("Approved" if approve_clicked else "Rejected") +
                f" · thread `{thread_id}`. The answer has been delivered to the chat tab."
            )
            st.rerun()


def _record_audit_from_streamlit(
    thread_id: str,
    entry: dict,
    decision: dict,
    user: dict,
    required_roles: list,
) -> None:
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
            maker_user_id=entry.get("maker_user_id"),
            approver_user_id=user.get("user_id"),
            approver_role=user.get("role"),
            required_roles=required_roles,
        )
    except Exception:
        pass


def tab_onboard_domain() -> None:
    """Schema-authoring agent — takes a few sample docs + a domain id and
    drives a bigger LLM to author ``schemas/<domain>.yaml`` end-to-end.
    Iterative: the agent may ask follow-up questions; the user answers
    inline and the agent re-emits.
    """
    import subprocess as _sub

    st.markdown("### 🌱 Onboard a new domain")
    st.caption(
        "Paste 1–5 sample documents and a domain id. An LLM agent extracts "
        "entity vocabularies, ID regex patterns, and UI copy, then writes "
        "the schema YAML. After saving, one click rebuilds the Qdrant "
        "index + KG and the domain lights up in every UI surface."
    )

    from config import ONBOARDING_MODEL, llm_available

    if not llm_available():
        st.error(
            "The onboarding agent needs an LLM backend. Set `OPENAI_API_KEY` "
            "(any OpenAI-compatible host) or point `ANSWER_MODEL` at a local "
            "Ollama model that's reachable."
        )
        return
    st.caption(f"Model: `{ONBOARDING_MODEL}` (override via `ONBOARDING_MODEL` env)")

    # Session state for the wizard
    state = st.session_state.setdefault(
        "onboard_state",
        {
            "domain_id": "",
            "domain_hint": "",
            "docs": [],            # list[str] of pasted/uploaded text
            "doc_names": [],
            "prior_qa": [],        # list[{question, answer}]
            "agent_response": None,  # last OnboardingResponse dict
            "current_answers": {},   # {question_idx: answer}
            "saved_to": None,
            "rebuild_output": "",
        },
    )

    inputs_col, output_col = st.columns([5, 7])

    # ── Inputs column ──────────────────────────────────────────────────
    with inputs_col:
        st.markdown("#### 1. Identify the domain")
        state["domain_id"] = st.text_input(
            "Domain id (lowercase a-z/0-9/_)",
            value=state["domain_id"],
            placeholder="e.g. medical, claims, semiconductor",
        )
        state["domain_hint"] = st.text_input(
            "Short label / hint (optional)",
            value=state["domain_hint"],
            placeholder="e.g. medical imaging service notes",
        )

        st.markdown("#### 2. Provide sample documents")
        uploads = st.file_uploader(
            "Upload .txt / .md / .pdf / .xlsx (up to 5 files)",
            accept_multiple_files=True,
            type=["txt", "md", "pdf", "xlsx"],
            key="onboard_uploads",
        )
        if uploads:
            from doc_pipeline.document_ingestion import DocumentIngestion
            ing = DocumentIngestion()
            docs: List[str] = []
            names: List[str] = []
            for uf in uploads[:5]:
                names.append(uf.name)
                suffix = "." + uf.name.rsplit(".", 1)[-1].lower()
                if suffix in (".txt", ".md"):
                    docs.append(uf.read().decode("utf-8", errors="ignore"))
                else:
                    # PDF / XLSX — drop to disk and use existing parsers.
                    with tempfile.NamedTemporaryFile(
                        suffix=suffix, delete=False
                    ) as tf:
                        tf.write(uf.read())
                        tmp_path = tf.name
                    try:
                        parsed = ing.ingest_file(tmp_path)
                        docs.append("\n\n".join(p.content for p in parsed))
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
            state["docs"] = docs
            state["doc_names"] = names

        pasted = st.text_area(
            "Or paste raw text (treated as one extra document)",
            height=150,
            placeholder="Paste a representative sample of your domain text…",
            key="onboard_paste",
        )
        if pasted.strip():
            existing = list(state["docs"])
            existing.append(pasted)
            state["docs"] = existing
            state["doc_names"].append("(pasted)")

        if state["docs"]:
            st.caption(
                f"Loaded {len(state['docs'])} doc(s), "
                f"{sum(len(d) for d in state['docs']):,} chars total"
            )

        st.markdown("#### 3. Analyse")
        if st.button(
            "🤖 Analyse & propose schema",
            use_container_width=True,
            disabled=not (state["domain_id"] and state["docs"]),
        ):
            try:
                from core.onboarding_agent import analyze
                with st.spinner(f"Authoring schema with {ONBOARDING_MODEL}…"):
                    resp = analyze(
                        state["domain_id"],
                        state["docs"],
                        domain_hint=state["domain_hint"],
                        prior_qa=state["prior_qa"],
                    )
                state["agent_response"] = resp.to_dict()
                state["current_answers"] = {}
            except Exception as e:
                st.error(f"Onboarding failed: {e}")

        if st.button("🔄 Reset wizard", use_container_width=True):
            for k in list(state.keys()):
                if k != "domain_id":
                    state[k] = [] if isinstance(state[k], list) else (
                        {} if isinstance(state[k], dict) else (
                            None if state[k] is None else ""
                        )
                    )
            st.rerun()

    # ── Output column ──────────────────────────────────────────────────
    with output_col:
        resp = state["agent_response"]
        if not resp:
            st.info(
                "No analysis yet. Provide a domain id + sample docs on the "
                "left, then click **Analyse & propose schema**."
            )
            return

        # Analysis
        if resp.get("analysis"):
            st.markdown("#### Agent analysis")
            st.markdown(f"> {resp['analysis']}")

        # Discovered entities
        discovered = resp.get("discovered_entities") or {}
        if discovered:
            st.markdown("#### Discovered entities")
            for et, vals in discovered.items():
                st.markdown(
                    f"- **{et}** ({len(vals)}): "
                    + ", ".join(f"`{v}`" for v in vals[:12])
                    + ("…" if len(vals) > 12 else "")
                )

        # Follow-up questions
        follow_ups = resp.get("follow_up_questions") or []
        if follow_ups and not resp.get("ready_to_generate"):
            st.markdown("#### Follow-up questions")
            st.caption(
                "Answer what you can, then **Submit answers & re-run** — or hit "
                "**Skip & generate now** to take a best-effort schema from "
                "current evidence and iterate later."
            )
            for i, q in enumerate(follow_ups):
                state["current_answers"][i] = st.text_area(
                    f"Q{i+1}. {q}",
                    value=state["current_answers"].get(i, ""),
                    key=f"onboard_answer_{i}",
                    height=70,
                )
            submit_col, skip_col = st.columns(2)
            with submit_col:
                if st.button("➡️ Submit answers & re-run", use_container_width=True):
                    pairs = [
                        {"question": q, "answer": state["current_answers"].get(i, "").strip()}
                        for i, q in enumerate(follow_ups)
                    ]
                    state["prior_qa"].extend([p for p in pairs if p["answer"]])
                    try:
                        from core.onboarding_agent import analyze
                        with st.spinner(f"Re-running with your answers…"):
                            resp2 = analyze(
                                state["domain_id"],
                                state["docs"],
                                domain_hint=state["domain_hint"],
                                prior_qa=state["prior_qa"],
                            )
                        state["agent_response"] = resp2.to_dict()
                        state["current_answers"] = {}
                        st.rerun()
                    except Exception as e:
                        st.error(f"Re-run failed: {e}")
            with skip_col:
                if st.button("⏩ Skip & generate now", use_container_width=True,
                             help="Force the agent to emit a YAML even if it still has questions."):
                    try:
                        from core.onboarding_agent import analyze
                        with st.spinner("Force-generating best-effort schema…"):
                            resp2 = analyze(
                                state["domain_id"],
                                state["docs"],
                                domain_hint=state["domain_hint"],
                                prior_qa=state["prior_qa"],
                                force_generate=True,
                            )
                        state["agent_response"] = resp2.to_dict()
                        state["current_answers"] = {}
                        st.rerun()
                    except Exception as e:
                        st.error(f"Force-generate failed: {e}")

        # YAML preview + validation report
        if resp.get("yaml"):
            preview_col, dl_col = st.columns([8, 2])
            with preview_col:
                st.markdown("#### Proposed schema YAML")
            with dl_col:
                # Always available — even if validation fails the user can
                # grab the YAML, hand-edit it, and re-run the pipeline.
                st.download_button(
                    label="⬇️ Download",
                    data=resp["yaml"],
                    file_name=f"{state['domain_id'] or 'new_domain'}.yaml",
                    mime="text/yaml",
                    use_container_width=True,
                    key="onboard_dl",
                )
            st.code(resp["yaml"], language="yaml")

            val = resp.get("validation") or {}
            if val:
                ok = val.get("all_passed")
                if ok:
                    st.success("✅ All validation gates passed.")
                else:
                    st.warning("⚠️ Validation found issues — fix before saving.")
                with st.expander("Validation report"):
                    st.json(val)
            sc = resp.get("self_check") or {}
            if sc:
                with st.expander("Agent self-check"):
                    st.json(sc)

            # ── Save + rebuild buttons ──
            save_col, rebuild_col = st.columns(2)
            with save_col:
                if st.button(
                    "💾 Save schema",
                    use_container_width=True,
                    disabled=not (val.get("all_passed") if val else False),
                ):
                    try:
                        from core.onboarding_agent import save_schema
                        dest = save_schema(state["domain_id"], resp["yaml"])
                        state["saved_to"] = str(dest)
                        st.success(f"Wrote {dest}")
                    except Exception as e:
                        st.error(f"Save failed: {e}")
            with rebuild_col:
                if st.button(
                    "🏗️ Save & build KG",
                    use_container_width=True,
                    disabled=not (val.get("all_passed") if val else False),
                    help="Saves the schema then runs scripts/onboard_domain.sh --rebuild-only",
                ):
                    try:
                        from core.onboarding_agent import save_schema
                        dest = save_schema(state["domain_id"], resp["yaml"])
                        state["saved_to"] = str(dest)
                        # Domain input dir must exist before the script runs.
                        from config import input_dir
                        Path(input_dir(state["domain_id"])).mkdir(
                            parents=True, exist_ok=True,
                        )
                    except Exception as e:
                        st.error(f"Save failed: {e}")
                    else:
                        with st.spinner("Building Qdrant index + KG…"):
                            try:
                                out = _sub.run(
                                    [
                                        "bash",
                                        "scripts/onboard_domain.sh",
                                        "--domain", state["domain_id"],
                                        "--rebuild-only",
                                    ],
                                    capture_output=True, text=True, timeout=900,
                                )
                                state["rebuild_output"] = (
                                    (out.stdout or "") + "\n" + (out.stderr or "")
                                )[-4000:]
                                if out.returncode != 0:
                                    st.error(
                                        f"Build script exited {out.returncode}. "
                                        f"See output below."
                                    )
                                else:
                                    st.success(
                                        "✅ Domain built. Restart `./run.sh` "
                                        "so the running UIs auto-discover it."
                                    )
                            except _sub.TimeoutExpired:
                                state["rebuild_output"] = "(build timed out after 15 min)"
                                st.error("Build timed out.")

            if state.get("saved_to"):
                st.caption(f"Last saved → `{state['saved_to']}`")
            if state.get("rebuild_output"):
                with st.expander("Build output"):
                    st.code(state["rebuild_output"])


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
        "🌱 Onboard Domain",
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
        tab_onboard_domain()
    with tabs[7]:
        tab_architecture()


main()
