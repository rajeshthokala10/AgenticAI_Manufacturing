"""Generate the Hybrid GraphRAG Manufacturing system design PDF.

Run from the repo root:

    python system_design/generate_diagram.py

Output:

    system_design/system_architecture.pdf

The PDF is an 8-page design document:

  Page 1 — Top-level architecture diagram
           Clients · API · Orchestration · NLU · Retrieval · LLMs ·
           Persistence · Knowledge Graph · Ingestion (with versions
           and key environment variables).

  Page 2 — Diagnostic-mode flow (LangGraph topology)
           START → format → retrieve → [rank_causes] → generate →
           critic → retry → END, plus a per-node reference table.

  Page 3 — Cost & latency breakdown
           Per-mode summary (Quick / Diagnostic / Chat / Classical RAG /
           Direct LLM), per-stage Diagnostic detail, and a
           cloud-vs-local pricing comparison.

  Page 4 — Human-in-the-Loop (HITL) approval gate
           criticality_check / human_approval topology, risk-score
           drivers, REST surface, and the pipeline status state machine.

  Page 5 — Low-level component sequence  (RBAC + Approvals UI)
           Lane-by-lane walk through a real $5,000 PO from the
           operator's chat box through auth, criticality classifier,
           checkpointer, the Next.js Approvals tab, and the audit log.

  Page 6 — Component interaction contracts
           Edge-level catalogue: from → to · payload · auth · failure
           modes · p50 latency. The single source of truth for every
           in-process and HTTP boundary in the system.

  Page 7 — Role-Based Knowledge-Base ACLs
           Three-tier document classification, role → tier read-sets,
           ingest-time tagging, ContextVar-scoped retriever filter, and
           the operator-vs-plant-manager evidence delta.

  Page 8 — Advanced patterns: rerank · cache · parallel · guardrails ·
           tools · offline eval. The six production-hardening layers
           that wrap the core engine, with flow diagrams, feature flags,
           and the new request flow that incorporates all of them.
"""

from __future__ import annotations

import math
from pathlib import Path

from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas


OUTPUT = Path(__file__).resolve().parent / "system_architecture.pdf"


# ─── Palette ────────────────────────────────────────────────────────────────

TITLE_COLOR = HexColor("#0F172A")
SUBTITLE_COLOR = HexColor("#334155")
PAGE_BG = HexColor("#FAFAFA")
ARROW_COLOR = HexColor("#1E293B")
TABLE_HEADER_BG = HexColor("#0F172A")
TABLE_ALT_ROW = HexColor("#F8FAFC")
TABLE_BORDER = HexColor("#94A3B8")
ACCENT = HexColor("#0EA5E9")

CLIENT = (HexColor("#E0E7FF"), HexColor("#4338CA"))
API_PAL = (HexColor("#FEF3C7"), HexColor("#B45309"))
ORCH = (HexColor("#DCFCE7"), HexColor("#15803D"))
NLU = (HexColor("#FDE2E2"), HexColor("#B91C1C"))
RET = (HexColor("#CFFAFE"), HexColor("#0E7490"))
LLM = (HexColor("#F3E8FF"), HexColor("#7E22CE"))
STORE = (HexColor("#F1F5F9"), HexColor("#475569"))
INGEST = (HexColor("#FFE4E6"), HexColor("#9F1239"))
OPTIONAL = (HexColor("#FEF9C3"), HexColor("#A16207"))


# ─── Generic primitives ─────────────────────────────────────────────────────


def draw_box(c, x, y, w, h, title, items, palette, dashed=False):
    fill, border = palette
    c.setFillColor(fill)
    c.setStrokeColor(border)
    c.setLineWidth(1.2)
    if dashed:
        c.setDash([4, 2], 0)
    c.roundRect(x, y, w, h, 6, fill=1, stroke=1)
    c.setDash([], 0)

    c.setFillColor(border)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x + 8, y + h - 14, title)

    c.setFillColor(black)
    c.setFont("Helvetica", 8)
    line_y = y + h - 28
    for item in items:
        c.drawString(x + 10, line_y, "\u2022 " + item)
        line_y -= 11


def draw_arrow(c, x1, y1, x2, y2, dashed=False, label=None, color=None):
    color = color or ARROW_COLOR
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.0)
    if dashed:
        c.setDash([3, 2], 0)
    c.line(x1, y1, x2, y2)
    c.setDash([], 0)

    angle = math.atan2(y2 - y1, x2 - x1)
    head = 7
    ax = x2 - head * math.cos(angle - math.pi / 8)
    ay = y2 - head * math.sin(angle - math.pi / 8)
    bx = x2 - head * math.cos(angle + math.pi / 8)
    by = y2 - head * math.sin(angle + math.pi / 8)
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(ax, ay)
    p.lineTo(bx, by)
    p.close()
    c.drawPath(p, fill=1, stroke=0)

    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 + 5
        c.setFillColor(SUBTITLE_COLOR)
        c.setFont("Helvetica-Oblique", 7)
        c.drawCentredString(mx, my, label)


def draw_table(
    c,
    x,
    y,
    headers,
    rows,
    col_widths,
    row_height=15,
    header_height=20,
    font_size=7.5,
    header_font_size=8,
):
    """Draw a table with the top-left corner at ``(x, y)``.

    Returns the y-coordinate of the bottom of the table so the caller can
    keep stacking content below it.
    """
    total_w = sum(col_widths)

    # Header row
    cur_y = y - header_height
    c.setFillColor(TABLE_HEADER_BG)
    c.rect(x, cur_y, total_w, header_height, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", header_font_size)
    cx = x
    for hdr, w in zip(headers, col_widths):
        c.drawString(cx + 5, cur_y + 6, str(hdr))
        cx += w

    # Body rows
    c.setFont("Helvetica", font_size)
    for i, row in enumerate(rows):
        cur_y -= row_height
        if i % 2 == 0:
            c.setFillColor(TABLE_ALT_ROW)
            c.rect(x, cur_y, total_w, row_height, fill=1, stroke=0)
        c.setFillColor(black)
        cx = x
        for cell, w in zip(row, col_widths):
            text = str(cell)
            # Hard-truncate to keep cells from overflowing.
            max_chars = int((w - 8) / (font_size * 0.55))
            if len(text) > max_chars:
                text = text[: max_chars - 1] + "\u2026"
            c.drawString(cx + 5, cur_y + 4, text)
            cx += w

    # Outer border
    c.setStrokeColor(TABLE_BORDER)
    c.setLineWidth(0.5)
    c.rect(x, cur_y, total_w, y - cur_y, fill=0, stroke=1)
    return cur_y


def draw_page_header(c, title, subtitle, page_num, total_pages):
    page_w, page_h = landscape(letter)
    c.setFillColor(PAGE_BG)
    c.rect(0, 0, page_w, page_h, fill=1, stroke=0)

    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(36, page_h - 36, title)

    c.setFillColor(SUBTITLE_COLOR)
    c.setFont("Helvetica", 9.5)
    c.drawString(36, page_h - 52, subtitle)

    c.setFont("Helvetica-Oblique", 8)
    c.drawRightString(page_w - 36, page_h - 36, f"page {page_num} / {total_pages}")


def draw_page_footer(c, footer_text):
    page_w, _ = landscape(letter)
    c.setFillColor(SUBTITLE_COLOR)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(36, 22, footer_text)
    c.drawRightString(
        page_w - 36, 22, "Generated by system_design/generate_diagram.py"
    )


# ─── Page 1 — Architecture diagram ──────────────────────────────────────────


def draw_page1(c):
    page_w, page_h = landscape(letter)
    draw_page_header(
        c,
        "Hybrid GraphRAG Manufacturing — System Architecture",
        "Multi-turn chat \u2022 LangGraph-optional orchestration \u2022 "
        "Hybrid retrieval (BM25 + FAISS + KG) \u2022 "
        "Cause-ranker \u2022 Critic + deterministic guardrails \u2022 ERP/MES tools (see p. 8)",
        page_num=1,
        total_pages=8,
    )

    # ─── Clients lane ──────────────────────────────────────────────────────
    y_clients = page_h - 120
    draw_box(
        c, 40, y_clients, 230, 64,
        "Next.js 14 Web UI  (web/)",
        [
            "App Router · TypeScript · Tailwind",
            "react-markdown · highlight.js",
            "Calls /api/* via next.config rewrites",
        ],
        CLIENT,
    )
    draw_box(
        c, 285, y_clients, 230, 64,
        "Streamlit \u22651.28  (app.py)",
        [
            "\U0001F4AC Chat tab (multi-turn slot-filling)",
            "Analytics dashboard (6 tabs)",
            "Plotly visualisations \u22655.18",
        ],
        CLIENT,
    )
    draw_box(
        c, 530, y_clients, 230, 64,
        "CLI / Python API",
        [
            "main.py \u00b7 from pipeline import \u2026",
            "ManufacturingPipeline \u00b7 ChatAgent",
            "Direct in-process; no HTTP",
        ],
        CLIENT,
    )

    # ─── API lane ──────────────────────────────────────────────────────────
    y_api = y_clients - 88
    draw_box(
        c, 200, y_api, 400, 60,
        "FastAPI \u22650.110  (api/server.py) · uvicorn \u22650.27",
        [
            "POST /api/chat  \u00b7  POST /api/reset",
            "GET  /api/health  \u00b7  /api/stats  \u00b7  /api/sessions/{id}",
            "Pydantic v2 models \u00b7 in-memory session store \u00b7 CORS-enabled",
        ],
        API_PAL,
    )
    draw_arrow(c, 155, y_clients, 290, y_api + 60, label="HTTP :8000")
    draw_arrow(c, 400, y_clients, 400, y_api + 60)
    draw_arrow(c, 645, y_clients, 510, y_api + 60, label="in-process")

    # ─── Orchestration lane ───────────────────────────────────────────────
    y_orch = y_api - 100
    draw_box(
        c, 30, y_orch, 240, 78,
        "ChatAgent  (pipeline/chat_agent.py)",
        [
            "Multi-turn conversation state",
            "Slot-filling: required \u2192 optional",
            "Resets \u00b7 skip tokens \u00b7 per-session",
        ],
        ORCH,
    )
    draw_box(
        c, 290, y_orch, 230, 78,
        "ManufacturingPipeline  (pipeline/)",
        [
            "Mode dispatch: Diagnostic | Quick",
            "build_or_load() \u00b7 stats[]",
            "Picks orchestrator engine",
        ],
        ORCH,
    )
    draw_box(
        c, 540, y_orch, 230, 78,
        "Diagnostic Engine",
        [
            "Default: core/orchestrator.py (procedural)",
            "Opt-in: pipeline/langgraph_orchestrator.py",
            "  USE_LANGGRAPH=true (LangGraph 1.x)",
        ],
        ORCH,
    )

    draw_arrow(c, 400, y_api, 150, y_orch + 78, label="user_msg + session_id")
    draw_arrow(c, 270, y_orch + 39, 290, y_orch + 39)
    draw_arrow(c, 520, y_orch + 39, 540, y_orch + 39)

    # ─── NLU / Retrieval / LLM trio ────────────────────────────────────────
    y_mid = y_orch - 100
    draw_box(
        c, 30, y_mid, 240, 88,
        "Query Understanding",
        [
            "QueryCorrector: spell + acronyms",
            "ClarifierAgent: intent \u00b7 entities \u00b7 slots",
            "Intent classifier \u2014 qwen2.5:3b (Ollama)",
            "format_query() \u2192 expanded + structured",
        ],
        NLU,
    )
    draw_box(
        c, 290, y_mid, 230, 88,
        "Hybrid Retrieval",
        [
            "BM25 (rank-bm25 \u22650.2 + pure-Py fallback)",
            "FAISS \u22651.7 (all-MiniLM-L6-v2, 384-dim)",
            "KG traversal (NetworkX \u22653.0)",
            "RRF fusion: K=60, top_k=5/10",
        ],
        RET,
    )
    draw_box(
        c, 540, y_mid, 230, 88,
        "Tiered LLMs  (core/llm_client.py)",
        [
            "gpt-4o          \u2192 answer + retry          (OpenAI)",
            "qwen2.5:3b   \u2192 critic + classifier      (Ollama)",
            "qwen2.5:3b   \u2192 cause-ranker (opt-in)  ",
            "gpt-4o-mini \u2192 comparison baselines",
        ],
        LLM,
    )

    draw_arrow(c, 130, y_orch, 130, y_mid + 88, label="raw query")
    draw_arrow(c, 200, y_mid + 88, 200, y_orch, label="normalised")
    draw_arrow(c, 400, y_orch, 400, y_mid + 88, label="search_query")
    draw_arrow(c, 470, y_mid + 88, 470, y_orch, label="top-k + KG paths")
    draw_arrow(c, 640, y_orch, 640, y_mid + 88, label="prompt+context")
    draw_arrow(c, 710, y_mid + 88, 710, y_orch, label="answer + critique")

    # ─── Persistence + Ingestion lane ─────────────────────────────────────
    y_store = y_mid - 92
    draw_box(
        c, 30, y_store, 240, 64,
        "Persistence",
        [
            "data/processed/  (chunks JSON \u00b7 KG JSON)",
            "doc_pipeline/vector_store/  (faiss.index)",
            "Session state: in-memory dict",
        ],
        STORE,
    )
    draw_box(
        c, 290, y_store, 230, 64,
        "Knowledge Graph",
        [
            "NetworkX DiGraph",
            "Entities: Equipment\u00b7Component\u00b7Alarm\u00b7\u2026",
            "Relations: TRIGGERS_ALARM \u00b7 RESOLVED_BY \u00b7 \u2026",
        ],
        STORE,
    )
    draw_box(
        c, 540, y_store, 230, 64,
        "Document Ingestion  (doc_pipeline/)",
        [
            "pdfplumber \u22650.10 \u00b7 openpyxl \u22653.1 \u00b7 pandas \u22652.0",
            "Semantic + recursive + sliding chunking",
            "KG builder \u00b7 sentence-transformers \u22652.2",
        ],
        INGEST,
    )

    draw_arrow(c, 155, y_mid, 155, y_store + 64, dashed=True)
    draw_arrow(c, 400, y_mid, 400, y_store + 64, dashed=True, label="reads")
    draw_arrow(c, 660, y_store + 64, 660, y_mid, label="builds index + KG")

    # ─── Legend ───────────────────────────────────────────────────────────
    c.setFillColor(SUBTITLE_COLOR)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(36, 70, "Legend")
    c.setFont("Helvetica", 7.5)
    c.setFillColor(black)
    c.drawString(36, 58, "Solid arrow \u2014 runtime data flow")
    c.drawString(36, 48, "Dashed arrow \u2014 reads from / writes to persisted state")

    draw_page_footer(
        c,
        "Default ports: API 8000 \u00b7 Streamlit 8501 \u00b7 Next.js 3000  "
        "\u2014  Stack: Python \u22653.10 \u00b7 Node \u226518",
    )


# ─── Page 2 — Diagnostic flow + LangGraph topology ──────────────────────────


def draw_page2(c):
    page_w, page_h = landscape(letter)
    draw_page_header(
        c,
        "Diagnostic-Mode Flow  \u2014  LangGraph Topology",
        "USE_LANGGRAPH=true routes every diagnostic query through "
        "this StateGraph. The procedural orchestrator follows the "
        "same logical flow.",
        page_num=2,
        total_pages=8,
    )

    # ─── Top half: graph topology ─────────────────────────────────────────
    cy = page_h - 145

    def node(c, cx, cy, w, h, label, sub, palette, dashed=False):
        draw_box(c, cx - w / 2, cy - h / 2, w, h, label, [sub] if sub else [],
                 palette, dashed=dashed)

    # START circle
    c.setFillColor(HexColor("#1E293B"))
    c.circle(80, cy, 14, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(80, cy - 3, "START")

    # Nodes
    node(c, 175, cy, 110, 50, "format",
         "format_query()  \u2022  intent + entities", NLU)
    node(c, 305, cy, 110, 50, "retrieve",
         "BM25 + FAISS + KG  \u2022  RRF", RET)
    node(c, 440, cy, 130, 50, "rank_causes  (optional)",
         "qwen2.5:3b  \u00b7  intent-gated", OPTIONAL, dashed=True)
    node(c, 590, cy, 110, 50, "generate",
         "gpt-4o (ANSWER_MODEL)", LLM)
    node(c, 720, cy, 110, 50, "critic",
         "qwen2.5:3b (CRITIC_MODEL)", LLM)

    # retry node below critic
    node(c, 720, cy - 90, 110, 50, "retry",
         "gpt-4o (RETRY_MODEL)", LLM)

    # END circle
    c.setFillColor(HexColor("#1E293B"))
    c.circle(720, cy - 175, 14, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(720, cy - 178, "END")

    # Edges
    draw_arrow(c, 94, cy, 120, cy)
    draw_arrow(c, 230, cy, 250, cy)
    draw_arrow(c, 360, cy, 375, cy, label="if USE_CAUSE_RANKING")
    draw_arrow(c, 505, cy, 535, cy, label="ranked causes")
    draw_arrow(c, 645, cy, 665, cy)
    # bypass arrow (skip rank_causes)
    draw_arrow(c, 360, cy + 30, 535, cy + 30, dashed=True,
               label="otherwise")
    draw_arrow(c, 720, cy - 25, 720, cy - 65, label="FAIL & attempts<MAX")
    draw_arrow(c, 665, cy - 90, 535, cy, dashed=False, label="re-evaluate")
    # Actually retry should loop back to critic — adjust:
    # retry → critic edge (up the right side back into critic)
    draw_arrow(c, 720, cy - 65, 720, cy - 25)
    # critic → END (PASS or attempts==MAX)
    draw_arrow(c, 720, cy - 25, 720, cy - 161, label="PASS / max retries", color=HexColor("#15803D"))

    # ─── Bottom half: per-node reference table ────────────────────────────
    headers = ["Node", "Function", "Model / Tool", "Env vars", "Latency p50", "Cost p50"]
    rows = [
        ["format", "Intent + entity extraction; query expansion",
         "regex \u2192 qwen2.5:3b fallback", "CLASSIFY_MODEL", "5–60 ms", "~$0  (local)"],
        ["retrieve", "BM25 + FAISS + KG traversal \u2192 RRF top-k",
         "rank-bm25 + FAISS + NetworkX", "TOP_K_RETRIEVAL=10  TOP_K_RERANK=5  RRF_K=60",
         "50–300 ms", "$0  (no LLM)"],
        ["rank_causes (opt-in)", "Score top-N root causes from evidence + KG",
         "CAUSE_RANK_MODEL  qwen2.5:3b", "USE_CAUSE_RANKING  CAUSE_RANK_TOP_K=5",
         "100–500 ms", "$0  (Ollama)"],
        ["generate", "First-pass evidence-grounded answer",
         "ANSWER_MODEL  gpt-4o", "OPENAI_API_KEY", "1.5–3 s", "~$0.0050"],
        ["critic", "Grounding / completeness / safety check",
         "CRITIC_MODEL  qwen2.5:3b", "—", "50–250 ms", "$0  (Ollama)"],
        ["retry", "Regenerate using critic feedback",
         "RETRY_MODEL  gpt-4o", "MAX_CRITIC_RETRIES=2", "1.5–3 s",
         "~$0.0050  (per retry)"],
    ]
    col_widths = [85, 195, 130, 175, 75, 80]
    table_y = cy - 225
    draw_table(c, 36, table_y, headers, rows, col_widths)

    draw_page_footer(
        c,
        "Default routing: PASS \u2192 END  \u00b7  FAIL & attempts<MAX_CRITIC_RETRIES \u2192 retry "
        "\u2192 critic  \u00b7  FAIL & attempts==MAX \u2192 END",
    )


# ─── Page 3 — Cost & latency breakdown ──────────────────────────────────────


def draw_page3(c):
    page_w, page_h = landscape(letter)
    draw_page_header(
        c,
        "Cost & Latency Breakdown",
        "Per-query estimates with the default tiered routing "
        "(answer = gpt-4o, critic = qwen2.5:3b on Ollama). "
        "Local models are free; OpenAI pricing per core/llm_client.py.",
        page_num=3,
        total_pages=8,
    )

    # ─── Section 1: Per-mode summary ──────────────────────────────────────
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, page_h - 88, "1.  Per-mode summary  (typical query, default config)")

    headers = ["Mode", "LLM calls per query", "Tokens (in / out)",
               "Cost / query", "Latency p50", "Notes"]
    rows = [
        ["Quick Search", "0",
         "0 / 0", "$0.0000",
         "120–400 ms", "FAISS + embeddings only; works fully offline"],
        ["Diagnostic (default)",
         "1 answer + 1 critic",
         "~1500 / ~400 + ~700 / ~150",
         "~$0.0080",
         "2.0–4.5 s", "ANSWER on gpt-4o, CRITIC on qwen2.5:3b (free)"],
        ["Diagnostic + cause-ranker",
         "+1 cause-ranker",
         "+~1200 / ~250",
         "+ $0.0000",
         "+100–500 ms", "USE_CAUSE_RANKING=true; intent-gated"],
        ["Diagnostic worst-case",
         "1 answer + 2 critics + 1 retry",
         "~3700 / ~950",
         "~$0.0160",
         "4.5–9 s", "Critic FAILs once, retry resolves it"],
        ["Chat (multi-turn)",
         "Same as Diagnostic",
         "Same as Diagnostic",
         "Same as Diagnostic",
         "Same as Diagnostic",
         "Plus per-turn slot-filling (no LLM)"],
        ["Classical RAG (baseline)",
         "1",
         "~1500 / ~400",
         "~$0.0005",
         "1.0–2.0 s", "FAISS-only retrieval, gpt-4o-mini answer"],
        ["Direct LLM (baseline)",
         "1",
         "~150 / ~400",
         "~$0.0003",
         "0.8–1.5 s", "No retrieval; gpt-4o-mini parametric only"],
    ]
    col_widths = [165, 130, 150, 75, 80, 120]
    y_after = draw_table(c, 36, page_h - 96, headers, rows, col_widths,
                         row_height=16, header_height=20)

    # ─── Section 2: Per-stage Diagnostic detail ───────────────────────────
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, y_after - 22,
                 "2.  Per-stage detail for Diagnostic mode "
                 "(USE_CAUSE_RANKING=true, 1 retry path)")

    headers2 = ["Stage", "Model", "Provider", "Tokens (in / out)",
                "Cost / call", "Latency p50", "Mandatory?"]
    rows2 = [
        ["format", "regex \u2192 qwen2.5:3b", "Ollama (fallback)",
         "0–250 / 0–80", "$0", "5–60 ms", "yes"],
        ["retrieve", "—", "FAISS + BM25 + KG",
         "0 / 0", "$0", "50–300 ms", "yes"],
        ["rank_causes", "qwen2.5:3b", "Ollama",
         "~1200 / ~250", "$0", "100–500 ms",
         "USE_CAUSE_RANKING + intent"],
        ["generate", "gpt-4o", "OpenAI",
         "~1500 / ~400", "~$0.0050", "1.5–3 s", "yes"],
        ["critic (1st)", "qwen2.5:3b", "Ollama",
         "~700 / ~150", "$0", "50–250 ms", "yes"],
        ["retry (if FAIL)", "gpt-4o", "OpenAI",
         "~1700 / ~400", "~$0.0050", "1.5–3 s",
         "FAIL & attempts<MAX"],
        ["critic (2nd)", "qwen2.5:3b", "Ollama",
         "~700 / ~150", "$0", "50–250 ms", "after retry"],
    ]
    col_widths2 = [115, 120, 115, 125, 75, 90, 80]
    y_after2 = draw_table(c, 36, y_after - 30, headers2, rows2, col_widths2,
                          row_height=15, header_height=20)

    # ─── Section 3: Cloud-vs-local pricing comparison ─────────────────────
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, y_after2 - 22,
                 "3.  Cloud vs local pricing  (per 1k tokens, from core/llm_client.py)")

    headers3 = ["Model role", "Default", "OpenAI in / out", "Local equivalent",
                "Switch via .env", "Net effect"]
    rows3 = [
        ["Answer / Retry", "gpt-4o",
         "$0.0025 / $0.010",
         "qwen2.5:7b on Ollama",
         "ANSWER_MODEL=qwen2.5:7b",
         "100% free, slower latency"],
        ["Critic", "qwen2.5:3b",
         "n/a (local)",
         "—",
         "CRITIC_MODEL=gpt-4o-mini",
         "$0 \u2192 ~$0.0001 / call"],
        ["Cause-ranker (opt-in)", "qwen2.5:3b",
         "n/a (local)",
         "—",
         "CAUSE_RANK_MODEL=gpt-4o-mini",
         "$0 \u2192 ~$0.0002 / call"],
        ["Comparison (Direct/Classical)", "gpt-4o-mini",
         "$0.00015 / $0.0006",
         "qwen2.5:3b on Ollama",
         "DIRECT_LLM_MODEL / CLASSICAL_RAG_MODEL",
         "Free at slight quality cost"],
        ["Embeddings", "MiniLM-L6-v2",
         "n/a (local)", "—", "EMBEDDING_MODEL=BAAI/bge-small-en-v1.5",
         "Rebuild FAISS after change"],
    ]
    col_widths3 = [155, 100, 110, 130, 175, 105]
    y_after3 = draw_table(c, 36, y_after2 - 30, headers3, rows3, col_widths3,
                          row_height=15, header_height=20)

    draw_page_footer(
        c,
        "Token estimates assume 5 evidence chunks @ ~300 tokens; "
        "actual usage scales linearly with TOP_K_RERANK and chunk size. "
        "Latency excludes retrieval cache hits.",
    )


# ─── Page 4 — HITL approval gate ───────────────────────────────────────────


def draw_page4(c):
    page_w, page_h = landscape(letter)
    draw_page_header(
        c,
        "Human-in-the-Loop (HITL) Approval Gate",
        "criticality_check + human_approval (interrupt) \u2022 SQLite checkpointer "
        "\u2022 audit log \u2022 USE_HITL=true",
        page_num=4,
        total_pages=8,
    )

    # ── Topology (rewritten as a clean two-tier flow) ──────────────────────
    #
    # Row 1 is the linear graph from START up to generate. Row 2 is the
    # branch tree: a decision diamond on `criticality_check`, the
    # interrupt-bordered `human_approval` node, the final `critic` node
    # plus the two terminal END pills (one for the rejected short-circuit,
    # one for the PASS path through critic).
    #
    # Every coordinate is anchored to `top_y` so the whole block can be
    # nudged vertically without disturbing the relative geometry.

    title_y = page_h - 88
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 11.5)
    c.drawString(36, title_y, "LangGraph topology with the HITL gate")
    c.setFillColor(SUBTITLE_COLOR)
    c.setFont("Helvetica-Oblique", 8.5)
    c.drawString(
        36, title_y - 13,
        "START \u2192 linear pipeline \u2192 criticality_check decision \u2192 "
        "auto-approve OR human_approval (interrupt) \u2192 critic \u2192 END",
    )

    # Palette tuned for print (muted, high-contrast on grey paper bg).
    COL_COMPUTE  = HexColor("#1F2A8E")  # blue – deterministic graph nodes
    COL_LLM      = HexColor("#5B21B6")  # purple – cause ranker (optional)
    COL_GEN      = HexColor("#0E7490")  # cyan – answer generation
    COL_DECISION = HexColor("#B45309")  # amber – decision diamond
    COL_INTR     = HexColor("#B91C1C")  # red – human interrupt
    COL_CRITIC   = HexColor("#15803D")  # green – critic
    COL_END      = HexColor("#334155")  # slate – terminal
    ARROW_GREY   = HexColor("#475569")

    # Reusable primitives ---------------------------------------------------

    def pipe_node(label, x_left, y_center, width, height, color,
                  dashed=False, sub=None):
        """Rounded rectangle with centred white bold label (+ optional sub)."""
        c.setFillColor(color)
        c.setStrokeColor(color)
        c.setLineWidth(1.0)
        if dashed:
            c.setDash([3, 2], 0)
        c.roundRect(x_left, y_center - height / 2, width, height, 5,
                    fill=1, stroke=1)
        c.setDash([], 0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 9)
        if sub:
            c.drawCentredString(x_left + width / 2, y_center + 2, label)
            c.setFont("Helvetica", 7)
            c.drawCentredString(x_left + width / 2, y_center - 8, sub)
        else:
            c.drawCentredString(x_left + width / 2, y_center - 3, label)

    def terminal_pill(label, cx, cy, color=COL_END):
        """START / END pill — slate background, white serif label."""
        w, h = 50, 20
        c.setFillColor(color)
        c.setStrokeColor(color)
        c.roundRect(cx - w / 2, cy - h / 2, w, h, 10, fill=1, stroke=1)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(cx, cy - 3, label)

    def diamond(label, sub, cx, cy, half_w, half_h, color):
        """Decision diamond with two-line label, white text."""
        c.setFillColor(color)
        c.setStrokeColor(color)
        c.setLineWidth(1.0)
        p = c.beginPath()
        p.moveTo(cx, cy + half_h)
        p.lineTo(cx + half_w, cy)
        p.lineTo(cx, cy - half_h)
        p.lineTo(cx - half_w, cy)
        p.close()
        c.drawPath(p, fill=1, stroke=1)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(cx, cy + 2, label)
        c.setFont("Helvetica", 7)
        c.drawCentredString(cx, cy - 8, sub)

    def arrow(x1, y1, x2, y2, label=None, label_above=True,
              color=ARROW_GREY, dashed=False):
        """Straight arrow with a closed arrowhead and optional italic label."""
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.1)
        if dashed:
            c.setDash([3, 2], 0)
        c.line(x1, y1, x2, y2)
        c.setDash([], 0)
        angle = math.atan2(y2 - y1, x2 - x1)
        head = 6.5
        ax = x2 - head * math.cos(angle - math.pi / 8)
        ay = y2 - head * math.sin(angle - math.pi / 8)
        bx = x2 - head * math.cos(angle + math.pi / 8)
        by = y2 - head * math.sin(angle + math.pi / 8)
        p = c.beginPath()
        p.moveTo(x2, y2)
        p.lineTo(ax, ay)
        p.lineTo(bx, by)
        p.close()
        c.drawPath(p, fill=1, stroke=0)
        if label:
            c.setFillColor(TITLE_COLOR)
            c.setFont("Helvetica-Oblique", 7.5)
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2 + (4 if label_above else -8)
            c.drawCentredString(mx, my, label)

    # ── Row 1: linear pipeline ─────────────────────────────────────────────
    row1_y = title_y - 45
    start_cx = 56
    NODE_H = 28
    pipeline_nodes = [
        ("format",          76,  COL_COMPUTE),
        ("detect_purchase", 102, COL_COMPUTE),
        ("retrieve",        76,  COL_COMPUTE),
        ("rank_causes",     86,  COL_LLM),
        ("generate",        76,  COL_GEN),
    ]
    gap = 12
    cursor = start_cx + 25 + gap  # right edge of START pill + gap
    centres = []
    for _, w, _ in pipeline_nodes:
        centres.append(cursor + w / 2)
        cursor += w + gap

    terminal_pill("START", start_cx, row1_y)
    arrow(start_cx + 25, row1_y,
          centres[0] - pipeline_nodes[0][1] / 2, row1_y)
    for i, ((label, w, color), cx) in enumerate(zip(pipeline_nodes, centres)):
        pipe_node(label, cx - w / 2, row1_y, w, NODE_H, color,
                  dashed=(label == "rank_causes"))
        if i + 1 < len(pipeline_nodes):
            next_w = pipeline_nodes[i + 1][1]
            arrow(cx + w / 2, row1_y,
                  centres[i + 1] - next_w / 2, row1_y)

    gen_cx = centres[-1]
    gen_bottom = row1_y - NODE_H / 2

    # ── Row 2: decision diamond directly under `generate` ─────────────────
    decision_cy = row1_y - 60
    diamond_half_w, diamond_half_h = 100, 28
    diamond_cx = gen_cx
    diamond_top_y = decision_cy + diamond_half_h
    diamond_bottom_y = decision_cy - diamond_half_h

    diamond("criticality_check",
            "risk.score \u2265 HITL_RISK_THRESHOLD ?",
            diamond_cx, decision_cy, diamond_half_w, diamond_half_h,
            COL_DECISION)

    # Short vertical arrow from generate down to the top of the diamond.
    arrow(gen_cx, gen_bottom, diamond_cx, diamond_top_y)

    # ── Row 3: critic (left) + human_approval (right) ─────────────────────
    row3_y = decision_cy - 60
    critic_w, critic_h = 110, NODE_H
    interrupt_w, interrupt_h = 130, NODE_H
    critic_cx = diamond_cx - 200      # well to the left of the diamond
    interrupt_cx = diamond_cx + 110   # close to the right vertex

    pipe_node("critic", critic_cx - critic_w / 2, row3_y, critic_w, critic_h,
              COL_CRITIC,
              sub="CRITIC_MODEL  \u00b7  qwen2.5:3b")
    pipe_node("human_approval",
              interrupt_cx - interrupt_w / 2, row3_y,
              interrupt_w, interrupt_h, COL_INTR, dashed=True,
              sub="interrupt() \u00b7 SqliteSaver pause")

    # Diamond → critic   (auto-approve, left branch)
    arrow(diamond_cx - diamond_half_w + 4, decision_cy - 4,
          critic_cx + critic_w / 2, row3_y + critic_h / 2,
          label="auto-approve",
          color=COL_DECISION)
    # Diamond → human_approval (needs_human, right branch)
    arrow(diamond_cx + diamond_half_w - 4, decision_cy - 4,
          interrupt_cx - interrupt_w / 2, row3_y + interrupt_h / 2,
          label="needs_human",
          color=COL_DECISION)

    # human_approval → critic (approved). Horizontal arrow at row 3's
    # centerline passing through the empty space between the two boxes
    # — it never crosses a body because the diamond above leaves a
    # clean corridor exactly here.
    approved_x1 = interrupt_cx - interrupt_w / 2 - 2
    approved_x2 = critic_cx + critic_w / 2 + 2
    arrow(approved_x1, row3_y, approved_x2, row3_y, color=COL_INTR)
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawCentredString((approved_x1 + approved_x2) / 2, row3_y + 5,
                        "approved")

    # ── Row 4: terminal END pills ─────────────────────────────────────────
    row4_y = row3_y - 42
    end_pass_cx = critic_cx
    end_reject_cx = interrupt_cx + interrupt_w / 2 + 50

    # critic → END (PASS). Straight vertical arrow; the label sits in
    # the empty space to the *right* of the arrow so it never lands on
    # the END pill.
    arrow(critic_cx, row3_y - critic_h / 2,
          end_pass_cx, row4_y + 11,
          color=COL_CRITIC)
    c.setFillColor(COL_CRITIC)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(critic_cx + 8, (row3_y - critic_h / 2 + row4_y + 11) / 2 - 2,
                 "PASS / max retries")
    terminal_pill("END", end_pass_cx, row4_y, COL_END)

    # human_approval → END (rejected) — drop right then over.
    rejected_label_x = (interrupt_cx + interrupt_w / 2 + end_reject_cx) / 2
    arrow(interrupt_cx + interrupt_w / 2 + 2, row3_y - 4,
          end_reject_cx - 25, row3_y - 4,
          color=COL_INTR)
    arrow(end_reject_cx - 25 + 12, row3_y - 4,
          end_reject_cx, row4_y + 8,
          label="rejected",
          color=COL_INTR)
    terminal_pill("END", end_reject_cx, row4_y, COL_END)

    # ── Compact legend strip under row 4 ──────────────────────────────────
    legend_y = row4_y - 20
    legend_items = [
        (COL_COMPUTE,  "deterministic"),
        (COL_LLM,      "optional cause-ranker"),
        (COL_GEN,      "answer LLM"),
        (COL_DECISION, "decision (risk gate)"),
        (COL_INTR,     "human interrupt"),
        (COL_CRITIC,   "critic"),
        (COL_END,      "terminal"),
    ]
    c.setFillColor(SUBTITLE_COLOR)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(36, legend_y, "Legend")
    c.setFont("Helvetica", 7.5)
    x_cursor = 76
    for color, label in legend_items:
        c.setFillColor(color)
        c.rect(x_cursor, legend_y - 1, 9, 8, fill=1, stroke=0)
        c.setFillColor(TITLE_COLOR)
        c.drawString(x_cursor + 13, legend_y, label)
        x_cursor += 13 + len(label) * 4.4 + 14

    # Anchor for the risk-drivers section below.
    y = legend_y - 6

    # ── Risk score breakdown table ─────────────────────────────────────────
    # `y` already points just below the topology legend, so the table sits
    # naturally beneath the diagram without any magic offset.
    table_y = y - 8
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, table_y, "Risk score drivers — core/criticality_classifier.py")

    headers = ["Driver", "Trigger", "Score bump", "Domain"]
    rows = [
        ("safety_keyword:*", "Substring of HITL_HIGH_RISK_KEYWORDS in query or proposed answer", "0.55 + 0.05 / extra hit", "diagnostic"),
        ("high_risk_intent", "Clarifier intent ∈ {shutdown, emergency, lockout_tagout, permit_to_work}", "0.90", "diagnostic"),
        ("low_critic_confidence", "Critic verdict.confidence < 0.5", "0.30", "diagnostic"),
        ("purchase_value≥threshold", "PurchaseRequest.total_usd ≥ HITL_AUTO_APPROVE_BELOW_USD", "0.70", "purchase_request"),
        ("single_source_vendor", "KG flags the part with single_source=true", "0.65", "purchase_request"),
        ("long_lead_time", "lead_time_days > 7", "0.55", "purchase_request"),
        ("class_A_equipment", "Used by an Equipment node tagged criticality=A", "0.70", "purchase_request"),
        ("llm_grader:*", "Tier-2 LLM grader (only fires for inconclusive 0.3–0.7 band)", "max(score, llm_score)", "any"),
    ]
    col_widths = [130, 360, 130, 90]
    table_y2 = draw_table(c, 36, table_y - 14, headers, rows, col_widths,
                           font_size=7.5, row_height=13)

    # ── Bottom: API surface + decision flow ────────────────────────────────
    bottom_y = table_y2 - 22
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(36, bottom_y, "REST surface (api/server.py)")
    api_box = [
        ("GET  /api/approvals/pending",            "list paused HITL workflows"),
        ("GET  /api/approvals/{thread_id}",        "snapshot of one paused workflow"),
        ("POST /api/approvals/{thread_id}/resume", "{approved, approver, comments, edited_answer?}"),
        ("GET  /api/approvals/my",                  "buckets pending / pending_for_me / actioned"),
        ("GET  /api/audit?limit=N&offset=M",       "recent decisions + approval-rate stats"),
    ]
    c.setFont("Helvetica", 8)
    for i, (route, desc) in enumerate(api_box):
        c.setFillColor(HexColor("#1E3A8A"))
        c.drawString(36, bottom_y - 14 - i * 11, route)
        c.setFillColor(SUBTITLE_COLOR)
        c.drawString(248, bottom_y - 14 - i * 11, desc)

    # State machine on the right (compact, no overlap with right margin).
    sm_x = 500
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(sm_x, bottom_y, "Pipeline status state machine")
    c.setFillColor(SUBTITLE_COLOR)
    c.setFont("Helvetica", 8)
    sm_lines = [
        "in_progress \u2192 awaiting_approval \u2192 complete   (approved)",
        "in_progress \u2192 awaiting_approval \u2192 rejected   (approved=false)",
        "in_progress \u2192 complete                         (auto-approve / USE_HITL=false)",
        "",
        "Checkpointer: SqliteSaver (HITL_DB_PATH) \u2014 survives restarts.",
        "Audit log:    core/audit_log.py \u2014 append-only, one row per decision.",
    ]
    for i, line in enumerate(sm_lines):
        c.drawString(sm_x, bottom_y - 14 - i * 11, line)

    draw_page_footer(
        c,
        "USE_HITL=true requires USE_LANGGRAPH=true. See system_design/HITL_DESIGN.md "
        "for the full PRD.",
    )


# ─── Page 5 — Low-level component sequence ─────────────────────────────────
#
# Goal: take the reader from the operator's first keystroke to the buyer
# clicking Approve, naming every component that participates and every
# wire that crosses a process boundary. Each numbered step references the
# exact source location so you can grep from this page straight to the
# implementation.

def draw_page5(c):
    page_w, page_h = landscape(letter)
    draw_page_header(
        c,
        "Low-Level Component Sequence  —  $5,000 PO end-to-end",
        "Operator submits a high-risk PO \u2192 LangGraph pauses at the HITL "
        "interrupt \u2192 buyer signs off from the Approvals tab. Each step names "
        "the file and wire payload.",
        page_num=5,
        total_pages=8,
    )

    # ─── Lane setup ────────────────────────────────────────────────────────
    #
    # Eight vertical lanes, drawn left-to-right. Lane centres are quoted so
    # the arrow/step drawing helpers below don't have to repeat the maths.
    LANES = [
        ("Operator UI\nweb/app/page.tsx",          HexColor("#E0E7FF")),
        ("Auth gate\ncomponents/AuthGate.tsx",     HexColor("#FEF3C7")),
        ("FastAPI\napi/server.py",                 HexColor("#FFE4E6")),
        ("Pipeline\nunified_pipeline.py",          HexColor("#DCFCE7")),
        ("LangGraph\nlanggraph_orchestrator.py",   HexColor("#CFFAFE")),
        ("Risk + RBAC\ncriticality · rbac",        HexColor("#FDE2E2")),
        ("Persistence\nSqliteSaver · audit · auth", HexColor("#F1F5F9")),
        ("Buyer UI\nApprovalsTab + banner",        HexColor("#F3E8FF")),
    ]
    lane_top = page_h - 75
    lane_bottom = 130
    lane_count = len(LANES)
    lane_left = 36
    lane_right = page_w - 36
    lane_width = (lane_right - lane_left) / lane_count

    def lane_x(i: int) -> float:
        return lane_left + lane_width * i + lane_width / 2

    # Lane header bands.
    for i, (label, color) in enumerate(LANES):
        x = lane_left + lane_width * i
        c.setFillColor(color)
        c.roundRect(x + 4, lane_top - 38, lane_width - 8, 36, 4,
                    fill=1, stroke=0)
        c.setFillColor(TITLE_COLOR)
        c.setFont("Helvetica-Bold", 7.5)
        for j, line in enumerate(label.split("\n")):
            c.drawCentredString(x + lane_width / 2, lane_top - 14 - j * 9, line)

        # Lane spine — long vertical guide.
        c.setStrokeColor(HexColor("#CBD5E1"))
        c.setLineWidth(0.5)
        c.setDash([2, 3], 0)
        c.line(x + lane_width / 2, lane_top - 44, x + lane_width / 2,
               lane_bottom)
        c.setDash([], 0)

    # ─── Sequence steps ────────────────────────────────────────────────────
    #
    # Each step is (label, from_lane, to_lane). The y-coordinate ticks down
    # 22 pt per step so the diagram reads top-to-bottom like a UML sequence.
    steps = [
        # Login phase.
        ("1. POST /api/auth/login\n   {user_id, password}",            0, 1),
        ("2. signup/login \u2192 auth_store.sqlite\n   bearer 32B, TTL 24h", 1, 6),
        ("3. token cached in localStorage",                            1, 0),
        # Chat phase.
        ("4. POST /api/chat\n   Authorization: Bearer \u2026",         0, 2),
        ("5. ChatAgent.respond()\n   slot-fill \u2192 pipeline.run()", 2, 3),
        ("6. langgraph.invoke(thread_id)\n   format \u2192 detect_purchase \u2192 retrieve", 3, 4),
        ("7. criticality_classifier.score(\u2026)\n   purchase_value=$5,000>=$2,000  \u2192 risk 0.70", 4, 5),
        ("8. required_roles_for(drivers)\n   \u2192 ['buyer']",        5, 4),
        ("9. interrupt({thread_id, risk, \u2026})\n   SqliteSaver checkpoints state", 4, 6),
        ("10. response.awaiting_approval=true\n    pending_approval_thread_id=\u2026", 2, 0),
        ("11. ApprovalBanner mounts\n    composer disabled; deep-link visible", 0, 0),
        # Buyer phase.
        ("12. Buyer signs in (steps 1–3 repeat)\n    role='buyer'",     7, 1),
        ("13. GET /api/approvals/my\n    Bearer (buyer)",               7, 2),
        ("14. bucket pending: maker / pending_for_me\n    can_approve(role, required) \u2227 \u00ac is_maker_locked", 2, 5),
        ("15. {stats, pending_for_me:[\u2026], actioned:[\u2026]}",     2, 7),
        ("16. Approvals tab renders PendingForMeCard\n    inline Approve / Reject",        7, 7),
        ("17. POST /api/approvals/{thread}/resume\n    {approved:true, comments}",         7, 2),
        ("18. require_user + can_approve + \u00ac maker_locked\n    \u2192 reject 403 / 409 if violated", 2, 5),
        ("19. graph.invoke(Command(resume=decision))\n    critic \u2192 END",              2, 4),
        ("20. audit_log.record(\u2026)\n    maker, approver, role, drivers",                4, 6),
        ("21. 200 OK \u2192 dashboardRefreshKey++\n    operator's banner clears next poll", 2, 7),
    ]

    # Render the sequence. Distinct colours for the three sub-phases keep
    # the diagram readable even printed in black and white.
    sub_colors = [
        HexColor("#1E3A8A"),  # blue – login
        HexColor("#15803D"),  # green – chat / pause
        HexColor("#7C2D12"),  # amber/brown – approval
    ]
    def color_for(i):
        if i <= 2:
            return sub_colors[0]
        if i <= 10:
            return sub_colors[1]
        return sub_colors[2]

    y = lane_top - 48
    step_dy = 18
    for idx, (label, src, dst) in enumerate(steps):
        col = color_for(idx)
        x1 = lane_x(src)
        x2 = lane_x(dst)
        # Self-call (src == dst) → render as a small bracket on that lane.
        if src == dst:
            bracket_w = lane_width * 0.45
            cx = x1
            # Place the label on whichever side has more room so the right
            # edge doesn't clip text for the last lane.
            label_on_right = src < lane_count - 1
            c.setStrokeColor(col)
            c.setLineWidth(1.0)
            c.line(cx - bracket_w / 2, y, cx - bracket_w / 2, y - 8)
            c.line(cx - bracket_w / 2, y - 8, cx + bracket_w / 2, y - 8)
            c.line(cx + bracket_w / 2, y, cx + bracket_w / 2, y - 8)
            # Arrow head pointing back into the lane spine.
            c.setFillColor(col)
            head_x = cx + bracket_w / 2 if label_on_right else cx - bracket_w / 2
            head_dir = -1 if label_on_right else 1  # tip points toward spine
            p = c.beginPath()
            p.moveTo(head_x, y - 8)
            p.lineTo(head_x + head_dir * 4, y - 11)
            p.lineTo(head_x + head_dir * 4, y - 5)
            p.close()
            c.drawPath(p, fill=1, stroke=0)
            # Label
            c.setFillColor(TITLE_COLOR)
            c.setFont("Helvetica", 7)
            for j, line in enumerate(label.split("\n")):
                if label_on_right:
                    c.drawString(cx + bracket_w / 2 + 6, y - 4 - j * 8, line)
                else:
                    c.drawRightString(cx - bracket_w / 2 - 6, y - 4 - j * 8, line)
        else:
            draw_arrow(c, x1, y, x2, y, color=col)
            c.setFillColor(TITLE_COLOR)
            c.setFont("Helvetica", 7)
            mx = (x1 + x2) / 2
            for j, line in enumerate(label.split("\n")):
                c.drawCentredString(mx, y + 4 + (1 - j) * 8 + 1, line) \
                    if False else c.drawCentredString(mx, y - 8 - j * 8, line)
        y -= step_dy

    # ─── Footer legend ─────────────────────────────────────────────────────
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(36, 92, "Phases")
    legend = [
        ("blue",   "1–3  Auth / token mint  (24h TTL, stdlib PBKDF2-SHA256)"),
        ("green",  "4–11 Chat \u2192 risk \u2192 interrupt; state durable via SqliteSaver"),
        ("amber",  "12–21 Buyer signs in, approves, audit row written, banner clears"),
    ]
    name_to_color = {
        "blue":  HexColor("#1E3A8A"),
        "green": HexColor("#15803D"),
        "amber": HexColor("#7C2D12"),
    }
    c.setFont("Helvetica", 7.5)
    for i, (name, txt) in enumerate(legend):
        col = name_to_color[name]
        c.setFillColor(col)
        c.rect(36, 78 - i * 11, 12, 8, fill=1, stroke=0)
        c.setFillColor(TITLE_COLOR)
        c.drawString(54, 79 - i * 11, txt)

    draw_page_footer(
        c,
        "Self-arrows = in-process call within a single component. "
        "Cross-lane arrows always cross a process boundary "
        "(HTTP / SQLite / LangGraph checkpoint).",
    )


# ─── Page 6 — Component interaction contracts ──────────────────────────────
#
# This page lists *every* edge in the system as a row: the wire payload,
# auth required, failure modes, and a p50 latency budget. Treat it as the
# single source of truth — any code change that breaks a row here is an
# observable architectural change that needs review.

def draw_page6(c):
    page_w, page_h = landscape(letter)
    draw_page_header(
        c,
        "Component Interaction Contracts",
        "Every cross-component edge in the system. Use this page as the "
        "single source of truth when changing a payload, an auth check, "
        "or a failure mode.",
        page_num=6,
        total_pages=8,
    )

    # ─── Section 1: UI → API (HTTP) ────────────────────────────────────────
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, page_h - 88, "1.  UI \u2192 API  (HTTP boundary)")

    headers = ["From", "To", "Route / Payload", "Auth", "Fails when", "p50"]
    rows = [
        ("Next.js page.tsx", "api/auth",
         "POST /api/auth/login {user_id, password}",
         "—", "wrong password \u2192 401", "20–50 ms"),
        ("Next.js page.tsx", "api/server",
         "POST /api/chat {session_id, message}",
         "Bearer (opt)", "422 / 500 on pipeline error", "0.4–9 s"),
        ("ApprovalsTab", "api/server", "GET /api/approvals/my",
         "Bearer", "401 on missing token", "30–80 ms"),
        ("PendingForMeCard", "api/server",
         "POST /api/approvals/{thread}/resume",
         "Bearer (checker)", "403 wrong role · 409 maker-lock · 404 unknown",
         "0.6–3.5 s"),
        ("ApprovalBanner", "api/server", "GET /api/approvals/{thread_id}",
         "Bearer (opt)", "404 unknown thread", "20–50 ms"),
        ("Streamlit app.py", "api/server",
         "same endpoints (in-tab login)",
         "Bearer (shared)", "same as above", "same"),
    ]
    col_widths = [105, 90, 200, 95, 180, 50]
    y_after = draw_table(c, 36, page_h - 96, headers, rows, col_widths,
                         row_height=15, header_height=20)

    # ─── Section 2: API → Domain (in-process) ──────────────────────────────
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, y_after - 22, "2.  FastAPI \u2192 domain  (in-process Python)")

    headers2 = ["From", "To", "Call / contract", "Notes", "Side effects", "p50"]
    rows2 = [
        ("api/server", "ChatAgent.respond",
         "respond(session_id, msg, user) \u2192 ChatTurn",
         "Slot fill \u2192 pipeline.run",
         "stamps maker_user_id", "0.4–9 s"),
        ("ChatAgent", "ManufacturingPipeline.run",
         "run(query, mode='diagnostic') \u2192 Result",
         "Intent + USE_LANGGRAPH",
         "may pause at interrupt", "0.4–9 s"),
        ("ManufacturingPipeline", "LangGraphOrchestrator",
         "invoke(state, config={thread_id})",
         "USE_LANGGRAPH=true",
         "checkpoint per node", "0.4–9 s"),
        ("LangGraph node", "criticality_classifier",
         "score(query, answer, purchase_request) \u2192 Risk",
         "Rules \u2192 drivers \u2192 score",
         "—", "1–5 ms"),
        ("criticality_classifier", "rbac.required_roles_for",
         "required_roles_for(drivers, pr) \u2192 [role_id]",
         "OR-set; stable order",
         "—", "<1 ms"),
        ("api/server (resume)", "rbac.can_approve / is_maker_locked",
         "two boolean guards before resume",
         "403 / 409 if violated",
         "—", "<1 ms"),
        ("LangGraph (resume)", "audit_log.record",
         "record(thread, decision, maker, approver, role, \u2026)",
         "Append-only",
         "SQLite write", "1–4 ms"),
    ]
    col_widths2 = [115, 140, 195, 115, 115, 40]
    y_after2 = draw_table(c, 36, y_after - 30, headers2, rows2, col_widths2,
                          row_height=15, header_height=20)

    # ─── Section 3: Domain → Persistence ───────────────────────────────────
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, y_after2 - 22, "3.  Domain \u2192 persistence")

    headers3 = ["Component", "Backend", "Lifecycle",
                "Failure mode", "Drop-in replacement"]
    rows3 = [
        ("auth_store",      "auth.sqlite (stdlib sqlite3)",
         "open per call; 24h token TTL",
         "delete file \u2192 re-seed accounts",
         "OIDC · Argon2"),
        ("SqliteSaver",     "audit.sqlite (lg_checkpoint_sqlite)",
         "one checkpoint per node",
         "missing pkg \u2192 MemorySaver (WARN)",
         "Postgres saver"),
        ("audit_log",       "audit.sqlite (append-only)",
         "writes on every approve/reject",
         "disk full \u2192 500",
         "Kafka topic"),
        ("FAISS index",     "doc_pipeline/vector_store/*",
         "rebuilt only when stale",
         "missing \u2192 rebuild on boot",
         "Qdrant / Pinecone"),
        ("Knowledge graph", "data/processed/knowledge_graph.json",
         "rebuilt when input docs change",
         "missing \u2192 rebuild on boot",
         "Neo4j / TigerGraph"),
        ("Session memory",  "in-memory dict (FastAPI process)",
         "wiped on restart",
         "crash \u2192 session lost (threads survive)",
         "Redis"),
    ]
    col_widths3 = [105, 170, 135, 190, 120]
    y_after3 = draw_table(c, 36, y_after2 - 30, headers3, rows3, col_widths3,
                          row_height=15, header_height=20)

    # ─── Section 4: Role-gating policy (tightening of the API guard) ──────
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, y_after3 - 22, "4.  Authorisation policy on /api/approvals/{thread}/resume")

    policy_lines = [
        "  1.  Bearer token resolves to a User (else 401 unauthorised).",
        "  2.  user.role \u2208 required_roles  (else 403 forbidden \u2014 wrong role).",
        "  3.  user.user_id \u2260 maker_user_id (else 409 conflict \u2014 maker locked).",
        "  4.  thread_id is still pending (else 404 not found / 409 if already resolved).",
        "Outcome: graph resumes \u2192 audit row written \u2192 stats refresh \u2192 both UIs refetch on next poll.",
    ]
    c.setFillColor(SUBTITLE_COLOR)
    c.setFont("Helvetica", 8)
    for i, line in enumerate(policy_lines):
        c.drawString(36, y_after3 - 36 - i * 10, line)

    draw_page_footer(
        c,
        "Latencies measured against a warm pipeline on an M2 MacBook; p50 cited, "
        "p99 typically 2\u20133\u00d7 these numbers. Numbers exclude OpenAI tail latencies.",
    )


# ─── Page 7 — Role-Based Knowledge-Base ACLs ────────────────────────────────


def draw_page7(c):
    """Document-level RBAC on the RAG corpus.

    Visual narrative:
      Top band   — three-tier classification card row (Public, Restricted,
                   Confidential) with example documents.
      Middle row — role-to-classification read-set matrix (rows = roles,
                   columns = tiers, dots = entitlement).
      Lower row — left  : ingest \u2192 tag flow (folder convention).
                  right : query-time filter flow (ContextVar \u2192 retriever).
      Footer ribbon — operator-vs-plant-manager evidence delta on the
                   smoke-test query, with the security guarantee tagline.
    """
    page_w, page_h = landscape(letter)
    draw_page_header(
        c,
        "Role-Based Knowledge-Base Access (Document ACLs)",
        "Three-tier classification \u2022 ContextVar-scoped retriever filter \u2022 "
        "smoke-test proven \u2022 core/document_acl.py",
        page_num=7,
        total_pages=8,
    )

    # Local palette ---------------------------------------------------------
    PUBLIC_COL       = HexColor("#15803D")  # green
    PUBLIC_FILL      = HexColor("#DCFCE7")
    RESTRICTED_COL   = HexColor("#B45309")  # amber
    RESTRICTED_FILL  = HexColor("#FEF3C7")
    CONFIDENTIAL_COL = HexColor("#B91C1C")  # red
    CONFIDENTIAL_FILL = HexColor("#FEE2E2")
    INK              = HexColor("#0F172A")
    INK_SOFT         = HexColor("#475569")
    PANEL_BORDER     = HexColor("#CBD5E1")
    PANEL_FILL       = HexColor("#FFFFFF")
    DOT_ON           = HexColor("#0F172A")
    DOT_OFF          = HexColor("#CBD5E1")

    # ── Band 1: classification cards ──────────────────────────────────────
    band1_top = page_h - 78
    band1_h = 100
    band1_y = band1_top - band1_h

    card_w = (page_w - 36 * 2 - 24) / 3  # 3 cards with 12px gutter
    card_h = band1_h - 4
    gutter = 12

    cards = [
        {
            "title": "public",
            "label": "Public",
            "subtitle": "everyone (incl. operators)",
            "border": PUBLIC_COL,
            "fill": PUBLIC_FILL,
            "examples": [
                "SOPs (CNC machining, conveyor PM)",
                "Alarm-response procedures",
                "Equipment manuals (pumps, presses)",
                "Public safety bulletins",
                "Production / quality plan summaries",
            ],
        },
        {
            "title": "restricted",
            "label": "Restricted",
            "subtitle": "every checker role (not operator)",
            "border": RESTRICTED_COL,
            "fill": RESTRICTED_FILL,
            "examples": [
                "Regulatory incident response playbook",
                "Internal RCA (Q4 spindle-bearing case)",
                "Litigation-sensitive draft findings",
                "Detailed work orders + RCAs",
                "Lockout / permit-to-work records",
            ],
        },
        {
            "title": "confidential",
            "label": "Confidential",
            "subtitle": "plant_manager + procurement_manager only",
            "border": CONFIDENTIAL_COL,
            "fill": CONFIDENTIAL_FILL,
            "examples": [
                "Q1 2026 financial review (EBITDA, capex)",
                "Project Meridian M&A diligence paper",
                "Strategic supplier pricing (SKF, Siemens, Sandvik)",
                "Leadership succession + retention plan",
                "MFN clauses, walk-away prices",
            ],
        },
    ]

    for i, card in enumerate(cards):
        x = 36 + i * (card_w + gutter)
        # Card body
        c.setFillColor(card["fill"])
        c.setStrokeColor(card["border"])
        c.setLineWidth(1.2)
        c.roundRect(x, band1_y, card_w, card_h, 8, fill=1, stroke=1)

        # Title + tier id
        c.setFillColor(card["border"])
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x + 12, band1_y + card_h - 18, card["label"])
        c.setFont("Helvetica", 7.5)
        c.drawString(x + 12, band1_y + card_h - 30,
                     f"tier id: \u201c{card['title']}\u201d  \u2022  {card['subtitle']}")

        # Hairline separator
        c.setStrokeColor(card["border"])
        c.setLineWidth(0.4)
        c.line(x + 12, band1_y + card_h - 36, x + card_w - 12,
               band1_y + card_h - 36)

        # Example bullets
        c.setFillColor(INK)
        c.setFont("Helvetica", 7.3)
        line_y = band1_y + card_h - 46
        for ex in card["examples"]:
            c.drawString(x + 14, line_y, "\u2022 " + ex)
            line_y -= 9.5

    # ── Band 2: role × tier read-set matrix ───────────────────────────────
    band2_top = band1_y - 22
    section_label_y = band2_top
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, section_label_y, "Role \u2192 tier read-sets")
    c.setFillColor(INK_SOFT)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(
        36, section_label_y - 11,
        "Each row is a signup role. A filled dot means the retrievers will "
        "return chunks at that tier for the signed-in user.",
    )

    # Matrix geometry
    matrix_top = section_label_y - 22
    role_col_w = 170
    tier_col_w = 95
    row_h = 13
    header_h = 16

    roles = [
        ("operator",              "line / control-room operator"),
        ("shift_supervisor",      "shift supervisor"),
        ("maintenance_planner",   "maintenance planner"),
        ("maintenance_engineer",  "maintenance engineer"),
        ("ehs_officer",           "EHS officer"),
        ("quality_engineer",      "quality engineer"),
        ("buyer",                 "buyer (POs \u2264 $10k)"),
        ("procurement_manager",   "procurement manager"),
        ("plant_manager",         "plant manager"),
    ]
    # read-set membership (must match ROLE_TO_CLASSIFICATIONS in code)
    membership = {
        "operator":             ("on", "off", "off"),
        "shift_supervisor":     ("on", "on",  "off"),
        "maintenance_planner":  ("on", "on",  "off"),
        "maintenance_engineer": ("on", "on",  "off"),
        "ehs_officer":          ("on", "on",  "off"),
        "quality_engineer":     ("on", "on",  "off"),
        "buyer":                ("on", "on",  "off"),
        "procurement_manager":  ("on", "on",  "on"),
        "plant_manager":        ("on", "on",  "on"),
    }
    tier_headers = [
        ("public",       PUBLIC_COL),
        ("restricted",   RESTRICTED_COL),
        ("confidential", CONFIDENTIAL_COL),
    ]

    matrix_w = role_col_w + tier_col_w * 3
    matrix_h = header_h + row_h * len(roles)
    matrix_y = matrix_top - matrix_h

    # Outer panel
    c.setFillColor(PANEL_FILL)
    c.setStrokeColor(PANEL_BORDER)
    c.setLineWidth(0.6)
    c.roundRect(36, matrix_y, matrix_w, matrix_h, 4, fill=1, stroke=1)

    # Header row
    header_y = matrix_top - header_h
    c.setFillColor(TABLE_HEADER_BG)
    c.rect(36, header_y, matrix_w, header_h, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(36 + 8, header_y + 5, "role  (rbac.py)")
    for i, (label, col) in enumerate(tier_headers):
        cx = 36 + role_col_w + tier_col_w * i + tier_col_w / 2
        c.drawCentredString(cx, header_y + 5, label)

    # Body rows
    for i, (role_id, role_desc) in enumerate(roles):
        row_y = header_y - (i + 1) * row_h
        if i % 2 == 0:
            c.setFillColor(TABLE_ALT_ROW)
            c.rect(36, row_y, matrix_w, row_h, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(36 + 8, row_y + 5, role_id)
        c.setFillColor(INK_SOFT)
        c.setFont("Helvetica", 7)
        c.drawString(36 + 8 + 110, row_y + 5, role_desc)

        # Dots
        for j, status in enumerate(membership[role_id]):
            cx = 36 + role_col_w + tier_col_w * j + tier_col_w / 2
            cy = row_y + row_h / 2
            tier_col = tier_headers[j][1]
            if status == "on":
                c.setFillColor(tier_col)
                c.setStrokeColor(tier_col)
                c.circle(cx, cy, 4, fill=1, stroke=1)
            else:
                c.setFillColor(DOT_OFF)
                c.setStrokeColor(DOT_OFF)
                c.circle(cx, cy, 2.5, fill=1, stroke=1)

    # ── Band 3: ingest pipeline (left) + query-time filter (right) ────────
    band3_top = matrix_y - 18
    panel_h = 132
    panel_w = (page_w - 36 * 2 - 18) / 2
    band3_y = band3_top - panel_h

    def section_title(x, y, title, subtitle):
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(x, y, title)
        c.setFillColor(INK_SOFT)
        c.setFont("Helvetica-Oblique", 7.5)
        c.drawString(x, y - 10, subtitle)

    # Left panel: ingest-time tagging
    section_title(
        36, band3_top,
        "Ingest-time tagging  (one folder convention, zero front-matter)",
        "doc_pipeline/document_ingestion.py + chunking.py + "
        "core.document_acl.classify_from_path",
    )
    c.setFillColor(PANEL_FILL)
    c.setStrokeColor(PANEL_BORDER)
    c.setLineWidth(0.6)
    c.roundRect(36, band3_y, panel_w, panel_h - 22, 4, fill=1, stroke=1)

    # Folder tree (left half of left panel)
    tree_x = 36 + 12
    tree_y = band3_y + panel_h - 22 - 16
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(tree_x, tree_y, "doc_pipeline/input_docs/")
    tree_lines = [
        ("\u251c\u2500 sop_cnc_machining.txt",              "public",       PUBLIC_COL),
        ("\u251c\u2500 quality_control_manual.pdf",          "public",       PUBLIC_COL),
        ("\u251c\u2500 production_metrics_q1_2026.xlsx",     "public",       PUBLIC_COL),
        ("\u251c\u2500 restricted/",                         "",             None),
        ("\u2502   \u251c\u2500 regulatory_incident_response_plan.txt",
                                                              "restricted",   RESTRICTED_COL),
        ("\u2502   \u2514\u2500 internal_incident_rca_2025_q4.txt",
                                                              "restricted",   RESTRICTED_COL),
        ("\u2514\u2500 management/",                          "",             None),
        ("    \u251c\u2500 q1_2026_financial_review.txt",     "confidential", CONFIDENTIAL_COL),
        ("    \u251c\u2500 acquisition_target_assessment.txt","confidential", CONFIDENTIAL_COL),
        ("    \u251c\u2500 strategic_supplier_pricing.txt",   "confidential", CONFIDENTIAL_COL),
        ("    \u2514\u2500 leadership_succession_2026.txt",   "confidential", CONFIDENTIAL_COL),
    ]
    c.setFont("Helvetica", 7)
    line_y = tree_y - 10
    for path, tag, col in tree_lines:
        c.setFillColor(INK)
        c.drawString(tree_x + 6, line_y, path)
        if tag and col:
            tag_x = tree_x + 245
            tag_w = 54
            c.setFillColor(col)
            c.setStrokeColor(col)
            c.roundRect(tag_x, line_y - 2, tag_w, 9, 4.5, fill=1, stroke=1)
            c.setFillColor(white)
            c.setFont("Helvetica-Bold", 6.2)
            c.drawCentredString(tag_x + tag_w / 2, line_y + 0.5, tag.upper())
            c.setFont("Helvetica", 7)
        line_y -= 9

    # Right panel: query-time filter
    right_panel_x = 36 + panel_w + 18
    section_title(
        right_panel_x, band3_top,
        "Query-time filter  (zero parameter plumbing)",
        "ContextVar set by FastAPI per request \u2192 retrievers honour it deep in the stack",
    )
    c.setFillColor(PANEL_FILL)
    c.setStrokeColor(PANEL_BORDER)
    c.setLineWidth(0.6)
    c.roundRect(right_panel_x, band3_y, panel_w, panel_h - 22, 4, fill=1, stroke=1)

    # Flow diagram inside the right panel
    flow_steps = [
        ("Bearer token", "POST /api/chat", HexColor("#FEF3C7"), HexColor("#B45309")),
        ("with_user_classifications(role)", "core/document_acl.py", HexColor("#DCFCE7"), HexColor("#15803D")),
        ("HybridRetriever.retrieve()", "BM25 \u22c4 vector \u22c4 graph \u2192 RRF", HexColor("#CFFAFE"), HexColor("#0E7490")),
        ("filter_chunks(fused)", "drops chunks whose .classification \u2209 read-set", HexColor("#F3E8FF"), HexColor("#7E22CE")),
        ("LLM prompt", "evidence the user is entitled to see", HexColor("#F1F5F9"), HexColor("#334155")),
    ]
    step_w = panel_w - 24
    step_h = 18
    step_x = right_panel_x + 12
    step_y = band3_y + panel_h - 22 - 10 - step_h
    for i, (title, sub, fill, border) in enumerate(flow_steps):
        c.setFillColor(fill)
        c.setStrokeColor(border)
        c.setLineWidth(1.0)
        c.roundRect(step_x, step_y, step_w, step_h, 4, fill=1, stroke=1)
        c.setFillColor(border)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(step_x + 10, step_y + step_h - 9, title)
        c.setFillColor(INK_SOFT)
        c.setFont("Helvetica", 6.5)
        c.drawString(step_x + 10, step_y + 4, sub)
        if i < len(flow_steps) - 1:
            arrow_y = step_y - 1
            c.setStrokeColor(INK_SOFT)
            c.setLineWidth(0.8)
            c.line(step_x + step_w / 2, arrow_y, step_x + step_w / 2, arrow_y - 2)
            # tiny arrowhead
            p = c.beginPath()
            p.moveTo(step_x + step_w / 2 - 2.5, arrow_y - 2)
            p.lineTo(step_x + step_w / 2 + 2.5, arrow_y - 2)
            p.lineTo(step_x + step_w / 2,       arrow_y - 5)
            p.close()
            c.setFillColor(INK_SOFT)
            c.drawPath(p, fill=1, stroke=0)
        step_y -= step_h + 2

    # ── Band 4: proof-point ribbon (operator vs plant_manager evidence) ───
    # Pin the ribbon a fixed distance above the footer so it never collides
    # with the page-footer line when the upper bands shrink.
    ribbon_h = 50
    ribbon_y = 42  # leaves ~20px clear above the footer baseline
    ribbon_top = ribbon_y + ribbon_h

    c.setFillColor(HexColor("#0F172A"))
    c.setStrokeColor(HexColor("#0F172A"))
    c.roundRect(36, ribbon_y, page_w - 36 * 2, ribbon_h, 6, fill=1, stroke=1)

    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(48, ribbon_y + ribbon_h - 13,
                 "Smoke-test proof  \u2014  scripts/smoke_test_acl.py")

    c.setFont("Helvetica", 7.4)
    c.setFillColor(HexColor("#E2E8F0"))
    c.drawString(
        48, ribbon_y + ribbon_h - 25,
        "Query: \u201cQ1 2026 EBITDA and capex execution plan\u201d  \u2192  same FAISS index, two roles:",
    )

    c.setFillColor(HexColor("#FCA5A5"))
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(48, ribbon_y + 16,
                 "operator (public-only) \u2192 0 confidential chunks; top-3:")
    c.setFillColor(white)
    c.setFont("Helvetica", 7.3)
    c.drawString(252, ribbon_y + 16,
                 "quality_control_manual.pdf, production_planning_report.pdf \u00d72")

    c.setFillColor(HexColor("#86EFAC"))
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(48, ribbon_y + 5,
                 "plant_manager (all tiers)  \u2192 5 confidential chunks; top-3:")
    c.setFillColor(white)
    c.setFont("Helvetica", 7.3)
    c.drawString(252, ribbon_y + 5,
                 "q1_2026_financial_review.txt \u00d73 (Header, Revenue & Margin, Capex Execution)")

    draw_page_footer(
        c,
        "Confidentiality is a data-pipeline guarantee enforced before the LLM ever sees the chunk \u2014 "
        "not a prompt-engineering pinky-swear.",
    )


# ─── Page 8 — Advanced patterns (rerank · cache · parallel · guardrails ·
#              tools · offline eval) ────────────────────────────────────────


def draw_page8(c):
    page_w, page_h = landscape(letter)
    draw_page_header(
        c,
        "Advanced Patterns  \u2014  rerank \u00b7 cache \u00b7 parallel \u00b7 guardrails \u00b7 tools \u00b7 offline eval",
        "Six production-hardening layers wrapping the core Hybrid GraphRAG engine. "
        "Each is gated by a single env flag.",
        page_num=8,
        total_pages=8,
    )

    # ─── Top row: the six pattern cards ───────────────────────────────────
    row1_top = page_h - 84
    card_h = 92
    card_w = (page_w - 36 * 2 - 12 * 5) / 6  # 6 cards, 12px gutters

    cards = [
        (
            "1. Cross-encoder rerank",
            RET,
            [
                "USE_RERANKER=false",
                "BAAI/bge-reranker-base",
                "2nd stage after RRF",
                "blend = 0.7\u00b7CE + 0.3\u00b7RRF",
                "+5\u201315% quality on noisy corpora",
                "core/retrieval/reranker.py",
            ],
        ),
        (
            "2. Async parallel retrieval",
            ORCH,
            [
                "USE_PARALLEL_RETRIEVAL=true \u2605",
                "ThreadPoolExecutor(3)",
                "BM25 \u2225 FAISS \u2225 Graph",
                "per-leg timeout 15 s",
                "~30% latency cut",
                "hybrid_retriever._run_retrievers",
            ],
        ),
        (
            "3. Semantic cache",
            STORE,
            [
                "USE_SEMANTIC_CACHE=false",
                "cosine \u2265 0.97 \u2192 HIT",
                "LRU \u00b7 max 256 \u00b7 TTL 3600 s",
                "skips entire LLM stack",
                "refuses paused / rejected runs",
                "core/semantic_cache.py",
            ],
        ),
        (
            "4. Deterministic guardrails",
            NLU,
            [
                "USE_GUARDRAILS=true \u2605",
                "citations required",
                "regex: loto bypass \u00b7 live elec \u00b7 PPE",
                "BLOCK / REWRITE / PASS",
                "feeds back into critic retry",
                "core/guardrails.py",
            ],
        ),
        (
            "5. ERP/MES/SAP tool-calling",
            LLM,
            [
                "USE_TOOLS=false",
                "Planner: rules + cheap LLM",
                "Read: inventory, WO status",
                "Write: PO, WO  \u2192  HITL gate",
                "MockBackend ships in-tree",
                "core/tools/{registry,planner}.py",
            ],
        ),
        (
            "6. RAGAS-style offline eval",
            INGEST,
            [
                "python -m comparison.eval.run",
                "GoldenItem dataset (curated)",
                "faithfulness \u00b7 relevancy",
                "context_precision \u00b7 citation_acc",
                "guardrail_pass_rate",
                "comparison/eval/",
            ],
        ),
    ]

    x_cur = 36
    for title, palette, items in cards:
        draw_box(c, x_cur, row1_top - card_h, card_w, card_h, title, items, palette)
        x_cur += card_w + 12

    # Star legend for the safe-on flags.
    c.setFillColor(SUBTITLE_COLOR)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(
        36, row1_top - card_h - 14,
        "\u2605  safe-on by default on a fresh .env  (run.sh enables parallel retrieval + guardrails; "
        "rerank \u00b7 cache \u00b7 tools stay opt-in to avoid surprise model downloads, memory pressure or write surface)",
    )

    # ─── Middle: updated request flow ─────────────────────────────────────
    flow_top = row1_top - card_h - 36
    flow_h = 240

    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, flow_top, "Updated request flow with all patterns engaged")

    flow_box_top = flow_top - 12
    # Node coordinates — laid out as a horizontal pipeline with vertical
    # branches for the cache, guardrails, and tool fan-out.
    node_w, node_h = 110, 30
    row_main_y = flow_box_top - 28

    def node(x, y, label, palette=RET, w=node_w, h=node_h, items=None):
        items = items or [label]
        draw_box(c, x, y, w, h, "", items, palette)

    # Linear chain
    chain_y = row_main_y
    chain_x_start = 38
    gap = 12
    labels = [
        ("User query", CLIENT),
        ("Semantic cache", STORE),
        ("Clarifier + QueryCorrector", NLU),
        ("Hybrid retrieval (parallel)", RET),
        ("Cross-encoder rerank", RET),
        ("Tool planner", LLM),
        ("LLM answer", LLM),
        ("Guardrails", NLU),
        ("Critic loop", LLM),
    ]
    xs = []
    x_cur = chain_x_start
    for _ in labels:
        xs.append(x_cur)
        x_cur += node_w + gap
    for x, (lab, pal) in zip(xs, labels):
        node(x, chain_y, lab, palette=pal, items=[lab])
    # Arrows between consecutive boxes.
    for i in range(len(xs) - 1):
        draw_arrow(
            c,
            xs[i] + node_w, chain_y + node_h / 2,
            xs[i + 1], chain_y + node_h / 2,
        )

    # Cache HIT short-circuit arrow up to the right edge.
    cache_x = xs[1] + node_w / 2
    out_y = chain_y + node_h + 18
    out_x = xs[-1] + node_w
    draw_arrow(c, cache_x, chain_y + node_h, cache_x, out_y, label="HIT")
    draw_arrow(c, cache_x, out_y, out_x, out_y)
    c.setFillColor(SUBTITLE_COLOR)
    c.setFont("Helvetica-Oblique", 7)
    c.drawString(out_x + 4, out_y - 3, "cached answer")

    # Tool planner write-tool branch down to HITL.
    tp_x = xs[5] + node_w / 2
    hitl_y = chain_y - 36
    draw_arrow(c, tp_x, chain_y, tp_x, hitl_y + node_h, label="write tool")
    hitl_x = tp_x - node_w / 2
    node(hitl_x, hitl_y, "HITL approval", palette=API_PAL,
         items=["HITL approval (LangGraph)"])
    # Read-tool feed into LLM answer.
    read_x = xs[6] + node_w / 2
    draw_arrow(c, tp_x + node_w / 2, chain_y + node_h / 2,
               read_x - node_w / 2, chain_y + node_h / 2,
               label="read tool result")

    # Guardrails BLOCK branch down to HITL.
    gr_x = xs[7] + node_w / 2
    draw_arrow(c, gr_x, chain_y, gr_x, hitl_y + node_h, label="BLOCK")
    node(gr_x - node_w / 2, hitl_y, "HITL approval", palette=API_PAL,
         items=["HITL approval (block)"])

    # Critic FAIL_REWRITE loop back to LLM answer.
    crit_x = xs[8] + node_w / 2
    llm_x = xs[6] + node_w / 2
    loop_y = chain_y - 18
    draw_arrow(c, crit_x, chain_y, crit_x, loop_y, label="FAIL_REWRITE")
    draw_arrow(c, crit_x, loop_y, llm_x, loop_y)
    draw_arrow(c, llm_x, loop_y, llm_x, chain_y)

    # Critic PASS arrow back into cache for write-through.
    pass_y = chain_y + node_h + 32
    cache_top = chain_y + node_h
    draw_arrow(c, crit_x, chain_y + node_h, crit_x, pass_y, label="PASS")
    draw_arrow(c, crit_x, pass_y, cache_x, pass_y)
    draw_arrow(c, cache_x, pass_y, cache_x, cache_top)
    c.setFillColor(SUBTITLE_COLOR)
    c.setFont("Helvetica-Oblique", 7)
    c.drawString(cache_x + 6, pass_y + 2, "write-through")

    # ─── Bottom: feature-flag + offline-eval reference table ──────────────
    tbl_top = hitl_y - 28
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, tbl_top, "Feature flags  (config.py)")

    headers = ["Flag", "Default", "Effect"]
    rows = [
        ("USE_RERANKER",              "false", "Cross-encoder rerank after RRF (5\u201315% quality lift)"),
        ("RERANKER_MODEL",            "BAAI/bge-reranker-base", "Any HF cross-encoder; first run downloads it"),
        ("RERANK_CANDIDATE_POOL",     "20",    "Pool forwarded to reranker; final cut uses TOP_K_RERANK"),
        ("RERANK_BLEND_WEIGHT",       "0.7",   "Final = w\u00b7CE + (1\u2212w)\u00b7RRF"),
        ("USE_PARALLEL_RETRIEVAL",    "true \u2605", "Fan BM25 / FAISS / Graph across a thread pool"),
        ("PARALLEL_RETRIEVAL_TIMEOUT_S", "15.0", "Per-leg timeout (empty result on miss)"),
        ("USE_SEMANTIC_CACHE",        "false", "In-memory cosine cache; hit short-circuits LLM stack"),
        ("SEMANTIC_CACHE_THRESHOLD",  "0.97",  "Cosine threshold for a hit"),
        ("SEMANTIC_CACHE_MAX_SIZE",   "256",   "LRU eviction limit"),
        ("SEMANTIC_CACHE_TTL_SECONDS","3600",  "Per-entry TTL"),
        ("USE_GUARDRAILS",            "true \u2605", "Citation + safety regex post-processor"),
        ("GUARDRAILS_REQUIRE_CITATIONS","true","Reject answers without [source, chunk_id]"),
        ("GUARDRAILS_MIN_CITATIONS",  "1",     "Minimum citations required"),
        ("GUARDRAILS_BLOCK_UNSAFE",   "true",  "Hard-block LOTO bypass, live electrical, \u2026"),
        ("USE_TOOLS",                 "false", "Enable ERP/MES tool planner + registry"),
        ("TOOL_PLANNER_MODEL",        "qwen2.5:3b", "Cheap LLM used by the planner"),
        ("TOOL_PLANNER_USE_LLM",      "true",  "When false, only rule-based routing fires"),
    ]
    col_widths = [180, 70, page_w - 36 * 2 - 180 - 70]
    y_after = draw_table(
        c, 36, tbl_top - 6, headers, rows, col_widths,
        row_height=12, header_height=16, font_size=7, header_font_size=7.5,
    )

    # ─── Eval harness ribbon ──────────────────────────────────────────────
    ribbon_y = y_after - 28
    ribbon_h = 56
    c.setFillColor(OPTIONAL[0])
    c.setStrokeColor(OPTIONAL[1])
    c.setLineWidth(1.0)
    c.roundRect(36, ribbon_y, page_w - 36 * 2, ribbon_h, 6, fill=1, stroke=1)
    c.setFillColor(OPTIONAL[1])
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(46, ribbon_y + ribbon_h - 16,
                 "RAGAS-style offline eval  \u2014  comparison/eval/  \u2014  one command")
    c.setFillColor(black)
    c.setFont("Courier", 8)
    c.drawString(
        46, ribbon_y + ribbon_h - 30,
        "python -m comparison.eval.run --pipelines hybrid_graphrag classical_rag direct_llm "
        "--output comparison/eval/report.md --json-output comparison/eval/report.json --cache-dir .eval_cache",
    )
    c.setFont("Helvetica", 7.5)
    c.setFillColor(SUBTITLE_COLOR)
    c.drawString(
        46, ribbon_y + 8,
        "Metrics: faithfulness \u00b7 answer_relevancy \u00b7 context_precision \u00b7 citation_accuracy "
        "\u00b7 must_mention_coverage \u00b7 guardrail_pass_rate \u00b7 forbidden_violation_rate.   "
        "Exit code is non-zero when --min-faithfulness floor is breached \u2014 CI-friendly.",
    )

    draw_page_footer(
        c,
        "Modules: core/retrieval/reranker.py \u00b7 core/semantic_cache.py \u00b7 core/guardrails.py "
        "\u00b7 core/tools/ \u00b7 comparison/eval/",
    )


# ─── Driver ─────────────────────────────────────────────────────────────────


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=landscape(letter))
    c.setTitle("Hybrid GraphRAG Manufacturing — System Design")
    c.setAuthor("hybrid-graphrag-manufacturing")
    c.setSubject(
        "Architecture · LangGraph topology · Cost & latency · HITL · "
        "Low-level component sequence · Interaction contracts · Document ACLs · "
        "Advanced patterns (rerank · cache · parallel · guardrails · tools · eval)"
    )

    draw_page1(c)
    c.showPage()

    draw_page2(c)
    c.showPage()

    draw_page3(c)
    c.showPage()

    draw_page4(c)
    c.showPage()

    draw_page5(c)
    c.showPage()

    draw_page6(c)
    c.showPage()

    draw_page7(c)
    c.showPage()

    draw_page8(c)
    c.showPage()

    c.save()
    print(f"Wrote: {OUTPUT}")


if __name__ == "__main__":
    main()
