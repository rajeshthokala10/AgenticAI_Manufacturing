"""Generate the Hybrid GraphRAG Manufacturing system design PDF.

Run from the repo root:

    python system_design/generate_diagram.py

Output:

    system_design/system_architecture.pdf

The PDF is a 6-page design document:

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
        "Optional cause-ranker \u2022 Critic-validated tiered LLMs",
        page_num=1,
        total_pages=6,
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
        total_pages=6,
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
        total_pages=6,
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
        total_pages=6,
    )

    # ── Topology ──
    y = page_h - 90
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, y, "LangGraph topology with the HITL gate (Phases A + B + C)")

    y -= 18
    boxes = [
        ("format", 36, y - 30, 90, 28, "#1E2761"),
        ("detect_purchase", 132, y - 30, 110, 28, "#7A2048"),
        ("retrieve", 248, y - 30, 90, 28, "#283593"),
        ("rank_causes", 344, y - 30, 100, 28, "#5e35b2"),
        ("generate", 450, y - 30, 90, 28, "#1565c0"),
        ("criticality_check", 546, y - 30, 120, 28, "#e65100"),
        ("human_approval\n(interrupt)", 672, y - 60, 130, 50, "#c62828"),
        ("critic", 672, y - 30, 90, 28, "#2e7d32"),
        ("END", 768, y - 30, 50, 28, "#424242"),
    ]
    for label, x, by, w, h, color in boxes:
        c.setFillColor(color)
        c.roundRect(x, by, w, h, 4, fill=1, stroke=0)
        c.setFillColor("white")
        c.setFont("Helvetica-Bold", 8.5)
        for i, line in enumerate(label.split("\n")):
            c.drawCentredString(x + w / 2, by + h - 12 - i * 10, line)

    # arrows row
    arrow_y = y - 16
    for x_from, w_from, gap in [
        (36, 90, 132),    # format → detect_purchase
        (132, 110, 248),  # detect_purchase → retrieve
        (248, 90, 344),   # retrieve → rank_causes (conditional)
        (344, 100, 450),  # rank_causes → generate
        (450, 90, 546),   # generate → criticality_check
    ]:
        c.setStrokeColor("#444")
        c.setLineWidth(0.8)
        c.line(x_from + w_from + 1, arrow_y, gap - 1, arrow_y)
        c.line(gap - 5, arrow_y - 3, gap - 1, arrow_y)
        c.line(gap - 5, arrow_y + 3, gap - 1, arrow_y)

    # criticality_check branches
    c.setStrokeColor("#e65100")
    c.setLineWidth(1.0)
    # → human_approval (down-right)
    c.line(606, y - 30, 672, y - 60)
    c.setFillColor("#e65100")
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(610, y - 38, "needs_human")
    # → critic (right)
    c.line(666, y - 16, 672, y - 16)
    c.line(670, y - 13, 672, y - 16)
    c.line(670, y - 19, 672, y - 16)
    c.drawString(610, y - 12, "auto-approve")
    # human_approval → critic (when approved)
    c.line(737, y - 35, 737, y - 16)
    c.drawString(680, y - 28, "approved")
    # human_approval → END (when rejected)
    c.line(802, y - 35, 793, y - 16)
    c.drawString(770, y - 38, "rejected")
    # critic → END
    c.line(762, y - 16, 768, y - 16)
    c.line(766, y - 13, 768, y - 16)
    c.line(766, y - 19, 768, y - 16)

    # ── Risk score breakdown table ──
    table_y = y - 110
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
    table_y2 = draw_table(c, 36, table_y - 18, headers, rows, col_widths,
                           font_size=8, row_height=15)

    # ── Bottom: API surface + decision flow ──
    bottom_y = table_y2 - 30
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(36, bottom_y, "REST surface (api/server.py)")
    api_box = [
        ("GET  /api/approvals/pending",            "list paused HITL workflows"),
        ("GET  /api/approvals/{thread_id}",        "snapshot of one paused workflow"),
        ("POST /api/approvals/{thread_id}/resume", "{approved, approver, comments, edited_answer?}"),
        ("GET  /api/audit?limit=N&offset=M",       "recent decisions + approval-rate stats"),
    ]
    c.setFont("Helvetica", 8.5)
    for i, (route, desc) in enumerate(api_box):
        c.setFillColor("#283593")
        c.drawString(36, bottom_y - 16 - i * 13, route)
        c.setFillColor(SUBTITLE_COLOR)
        c.drawString(280, bottom_y - 16 - i * 13, desc)

    # State machine on the right
    sm_x = 540
    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(sm_x, bottom_y, "Pipeline status state machine")
    c.setFillColor(SUBTITLE_COLOR)
    c.setFont("Helvetica", 8.5)
    sm_lines = [
        "in_progress  →  awaiting_approval  →  complete    (approved)",
        "in_progress  →  awaiting_approval  →  rejected    (approved=false)",
        "in_progress  →  complete                          (auto-approve / USE_HITL=false)",
        "",
        "Checkpointer: SqliteSaver (HITL_DB_PATH) — survives restarts.",
        "Audit log:    core/audit_log.py — append-only, one row per decision.",
    ]
    for i, line in enumerate(sm_lines):
        c.drawString(sm_x, bottom_y - 16 - i * 12, line)

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
        total_pages=6,
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
        total_pages=6,
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


# ─── Driver ─────────────────────────────────────────────────────────────────


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=landscape(letter))
    c.setTitle("Hybrid GraphRAG Manufacturing — System Design")
    c.setAuthor("hybrid-graphrag-manufacturing")
    c.setSubject(
        "Architecture · LangGraph topology · Cost & latency · HITL · "
        "Low-level component sequence · Interaction contracts"
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

    c.save()
    print(f"Wrote: {OUTPUT}")


if __name__ == "__main__":
    main()
