"""Generate the Hybrid GraphRAG Manufacturing system architecture PDF.

Run from the repo root:

    python system_design/generate_diagram.py

Output:

    system_design/system_architecture.pdf
"""

from __future__ import annotations

import math
from pathlib import Path

from reportlab.lib.colors import HexColor, black
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas


OUTPUT = Path(__file__).resolve().parent / "system_architecture.pdf"


# Palette — pairs of (fill, border) for each architectural lane.
TITLE_COLOR = HexColor("#0F172A")
SUBTITLE_COLOR = HexColor("#334155")
PAGE_BG = HexColor("#FAFAFA")
ARROW_COLOR = HexColor("#1E293B")

CLIENT = (HexColor("#E0E7FF"), HexColor("#4338CA"))
API = (HexColor("#FEF3C7"), HexColor("#B45309"))
ORCH = (HexColor("#DCFCE7"), HexColor("#15803D"))
NLU = (HexColor("#FDE2E2"), HexColor("#B91C1C"))
RET = (HexColor("#CFFAFE"), HexColor("#0E7490"))
LLM = (HexColor("#F3E8FF"), HexColor("#7E22CE"))
STORE = (HexColor("#F1F5F9"), HexColor("#475569"))
INGEST = (HexColor("#FFE4E6"), HexColor("#9F1239"))


def draw_box(c, x, y, w, h, title, items, palette):
    fill, border = palette
    c.setFillColor(fill)
    c.setStrokeColor(border)
    c.setLineWidth(1.2)
    c.roundRect(x, y, w, h, 6, fill=1, stroke=1)

    c.setFillColor(border)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x + 8, y + h - 14, title)

    c.setFillColor(black)
    c.setFont("Helvetica", 8)
    line_y = y + h - 28
    for item in items:
        c.drawString(x + 10, line_y, "\u2022 " + item)
        line_y -= 11


def draw_arrow(c, x1, y1, x2, y2, dashed=False, label=None):
    c.setStrokeColor(ARROW_COLOR)
    c.setFillColor(ARROW_COLOR)
    c.setLineWidth(1.0)
    if dashed:
        c.setDash([3, 2], 0)
    else:
        c.setDash([], 0)
    c.line(x1, y1, x2, y2)
    c.setDash([], 0)

    angle = math.atan2(y2 - y1, x2 - x1)
    head = 7
    ax = x2 - head * math.cos(angle - math.pi / 8)
    ay = y2 - head * math.sin(angle - math.pi / 8)
    bx = x2 - head * math.cos(angle + math.pi / 8)
    by = y2 - head * math.sin(angle + math.pi / 8)
    path = c.beginPath()
    path.moveTo(x2, y2)
    path.lineTo(ax, ay)
    path.lineTo(bx, by)
    path.close()
    c.drawPath(path, fill=1, stroke=0)

    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 + 5
        c.setFillColor(SUBTITLE_COLOR)
        c.setFont("Helvetica-Oblique", 7)
        c.drawCentredString(mx, my, label)


def draw_legend(c, x, y):
    c.setFillColor(SUBTITLE_COLOR)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(x, y, "Legend")

    items = [
        ("Solid arrow", "runtime data flow"),
        ("Dashed arrow", "reads from / writes to persisted state"),
    ]
    c.setFont("Helvetica", 7.5)
    yy = y - 12
    for label, desc in items:
        c.setFillColor(black)
        c.drawString(x, yy, f"{label} — {desc}")
        yy -= 10


def draw_diagram(c):
    page_w, page_h = landscape(letter)

    c.setFillColor(PAGE_BG)
    c.rect(0, 0, page_w, page_h, fill=1, stroke=0)

    c.setFillColor(TITLE_COLOR)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(36, page_h - 38, "Hybrid GraphRAG Manufacturing — System Architecture")
    c.setFillColor(SUBTITLE_COLOR)
    c.setFont("Helvetica", 10)
    c.drawString(
        36,
        page_h - 54,
        "Multi-turn chat \u2022 LangGraph-optional orchestration \u2022 "
        "Hybrid retrieval (BM25 + FAISS + KG) \u2022 "
        "Optional cause-ranker \u2022 "
        "Critic-validated tiered LLMs (OpenAI + Ollama)",
    )

    y_clients = page_h - 130
    draw_box(
        c,
        50,
        y_clients,
        220,
        62,
        "Next.js Web UI  (web/)",
        ["Claude-style chat", "react-markdown · Tailwind", "Calls /api/chat via rewrites"],
        CLIENT,
    )
    draw_box(
        c,
        300,
        y_clients,
        220,
        62,
        "Streamlit UI  (app.py)",
        ["\U0001F4AC Chat tab", "Analytics dashboard (6 tabs)", "Direct in-process pipeline"],
        CLIENT,
    )
    draw_box(
        c,
        550,
        y_clients,
        220,
        62,
        "CLI / Python API",
        ["main.py · pipeline.*", "ChatAgent · ManufacturingPipeline"],
        CLIENT,
    )

    y_api = y_clients - 92
    draw_box(
        c,
        220,
        y_api,
        380,
        62,
        "FastAPI Backend  (api/server.py)",
        [
            "POST /api/chat   POST /api/reset",
            "GET  /api/health · /api/stats · /api/sessions/{id}",
            "In-memory session store · CORS-enabled",
        ],
        API,
    )

    draw_arrow(c, 160, y_clients, 300, y_api + 62, label="HTTP")
    draw_arrow(c, 410, y_clients, 410, y_api + 62)
    draw_arrow(c, 660, y_clients, 520, y_api + 62, label="in-process")

    y_orch = y_api - 108
    draw_box(
        c,
        40,
        y_orch,
        230,
        82,
        "ChatAgent  (pipeline/chat_agent.py)",
        [
            "Multi-turn conversation state",
            "Slot-filling (required → optional)",
            "Resets · skip tokens · per-session",
        ],
        ORCH,
    )
    draw_box(
        c,
        300,
        y_orch,
        220,
        82,
        "ManufacturingPipeline  (pipeline/)",
        [
            "Mode dispatch: Diagnostic | Quick",
            "Normalises retrieval ↔ LLM I/O",
            "Builds context for orchestrator",
        ],
        ORCH,
    )
    draw_box(
        c,
        550,
        y_orch,
        220,
        82,
        "HybridOrchestrator   (procedural · default)",
        [
            "core/orchestrator.py",
            "Runs critic loop (MAX_CRITIC_RETRIES)",
            "Opt-in: LangGraph StateGraph (USE_LANGGRAPH=true)",
        ],
        ORCH,
    )

    draw_arrow(c, 410, y_api, 155, y_orch + 82, label="user_msg + session")
    draw_arrow(c, 270, y_orch + 40, 300, y_orch + 40)
    draw_arrow(c, 520, y_orch + 40, 550, y_orch + 40)

    y_mid = y_orch - 108
    draw_box(
        c,
        40,
        y_mid,
        230,
        82,
        "Query Understanding",
        [
            "QueryCorrector (spell + acronyms)",
            "ClarifierAgent (intent · entities · slots)",
            "Intent classifier — qwen2.5:3b (Ollama)",
        ],
        NLU,
    )
    draw_box(
        c,
        300,
        y_mid,
        220,
        82,
        "Hybrid Retrieval",
        [
            "BM25 (rank-bm25 + pure-Python fallback)",
            "FAISS dense (all-MiniLM-L6-v2)",
            "KG traversal (NetworkX) · RRF fusion",
        ],
        RET,
    )
    draw_box(
        c,
        550,
        y_mid,
        220,
        92,
        "Tiered LLMs  (core/llm_client.py)",
        [
            "gpt-4o          → answer / retry    (OpenAI)",
            "qwen2.5:3b   → critic + classifier  (Ollama)",
            "qwen2.5:3b   → cause-ranker  (opt-in, intent-gated)",
            "gpt-4o-mini → comparison baselines",
        ],
        LLM,
    )

    draw_arrow(c, 130, y_orch, 130, y_mid + 82, label="raw query")
    draw_arrow(c, 200, y_mid + 82, 200, y_orch, label="normalised")

    draw_arrow(c, 400, y_orch, 400, y_mid + 82, label="query")
    draw_arrow(c, 470, y_mid + 82, 470, y_orch, label="top-k + KG paths")

    draw_arrow(c, 640, y_orch, 640, y_mid + 82, label="prompt+context")
    draw_arrow(c, 710, y_mid + 82, 710, y_orch, label="answer + critique")

    y_store = y_mid - 96
    draw_box(
        c,
        40,
        y_store,
        230,
        66,
        "Persistence",
        [
            "data/processed/  (chunks · KG JSON)",
            "doc_pipeline/vector_store/faiss.index",
            "Session state (in-memory dict)",
        ],
        STORE,
    )
    draw_box(
        c,
        300,
        y_store,
        220,
        66,
        "Knowledge Graph",
        [
            "NetworkX DiGraph",
            "Entities: Equipment·Component·Alarm·…",
            "Relations: TRIGGERS_ALARM · RESOLVED_BY · …",
        ],
        STORE,
    )
    draw_box(
        c,
        550,
        y_store,
        220,
        66,
        "Document Ingestion  (doc_pipeline/)",
        [
            "pdfplumber · openpyxl · pandas",
            "Semantic + sliding chunking",
            "KG builder · MiniLM-L6-v2 embeddings",
        ],
        INGEST,
    )

    draw_arrow(c, 155, y_mid, 155, y_store + 66, dashed=True)
    draw_arrow(c, 400, y_mid, 400, y_store + 66, dashed=True, label="reads")
    draw_arrow(c, 660, y_store + 66, 660, y_mid, label="builds index + KG")

    draw_legend(c, 36, 60)
    c.setFillColor(SUBTITLE_COLOR)
    c.setFont("Helvetica-Oblique", 8)
    c.drawRightString(
        page_w - 36,
        24,
        "Generated by system_design/generate_diagram.py · reportlab",
    )


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=landscape(letter))
    c.setTitle("Hybrid GraphRAG Manufacturing — System Architecture")
    c.setAuthor("hybrid-graphrag-manufacturing")
    c.setSubject("System architecture overview")
    draw_diagram(c)
    c.showPage()
    c.save()
    print(f"Wrote: {OUTPUT}")


if __name__ == "__main__":
    main()
