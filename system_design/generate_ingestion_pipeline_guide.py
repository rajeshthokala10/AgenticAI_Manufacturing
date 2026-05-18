"""Generate the ingestion-pipeline reference PDF.

A precise walkthrough of what happens between "schema YAML is saved" and
"answers reference the knowledge graph". Covers ingestion → chunking →
embedding → adapter → KG construction → persistence → retrieval.

Output is letter-portrait, body-text oriented so the tables and code
blocks fit comfortably for reading on screen / printing.

Run from the repo root:

    python system_design/generate_ingestion_pipeline_guide.py

Output:

    system_design/ingestion_pipeline_guide.pdf
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT = Path(__file__).resolve().parent / "ingestion_pipeline_guide.pdf"


# ─── Palette (matches the rest of system_design/) ──────────────────────────

INK = HexColor("#0F172A")
INK_MID = HexColor("#334155")
INK_SOFT = HexColor("#64748B")
ACCENT = HexColor("#B45309")     # copper
RULE = HexColor("#E2E8F0")
CODE_BG = HexColor("#F1F5F9")
HEADER_BG = HexColor("#0F172A")
ALT_ROW = HexColor("#F8FAFC")


def _styles() -> dict:
    base = getSampleStyleSheet()
    out = {}
    out["Title"] = ParagraphStyle(
        "Title", parent=base["Title"],
        fontName="Helvetica-Bold", fontSize=22, leading=26,
        textColor=INK, spaceAfter=4,
    )
    out["Subtitle"] = ParagraphStyle(
        "Subtitle", parent=base["Normal"],
        fontName="Helvetica", fontSize=11, leading=14,
        textColor=INK_SOFT, spaceAfter=14,
    )
    out["H1"] = ParagraphStyle(
        "H1", parent=base["Heading1"],
        fontName="Helvetica-Bold", fontSize=14, leading=18,
        textColor=ACCENT, spaceBefore=14, spaceAfter=6,
    )
    out["H2"] = ParagraphStyle(
        "H2", parent=base["Heading2"],
        fontName="Helvetica-Bold", fontSize=11, leading=14,
        textColor=INK, spaceBefore=8, spaceAfter=4,
    )
    out["Body"] = ParagraphStyle(
        "Body", parent=base["BodyText"],
        fontName="Helvetica", fontSize=9.5, leading=13,
        textColor=INK, alignment=TA_LEFT, spaceAfter=4,
    )
    out["Code"] = ParagraphStyle(
        "Code", parent=base["Code"],
        fontName="Courier", fontSize=8, leading=10.5,
        textColor=INK, backColor=CODE_BG,
        leftIndent=6, rightIndent=6,
        spaceBefore=4, spaceAfter=8,
        borderColor=RULE, borderWidth=0.5, borderPadding=6,
    )
    return out


def _table(data, col_widths, header_row=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header_row else 0)
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.3, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_ROW]),
    ]
    if header_row:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
        ]
    t.setStyle(TableStyle(style))
    return t


def P(text, style):
    return Paragraph(text, style)


# ─── Content ───────────────────────────────────────────────────────────────


def build_story(s: dict):
    story = []
    body = s["Body"]

    # ── Cover ─────────────────────────────────────────────────────────────
    story.append(P("From schema to knowledge graph", s["Title"]))
    story.append(P(
        "Manufacturing Hybrid GraphRAG — the exact path documents take "
        "between &lsquo;schema YAML saved&rsquo; and &lsquo;answers cite "
        "graph nodes&rsquo;. Six stages, four extractors, one schema as "
        "the source of truth.", s["Subtitle"]))

    # ── Trigger ───────────────────────────────────────────────────────────
    story.append(P("0. The trigger", s["H1"]))
    story.append(P(
        "Either the Streamlit <b>Save &amp; build KG</b> button or the CLI "
        "<font face='Courier'>python main.py --rebuild --domain &lt;id&gt; "
        "--no-llm</font>. Both end up in "
        "<font face='Courier'>pipeline/unified_pipeline.py:149</font> "
        "calling <font face='Courier'>ManufacturingPipeline(domain=&lt;id&gt;)"
        ".build_or_load(rebuild=True)</font>.", body))

    # ── Stage 1 ─────────────────────────────────────────────────────────
    story.append(P("Stage 1 — Ingestion", s["H1"]))
    story.append(P(
        "<font face='Courier'>doc_pipeline/document_ingestion.py</font>",
        s["H2"]))
    story.append(Preformatted("""\
doc_pipeline/input_docs/<domain>/**             <- what you staged

DocumentIngestion.ingest_directory()    (recursive rglob)
   per file -> ingest_file() -> parser by extension:
       .pdf   -> PDFParser   (pdfplumber: per-page text + tables)
       .txt   -> TXTParser   (whole file = one Document)
       .xlsx  -> ExcelParser (one Document per sheet)

list[Document]  with .content + .metadata
   (page, sheet_name, source, classification, ...)""", s["Code"]))
    story.append(P(
        "Folder name &rarr; <font face='Courier'>metadata.classification</font> "
        "via <font face='Courier'>core/document_acl.py:60</font>:", body))
    story.append(_table([
        ["Subfolder segment", "Classification"],
        ["management/ or confidential/", "confidential"],
        ["restricted/ or internal/", "restricted"],
        ["anything else", "public"],
    ], col_widths=[3.0 * inch, 2.0 * inch]))

    # ── Stage 2 ───────────────────────────────────────────────────────
    story.append(P("Stage 2 — Chunking", s["H1"]))
    story.append(P(
        "<font face='Courier'>doc_pipeline/chunking.py</font>", s["H2"]))
    story.append(Preformatted("""\
HybridChunker.chunk_documents(documents)
   three strategies, picked per Document:
      recursive       (long prose, paragraph-aware split)
      semantic        (embedding-similarity boundary detection)
      sliding_window  (fallback for short texts)

list[Chunk]   with .text, .metadata, .chunk_id (int), .strategy""", s["Code"]))
    story.append(P(
        "Chunks inherit the source Document&apos;s metadata (page, sheet, "
        "classification, etc.).", body))

    # ── Stage 3 ───────────────────────────────────────────────────────
    story.append(P("Stage 3 — Embedding &rarr; Qdrant", s["H1"]))
    story.append(P(
        "<font face='Courier'>doc_pipeline/embeddings.py</font>", s["H2"]))
    story.append(Preformatted("""\
EmbeddingPipeline(domain=<id>)
   bge-small-en-v1.5 encodes each chunk (384-dim, shared model cache)
   Qdrant upsert into collection `<domain>_corpus`
      (single Qdrant store, per-domain collection name)

save() writes:
   doc_pipeline/vector_store/<domain>_index_chunks.json
   doc_pipeline/vector_store/<domain>_index.manifest.json""", s["Code"]))

    # ── Stage 4 ───────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(P("Stage 4 — Adapter: chunks &rarr; core docs", s["H1"]))
    story.append(P(
        "<font face='Courier'>pipeline/adapter.py</font> &mdash; "
        "this is where the schema&apos;s Equipment "
        "<font face='Courier'>id_pattern</font> drives extraction.",
        s["H2"]))
    story.append(Preformatted("""\
chunks_to_core_docs(chunks, domain=<id>):
   for each chunk:
      derive stable chunk_id (md5 of source + index)
      regex-extract entity ids from chunk.text into metadata:
         equipment_ids  <- schema's Equipment id_pattern (anchored,
                              word-bound, stripped of ^$ for findall)
         alarm_codes    <- ALM-[A-Z]\\d{3}
         part_numbers   <- SP-/TH-/BRK-/SFT-/HSG-/GR-\\d{4}
         fault_codes    <- FC-\\d{3}

list[{"chunk_id": "...", "text": "...", "metadata": {...}}]""", s["Code"]))
    story.append(P(
        "<b>Why this matters:</b> when you author a clean "
        "<font face='Courier'>id_pattern</font> in your schema, "
        "asset-tag extraction is automatic. New domains&apos; tags get "
        "lifted into <font face='Courier'>metadata.equipment_ids</font> "
        "without any Python edits.", body))

    # ── Stage 5 ───────────────────────────────────────────────────────
    story.append(P("Stage 5 — KG construction", s["H1"]))
    story.append(P(
        "<font face='Courier'>core/knowledge_graph.py</font> &mdash; "
        "<font face='Courier'>KnowledgeGraph(domain=&lt;id&gt;)"
        ".build_from_documents(docs)</font> runs four extractors per "
        "chunk in priority order. Each extractor reads YOUR schema and "
        "emits Mentions + EdgeCandidates accordingly.", body))

    story.append(_table([
        ["#", "Extractor", "Author tag", "Conf.", "Reads", "What it emits"],
        ["1", "CodeExtractor", "system:code", "1.00",
         "chunk text via every entity's id_pattern regex",
         "Mentions for Equipment / Alarm / FailureMode / SparePart ids"],
        ["2", "MetadataExtractor", "system:metadata", "0.95",
         "metadata.{equipment_ids, alarm_codes, part_numbers, fault_codes}",
         "Mentions + co-occurrence edges declared in the schema "
         "(Equipment->Alarm, Equipment->SparePart, Alarm->FailureMode)"],
        ["3", "KeywordExtractor", "system:keyword", "0.95",
         "every entity_type with a 'vocabulary' block (Component, Cause, etc.)",
         "Mentions per word-boundary vocab match + edges from "
         "metadata.equipment_ids to the matched phrase via schema edge_types"],
        ["4", "NarrativeExtractor", "system:llm_extract", "0.50",
         "regex over prose for open-vocab types",
         "Low-confidence Symptom / Procedure mentions; HITL gap detector "
         "surfaces them for review"],
    ], col_widths=[0.2*inch, 1.0*inch, 1.0*inch, 0.5*inch, 1.7*inch, 2.1*inch]))

    story.append(Spacer(1, 6))
    story.append(P(
        "Every Mention and EdgeCandidate goes through schema validation:",
        body))
    story.append(P(
        "&bull; <font face='Courier'>Schema.validate_entity(identifier, "
        "type_name)</font> &mdash; checks the <font face='Courier'>"
        "id_pattern</font> or <font face='Courier'>vocabulary</font> "
        "allow-list", body))
    story.append(P(
        "&bull; <font face='Courier'>Schema.validate_edge(relation, "
        "source_type, target_type)</font> &mdash; checks the edge is "
        "declared and endpoints match its source/target types", body))
    story.append(P(
        "Rejects go into <font face='Courier'>kg._rejected</font> "
        "(inspectable for diagnosing schema drift). Accepted entries land "
        "in a NetworkX <font face='Courier'>DiGraph</font> with full "
        "provenance:", body))

    story.append(Preformatted("""\
graph.add_node(identifier,
    entity_type="Component",
    chunk_ids={"chunk_42", "chunk_103"},
    provenance=Provenance(
        author="system:keyword",
        confidence=0.95,
        source_chunk_id="chunk_42",
        timestamp=1747408234.5,
        supersedes=None,         # set when HITL writes back
        notes="",
    ),
)

graph.add_edge("CV-301", "belt",
    relation="HAS_COMPONENT",
    weight=2,
    chunk_ids={...},
    provenance=Provenance(...),
)""", s["Code"]))

    # ── Stage 6 ───────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(P("Stage 6 — Persistence", s["H1"]))
    story.append(P(
        "After every node + edge is validated and stamped:", body))
    story.append(Preformatted("""\
data/processed/knowledge_graph.<domain>.json     <- KG snapshot

{
  "nodes": {
    "CV-301": {"entity_type": "Equipment",
               "chunk_ids": [...],
               "provenance": {"author": "system:metadata",
                              "confidence": 0.95, ...}},
    "belt":   {"entity_type": "Component",
               "chunk_ids": [...],
               "provenance": {...}}
  },
  "edges": [
    {"source": "CV-301", "target": "belt",
     "relation": "HAS_COMPONENT",
     "weight": 2, "chunk_ids": [...],
     "provenance": {...}}
  ]
}""", s["Code"]))

    # ── Stage 7 ───────────────────────────────────────────────────────
    story.append(P("What gets queried at runtime", s["H1"]))
    story.append(P(
        "When a user types a question, the KG isn&apos;t queried for "
        "facts directly &mdash; it acts as a <b>retrieval bias</b>. "
        "Chunks linked to query-relevant entities get boosted; "
        "unrelated chunks get demoted. That&apos;s how the answer&apos;s "
        "symptom &rarr; cause &rarr; procedure chain stays grounded in "
        "evidence rather than LLM hallucination.", body))

    story.append(Preformatted("""\
query
   ClarifierAgent -> extracts entity-typed slots
                     (equipment_id, metric, etc.)

   KnowledgeGraph.get_subgraph_for_query()
      - matches query entities to KG nodes
      - walks declared edges per the schema's `traversal_routes`
      - returns a subgraph + a chunk allow-list

   HybridRetriever (BM25 + Qdrant vector + RRF + cross-encoder reranker)
      - scores chunks
      - prefers chunks in the KG allow-list

   LLM (task_model("answer") per the router) generates the answer

   Critic + Guardrails verify citations against the allow-list""",
                              s["Code"]))

    # ── One-glance ────────────────────────────────────────────────────
    story.append(P("One-glance diagram", s["H1"]))
    story.append(Preformatted("""\
schemas/<domain>.yaml --------------------------------+
                                                     |
doc_pipeline/input_docs/<domain>/**                  |
   |                                                 |
   v DocumentIngestion (PDF / TXT / XLSX parsers)    |
list[Document]                                       |
   |                                                 |
   v HybridChunker                                   |
list[Chunk]                                          |
   |                                                 |
   +--> EmbeddingPipeline -> Qdrant                  |
   |       <domain>_corpus collection                |
   |                                                 |
   v adapter.chunks_to_core_docs                     |  schema drives:
[{chunk_id, text, metadata}]                         |    EQUIPMENT_RE,
   |                                                 |    vocabularies,
   v KnowledgeGraph.build_from_documents             |    edge declarations
       4 extractors per chunk         <--------------+
       schema validates every Mention / Edge <-------+
       provenance stamped on every node / edge
       rejects go to kg._rejected
   |
   v
data/processed/knowledge_graph.<domain>.json""", s["Code"]))

    story.append(Spacer(1, 10))
    story.append(P(
        "Two artifacts result: the Qdrant collection for similarity "
        "search, and the KG json for typed retrieval routing. Both are "
        "consumed at query time by <font face='Courier'>"
        "pipeline/unified_pipeline.diagnostic()</font>.", body))

    return story


# ─── Page chrome ───────────────────────────────────────────────────────────


def _draw_chrome(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(INK_SOFT)
    canvas.drawString(0.6 * inch, LETTER[1] - 0.45 * inch,
                      "Manufacturing Hybrid GraphRAG — ingestion pipeline")
    canvas.drawRightString(LETTER[0] - 0.6 * inch, LETTER[1] - 0.45 * inch,
                           "system_design/ingestion_pipeline_guide.pdf")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.3)
    canvas.line(0.6 * inch, LETTER[1] - 0.55 * inch,
                LETTER[0] - 0.6 * inch, LETTER[1] - 0.55 * inch)
    canvas.drawString(0.6 * inch, 0.45 * inch, "Anthropic / Claude Code generated")
    canvas.drawRightString(LETTER[0] - 0.6 * inch, 0.45 * inch,
                           f"Page {doc.page}")
    canvas.restoreState()


def main() -> Path:
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=LETTER,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.75 * inch, bottomMargin=0.65 * inch,
        title="From schema to knowledge graph — Manufacturing Hybrid GraphRAG",
        author="Manufacturing Hybrid GraphRAG",
    )
    s = _styles()
    doc.build(build_story(s), onFirstPage=_draw_chrome, onLaterPages=_draw_chrome)
    return OUTPUT


if __name__ == "__main__":
    path = main()
    print(f"Wrote {path} ({path.stat().st_size // 1024} KB)")
