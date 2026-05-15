"""Generate the domain-onboarding reference PDF.

A precise reference for adding a new domain's data to the Manufacturing
Hybrid GraphRAG pipeline: drop locations, supported file types,
ACL-by-folder, schema declaration, the four-extractor stack, the end-to-
end pipeline flow, commands, output artefacts, and a worked example.

Output is letter-portrait, body-text oriented (not a slide deck) so the
tables and code blocks fit comfortably for reading on screen / printing.

Run from the repo root:

    python system_design/generate_domain_onboarding_guide.py

Output:

    system_design/domain_onboarding_guide.pdf
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
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT = Path(__file__).resolve().parent / "domain_onboarding_guide.pdf"


# ─── Palette (matches the rest of system_design/) ──────────────────────────

INK = HexColor("#0F172A")
INK_MID = HexColor("#334155")
INK_SOFT = HexColor("#64748B")
ACCENT = HexColor("#B45309")     # copper
RULE = HexColor("#E2E8F0")
CODE_BG = HexColor("#F1F5F9")
HEADER_BG = HexColor("#0F172A")
ALT_ROW = HexColor("#F8FAFC")


# ─── Styles ────────────────────────────────────────────────────────────────


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
    out["BodySmall"] = ParagraphStyle(
        "BodySmall", parent=out["Body"],
        fontSize=8.5, leading=11.5, textColor=INK_MID,
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


# ─── Table helpers ─────────────────────────────────────────────────────────


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
    small = s["BodySmall"]

    # ── Cover ─────────────────────────────────────────────────────────────
    story.append(P("Adding a new domain", s["Title"]))
    story.append(P("Manufacturing Hybrid GraphRAG — strict onboarding reference. "
                   "Drop locations, supported file types, ACL-by-folder, schema "
                   "declaration, the four-extractor stack, end-to-end flow, "
                   "commands, output artefacts, and a worked example.", s["Subtitle"]))

    # ── 1. Inputs ─────────────────────────────────────────────────────────
    story.append(P("1. Inputs", s["H1"]))
    story.append(P(
        "<b>Drop location:</b> <font face='Courier'>doc_pipeline/input_docs/</font> — "
        "loaded recursively via <font face='Courier'>Path.rglob(&quot;*&quot;)</font> "
        "at <font face='Courier'>doc_pipeline/document_ingestion.py:250</font>.", body))

    story.append(P("Supported file types", s["H2"]))
    story.append(P(
        "Parsers wired at <font face='Courier'>doc_pipeline/document_ingestion.py:215-218</font>. "
        "Anything else is rejected with <i>Unsupported file type: &lt;ext&gt;</i>. "
        "There is no <font face='Courier'>.md / .csv / .docx / .html</font> parser — "
        "convert to <font face='Courier'>.txt</font> first (see "
        "<font face='Courier'>scripts/ingest_piston_work_orders.py</font> for the pattern).",
        body))

    story.append(_table([
        ["Extension", "Parser", "What is extracted"],
        [".pdf", "PDFParser (pdfplumber)",
         "Per-page text + table cells; metadata.page, metadata.has_tables"],
        [".txt", "TXTParser",
         "Whole file as one document"],
        [".xlsx / .xls", "ExcelParser (openpyxl)",
         "One document segment per sheet; metadata.sheet_name, "
         "metadata.columns, metadata.row_count"],
    ], col_widths=[0.95 * inch, 1.65 * inch, 3.9 * inch]))
    story.append(Spacer(1, 8))

    story.append(P("ACL / classification by folder name", s["H2"]))
    story.append(P(
        "Defined at <font face='Courier'>core/document_acl.py:60</font>. The "
        "classification is attached to chunk metadata at ingestion; retrievers "
        "filter against the caller&rsquo;s role.", body))

    story.append(_table([
        ["Subfolder segment", "Classification", "Visible to roles"],
        ["management/ or confidential/", "confidential",
         "plant_manager, procurement_manager"],
        ["restricted/ or internal/", "restricted", "checker tier + above"],
        ["anything else (e.g. aviation/, work_orders/)", "public", "everyone"],
    ], col_widths=[2.4 * inch, 1.4 * inch, 2.7 * inch]))

    # ── 2. Schema ─────────────────────────────────────────────────────────
    story.append(P("2. Schema declaration", s["H1"]))
    story.append(P(
        "<b>File:</b> <font face='Courier'>schemas/manufacturing.yaml</font> "
        "(override path via env <font face='Courier'>KG_SCHEMA_PATH</font>). "
        "Two YAML keys you edit per domain:", body))

    story.append(P("entity_types", s["H2"]))
    story.append(P(
        "Each entry needs <b>name</b>. Then <i>one</i> of:", body))
    story.append(P(
        "&bull; <b>vocabulary: [phrase, phrase, &hellip;]</b> — closed list. "
        "The KeywordExtractor matches these phrases verbatim in chunk text and "
        "emits Mentions where <font face='Courier'>identifier == phrase</font>.", body))
    story.append(P(
        "&bull; <b>id_pattern: '&lt;regex&gt;'</b> — closed by regex. The "
        "CodeExtractor and the adapter&rsquo;s <font face='Courier'>EQUIPMENT_RE</font> "
        "lift matches out of text; the schema validator checks them.", body))
    story.append(P(
        "&bull; <b>Neither</b> &rarr; open-vocab. Only the NarrativeExtractor will "
        "produce Mentions (low confidence) here.", body))
    story.append(P(
        "Optional: <b>case_sensitive: false</b> makes both <font face='Courier'>vocabulary</font> "
        "and <font face='Courier'>id_pattern</font> case-insensitive.", body))

    story.append(P("edge_types", s["H2"]))
    story.append(P(
        "Each entry needs <b>name</b> (uppercase by convention), <b>source</b> "
        "(entity type name or list), <b>target</b> (entity type name or list). "
        "Optional: <b>min_cardinality</b>, <b>max_cardinality</b>.", body))

    story.append(P("traversal_routes", s["H2"]))
    story.append(P(
        "Used by the orchestrator to decide which edges to expand for a given "
        "query intent. Each route declares <b>seed_types</b> and <b>walk</b> (an "
        "ordered list of edge_type names).", body))

    # ── 3. KeywordExtractor edge emission ─────────────────────────────────
    story.append(PageBreak())
    story.append(P("3. Making KeywordExtractor emit edges (not just Mentions)", s["H1"]))
    story.append(P(
        "The KeywordExtractor emits an EdgeCandidate "
        "<font face='Courier'>equipment_id &rarr; vocab_phrase</font> only when:", body))
    story.append(P(
        "1. The chunk&rsquo;s <font face='Courier'>metadata.equipment_ids</font> is "
        "populated, <b>and</b>", body))
    story.append(P(
        "2. The schema declares an edge type whose <b>source</b> includes "
        "<font face='Courier'>Equipment</font> and <b>target</b> includes the "
        "vocab&rsquo;s entity type.", body))
    story.append(P(
        "<font face='Courier'>metadata.equipment_ids</font> is populated by "
        "<font face='Courier'>pipeline/adapter.py:_extract_entity_metadata</font> — "
        "it regex-scans chunk text with <font face='Courier'>EQUIPMENT_RE</font>. "
        "So the chunk text must contain a string matching that regex, or upstream "
        "must hand-set <font face='Courier'>equipment_ids</font> on the metadata.",
        body))
    story.append(P(
        "If your new domain has its own asset ID convention (e.g. "
        "<font face='Courier'>MED-IMG-NNNN</font> for medical imagers), you must:", body))
    story.append(P(
        "1. Add that pattern to the <b>Equipment id_pattern</b> in "
        "<font face='Courier'>schemas/manufacturing.yaml</font>.", body))
    story.append(P(
        "2. Add the same pattern to <b>EQUIPMENT_RE</b> in "
        "<font face='Courier'>pipeline/adapter.py:21</font>.", body))
    story.append(P(
        "3. Ensure documents contain the ID string in their body text.", body))
    story.append(P(
        "This is the only pure-Python edit per domain. Everything else is YAML.", body))

    # ── 4. Extractor stack ────────────────────────────────────────────────
    story.append(P("4. Extractor stack (runs on every chunk)", s["H1"]))
    story.append(P(
        "Order matters — earlier extractors win ties "
        "(<font face='Courier'>core/knowledge_graph.py:72</font>).", body))

    story.append(_table([
        ["#", "Extractor", "Author tag", "Conf.", "Reads", "Emits"],
        ["1", "CodeExtractor", "system:code", "1.00",
         "Chunk text via schema id_pattern regexes", "Mentions only"],
        ["2", "MetadataExtractor", "system:metadata", "0.95",
         "metadata.{equipment_ids, alarm_codes, part_numbers, fault_codes}",
         "Mentions + co-occurrence edges (Equipment→Alarm, Equipment→SparePart, Alarm→FailureMode)"],
        ["3", "KeywordExtractor", "system:keyword", "0.95",
         "Chunk text against every vocabulary in the schema",
         "Mentions per vocab hit + Equipment→<vocab> edges via schema-declared edge types"],
        ["4", "NarrativeExtractor", "system:llm_extract", "0.50",
         "Chunk text via regex (_SYMPTOM_PATTERNS, _PROCEDURE_PATTERNS)",
         "Open-vocab Symptom + Procedure Mentions; Equipment→Symptom edges if equipment_ids present"],
    ], col_widths=[0.2*inch, 1.05*inch, 0.95*inch, 0.45*inch, 1.85*inch, 2.0*inch]))
    story.append(Spacer(1, 6))
    story.append(P(
        "The schema validates every Mention "
        "(<font face='Courier'>Schema.validate_entity</font>) and every "
        "EdgeCandidate (<font face='Courier'>Schema.validate_edge</font>). "
        "Rejects land in <font face='Courier'>kg._rejected</font> — inspectable "
        "for diagnosing schema drift.", body))

    # ── 5. Pipeline flow ─────────────────────────────────────────────────
    story.append(P("5. Pipeline flow end-to-end", s["H1"]))
    flow = """doc_pipeline/input_docs/<your-domain>/*.{pdf,txt,xlsx}
   |
   v  DocumentIngestion.ingest_directory()       (one Document per file/page/sheet)
HybridChunker.chunk_documents()                  (recursive / semantic / sliding_window)
   |
   v  EmbeddingPipeline.build_index()
   |   |- bge-small-en-v1.5 (384-dim) encodes every chunk
   |   |- Qdrant upsert -> doc_pipeline/vector_store/qdrant/
   |   |- Saves doc_pipeline/vector_store/manufacturing_index.manifest.json
   |   '- Saves doc_pipeline/vector_store/manufacturing_index_chunks.json
   |
   v  pipeline/adapter.py:chunks_to_core_docs()
   |   |- stable_chunk_id(source, idx)
   |   |- regex-extracts equipment_ids / alarm_codes / part_numbers / fault_codes
   |   |  from chunk text into metadata
   |   '- outputs [{chunk_id, text, metadata}, ...]
   |
   v  KnowledgeGraph.build_from_documents()
       |- runs all 4 extractors on each doc
       |- validates Mentions + edges against schema
       |- stamps Provenance{author, confidence, source_chunk_id, timestamp}
       '- Saves data/processed/knowledge_graph.json"""
    story.append(Preformatted(flow, s["Code"]))

    # ── 6. Commands ──────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(P("6. Commands", s["H1"]))
    story.append(_table([
        ["Goal", "Command"],
        ["Add a new domain's docs",
         "Drop files under doc_pipeline/input_docs/<domain>/ "
         "(use a sub-folder for ACL)"],
        ["Convert JSONL/CSV/other to .txt",
         "Write a script under scripts/ modeled on "
         "scripts/ingest_piston_work_orders.py (deterministic filenames, "
         "body contains canonical IDs)"],
        ["Edit the ontology",
         "Edit schemas/manufacturing.yaml — add entity types with "
         "vocabulary / id_pattern, add edge types, add traversal routes"],
        ["Extend Equipment regex (rare)",
         "Edit pipeline/adapter.py:EQUIPMENT_RE"],
        ["Full rebuild (index + KG)",
         "python main.py --rebuild --no-llm"],
        ["Rebuild + run a query",
         "python main.py --rebuild --query \"your question\""],
        ["Rebuild + diagnostic (LLM)",
         "python main.py --rebuild --diagnostic \"your question\""],
        ["Inspect resulting KG",
         "python -c \"from core.knowledge_graph import KnowledgeGraph; "
         "kg=KnowledgeGraph(); kg.load(); "
         "print(kg.graph.number_of_nodes(), kg.graph.number_of_edges())\""],
        ["Inspect schema rejects",
         "After build: kg._rejected (list of dicts with kind, "
         "entity_type/relation, chunk_id, author)"],
        ["Launch Streamlit + FastAPI",
         "./run.sh   (FastAPI on :8000, Streamlit on :8501)"],
        ["Stop",
         "./stop.sh"],
    ], col_widths=[2.0 * inch, 4.5 * inch]))

    # ── 7. Output artefacts ──────────────────────────────────────────────
    story.append(P("7. Output artefacts (what gets written)", s["H1"]))
    story.append(_table([
        ["Path", "Content", "Persistence"],
        ["doc_pipeline/vector_store/qdrant/",
         "Qdrant collection \"manufacturing_corpus\" (embedded mode)",
         "Survives across runs; --rebuild overwrites"],
        ["doc_pipeline/vector_store/manufacturing_index.manifest.json",
         "{collection, count, embedding_model, dim} — sanity manifest",
         "Overwritten on rebuild"],
        ["doc_pipeline/vector_store/manufacturing_index_chunks.json",
         "All chunks with text + metadata (BM25 / KG source)",
         "Overwritten on rebuild"],
        ["data/processed/knowledge_graph.json",
         "NetworkX graph snapshot + provenance",
         "Overwritten on rebuild"],
        ["data/processed/audit.sqlite",
         "HITL approval log + LangGraph checkpoints",
         "Append-only"],
    ], col_widths=[2.45 * inch, 2.6 * inch, 1.45 * inch]))

    # ── 8. Worked example ───────────────────────────────────────────────
    story.append(PageBreak())
    story.append(P("8. Minimal worked example (medical-device domain)", s["H1"]))

    story.append(P("schemas/manufacturing.yaml — additions", s["H2"]))
    yaml_block = """entity_types:
  - name: Equipment
    id_pattern: '<existing>|^MED-IMG-\\d{4}$'          # add med-imager IDs
    case_sensitive: false

  - name: Component                                     # extend existing vocab
    case_sensitive: false
    vocabulary:
      - <existing terms...>
      - x-ray tube
      - detector panel
      - gantry motor
      - high-voltage generator

  - name: Cause
    case_sensitive: false
    vocabulary:
      - <existing terms...>
      - calibration drift
      - detector burn-in
      - cooling failure"""
    story.append(Preformatted(yaml_block, s["Code"]))

    story.append(P("pipeline/adapter.py — single regex edit", s["H2"]))
    story.append(Preformatted(
        "EQUIPMENT_RE = re.compile(r'...existing...|MED-IMG-\\d{4}')",
        s["Code"]))

    story.append(P("Stage data + rebuild + verify", s["H2"]))
    bash_block = """# stage data
mkdir -p doc_pipeline/input_docs/medical/{manuals,work_orders}
cp /path/to/service_manuals/*.pdf  doc_pipeline/input_docs/medical/manuals/
# convert your JSONL/CSV of work orders to .txt with the same shape as
# scripts/ingest_piston_work_orders.py

# rebuild
python main.py --rebuild --no-llm

# verify
python -c "
from core.knowledge_graph import KnowledgeGraph
kg = KnowledgeGraph(); kg.load()
from collections import Counter
c = Counter(d.get('entity_type') for _,d in kg.graph.nodes(data=True))
print(c)
print('rejects:', len(kg._rejected))
" """
    story.append(Preformatted(bash_block, s["Code"]))

    # ── 9. Code-change matrix ───────────────────────────────────────────
    story.append(P("9. What you CANNOT do without code changes", s["H1"]))
    story.append(_table([
        ["Wish", "Why it requires Python"],
        ["Ingest .md / .docx / .csv / .html",
         "Only .pdf/.txt/.xlsx/.xls have parsers — add a class to "
         "doc_pipeline/document_ingestion.py and register in the dispatch dict"],
        ["Custom entity-ID convention auto-tagged from chunk text",
         "pipeline/adapter.py:EQUIPMENT_RE is hard-coded"],
        ["Aliasing (e.g. \"mag\" and \"magneto\" -> same Component)",
         "KeywordExtractor uses the vocab phrase as the canonical identifier. "
         "For aliases, either add each alias as a separate vocab entry or port "
         "kgrag's full KeywordRule(canonical, phrases=[...]) form (~30 LOC)"],
        ["LLM-driven open-vocab Symptom/Procedure extraction",
         "NarrativeExtractor is regex-only; the interface accepts any Extractor "
         "subclass — swap it out (deferred item in DECISIONS.md)"],
        ["New extractor confidence tier (e.g. CMMS import)",
         "Subclass core/kg/extractors/base.Extractor, set "
         "author=ProvenanceAuthor.IMPORT_CMMS, register in "
         "KnowledgeGraph.__init__"],
    ], col_widths=[2.4 * inch, 4.1 * inch]))

    story.append(Spacer(1, 12))
    story.append(P(
        "<b>Bottom line.</b> Schema YAML + a converter script + "
        "<font face='Courier'>python main.py --rebuild</font> covers every "
        "domain whose data fits the supported file types and uses an Equipment "
        "ID convention you have declared.", body))

    return story


# ─── Page chrome ───────────────────────────────────────────────────────────


def _draw_chrome(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(INK_SOFT)
    # Header
    canvas.drawString(0.6 * inch, LETTER[1] - 0.45 * inch,
                      "Manufacturing Hybrid GraphRAG — domain onboarding")
    canvas.drawRightString(LETTER[0] - 0.6 * inch, LETTER[1] - 0.45 * inch,
                           "system_design/domain_onboarding_guide.pdf")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.3)
    canvas.line(0.6 * inch, LETTER[1] - 0.55 * inch,
                LETTER[0] - 0.6 * inch, LETTER[1] - 0.55 * inch)
    # Footer
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
        title="Adding a new domain — Manufacturing Hybrid GraphRAG",
        author="Manufacturing Hybrid GraphRAG",
    )
    s = _styles()
    doc.build(build_story(s), onFirstPage=_draw_chrome, onLaterPages=_draw_chrome)
    return OUTPUT


if __name__ == "__main__":
    path = main()
    print(f"Wrote {path} ({path.stat().st_size // 1024} KB)")
