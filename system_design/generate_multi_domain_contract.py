"""Generate the multi-domain contract PDF.

Documents the schema-driven multi-domain contract introduced on the
``multi-domain-onboarding-anil_y`` branch: which files are
domain-specific, what schema YAML blocks now exist, what the system
auto-derives, what's still global, and the minimum-viable workflow for
onboarding a new domain.

Companion to ``generate_domain_onboarding_guide.py`` (which covers the
original ingestion path). This file documents what the schema YAML can
*now* control without any Python edit.

Run from the repo root::

    python system_design/generate_multi_domain_contract.py

Output::

    system_design/multi_domain_contract.pdf
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

OUTPUT = Path(__file__).resolve().parent / "multi_domain_contract.pdf"


# ─── Palette (matches the rest of system_design/) ─────────────────────────

INK = HexColor("#0F172A")
INK_MID = HexColor("#334155")
INK_SOFT = HexColor("#64748B")
ACCENT = HexColor("#B45309")     # copper
RULE = HexColor("#E2E8F0")
CODE_BG = HexColor("#F1F5F9")
HEADER_BG = HexColor("#0F172A")
ALT_ROW = HexColor("#F8FAFC")
GOOD_BG = HexColor("#ECFDF5")
WARN_BG = HexColor("#FEF3C7")


# ─── Styles ────────────────────────────────────────────────────────────────


def _styles() -> dict:
    base = getSampleStyleSheet()
    out: dict = {}
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


def _code_lines(text: str) -> Preformatted:
    return Preformatted(text, style=_STYLES["Code"])


_STYLES: dict = {}


# ─── Content ───────────────────────────────────────────────────────────────


def build_story(s: dict):
    global _STYLES
    _STYLES = s
    story = []
    body = s["Body"]
    small = s["BodySmall"]

    # ── Cover ─────────────────────────────────────────────────────────────
    story.append(P("Multi-Domain Contract", s["Title"]))
    story.append(P(
        "Schema-driven domain onboarding for the AgenticAI Manufacturing "
        "Hybrid GraphRAG pipeline. What's domain-specific, what's "
        "auto-derived, what the schema YAML can now control without any "
        "Python edit, and what's still global.",
        s["Subtitle"],
    ))

    # ── 1. The promise ────────────────────────────────────────────────────
    story.append(P("1. The promise", s["H1"]))
    story.append(P(
        "Adding a new domain is a <b>two-path</b> change:", body))
    story.append(P(
        "&bull; <b>Author</b> a single YAML at "
        "<font face='Courier'>schemas/&lt;domain&gt;.yaml</font>", body))
    story.append(P(
        "&bull; <b>Drop</b> source documents into "
        "<font face='Courier'>doc_pipeline/input_docs/&lt;domain&gt;/</font>",
        body))
    story.append(P(
        "Everything else &mdash; UI selector, FastAPI routing, Qdrant "
        "collection, KG snapshot, FAISS index, per-domain LLM persona, "
        "HITL escalation vocabulary, clarifier intent / slot prompts, "
        "structured procedure drafter opt-in &mdash; is derived from the "
        "schema YAML at startup. Restart the API and the new domain "
        "appears in every surface.",
        body,
    ))
    story.append(P(
        "Manufacturing remains byte-identical: every schema-driven path "
        "falls back to the legacy module-level constants when the YAML "
        "block is absent, so no regression risk for the existing domain.",
        small,
    ))

    # ── 2. What you author per domain ─────────────────────────────────────
    story.append(P("2. Files you author per domain", s["H1"]))
    story.append(_table([
        ["Path", "What goes in it", "Required?"],
        ["schemas/<domain>.yaml",
         "The whole domain contract — ontology, prompts, safety "
         "keywords, clarifier vocab, procedure config, UI copy.",
         "Yes (single file)"],
        ["doc_pipeline/input_docs/<domain>/",
         "Raw PDFs / TXT / XLSX files for ingestion. Subfolders "
         "named management/ confidential/ restricted/ internal/ "
         "drive ACL classification (see existing "
         "domain_onboarding_guide.pdf).",
         "Yes (drop files here)"],
    ], col_widths=[2.0 * inch, 4.0 * inch, 1.0 * inch]))

    # ── 3. What the system auto-creates ───────────────────────────────────
    story.append(P("3. Auto-derived artifacts (don't touch)", s["H1"]))
    story.append(_table([
        ["Path / resource", "Generated by"],
        ["data/processed/knowledge_graph.<domain>.json",
         "KG builder on first ingest (config.kg_path)"],
        ["Qdrant collection <domain>_corpus",
         "Vector indexer (config.qdrant_collection)"],
        ["FAISS index name <domain>_index",
         "Embedding pipeline (config.index_name)"],
        ["/api/domains entry",
         "config._discover_domains() scans schemas/*.yaml at startup"],
        ["Streamlit + Next.js header switcher",
         "Both UIs read from /api/domains"],
        ["Per-domain ManufacturingPipeline instance",
         "api/server.py builds one pipeline per discovered domain"],
    ], col_widths=[3.4 * inch, 3.6 * inch]))

    # ── 4. Schema anatomy ─────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(P("4. Anatomy of schemas/<domain>.yaml", s["H1"]))
    story.append(P(
        "Each block below is independent. Fill in the ones you need; "
        "omitted blocks fall back to manufacturing-flavoured defaults.",
        body,
    ))

    story.append(P("4.1 Identity &amp; UI", s["H2"]))
    story.append(_table([
        ["Block", "Effect when set", "Fallback when omitted"],
        ["domain: <id>, version: 1",
         "Required identity",
         "—"],
        ["display: {label, emoji, color}",
         "Shows up in selectors / header chips",
         "Title-cased id + 📁 + slate-500"],
        ["placeholder:, empty_state:, examples:",
         "Chat input copy + landing page",
         "Generic fallback"],
    ], col_widths=[1.9 * inch, 2.6 * inch, 2.5 * inch]))

    story.append(P("4.2 Knowledge graph ontology", s["H2"]))
    story.append(_table([
        ["Block", "Effect when set", "Fallback when omitted"],
        ["entity_types: [...]",
         "Closed/regex/open-vocab entity types. Drives "
         "KeywordExtractor + CodeExtractor + schema validator.",
         "KG can't extract anything meaningfully"],
        ["edge_types: [...]",
         "Edges the KG will accept (with source/target type "
         "constraints and cardinality).",
         "Same — required"],
        ["traversal_routes: {...}",
         "Named walks the orchestrator uses for query-driven "
         "subgraph expansion.",
         "Routes silently skipped"],
        ["gap_thresholds: {...}",
         "KG gap detector's low-confidence / retrieval floors.",
         "Defaults from config"],
    ], col_widths=[1.9 * inch, 3.4 * inch, 1.7 * inch]))

    story.append(P("4.3 LLM personas &amp; prompts (NEW)", s["H2"]))
    story.append(P(
        "Read by <font face='Courier'>core/domain_prompts.get_prompt()</font>. "
        "Every block below is optional; any unset key falls back to the "
        "module-level constant in the matching <font face='Courier'>core/*.py</font> "
        "file (manufacturing-flavoured).",
        body,
    ))
    story.append(_table([
        ["Block (under prompts:)", "Consumed by", "Effect"],
        ["persona", "free-form, used by your own templates",
         "Short label — \"aviation maintenance copilot\""],
        ["answer_system", "core/orchestrator.py, pipeline/langgraph_orchestrator.py",
         "The system prompt for the free-form answer LLM call"],
        ["retry_system", "both orchestrators (critic-rejected retry path)",
         "Improved-answer system prompt (uses {critic_feedback})"],
        ["critic_rules", "core/critic.py",
         "The quality critic's evaluation criteria"],
        ["procedure_system", "core/procedure_drafter.py",
         "Structured drafter persona; controls safety-precondition "
         "phrasing (LOTO vs mag-ground vs none)"],
        ["cause_rank_system", "core/cause_ranker.py",
         "Root-cause ranking persona (uses {top_k}, {taxonomy_clause})"],
        ["classify_system", "core/query_formatter.py",
         "Intent classifier LLM prompt"],
        ["risk_grader_system / risk_grader_user",
         "core/criticality_classifier.py",
         "Tier-2 HITL risk grader prompts"],
    ], col_widths=[1.85 * inch, 2.35 * inch, 2.8 * inch]))

    story.append(P("4.4 Safety / HITL escalation (NEW)", s["H2"]))
    story.append(_table([
        ["Block", "Consumed by", "Effect"],
        ["safety.high_risk_keywords: [...]",
         "core/criticality_classifier.classify()",
         "Domain-specific keyword list that escalates a query/answer "
         "to a human supervisor. Aviation drops LOTO/H2S/arc-flash, "
         "adds mayday/in-flight-fire/AD-compliance/etc."],
    ], col_widths=[1.85 * inch, 2.35 * inch, 2.8 * inch]))

    story.append(PageBreak())
    story.append(P("4.5 Clarifier vocabulary (NEW)", s["H2"]))
    story.append(P(
        "Read by <font face='Courier'>doc_pipeline/clarifier_agent.py</font> "
        "via the <font face='Courier'>clarifier:</font> top-level block.",
        body,
    ))
    story.append(_table([
        ["Sub-block", "Effect when set", "Layering behaviour"],
        ["equipment_patterns: [{pattern, type}]",
         "Extra regexes for entity extraction (e.g. WO:ASRS:NNN, "
         "ENG:O-360-A4M).",
         "Appended to manufacturing defaults"],
        ["part_number_patterns: [{pattern, type}]",
         "Domain-specific part-number regexes.",
         "Appended"],
        ["supplier_names: {key: label}",
         "Dictionary lookups for supplier entities.",
         "Merged (schema wins on collision)"],
        ["metric_names: {key: label}",
         "EGT/CHT/MP/RPM for aviation; OEE/MTBF for manufacturing.",
         "Merged"],
        ["department_names: {key: label}",
         "Domain-specific department vocabulary.",
         "Merged"],
        ["intent_patterns: [{intent, patterns, boost}]",
         "Extra intent regexes — e.g. aviation's \"mag drop\" boosts "
         "TROUBLESHOOTING to 0.93 confidence.",
         "Appended to manufacturing defaults"],
        ["slot_templates: {INTENT: [...]}",
         "Per-intent clarification prompts the user sees (\"Which "
         "engine?\" not \"Which CNC line?\").",
         "FULL replacement for the listed intents"],
    ], col_widths=[1.95 * inch, 2.85 * inch, 2.2 * inch]))

    story.append(P("4.6 Query auto-correction", s["H2"]))
    story.append(P(
        "Read by <font face='Courier'>doc_pipeline/query_correction.py</font> "
        "via the <font face='Courier'>corrections:</font> top-level block.",
        body,
    ))
    story.append(_table([
        ["Sub-block", "Effect when set"],
        ["corrections.acronyms: {key: expansion}",
         "Domain acronym expansions (asrs, far, amt, mel, &hellip;). "
         "Merged with manufacturing defaults; schema wins on collision."],
        ["corrections.misspellings: {wrong: right}",
         "Per-domain typo fixes (carberator → carburetor)."],
        ["corrections.synonyms: {key: [&hellip;]}",
         "Term enrichment (\"carb ice\" → carburetor icing, induction icing)."],
        ["corrections.vocabulary: [...]",
         "Extra protected vocabulary so close-match correction won't "
         "rewrite domain terms."],
    ], col_widths=[2.55 * inch, 4.45 * inch]))

    story.append(P("4.7 Structured procedure drafter opt-in (NEW)", s["H2"]))
    story.append(_table([
        ["Block", "Effect", "Default"],
        ["procedure.enabled: bool",
         "Master switch. Set false to skip the structured drafter "
         "entirely (legal lookup, market research, medical reference, "
         "etc.).",
         "true"],
        ["procedure.trigger_intents: [...]",
         "Substring list — drafter fires when any trigger is a "
         "substring of the classified intent. Match rule: "
         "`trigger in intent`.",
         "Falls back to core/cause_ranker._TROUBLESHOOTING_TRIGGERS"],
    ], col_widths=[1.95 * inch, 3.45 * inch, 1.6 * inch]))

    # ── 5. Known gaps ─────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(P("5. Known gaps still global (not yet schema-driven)", s["H1"]))
    story.append(P(
        "These remain manufacturing-leaning and may need a small code "
        "change for non-adjacent domains. None of them block read-only "
        "Q&amp;A on a new domain &mdash; they're mostly noise rather than "
        "failure paths.",
        body,
    ))
    story.append(_table([
        ["Where", "What's still global", "When it bites"],
        ["core/query_formatter.py",
         "MANUFACTURING_ABBREVIATIONS, INTENT_PATTERNS, _extract_entities "
         "regexes",
         "Only the regex fallback when the LLM classifier is off"],
        ["core/cause_ranker.py",
         "_TROUBLESHOOTING_TRIGGERS substring list gates the cause "
         "ranker (only the procedure drafter gate became schema-aware)",
         "Domains with uncommon intent vocabulary may skip cause ranking"],
        ["config.CAUSE_TAXONOMY",
         "Closed list of allowed cause names (manufacturing-flavoured: "
         "bearing wear, seal leak, &hellip;)",
         "New domains likely want to bypass or override"],
        ["doc_pipeline/clarifier_agent.Intent",
         "Fixed enum (LOOKUP / COMPARISON / TROUBLESHOOTING / &hellip;); "
         "schemas can add patterns but not invent new intent names",
         "Domains with fundamentally different intents"],
        ["core/purchase_request.py",
         "Assumes USD purchase-order parsing for HITL",
         "Irrelevant but inert on non-PO domains"],
    ], col_widths=[1.85 * inch, 3.25 * inch, 1.9 * inch]))

    # ── 6. Minimum viable workflow ────────────────────────────────────────
    story.append(P("6. Minimum viable new domain (30-second version)", s["H1"]))
    story.append(P(
        "1. Create <font face='Courier'>schemas/&lt;domain&gt;.yaml</font> "
        "with at minimum: <font face='Courier'>domain</font>, "
        "<font face='Courier'>version</font>, <font face='Courier'>display</font>, "
        "<font face='Courier'>entity_types</font>, <font face='Courier'>edge_types</font>.",
        body,
    ))
    story.append(P(
        "2. Drop documents into "
        "<font face='Courier'>doc_pipeline/input_docs/&lt;domain&gt;/</font>.",
        body,
    ))
    story.append(P(
        "3. Restart the API. Discovery auto-registers the domain, the UI "
        "gets a new tab, chat/copilot route queries to a fresh Qdrant "
        "collection + KG, and the system answers with the manufacturing "
        "persona (fallback).",
        body,
    ))
    story.append(P(
        "4. (Recommended) Add <font face='Courier'>prompts:</font>, "
        "<font face='Courier'>safety:</font>, "
        "<font face='Courier'>clarifier:</font>, and "
        "<font face='Courier'>procedure:</font> blocks to stop sounding "
        "like a manufacturing copilot. Author each block once; the "
        "system reads them on the next restart.",
        body,
    ))

    # ── 7. Worked example: aviation overrides ─────────────────────────────
    story.append(P("7. Worked example — aviation schema overrides", s["H1"]))
    story.append(P(
        "Lifted from <font face='Courier'>schemas/aviation.yaml</font> — "
        "shows the shape of every block.",
        body,
    ))
    story.append(_code_lines("""prompts:
  persona: aviation piston-engine maintenance copilot
  answer_system: |
    You are an aviation maintenance copilot specialising in
    piston-engine diagnostics, FAA AMT chapters, and ASRS work-order
    analysis. ... [full text in schemas/aviation.yaml]
  critic_rules: |
    You are a strict quality critic for aviation piston-engine
    diagnostic answers. ... [full text]
  procedure_system: |
    Sequence the steps so safety preconditions (mag-ground, prop
    clear, fuel shutoff, master off) come BEFORE any inspection.

safety:
  high_risk_keywords:
    - emergency
    - mayday
    - in-flight
    - airworthiness
    - prop strike
    - fuel leak
    - in-flight fire
    - ad compliance

clarifier:
  equipment_patterns:
    - { pattern: '\\\\b(WO:ASRS:\\\\d+)\\\\b',   type: work_order_id }
    - { pattern: '\\\\b(ENG:[A-Z0-9_-]+)\\\\b', type: engine_id }
  metric_names:
    egt: Exhaust Gas Temperature
    cht: Cylinder Head Temperature
  intent_patterns:
    - intent: TROUBLESHOOTING
      patterns:
        - '\\\\b(mag.?drop|magneto drop)\\\\b'
        - '\\\\b(carb ice|carburet?or ic(?:ing|e))\\\\b'
      boost: 0.93
  slot_templates:
    TROUBLESHOOTING:
      - name: aircraft_or_engine
        entity_types: [engine_id, equipment_id]
        required: true
        prompt: "Which engine or aircraft? (e.g., ENG:O-360-A4M, N12345)"

corrections:
  acronyms: { asrs: Aviation Safety Reporting System, amt: aviation maintenance technician }
  misspellings: { carberator: carburetor, magnetto: magneto }

procedure:
  enabled: true
  trigger_intents: [troubleshoot, root_cause, diagnos, repair]
"""))

    # ── 8. Source files touched ───────────────────────────────────────────
    story.append(P("8. Source files implementing the contract", s["H1"]))
    story.append(_table([
        ["File", "Role"],
        ["schemas/<domain>.yaml",
         "The authoring surface — everything domain-specific lives here"],
        ["core/domain_prompts.py",
         "Cached loader: get_prompt(), get_high_risk_keywords(), "
         "get_procedure_config(), procedure_should_run()"],
        ["config.py (_discover_domains)",
         "Startup scan of schemas/*.yaml → DOMAINS, SCHEMA_PATHS, "
         "DOMAIN_QDRANT_COLLECTIONS, DOMAIN_KG_PATHS, DOMAIN_DISPLAY, "
         "DOMAIN_EXAMPLES, etc."],
        ["core/orchestrator.py, pipeline/langgraph_orchestrator.py",
         "Read self._domain from KG; pass it to every LLM call; gate "
         "procedure drafter on procedure_should_run()"],
        ["core/critic.py, core/procedure_drafter.py, "
         "core/cause_ranker.py, core/query_formatter.py, "
         "core/criticality_classifier.py",
         "Each takes a domain kwarg and resolves its system prompt via "
         "get_prompt() (or keyword list via get_high_risk_keywords())"],
        ["doc_pipeline/clarifier_agent.py",
         "IntentClassifier, EntityExtractor, SlotFiller all take "
         "domain and merge schema's clarifier: block"],
        ["doc_pipeline/query_correction.py",
         "QueryCorrector(domain=) merges schema's corrections: block"],
    ], col_widths=[2.75 * inch, 4.25 * inch]))

    return story


def main() -> Path:
    s = _styles()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Multi-Domain Contract",
        author="AgenticAI Manufacturing — Hybrid GraphRAG",
    )
    doc.build(build_story(s))
    return OUTPUT


if __name__ == "__main__":
    out = main()
    print(f"wrote {out}")
