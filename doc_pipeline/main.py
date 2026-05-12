"""
Main runner — generates sample documents, indexes them, and runs demo queries
showing the full pipeline:
  ingestion → chunking → embedding → clarifier (intent/entity/slot) → query correction → retrieval
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PIPELINE_DIR = Path(__file__).resolve().parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from config import INPUT_DOCS_DIR, VECTOR_STORE_DIR, EMBEDDING_MODEL, ensure_dirs
from create_sample_docs import (
    create_quality_control_pdf,
    create_production_planning_pdf,
    create_safety_compliance_pdf,
    create_maintenance_pdf,
    create_sop_txt,
    create_supply_chain_txt,
    create_production_metrics_excel,
)
from rag_engine import RAGEngine
from query_correction import QueryCorrector
from clarifier_agent import ClarifierAgent


SAMPLE_DOC_FACTORIES = (
    create_quality_control_pdf,
    create_production_planning_pdf,
    create_safety_compliance_pdf,
    create_maintenance_pdf,
    create_sop_txt,
    create_supply_chain_txt,
    create_production_metrics_excel,
)


def banner(title: str, char: str = "=", width: int = 72) -> None:
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width + "\n")


def generate_documents() -> None:
    banner("GENERATING SAMPLE MANUFACTURING DOCUMENTS")
    for factory in SAMPLE_DOC_FACTORIES:
        factory()
    print(f"\n  All {len(SAMPLE_DOC_FACTORIES)} documents generated.\n")


QUERY_CORRECTION_SAMPLES = [
    "what is the maintanance schedul for CNC machines?",
    "show me the OEE and MTBF for plant A equiment",
    "safty compliace report for OSHA inpsection",
    "supplier scorcard for steel procudre",
    "what is the scrap rate for titanim parts in the stamping line?",
    "how do we handle deffect in qualitiy control?",
    "tell me about vibation analysis on spinle bearings",
    "what is the CAPA process for NCR",
    "inventry management and kanban system for spare parts",
    "hydralic pressure specificaiton for CNC machning center",
]

CLARIFIER_SAMPLES = [
    "What is the OEE for CNC Machining in Plant A for Q1 2026?",
    "What are the numbers for Plant B?",
    "Why did CNC Line 4 shut down in February?",
    "Something keeps breaking down, how do I fix it?",
    "Compare Nippon Steel vs ArcelorMittal on quality and delivery scores",
    "Are we compliant with OSHA 29 CFR 1910.147 lockout tagout requirements?",
    "How do I perform a tool change on the Mori Seiki NHX5000?",
    "How has MTBF improved from Q4 2025 to Q1 2026?",
    "What was the root cause of the spindle bearing failure on CNC-A-004?",
    "What is the current status of the heat treatment furnace HT-B-001?",
    "Tell me about steel",
    "What is the CPK and scrap rate for part TH-4401 on STAMP-A-002 in March?",
]

PIPELINE_SAMPLES = [
    "What is the OEE target for Q2 2026?",
    "How do we handle non-conformance reports?",
    "What is the maintanance schedul for CNC spindle bearings?",
    "Tell me about supplier scorcard for steel vendors",
    "What are the safty training requirements for new operators?",
    "What happened with CNC Line 4 shutdown in February?",
    "What is the CAPA process timeline for critical defects?",
    "environmental compliace VOC emissions status",
    "Compare the scrap rate between Plant A and Plant B welding",
    "How does the kanban system work for spare parts inventry?",
]


def demo_query_correction() -> None:
    banner("QUERY AUTO-CORRECTION DEMO")
    corrector = QueryCorrector()
    for q in QUERY_CORRECTION_SAMPLES:
        result = corrector.correct(q)
        print(f"  Original:  {result.original}")
        print(f"  Corrected: {result.corrected}")
        for fix in result.corrections_applied:
            print(f"    * {fix}")
        print(f"  Expanded:  {result.expanded[:120]}...\n")


def demo_clarifier_agent() -> None:
    banner("CLARIFIER AGENT DEMO — Intent / Entity / Slot Analysis")
    agent = ClarifierAgent()
    for q in CLARIFIER_SAMPLES:
        print(agent.format_analysis(agent.analyze(q)) + "\n")


def demo_queries(engine: RAGEngine) -> None:
    banner("FULL PIPELINE DEMO — Clarifier + Correction + Retrieval")
    for q in PIPELINE_SAMPLES:
        response = engine.query(q, top_k=3, show_corrections=True, show_clarifier=True)
        print(response.formatted_output + "\n")


def print_index_stats(stats: dict) -> None:
    print("\n-- Index Statistics --")
    print(f"  Documents ingested:  {stats['documents_ingested']}")
    print(f"  Chunks created:      {stats['chunks_created']}")
    print(f"  FAISS vectors:       {stats['index_vectors']}")
    print(f"  Embedding dimension: {stats['embedding_dim']}")
    print(f"  Sources:")
    for src in stats['sources']:
        print(f"    * {Path(src).name}")


def main() -> None:
    start_time = time.time()

    print("\n" + "#" * 72)
    print("#" + " " * 70 + "#")
    print("#   MANUFACTURING DOCUMENT PROCESSING & RAG PIPELINE" + " " * 19 + "#")
    print("#   with Clarifier Agent (Intent / Entity / Slot Filling)" + " " * 13 + "#")
    print("#" + " " * 70 + "#")
    print("#" * 72)

    ensure_dirs()
    generate_documents()
    demo_query_correction()
    demo_clarifier_agent()

    banner("BUILDING RAG INDEX", char="#")
    engine = RAGEngine(
        input_dir=INPUT_DOCS_DIR,
        index_dir=VECTOR_STORE_DIR,
        model_name=EMBEDDING_MODEL,
    )
    print_index_stats(engine.index_documents(save=True))

    demo_queries(engine)

    print(f"\n  Total pipeline time: {time.time() - start_time:.1f}s")
    print("  Pipeline complete.\n")


if __name__ == "__main__":
    main()
