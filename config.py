"""
Unified configuration for the Manufacturing Hybrid GraphRAG pipeline.

This is the single source of truth for paths, models, and thresholds shared
across the doc_pipeline ingestion layer, the core retrieval/KG layer, and the
top-level Streamlit application.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent

# Canonical ingestion directory. The doc_pipeline ships seven realistic sample
# documents here; the application falls back to data/ if these are removed.
INPUT_DOCS_DIR: Path = BASE_DIR / "doc_pipeline" / "input_docs"

# Legacy/auxiliary corpora that the unified pipeline will also pick up.
DATA_DIR: Path = BASE_DIR / "data"
PDF_DIR: Path = DATA_DIR / "pdfs"
EXCEL_DIR: Path = DATA_DIR / "excel"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# Vector store (FAISS) and KG live alongside the doc_pipeline so we reuse the
# same artefacts whether the user launches the CLI, the Streamlit app, or the
# Hybrid GraphRAG orchestrator.
VECTOR_STORE_DIR: Path = BASE_DIR / "doc_pipeline" / "vector_store"
OUTPUT_DIR: Path = BASE_DIR / "doc_pipeline" / "output"

INDEX_NAME: str = "manufacturing_index"
GRAPH_PATH: Path = PROCESSED_DIR / "knowledge_graph.json"

# ── Models / credentials ────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Tiered model routing (Ollama + OpenAI) ──────────────────────────────────
# Strong cloud models for user-facing answers; local Qwen for auxiliary tasks.
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
ANSWER_MODEL: str = os.getenv("ANSWER_MODEL", "gpt-4o")
CRITIC_MODEL: str = os.getenv("CRITIC_MODEL", "qwen2.5:3b")
RETRY_MODEL: str = os.getenv("RETRY_MODEL", "gpt-4o")
CLASSIFY_MODEL: str = os.getenv("CLASSIFY_MODEL", "qwen2.5:3b")
DIRECT_LLM_MODEL: str = os.getenv("DIRECT_LLM_MODEL", "gpt-4o-mini")
CLASSICAL_RAG_MODEL: str = os.getenv("CLASSICAL_RAG_MODEL", "gpt-4o-mini")

# Optional dedicated cause-ranking LLM (free local default via Ollama).
# Used only when USE_CAUSE_RANKING=true *and* the query is a troubleshooting /
# failure-analysis intent. See core/cause_ranker.py for details.
CAUSE_RANK_MODEL: str = os.getenv("CAUSE_RANK_MODEL", "qwen2.5:3b")
CAUSE_RANK_TOP_K: int = int(os.getenv("CAUSE_RANK_TOP_K", "5"))

# ── Retrieval ───────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "10"))
TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", "5"))
RRF_K: int = 60
MAX_CRITIC_RETRIES: int = int(os.getenv("MAX_CRITIC_RETRIES", "2"))
DEFAULT_TOP_K: int = 5
DEFAULT_CONTEXT_WINDOW: int = 1

# ── Orchestration ──────────────────────────────────────────────────────────
# Switch between the legacy procedural Orchestrator (core/orchestrator.py) and
# the LangGraph StateGraph-based orchestrator (pipeline/langgraph_orchestrator.py).
# Both run the same retrieval → answer → critic → retry flow but the LangGraph
# version makes the state transitions explicit and integrates with the
# wider LangChain ecosystem (tracing, checkpointing, etc.).
USE_LANGGRAPH: bool = os.getenv("USE_LANGGRAPH", "false").strip().lower() in (
    "1", "true", "yes", "on",
)

# Insert a dedicated cause-ranking LLM stage between retrieval and answer
# generation. Active only for troubleshooting / failure-analysis intents — the
# stage short-circuits to an empty result for unrelated queries.
USE_CAUSE_RANKING: bool = os.getenv("USE_CAUSE_RANKING", "false").strip().lower() in (
    "1", "true", "yes", "on",
)

# ── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 64

SEMANTIC_SIMILARITY_THRESHOLD: float = 0.45
SEMANTIC_MIN_CHUNK_SIZE: int = 100
SEMANTIC_MAX_CHUNK_SIZE: int = 1500

RECURSIVE_CHUNK_SIZE: int = 1000
RECURSIVE_CHUNK_OVERLAP: int = 150

SLIDING_WINDOW_SIZE: int = 800
SLIDING_WINDOW_STEP: int = 400

FAISS_IVF_MIN_CHUNKS: int = 100
FAISS_NPROBE_CAP: int = 10

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt", ".xlsx", ".xls")

# ── Vector store (legacy ChromaDB path — only used if explicitly enabled) ───
CHROMAb_COLLECTION: str = "manufacturing_docs"
CHROMA_DIR: str = str(PROCESSED_DIR / "chromadb")

# ── Domain ontology (used by KG builder) ────────────────────────────────────
DOMAIN_ONTOLOGY = {
    "entity_types": [
        "Equipment", "Component", "Alarm", "FailureMode",
        "Symptom", "Cause", "Procedure", "SparePart", "Specification",
    ],
    "relation_types": [
        "HAS_COMPONENT", "TRIGGERS_ALARM", "CAUSES_FAILURE",
        "HAS_SYMPTOM", "RESOLVED_BY", "REQUIRES_PART",
        "FOLLOWS_PROCEDURE", "HAS_SPECIFICATION",
    ],
    "traversal_routes": {
        "symptom_to_fix": ["Symptom", "Cause", "FailureMode", "Procedure"],
        "alarm_to_procedure": ["Alarm", "Equipment", "FailureMode", "Procedure"],
        "equipment_to_parts": ["Equipment", "Component", "SparePart"],
    },
}


def ensure_dirs() -> None:
    """Create runtime directories the pipeline writes to."""
    for d in (INPUT_DOCS_DIR, VECTOR_STORE_DIR, OUTPUT_DIR, PROCESSED_DIR,
              PDF_DIR, EXCEL_DIR):
        d.mkdir(parents=True, exist_ok=True)


def llm_available() -> bool:
    """Return True if an OpenAI API key is configured."""
    return bool(OPENAI_API_KEY) and OPENAI_API_KEY.startswith(("sk-", "sk_"))
