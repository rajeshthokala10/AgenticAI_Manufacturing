"""
Unified configuration for the Manufacturing Hybrid GraphRAG pipeline.

Backed by ``pydantic-settings``: env vars are validated, defaults are typed,
and the canonical instance lives at ``config.settings``.

Module-level constants (``EMBEDDING_MODEL``, ``USE_RERANKER``, …) are
re-exported from that instance for back-compat with existing
``from config import X`` call-sites across ``core/``, ``pipeline/``,
``doc_pipeline/``, ``comparison/`` and the Streamlit / FastAPI front-ends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Paths (filesystem layout — not env-driven) ──────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent

# Canonical ingestion directory. The doc_pipeline ships realistic sample
# documents here; the application falls back to data/ if these are removed.
INPUT_DOCS_DIR: Path = BASE_DIR / "doc_pipeline" / "input_docs"

# Legacy/auxiliary corpora that the unified pipeline will also pick up.
DATA_DIR: Path = BASE_DIR / "data"
PDF_DIR: Path = DATA_DIR / "pdfs"
EXCEL_DIR: Path = DATA_DIR / "excel"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# Vector store and KG live alongside the doc_pipeline so we reuse the
# same artefacts whether the user launches the CLI, the Streamlit app, or the
# Hybrid GraphRAG orchestrator.
VECTOR_STORE_DIR: Path = BASE_DIR / "doc_pipeline" / "vector_store"
OUTPUT_DIR: Path = BASE_DIR / "doc_pipeline" / "output"

INDEX_NAME: str = "manufacturing_index"
GRAPH_PATH: Path = PROCESSED_DIR / "knowledge_graph.json"

# ── Domains (auto-discovered from schemas/*.yaml) ───────────────────────────
# The system runs N independent domains side-by-side. Each domain owns:
#   - schemas/<domain>.yaml      (ontology)
#   - doc_pipeline/input_docs/<domain>/      (raw documents)
#   - Qdrant collection ``<domain>_corpus``  (vector store)
#   - data/processed/knowledge_graph.<domain>.json   (KG snapshot)
#
# Adding a new domain is therefore a *zero-Python-edit* operation: drop a
# schema YAML into ``schemas/`` and the discovery below picks it up. This
# function is intentionally lazy + cached so the import has no I/O when the
# file list is unchanged.

SCHEMAS_DIR: Path = BASE_DIR / "schemas"


def _discover_domains() -> Dict[str, Dict[str, Any]]:
    """Scan ``schemas/*.yaml`` and return a registry keyed by domain id.

    Each entry contains the schema path, the display block, and the UX
    copy block (examples + empty_state + placeholder) — everything the
    UI layers need to render the domain without any Python edits.
    """
    import yaml as _yaml  # local import; PyYAML is optional in some envs

    registry: Dict[str, Dict[str, Any]] = {}
    if not SCHEMAS_DIR.exists():
        return registry

    for path in sorted(SCHEMAS_DIR.glob("*.yaml")):
        try:
            raw = _yaml.safe_load(path.read_text()) or {}
        except Exception:  # pragma: no cover - malformed YAML
            continue
        if not isinstance(raw, dict):
            continue
        domain_id = str(raw.get("domain") or path.stem).strip().lower()
        if not domain_id:
            continue
        display = dict(raw.get("display") or {})
        registry[domain_id] = {
            "schema_path": path,
            "display": {
                "label": display.get("label", domain_id.replace("_", " ").title()),
                "emoji": display.get("emoji", "📁"),
                "color": display.get("color", "#64748B"),  # slate-500 fallback
            },
            "examples": [str(x) for x in (raw.get("examples") or []) if x],
            "empty_state": {
                str(k): str(v).strip()
                for k, v in (raw.get("empty_state") or {}).items()
            },
            "placeholder": str(raw.get("placeholder") or "").strip(),
        }
    return registry


_DOMAIN_REGISTRY: Dict[str, Dict[str, Any]] = _discover_domains()


def _ordered_domains(reg: Dict[str, Dict[str, Any]]) -> Tuple[str, ...]:
    """Stable ordering: ``manufacturing`` first if present, then alphabetical."""
    keys = sorted(reg)
    if "manufacturing" in keys:
        keys.remove("manufacturing")
        return ("manufacturing", *keys)
    return tuple(keys)


DOMAINS: Tuple[str, ...] = _ordered_domains(_DOMAIN_REGISTRY)
DEFAULT_DOMAIN: str = DOMAINS[0] if DOMAINS else "manufacturing"

SCHEMA_PATHS: Dict[str, Path] = {
    d: _DOMAIN_REGISTRY[d]["schema_path"] for d in DOMAINS
}
DOMAIN_INPUT_DIRS: Dict[str, Path] = {
    d: INPUT_DOCS_DIR / d for d in DOMAINS
}
DOMAIN_QDRANT_COLLECTIONS: Dict[str, str] = {
    d: f"{d}_corpus" for d in DOMAINS
}
DOMAIN_KG_PATHS: Dict[str, Path] = {
    d: PROCESSED_DIR / f"knowledge_graph.{d}.json" for d in DOMAINS
}
DOMAIN_INDEX_NAMES: Dict[str, str] = {
    d: f"{d}_index" for d in DOMAINS
}
DOMAIN_DISPLAY: Dict[str, Dict[str, str]] = {
    d: _DOMAIN_REGISTRY[d]["display"] for d in DOMAINS
}
DOMAIN_EXAMPLES: Dict[str, list] = {
    d: list(_DOMAIN_REGISTRY[d].get("examples") or []) for d in DOMAINS
}
DOMAIN_EMPTY_STATE: Dict[str, Dict[str, str]] = {
    d: dict(_DOMAIN_REGISTRY[d].get("empty_state") or {}) for d in DOMAINS
}
DOMAIN_PLACEHOLDER: Dict[str, str] = {
    d: _DOMAIN_REGISTRY[d].get("placeholder") or "" for d in DOMAINS
}


def normalize_domain(domain: str | None) -> str:
    """Validate and canonicalize a domain string. Falls back to default."""
    if not domain:
        return DEFAULT_DOMAIN
    d = domain.strip().lower()
    if d not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
    return d


def schema_path(domain: str) -> Path:
    return SCHEMA_PATHS[normalize_domain(domain)]


def kg_path(domain: str) -> Path:
    return DOMAIN_KG_PATHS[normalize_domain(domain)]


def qdrant_collection(domain: str) -> str:
    return DOMAIN_QDRANT_COLLECTIONS[normalize_domain(domain)]


def index_name(domain: str) -> str:
    return DOMAIN_INDEX_NAMES[normalize_domain(domain)]


def input_dir(domain: str) -> Path:
    return DOMAIN_INPUT_DIRS[normalize_domain(domain)]


def domain_display(domain: str) -> Dict[str, str]:
    return DOMAIN_DISPLAY[normalize_domain(domain)]

# Qdrant lives under the vector store dir so a single ``rm -rf`` clears the
# whole indexed corpus. On-disk by default; point ``QDRANT_URL`` at a remote
# instance for distributed deployments.
QDRANT_PATH: Path = VECTOR_STORE_DIR / "qdrant"


class Settings(BaseSettings):
    """Runtime configuration sourced from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Models / credentials ────────────────────────────────────────────
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    # bge-small-en-v1.5 is the new default — same dim as MiniLM (384) but
    # consistently outperforms it on industrial / technical retrieval.
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # ── Tiered model routing (Ollama + OpenAI) ──────────────────────────
    ollama_base_url: str = "http://localhost:11434/v1"
    answer_model: str = "gpt-4o"
    critic_model: str = "qwen2.5:3b"
    retry_model: str = "gpt-4o"
    classify_model: str = "qwen2.5:3b"
    direct_llm_model: str = "gpt-4o-mini"
    classical_rag_model: str = "gpt-4o-mini"

    # ── Optional dedicated cause-ranking LLM ────────────────────────────
    cause_rank_model: str = "qwen2.5:3b"
    cause_rank_top_k: int = 5

    # ── Schema-onboarding agent (Streamlit "Onboard Domain" tab) ────────
    # Authoring a fresh ``schemas/<domain>.yaml`` is a high-stakes one-shot
    # task — closed vocabularies + regex id_patterns + KG edge declarations
    # all need to be coherent. qwen2.5:3b can't do it reliably. Defaults to
    # ``answer_model`` so it Just Works on cloud setups; override on local-
    # only deployments by pointing this at a larger Ollama model
    # (e.g. ``qwen2.5:14b``) or at a different OpenAI model.
    onboarding_model: str = ""

    # Optional fixed cause taxonomy (piston-style). Comma-separated list of
    # canonical cause names. When non-empty, the cause-ranker drops any
    # cause that is not in the list (anti-hallucination guarantee).
    cause_taxonomy: str = ""

    # ── Retrieval ───────────────────────────────────────────────────────
    top_k_retrieval: int = 10
    top_k_rerank: int = 5
    rrf_k: int = 60
    max_critic_retries: int = 2
    default_top_k: int = 5
    default_context_window: int = 1

    # ── Cross-encoder reranker (always-on by default now) ───────────────
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_candidate_pool: int = 20
    rerank_blend_weight: float = 0.7

    # ── Async parallel retrieval ────────────────────────────────────────
    use_parallel_retrieval: bool = True
    parallel_retrieval_timeout_s: float = 15.0

    # ── Semantic cache ──────────────────────────────────────────────────
    use_semantic_cache: bool = False
    semantic_cache_threshold: float = 0.97
    semantic_cache_max_size: int = 256
    semantic_cache_ttl_seconds: int = 3600

    # ── Guardrails ──────────────────────────────────────────────────────
    use_guardrails: bool = True
    guardrails_require_citations: bool = True
    guardrails_min_citations: int = 1
    guardrails_block_unsafe: bool = True

    # ── Tool-calling ────────────────────────────────────────────────────
    use_tools: bool = False
    tool_planner_model: str = "qwen2.5:3b"
    tool_planner_use_llm: bool = True

    # ── Orchestration ───────────────────────────────────────────────────
    use_langgraph: bool = False
    use_cause_ranking: bool = False

    # ── Two-stage generation (procedure drafting) ───────────────────────
    # When true, after retrieval + cause-ranking the pipeline emits a
    # structured procedure { steps: [{step, action, citations}] } via a
    # dedicated LLM call. Renders as Markdown for legacy answer surfaces.
    use_procedure_drafting: bool = False
    procedure_model: str = "gpt-4o"

    # ── HITL approval gate ──────────────────────────────────────────────
    use_hitl: bool = False
    hitl_risk_threshold: float = 0.6
    hitl_auto_approve_below_usd: float = 2000.0
    hitl_high_risk_keywords: str = (
        "lockout,tagout,hot work,fire,explosion,h2s,arc flash,confined space,"
        "fatal,injury,death,toxic,asphyxiation,radiation,permit-to-work,"
        "shutdown,emergency"
    )
    hitl_db_path: str = ""
    hitl_checkpoint_backend: str = "sqlite"

    # ── Qdrant vector store ─────────────────────────────────────────────
    # Empty string → embedded on-disk mode at ``QDRANT_PATH``. Use ":memory:"
    # for tests or an "http://host:6333" URL for a remote Qdrant server.
    qdrant_url: str = ""
    qdrant_collection: str = "manufacturing_corpus"

    # ── KG retrieval floor ──────────────────────────────────────────────
    # Minimum provenance confidence a node must have to contribute chunks
    # to the graph allow-list. 0.0 = trust everything (legacy behaviour).
    # 0.9 = drop narrative / LLM-extracted nodes; only structured /
    # deterministic / human-confirmed sources contribute. Trade-off
    # documented in DECISIONS.md.
    kg_retrieval_min_confidence: float = 0.0

    # ── Chunking ────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    semantic_similarity_threshold: float = 0.45
    semantic_min_chunk_size: int = 100
    semantic_max_chunk_size: int = 1500
    recursive_chunk_size: int = 1000
    recursive_chunk_overlap: int = 150
    sliding_window_size: int = 800
    sliding_window_step: int = 400
    faiss_ivf_min_chunks: int = 100
    faiss_nprobe_cap: int = 10

    # ── Legacy ChromaDB path (only used if explicitly enabled) ──────────
    chroma_collection: str = "manufacturing_docs"


settings = Settings()


# ── Back-compat module-level constants ──────────────────────────────────────
# Every existing call-site does `from config import X`. We expose the
# settings fields as upper-case constants so that import surface keeps
# working without any module changes elsewhere.

OPENAI_API_KEY: str = settings.openai_api_key
LLM_MODEL: str = settings.llm_model
EMBEDDING_MODEL: str = settings.embedding_model

OLLAMA_BASE_URL: str = settings.ollama_base_url
ANSWER_MODEL: str = settings.answer_model
ONBOARDING_MODEL: str = settings.onboarding_model.strip() or settings.answer_model
CRITIC_MODEL: str = settings.critic_model
RETRY_MODEL: str = settings.retry_model
CLASSIFY_MODEL: str = settings.classify_model
DIRECT_LLM_MODEL: str = settings.direct_llm_model
CLASSICAL_RAG_MODEL: str = settings.classical_rag_model

CAUSE_RANK_MODEL: str = settings.cause_rank_model
CAUSE_RANK_TOP_K: int = settings.cause_rank_top_k
CAUSE_TAXONOMY: Tuple[str, ...] = tuple(
    c.strip() for c in settings.cause_taxonomy.split(",") if c.strip()
)

TOP_K_RETRIEVAL: int = settings.top_k_retrieval
TOP_K_RERANK: int = settings.top_k_rerank
RRF_K: int = settings.rrf_k
MAX_CRITIC_RETRIES: int = settings.max_critic_retries
DEFAULT_TOP_K: int = settings.default_top_k
DEFAULT_CONTEXT_WINDOW: int = settings.default_context_window

USE_RERANKER: bool = settings.use_reranker
RERANKER_MODEL: str = settings.reranker_model
RERANK_CANDIDATE_POOL: int = settings.rerank_candidate_pool
RERANK_BLEND_WEIGHT: float = settings.rerank_blend_weight

USE_PARALLEL_RETRIEVAL: bool = settings.use_parallel_retrieval
PARALLEL_RETRIEVAL_TIMEOUT_S: float = settings.parallel_retrieval_timeout_s

USE_SEMANTIC_CACHE: bool = settings.use_semantic_cache
SEMANTIC_CACHE_THRESHOLD: float = settings.semantic_cache_threshold
SEMANTIC_CACHE_MAX_SIZE: int = settings.semantic_cache_max_size
SEMANTIC_CACHE_TTL_SECONDS: int = settings.semantic_cache_ttl_seconds

USE_GUARDRAILS: bool = settings.use_guardrails
GUARDRAILS_REQUIRE_CITATIONS: bool = settings.guardrails_require_citations
GUARDRAILS_MIN_CITATIONS: int = settings.guardrails_min_citations
GUARDRAILS_BLOCK_UNSAFE: bool = settings.guardrails_block_unsafe

USE_TOOLS: bool = settings.use_tools
TOOL_PLANNER_MODEL: str = settings.tool_planner_model
TOOL_PLANNER_USE_LLM: bool = settings.tool_planner_use_llm

USE_LANGGRAPH: bool = settings.use_langgraph
USE_CAUSE_RANKING: bool = settings.use_cause_ranking

USE_PROCEDURE_DRAFTING: bool = settings.use_procedure_drafting
PROCEDURE_MODEL: str = settings.procedure_model

USE_HITL: bool = settings.use_hitl
HITL_RISK_THRESHOLD: float = settings.hitl_risk_threshold
HITL_AUTO_APPROVE_BELOW_USD: float = settings.hitl_auto_approve_below_usd
HITL_HIGH_RISK_KEYWORDS: Tuple[str, ...] = tuple(
    kw.strip().lower() for kw in settings.hitl_high_risk_keywords.split(",") if kw.strip()
)
HITL_DB_PATH: Path = (
    Path(settings.hitl_db_path) if settings.hitl_db_path
    else PROCESSED_DIR / "audit.sqlite"
)
HITL_CHECKPOINT_BACKEND: str = settings.hitl_checkpoint_backend.strip().lower()

QDRANT_URL: str = settings.qdrant_url
QDRANT_COLLECTION: str = settings.qdrant_collection

KG_RETRIEVAL_MIN_CONFIDENCE: float = settings.kg_retrieval_min_confidence

CHUNK_SIZE: int = settings.chunk_size
CHUNK_OVERLAP: int = settings.chunk_overlap
SEMANTIC_SIMILARITY_THRESHOLD: float = settings.semantic_similarity_threshold
SEMANTIC_MIN_CHUNK_SIZE: int = settings.semantic_min_chunk_size
SEMANTIC_MAX_CHUNK_SIZE: int = settings.semantic_max_chunk_size
RECURSIVE_CHUNK_SIZE: int = settings.recursive_chunk_size
RECURSIVE_CHUNK_OVERLAP: int = settings.recursive_chunk_overlap
SLIDING_WINDOW_SIZE: int = settings.sliding_window_size
SLIDING_WINDOW_STEP: int = settings.sliding_window_step
FAISS_IVF_MIN_CHUNKS: int = settings.faiss_ivf_min_chunks
FAISS_NPROBE_CAP: int = settings.faiss_nprobe_cap

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".pdf", ".txt", ".xlsx", ".xls")

# Legacy ChromaDB constants (kept for the old VectorRetriever path).
CHROMA_COLLECTION: str = settings.chroma_collection
CHROMA_DIR: str = str(PROCESSED_DIR / "chromadb")
# Historical mis-spelling used by one or two call-sites — keep alias.
CHROMAb_COLLECTION: str = CHROMA_COLLECTION

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
              PDF_DIR, EXCEL_DIR, QDRANT_PATH):
        d.mkdir(parents=True, exist_ok=True)


def _openai_key_valid() -> bool:
    return bool(OPENAI_API_KEY) and OPENAI_API_KEY.startswith(("sk-", "sk_"))


def _is_local_model(model: str) -> bool:
    """Mirror of ``core.llm_client._is_local_model`` — kept here to avoid
    a circular import. Anything routed to Ollama is "local"."""
    if not model:
        return False
    return model.startswith(("qwen", "llama", "phi", "mistral:"))


_OLLAMA_PROBE_CACHE: dict = {}


def _ollama_reachable(base_url: str = "", timeout: float = 1.0) -> bool:
    """Cheap liveness probe against Ollama. Cached for the process lifetime."""
    url = (base_url or OLLAMA_BASE_URL or "").rstrip("/")
    if not url:
        return False
    if url in _OLLAMA_PROBE_CACHE:
        return _OLLAMA_PROBE_CACHE[url]
    # ``/v1`` is the OpenAI-compatible suffix; the native endpoint lives at
    # the parent. Probe ``/api/tags`` on the parent — that's the standard
    # liveness check.
    parent = url.rsplit("/v1", 1)[0]
    try:
        import urllib.request
        req = urllib.request.Request(f"{parent}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ok = 200 <= resp.status < 300
    except Exception:
        ok = False
    _OLLAMA_PROBE_CACHE[url] = ok
    return ok


def llm_available() -> bool:
    """Return True if at least one LLM backend can serve the configured
    answer/procedure path.

    Resolves in this order:

    1. Valid OpenAI key  → True (covers all cloud-model routes).
    2. Configured answer model routes to Ollama AND Ollama is reachable
       → True (free local stack only; cloud routes will still fail but the
       diagnostic copilot can run end-to-end).
    3. Otherwise → False.
    """
    if _openai_key_valid():
        return True
    # If the user has wired the answer surface to a local model, accept it
    # as long as Ollama is actually responding.
    if _is_local_model(ANSWER_MODEL) and _ollama_reachable():
        return True
    return False
