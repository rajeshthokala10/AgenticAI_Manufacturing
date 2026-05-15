"""
ManufacturingPipeline — the single object that wires every layer together.

Stages
------
1. Ingest documents from `input_docs/` + `data/pdfs/` + `data/excel/`
   via doc_pipeline's PDF/TXT/Excel parsers.
2. Smart-chunk them with the doc_pipeline HybridChunker (semantic / recursive /
   sliding window per file type).
3. Embed (bge-small-en-v1.5 by default) and index with Qdrant
   (``doc_pipeline.embeddings.EmbeddingPipeline``).
4. Build the knowledge graph (core/knowledge_graph.py) from chunk metadata.
5. At query time:
   * Quick mode      → Clarifier + QueryCorrector + Qdrant top-k.
   * Diagnostic mode → Clarifier + Hybrid (BM25 + Qdrant + Graph + RRF)
                       + bge-reranker + LLM answer + Critic loop
                       (if OPENAI_API_KEY set).
   * Classical RAG   → Qdrant only + LLM answer (baseline for comparison).
   * Direct LLM      → no retrieval (baseline for comparison).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

logger = logging.getLogger("pipeline.unified")


class PipelineMode(str, Enum):
    QUICK = "quick"
    DIAGNOSTIC = "diagnostic"
    CLASSICAL = "classical_rag"
    DIRECT = "direct_llm"


@dataclass
class PipelineResult:
    mode: str
    query: str
    answer: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    clarification: Optional[Any] = None
    correction: Optional[Any] = None
    graph_context: Optional[Dict] = None
    cause_ranking: Optional[Dict] = None
    procedure: Optional[Dict] = None
    critic: Optional[Dict] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    formatted_output: str = ""

    # HITL extensions (Phases A + B + C). All optional; default = no approval.
    risk: Optional[Dict[str, Any]] = None
    purchase_request: Optional[Dict[str, Any]] = None
    requires_approval: bool = False
    approval_thread_id: Optional[str] = None
    rejected: bool = False
    human_decision: Optional[Dict[str, Any]] = None
    interrupt_payload: Optional[Dict[str, Any]] = None
    pipeline_status: str = "complete"  # 'complete' | 'awaiting_approval' | 'rejected'

    # Guardrails + tool-calling extensions
    guardrails: Optional[Dict[str, Any]] = None
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    pending_tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        out = {
            "mode": self.mode,
            "query": self.query,
            "answer": self.answer,
            "evidence": self.evidence,
            "graph_context": self.graph_context,
            "cause_ranking": self.cause_ranking,
            "procedure": self.procedure,
            "critic": self.critic,
            "metrics": self.metrics,
            "formatted_output": self.formatted_output,
            "risk": self.risk,
            "purchase_request": self.purchase_request,
            "requires_approval": self.requires_approval,
            "approval_thread_id": self.approval_thread_id,
            "rejected": self.rejected,
            "human_decision": self.human_decision,
            "interrupt_payload": self.interrupt_payload,
            "pipeline_status": self.pipeline_status,
            "guardrails": self.guardrails,
            "tool_results": self.tool_results,
            "pending_tool_calls": self.pending_tool_calls,
        }
        if self.clarification is not None:
            out["intent"] = self.clarification.intent.value
            out["entities"] = [(e.entity_type, e.normalized) for e in self.clarification.entities]
            out["is_complete"] = self.clarification.is_complete
        return out


_PIPELINE_CACHE: Dict[str, "ManufacturingPipeline"] = {}


def get_pipeline(domain: Optional[str] = None) -> "ManufacturingPipeline":
    """Return the cached pipeline for ``domain``, building one on first call.

    Two domains share heavy ML models (embedder, reranker) via the
    process-wide cache so a second pipeline is cheap to construct.
    """
    from config import normalize_domain
    d = normalize_domain(domain)
    pipe = _PIPELINE_CACHE.get(d)
    if pipe is None:
        pipe = ManufacturingPipeline(domain=d)
        _PIPELINE_CACHE[d] = pipe
    return pipe


class ManufacturingPipeline:
    """Single end-to-end pipeline for ingestion, indexing, and querying."""

    def __init__(
        self,
        input_dirs: Optional[Iterable[Path | str]] = None,
        index_dir: Optional[Path | str] = None,
        embedding_model: Optional[str] = None,
        domain: Optional[str] = None,
    ):
        from config import (
            INPUT_DOCS_DIR, PDF_DIR, EXCEL_DIR, VECTOR_STORE_DIR,
            EMBEDDING_MODEL, ensure_dirs, normalize_domain, input_dir,
        )
        ensure_dirs()

        # Per-domain state — the schema, KG file, Qdrant collection and
        # input directory all key off ``self.domain``.
        self.domain: str = normalize_domain(domain)

        # When no explicit input_dirs override is given, default to the
        # domain's own ingestion folder. Legacy data/ folders are only added
        # for the manufacturing domain (back-compat with existing layouts).
        if input_dirs is None:
            default_inputs: List[Path] = [input_dir(self.domain)]
            if self.domain == "manufacturing":
                default_inputs += [PDF_DIR, EXCEL_DIR]
            self.input_dirs = [Path(p) for p in default_inputs if Path(p).exists()]
        else:
            self.input_dirs = [Path(p) for p in input_dirs if Path(p).exists()]

        self.index_dir = Path(index_dir or VECTOR_STORE_DIR)
        self.embedding_model = embedding_model or EMBEDDING_MODEL

        from doc_pipeline.document_ingestion import DocumentIngestion
        from doc_pipeline.embeddings import EmbeddingPipeline
        from doc_pipeline.chunking import HybridChunker
        from doc_pipeline.clarifier_agent import ClarifierAgent
        from doc_pipeline.query_correction import QueryCorrector

        self.ingestion = DocumentIngestion()
        self.embedding_pipeline = EmbeddingPipeline(
            model_name=self.embedding_model,
            index_dir=self.index_dir,
            domain=self.domain,
        )
        self.chunker = HybridChunker(embedding_model=self.embedding_pipeline.get_model())
        self.clarifier = ClarifierAgent(domain=self.domain)
        self.query_corrector = QueryCorrector(domain=self.domain)

        self.documents: List[Dict] = []
        self.kg = None
        self.orchestrator = None
        self.classical_rag = None
        self.faiss_retriever = None
        self._ready = False
        self._llm_enabled = False
        self._orchestrator_engine: str = "procedural"  # or "langgraph"

    # ─── Build ───────────────────────────────────────────────────────────

    def build_or_load(self, rebuild: bool = False, enable_llm: bool = True) -> Dict[str, Any]:
        """Build (or reload) all indexes. Returns stats dict."""
        if self._ready and not rebuild:
            return self._stats()

        stats: Dict[str, Any] = {}
        if not rebuild and self.embedding_pipeline.has_saved_index():
            logger.info("Loading FAISS index from %s", self.index_dir)
            self.embedding_pipeline.load()
            stats["index_source"] = "loaded"
        else:
            self._ingest_and_index()
            stats["index_source"] = "built"

        # Always build documents (from chunks already in the embedding pipeline)
        # so the KG / BM25 layers have the right shape.
        self._materialize_core_documents()

        # Knowledge graph (per-domain schema + KG file)
        from core.knowledge_graph import KnowledgeGraph
        self.kg = KnowledgeGraph(domain=self.domain)
        if not rebuild and self.kg.load():
            logger.info("Loaded knowledge graph (%d nodes, %d edges)",
                        self.kg.graph.number_of_nodes(), self.kg.graph.number_of_edges())
        else:
            logger.info("Building knowledge graph...")
            self.kg.build_from_documents(self.documents)

        # Qdrant-backed vector retriever shared by orchestrator & classical RAG.
        # Variable name kept as ``faiss_retriever`` for back-compat with the
        # existing call-sites; the class itself is now ``QdrantVectorRetriever``.
        from pipeline.faiss_retriever import QdrantVectorRetriever
        self.faiss_retriever = QdrantVectorRetriever(embedding_pipeline=self.embedding_pipeline)
        self.faiss_retriever.attach(self.embedding_pipeline, self.documents)

        # Decide LLM availability
        from config import llm_available
        self._llm_enabled = enable_llm and llm_available()

        if self._llm_enabled:
            from comparison.classical_rag import ClassicalRAG
            from config import USE_LANGGRAPH

            self.orchestrator = self._build_orchestrator(USE_LANGGRAPH)
            self.orchestrator.initialize()

            self.classical_rag = ClassicalRAG(
                self.documents,
                vector_retriever=self.faiss_retriever,
                skip_index_build=True,
            )
            self.classical_rag.initialize()

        self._ready = True
        stats.update(self._stats())
        return stats

    def _ingest_and_index(self) -> None:
        all_documents = []
        for directory in self.input_dirs:
            try:
                all_documents.extend(self.ingestion.ingest_directory(directory))
            except Exception as exc:
                logger.warning("Skipping %s: %s", directory, exc)

        if not all_documents:
            raise RuntimeError(
                f"No supported documents found in any of: {self.input_dirs}"
            )

        chunks = self.chunker.chunk_documents(all_documents)
        self.embedding_pipeline.build_index(chunks)
        self.embedding_pipeline.save()

    def _materialize_core_documents(self) -> None:
        """Turn doc_pipeline Chunks into the core dict shape (with entity metadata).

        ``domain`` is passed to the adapter so it can union the schema's
        Equipment id_pattern with the legacy regex — every domain's asset
        tags get auto-lifted into ``metadata.equipment_ids`` without any
        Python edits to ``pipeline/adapter.py``.
        """
        from pipeline.adapter import chunks_to_core_docs
        self.documents = chunks_to_core_docs(
            self.embedding_pipeline.chunks, domain=self.domain,
        )
        logger.info("Materialized %d core documents", len(self.documents))

    def _build_orchestrator(self, use_langgraph: bool):
        """Construct the diagnostic orchestrator.

        Prefers the LangGraph engine when ``USE_LANGGRAPH=true`` and falls back
        to the procedural ``core.orchestrator.Orchestrator`` if langgraph is
        not installed or the env flag is off.
        """
        from core.orchestrator import Orchestrator as ProceduralOrchestrator

        embed_fn = self._make_embed_fn()

        if use_langgraph:
            try:
                from pipeline.langgraph_orchestrator import LangGraphOrchestrator

                logger.info("Diagnostic engine: LangGraph (USE_LANGGRAPH=true)")
                self._orchestrator_engine = "langgraph"
                return LangGraphOrchestrator(
                    self.documents,
                    self.kg,
                    vector_retriever=self.faiss_retriever,
                    skip_vector_build=True,
                    embed_fn=embed_fn,
                )
            except ImportError as exc:
                logger.warning(
                    "USE_LANGGRAPH=true but langgraph is not installed (%s); "
                    "falling back to the procedural orchestrator.",
                    exc,
                )

        logger.info("Diagnostic engine: procedural Orchestrator")
        self._orchestrator_engine = "procedural"
        return ProceduralOrchestrator(
            self.documents,
            self.kg,
            vector_retriever=self.faiss_retriever,
            skip_vector_build=True,
            embed_fn=embed_fn,
        )

    def _make_embed_fn(self):
        """Return a callable ``str -> np.ndarray`` reusing the FAISS embedder.

        Used by the semantic cache to embed queries without loading a second
        sentence-transformers model into memory.
        """
        ep = self.embedding_pipeline

        def _embed(text: str):
            model = ep.get_model()
            vec = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
            return vec[0]

        return _embed

    def _stats(self) -> Dict[str, Any]:
        kg_stats = self.kg.get_stats() if self.kg is not None else {}
        return {
            "documents": len(self.documents),
            "vectors": self.embedding_pipeline.index.ntotal if self.embedding_pipeline.index else 0,
            "embedding_dim": self.embedding_pipeline.dimension,
            "kg_nodes": kg_stats.get("total_nodes", 0),
            "kg_edges": kg_stats.get("total_edges", 0),
            "kg_entity_types": kg_stats.get("entity_types", {}),
            "kg_relation_types": kg_stats.get("relation_types", {}),
            "llm_enabled": self._llm_enabled,
            "orchestrator_engine": self._orchestrator_engine,
            "input_dirs": [str(d) for d in self.input_dirs],
        }

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats()

    @property
    def llm_enabled(self) -> bool:
        return self._llm_enabled

    # ─── Query ───────────────────────────────────────────────────────────

    def quick_search(self, query: str, top_k: int = 5,
                     use_context_window: bool = True) -> PipelineResult:
        self._require_ready()
        start = time.time()

        clarification = self.clarifier.analyze(query)
        correction = self.query_corrector.correct(query)

        base = correction.expanded if correction.corrections_applied else correction.corrected
        search_query = clarification.enriched_query if clarification.entities else base

        if use_context_window:
            raw = self.embedding_pipeline.search_with_context(search_query, top_k=top_k, context_window=1)
        else:
            raw = self.embedding_pipeline.search(search_query, top_k=top_k)

        evidence = [{
            "chunk_id": self.documents[r.chunk_id]["chunk_id"] if 0 <= r.chunk_id < len(self.documents) else "?",
            "text": r.text,
            "metadata": r.metadata,
            "vector_score": float(r.score),
        } for r in raw]

        graph_ctx = self.kg.get_subgraph_for_query(query) if self.kg else {"nodes": [], "edges": []}

        return PipelineResult(
            mode=PipelineMode.QUICK.value,
            query=query,
            answer="",
            evidence=evidence,
            clarification=clarification,
            correction=correction,
            graph_context=graph_ctx,
            critic=None,
            metrics={"total_latency_ms": (time.time() - start) * 1000},
            formatted_output="",
        )

    def diagnostic(self, query: str, thread_id: Optional[str] = None) -> PipelineResult:
        self._require_ready()
        if not self._llm_enabled or self.orchestrator is None:
            raise RuntimeError(
                "Diagnostic mode requires OPENAI_API_KEY. Use quick_search() instead."
            )

        clarification = self.clarifier.analyze(query)
        correction = self.query_corrector.correct(query)

        # Use the corrected & enriched query for retrieval.
        enriched = clarification.enriched_query if clarification.entities else correction.corrected

        # LangGraph orchestrator supports thread_id (HITL); the procedural one
        # ignores the kwarg to keep the call shape identical.
        if self._orchestrator_engine == "langgraph":
            result = self.orchestrator.process_query(enriched, thread_id=thread_id)
        else:
            result = self.orchestrator.process_query(enriched)

        return self._wrap_diagnostic_result(query, result, clarification, correction)

    def diagnostic_stream(
        self,
        query: str,
        thread_id: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Stream per-node updates from the diagnostic pipeline.

        Requires the LangGraph orchestrator (``USE_LANGGRAPH=true``). The
        procedural orchestrator does not expose intermediate states.

        Each yielded item matches the contract emitted by
        :meth:`LangGraphOrchestrator.stream_query`:

        * ``{"event": "node_update", "node": "...", "update": {...}}``
        * ``{"event": "complete", "response": {...}}``     — terminal, success
        * ``{"event": "interrupted", "response": {...}}``  — HITL pause
        """
        self._require_ready()
        if not self._llm_enabled or self.orchestrator is None:
            raise RuntimeError("Diagnostic streaming requires OPENAI_API_KEY.")
        if self._orchestrator_engine != "langgraph":
            raise RuntimeError(
                "Diagnostic streaming requires the LangGraph orchestrator. "
                "Set USE_LANGGRAPH=true."
            )
        if not hasattr(self.orchestrator, "stream_query"):
            raise RuntimeError("Active orchestrator does not support streaming.")

        clarification = self.clarifier.analyze(query)
        correction = self.query_corrector.correct(query)
        enriched = clarification.enriched_query if clarification.entities else correction.corrected

        # Emit the pre-LangGraph stages first so the UI has progress before
        # the StateGraph spins up.
        yield {
            "event": "node_update",
            "node": "clarify",
            "update": {
                "intent": clarification.intent.value,
                "entities": [(e.entity_type, e.normalized) for e in clarification.entities],
                "is_complete": clarification.is_complete,
            },
        }
        yield {
            "event": "node_update",
            "node": "correct",
            "update": {
                "corrected": correction.corrected,
                "expanded": correction.expanded,
                "corrections_applied": correction.corrections_applied,
            },
        }

        yield from self.orchestrator.stream_query(enriched, thread_id=thread_id)

    def resume_diagnostic(
        self,
        thread_id: str,
        decision: Dict[str, Any],
    ) -> PipelineResult:
        """Resume a paused diagnostic graph (after a human approve/reject).

        Only supported when the LangGraph orchestrator is active and HITL is
        enabled. Raises a clear ``RuntimeError`` otherwise.
        """
        self._require_ready()
        if self._orchestrator_engine != "langgraph" or self.orchestrator is None:
            raise RuntimeError(
                "Resume requires the LangGraph orchestrator. "
                "Set USE_LANGGRAPH=true and USE_HITL=true."
            )
        if not hasattr(self.orchestrator, "resume"):
            raise RuntimeError("Active orchestrator does not support resume.")
        result = self.orchestrator.resume(thread_id, decision)
        return self._wrap_diagnostic_result(result.get("query", {}).get("original", ""), result)

    def pending_approvals(self) -> List[Dict[str, Any]]:
        """Snapshot of all paused approvals (LangGraph engine only)."""
        if self._orchestrator_engine != "langgraph" or self.orchestrator is None:
            return []
        if not hasattr(self.orchestrator, "pending_approvals"):
            return []
        return self.orchestrator.pending_approvals()

    def get_pending_approval(self, thread_id: str) -> Optional[Dict[str, Any]]:
        if self._orchestrator_engine != "langgraph" or self.orchestrator is None:
            return None
        if not hasattr(self.orchestrator, "get_pending"):
            return None
        return self.orchestrator.get_pending(thread_id)

    def annotate_pending(
        self,
        thread_id: str,
        *,
        maker_user_id: Optional[str] = None,
        required_roles: Optional[List[str]] = None,
    ) -> bool:
        """Attach RBAC metadata to an existing pending approval.

        Called by the API layer *after* ``diagnostic()`` produced a pending
        thread so the policy and the maker identity are stored alongside the
        proposal. Returns ``True`` if the pending entry was found.
        """
        if self._orchestrator_engine != "langgraph" or self.orchestrator is None:
            return False
        pending = getattr(self.orchestrator, "_pending", None)
        if pending is None or thread_id not in pending:
            return False
        entry = pending[thread_id]
        if maker_user_id is not None:
            entry["maker_user_id"] = maker_user_id
        if required_roles is not None:
            entry["required_roles"] = list(required_roles)
        return True

    def _wrap_diagnostic_result(
        self,
        query: str,
        result: Dict[str, Any],
        clarification: Any = None,
        correction: Any = None,
    ) -> PipelineResult:
        status = result.get("pipeline_status", "complete")
        return PipelineResult(
            mode=PipelineMode.DIAGNOSTIC.value,
            query=query,
            answer=result.get("answer", ""),
            evidence=result.get("evidence", []),
            clarification=clarification,
            correction=correction,
            graph_context=result.get("graph_context"),
            cause_ranking=result.get("cause_ranking"),
            procedure=result.get("procedure"),
            critic=result.get("critic"),
            metrics=result.get("metrics", {}),
            formatted_output="",
            risk=result.get("risk"),
            purchase_request=result.get("purchase_request"),
            requires_approval=bool(result.get("awaiting_approval", False)),
            approval_thread_id=result.get("approval_thread_id"),
            rejected=(status == "rejected"),
            human_decision=result.get("human_decision"),
            interrupt_payload=result.get("interrupt_payload"),
            pipeline_status=status,
            guardrails=result.get("guardrails"),
            tool_results=result.get("tool_results", []) or [],
            pending_tool_calls=result.get("pending_tool_calls", []) or [],
        )

    def classical(self, query: str) -> PipelineResult:
        self._require_ready()
        if not self._llm_enabled or self.classical_rag is None:
            raise RuntimeError("Classical RAG mode requires OPENAI_API_KEY.")
        result = self.classical_rag.query(query)
        return PipelineResult(
            mode=PipelineMode.CLASSICAL.value,
            query=query,
            answer=result.get("answer", ""),
            evidence=result.get("evidence", []),
            graph_context=result.get("graph_context"),
            critic=result.get("critic"),
            metrics=result.get("metrics", {}),
        )

    def direct(self, query: str) -> PipelineResult:
        if not self._llm_enabled:
            raise RuntimeError("Direct LLM mode requires OPENAI_API_KEY.")
        from comparison.direct_llm import direct_llm_query
        result = direct_llm_query(query)
        return PipelineResult(
            mode=PipelineMode.DIRECT.value,
            query=query,
            answer=result.get("answer", ""),
            evidence=[],
            graph_context=result.get("graph_context"),
            critic=result.get("critic"),
            metrics=result.get("metrics", {}),
        )

    def compare(self, query: str) -> Dict[str, PipelineResult]:
        """Run direct / classical / diagnostic in parallel for benchmarking."""
        if not self._llm_enabled:
            raise RuntimeError("Comparison mode requires OPENAI_API_KEY.")
        return {
            "direct_llm": self.direct(query),
            "classical_rag": self.classical(query),
            "hybrid_graphrag": self.diagnostic(query),
        }

    # ─── Tool-calling (ERP / MES) ────────────────────────────────────────

    def execute_approved_tool_call(
        self,
        tool_call: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a previously-paused write tool call after human approval.

        ``tool_call`` should be one of the dicts emitted in
        ``PipelineResult.pending_tool_calls`` (which mirrors
        :class:`core.tools.registry.ToolCall.to_dict`). Returns the result
        dict from the registry. Side-effect-free tool calls run inline in
        ``diagnostic()`` and do not require this path.
        """
        from core.tools import ToolCall, get_registry

        name = tool_call.get("name") or ""
        if not name:
            raise ValueError("tool_call missing 'name'")
        call = ToolCall(
            name=name,
            arguments=dict(tool_call.get("arguments") or {}),
            side_effect=str(tool_call.get("side_effect") or "write"),
            requires_approval=bool(tool_call.get("requires_approval", True)),
            risk_score=float(tool_call.get("risk_score", 0.0) or 0.0),
            rationale=str(tool_call.get("rationale") or "human-approved"),
            call_id=str(tool_call.get("call_id") or f"tc_{int(time.time() * 1000)}"),
        )
        result = get_registry().execute(call)
        return result.to_dict()

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("Pipeline not initialised — call build_or_load() first.")
