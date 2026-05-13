"""
ManufacturingPipeline — the single object that wires every layer together.

Stages
------
1. Ingest documents from `input_docs/` + `data/pdfs/` + `data/excel/`
   via doc_pipeline's PDF/TXT/Excel parsers.
2. Smart-chunk them with the doc_pipeline HybridChunker (semantic / recursive /
   sliding window per file type).
3. Embed + index with FAISS (doc_pipeline.embeddings).
4. Build the knowledge graph (core/knowledge_graph.py) from chunk metadata.
5. At query time:
   * Quick mode      → Clarifier + QueryCorrector + FAISS top-k.
   * Diagnostic mode → Clarifier + Hybrid (BM25 + FAISS + Graph + RRF)
                       + LLM answer + Critic loop (if OPENAI_API_KEY set).
   * Classical RAG   → FAISS only + LLM answer (baseline for comparison).
   * Direct LLM      → no retrieval (baseline for comparison).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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

    def to_dict(self) -> Dict:
        out = {
            "mode": self.mode,
            "query": self.query,
            "answer": self.answer,
            "evidence": self.evidence,
            "graph_context": self.graph_context,
            "cause_ranking": self.cause_ranking,
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
        }
        if self.clarification is not None:
            out["intent"] = self.clarification.intent.value
            out["entities"] = [(e.entity_type, e.normalized) for e in self.clarification.entities]
            out["is_complete"] = self.clarification.is_complete
        return out


class ManufacturingPipeline:
    """Single end-to-end pipeline for ingestion, indexing, and querying."""

    def __init__(
        self,
        input_dirs: Optional[Iterable[Path | str]] = None,
        index_dir: Optional[Path | str] = None,
        embedding_model: Optional[str] = None,
    ):
        from config import (
            INPUT_DOCS_DIR, PDF_DIR, EXCEL_DIR, VECTOR_STORE_DIR,
            EMBEDDING_MODEL, ensure_dirs,
        )
        ensure_dirs()

        self.input_dirs: List[Path] = [
            Path(p) for p in (input_dirs or [INPUT_DOCS_DIR, PDF_DIR, EXCEL_DIR])
            if Path(p).exists()
        ]
        self.index_dir = Path(index_dir or VECTOR_STORE_DIR)
        self.embedding_model = embedding_model or EMBEDDING_MODEL

        from doc_pipeline.document_ingestion import DocumentIngestion
        from doc_pipeline.embeddings import EmbeddingPipeline
        from doc_pipeline.chunking import HybridChunker
        from doc_pipeline.clarifier_agent import ClarifierAgent
        from doc_pipeline.query_correction import QueryCorrector

        self.ingestion = DocumentIngestion()
        self.embedding_pipeline = EmbeddingPipeline(
            model_name=self.embedding_model, index_dir=self.index_dir,
        )
        self.chunker = HybridChunker(embedding_model=self.embedding_pipeline.get_model())
        self.clarifier = ClarifierAgent()
        self.query_corrector = QueryCorrector()

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

        # Knowledge graph
        from core.knowledge_graph import KnowledgeGraph
        self.kg = KnowledgeGraph()
        if not rebuild and self.kg.load():
            logger.info("Loaded knowledge graph (%d nodes, %d edges)",
                        self.kg.graph.number_of_nodes(), self.kg.graph.number_of_edges())
        else:
            logger.info("Building knowledge graph...")
            self.kg.build_from_documents(self.documents)

        # FAISS-backed vector retriever shared by orchestrator & classical RAG
        from pipeline.faiss_retriever import FaissVectorRetriever
        self.faiss_retriever = FaissVectorRetriever(embedding_pipeline=self.embedding_pipeline)
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
        """Turn doc_pipeline Chunks into the core dict shape (with entity metadata)."""
        from pipeline.adapter import chunks_to_core_docs
        self.documents = chunks_to_core_docs(self.embedding_pipeline.chunks)
        logger.info("Materialized %d core documents", len(self.documents))

    def _build_orchestrator(self, use_langgraph: bool):
        """Construct the diagnostic orchestrator.

        Prefers the LangGraph engine when ``USE_LANGGRAPH=true`` and falls back
        to the procedural ``core.orchestrator.Orchestrator`` if langgraph is
        not installed or the env flag is off.
        """
        from core.orchestrator import Orchestrator as ProceduralOrchestrator

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
        )

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

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("Pipeline not initialised — call build_or_load() first.")
