"""Hybrid retriever — BM25 + Vector + Graph fused via RRF.

Pipeline (each step is feature-flagged so the legacy path stays the default):

1. **Parallel retrieval** — BM25 / Vector / Graph queries run concurrently in
   a thread pool (``USE_PARALLEL_RETRIEVAL``). They are I/O- and CPU-bound in
   roughly equal measure, so a thread pool wins ~30% latency vs the original
   sequential implementation without touching the response shape.

2. **Reciprocal Rank Fusion + edge-prior boost** — same algorithm as before.

3. **Cross-encoder rerank** — optional second stage (``USE_RERANKER``) that
   jointly scores ``(query, chunk_text)`` pairs and re-sorts a wider pool
   (``RERANK_CANDIDATE_POOL``). Falls back to the unranked RRF order on any
   model load / inference failure.

4. **Document ACL** — per-request entitlement filter (unchanged).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Callable, Dict, List, Optional, Set

from config import (
    KG_RETRIEVAL_MIN_CONFIDENCE,
    PARALLEL_RETRIEVAL_TIMEOUT_S,
    RERANK_BLEND_WEIGHT,
    RERANK_CANDIDATE_POOL,
    RERANKER_MODEL,
    RRF_K,
    TOP_K_RERANK,
    TOP_K_RETRIEVAL,
    USE_PARALLEL_RETRIEVAL,
    USE_RERANKER,
)
from core.retrieval.bm25_retriever import BM25Retriever
from core.retrieval.graph_retriever import GraphRetriever
from core.knowledge_graph import KnowledgeGraph
from core.document_acl import filter_chunks

# Qdrant-backed retriever is the default. The legacy ChromaDB-backed
# ``core.retrieval.vector_retriever.VectorRetriever`` is still importable
# but only used by callers that explicitly opt in (e.g. the comparison
# benchmark with ``USE_LEGACY_CHROMA=true``).
from pipeline.faiss_retriever import QdrantVectorRetriever as VectorRetriever  # noqa: F401

logger = logging.getLogger("core.retrieval.hybrid")


class HybridRetriever:
    def __init__(
        self,
        documents: List[Dict],
        knowledge_graph: KnowledgeGraph,
        vector_retriever: Optional[object] = None,
    ):
        self.documents = documents
        self.doc_map = {doc["chunk_id"]: doc for doc in documents}
        self.knowledge_graph = knowledge_graph

        self.bm25 = BM25Retriever()
        # Caller can inject a FAISS-backed retriever for the unified pipeline.
        self.vector = vector_retriever if vector_retriever is not None else VectorRetriever()
        self.graph = GraphRetriever(knowledge_graph)

    def build_indexes(self, skip_vector: bool = False) -> None:
        print("  Building BM25 index...")
        self.bm25.build_index(self.documents)
        if not skip_vector:
            print("  Building vector index...")
            self.vector.build_index(self.documents)
        print("  Indexes built.")

    def retrieve(
        self,
        query: str,
        use_graph_filter: bool = True,
        top_k: int = TOP_K_RERANK,
    ) -> List[Dict]:
        allow_list = None
        if use_graph_filter:
            # Provenance-aware allow-list: when KG_RETRIEVAL_MIN_CONFIDENCE
            # is raised, only high-trust (structured / human-confirmed)
            # nodes contribute chunks. 0.0 preserves legacy behaviour.
            allow_list = self.graph.get_allow_list(
                query, min_confidence=KG_RETRIEVAL_MIN_CONFIDENCE,
            )
            if not allow_list:
                allow_list = None

        bm25_results, vector_results, graph_results = self._run_retrievers(query, allow_list)

        fused = self._reciprocal_rank_fusion(bm25_results, vector_results, graph_results)

        edge_priors = self.graph.get_edge_priors({r["chunk_id"] for r in fused})
        for result in fused:
            prior_boost = edge_priors.get(result["chunk_id"], 0.0)
            result["rrf_score"] += prior_boost * 0.1
            result["prior_boost"] = prior_boost

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

        # Attach text/metadata to the candidates the reranker will see.
        for result in fused:
            if result["chunk_id"] in self.doc_map:
                doc = self.doc_map[result["chunk_id"]]
                result["text"] = doc["text"]
                result["metadata"] = doc.get("metadata", {})

        # Optional second-stage cross-encoder rerank over a wider pool.
        reranked = self._maybe_rerank(query, fused, top_k=top_k)

        # Apply per-request document ACL last so any chunk the active user
        # is not entitled to read never reaches the LLM prompt or the UI.
        return filter_chunks(reranked)

    def retrieve_vector_only(self, query: str, top_k: int = TOP_K_RERANK) -> List[Dict]:
        return self.vector.retrieve(query, top_k=top_k)

    # ─── Retriever fan-out ───────────────────────────────────────────────

    def _run_retrievers(
        self,
        query: str,
        allow_list: Optional[Set[str]],
    ) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """Run BM25 / Vector / Graph retrievers, parallel when enabled."""

        def _bm25() -> List[Dict]:
            return self.bm25.retrieve(query, top_k=TOP_K_RETRIEVAL, allow_list=allow_list)

        def _vector() -> List[Dict]:
            return self.vector.retrieve(query, top_k=TOP_K_RETRIEVAL, allow_list=allow_list)

        def _graph() -> List[Dict]:
            return self.graph.retrieve_by_entity(query, top_k=TOP_K_RETRIEVAL)

        if not USE_PARALLEL_RETRIEVAL:
            return _bm25(), _vector(), _graph()

        # Three tasks, three threads — the cost of the pool is dominated by
        # FAISS+BM25 work, so we don't bother caching the executor.
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="hybrid-retr") as pool:
            future_bm25 = pool.submit(_safe_call, _bm25, "bm25")
            future_vec = pool.submit(_safe_call, _vector, "vector")
            future_graph = pool.submit(_safe_call, _graph, "graph")
            try:
                bm = future_bm25.result(timeout=PARALLEL_RETRIEVAL_TIMEOUT_S)
            except FuturesTimeout:
                logger.warning("bm25 retrieval timed out — empty result")
                bm = []
            try:
                vec = future_vec.result(timeout=PARALLEL_RETRIEVAL_TIMEOUT_S)
            except FuturesTimeout:
                logger.warning("vector retrieval timed out — empty result")
                vec = []
            try:
                gr = future_graph.result(timeout=PARALLEL_RETRIEVAL_TIMEOUT_S)
            except FuturesTimeout:
                logger.warning("graph retrieval timed out — empty result")
                gr = []
        return bm, vec, gr

    # ─── Reranker hook ───────────────────────────────────────────────────

    def _maybe_rerank(
        self,
        query: str,
        fused: List[Dict[str, Any]],
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not USE_RERANKER or not fused:
            return fused[:top_k]
        try:
            from core.retrieval.reranker import rerank
        except Exception as exc:  # pragma: no cover - import-time failure
            logger.warning("Reranker import failed (%s) — passthrough.", exc)
            return fused[:top_k]
        pool = fused[: max(top_k, min(len(fused), RERANK_CANDIDATE_POOL))]
        try:
            return rerank(
                query=query,
                candidates=pool,
                model_name=RERANKER_MODEL,
                top_k=top_k,
                blend_weight=RERANK_BLEND_WEIGHT,
            )
        except Exception as exc:  # pragma: no cover - inference failure
            logger.warning("Reranker failed (%s) — passthrough.", exc)
            return fused[:top_k]

    # ─── RRF fusion (unchanged algorithm, extracted for readability) ─────

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        graph_results: List[Dict],
    ) -> List[Dict]:
        scores: Dict[str, Dict] = {}

        def _slot(cid: str) -> Dict[str, Any]:
            if cid not in scores:
                scores[cid] = {
                    "chunk_id": cid,
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "vector_rank": None,
                    "graph_rank": None,
                    "bm25_score": 0.0,
                    "vector_score": 0.0,
                    "graph_score": 0.0,
                }
            return scores[cid]

        for rank, result in enumerate(bm25_results):
            slot = _slot(result["chunk_id"])
            slot["rrf_score"] += 1.0 / (RRF_K + rank + 1)
            slot["bm25_rank"] = rank + 1
            slot["bm25_score"] = result.get("bm25_score", 0.0)

        for rank, result in enumerate(vector_results):
            slot = _slot(result["chunk_id"])
            slot["rrf_score"] += 1.0 / (RRF_K + rank + 1)
            slot["vector_rank"] = rank + 1
            slot["vector_score"] = result.get("vector_score", 0.0)

        for rank, result in enumerate(graph_results):
            slot = _slot(result["chunk_id"])
            slot["rrf_score"] += 1.0 / (RRF_K + rank + 1)
            slot["graph_rank"] = rank + 1
            slot["graph_score"] = result.get("graph_score", 0.0)

        results = list(scores.values())
        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results


def _safe_call(fn: Callable[[], List[Dict]], label: str) -> List[Dict]:
    """Run ``fn`` inside the thread pool, swallowing exceptions."""
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - retriever failure
        logger.warning("%s retriever failed: %s", label, exc)
        return []
