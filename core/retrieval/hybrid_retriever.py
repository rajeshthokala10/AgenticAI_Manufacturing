from typing import List, Dict, Set, Optional

from config import RRF_K, TOP_K_RETRIEVAL, TOP_K_RERANK
from core.retrieval.bm25_retriever import BM25Retriever
from core.retrieval.graph_retriever import GraphRetriever
from core.knowledge_graph import KnowledgeGraph
from core.document_acl import filter_chunks

try:
    from core.retrieval.vector_retriever import VectorRetriever
except ImportError:
    # ChromaDB unavailable — fall back to FAISS-backed retriever.
    from pipeline.faiss_retriever import FaissVectorRetriever as VectorRetriever  # type: ignore


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

    def retrieve(self, query: str, use_graph_filter: bool = True, top_k: int = TOP_K_RERANK) -> List[Dict]:
        allow_list = None
        if use_graph_filter:
            allow_list = self.graph.get_allow_list(query)
            if not allow_list:
                allow_list = None

        bm25_results = self.bm25.retrieve(query, top_k=TOP_K_RETRIEVAL, allow_list=allow_list)
        vector_results = self.vector.retrieve(query, top_k=TOP_K_RETRIEVAL, allow_list=allow_list)
        graph_results = self.graph.retrieve_by_entity(query, top_k=TOP_K_RETRIEVAL)

        fused = self._reciprocal_rank_fusion(bm25_results, vector_results, graph_results)

        edge_priors = self.graph.get_edge_priors({r["chunk_id"] for r in fused})
        for result in fused:
            prior_boost = edge_priors.get(result["chunk_id"], 0.0)
            result["rrf_score"] += prior_boost * 0.1
            result["prior_boost"] = prior_boost

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        fused = fused[:top_k]

        for result in fused:
            if result["chunk_id"] in self.doc_map:
                doc = self.doc_map[result["chunk_id"]]
                result["text"] = doc["text"]
                result["metadata"] = doc.get("metadata", {})

        # Apply per-request document ACL last so any chunk the active user
        # is not entitled to read never reaches the LLM prompt or the UI.
        return filter_chunks(fused)

    def retrieve_vector_only(self, query: str, top_k: int = TOP_K_RERANK) -> List[Dict]:
        return self.vector.retrieve(query, top_k=top_k)

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        graph_results: List[Dict],
    ) -> List[Dict]:
        scores: Dict[str, Dict] = {}

        for rank, result in enumerate(bm25_results):
            cid = result["chunk_id"]
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
            scores[cid]["rrf_score"] += 1.0 / (RRF_K + rank + 1)
            scores[cid]["bm25_rank"] = rank + 1
            scores[cid]["bm25_score"] = result.get("bm25_score", 0.0)

        for rank, result in enumerate(vector_results):
            cid = result["chunk_id"]
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
            scores[cid]["rrf_score"] += 1.0 / (RRF_K + rank + 1)
            scores[cid]["vector_rank"] = rank + 1
            scores[cid]["vector_score"] = result.get("vector_score", 0.0)

        for rank, result in enumerate(graph_results):
            cid = result["chunk_id"]
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
            scores[cid]["rrf_score"] += 1.0 / (RRF_K + rank + 1)
            scores[cid]["graph_rank"] = rank + 1
            scores[cid]["graph_score"] = result.get("graph_score", 0.0)

        results = list(scores.values())
        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results
