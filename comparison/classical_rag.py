import time
from typing import Dict, List

from config import TOP_K_RERANK, CLASSICAL_RAG_MODEL
from core.retrieval.vector_retriever import VectorRetriever
from core.llm_client import call_llm_with_metrics


CLASSICAL_RAG_PROMPT = """You are a manufacturing equipment assistant. Use ONLY the provided context
to answer questions. If the context doesn't contain enough information, say so.

CONTEXT:
{context}

Answer the user's question based on the context above."""


class ClassicalRAG:
    def __init__(self, documents: List[Dict]):
        self.documents = documents
        self.vector = VectorRetriever()
        self._indexed = False

    def initialize(self) -> None:
        if not self._indexed:
            self.vector.build_index(self.documents)
            self._indexed = True

    def query(self, raw_query: str) -> Dict:
        start = time.time()

        ret_start = time.time()
        results = self.vector.retrieve(raw_query, top_k=TOP_K_RERANK)
        ret_time = (time.time() - ret_start) * 1000

        context = "\n\n".join([
            f"[{r.get('metadata', {}).get('source', 'unknown')}]: {r['text']}"
            for r in results
        ])

        gen_start = time.time()
        llm_result = call_llm_with_metrics(
            system_prompt=CLASSICAL_RAG_PROMPT.format(context=context),
            user_prompt=raw_query,
            temperature=0.3,
            model=CLASSICAL_RAG_MODEL,
        )
        gen_time = (time.time() - gen_start) * 1000

        total_time = (time.time() - start) * 1000

        return {
            "query": {"original": raw_query},
            "answer": llm_result["response"],
            "evidence": results,
            "graph_context": {"nodes": [], "edges": []},
            "graph_filter": {"allow_list_size": 0, "total_docs": len(self.documents), "filter_ratio": "N/A (no graph)"},
            "critic": {
                "final_verdict": {
                    "verdict": "SKIP",
                    "confidence": 0.0,
                    "issues": ["No critic loop in classical RAG — answers not verified"],
                    "suggestion": "Add hybrid retrieval and critic for higher accuracy",
                },
                "attempts": [],
                "total_attempts": 0,
            },
            "metrics": {
                "total_latency_ms": total_time,
                "query_formatting_ms": 0,
                "retrieval_ms": ret_time,
                "generation_ms": gen_time,
                "prompt_tokens": llm_result["prompt_tokens"],
                "completion_tokens": llm_result["completion_tokens"],
                "total_tokens": llm_result["total_tokens"],
                "cost_estimate_usd": llm_result["cost_estimate"],
                "model": llm_result["model"],
            },
            "pipeline": "classical_rag",
        }
