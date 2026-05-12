import time
from typing import Dict, List, Optional

from config import MAX_CRITIC_RETRIES, TOP_K_RERANK
from core.query_formatter import format_query
from core.knowledge_graph import KnowledgeGraph
from core.retrieval.hybrid_retriever import HybridRetriever
from core.critic import critic_evaluate
from core.llm_client import call_llm, call_llm_with_metrics


ANSWER_SYSTEM_PROMPT = """You are a manufacturing diagnostic copilot. You provide evidence-grounded answers
to equipment troubleshooting, maintenance, and operational queries.

RULES:
1. Only use information from the provided evidence chunks. Do not hallucinate.
2. Cite your sources using [source_name, chunk_id] format.
3. For troubleshooting queries, provide: Diagnosis, Root Cause candidates, Recommended Procedure, and Safety notes.
4. Always reference specific equipment IDs, alarm codes, and part numbers when available.
5. If evidence is insufficient, state what is missing rather than guessing.
6. Prioritize safety-critical information."""


RETRY_SYSTEM_PROMPT = """You are a manufacturing diagnostic copilot. Your previous answer was rejected by the
quality critic for the following reasons. Generate an improved answer that addresses the issues.

CRITIC FEEDBACK:
{critic_feedback}

RULES:
1. Only use information from the provided evidence chunks. Do not hallucinate.
2. Address every issue raised by the critic.
3. Cite your sources using [source_name, chunk_id] format.
4. Be more conservative — if evidence is uncertain, say so explicitly."""


class Orchestrator:
    def __init__(self, documents: List[Dict], knowledge_graph: KnowledgeGraph):
        self.documents = documents
        self.knowledge_graph = knowledge_graph
        self.retriever = HybridRetriever(documents, knowledge_graph)
        self._indexed = False

    def initialize(self) -> None:
        if not self._indexed:
            self.retriever.build_indexes()
            self._indexed = True

    def process_query(self, raw_query: str) -> Dict:
        total_start = time.time()

        fmt_start = time.time()
        formatted = format_query(raw_query)
        fmt_time = (time.time() - fmt_start) * 1000

        search_query = formatted["structured_query"]

        ret_start = time.time()
        retrieved_chunks = self.retriever.retrieve(search_query, top_k=TOP_K_RERANK)
        ret_time = (time.time() - ret_start) * 1000

        graph_info = self.knowledge_graph.get_subgraph_for_query(raw_query)
        allow_list = self.knowledge_graph.get_allow_list(raw_query)

        evidence_text = self._format_evidence(retrieved_chunks)
        user_prompt = f"""QUERY: {formatted['expanded']}

INTENT: {formatted['intent']}
ENTITIES: {formatted['entities']}

EVIDENCE CHUNKS:
{evidence_text}

Provide a comprehensive, evidence-grounded answer. Cite sources for every claim."""

        gen_start = time.time()
        llm_result = call_llm_with_metrics(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        gen_time = (time.time() - gen_start) * 1000

        answer = llm_result["response"]
        critic_results = []
        final_verdict = None

        for attempt in range(1, MAX_CRITIC_RETRIES + 1):
            crit_start = time.time()
            critic_result = critic_evaluate(raw_query, answer, retrieved_chunks, attempt)
            crit_time = (time.time() - crit_start) * 1000
            critic_result["latency_ms"] = crit_time
            critic_results.append(critic_result)

            if critic_result["verdict"] == "PASS":
                final_verdict = critic_result
                break

            if attempt < MAX_CRITIC_RETRIES:
                feedback = f"Issues: {critic_result['issues']}\nSuggestion: {critic_result['suggestion']}"
                retry_prompt = f"""QUERY: {formatted['expanded']}

PREVIOUS ANSWER (REJECTED):
{answer}

CRITIC ISSUES:
{feedback}

EVIDENCE CHUNKS:
{evidence_text}

Generate an improved answer that addresses the critic's concerns. Cite all sources."""

                retry_result = call_llm_with_metrics(
                    system_prompt=RETRY_SYSTEM_PROMPT.format(critic_feedback=feedback),
                    user_prompt=retry_prompt,
                )
                answer = retry_result["response"]
                llm_result["prompt_tokens"] += retry_result["prompt_tokens"]
                llm_result["completion_tokens"] += retry_result["completion_tokens"]
                llm_result["total_tokens"] += retry_result["total_tokens"]
                llm_result["cost_estimate"] += retry_result["cost_estimate"]
            else:
                final_verdict = critic_result

        total_time = (time.time() - total_start) * 1000

        return {
            "query": {
                "original": raw_query,
                "formatted": formatted,
            },
            "answer": answer,
            "evidence": retrieved_chunks,
            "graph_context": graph_info,
            "graph_filter": {
                "allow_list_size": len(allow_list),
                "total_docs": len(self.documents),
                "filter_ratio": f"{len(allow_list)}/{len(self.documents)}" if allow_list else "no filter",
            },
            "critic": {
                "final_verdict": final_verdict,
                "attempts": critic_results,
                "total_attempts": len(critic_results),
            },
            "metrics": {
                "total_latency_ms": total_time,
                "query_formatting_ms": fmt_time,
                "retrieval_ms": ret_time,
                "generation_ms": gen_time,
                "prompt_tokens": llm_result["prompt_tokens"],
                "completion_tokens": llm_result["completion_tokens"],
                "total_tokens": llm_result["total_tokens"],
                "cost_estimate_usd": llm_result["cost_estimate"],
                "model": llm_result["model"],
            },
            "pipeline": "hybrid_graphrag",
        }

    def _format_evidence(self, chunks: List[Dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "unknown")
            doc_type = meta.get("doc_type", "unknown")
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id", "N/A")

            scores = []
            if chunk.get("bm25_rank"):
                scores.append(f"BM25 rank: {chunk['bm25_rank']}")
            if chunk.get("vector_rank"):
                scores.append(f"Vector rank: {chunk['vector_rank']}")
            if chunk.get("graph_rank"):
                scores.append(f"Graph rank: {chunk['graph_rank']}")
            score_str = " | ".join(scores) if scores else f"RRF: {chunk.get('rrf_score', 0):.4f}"

            parts.append(
                f"--- Evidence {i+1} [{source} | {doc_type} | {chunk_id}] ({score_str}) ---\n{text}"
            )
        return "\n\n".join(parts)
