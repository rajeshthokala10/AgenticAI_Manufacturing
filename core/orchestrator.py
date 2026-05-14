import logging
import time
from typing import Any, Callable, Dict, List, Optional

from config import (
    ANSWER_MODEL,
    CAUSE_RANK_TOP_K,
    GUARDRAILS_BLOCK_UNSAFE,
    GUARDRAILS_MIN_CITATIONS,
    GUARDRAILS_REQUIRE_CITATIONS,
    MAX_CRITIC_RETRIES,
    RETRY_MODEL,
    TOOL_PLANNER_MODEL,
    TOOL_PLANNER_USE_LLM,
    TOP_K_RERANK,
    USE_CAUSE_RANKING,
    USE_GUARDRAILS,
    USE_SEMANTIC_CACHE,
    USE_TOOLS,
)
from core.cause_ranker import (
    _intent_is_troubleshooting,
    format_for_prompt as format_causes_for_prompt,
    rank_causes,
)
from core.critic import critic_evaluate
from core.guardrails import evaluate as guardrails_evaluate, merge_into_critic
from core.knowledge_graph import KnowledgeGraph
from core.llm_client import call_llm_with_metrics
from core.query_formatter import format_query
from core.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger("core.orchestrator")


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
    def __init__(
        self,
        documents: List[Dict],
        knowledge_graph: KnowledgeGraph,
        vector_retriever: Optional[object] = None,
        skip_vector_build: bool = False,
        embed_fn: Optional[Callable[[str], Any]] = None,
    ):
        self.documents = documents
        self.knowledge_graph = knowledge_graph
        self.retriever = HybridRetriever(documents, knowledge_graph, vector_retriever)
        self._indexed = False
        self._skip_vector_build = skip_vector_build
        self._embed_fn = embed_fn

    def initialize(self) -> None:
        if not self._indexed:
            self.retriever.build_indexes(skip_vector=self._skip_vector_build)
            self._indexed = True

    def process_query(self, raw_query: str) -> Dict:
        total_start = time.time()

        # ─── Semantic cache lookup ────────────────────────────────────────
        cache = self._get_cache()
        if cache is not None:
            cached = cache.get(raw_query, namespace="diagnostic")
            if cached is not None:
                logger.info("semantic-cache HIT for query: %r", raw_query[:80])
                cached.setdefault("metrics", {})["total_latency_ms"] = (
                    (time.time() - total_start) * 1000
                )
                cached["pipeline"] = "hybrid_graphrag"
                return cached

        fmt_start = time.time()
        formatted = format_query(raw_query)
        fmt_time = (time.time() - fmt_start) * 1000

        search_query = formatted["structured_query"]

        ret_start = time.time()
        retrieved_chunks = self.retriever.retrieve(search_query, top_k=TOP_K_RERANK)
        ret_time = (time.time() - ret_start) * 1000

        graph_info = self.knowledge_graph.get_subgraph_for_query(raw_query)
        allow_list = self.knowledge_graph.get_allow_list(raw_query)

        # ─── Optional read-only tool calls (folded into the evidence) ─────
        tool_results: List[Dict[str, Any]] = []
        pending_writes: List[Dict[str, Any]] = []
        tool_block = ""
        if USE_TOOLS:
            tool_results, pending_writes = self._run_read_tools(
                raw_query, formatted.get("intent")
            )
            tool_block = self._format_tools_for_prompt(tool_results)

        evidence_text = self._format_evidence(retrieved_chunks)

        cause_ranking: Optional[Dict] = None
        cause_ranking_ms = 0.0
        cause_block = ""
        if USE_CAUSE_RANKING and _intent_is_troubleshooting(formatted.get("intent")):
            cr_start = time.time()
            cause_ranking = rank_causes(
                query=raw_query,
                intent=formatted.get("intent"),
                evidence_chunks=retrieved_chunks,
                graph_context=graph_info,
                top_k=CAUSE_RANK_TOP_K,
            )
            cause_ranking_ms = (time.time() - cr_start) * 1000
            cause_block = format_causes_for_prompt(cause_ranking.get("candidates", []))

        user_prompt = (
            f"QUERY: {formatted['expanded']}\n\n"
            f"INTENT: {formatted['intent']}\n"
            f"ENTITIES: {formatted['entities']}\n\n"
            + (cause_block + "\n\n" if cause_block else "")
            + (tool_block + "\n\n" if tool_block else "")
            + f"EVIDENCE CHUNKS:\n{evidence_text}\n\n"
            + "Provide a comprehensive, evidence-grounded answer. Cite sources for every claim."
        )

        gen_start = time.time()
        llm_result = call_llm_with_metrics(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=ANSWER_MODEL,
        )
        gen_time = (time.time() - gen_start) * 1000

        # Fold the cause-ranker's token spend into the running total so the
        # final metrics reflect the true cost of the query.
        if cause_ranking:
            llm_result["prompt_tokens"] += cause_ranking.get("prompt_tokens", 0)
            llm_result["completion_tokens"] += cause_ranking.get("completion_tokens", 0)
            llm_result["total_tokens"] += cause_ranking.get("total_tokens", 0)
            llm_result["cost_estimate"] += cause_ranking.get("cost_estimate", 0.0)

        answer = llm_result["response"]
        critic_results = []
        final_verdict = None
        last_guardrail_report: Optional[Dict[str, Any]] = None

        for attempt in range(1, MAX_CRITIC_RETRIES + 1):
            crit_start = time.time()
            critic_result = critic_evaluate(raw_query, answer, retrieved_chunks, attempt)
            crit_time = (time.time() - crit_start) * 1000
            critic_result["latency_ms"] = crit_time

            if USE_GUARDRAILS:
                report = guardrails_evaluate(
                    answer,
                    retrieved_chunks,
                    require_citations=GUARDRAILS_REQUIRE_CITATIONS,
                    min_citations=GUARDRAILS_MIN_CITATIONS,
                    block_on_unsafe=GUARDRAILS_BLOCK_UNSAFE,
                )
                critic_result = merge_into_critic(critic_result, report)
                last_guardrail_report = report.to_dict()
                if critic_result.get("guardrails_blocked"):
                    # Hard refusal — short-circuit without delivering the answer.
                    final_verdict = critic_result
                    critic_results.append(critic_result)
                    answer = (
                        "🚫 This answer was blocked by the safety guardrails. "
                        "A human supervisor must review the request before any "
                        "action is taken. Reason(s): "
                        + "; ".join(v["message"] for v in report.to_dict()["violations"])
                    )
                    break

            critic_results.append(critic_result)

            if critic_result["verdict"] == "PASS":
                final_verdict = critic_result
                break

            if attempt < MAX_CRITIC_RETRIES:
                feedback = (
                    f"Issues: {critic_result['issues']}\n"
                    f"Suggestion: {critic_result.get('suggestion', '')}"
                )
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
                    model=RETRY_MODEL,
                )
                answer = retry_result["response"]
                llm_result["prompt_tokens"] += retry_result["prompt_tokens"]
                llm_result["completion_tokens"] += retry_result["completion_tokens"]
                llm_result["total_tokens"] += retry_result["total_tokens"]
                llm_result["cost_estimate"] += retry_result["cost_estimate"]
            else:
                final_verdict = critic_result

        total_time = (time.time() - total_start) * 1000

        response: Dict[str, Any] = {
            "query": {
                "original": raw_query,
                "formatted": formatted,
                "intent_classification": formatted.get("intent_metadata", {}),
            },
            "answer": answer,
            "evidence": retrieved_chunks,
            "graph_context": graph_info,
            "graph_filter": {
                "allow_list_size": len(allow_list),
                "total_docs": len(self.documents),
                "filter_ratio": f"{len(allow_list)}/{len(self.documents)}" if allow_list else "no filter",
            },
            "cause_ranking": cause_ranking,
            "critic": {
                "final_verdict": final_verdict,
                "attempts": critic_results,
                "total_attempts": len(critic_results),
            },
            "guardrails": last_guardrail_report,
            "tool_results": tool_results,
            "pending_tool_calls": pending_writes,
            "metrics": {
                "total_latency_ms": total_time,
                "query_formatting_ms": fmt_time,
                "retrieval_ms": ret_time,
                "cause_ranking_ms": cause_ranking_ms,
                "generation_ms": gen_time,
                "prompt_tokens": llm_result["prompt_tokens"],
                "completion_tokens": llm_result["completion_tokens"],
                "total_tokens": llm_result["total_tokens"],
                "cost_estimate_usd": llm_result["cost_estimate"],
                "model": llm_result["model"],
                "cache_hit": False,
            },
            "pipeline": "hybrid_graphrag",
        }

        if cache is not None and final_verdict and final_verdict.get("verdict") == "PASS":
            cache.put(raw_query, response, namespace="diagnostic")

        return response

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _get_cache(self):
        if not USE_SEMANTIC_CACHE or self._embed_fn is None:
            return None
        from core.semantic_cache import get_cache
        return get_cache(embed_fn=self._embed_fn)

    def _run_read_tools(
        self,
        raw_query: str,
        intent: Optional[str],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Plan + execute read-only tools; defer write tools to HITL."""
        try:
            from core.tools import get_registry
            from core.tools.planner import plan_tool_calls, split_pending_calls
        except Exception as exc:  # pragma: no cover - optional dep
            logger.warning("Tool planner unavailable: %s", exc)
            return [], []

        calls = plan_tool_calls(
            raw_query,
            intent=intent,
            use_llm=TOOL_PLANNER_USE_LLM,
            model=TOOL_PLANNER_MODEL,
        )
        if not calls:
            return [], []

        registry = get_registry()
        buckets = split_pending_calls(calls)
        executed: List[Dict[str, Any]] = []
        for call in buckets["read"]:
            result = registry.execute(call)
            executed.append({
                "tool": call.name,
                "arguments": call.arguments,
                "status": result.status,
                "output": result.output,
                "elapsed_ms": result.elapsed_ms,
                "error": result.error,
            })
        pending = [c.to_dict() for c in buckets["write"]]
        return executed, pending

    @staticmethod
    def _format_tools_for_prompt(tool_results: List[Dict[str, Any]]) -> str:
        if not tool_results:
            return ""
        lines = ["TOOL RESULTS (live data from ERP/MES):"]
        for r in tool_results:
            lines.append(
                f"- {r['tool']}({r.get('arguments')}) → {r.get('output')}"
            )
        return "\n".join(lines)

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
            if chunk.get("rerank_score") is not None:
                scores.append(f"Rerank: {chunk['rerank_score']:.3f}")
            score_str = " | ".join(scores) if scores else f"RRF: {chunk.get('rrf_score', 0):.4f}"

            parts.append(
                f"--- Evidence {i+1} [{source} | {doc_type} | {chunk_id}] ({score_str}) ---\n{text}"
            )
        return "\n\n".join(parts)
