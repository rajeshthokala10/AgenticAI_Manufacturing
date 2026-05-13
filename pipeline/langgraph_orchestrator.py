"""LangGraph-based orchestrator for the Hybrid GraphRAG pipeline.

Wraps the same retrieval / LLM / critic primitives used by
``core.orchestrator.Orchestrator`` inside an explicit ``langgraph.StateGraph``.
Activate by setting ``USE_LANGGRAPH=true`` in ``.env`` (or by instantiating
this class directly and passing it to ``ManufacturingPipeline``).

Graph topology
--------------

::

    START
      │
      ▼
    format ── format_query() (intent, entities, expansion)
      │
      ▼
    retrieve ── HybridRetriever + KG subgraph + allow-list
      │
      ▼
    rank_causes ── call_llm_with_metrics(CAUSE_RANK_MODEL)   [optional]
      │           (active only when USE_CAUSE_RANKING=true and intent is
      │            a troubleshooting / failure-analysis type)
      ▼
    generate ── call_llm_with_metrics(ANSWER_MODEL)
      │
      ▼
    critic ── critic_evaluate(CRITIC_MODEL)
      │
      ├── PASS or attempts == MAX_CRITIC_RETRIES ──► END
      │
      └── FAIL & attempts < MAX_CRITIC_RETRIES
              │
              ▼
            retry ── call_llm_with_metrics(RETRY_MODEL)
              │
              └──► critic   (loops)

The response dict mirrors ``Orchestrator.process_query`` exactly, except the
``pipeline`` field is set to ``"hybrid_graphrag_langgraph"`` so callers can
tell which engine produced the result.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - guarded at construction time
    raise ImportError(
        "langgraph is required for LangGraphOrchestrator. "
        "Install with: pip install langgraph langchain-core"
    ) from exc

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover - py<3.8
    from typing_extensions import TypedDict  # type: ignore

from config import (
    ANSWER_MODEL,
    CAUSE_RANK_TOP_K,
    MAX_CRITIC_RETRIES,
    RETRY_MODEL,
    TOP_K_RERANK,
    USE_CAUSE_RANKING,
)
from core.cause_ranker import format_for_prompt as format_causes_for_prompt
from core.cause_ranker import rank_causes
from core.critic import critic_evaluate
from core.knowledge_graph import KnowledgeGraph
from core.llm_client import call_llm_with_metrics
from core.query_formatter import format_query
from core.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger("pipeline.langgraph")


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


class GraphState(TypedDict, total=False):
    """Mutable state passed between StateGraph nodes."""

    raw_query: str
    formatted: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    graph_context: Dict[str, Any]
    allow_list: List[str]
    cause_ranking: Dict[str, Any]
    answer: str
    attempts: List[Dict[str, Any]]
    attempt_idx: int
    llm_metrics: Dict[str, Any]
    timings: Dict[str, float]


class LangGraphOrchestrator:
    """Drop-in replacement for ``core.orchestrator.Orchestrator``.

    Public surface (``initialize`` + ``process_query``) matches the legacy
    orchestrator exactly so ``ManufacturingPipeline`` can swap engines based
    on the ``USE_LANGGRAPH`` config flag without further changes.
    """

    def __init__(
        self,
        documents: List[Dict],
        knowledge_graph: KnowledgeGraph,
        vector_retriever: Optional[object] = None,
        skip_vector_build: bool = False,
    ):
        self.documents = documents
        self.knowledge_graph = knowledge_graph
        self.retriever = HybridRetriever(documents, knowledge_graph, vector_retriever)
        self._indexed = False
        self._skip_vector_build = skip_vector_build
        self.graph = self._build_graph()

    # ─── Public API ──────────────────────────────────────────────────────

    def initialize(self) -> None:
        if not self._indexed:
            self.retriever.build_indexes(skip_vector=self._skip_vector_build)
            self._indexed = True

    def process_query(self, raw_query: str) -> Dict[str, Any]:
        total_start = time.time()
        initial: GraphState = {
            "raw_query": raw_query,
            "attempts": [],
            "attempt_idx": 0,
            "llm_metrics": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_estimate": 0.0,
                "model": ANSWER_MODEL,
            },
            "timings": {},
        }

        final_state: GraphState = self.graph.invoke(initial)
        total_ms = (time.time() - total_start) * 1000
        timings = dict(final_state.get("timings", {}))
        timings["total_latency_ms"] = total_ms
        final_state["timings"] = timings
        return self._to_response(raw_query, final_state)

    # ─── Graph construction ──────────────────────────────────────────────

    def _build_graph(self):
        g: StateGraph = StateGraph(GraphState)

        g.add_node("format", self._format_node)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("rank_causes", self._rank_causes_node)
        g.add_node("generate", self._generate_node)
        g.add_node("critic", self._critic_node)
        g.add_node("retry", self._retry_node)

        g.add_edge(START, "format")
        g.add_edge("format", "retrieve")
        # Optional cause-ranking stage — short-circuited by the conditional
        # edge when USE_CAUSE_RANKING=false (it would otherwise idle-fire and
        # return immediately, but this keeps the graph picture cleaner).
        g.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {"rank_causes": "rank_causes", "generate": "generate"},
        )
        g.add_edge("rank_causes", "generate")
        g.add_edge("generate", "critic")
        g.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {"retry": "retry", "end": END},
        )
        g.add_edge("retry", "critic")

        return g.compile()

    # ─── Nodes ───────────────────────────────────────────────────────────

    def _format_node(self, state: GraphState) -> Dict[str, Any]:
        t0 = time.time()
        formatted = format_query(state["raw_query"])
        timings = dict(state.get("timings", {}))
        timings["query_formatting_ms"] = (time.time() - t0) * 1000
        return {"formatted": formatted, "timings": timings}

    def _retrieve_node(self, state: GraphState) -> Dict[str, Any]:
        t0 = time.time()
        search_query = state["formatted"]["structured_query"]
        chunks = self.retriever.retrieve(search_query, top_k=TOP_K_RERANK)
        graph_ctx = self.knowledge_graph.get_subgraph_for_query(state["raw_query"])
        allow_list = self.knowledge_graph.get_allow_list(state["raw_query"])
        timings = dict(state.get("timings", {}))
        timings["retrieval_ms"] = (time.time() - t0) * 1000
        return {
            "evidence": chunks,
            "graph_context": graph_ctx,
            "allow_list": allow_list,
            "timings": timings,
        }

    def _rank_causes_node(self, state: GraphState) -> Dict[str, Any]:
        t0 = time.time()
        formatted = state.get("formatted", {})
        result = rank_causes(
            query=state["raw_query"],
            intent=formatted.get("intent"),
            evidence_chunks=state.get("evidence", []),
            graph_context=state.get("graph_context"),
            top_k=CAUSE_RANK_TOP_K,
        )
        timings = dict(state.get("timings", {}))
        timings["cause_ranking_ms"] = (time.time() - t0) * 1000
        return {
            "cause_ranking": result,
            "llm_metrics": self._merge_metrics(state.get("llm_metrics", {}), result),
            "timings": timings,
        }

    def _generate_node(self, state: GraphState) -> Dict[str, Any]:
        t0 = time.time()
        formatted = state["formatted"]
        evidence_text = self._format_evidence(state["evidence"])

        cause_ranking = state.get("cause_ranking") or {}
        cause_block = format_causes_for_prompt(cause_ranking.get("candidates", []))

        user_prompt = (
            f"QUERY: {formatted['expanded']}\n\n"
            f"INTENT: {formatted['intent']}\n"
            f"ENTITIES: {formatted['entities']}\n\n"
            + (cause_block + "\n\n" if cause_block else "")
            + f"EVIDENCE CHUNKS:\n{evidence_text}\n\n"
            + "Provide a comprehensive, evidence-grounded answer. Cite sources for every claim."
        )
        result = call_llm_with_metrics(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=ANSWER_MODEL,
        )
        timings = dict(state.get("timings", {}))
        timings["generation_ms"] = (time.time() - t0) * 1000
        return {
            "answer": result["response"],
            "llm_metrics": self._merge_metrics(state.get("llm_metrics", {}), result),
            "timings": timings,
        }

    def _critic_node(self, state: GraphState) -> Dict[str, Any]:
        t0 = time.time()
        attempt_idx = int(state.get("attempt_idx", 0)) + 1
        critic_result = critic_evaluate(
            state["raw_query"],
            state["answer"],
            state["evidence"],
            attempt_idx,
        )
        critic_result["latency_ms"] = (time.time() - t0) * 1000
        attempts = list(state.get("attempts", [])) + [critic_result]
        return {"attempts": attempts, "attempt_idx": attempt_idx}

    def _retry_node(self, state: GraphState) -> Dict[str, Any]:
        t0 = time.time()
        last = state["attempts"][-1]
        feedback = f"Issues: {last['issues']}\nSuggestion: {last['suggestion']}"
        evidence_text = self._format_evidence(state["evidence"])
        formatted = state["formatted"]
        retry_prompt = (
            f"QUERY: {formatted['expanded']}\n\n"
            f"PREVIOUS ANSWER (REJECTED):\n{state['answer']}\n\n"
            f"CRITIC ISSUES:\n{feedback}\n\n"
            f"EVIDENCE CHUNKS:\n{evidence_text}\n\n"
            "Generate an improved answer that addresses the critic's concerns. Cite all sources."
        )
        result = call_llm_with_metrics(
            system_prompt=RETRY_SYSTEM_PROMPT.format(critic_feedback=feedback),
            user_prompt=retry_prompt,
            model=RETRY_MODEL,
        )
        timings = dict(state.get("timings", {}))
        timings["retry_ms"] = timings.get("retry_ms", 0.0) + (time.time() - t0) * 1000
        return {
            "answer": result["response"],
            "llm_metrics": self._merge_metrics(state.get("llm_metrics", {}), result),
            "timings": timings,
        }

    # ─── Routing ─────────────────────────────────────────────────────────

    def _route_after_retrieve(self, state: GraphState) -> str:
        """Branch into the cause-ranking node only when the feature is enabled
        and the query intent looks like troubleshooting.

        The ranker itself is also intent-gated, so this is just a graph-level
        short-circuit to keep the trace clean for non-troubleshooting queries.
        """
        if not USE_CAUSE_RANKING:
            return "generate"
        formatted = state.get("formatted", {}) or {}
        from core.cause_ranker import _intent_is_troubleshooting  # local import to avoid cycle

        if _intent_is_troubleshooting(formatted.get("intent")):
            return "rank_causes"
        return "generate"

    def _route_after_critic(self, state: GraphState) -> str:
        attempts = state.get("attempts", [])
        if not attempts:
            return "end"
        last = attempts[-1]
        if last.get("verdict") == "PASS":
            return "end"
        if int(state.get("attempt_idx", 0)) >= MAX_CRITIC_RETRIES:
            return "end"
        return "retry"

    # ─── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _merge_metrics(running: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "prompt_tokens": running.get("prompt_tokens", 0) + new.get("prompt_tokens", 0),
            "completion_tokens": running.get("completion_tokens", 0)
            + new.get("completion_tokens", 0),
            "total_tokens": running.get("total_tokens", 0) + new.get("total_tokens", 0),
            "cost_estimate": running.get("cost_estimate", 0.0) + new.get("cost_estimate", 0.0),
            "model": new.get("model") or running.get("model"),
        }

    @staticmethod
    def _format_evidence(chunks: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for i, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "unknown")
            doc_type = meta.get("doc_type", "unknown")
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id", "N/A")

            scores: List[str] = []
            if chunk.get("bm25_rank"):
                scores.append(f"BM25 rank: {chunk['bm25_rank']}")
            if chunk.get("vector_rank"):
                scores.append(f"Vector rank: {chunk['vector_rank']}")
            if chunk.get("graph_rank"):
                scores.append(f"Graph rank: {chunk['graph_rank']}")
            score_str = " | ".join(scores) if scores else f"RRF: {chunk.get('rrf_score', 0):.4f}"

            parts.append(
                f"--- Evidence {i + 1} [{source} | {doc_type} | {chunk_id}] ({score_str}) ---\n{text}"
            )
        return "\n\n".join(parts)

    def _to_response(self, raw_query: str, state: GraphState) -> Dict[str, Any]:
        attempts = state.get("attempts", [])
        final_verdict = attempts[-1] if attempts else None
        formatted = state.get("formatted", {})
        timings = state.get("timings", {})
        llm = state.get("llm_metrics", {})
        allow_list = state.get("allow_list", [])

        cause_ranking = state.get("cause_ranking")

        return {
            "query": {
                "original": raw_query,
                "formatted": formatted,
                "intent_classification": formatted.get("intent_metadata", {}),
            },
            "answer": state.get("answer", ""),
            "evidence": state.get("evidence", []),
            "graph_context": state.get("graph_context", {}),
            "graph_filter": {
                "allow_list_size": len(allow_list),
                "total_docs": len(self.documents),
                "filter_ratio": f"{len(allow_list)}/{len(self.documents)}"
                if allow_list
                else "no filter",
            },
            "cause_ranking": cause_ranking,
            "critic": {
                "final_verdict": final_verdict,
                "attempts": attempts,
                "total_attempts": len(attempts),
            },
            "metrics": {
                "total_latency_ms": timings.get("total_latency_ms", 0.0),
                "query_formatting_ms": timings.get("query_formatting_ms", 0.0),
                "retrieval_ms": timings.get("retrieval_ms", 0.0),
                "cause_ranking_ms": timings.get("cause_ranking_ms", 0.0),
                "generation_ms": timings.get("generation_ms", 0.0),
                "retry_ms": timings.get("retry_ms", 0.0),
                "prompt_tokens": llm.get("prompt_tokens", 0),
                "completion_tokens": llm.get("completion_tokens", 0),
                "total_tokens": llm.get("total_tokens", 0),
                "cost_estimate_usd": llm.get("cost_estimate", 0.0),
                "model": llm.get("model"),
            },
            "pipeline": "hybrid_graphrag_langgraph",
        }
