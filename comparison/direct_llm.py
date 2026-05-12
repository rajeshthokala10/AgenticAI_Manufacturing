import time
from typing import Dict

from core.llm_client import call_llm_with_metrics


DIRECT_SYSTEM_PROMPT = """You are a manufacturing equipment assistant. Answer questions about
industrial equipment maintenance, troubleshooting, and operations based on your general knowledge.
Provide helpful and detailed answers."""


def direct_llm_query(raw_query: str) -> Dict:
    start = time.time()

    result = call_llm_with_metrics(
        system_prompt=DIRECT_SYSTEM_PROMPT,
        user_prompt=raw_query,
        temperature=0.7,
        max_tokens=1500,
    )

    total_time = (time.time() - start) * 1000

    return {
        "query": {"original": raw_query},
        "answer": result["response"],
        "evidence": [],
        "graph_context": {"nodes": [], "edges": []},
        "graph_filter": {"allow_list_size": 0, "total_docs": 0, "filter_ratio": "N/A"},
        "critic": {
            "final_verdict": {
                "verdict": "SKIP",
                "confidence": 0.0,
                "issues": ["No evidence grounding — direct LLM has no retrieval"],
                "suggestion": "Cannot verify claims without source documents",
            },
            "attempts": [],
            "total_attempts": 0,
        },
        "metrics": {
            "total_latency_ms": total_time,
            "query_formatting_ms": 0,
            "retrieval_ms": 0,
            "generation_ms": total_time,
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
            "cost_estimate_usd": result["cost_estimate"],
            "model": result["model"],
        },
        "pipeline": "direct_llm",
    }
