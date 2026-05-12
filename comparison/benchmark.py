import time
from typing import Dict, List

from comparison.direct_llm import direct_llm_query
from comparison.classical_rag import ClassicalRAG
from core.orchestrator import Orchestrator


SAMPLE_QUERIES = [
    {
        "query": "Pump P-203 has high vibration alarm ALM-P001. What is the likely cause and fix procedure?",
        "category": "troubleshoot",
        "difficulty": "medium",
    },
    {
        "query": "Belt tracking deviation on conveyor CV-301. Alarm ALM-C002 triggered repeatedly.",
        "category": "troubleshoot",
        "difficulty": "medium",
    },
    {
        "query": "Hydraulic press HP-401 showing pressure loss. Cycle time increased by 40%.",
        "category": "troubleshoot",
        "difficulty": "hard",
    },
    {
        "query": "What is the preventive maintenance schedule for pump P-201 mechanical seal?",
        "category": "procedure",
        "difficulty": "easy",
    },
    {
        "query": "What spare parts are needed for bearing replacement on P-203?",
        "category": "inventory",
        "difficulty": "easy",
    },
    {
        "query": "PLC fault code FC-003 on conveyor CV-302. Communication loss with VFD.",
        "category": "alarm",
        "difficulty": "hard",
    },
    {
        "query": "What is the maximum operating pressure for HP-402 hydraulic system?",
        "category": "specification",
        "difficulty": "easy",
    },
    {
        "query": "Multiple alarms: ALM-P003 seal leak on P-202 and ALM-P001 vibration on P-203. Are they related?",
        "category": "troubleshoot",
        "difficulty": "hard",
    },
]


def run_benchmark(
    orchestrator: Orchestrator,
    classical_rag: ClassicalRAG,
    queries: List[Dict] = None,
) -> Dict:
    if queries is None:
        queries = SAMPLE_QUERIES

    results = []

    for q in queries:
        query_text = q["query"]

        direct_result = direct_llm_query(query_text)
        classical_result = classical_rag.query(query_text)
        hybrid_result = orchestrator.process_query(query_text)

        results.append({
            "query": query_text,
            "category": q.get("category", "unknown"),
            "difficulty": q.get("difficulty", "unknown"),
            "direct_llm": _extract_summary(direct_result),
            "classical_rag": _extract_summary(classical_result),
            "hybrid_graphrag": _extract_summary(hybrid_result),
        })

    summary = _compute_summary(results)
    return {"results": results, "summary": summary}


def run_single_comparison(
    query: str,
    orchestrator: Orchestrator,
    classical_rag: ClassicalRAG,
) -> Dict:
    direct_result = direct_llm_query(query)
    classical_result = classical_rag.query(query)
    hybrid_result = orchestrator.process_query(query)

    return {
        "query": query,
        "direct_llm": direct_result,
        "classical_rag": classical_result,
        "hybrid_graphrag": hybrid_result,
    }


def _extract_summary(result: Dict) -> Dict:
    metrics = result.get("metrics", {})
    critic = result.get("critic", {})
    verdict = critic.get("final_verdict", {})

    return {
        "answer_preview": result.get("answer", "")[:200] + "...",
        "latency_ms": metrics.get("total_latency_ms", 0),
        "total_tokens": metrics.get("total_tokens", 0),
        "cost_usd": metrics.get("cost_estimate_usd", 0),
        "evidence_count": len(result.get("evidence", [])),
        "critic_verdict": verdict.get("verdict", "N/A"),
        "critic_confidence": verdict.get("confidence", 0),
        "has_citations": "[" in result.get("answer", ""),
        "pipeline": result.get("pipeline", "unknown"),
    }


def _compute_summary(results: List[Dict]) -> Dict:
    pipelines = ["direct_llm", "classical_rag", "hybrid_graphrag"]
    summary = {}

    for pipeline in pipelines:
        latencies = [r[pipeline]["latency_ms"] for r in results]
        tokens = [r[pipeline]["total_tokens"] for r in results]
        costs = [r[pipeline]["cost_usd"] for r in results]
        citations = [r[pipeline]["has_citations"] for r in results]
        evidence = [r[pipeline]["evidence_count"] for r in results]

        critic_pass = sum(
            1 for r in results if r[pipeline]["critic_verdict"] == "PASS"
        )

        summary[pipeline] = {
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "avg_tokens": sum(tokens) / len(tokens) if tokens else 0,
            "total_cost_usd": sum(costs),
            "citation_rate": sum(citations) / len(citations) if citations else 0,
            "avg_evidence_chunks": sum(evidence) / len(evidence) if evidence else 0,
            "critic_pass_rate": critic_pass / len(results) if results else 0,
            "query_count": len(results),
        }

    return summary
