from typing import Dict, List


def format_latency(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms/1000:.1f}s"


def format_cost(usd: float) -> str:
    if usd < 0.01:
        return f"${usd:.4f}"
    return f"${usd:.2f}"


def compute_accuracy_estimates(pipeline: str) -> Dict:
    estimates = {
        "direct_llm": {
            "answer_accuracy": 45,
            "hallucination_rate": 40,
            "grounded_claims": 0,
            "citation_support": False,
            "audit_compliance": False,
            "id_handling": "Poor",
            "self_correction": False,
        },
        "classical_rag": {
            "answer_accuracy": 60,
            "hallucination_rate": 25,
            "grounded_claims": 65,
            "citation_support": "Partial",
            "audit_compliance": "Limited",
            "id_handling": "Limited",
            "self_correction": False,
        },
        "hybrid_graphrag": {
            "answer_accuracy": 85,
            "hallucination_rate": 8,
            "grounded_claims": 88,
            "citation_support": True,
            "audit_compliance": True,
            "id_handling": "Excellent",
            "self_correction": True,
        },
    }
    return estimates.get(pipeline, estimates["direct_llm"])


def compute_cost_projection(
    queries_per_month: int = 100000,
    cost_per_wrong_answer: float = 300,
) -> Dict:
    pipelines = {
        "direct_llm": {"hallucination_rate": 0.40, "token_cost_per_query": 0.002},
        "classical_rag": {"hallucination_rate": 0.25, "token_cost_per_query": 0.003},
        "hybrid_graphrag": {"hallucination_rate": 0.08, "token_cost_per_query": 0.005},
    }

    projections = {}
    for name, config in pipelines.items():
        wrong_answers = queries_per_month * config["hallucination_rate"]
        wrong_answer_cost = wrong_answers * cost_per_wrong_answer
        token_cost = queries_per_month * config["token_cost_per_query"]
        total = wrong_answer_cost + token_cost

        projections[name] = {
            "queries_per_month": queries_per_month,
            "wrong_answers": int(wrong_answers),
            "wrong_answer_cost": wrong_answer_cost,
            "token_cost": token_cost,
            "total_monthly_cost": total,
            "cost_per_query": total / queries_per_month,
        }

    hybrid = projections["hybrid_graphrag"]
    classical = projections["classical_rag"]
    direct = projections["direct_llm"]

    projections["savings_vs_classical"] = classical["total_monthly_cost"] - hybrid["total_monthly_cost"]
    projections["savings_vs_direct"] = direct["total_monthly_cost"] - hybrid["total_monthly_cost"]
    projections["roi_vs_classical"] = (
        projections["savings_vs_classical"] / max(hybrid["token_cost"] - classical["token_cost"], 1)
    )

    return projections
