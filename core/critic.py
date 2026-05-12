import re
from typing import Dict, List, Optional, Tuple

from config import CRITIC_MODEL
from core.llm_client import call_llm


CRITIC_SYSTEM_PROMPT = """You are a strict quality critic for manufacturing diagnostic answers.
Your job is to evaluate whether an answer is GROUNDED in the provided evidence.

Evaluate the answer against these criteria:
1. FACTUAL GROUNDING: Every claim must be traceable to a source chunk. No hallucinated facts.
2. COMPLETENESS: The answer should address the core question.
3. TECHNICAL ACCURACY: Equipment IDs, alarm codes, procedures must match the source data.
4. ACTIONABILITY: For troubleshooting queries, the answer must provide concrete next steps.
5. SAFETY: Any safety-critical procedures must include proper warnings.

Respond in this exact format:
VERDICT: PASS or FAIL
CONFIDENCE: 0.0 to 1.0
ISSUES: (list any issues found, or "None")
SUGGESTION: (if FAIL, describe what needs to change)"""


def critic_evaluate(
    query: str,
    answer: str,
    evidence_chunks: List[Dict],
    attempt: int = 1,
) -> Dict:
    evidence_text = "\n\n".join([
        f"[Source: {c.get('metadata', {}).get('source', 'unknown')} | "
        f"Chunk: {c.get('chunk_id', 'N/A')}]\n{c.get('text', '')}"
        for c in evidence_chunks
    ])

    user_prompt = f"""QUERY: {query}

EVIDENCE CHUNKS:
{evidence_text}

ANSWER TO EVALUATE:
{answer}

Evaluate whether this answer is fully grounded in the evidence chunks provided.
Any claim not supported by the evidence is a hallucination."""

    response = call_llm(
        system_prompt=CRITIC_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=500,
        model=CRITIC_MODEL,
    )

    return _parse_critic_response(response, attempt)


def _parse_critic_response(response: str, attempt: int) -> Dict:
    verdict = "FAIL"
    confidence = 0.5
    issues = []
    suggestion = ""

    verdict_match = re.search(r'VERDICT:\s*(PASS|FAIL)', response, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).upper()

    conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
    if conf_match:
        confidence = min(float(conf_match.group(1)), 1.0)

    issues_match = re.search(r'ISSUES:\s*(.+?)(?=SUGGESTION:|$)', response, re.DOTALL)
    if issues_match:
        issues_text = issues_match.group(1).strip()
        if issues_text.lower() != "none":
            issues = [line.strip("- ").strip() for line in issues_text.split("\n") if line.strip()]

    suggestion_match = re.search(r'SUGGESTION:\s*(.+?)$', response, re.DOTALL)
    if suggestion_match:
        suggestion = suggestion_match.group(1).strip()

    return {
        "verdict": verdict,
        "confidence": confidence,
        "issues": issues,
        "suggestion": suggestion,
        "attempt": attempt,
        "raw_response": response,
    }
