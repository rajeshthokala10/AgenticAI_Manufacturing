import time
from typing import Optional

from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1500,
    model: Optional[str] = None,
) -> str:
    client = _get_client()
    model = model or LLM_MODEL

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def call_llm_with_metrics(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1500,
    model: Optional[str] = None,
) -> dict:
    client = _get_client()
    model = model or LLM_MODEL

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed_ms = (time.time() - start) * 1000

    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    cost_per_1k_input = 0.00015
    cost_per_1k_output = 0.0006
    if "gpt-4o" in model and "mini" not in model:
        cost_per_1k_input = 0.0025
        cost_per_1k_output = 0.01
    elif "gpt-4" in model:
        cost_per_1k_input = 0.03
        cost_per_1k_output = 0.06

    cost = (prompt_tokens * cost_per_1k_input + completion_tokens * cost_per_1k_output) / 1000

    return {
        "response": response.choices[0].message.content,
        "latency_ms": elapsed_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "model": model,
        "cost_estimate": cost,
    }
