import time
from typing import Optional

from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL, OLLAMA_BASE_URL

_openai_client = None
_ollama_client = None

OLLAMA_MODELS = {"qwen2.5:3b", "qwen2.5:1.5b", "qwen2.5:7b", "llama3.2:3b", "phi3:3.8b", "mistral:7b"}

MODEL_PRICING = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4": (0.03, 0.06),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-3.5-turbo": (0.0005, 0.0015),
}


def _is_local_model(model: str) -> bool:
    return model in OLLAMA_MODELS or model.startswith(("qwen", "llama", "phi", "mistral:"))


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    return _ollama_client


def _get_client(model: str):
    if _is_local_model(model):
        return _get_ollama_client()
    return _get_openai_client()


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1500,
    model: Optional[str] = None,
) -> str:
    model = model or LLM_MODEL
    client = _get_client(model)

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
    model = model or LLM_MODEL
    client = _get_client(model)

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

    cost_per_1k_input, cost_per_1k_output = MODEL_PRICING.get(
        model, (0.0, 0.0) if _is_local_model(model) else (0.00015, 0.0006)
    )

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
