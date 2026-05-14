"""Cross-encoder reranker — second-stage reordering after RRF fusion.

Plugged in by ``HybridRetriever`` between the RRF/edge-prior step and the
final top-K cut. The reranker scores ``(query, chunk_text)`` pairs with a
cross-encoder (default ``BAAI/bge-reranker-base``) and re-sorts the candidate
pool. Cross-encoders typically lift answer quality 5–15 % on noisy
industrial corpora because they read query+chunk *jointly* rather than
matching independent embeddings.

The model is lazy-loaded (downloaded on first use) and the whole stage is
gated by ``USE_RERANKER`` so a deployment without `sentence-transformers`
extras still works. Failures degrade gracefully to the unrebanked RRF order.

Design choices:
* Pure inference, no training — drop-in upgrade.
* Caller-controlled top-K so we can rerank a larger candidate pool than we
  ultimately surface (`RERANK_CANDIDATE_POOL`).
* Returns the same dict shape as the input so downstream code is unchanged.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("core.retrieval.reranker")

_MODEL_LOCK = threading.Lock()
_MODEL: Optional[Any] = None
_MODEL_NAME_LOADED: Optional[str] = None


def _load_model(model_name: str):
    """Load (and cache) the cross-encoder model. Returns ``None`` on failure."""
    global _MODEL, _MODEL_NAME_LOADED
    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_NAME_LOADED == model_name:
            return _MODEL
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            logger.warning(
                "Reranker disabled — sentence-transformers CrossEncoder "
                "unavailable (%s). Install with: pip install sentence-transformers",
                exc,
            )
            return None
        try:
            logger.info("Loading cross-encoder reranker: %s", model_name)
            _MODEL = CrossEncoder(model_name)
            _MODEL_NAME_LOADED = model_name
            return _MODEL
        except Exception as exc:  # pragma: no cover - download / network
            logger.warning("Reranker load failed (%s) — passthrough.", exc)
            return None


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    model_name: str,
    top_k: int,
    text_key: str = "text",
    score_key: str = "rerank_score",
    blend_weight: float = 0.7,
    rrf_key: str = "rrf_score",
) -> List[Dict[str, Any]]:
    """Rerank ``candidates`` with a cross-encoder and return the top ``top_k``.

    The original RRF score is preserved and blended with the cross-encoder
    score to keep the lexical/graph signals from being completely
    overwritten by the dense reranker:

        final = blend_weight * normalized_rerank + (1 - blend_weight) * normalized_rrf

    If the cross-encoder cannot be loaded (no internet, missing extras, …),
    we return ``candidates`` untouched — capped to ``top_k`` — so the
    pipeline never breaks because of the reranker.
    """
    if not candidates:
        return []

    pool = candidates[: max(top_k, len(candidates))]

    # Hydrate missing text fields so the model sees something. The
    # HybridRetriever attaches `text` after fusion; reranker may be called
    # before that completes in some code paths, so we guard.
    pairs: List[List[str]] = []
    valid_idxs: List[int] = []
    for idx, item in enumerate(pool):
        text = item.get(text_key) or ""
        if not text:
            continue
        pairs.append([query, text[:2000]])  # truncate long chunks for speed
        valid_idxs.append(idx)

    if not pairs:
        return pool[:top_k]

    model = _load_model(model_name)
    if model is None:
        return pool[:top_k]

    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as exc:  # pragma: no cover - inference failure
        logger.warning("Reranker inference failed (%s) — passthrough.", exc)
        return pool[:top_k]

    # Normalise the cross-encoder scores into [0,1] for stable blending.
    s_min = float(min(scores)) if len(scores) else 0.0
    s_max = float(max(scores)) if len(scores) else 0.0
    s_range = (s_max - s_min) or 1.0

    rrf_values = [float(item.get(rrf_key, 0.0) or 0.0) for item in pool]
    r_min = min(rrf_values) if rrf_values else 0.0
    r_max = max(rrf_values) if rrf_values else 0.0
    r_range = (r_max - r_min) or 1.0

    for s, i in zip(scores, valid_idxs):
        normalized = (float(s) - s_min) / s_range
        rrf_norm = (float(pool[i].get(rrf_key, 0.0) or 0.0) - r_min) / r_range
        pool[i][score_key] = round(float(s), 6)
        pool[i]["rerank_normalized"] = round(normalized, 6)
        pool[i]["rerank_blended"] = round(
            blend_weight * normalized + (1.0 - blend_weight) * rrf_norm, 6
        )

    # Items without text get a neutral blended score so they don't outrank
    # genuinely relevant candidates.
    for idx, item in enumerate(pool):
        item.setdefault("rerank_blended", 0.0)
        item.setdefault(score_key, 0.0)

    pool.sort(key=lambda x: x.get("rerank_blended", 0.0), reverse=True)
    return pool[:top_k]


def warmup(model_name: str) -> bool:
    """Eagerly load the model. Returns True on success (used by health checks)."""
    return _load_model(model_name) is not None
