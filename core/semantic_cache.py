"""Semantic cache — embed-and-match results to avoid recomputing answers.

Manufacturing operators tend to ask the same questions ("MTBF for press 3",
"vibration alarm on P-203") again and again. The semantic cache:

1. **Embeds** the incoming query (re-using the FAISS embedding model so we
   don't pull a second model into memory).
2. Finds the nearest cached query by cosine similarity.
3. If similarity ≥ ``threshold`` and the cached entry is younger than
   ``ttl_seconds``, replays the cached answer/evidence/metrics and stamps
   ``metrics.cache_hit = True``.
4. Otherwise, lets the caller compute the answer normally and writes it
   back via :meth:`SemanticCache.put`.

The cache is opt-in via ``USE_SEMANTIC_CACHE=true``. The store is in-process
and bounded — perfect for a single-host deployment. A future iteration can
swap the backing store to Redis without changing the API.

Concurrency: a single lock guards reads + writes. Hot path is O(N) over the
cache size (defaults to 256 entries) which is negligible compared to LLM
latency.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("core.semantic_cache")


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


class SemanticCache:
    """In-memory semantic cache keyed by query embedding cosine similarity."""

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        *,
        threshold: float = 0.97,
        max_size: int = 256,
        ttl_seconds: int = 3600,
    ):
        self._embed_fn = embed_fn
        self.threshold = float(threshold)
        self.max_size = int(max_size)
        self.ttl_seconds = int(ttl_seconds)

        self._lock = threading.Lock()
        # OrderedDict for LRU eviction: key = (namespace, normalized_query_hash)
        self._store: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # ─── Public API ──────────────────────────────────────────────────────

    def get(self, query: str, *, namespace: str = "default") -> Optional[Dict[str, Any]]:
        """Return a cached payload for ``query`` or None on miss."""
        if not query:
            return None
        try:
            q_emb = self._embed_query(query)
        except Exception as exc:  # pragma: no cover - model missing
            logger.warning("SemanticCache embed failed (%s) — passthrough miss.", exc)
            self._misses += 1
            return None

        now = time.time()
        best: Tuple[float, Optional[str]] = (-1.0, None)
        with self._lock:
            stale_keys: List[str] = []
            for key, entry in self._store.items():
                if entry["namespace"] != namespace:
                    continue
                if now - entry["ts"] > self.ttl_seconds:
                    stale_keys.append(key)
                    continue
                sim = _cosine(q_emb, entry["embedding"])
                if sim > best[0]:
                    best = (sim, key)
            for key in stale_keys:
                self._store.pop(key, None)

            sim, key = best
            if key is None or sim < self.threshold:
                self._misses += 1
                return None

            # LRU touch + return a deep copy so callers cannot mutate the
            # cached payload (which would corrupt subsequent hits).
            entry = self._store.pop(key)
            self._store[key] = entry
            self._hits += 1
            payload = deepcopy(entry["payload"])
            payload.setdefault("metrics", {})
            payload["metrics"]["cache_hit"] = True
            payload["metrics"]["cache_similarity"] = round(float(sim), 4)
            payload["metrics"]["cache_age_s"] = round(now - entry["ts"], 2)
            return payload

    def put(
        self,
        query: str,
        payload: Dict[str, Any],
        *,
        namespace: str = "default",
    ) -> None:
        """Insert ``payload`` keyed by an embedding of ``query``.

        Skips no-op storage when the payload is missing an answer (don't
        cache failures or paused/rejected HITL runs).
        """
        if not query or not payload:
            return
        if payload.get("pipeline_status") in ("awaiting_approval", "rejected"):
            return
        if not payload.get("answer"):
            return
        try:
            q_emb = self._embed_query(query)
        except Exception as exc:  # pragma: no cover - model missing
            logger.warning("SemanticCache embed failed (%s) — skip put.", exc)
            return

        key = self._make_key(query, namespace)
        with self._lock:
            self._store.pop(key, None)
            self._store[key] = {
                "namespace": namespace,
                "query": query,
                "embedding": q_emb,
                "payload": deepcopy(payload),
                "ts": time.time(),
            }
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)
                self._evictions += 1

    def invalidate(self, namespace: Optional[str] = None) -> int:
        """Drop all entries (optionally for a single namespace). Returns count."""
        with self._lock:
            if namespace is None:
                n = len(self._store)
                self._store.clear()
                return n
            keys = [k for k, v in self._store.items() if v["namespace"] == namespace]
            for k in keys:
                self._store.pop(k, None)
            return len(keys)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                "threshold": self.threshold,
                "ttl_seconds": self.ttl_seconds,
            }

    # ─── Internals ───────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        emb = self._embed_fn(_normalize(query))
        arr = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr

    @staticmethod
    def _make_key(query: str, namespace: str) -> str:
        h = hashlib.sha1(_normalize(query).encode("utf-8")).hexdigest()[:16]
        return f"{namespace}:{h}"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # Embeddings are stored pre-normalized so cosine == dot product.
    return float(np.dot(a, b))


# ─── Module-level singleton accessor ────────────────────────────────────

_INSTANCE: Optional[SemanticCache] = None
_INSTANCE_LOCK = threading.Lock()


def get_cache(
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    *,
    threshold: Optional[float] = None,
    max_size: Optional[int] = None,
    ttl_seconds: Optional[int] = None,
) -> Optional[SemanticCache]:
    """Return the process-wide ``SemanticCache`` singleton.

    The first call must supply ``embed_fn`` (typically wired from the
    embedding pipeline at orchestrator construction). Subsequent calls
    can pass ``None`` and just retrieve the existing instance.
    """
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            return _INSTANCE
        if embed_fn is None:
            return None
        from config import (
            SEMANTIC_CACHE_MAX_SIZE,
            SEMANTIC_CACHE_THRESHOLD,
            SEMANTIC_CACHE_TTL_SECONDS,
        )
        _INSTANCE = SemanticCache(
            embed_fn=embed_fn,
            threshold=threshold if threshold is not None else SEMANTIC_CACHE_THRESHOLD,
            max_size=max_size if max_size is not None else SEMANTIC_CACHE_MAX_SIZE,
            ttl_seconds=ttl_seconds if ttl_seconds is not None else SEMANTIC_CACHE_TTL_SECONDS,
        )
        return _INSTANCE


def reset_cache() -> None:
    """Test hook — drop the singleton (used by the eval harness)."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = None
