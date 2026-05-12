"""
BM25 retriever.

Tries to use the `rank_bm25` package; if it's not installed, falls back to an
in-tree pure-Python implementation of BM25 Okapi so the unified pipeline works
without an extra dependency.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set

try:
    from rank_bm25 import BM25Okapi as _RankBM25  # type: ignore
    _HAS_RANK_BM25 = True
except Exception:
    _RankBM25 = None
    _HAS_RANK_BM25 = False


class _SimpleBM25Okapi:
    """Minimal BM25 Okapi implementation used as a fallback for `rank_bm25`."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(d) for d in corpus]
        self.avgdl = (sum(self.doc_lens) / self.corpus_size) if self.corpus_size else 0.0

        self.doc_freqs: List[Counter] = [Counter(d) for d in corpus]

        df: Counter = Counter()
        for d in corpus:
            df.update(set(d))

        self.idf: Dict[str, float] = {}
        for term, n in df.items():
            self.idf[term] = math.log(((self.corpus_size - n + 0.5) / (n + 0.5)) + 1.0)

    def get_scores(self, query: List[str]) -> List[float]:
        if not self.corpus_size:
            return []
        scores = [0.0] * self.corpus_size
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lens[i] or 1
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[i] += idf * (tf * (self.k1 + 1.0)) / denom
        return scores


class BM25Retriever:
    def __init__(self):
        self._index = None
        self._documents: List[Dict] = []

    def build_index(self, documents: List[Dict]) -> None:
        self._documents = documents
        tokenized_corpus = [self._tokenize(doc["text"]) for doc in documents]
        BM25Cls = _RankBM25 if _HAS_RANK_BM25 else _SimpleBM25Okapi
        self._index = BM25Cls(tokenized_corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        allow_list: Optional[Set[str]] = None,
    ) -> List[Dict]:
        if not self._index:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        scored_docs = []
        for i, score in enumerate(scores):
            doc = self._documents[i]
            if allow_list and doc["chunk_id"] not in allow_list:
                continue
            scored_docs.append({
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "bm25_score": float(score),
            })

        scored_docs.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored_docs[:top_k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\b[\w\-]+\b', text.lower())
