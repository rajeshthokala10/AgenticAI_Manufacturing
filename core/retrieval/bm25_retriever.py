import re
from typing import List, Dict, Optional, Set

from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self):
        self._index: Optional[BM25Okapi] = None
        self._documents: List[Dict] = []

    def build_index(self, documents: List[Dict]) -> None:
        self._documents = documents
        tokenized_corpus = [self._tokenize(doc["text"]) for doc in documents]
        self._index = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 10, allow_list: Optional[Set[str]] = None) -> List[Dict]:
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

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\b[\w\-]+\b', text)
        return tokens
