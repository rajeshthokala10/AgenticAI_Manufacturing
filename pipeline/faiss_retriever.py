"""
Qdrant-backed VectorRetriever for the unified pipeline.

Drop-in replacement for ``core.retrieval.vector_retriever.VectorRetriever``
(legacy ChromaDB) that reuses the single Qdrant collection built by
``doc_pipeline.embeddings.EmbeddingPipeline``. One vector store across the
whole system instead of duplicate embeddings.

The file is named ``faiss_retriever.py`` for historical reasons — the index
was originally FAISS-backed. The class name is now ``QdrantVectorRetriever``
and ``FaissVectorRetriever`` is kept as an alias so existing call-sites keep
importing the symbol they already had.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger("pipeline.qdrant_retriever")


class QdrantVectorRetriever:
    """API-compatible with ``core.retrieval.vector_retriever.VectorRetriever``.

    Uses ``doc_pipeline.embeddings.EmbeddingPipeline`` (which wraps Qdrant)
    as the underlying engine.
    """

    def __init__(self, embedding_pipeline=None):
        from doc_pipeline.embeddings import EmbeddingPipeline

        self._docs: List[Dict] = []
        self._id_to_doc: Dict[str, Dict] = {}
        self._embedding_pipeline = embedding_pipeline or EmbeddingPipeline()

    def build_index(self, documents: List[Dict]) -> None:
        """Build the Qdrant index from ``{chunk_id, text, metadata}`` dicts.

        Note: when the pipeline owner already has a prebuilt EmbeddingPipeline
        (because doc_pipeline indexed the source chunks), call ``attach()`` to
        reuse it without re-embedding.
        """
        from doc_pipeline.chunking import Chunk

        self._docs = documents
        self._id_to_doc = {d["chunk_id"]: d for d in documents}

        as_chunks = [
            Chunk(
                text=d["text"],
                metadata={**d.get("metadata", {}), "chunk_id": d["chunk_id"]},
                chunk_id=i,
                strategy=d.get("metadata", {}).get("doc_type", "core"),
            )
            for i, d in enumerate(documents)
        ]
        self._embedding_pipeline.build_index(as_chunks)

    def attach(self, embedding_pipeline, documents: List[Dict]) -> None:
        """Reuse an externally-built EmbeddingPipeline + document list."""
        self._embedding_pipeline = embedding_pipeline
        self._docs = documents
        self._id_to_doc = {d["chunk_id"]: d for d in documents}

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        allow_list: Optional[Set[str]] = None,
    ) -> List[Dict]:
        if self._embedding_pipeline.index is None or not self._docs:
            return []

        fetch_k = top_k * 3 if allow_list else top_k
        raw_results = self._embedding_pipeline.search(query, top_k=fetch_k)

        out: List[Dict] = []
        for r in raw_results:
            if r.chunk_id is None or r.chunk_id >= len(self._docs):
                continue
            doc = self._docs[r.chunk_id]
            if allow_list and doc["chunk_id"] not in allow_list:
                continue
            out.append({
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "vector_score": float(r.score),
            })
            if len(out) >= top_k:
                break

        out.sort(key=lambda x: x["vector_score"], reverse=True)
        return out[:top_k]


# Back-compat alias — older code imported ``FaissVectorRetriever``.
FaissVectorRetriever = QdrantVectorRetriever
