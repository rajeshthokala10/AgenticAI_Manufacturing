"""
Embedding Pipeline — Sentence-transformer embeddings with FAISS vector index.

Handles encoding text chunks, building/saving/loading a FAISS index,
and performing similarity search with metadata retrieval.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from chunking import Chunk
    from config import (
        EMBEDDING_MODEL, VECTOR_STORE_DIR, INDEX_NAME,
        FAISS_IVF_MIN_CHUNKS, FAISS_NPROBE_CAP,
    )
except ImportError:
    from .chunking import Chunk
    from .config import (
        EMBEDDING_MODEL, VECTOR_STORE_DIR, INDEX_NAME,
        FAISS_IVF_MIN_CHUNKS, FAISS_NPROBE_CAP,
    )


logger = logging.getLogger("doc_pipeline.embeddings")


@dataclass
class SearchResult:
    text: str
    metadata: dict
    score: float
    chunk_id: int


class EmbeddingPipeline:
    """Encodes chunks with sentence-transformers and indexes them in FAISS."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        index_dir: str | Path = VECTOR_STORE_DIR,
    ):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def build_index(self, chunks: list[Chunk], use_ivf: bool = True) -> None:
        """Encode all chunks and build a FAISS index."""
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        self.chunks = chunks
        texts = [c.text for c in chunks]

        logger.info("Encoding %d chunks (dim=%d)", len(texts), self.dimension)
        embs = self.model.encode(
            texts, show_progress_bar=True, batch_size=64,
            normalize_embeddings=True,
        )
        self.embeddings = np.asarray(embs, dtype=np.float32)

        if use_ivf and len(chunks) > FAISS_IVF_MIN_CHUNKS:
            n_clusters = min(int(np.sqrt(len(chunks))), 64)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, n_clusters, faiss.METRIC_INNER_PRODUCT,
            )
            self.index.train(self.embeddings)
            self.index.add(self.embeddings)
            self.index.nprobe = min(n_clusters, FAISS_NPROBE_CAP)
            logger.info("Built IVF index: %d clusters, nprobe=%d", n_clusters, self.index.nprobe)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(self.embeddings)
            logger.info("Built flat index with %d vectors", self.index.ntotal)

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Search the index for chunks most similar to the query."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() or load() first.")
        if not self.chunks:
            return []

        query_embedding = self.model.encode(
            [query], normalize_embeddings=True,
        ).astype(np.float32)

        # Over-fetch so we still return ``top_k`` results after the ACL
        # filter strips chunks the current user is not entitled to read.
        # The factor of 4 + offset is generous enough that even a heavy
        # confidential corpus rarely starves a public-only operator.
        k = min(max(top_k * 4, top_k + 10), self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        from core.document_acl import active_classifications

        allowed = active_classifications()
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks) or score < score_threshold:
                continue
            chunk = self.chunks[idx]
            cls = (chunk.metadata or {}).get("classification", "public")
            if cls not in allowed:
                continue
            results.append(SearchResult(
                text=chunk.text, metadata=chunk.metadata,
                score=float(score), chunk_id=chunk.chunk_id,
            ))
            if len(results) >= top_k:
                break
        return results

    def search_with_context(
        self,
        query: str,
        top_k: int = 5,
        context_window: int = 1,
    ) -> list[SearchResult]:
        """Search and include neighboring chunks (same source) for broader context."""
        base_results = self.search(query, top_k=top_k)
        if context_window <= 0:
            return base_results

        expanded: list[SearchResult] = []
        seen_ids: set[int] = set()

        for result in base_results:
            cid = result.chunk_id
            for offset in range(-context_window, context_window + 1):
                neighbor_id = cid + offset
                if not 0 <= neighbor_id < len(self.chunks):
                    continue
                if neighbor_id in seen_ids:
                    continue
                neighbor = self.chunks[neighbor_id]
                if neighbor.metadata.get("source") != result.metadata.get("source"):
                    continue
                seen_ids.add(neighbor_id)
                expanded.append(SearchResult(
                    text=neighbor.text, metadata=neighbor.metadata,
                    score=result.score if offset == 0 else result.score * 0.8,
                    chunk_id=neighbor_id,
                ))

        expanded.sort(key=lambda r: r.score, reverse=True)
        return expanded[:top_k + context_window * 2]

    def save(self, name: str = INDEX_NAME) -> None:
        """Persist index, embeddings, and chunk metadata to disk."""
        if self.index is None or self.embeddings is None:
            raise RuntimeError("Nothing to save — call build_index() first.")

        faiss.write_index(self.index, str(self.index_dir / f"{name}.faiss"))
        np.save(str(self.index_dir / f"{name}_embeddings.npy"), self.embeddings)

        chunk_data = [{
            "text": c.text, "metadata": c.metadata,
            "chunk_id": c.chunk_id, "strategy": c.strategy,
        } for c in self.chunks]
        (self.index_dir / f"{name}_chunks.json").write_text(
            json.dumps(chunk_data, indent=2, default=str)
        )

        logger.info("Saved index to %s/%s.*", self.index_dir, name)

    def load(self, name: str = INDEX_NAME) -> None:
        """Load a previously saved index from disk."""
        idx_path = self.index_dir / f"{name}.faiss"
        emb_path = self.index_dir / f"{name}_embeddings.npy"
        chunks_path = self.index_dir / f"{name}_chunks.json"

        for p in (idx_path, emb_path, chunks_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing index file: {p}")

        self.index = faiss.read_index(str(idx_path))
        self.embeddings = np.load(str(emb_path))
        chunk_data = json.loads(chunks_path.read_text())

        self.chunks = [
            Chunk(text=c["text"], metadata=c["metadata"],
                  chunk_id=c["chunk_id"], strategy=c["strategy"])
            for c in chunk_data
        ]
        logger.info("Loaded index: %d vectors, %d chunks", self.index.ntotal, len(self.chunks))

    def has_saved_index(self, name: str = INDEX_NAME) -> bool:
        return all((self.index_dir / f"{name}{suffix}").exists()
                   for suffix in (".faiss", "_embeddings.npy", "_chunks.json"))

    def get_model(self) -> SentenceTransformer:
        return self.model
