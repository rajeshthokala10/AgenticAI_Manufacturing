"""
Embedding Pipeline — sentence-transformer embeddings with a Qdrant vector
index.

Public surface (kept stable so all existing callers continue to work):

    pipeline = EmbeddingPipeline()
    pipeline.build_index(chunks)
    pipeline.save()
    pipeline.load()
    pipeline.has_saved_index()
    pipeline.dimension                 # int
    pipeline.index.ntotal              # int (vector count)
    pipeline.chunks                    # list[Chunk]
    pipeline.search(query, top_k=...)  # list[SearchResult]
    pipeline.search_with_context(...)  # list[SearchResult]
    pipeline.get_model()               # SentenceTransformer

Why Qdrant: it gives us first-class metadata filtering (used by the
document-ACL layer), persistent on-disk storage, and a clean upgrade path
to a remote server (set ``QDRANT_URL=http://host:6333``) without changing
the application code.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from chunking import Chunk
    from config import (
        EMBEDDING_MODEL, VECTOR_STORE_DIR, INDEX_NAME,
        QDRANT_PATH, QDRANT_URL, QDRANT_COLLECTION,
    )
except ImportError:
    from .chunking import Chunk
    from .config import (
        EMBEDDING_MODEL, VECTOR_STORE_DIR, INDEX_NAME,
        QDRANT_PATH, QDRANT_URL, QDRANT_COLLECTION,
    )


logger = logging.getLogger("doc_pipeline.embeddings")


# Embedded (on-disk) Qdrant takes an exclusive file lock on its storage
# folder; the second client to open the same path raises ``RuntimeError:
# already accessed by another instance``. Within a single process we
# share one QdrantClient instance keyed by (url, path) so that multiple
# EmbeddingPipeline objects don't collide. Cross-process sharing is not
# possible with embedded mode — run a Qdrant server (set ``QDRANT_URL``)
# for the multi-service ``run.sh`` deployment.
_CLIENT_CACHE: dict = {}


def _resolve_client():
    """Return a shared QdrantClient for the configured URL/path."""
    from qdrant_client import QdrantClient

    url = (QDRANT_URL or "").strip()
    key = url if url else f"path:{QDRANT_PATH}"
    client = _CLIENT_CACHE.get(key)
    if client is not None:
        return client
    if url == ":memory:":
        client = QdrantClient(":memory:")
    elif url:
        client = QdrantClient(url=url)
    else:
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        try:
            client = QdrantClient(path=str(QDRANT_PATH))
        except RuntimeError as exc:
            # Lock conflict — surface a clear, actionable error message.
            raise RuntimeError(
                "Embedded Qdrant lock conflict on "
                f"{QDRANT_PATH}.\n\nEmbedded mode is single-process. To run "
                "FastAPI + Streamlit + Next.js together:\n"
                "  1. Stop everything:  ./stop.sh\n"
                "  2. ./run.sh will auto-boot a Qdrant docker container and "
                "set QDRANT_URL.\n"
                "Or, to point at an existing Qdrant server, set "
                "QDRANT_URL=http://host:6333 in .env."
            ) from exc
    _CLIENT_CACHE[key] = client
    return client


@dataclass
class SearchResult:
    text: str
    metadata: dict
    score: float
    chunk_id: int  # ordinal position in ``EmbeddingPipeline.chunks``


class _QdrantIndexShim:
    """Tiny adapter exposing ``.ntotal`` so legacy code reading
    ``pipeline.index.ntotal`` keeps working with the Qdrant backend.
    """

    __slots__ = ("_client", "_collection")

    def __init__(self, client, collection: str):
        self._client = client
        self._collection = collection

    @property
    def ntotal(self) -> int:
        try:
            info = self._client.get_collection(self._collection)
            return int(info.points_count or 0)
        except Exception:
            return 0

    def __bool__(self) -> bool:  # so `if self.index is None` style checks behave
        return True


# Process-wide cache so a second pipeline (e.g. the aviation domain on top
# of manufacturing) doesn't pay the model load twice. Keyed by model_name —
# the SentenceTransformer object is stateless across queries.
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _load_model(model_name: str) -> SentenceTransformer:
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    return model


class EmbeddingPipeline:
    """Encodes chunks with sentence-transformers and indexes them in Qdrant."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        index_dir: str | Path = VECTOR_STORE_DIR,
        collection_name: Optional[str] = None,
        domain: Optional[str] = None,
    ):
        self.model = _load_model(model_name)
        self.dimension: int = self.model.get_sentence_embedding_dimension()
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None  # kept around for callers
        self._client = None
        # Collection resolution order:
        #   1. explicit ``collection_name``
        #   2. ``config.qdrant_collection(domain)`` when ``domain`` is set
        #   3. legacy ``QDRANT_COLLECTION``
        if collection_name:
            self._collection: str = collection_name
        elif domain:
            from config import qdrant_collection
            self._collection = qdrant_collection(domain)
        else:
            self._collection = QDRANT_COLLECTION
        self.domain: Optional[str] = domain
        # Per-domain index/manifest filename. When a domain is set, we use
        # ``<domain>_index_*`` so the manufacturing and aviation chunk JSONs
        # never collide on disk.
        if domain:
            from config import index_name as _index_name
            self.index_name: str = _index_name(domain)
        else:
            self.index_name = INDEX_NAME
        self.index: Optional[_QdrantIndexShim] = None

    # ─── Qdrant client management ────────────────────────────────────────

    def _get_client(self):
        if self._client is not None:
            return self._client
        self._client = _resolve_client()
        return self._client

    def _ensure_collection(self, recreate: bool = False) -> None:
        from qdrant_client.http import models as qm

        client = self._get_client()
        exists = client.collection_exists(self._collection)
        if exists and recreate:
            client.delete_collection(self._collection)
            exists = False
        if not exists:
            client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(
                    size=self.dimension, distance=qm.Distance.COSINE,
                ),
            )
        self.index = _QdrantIndexShim(client, self._collection)

    # ─── Build ──────────────────────────────────────────────────────────

    def build_index(self, chunks: List[Chunk], use_ivf: bool = True) -> None:
        """Encode all chunks and upsert them into Qdrant.

        The ``use_ivf`` flag is preserved for back-compat and ignored — Qdrant
        chooses its own ANN index and tunes nprobe internally.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]

        logger.info("Encoding %d chunks (dim=%d)", len(texts), self.dimension)
        embs = self.model.encode(
            texts, show_progress_bar=True, batch_size=64,
            normalize_embeddings=True,
        )
        self.embeddings = np.asarray(embs, dtype=np.float32)

        self._ensure_collection(recreate=True)
        self._upsert_points(self.embeddings, self.chunks)

        logger.info(
            "Built Qdrant index: %d vectors, collection=%s",
            int(self.index.ntotal), self._collection,
        )

    def _upsert_points(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        from qdrant_client.http import models as qm

        client = self._get_client()
        batch_size = 256
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            batch_vecs = vectors[start:end]
            batch_chunks = chunks[start:end]
            points = [
                qm.PointStruct(
                    id=start + i,
                    vector=vec.tolist(),
                    payload={
                        "chunk_id": start + i,
                        "text": chunk.text,
                        "metadata": chunk.metadata or {},
                        "strategy": getattr(chunk, "strategy", "core"),
                    },
                )
                for i, (vec, chunk) in enumerate(zip(batch_vecs, batch_chunks))
            ]
            client.upsert(collection_name=self._collection, points=points)

    # ─── Search ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """Search the index for chunks most similar to the query.

        Applies the document-ACL filter from ``core.document_acl`` so a chunk
        classified above the active user's clearance never leaves this call.
        """
        if self.index is None or not self.chunks:
            return []

        client = self._get_client()
        query_vec = self.model.encode(
            [query], normalize_embeddings=True,
        )[0].astype(np.float32).tolist()

        # Over-fetch so we still return ``top_k`` after the ACL filter drops
        # entries the active user is not entitled to read.
        fetch_k = min(max(top_k * 4, top_k + 10), int(self.index.ntotal) or top_k)

        hits = client.query_points(
            collection_name=self._collection,
            query=query_vec,
            limit=fetch_k,
            with_payload=True,
        ).points

        from core.document_acl import active_classifications

        allowed = active_classifications()
        results: List[SearchResult] = []
        for hit in hits:
            score = float(hit.score)
            if score < score_threshold:
                continue
            payload = hit.payload or {}
            cid = int(payload.get("chunk_id", hit.id))
            if cid < 0 or cid >= len(self.chunks):
                continue
            metadata = payload.get("metadata") or {}
            cls = metadata.get("classification", "public")
            if cls not in allowed:
                continue
            results.append(SearchResult(
                text=payload.get("text", self.chunks[cid].text),
                metadata=metadata,
                score=score,
                chunk_id=cid,
            ))
            if len(results) >= top_k:
                break
        return results

    def search_with_context(
        self,
        query: str,
        top_k: int = 5,
        context_window: int = 1,
    ) -> List[SearchResult]:
        """Search and include neighboring chunks (same source) for broader context."""
        base_results = self.search(query, top_k=top_k)
        if context_window <= 0:
            return base_results

        expanded: List[SearchResult] = []
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
        return expanded[: top_k + context_window * 2]

    # ─── Persistence ────────────────────────────────────────────────────

    def save(self, name: Optional[str] = None) -> None:
        name = name or self.index_name
        """Persist chunk metadata to disk. Qdrant itself is already on-disk
        unless ``QDRANT_URL`` points elsewhere, so this only writes the
        chunk index + a manifest file used by ``has_saved_index``.
        """
        if self.index is None or not self.chunks:
            raise RuntimeError("Nothing to save — call build_index() first.")

        chunk_data = [{
            "text": c.text, "metadata": c.metadata,
            "chunk_id": c.chunk_id, "strategy": c.strategy,
        } for c in self.chunks]
        (self.index_dir / f"{name}_chunks.json").write_text(
            json.dumps(chunk_data, indent=2, default=str)
        )

        manifest = {
            "backend": "qdrant",
            "collection": self._collection,
            "qdrant_url": QDRANT_URL or "",
            "qdrant_path": str(QDRANT_PATH),
            "dimension": self.dimension,
            "n_chunks": len(self.chunks),
        }
        (self.index_dir / f"{name}.manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )

        logger.info(
            "Saved index manifest %s (collection=%s, n=%d)",
            name, self._collection, len(self.chunks),
        )

    def load(self, name: Optional[str] = None) -> None:
        """Reload chunks from disk and bind to an existing Qdrant collection."""
        name = name or self.index_name
        chunks_path = self.index_dir / f"{name}_chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(f"Missing chunk index: {chunks_path}")

        chunk_data = json.loads(chunks_path.read_text())
        self.chunks = [
            Chunk(
                text=c["text"], metadata=c["metadata"],
                chunk_id=c["chunk_id"], strategy=c["strategy"],
            )
            for c in chunk_data
        ]

        client = self._get_client()
        if not client.collection_exists(self._collection):
            raise RuntimeError(
                f"Qdrant collection '{self._collection}' is missing. "
                "Re-run with --rebuild to regenerate the index."
            )
        self.index = _QdrantIndexShim(client, self._collection)

        # We deliberately do not reload the numpy ``embeddings`` array — Qdrant
        # is the source of truth now. Callers that historically reached into
        # ``self.embeddings`` get None and should embed lazily via ``self.model``.
        self.embeddings = None
        logger.info(
            "Loaded index: %d vectors, %d chunks",
            int(self.index.ntotal), len(self.chunks),
        )

    def has_saved_index(self, name: Optional[str] = None) -> bool:
        """Return True iff there is a usable on-disk + Qdrant snapshot."""
        name = name or self.index_name
        chunks_path = self.index_dir / f"{name}_chunks.json"
        if not chunks_path.exists():
            return False
        try:
            client = self._get_client()
            if not client.collection_exists(self._collection):
                return False
            info = client.get_collection(self._collection)
            return int(info.points_count or 0) > 0
        except Exception:
            return False

    def get_model(self) -> SentenceTransformer:
        return self.model

    # ─── Optional helpers (used by the unified pipeline) ────────────────

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Encode arbitrary texts using the loaded model. Returns float32."""
        vecs = self.model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)
