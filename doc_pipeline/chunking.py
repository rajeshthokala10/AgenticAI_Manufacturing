"""
Smart Chunking Strategies for manufacturing documents.

Provides three strategies:
1. Semantic Chunking — splits on topic boundaries using sentence similarity
2. Recursive Character Chunking — splits hierarchically by separators
3. Sliding Window Chunking — overlapping fixed-size windows

A HybridChunker auto-selects the best strategy per document type.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

try:
    from document_ingestion import Document, DocType
    from config import (
        SEMANTIC_SIMILARITY_THRESHOLD, SEMANTIC_MIN_CHUNK_SIZE, SEMANTIC_MAX_CHUNK_SIZE,
        RECURSIVE_CHUNK_SIZE, RECURSIVE_CHUNK_OVERLAP,
        SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STEP,
    )
except ImportError:
    from .document_ingestion import Document, DocType
    from .config import (
        SEMANTIC_SIMILARITY_THRESHOLD, SEMANTIC_MIN_CHUNK_SIZE, SEMANTIC_MAX_CHUNK_SIZE,
        RECURSIVE_CHUNK_SIZE, RECURSIVE_CHUNK_OVERLAP,
        SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STEP,
    )


logger = logging.getLogger("doc_pipeline.chunking")


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: int = 0
    strategy: str = ""


_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


class SemanticChunker:
    """
    Splits text at semantic boundaries by measuring cosine similarity between
    consecutive sentence embeddings. When similarity drops below a threshold,
    a chunk boundary is inserted.
    """

    def __init__(
        self,
        model=None,
        similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
        min_chunk_size: int = SEMANTIC_MIN_CHUNK_SIZE,
        max_chunk_size: int = SEMANTIC_MAX_CHUNK_SIZE,
    ):
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, doc: Document) -> list[Chunk]:
        sentences = _split_sentences(doc.content)
        if len(sentences) <= 2:
            return [Chunk(text=doc.content, metadata=doc.metadata.copy(),
                          chunk_id=0, strategy="semantic")]

        if self.model is None:
            return self._fallback_chunk(doc, sentences)

        embeddings = self.model.encode(sentences, show_progress_bar=False)
        boundaries = self._find_boundaries(np.asarray(embeddings))
        return self._build_chunks(doc, sentences, boundaries)

    def _find_boundaries(self, embeddings: np.ndarray) -> list[int]:
        boundaries = [0]
        for i in range(len(embeddings) - 1):
            a, b = embeddings[i], embeddings[i + 1]
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            if sim < self.similarity_threshold:
                boundaries.append(i + 1)
        return boundaries

    def _build_chunks(self, doc: Document, sentences: list[str],
                      boundaries: list[int]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(sentences)
            text = " ".join(sentences[start:end])

            if len(text) < self.min_chunk_size and chunks:
                chunks[-1].text += " " + text
                continue
            if len(text) > self.max_chunk_size:
                chunks.extend(self._split_large(text, doc, len(chunks)))
                continue

            meta = doc.metadata.copy()
            meta["sentence_range"] = f"{start}-{end - 1}"
            chunks.append(Chunk(text=text, metadata=meta,
                                chunk_id=len(chunks), strategy="semantic"))
        return chunks

    def _split_large(self, text: str, doc: Document, start_id: int) -> list[Chunk]:
        words = text.split()
        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0

        for word in words:
            current.append(word)
            current_len += len(word) + 1
            if current_len >= self.max_chunk_size:
                chunks.append(Chunk(
                    text=" ".join(current), metadata=doc.metadata.copy(),
                    chunk_id=start_id + len(chunks), strategy="semantic",
                ))
                current, current_len = [], 0

        if current:
            if chunks and current_len < self.min_chunk_size:
                chunks[-1].text += " " + " ".join(current)
            else:
                chunks.append(Chunk(
                    text=" ".join(current), metadata=doc.metadata.copy(),
                    chunk_id=start_id + len(chunks), strategy="semantic",
                ))
        return chunks

    def _fallback_chunk(self, doc: Document, sentences: list[str]) -> list[Chunk]:
        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        for sent in sentences:
            if current_len + len(sent) > self.max_chunk_size and current:
                chunks.append(Chunk(
                    text=" ".join(current), metadata=doc.metadata.copy(),
                    chunk_id=len(chunks), strategy="semantic-fallback",
                ))
                current, current_len = [], 0
            current.append(sent)
            current_len += len(sent) + 1

        if current:
            chunks.append(Chunk(
                text=" ".join(current), metadata=doc.metadata.copy(),
                chunk_id=len(chunks), strategy="semantic-fallback",
            ))
        return chunks


class RecursiveChunker:
    """
    Splits text hierarchically using separators in priority order:
    section headers → paragraphs → sentences → words.
    Ideal for structured documents like SOPs and manuals.
    """

    SEPARATORS = (
        r'\n===.*?===\n',
        r'\n\d+\.\d+\s',
        r'\n\n',
        r'\n',
        r'(?<=[.!?])\s+',
        r'\s+',
    )

    def __init__(
        self,
        chunk_size: int = RECURSIVE_CHUNK_SIZE,
        chunk_overlap: int = RECURSIVE_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        pieces = self._recursive_split(doc.content, 0)
        return self._merge_pieces(pieces, doc)

    def _recursive_split(self, text: str, level: int) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if level >= len(self.SEPARATORS):
            return [text[:self.chunk_size]]

        splits = [s for s in re.split(self.SEPARATORS[level], text) if s.strip()]
        if len(splits) <= 1:
            return self._recursive_split(text, level + 1)

        result: list[str] = []
        for split in splits:
            if len(split) <= self.chunk_size:
                result.append(split)
            else:
                result.extend(self._recursive_split(split, level + 1))
        return result

    def _merge_pieces(self, pieces: list[str], doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        current_text = ""

        for piece in pieces:
            if len(current_text) + len(piece) <= self.chunk_size:
                current_text += ("\n" if current_text else "") + piece
                continue

            if current_text.strip():
                chunks.append(Chunk(
                    text=current_text.strip(), metadata=doc.metadata.copy(),
                    chunk_id=len(chunks), strategy="recursive",
                ))
            overlap = current_text[-self.chunk_overlap:] if len(current_text) > self.chunk_overlap else ""
            current_text = overlap + "\n" + piece

        if current_text.strip():
            chunks.append(Chunk(
                text=current_text.strip(), metadata=doc.metadata.copy(),
                chunk_id=len(chunks), strategy="recursive",
            ))
        return chunks


class SlidingWindowChunker:
    """
    Fixed-size sliding window with configurable overlap.
    Best for dense tabular or uniform content like Excel data.
    """

    def __init__(
        self,
        window_size: int = SLIDING_WINDOW_SIZE,
        step_size: int = SLIDING_WINDOW_STEP,
    ):
        self.window_size = window_size
        self.step_size = step_size

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content
        if len(text) <= self.window_size:
            return [Chunk(text=text, metadata=doc.metadata.copy(),
                          chunk_id=0, strategy="sliding_window")]

        chunks: list[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + self.window_size, len(text))
            window = text[start:end]

            if end < len(text):
                break_at = max(window.rfind("."), window.rfind("\n"))
                if break_at > self.window_size * 0.5:
                    window = text[start:start + break_at + 1]
                    end = start + break_at + 1

            meta = doc.metadata.copy()
            meta["window_start"] = start
            meta["window_end"] = end
            chunks.append(Chunk(
                text=window.strip(), metadata=meta,
                chunk_id=len(chunks), strategy="sliding_window",
            ))

            start += self.step_size
            if end >= len(text):
                break

        return chunks


class HybridChunker:
    """
    Auto-selects the best chunking strategy based on document type and content:
    - PDF text: Semantic chunking (topic-aware splits)
    - PDF with tables / TXT / SOP: Recursive chunking (respects section structure)
    - Excel data: Sliding window (handles tabular data)
    """

    def __init__(self, embedding_model=None):
        self.semantic = SemanticChunker(model=embedding_model)
        self.recursive = RecursiveChunker()
        self.sliding = SlidingWindowChunker()

    def chunk(self, doc: Document) -> list[Chunk]:
        dt = doc.doc_type
        if dt == DocType.EXCEL.value:
            chunks = self.sliding.chunk(doc)
        elif dt == DocType.TXT.value:
            chunks = self.recursive.chunk(doc)
        elif dt == DocType.PDF.value:
            chunks = self.recursive.chunk(doc) if doc.metadata.get("has_tables") \
                else self.semantic.chunk(doc)
        else:
            chunks = self.recursive.chunk(doc)

        from core.document_acl import classify_from_path

        classification = doc.metadata.get("classification") or classify_from_path(doc.source)
        for chunk in chunks:
            chunk.metadata["source"] = doc.source
            chunk.metadata["doc_type"] = doc.doc_type
            chunk.metadata["classification"] = classification

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for doc in documents:
            doc_chunks = self.chunk(doc)
            for i, chunk in enumerate(doc_chunks):
                chunk.chunk_id = len(all_chunks) + i
            all_chunks.extend(doc_chunks)

        strategy_counts: dict[str, int] = {}
        for c in all_chunks:
            strategy_counts[c.strategy] = strategy_counts.get(c.strategy, 0) + 1
        avg_len = sum(len(c.text) for c in all_chunks) / max(len(all_chunks), 1)

        logger.info(
            "Chunking complete: %d chunks from %d documents (avg=%.0f chars, strategies=%s)",
            len(all_chunks), len(documents), avg_len, strategy_counts,
        )
        return all_chunks
