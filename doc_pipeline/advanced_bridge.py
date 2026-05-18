"""Bridge between the vendored ``advanced_parser`` and our existing
``doc_pipeline`` shape.

The advanced parser emits a flat list of ``ProcessedChunk`` per file —
text / table / form / chart / footnote chunks all carrying their own
metadata (page, section, content_hash, content_type). Our downstream
pipeline (chunker → embedder → adapter → KG) consumes
``list[Document]`` where each ``Document`` has a ``content`` string and
a ``metadata`` dict.

This module bridges the two **one-to-one** — each ``ProcessedChunk``
becomes one ``Document``. The advanced parser is already a chunker
(section-aware for text, row-bucketed for tables, one-per-form, etc.),
so re-chunking via our ``HybridChunker`` mostly leaves things intact
because the recursive strategy's ``chunk_size`` is comfortably larger
than typical advanced-parser chunks (~500–1000 chars each).

Public surface:

    parse_pdf_advanced(file_path) -> list[Document]

A drop-in replacement for the legacy ``PDFParser.parse(path)`` when
``USE_ADVANCED_PARSER=true``. See ``DocumentIngestion.ingest_file``
for the dispatch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from doc_pipeline.advanced_parser import (
    PipelineConfig,
    ProcessedChunk,
    ProductionRAGPipeline,
)
from doc_pipeline.document_ingestion import Document

logger = logging.getLogger("doc_pipeline.advanced_bridge")


def _build_pipeline_config() -> PipelineConfig:
    """Translate the relevant bits of ``config.settings`` into the
    advanced parser's ``PipelineConfig``. Conservative defaults: OCR off,
    VLM-charts off, deduplication off (the KG builder + Qdrant upsert
    handle dedup downstream)."""
    # Local import so this module stays importable even when config has
    # a transient error.
    from config import (
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        EMBEDDING_MODEL,
    )
    return PipelineConfig(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # We don't want the advanced parser computing embeddings — our
        # EmbeddingPipeline owns that stage. Skipping saves a pass over
        # the corpus.
        generate_embeddings=False,
        embedding_model=EMBEDDING_MODEL,
        # Headers / footers / watermarks / redactions: strip aggressively
        # (these are noise in the KG layer).
        strip_watermarks=True,
        detect_redactions=True,
        filter_toc_pages=True,
        # Tables: keep cross-page merging on — it's the headline win.
        merge_cross_page_tables=True,
        # OCR: enable but fall back gracefully when tesseract isn't on
        # the host. The advanced parser logs per-page warnings rather
        # than failing the whole document.
        ocr_engine="pytesseract",
        # Charts / VLM: leave off by default — the gpt-4o call adds cost
        # and the legacy KG layer doesn't currently consume chart
        # descriptions. Enable later via an env knob if we add a
        # ChartExtractor on the KG side.
        enable_vlm_charts=False,
        # Dedup is downstream's job (Qdrant upsert + KG provenance).
        enable_dedup=False,
        # Versioning is downstream's job too (we have stable_chunk_id).
        enable_versioning=False,
        # Per-page fault tolerance: never crash a domain rebuild on one
        # bad page.
        fail_on_error=False,
    )


_pipeline_singleton: ProductionRAGPipeline | None = None


def _get_pipeline() -> ProductionRAGPipeline:
    """Lazy-init the advanced parser. One instance per process is fine —
    it's stateless across files."""
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = ProductionRAGPipeline(_build_pipeline_config())
        logger.info("Initialised advanced PDF parser (vendored production pipeline)")
    return _pipeline_singleton


def parse_pdf_advanced(file_path: str | Path, password: str = "") -> List[Document]:
    """Run the advanced parser against a PDF and return our ``Document`` shape.

    Drop-in replacement for ``PDFParser.parse(path)`` when
    ``USE_ADVANCED_PARSER=true``.
    """
    path = Path(file_path)
    pipeline = _get_pipeline()

    try:
        processed: List[ProcessedChunk] = pipeline.process(str(path), password=password)
    except Exception as exc:
        # Bubble up — DocumentIngestion's caller already catches per-file
        # errors and logs a warning; we just need the original exception
        # type to surface so the user sees what actually broke.
        logger.error("Advanced parser failed on %s: %s", path.name, exc)
        raise

    if not processed:
        logger.warning("Advanced parser produced 0 chunks for %s", path.name)
        return []

    return _to_documents(processed, path)


def _to_documents(
    chunks: List[ProcessedChunk],
    path: Path,
) -> List[Document]:
    """1:1 convert advanced-parser chunks into our ``Document`` list.

    Every ``ProcessedChunk`` becomes one ``Document``. Metadata is
    forwarded plus a few normalised fields (``source``, ``content_type``,
    ``content_hash``, ``advanced_parser=True``) so downstream extractors
    can branch on the kind of content.
    """
    docs: List[Document] = []
    type_counts: dict[str, int] = {}
    for ch in chunks:
        meta = dict(ch.metadata or {})
        ctype = (ch.content_type or "text").lower()
        type_counts[ctype] = type_counts.get(ctype, 0) + 1
        docs.append(Document(
            content=ch.content,
            metadata={
                **meta,
                "source": str(path),
                "filename": path.name,
                "content_type": ctype,
                "content_hash": ch.content_hash,
                "advanced_parser": True,
            },
            source=str(path),
            doc_type=ctype,
        ))
    logger.info(
        "Advanced parser: %s -> %d Documents (%s)",
        path.name, len(docs),
        ", ".join(f"{n} {t}" for t, n in sorted(type_counts.items())),
    )
    return docs
