"""
Manufacturing document RAG pipeline package.

Exposes the high-level objects so callers can do:

    from doc_pipeline import RAGEngine, ClarifierAgent, QueryCorrector
"""

import logging

logger = logging.getLogger("doc_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                           datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _try_import():
    try:
        from .rag_engine import RAGEngine, QueryResponse
        from .clarifier_agent import ClarifierAgent, ClarifierResult, Intent
        from .query_correction import QueryCorrector, CorrectedQuery
        from .document_ingestion import Document, DocType, DocumentIngestion
        from .chunking import Chunk, HybridChunker
        from .embeddings import EmbeddingPipeline, SearchResult
        return {
            "RAGEngine": RAGEngine,
            "QueryResponse": QueryResponse,
            "ClarifierAgent": ClarifierAgent,
            "ClarifierResult": ClarifierResult,
            "Intent": Intent,
            "QueryCorrector": QueryCorrector,
            "CorrectedQuery": CorrectedQuery,
            "Document": Document,
            "DocType": DocType,
            "DocumentIngestion": DocumentIngestion,
            "Chunk": Chunk,
            "HybridChunker": HybridChunker,
            "EmbeddingPipeline": EmbeddingPipeline,
            "SearchResult": SearchResult,
        }
    except Exception:
        return {}


globals().update(_try_import())

__all__ = [
    "RAGEngine", "QueryResponse",
    "ClarifierAgent", "ClarifierResult", "Intent",
    "QueryCorrector", "CorrectedQuery",
    "Document", "DocType", "DocumentIngestion",
    "Chunk", "HybridChunker",
    "EmbeddingPipeline", "SearchResult",
    "logger",
]
