"""
Unified Manufacturing Hybrid GraphRAG pipeline.

This package wires together:
  * doc_pipeline   — PDF/TXT/Excel ingestion, smart chunking, clarifier, query
                     correction, FAISS embeddings.
  * core           — Knowledge graph, BM25 + Vector + Graph hybrid retrieval,
                     LLM orchestrator, critic loop.

Public surface:
    from pipeline import ManufacturingPipeline
    pipe = ManufacturingPipeline()
    pipe.build_or_load()
    result = pipe.query("Pump P-203 has high vibration. Cause and fix?")
"""

from __future__ import annotations

import logging

logger = logging.getLogger("pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


from .unified_pipeline import ManufacturingPipeline, PipelineMode, PipelineResult
from .chat_agent import ChatAgent, ChatState, ChatTurn

# Optional LangGraph orchestrator — only available when ``langgraph`` is installed.
try:
    from .langgraph_orchestrator import LangGraphOrchestrator  # noqa: F401

    _LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    LangGraphOrchestrator = None  # type: ignore[assignment]
    _LANGGRAPH_AVAILABLE = False

__all__ = [
    "ManufacturingPipeline",
    "PipelineMode",
    "PipelineResult",
    "ChatAgent",
    "ChatState",
    "ChatTurn",
    "LangGraphOrchestrator",
    "logger",
]
