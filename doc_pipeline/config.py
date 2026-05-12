"""
Configuration for the document pipeline.

When the doc_pipeline is used as part of the unified Manufacturing
Hybrid GraphRAG project, this module proxies to the root-level `config.py`
so that there is a single source of truth for paths and thresholds.

When run stand-alone (root config not importable), it falls back to its own
hard-coded defaults so the pipeline still works.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


_PIPELINE_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _PIPELINE_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))


try:
    from config import (  # type: ignore[import-not-found]
        INPUT_DOCS_DIR, VECTOR_STORE_DIR, OUTPUT_DIR, INDEX_NAME,
        EMBEDDING_MODEL,
        SEMANTIC_SIMILARITY_THRESHOLD, SEMANTIC_MIN_CHUNK_SIZE, SEMANTIC_MAX_CHUNK_SIZE,
        RECURSIVE_CHUNK_SIZE, RECURSIVE_CHUNK_OVERLAP,
        SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STEP,
        FAISS_IVF_MIN_CHUNKS, FAISS_NPROBE_CAP,
        DEFAULT_TOP_K, DEFAULT_CONTEXT_WINDOW,
        SUPPORTED_EXTENSIONS,
        ensure_dirs,
    )
    PIPELINE_DIR = _PIPELINE_DIR
except Exception:
    PIPELINE_DIR = _PIPELINE_DIR
    INPUT_DOCS_DIR = _PIPELINE_DIR / "input_docs"
    VECTOR_STORE_DIR = _PIPELINE_DIR / "vector_store"
    OUTPUT_DIR = _PIPELINE_DIR / "output"

    INDEX_NAME = "manufacturing_index"
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    SEMANTIC_SIMILARITY_THRESHOLD = 0.45
    SEMANTIC_MIN_CHUNK_SIZE = 100
    SEMANTIC_MAX_CHUNK_SIZE = 1500

    RECURSIVE_CHUNK_SIZE = 1000
    RECURSIVE_CHUNK_OVERLAP = 150

    SLIDING_WINDOW_SIZE = 800
    SLIDING_WINDOW_STEP = 400

    FAISS_IVF_MIN_CHUNKS = 100
    FAISS_NPROBE_CAP = 10

    DEFAULT_TOP_K = 5
    DEFAULT_CONTEXT_WINDOW = 1

    SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".xlsx", ".xls")

    def ensure_dirs() -> None:
        for d in (INPUT_DOCS_DIR, VECTOR_STORE_DIR, OUTPUT_DIR):
            d.mkdir(parents=True, exist_ok=True)
