"""
RAG Query Engine — ties together ingestion, chunking, embeddings, query correction,
and the clarifier agent into a unified retrieval-augmented generation pipeline.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    from document_ingestion import DocumentIngestion
    from chunking import HybridChunker
    from embeddings import EmbeddingPipeline, SearchResult
    from query_correction import QueryCorrector, CorrectedQuery
    from clarifier_agent import ClarifierAgent, ClarifierResult, Intent
    from config import EMBEDDING_MODEL, INPUT_DOCS_DIR, VECTOR_STORE_DIR, DEFAULT_TOP_K, DEFAULT_CONTEXT_WINDOW
except ImportError:
    from .document_ingestion import DocumentIngestion
    from .chunking import HybridChunker
    from .embeddings import EmbeddingPipeline, SearchResult
    from .query_correction import QueryCorrector, CorrectedQuery
    from .clarifier_agent import ClarifierAgent, ClarifierResult, Intent
    from .config import EMBEDDING_MODEL, INPUT_DOCS_DIR, VECTOR_STORE_DIR, DEFAULT_TOP_K, DEFAULT_CONTEXT_WINDOW


logger = logging.getLogger("doc_pipeline.rag")


@dataclass
class IndexStats:
    documents_ingested: int
    chunks_created: int
    index_vectors: int
    embedding_dim: int
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueryResponse:
    query: str
    clarification: ClarifierResult
    correction: CorrectedQuery
    results: list[SearchResult]
    formatted_output: str

    @property
    def num_results(self) -> int:
        return len(self.results)

    @property
    def intent(self) -> str:
        return self.clarification.intent.value

    @property
    def entities(self) -> list[tuple[str, str]]:
        return [(e.entity_type, e.normalized) for e in self.clarification.entities]

    @property
    def is_complete(self) -> bool:
        return self.clarification.is_complete

    def __getitem__(self, key: str):
        """Backwards-compat dict-style access for existing callers."""
        mapping = {
            "query": self.query,
            "clarification": self.clarification,
            "correction": self.correction,
            "results": self.results,
            "formatted_output": self.formatted_output,
            "num_results": self.num_results,
            "intent": self.intent,
            "entities": self.entities,
            "is_complete": self.is_complete,
        }
        return mapping[key]


class RAGEngine:
    """
    End-to-end RAG pipeline for manufacturing documents.

    Pipeline:
    1. Ingest documents (PDF, TXT, Excel)
    2. Chunk with hybrid strategy (semantic / recursive / sliding window)
    3. Embed chunks → FAISS index
    4. At query time: clarify → correct → search → format context
    """

    def __init__(
        self,
        input_dir: str | Path = INPUT_DOCS_DIR,
        index_dir: str | Path = VECTOR_STORE_DIR,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.input_dir = Path(input_dir)
        self.index_dir = Path(index_dir)
        self.ingestion = DocumentIngestion()
        self.embedding_pipeline = EmbeddingPipeline(model_name=model_name, index_dir=index_dir)
        self.chunker = HybridChunker(embedding_model=self.embedding_pipeline.get_model())
        self.query_corrector = QueryCorrector()
        self.clarifier = ClarifierAgent()
        self.is_indexed = False

    def index_documents(self, save: bool = True) -> dict:
        """Ingest, chunk, embed, and index all documents in the input directory."""
        logger.info("STEP 1: Document Ingestion")
        documents = self.ingestion.ingest_directory(self.input_dir)
        if not documents:
            raise RuntimeError(f"No supported documents found in {self.input_dir}")

        logger.info("STEP 2: Smart Chunking")
        chunks = self.chunker.chunk_documents(documents)

        logger.info("STEP 3: Embedding & Indexing")
        self.embedding_pipeline.build_index(chunks)

        if save:
            self.embedding_pipeline.save()

        self.is_indexed = True
        stats = IndexStats(
            documents_ingested=len(documents),
            chunks_created=len(chunks),
            index_vectors=self.embedding_pipeline.index.ntotal,
            embedding_dim=self.embedding_pipeline.dimension,
            sources=sorted({d.source for d in documents}),
        )
        return stats.to_dict()

    def load_index(self) -> None:
        """Load a previously built index from disk."""
        self.embedding_pipeline.load()
        self.is_indexed = True

    def ensure_indexed(self) -> dict | None:
        """Load existing index, or build one from input_docs if missing.

        Returns build stats if a new index was created, otherwise None.
        """
        if self.embedding_pipeline.has_saved_index():
            self.load_index()
            return None
        logger.info("No saved index found — building from %s", self.input_dir)
        return self.index_documents(save=True)

    def query(
        self,
        user_query: str,
        top_k: int = DEFAULT_TOP_K,
        use_context_window: bool = True,
        show_corrections: bool = True,
        show_clarifier: bool = True,
    ) -> QueryResponse:
        """
        Process a user query through the full RAG pipeline:
        1. Clarifier Agent: classify intent, extract entities, fill slots
        2. Auto-correct and enhance the query
        3. Retrieve relevant chunks (using enriched query)
        4. Format results with source attribution
        """
        if not self.is_indexed:
            raise RuntimeError("No index loaded. Call index_documents() or load_index() first.")

        clarification = self.clarifier.analyze(user_query)
        correction = self.query_corrector.correct(user_query)

        base_query = correction.expanded if correction.corrections_applied else correction.corrected
        search_query = clarification.enriched_query if clarification.entities else base_query

        if use_context_window:
            results = self.embedding_pipeline.search_with_context(
                search_query, top_k=top_k, context_window=DEFAULT_CONTEXT_WINDOW,
            )
        else:
            results = self.embedding_pipeline.search(search_query, top_k=top_k)

        formatted = self._format_results(
            results, correction, clarification, show_corrections, show_clarifier,
        )

        return QueryResponse(
            query=user_query,
            clarification=clarification,
            correction=correction,
            results=results,
            formatted_output=formatted,
        )

    def _format_results(
        self,
        results: list[SearchResult],
        correction: CorrectedQuery,
        clarification: ClarifierResult,
        show_corrections: bool,
        show_clarifier: bool,
    ) -> str:
        lines = ["=" * 72, f"  QUERY: {correction.original}"]

        if show_clarifier:
            lines.append(self.clarifier.format_analysis(clarification))

        if show_corrections and correction.corrections_applied:
            lines.append(f"  CORRECTED: {correction.corrected}")
            for fix in correction.corrections_applied:
                lines.append(f"    * {fix}")

        lines.append("=" * 72)

        if clarification.clarification_prompt and not clarification.is_complete:
            lines.append("")
            lines.append("  >> CLARIFICATION NEEDED:")
            for line in clarification.clarification_prompt.split("\n"):
                lines.append(f"     {line}")
            lines.append("")

        if not results:
            lines.append("\n  No relevant results found.\n")
            return "\n".join(lines)

        for i, result in enumerate(results, 1):
            lines.extend(self._format_single_result(i, result))

        lines.append("\n" + "=" * 72)
        summary = " | ".join([
            f"{len(results)} results",
            f"intent: {clarification.intent.value}",
            f"entities: {len(clarification.entities)}",
            f"confidence: {correction.confidence:.0%}",
        ])
        lines.append(f"  {summary}")
        lines.append("=" * 72)

        return "\n".join(lines)

    @staticmethod
    def _format_single_result(idx: int, result: SearchResult) -> list[str]:
        source_name = Path(result.metadata.get("source", "unknown")).name
        doc_type = result.metadata.get("doc_type", "unknown")
        lines = [f"\n-- Result {idx} -- [{doc_type.upper()}] {source_name} "
                 f"(relevance: {result.score:.3f})"]

        extra = []
        if "page" in result.metadata:
            extra.append(f"page {result.metadata['page']}")
        if "sheet_name" in result.metadata:
            extra.append(f"sheet: {result.metadata['sheet_name']}")
        if "section_title" in result.metadata:
            extra.append(f"section: {result.metadata['section_title']}")
        if extra:
            lines.append(f"   Location: {', '.join(extra)}")

        preview = result.text[:600]
        if len(result.text) > 600:
            last_sentence = preview.rfind(".")
            if last_sentence > 400:
                preview = preview[:last_sentence + 1]
            preview += " [...]"

        lines.append(textwrap.fill(preview, width=72,
                                    initial_indent="   ", subsequent_indent="   "))
        return lines

    def interactive_session(self) -> None:
        """Run an interactive query loop in the terminal."""
        print("\n" + "=" * 70)
        print("  MANUFACTURING DOCUMENT QUERY SYSTEM")
        print("  Type your question (or 'quit' to exit)")
        print("=" * 70)

        while True:
            try:
                user_input = input("\n> Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                print("Session ended.")
                break

            response = self.query(user_input)
            print(response.formatted_output)
