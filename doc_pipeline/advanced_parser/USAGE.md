# Production RAG Pipeline - Usage Guide

A production-grade document processing pipeline that extracts, cleans, chunks, and indexes content from PDFs, DOCX, HTML, and plain text files for RAG (Retrieval-Augmented Generation) systems.

---

## Installation

### Core Dependencies

```bash
pip install pymupdf pdfplumber pytesseract Pillow
pip install img2table numpy
```

### Optional Dependencies

```bash
# Embeddings
pip install sentence-transformers

# DOCX support
pip install python-docx

# HTML extraction
pip install trafilatura beautifulsoup4

# Better OCR
pip install surya-ocr easyocr paddleocr

# Azure Document Intelligence
pip install azure-ai-documentintelligence
```

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler
```

---

## Quick Start

### Minimal Usage

```python
from rag_pipeline import ProductionRAGPipeline

pipeline = ProductionRAGPipeline()
chunks = pipeline.process("document.pdf")

for chunk in chunks:
    print(f"[{chunk.content_type}] {chunk.content[:100]}...")
```

### With Configuration

```python
from rag_pipeline import ProductionRAGPipeline, PipelineConfig

config = PipelineConfig(
    chunk_size=1000,
    chunk_overlap=200,
    enable_vlm_charts=False,
    generate_embeddings=False,
)

pipeline = ProductionRAGPipeline(config)
chunks = pipeline.process("document.pdf")
```

### Processing Different File Types

```python
pipeline = ProductionRAGPipeline()

# PDF (text-based or scanned)
chunks = pipeline.process("report.pdf")

# Password-protected PDF
chunks = pipeline.process("secured.pdf", password="secret123")

# DOCX
chunks = pipeline.process("contract.docx")

# HTML
chunks = pipeline.process("webpage.html")

# Plain text
chunks = pipeline.process("notes.txt")
```

---

## Configuration Reference

### PipelineConfig Parameters

#### Chunking

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1000 | Maximum characters per chunk |
| `chunk_overlap` | 200 | Overlap between consecutive chunks |

#### OCR

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ocr_engine` | `"pytesseract"` | OCR engine (`"pytesseract"`, `"surya"`, `"azure"`) |
| `ocr_lang` | `"eng"` | Tesseract language code |
| `ocr_dpi` | 300 | DPI for rendering scanned pages |
| `min_ocr_confidence` | 30.0 | Minimum OCR confidence threshold (%) |
| `min_chars_for_digital` | 50 | Character count below which a page is classified as scanned |

#### Tables

| Parameter | Default | Description |
|-----------|---------|-------------|
| `table_engine` | `"img2table"` | Table extraction engine |
| `max_table_rows_per_chunk` | 15 | Max rows per table chunk (large tables are split) |
| `merge_cross_page_tables` | `True` | Merge tables that span consecutive pages |

#### Charts

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_vlm_charts` | `False` | Use a Vision-Language Model to describe charts |
| `vlm_model` | `"gpt-4o"` | VLM model name (requires OpenAI API key) |
| `min_chart_width` | 200 | Minimum image width (px) to consider as a chart |
| `min_chart_height` | 150 | Minimum image height (px) to consider as a chart |

#### Cleaning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `header_footer_margin_pct` | 0.08 | Top/bottom margin percentage for header/footer detection |
| `header_footer_frequency_threshold` | 0.5 | Fraction of pages text must appear on to be classified as header/footer |
| `strip_watermarks` | `True` | Detect and remove watermark text |
| `detect_redactions` | `True` | Detect redacted areas and mark with `[REDACTED]` |
| `filter_toc_pages` | `True` | Exclude table-of-contents pages from text chunks |

#### References

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_cross_ref_expansion` | `True` | Expand cross-references inline |
| `max_footnote_expansion` | 200 | Max characters for inline footnote expansion |

#### Deduplication

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_dedup` | `True` | Remove duplicate/near-duplicate chunks |
| `dedup_similarity_threshold` | 0.95 | Similarity threshold for near-duplicate detection |

#### Encoding

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoding_fixes` | `True` | Fix ligatures, smart quotes, mojibake |

#### Scale / Fault Tolerance

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_pages` | 0 | Max pages to process (0 = unlimited) |
| `batch_size` | 50 | Pages per batch for memory management |
| `fail_on_error` | `False` | `True` = raise on error, `False` = skip bad pages and log warning |

#### Embeddings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `generate_embeddings` | `False` | Generate vector embeddings for each chunk |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Sentence-transformers model name |

#### Versioning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_versioning` | `False` | Track document versions across processing runs |
| `version_store_path` | `None` | Directory path for version store JSON |

---

## Working with Chunks

### ProcessedChunk Structure

Each chunk returned by the pipeline is a `ProcessedChunk` object:

```python
chunk.content        # The text content of the chunk
chunk.content_type   # "text", "table", "form", "chart", or "footnote"
chunk.metadata       # Dict with page number, source info, quality flags, etc.
chunk.embedding      # List[float] if embeddings are enabled, else None
chunk.content_hash   # MD5 hash of content (for dedup)
chunk.to_dict()      # Serialize to dictionary
```

### Content Types

| Type | Description | How it's generated |
|------|-------------|-------------------|
| `text` | Body text from the document | Section-aware or semantic chunking |
| `table` | Table data (markdown or natural language) | Each table produces 2 chunks: structured + NL summary |
| `form` | Form fields / key-value pairs | Grouped by page, output as natural language |
| `chart` | Chart/graph description | VLM-generated text description (if enabled) |
| `footnote` | PDF annotations (sticky notes, comments) | Extracted separately from main text |

### Metadata Fields

Common metadata fields attached to chunks:

```python
{
    "page": 3,                          # Source page number
    "type": "text",                     # Content type
    "section_path": "1 Intro > 1.1 ...",# Hierarchical section path (if detected)
    "section_number": "1.1",            # Section number
    "source_file": "report.pdf",        # Source filename
    "title": "Annual Report 2025",      # Document title (from PDF metadata)
    "author": "Jane Smith",             # Document author
    "doc_type": "pdf",                  # Document type
    "headers": ["Col1", "Col2"],        # Table headers (table chunks only)
    "row_range": "1-15",                # Row range (table chunks only)
    "total_rows": 42,                   # Total table rows (table chunks only)
    "low_quality": True,                # Quality flag: low alpha ratio
    "short_chunk": True,                # Quality flag: under 50 chars
    "repetitive": True,                 # Quality flag: high word repetition
    "has_redactions": True,             # Page contains redacted content
    "warnings": ["Error on page 5: ..."] # Processing warnings
}
```

---

## Evaluation

### Coverage Report

```python
from rag_pipeline.evaluate import pipeline_coverage_report

report = pipeline_coverage_report(chunks)
print(report)
# {
#   "total_chunks": 551,
#   "type_distribution": {"table": 419, "form": 55, "text": 77},
#   "has_tables": True,
#   "has_forms": True,
#   "has_charts": False,
#   "has_redactions": False,
#   "quality_flags": {"repetitive": 123, "short_chunk": 3},
#   "avg_chunk_length": 394,
#   "pages_covered": [1, 2, 3, ...]
# }
```

### Extraction Accuracy (requires ground truth)

```python
from rag_pipeline.evaluate import evaluate_extraction

result = evaluate_extraction(
    ground_truth="The revenue was $1.2M in Q3...",
    extracted="The revenue was $1.2M in Q3..."
)
# {"text_similarity": 1.0, "number_recall": 1.0, "noise_count": 0, "length_ratio": 1.0}
```

### Retrieval Quality (requires test queries)

```python
from rag_pipeline.evaluate import evaluate_retrieval

result = evaluate_retrieval(
    questions=["What was Q3 revenue?"],
    expected=["The revenue was $1.2M"],
    retrieved=[["The revenue was $1.2M in Q3...", "Other chunk..."]],
    k=5
)
# {"recall_at_k": 1.0, "mrr": 1.0}
```

---

## Integration Examples

### With a Vector Database (ChromaDB)

```python
import chromadb
from rag_pipeline import ProductionRAGPipeline, PipelineConfig

config = PipelineConfig(generate_embeddings=True)
pipeline = ProductionRAGPipeline(config)
chunks = pipeline.process("document.pdf")

client = chromadb.Client()
collection = client.create_collection("documents")

collection.add(
    ids=[chunk.content_hash for chunk in chunks],
    documents=[chunk.content for chunk in chunks],
    embeddings=[chunk.embedding for chunk in chunks],
    metadatas=[chunk.metadata for chunk in chunks],
)

results = collection.query(query_texts=["What is the revenue?"], n_results=5)
```

### With LangChain

```python
from langchain.schema import Document
from rag_pipeline import ProductionRAGPipeline

pipeline = ProductionRAGPipeline()
chunks = pipeline.process("document.pdf")

langchain_docs = [
    Document(page_content=chunk.content, metadata=chunk.metadata)
    for chunk in chunks
]
```

### Batch Processing Multiple Files

```python
from pathlib import Path
from rag_pipeline import ProductionRAGPipeline, PipelineConfig

config = PipelineConfig(fail_on_error=False)
pipeline = ProductionRAGPipeline(config)

all_chunks = []
for file in Path("documents/").glob("*.*"):
    if file.suffix.lower() in (".pdf", ".docx", ".html", ".txt"):
        try:
            chunks = pipeline.process(str(file))
            all_chunks.extend(chunks)
            print(f"{file.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"{file.name}: FAILED - {e}")

print(f"\nTotal: {len(all_chunks)} chunks from {len(list(Path('documents/').glob('*.*')))} files")
```

### With Document Versioning

```python
from rag_pipeline import ProductionRAGPipeline, PipelineConfig

config = PipelineConfig(
    enable_versioning=True,
    version_store_path="./version_store"
)
pipeline = ProductionRAGPipeline(config)

# First run: version 1
chunks = pipeline.process("policy.pdf")
# Logs: "new, version 1"

# After document is updated, re-process:
chunks = pipeline.process("policy.pdf")
# Logs: "changed, version 2" (only if content changed)
# Logs: "unchanged, version 1" (if identical)
```

---

## Pipeline Stages

The pipeline processes PDFs in the following order:

```
1.  Open PDF (handle encryption)
2.  Extract metadata (title, author, dates, etc.)
3.  Classify pages (digital vs scanned vs empty)
4.  Detect headers/footers (frequency-based)
5.  Detect redactions (annotations + black rectangles)
6.  Detect document boundaries (merged PDFs)
7.  Extract text per page:
    - Digital pages: column-aware extraction
    - Scanned pages: OCR with confidence scoring
    - Clean: strip headers/footers
    - Clean: remove watermarks
    - Clean: mark redactions with [REDACTED]
    - Flag: TOC/index pages
8.  Extract tables (img2table + cross-page merge)
9.  Extract forms (AcroForms + spatial KV detection)
10. Extract charts (optional VLM description)
11. Extract annotations (sticky notes, comments)
12. Extract embedded attachments
13. Parse section hierarchy + footnotes
14. Chunk text (hierarchical or semantic)
15. Resolve cross-references inline
16. Validate chunk quality
17. Deduplicate chunks
18. Generate embeddings (optional)
19. Check document version (optional)
```

---

## Troubleshooting

### No text extracted from PDF

The PDF is likely image-based (scanned). The pipeline auto-detects this and uses OCR. Ensure Tesseract is installed:

```bash
tesseract --version
```

### Low OCR quality

Increase DPI or switch OCR engine:

```python
config = PipelineConfig(ocr_dpi=400, ocr_lang="eng+fra")
```

### Code blocks detected as tables

PDFs with code samples inside bordered boxes will be extracted as tables by img2table. This is expected behavior since they visually appear as tables.

### Too many chunks

Increase chunk size or disable dual table indexing:

```python
config = PipelineConfig(chunk_size=2000, max_table_rows_per_chunk=30)
```

### Memory issues with large PDFs

Limit pages per run:

```python
config = PipelineConfig(max_pages=100)
```

### Pipeline crashes on one bad page

By default, errors are caught per-page and logged. To enforce strict mode:

```python
config = PipelineConfig(fail_on_error=True)
```
