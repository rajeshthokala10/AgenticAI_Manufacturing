from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PipelineConfig:
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # OCR
    ocr_engine: str = "pytesseract"  # "pytesseract", "surya", "azure"
    ocr_lang: str = "eng"
    ocr_dpi: int = 300
    min_ocr_confidence: float = 30.0
    min_chars_for_digital: int = 50

    # Tables
    table_engine: str = "img2table"  # "img2table", "pdfplumber", "azure"
    max_table_rows_per_chunk: int = 15
    merge_cross_page_tables: bool = True

    # Charts
    enable_vlm_charts: bool = False
    vlm_model: str = "gpt-4o"
    min_chart_width: int = 200
    min_chart_height: int = 150

    # Cleaning
    header_footer_margin_pct: float = 0.08
    header_footer_frequency_threshold: float = 0.5
    strip_watermarks: bool = True
    detect_redactions: bool = True
    filter_toc_pages: bool = True

    # References
    enable_cross_ref_expansion: bool = True
    max_footnote_expansion: int = 200

    # Deduplication
    enable_dedup: bool = True
    dedup_similarity_threshold: float = 0.95

    # Encoding
    encoding_fixes: bool = True
    common_ligature_map: dict = field(default_factory=lambda: {
        "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--", "\u2026": "...", "\u00a0": " ",
    })

    # Scale / fault tolerance
    max_pages: int = 0  # 0 = unlimited
    batch_size: int = 50  # pages per batch for large docs
    fail_on_error: bool = False  # False = skip bad pages, log warning

    # i18n
    rtl_languages: List[str] = field(default_factory=lambda: [
        "ara", "heb", "fas", "urd",
    ])

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    generate_embeddings: bool = False

    # Versioning
    enable_versioning: bool = False
    version_store_path: Optional[str] = None
