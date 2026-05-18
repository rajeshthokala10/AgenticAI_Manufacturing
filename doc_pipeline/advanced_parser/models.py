from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import hashlib


class DocumentType(Enum):
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    TXT = "txt"


class PageType(Enum):
    DIGITAL = "digital"
    SCANNED = "scanned"
    EMPTY = "empty"


@dataclass
class PageInfo:
    page_num: int
    text: str
    page_type: PageType
    confidence: float = 1.0
    has_redactions: bool = False
    is_toc: bool = False


@dataclass
class TableData:
    headers: List[str]
    rows: List[List[str]]
    page: int
    markdown: str
    natural_language: str
    continues_from_previous: bool = False
    continues_to_next: bool = False


@dataclass
class FormField:
    key: str
    value: str
    page: int
    method: str = "unknown"
    confidence: float = 1.0


@dataclass
class ChartInfo:
    page: int
    image_bytes: bytes
    width: int
    height: int
    description: str = ""


@dataclass
class ProcessedChunk:
    content: str
    content_type: str  # "text", "table", "form", "chart", "footnote"
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @property
    def content_hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "content_type": self.content_type,
            "metadata": self.metadata,
            "content_hash": self.content_hash,
        }


@dataclass
class ExtractionResult:
    pages: List[PageInfo]
    tables: List[TableData]
    forms: List[FormField]
    charts: List[ChartInfo]
    metadata: Dict
    footnotes: Dict[str, str]
    section_lookup: Dict[str, str]
    warnings: List[str] = field(default_factory=list)
