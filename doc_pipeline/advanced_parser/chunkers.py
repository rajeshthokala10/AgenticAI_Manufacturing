"""Chunking: hierarchical sections, table chunking, quality validation."""

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .models import ProcessedChunk, TableData, FormField
from .config import PipelineConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Section hierarchy parsing
# =============================================================================

@dataclass
class SectionNode:
    number: str
    title: str
    content: str
    level: int
    children: List["SectionNode"] = field(default_factory=list)
    parent: Optional["SectionNode"] = None

    def full_path(self) -> str:
        parts = []
        node = self
        while node:
            parts.append(f"{node.number} {node.title}".strip())
            node = node.parent
        return " > ".join(reversed(parts))


SECTION_PATTERNS = [
    (r"^(\d+)\.\s+(.+)", 1),               # "1. Title"
    (r"^(\d+\.\d+)\s+(.+)", 2),            # "1.1 Title"
    (r"^(\d+\.\d+\.\d+)\s+(.+)", 3),       # "1.1.1 Title"
    (r"^\(([a-z])\)\s+(.+)", 4),            # "(a) Text"
    (r"^\(([ivxlc]+)\)\s+(.+)", 5),         # "(ii) Text"
    (r"^([A-Z])\.\s+(.+)", 4),             # "A. Text"
]


def parse_sections(text: str) -> List[SectionNode]:
    """Parse document into hierarchical sections."""
    sections = []
    current_stack = {}

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        matched = False
        for pattern, level in SECTION_PATTERNS:
            m = re.match(pattern, line)
            if m:
                node = SectionNode(
                    number=m.group(1),
                    title=m.group(2)[:100] if level <= 3 else "",
                    content=m.group(2), level=level,
                )
                if level > 1 and (level - 1) in current_stack:
                    node.parent = current_stack[level - 1]
                    current_stack[level - 1].children.append(node)

                current_stack[level] = node
                sections.append(node)
                matched = True
                break

        if not matched and sections:
            sections[-1].content += "\n" + line

    return sections


# =============================================================================
# Text chunking with hierarchy
# =============================================================================

def chunk_with_hierarchy(sections: List[SectionNode], config: PipelineConfig) -> List[Dict]:
    """Create chunks preserving section path context."""
    chunks = []
    for section in sections:
        path = section.full_path()
        content = section.content
        chunk_text = f"[{path}]\n{content}"

        if len(chunk_text) > config.chunk_size:
            # Split long sections, keeping path prefix
            words = content.split()
            current, current_len = [], 0
            max_content = config.chunk_size - len(path) - 10

            for word in words:
                if current_len + len(word) + 1 > max_content:
                    chunks.append({
                        "content": f"[{path}]\n{' '.join(current)}",
                        "metadata": {"section_path": path, "section_number": section.number,
                                     "level": section.level, "type": "text"},
                    })
                    current, current_len = [word], len(word)
                else:
                    current.append(word)
                    current_len += len(word) + 1

            if current:
                chunks.append({
                    "content": f"[{path}]\n{' '.join(current)}",
                    "metadata": {"section_path": path, "section_number": section.number,
                                 "level": section.level, "type": "text"},
                })
        else:
            chunks.append({
                "content": chunk_text,
                "metadata": {"section_path": path, "section_number": section.number,
                             "level": section.level, "type": "text"},
            })
    return chunks


def semantic_chunk(pages: List[Dict], config: PipelineConfig) -> List[Dict]:
    """Fallback: chunk by paragraph/sentence boundaries with overlap."""
    chunks = []
    for page in pages:
        text = page.get("text", "")
        if not text.strip():
            continue

        # Split on paragraph boundaries
        paragraphs = re.split(r"\n\n+", text)
        current, current_len = [], 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if current_len + len(para) > config.chunk_size and current:
                chunk_text = "\n\n".join(current)
                chunks.append({
                    "content": chunk_text,
                    "metadata": {"page": page.get("page", 0), "type": "text"},
                })
                # Overlap: keep last paragraph
                overlap_text = current[-1] if current else ""
                current = [overlap_text, para] if overlap_text else [para]
                current_len = len(overlap_text) + len(para)
            else:
                current.append(para)
                current_len += len(para)

        if current:
            chunks.append({
                "content": "\n\n".join(current),
                "metadata": {"page": page.get("page", 0), "type": "text"},
            })
    return chunks


# =============================================================================
# Table chunking
# =============================================================================

def chunk_tables(tables: List[TableData], config: PipelineConfig) -> List[ProcessedChunk]:
    """Create chunks from tables (split large ones, dual-index with NL)."""
    chunks = []
    max_rows = config.max_table_rows_per_chunk

    for table in tables:
        rows = table.rows
        for i in range(0, len(rows), max_rows):
            chunk_rows = rows[i:i + max_rows]
            md = _rows_to_md(table.headers, chunk_rows)
            nl = _rows_to_nl(table.headers, chunk_rows, table.page)

            # Markdown chunk (for structured retrieval)
            chunks.append(ProcessedChunk(
                content=md, content_type="table",
                metadata={
                    "page": table.page, "row_range": f"{i+1}-{i+len(chunk_rows)}",
                    "total_rows": len(rows), "headers": table.headers,
                    "table_markdown": md,
                },
            ))
            # Natural language chunk (for semantic search)
            chunks.append(ProcessedChunk(
                content=nl, content_type="table",
                metadata={
                    "page": table.page, "row_range": f"{i+1}-{i+len(chunk_rows)}",
                    "total_rows": len(rows), "is_nl_summary": True,
                },
            ))
    return chunks


def _rows_to_md(headers, rows):
    md = "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        md += "| " + " | ".join(padded[:len(headers)]) + " |\n"
    return md


def _rows_to_nl(headers, rows, page):
    lines = [f"Table on page {page} with columns: {', '.join(headers)}."]
    for row in rows[:20]:
        parts = [f"{h}: {v}" for h, v in zip(headers, row) if v]
        if parts:
            lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


# =============================================================================
# Form chunking
# =============================================================================

def chunk_forms(forms: List[FormField], source: str) -> List[ProcessedChunk]:
    """Create chunks from form fields, grouped by page."""
    if not forms:
        return []

    by_page = {}
    for f in forms:
        by_page.setdefault(f.page, []).append(f)

    chunks = []
    for page, fields in by_page.items():
        # Natural language version
        nl = f"Form data from {source}, page {page}:\n"
        nl += "\n".join(f"- {f.key}: {f.value}" for f in fields if f.key and f.value)

        # Structured JSON version
        structured = {f.key: f.value for f in fields if f.key and f.value}

        chunks.append(ProcessedChunk(
            content=nl, content_type="form",
            metadata={"page": page, "source": source,
                       "field_count": len(fields), "structured": structured},
        ))
    return chunks


# =============================================================================
# Cross-reference & footnote resolution
# =============================================================================

def resolve_references(chunks: List[Dict], footnotes: Dict[str, str],
                       section_lookup: Dict[str, str], config: PipelineConfig) -> List[Dict]:
    """Expand cross-references and footnotes inline."""
    if not config.enable_cross_ref_expansion:
        return chunks

    ref_pattern = re.compile(
        r"(?:as per|per|see|refer to|in accordance with)\s+"
        r"(?:Section|Clause|Article|Appendix)\s+([\w\d.]+(?:\([a-z]+\))?)",
        re.IGNORECASE,
    )

    enriched = []
    for chunk in chunks:
        content = chunk["content"]
        expansions = []

        # Expand section references
        for m in ref_pattern.finditer(content):
            target = m.group(1)
            if target in section_lookup:
                expansions.append(f"[Referenced {target}: \"{section_lookup[target][:200]}\"]")

        # Expand footnotes
        for fn_id, fn_text in footnotes.items():
            markers = [f"[{fn_id}]", f"({fn_id})", f" {fn_id}"]
            for marker in markers:
                if marker in content:
                    content = content.replace(
                        marker, f"{marker} [fn: {fn_text[:config.max_footnote_expansion]}]", 1
                    )
                    break

        if expansions:
            content += "\n\n--- Referenced Content ---\n" + "\n".join(expansions)

        enriched.append({**chunk, "content": content})
    return enriched


def extract_footnotes(text: str) -> Dict[str, str]:
    """Extract footnotes and build ID-to-content mapping."""
    footnotes = {}
    patterns = [
        r"^(\d+)\.\s+(.+?)(?=^\d+\.|$)",    # "1. text"
        r"^\[(\d+)\]\s+(.+?)(?=^\[\d+\]|$)", # "[1] text"
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.MULTILINE | re.DOTALL):
            fn_text = m.group(2).strip()
            if fn_text and len(fn_text) > 10:
                footnotes[m.group(1)] = fn_text
    return footnotes


# =============================================================================
# Chunk quality validation
# =============================================================================

def validate_chunk_quality(chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
    """Filter out low-quality chunks and flag issues."""
    valid = []
    for chunk in chunks:
        content = chunk.content.strip()

        # Skip empty or near-empty chunks
        if len(content) < 20:
            logger.debug(f"Dropping tiny chunk: {content[:50]}")
            continue

        # Skip chunks that are mostly numbers/symbols (likely garbled OCR)
        alpha_ratio = sum(c.isalpha() for c in content) / max(len(content), 1)
        if alpha_ratio < 0.3 and chunk.content_type == "text":
            logger.warning(f"Low alpha ratio ({alpha_ratio:.2f}), possible OCR garbage")
            chunk.metadata["low_quality"] = True

        # Flag very short chunks
        if len(content) < 50:
            chunk.metadata["short_chunk"] = True

        # Check for excessive repetition
        words = content.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                chunk.metadata["repetitive"] = True
                logger.debug(f"Highly repetitive chunk on page {chunk.metadata.get('page')}")

        valid.append(chunk)

    dropped = len(chunks) - len(valid)
    if dropped:
        logger.info(f"Dropped {dropped} low-quality chunks")
    return valid
