"""Cleaning: headers/footers, watermarks, redactions, TOC, noise."""

import re
import logging
from typing import List, Dict, Set, Tuple
from collections import Counter

from .models import PageInfo
from .config import PipelineConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Header / Footer detection & removal
# =============================================================================

def detect_headers_footers(doc, config: PipelineConfig) -> Tuple[Set[str], Set[str]]:
    """Find repeating text in top/bottom margins across pages."""
    margin_pct = config.header_footer_margin_pct
    top_texts = Counter()
    bottom_texts = Counter()

    for page in doc:
        height = page.rect.height
        top_margin = height * margin_pct
        bottom_margin = height * (1 - margin_pct)

        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            text = _block_text(block).strip()
            if not text or len(text) < 3:
                continue

            normalized = _normalize_for_matching(text)
            block_top = block["bbox"][1]
            block_bottom = block["bbox"][3]

            if block_top < top_margin:
                top_texts[normalized] += 1
            elif block_bottom > bottom_margin:
                bottom_texts[normalized] += 1

    threshold = len(doc) * config.header_footer_frequency_threshold
    headers = {t for t, c in top_texts.items() if c >= threshold}
    footers = {t for t, c in bottom_texts.items() if c >= threshold}

    if headers or footers:
        logger.info(f"Detected {len(headers)} header and {len(footers)} footer patterns")

    return headers, footers


def strip_headers_footers(text: str, headers: Set[str], footers: Set[str]) -> str:
    """Remove detected header/footer patterns from text."""
    for pattern in headers | footers:
        text = text.replace(pattern, "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _normalize_for_matching(text: str) -> str:
    """Normalize text for header/footer matching (strip page numbers)."""
    text = re.sub(r"page\s*\d+\s*(of\s*\d+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*-?\s*\d+\s*-?\s*$", "[PAGE_NUM]", text)
    return text.strip()


def _block_text(block) -> str:
    text = ""
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            text += span["text"]
    return text


# =============================================================================
# Watermark detection & removal
# =============================================================================

WATERMARK_PATTERNS = [
    r"^(DRAFT|CONFIDENTIAL|COPY|SAMPLE|VOID|DO NOT COPY)$",
    r"^(APPROVED|REJECTED|PRELIMINARY|FOR REVIEW|INTERNAL USE ONLY)$",
]


def detect_watermarks(page) -> Set[str]:
    """Detect watermark text by size, color, and content heuristics."""
    watermarks = set()
    try:
        blocks = page.get_text("rawdict")["blocks"]
    except Exception:
        return watermarks

    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue

                # Large font = likely watermark
                if span.get("size", 0) > 40:
                    watermarks.add(text)
                    continue

                # Light gray text
                color = span.get("color", 0)
                if isinstance(color, int) and color > 0xCCCCCC:
                    watermarks.add(text)
                    continue

                # Known watermark phrases
                for p in WATERMARK_PATTERNS:
                    if re.match(p, text, re.IGNORECASE):
                        watermarks.add(text)
    return watermarks


def remove_watermarks(text: str, watermarks: Set[str]) -> str:
    """Remove detected watermark text."""
    for wm in watermarks:
        text = text.replace(wm, "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# =============================================================================
# Redaction detection
# =============================================================================

def detect_redactions(doc) -> Dict[int, List]:
    """Detect redacted areas (annotations + black rectangles)."""
    import fitz
    redactions_by_page = {}

    for page_num, page in enumerate(doc):
        rects = []

        # Method 1: Redaction annotations
        for annot in page.annots() or []:
            if annot.type[0] == 12:
                rects.append(fitz.Rect(annot.rect))

        # Method 2: Black filled rectangles
        for drawing in page.get_drawings():
            if drawing.get("fill") and drawing["fill"] == (0, 0, 0):
                rect = drawing.get("rect")
                if rect:
                    w, h = rect[2] - rect[0], rect[3] - rect[1]
                    if 20 < w < page.rect.width * 0.9 and 5 < h < 50:
                        rects.append(fitz.Rect(rect))

        if rects:
            redactions_by_page[page_num] = rects

    if redactions_by_page:
        logger.info(f"Detected redactions on {len(redactions_by_page)} pages")

    return redactions_by_page


def apply_redaction_markers(page, redaction_rects: List, text: str) -> Tuple[str, bool]:
    """Replace redacted text spans with [REDACTED] marker."""
    import fitz
    if not redaction_rects:
        return text, False

    blocks = page.get_text("dict")["blocks"]
    output = []
    has_redactions = False

    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                span_rect = fitz.Rect(span["bbox"])
                if any(span_rect.intersects(r) for r in redaction_rects):
                    line_text += "[REDACTED] "
                    has_redactions = True
                else:
                    line_text += span["text"]
            output.append(line_text.strip())

    return "\n".join(output), has_redactions


# =============================================================================
# TOC / Index page detection
# =============================================================================

def is_toc_page(text: str) -> bool:
    """Detect table of contents or index pages that would pollute retrieval."""
    lines = text.strip().split("\n")
    if not lines:
        return False

    # High ratio of lines with page numbers at end
    page_ref_count = sum(1 for l in lines if re.search(r"\.\s*\d+\s*$", l))
    if page_ref_count > len(lines) * 0.4 and len(lines) > 5:
        return True

    # Explicit markers
    first_lines = " ".join(lines[:3]).lower()
    if any(marker in first_lines for marker in ["table of contents", "contents", "index"]):
        if page_ref_count > 3:
            return True

    return False


# =============================================================================
# Annotation extraction
# =============================================================================

def extract_annotations(doc) -> List[Dict]:
    """Extract PDF annotations (sticky notes, comments) separately."""
    annotations = []
    for page_num, page in enumerate(doc):
        for annot in page.annots() or []:
            annot_type = annot.type[1]
            content = annot.info.get("content", "")
            if content.strip() and annot_type not in ("Redact",):
                annotations.append({
                    "type": annot_type,
                    "content": content.strip(),
                    "page": page_num + 1,
                    "author": annot.info.get("title", ""),
                })
    return annotations
