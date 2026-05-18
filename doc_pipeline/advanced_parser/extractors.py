"""All extraction logic: text, tables, forms, charts, attachments."""

import io
import re
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter

from .models import (
    PageInfo, PageType, TableData, FormField, ChartInfo, ExtractionResult
)
from .config import PipelineConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Page classification
# =============================================================================

def classify_pages(doc, config: PipelineConfig) -> Dict[int, PageType]:
    """Classify each page as digital, scanned, or empty."""
    result = {}
    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if len(text) >= config.min_chars_for_digital:
            result[page_num] = PageType.DIGITAL
        else:
            images = page.get_images(full=True)
            result[page_num] = PageType.SCANNED if images else PageType.EMPTY
    return result


# =============================================================================
# Text extraction (digital + OCR + encoding + columns + RTL)
# =============================================================================

def extract_text_digital(page, config: PipelineConfig) -> str:
    """Extract text from digital page, respecting column layout."""
    try:
        blocks = page.get_text("dict")["blocks"]
        text_blocks = [b for b in blocks if b.get("type") == 0]
    except Exception:
        text_blocks = []

    if not text_blocks:
        return page.get_text("text") or ""

    # Try column detection, fall back to simple extraction
    try:
        columns = _detect_columns(text_blocks, page.rect.width)
    except Exception:
        columns = [(0, page.rect.width)]

    if len(columns) <= 1:
        text = page.get_text("text")
    else:
        col_blocks = {i: [] for i in range(len(columns))}
        for b in text_blocks:
            mid_x = (b["bbox"][0] + b["bbox"][2]) / 2
            for ci, (cs, ce) in enumerate(columns):
                if cs <= mid_x <= ce:
                    col_blocks[ci].append(b)
                    break

        parts = []
        for ci in sorted(col_blocks.keys()):
            col = sorted(col_blocks[ci], key=lambda b: b["bbox"][1])
            for b in col:
                for line in b.get("lines", []):
                    line_text = "".join(s.get("text", "") for s in line.get("spans", []))
                    if line_text.strip():
                        parts.append(line_text.strip())
        text = "\n".join(parts) if parts else page.get_text("text")

    if config.encoding_fixes:
        text = fix_encoding(text, config)

    return text or ""


def _detect_columns(text_blocks, page_width, gap_threshold=50) -> List[Tuple[float, float]]:
    """Detect column boundaries from text block positions."""
    if not text_blocks:
        return [(0, page_width)]

    import numpy as np
    x_positions = []
    for b in text_blocks:
        x_positions.extend([b["bbox"][0], b["bbox"][2]])

    if len(set(x_positions)) < 4:
        return [(0, page_width)]

    bins = np.linspace(0, page_width, 50)
    hist, edges = np.histogram(x_positions, bins=bins)
    threshold = max(hist) * 0.05

    gap_regions = []
    in_gap = False
    gap_start = 0
    for i, count in enumerate(hist):
        if count <= threshold:
            if not in_gap:
                gap_start = edges[i]
                in_gap = True
        elif in_gap:
            gap_width = edges[i] - gap_start
            if gap_width > gap_threshold:
                gap_regions.append((gap_start, edges[i]))
            in_gap = False

    if not gap_regions:
        return [(0, page_width)]

    columns = []
    col_start = 0
    for gs, ge in gap_regions:
        columns.append((col_start, gs))
        col_start = ge
    columns.append((col_start, page_width))
    return columns


def extract_text_ocr(pdf_path: str, page_num: int, config: PipelineConfig) -> Tuple[str, float]:
    """OCR a scanned page. Returns (text, avg_confidence)."""
    import os
    import fitz
    import pytesseract
    from PIL import Image

    doc = fitz.open(os.path.abspath(pdf_path))
    page = doc[page_num]
    mat = fitz.Matrix(config.ocr_dpi / 72, config.ocr_dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close()

    # Get structured output for confidence
    data = pytesseract.image_to_data(img, lang=config.ocr_lang, output_type=pytesseract.Output.DICT)
    confidences = [c for c in data["conf"] if c > 0]
    avg_conf = sum(confidences) / max(len(confidences), 1)

    if avg_conf < config.min_ocr_confidence:
        logger.warning(f"Low OCR confidence ({avg_conf:.1f}%) on page {page_num + 1}")

    text = pytesseract.image_to_string(img, lang=config.ocr_lang, config="--oem 3 --psm 6")

    if config.encoding_fixes:
        text = fix_encoding(text, config)

    return text.strip(), avg_conf


def fix_encoding(text: str, config: PipelineConfig) -> str:
    """Fix common encoding issues: ligatures, smart quotes, mojibake."""
    for bad, good in config.common_ligature_map.items():
        text = text.replace(bad, good)

    # Fix common mojibake patterns
    mojibake = {
        "Ã©": "e", "Ã¨": "e", "Ã¢": "a", "Ã®": "i",
        "Ã´": "o", "Ã¼": "u", "Ã±": "n", "Â": "",
    }
    for bad, good in mojibake.items():
        text = text.replace(bad, good)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# =============================================================================
# Table extraction
# =============================================================================

def extract_tables(pdf_path: str, config: PipelineConfig) -> List[TableData]:
    """Extract tables using img2table + OCR, with cross-page merge."""
    tables = _extract_tables_raw(pdf_path, config)

    if config.merge_cross_page_tables and len(tables) > 1:
        tables = _merge_cross_page_tables(tables)

    return tables


def _extract_tables_raw(pdf_path: str, config: PipelineConfig) -> List[TableData]:
    """Raw table extraction via img2table."""
    import os
    from img2table.document import PDF
    from img2table.ocr import TesseractOCR

    ocr = TesseractOCR(n_threads=1, lang=config.ocr_lang)
    doc = PDF(src=os.path.abspath(pdf_path))
    extracted = doc.extract_tables(ocr=ocr, borderless_tables=True)

    tables = []
    for page_num, page_tables in extracted.items():
        for table in page_tables:
            df = table.df
            if df is None or df.empty or len(df) < 2:
                continue

            headers = [str(c).strip() for c in df.iloc[0]]
            rows = [[str(v).strip() for v in row] for _, row in df.iloc[1:].iterrows()]

            # Skip if all empty
            if not any(any(c for c in row) for row in rows):
                continue

            md = _rows_to_markdown(headers, rows)
            nl = _table_to_nl(headers, rows)

            tables.append(TableData(
                headers=headers, rows=rows, page=page_num + 1,
                markdown=md, natural_language=nl,
            ))
    return tables


def _merge_cross_page_tables(tables: List[TableData]) -> List[TableData]:
    """Merge tables that span across pages (same headers, consecutive pages)."""
    if not tables:
        return tables

    merged = [tables[0]]
    for t in tables[1:]:
        prev = merged[-1]
        # Same headers and consecutive pages = continuation
        if (t.headers == prev.headers and t.page == prev.page + 1):
            prev.rows.extend(t.rows)
            prev.continues_to_next = False
            prev.markdown = _rows_to_markdown(prev.headers, prev.rows)
            prev.natural_language = _table_to_nl(prev.headers, prev.rows)
            logger.info(f"Merged table from page {t.page} into page {prev.page}")
        else:
            merged.append(t)
    return merged


def _rows_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        md += "| " + " | ".join(padded[:len(headers)]) + " |\n"
    return md


def _table_to_nl(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    for row in rows[:20]:
        parts = [f"{h}: {v}" for h, v in zip(headers, row) if v]
        if parts:
            lines.append(", ".join(parts))
    summary = f"Table with {len(rows)} rows. Columns: {', '.join(h for h in headers if h)}.\n"
    return summary + "\n".join(f"- {l}" for l in lines)


# =============================================================================
# Form / Key-Value extraction
# =============================================================================

def extract_forms(doc, config: PipelineConfig) -> List[FormField]:
    """Extract form fields (AcroForms) and spatial KV pairs."""
    fields = []

    # Method 1: AcroForm widgets
    for page_num, page in enumerate(doc):
        widgets = page.widgets()
        if widgets:
            for w in widgets:
                fields.append(FormField(
                    key=w.field_name or "", value=w.field_value or "",
                    page=page_num + 1, method="acroform", confidence=1.0,
                ))

    # Method 2: Spatial KV detection (colon-separated or side-by-side)
    if not fields:
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            spans = _collect_line_spans(blocks)
            fields.extend(_detect_kv_patterns(spans, page_num + 1))

    return fields


def _collect_line_spans(blocks) -> List[Dict]:
    spans = []
    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line.get("spans", []))
            bbox = line["bbox"]
            spans.append({
                "text": text.strip(), "bbox": bbox,
                "y_mid": (bbox[1] + bbox[3]) / 2,
                "x_start": bbox[0], "x_end": bbox[2],
            })
    return spans


def _detect_kv_patterns(spans: List[Dict], page: int) -> List[FormField]:
    fields = []
    for span in spans:
        text = span["text"]
        if ":" in text:
            parts = text.split(":", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                fields.append(FormField(
                    key=parts[0].strip(), value=parts[1].strip(),
                    page=page, method="colon_split",
                ))
    return fields


# =============================================================================
# Chart/Image extraction
# =============================================================================

def extract_charts(doc, config: PipelineConfig) -> List[ChartInfo]:
    """Extract chart/graph images from PDF pages."""
    charts = []
    for page_num, page in enumerate(doc):
        for img_info in page.get_images(full=True):
            try:
                base_image = doc.extract_image(img_info[0])
                if not base_image:
                    continue
                w, h = base_image["width"], base_image["height"]
                if w < config.min_chart_width or h < config.min_chart_height:
                    continue
                aspect = w / max(h, 1)
                if aspect > 5 or aspect < 0.2:
                    continue
                charts.append(ChartInfo(
                    page=page_num + 1, image_bytes=base_image["image"],
                    width=w, height=h,
                ))
            except Exception as e:
                logger.warning(f"Failed to extract image on page {page_num + 1}: {e}")

        # Full-page vector charts (little text, no embedded images)
        text = page.get_text("text").strip()
        images = page.get_images(full=True)
        if len(text) < 100 and len(images) < 2:
            import fitz
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            charts.append(ChartInfo(
                page=page_num + 1, image_bytes=pix.tobytes("png"),
                width=pix.width, height=pix.height,
            ))

    return charts


def describe_chart_with_vlm(chart: ChartInfo, config: PipelineConfig) -> str:
    """Use a VLM to generate text description of a chart."""
    import base64
    try:
        from openai import OpenAI
        client = OpenAI()
        b64 = base64.b64encode(chart.image_bytes).decode("utf-8")
        resp = client.chat.completions.create(
            model=config.vlm_model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Describe this chart for a search index. Include: chart type, axes, key data points, trends, and conclusions. Be specific with numbers."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
            ]}],
            max_tokens=500,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning(f"VLM chart description failed: {e}")
        return f"[Chart on page {chart.page}, {chart.width}x{chart.height}px]"


# =============================================================================
# Embedded attachments
# =============================================================================

def extract_attachments(doc) -> List[Dict]:
    """Extract embedded file attachments from PDF."""
    attachments = []
    try:
        for i in range(doc.embfile_count()):
            info = doc.embfile_info(i)
            data = doc.embfile_get(i)
            attachments.append({
                "name": info.get("filename", f"attachment_{i}"),
                "size": len(data),
                "data": data,
            })
    except Exception:
        pass
    return attachments


# =============================================================================
# Metadata extraction
# =============================================================================

def extract_metadata(doc, file_path: str) -> Dict:
    """Extract all PDF metadata."""
    meta = doc.metadata or {}
    return {
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "subject": meta.get("subject", ""),
        "keywords": meta.get("keywords", ""),
        "creator": meta.get("creator", ""),
        "producer": meta.get("producer", ""),
        "creation_date": _parse_pdf_date(meta.get("creationDate", "")),
        "modification_date": _parse_pdf_date(meta.get("modDate", "")),
        "page_count": len(doc),
        "is_encrypted": doc.is_encrypted,
        "has_forms": any(page.widgets() for page in doc),
        "source_file": file_path,
    }


def _parse_pdf_date(date_str: str) -> str:
    if not date_str:
        return ""
    try:
        from datetime import datetime
        clean = date_str.replace("D:", "")[:14]
        return datetime.strptime(clean, "%Y%m%d%H%M%S").isoformat()
    except (ValueError, IndexError):
        return date_str
