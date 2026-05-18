"""Utilities: dedup, versioning, document boundaries, embeddings."""

import hashlib
import json
import logging
import os
import re
from typing import List, Dict, Optional

from .models import ProcessedChunk
from .config import PipelineConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Deduplication
# =============================================================================

def deduplicate_chunks(chunks: List[ProcessedChunk], config: PipelineConfig) -> List[ProcessedChunk]:
    """Remove duplicate/near-duplicate chunks by content hash + similarity."""
    if not config.enable_dedup:
        return chunks

    seen_hashes = set()
    unique = []

    for chunk in chunks:
        h = chunk.content_hash
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # Check near-duplicates via normalized content
        normalized = _normalize_for_dedup(chunk.content)
        norm_hash = hashlib.md5(normalized.encode()).hexdigest()
        if norm_hash in seen_hashes:
            continue
        seen_hashes.add(norm_hash)

        unique.append(chunk)

    dropped = len(chunks) - len(unique)
    if dropped:
        logger.info(f"Deduplication removed {dropped} chunks")
    return unique


def _normalize_for_dedup(text: str) -> str:
    """Normalize text for dedup comparison."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


# =============================================================================
# Document boundary detection (merged PDFs)
# =============================================================================

def detect_document_boundaries(doc) -> List[int]:
    """Detect where one logical document ends and another begins."""
    boundaries = [0]
    prev_features = None

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        features = _page_features(page, text)

        if prev_features and _is_boundary(prev_features, features, text):
            boundaries.append(page_num)

        prev_features = features

    if len(boundaries) > 1:
        logger.info(f"Detected {len(boundaries)} logical sub-documents")
    return sorted(set(boundaries))


def _page_features(page, text: str) -> Dict:
    return {
        "has_page_1": bool(re.search(r"page\s*1\s*(of|/)", text, re.IGNORECASE)),
        "has_letterhead": _has_letterhead(page),
        "has_date_header": bool(re.search(r"^\s*(Date|Dear|To:|From:)", text, re.IGNORECASE | re.MULTILINE)),
        "dominant_font": _get_dominant_font(page),
        "text_length": len(text),
        "has_signature": bool(re.search(r"(sincerely|regards|signature)", text, re.IGNORECASE)),
    }


def _has_letterhead(page) -> bool:
    top_zone = page.rect.height * 0.15
    for img_info in page.get_images(full=True):
        try:
            rects = page.get_image_rects(img_info[0])
            for rect in rects:
                if rect.y0 < top_zone:
                    return True
        except Exception:
            pass
    return False


def _get_dominant_font(page) -> str:
    from collections import Counter
    fonts = Counter()
    blocks = page.get_text("dict")["blocks"]
    for b in blocks:
        if b["type"] != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                fonts[span["font"]] += len(span["text"])
    return fonts.most_common(1)[0][0] if fonts else ""


def _is_boundary(prev: Dict, curr: Dict, text: str) -> bool:
    signals = 0
    if curr["has_page_1"]:
        signals += 3
    if curr["has_letterhead"] and not prev["has_letterhead"]:
        signals += 2
    if curr["has_date_header"]:
        signals += 1
    if prev["has_signature"]:
        signals += 2
    if curr["dominant_font"] != prev["dominant_font"]:
        signals += 1
    return signals >= 3


# =============================================================================
# Document versioning
# =============================================================================

def compute_doc_fingerprint(chunks: List[ProcessedChunk]) -> str:
    """Compute a fingerprint for the document based on chunk content."""
    content = "".join(sorted(c.content_hash for c in chunks))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def check_version_changes(
    fingerprint: str, source_file: str, config: PipelineConfig
) -> Optional[Dict]:
    """Check if document has changed since last processing."""
    if not config.enable_versioning or not config.version_store_path:
        return None

    store_path = config.version_store_path
    os.makedirs(store_path, exist_ok=True)
    version_file = os.path.join(store_path, "versions.json")

    versions = {}
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            versions = json.load(f)

    prev = versions.get(source_file)
    is_new = prev is None
    is_changed = prev is not None and prev["fingerprint"] != fingerprint

    # Update version store
    from datetime import datetime
    versions[source_file] = {
        "fingerprint": fingerprint,
        "last_processed": datetime.now().isoformat(),
        "version": (prev["version"] + 1) if prev else 1,
    }
    with open(version_file, "w") as f:
        json.dump(versions, f, indent=2)

    if is_new:
        return {"status": "new", "version": versions[source_file]["version"]}
    elif is_changed:
        logger.info(f"Document changed: {source_file} (v{versions[source_file]['version']})")
        return {"status": "changed", "version": versions[source_file]["version"],
                "prev_fingerprint": prev["fingerprint"]}
    else:
        return {"status": "unchanged", "version": prev["version"]}


# =============================================================================
# Embeddings
# =============================================================================

def generate_embeddings(chunks: List[ProcessedChunk], config: PipelineConfig) -> None:
    """Generate embeddings for all chunks using sentence-transformers."""
    if not config.generate_embeddings:
        return

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.embedding_model)
        texts = [c.content for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=True)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
    except ImportError:
        logger.warning("sentence-transformers not installed, skipping embeddings")


# =============================================================================
# Password / encryption handling
# =============================================================================

def open_pdf_safe(pdf_path: str, password: str = "") -> "fitz.Document":
    """Open PDF with optional password, handling encrypted documents."""
    import fitz
    import os
    pdf_path = os.path.abspath(pdf_path)
    doc = fitz.open(pdf_path)
    if doc.is_encrypted:
        if not password:
            raise ValueError(f"PDF is encrypted: {pdf_path}. Provide password.")
        if not doc.authenticate(password):
            raise ValueError(f"Wrong password for: {pdf_path}")
        logger.info(f"Decrypted PDF: {pdf_path}")
    return doc
