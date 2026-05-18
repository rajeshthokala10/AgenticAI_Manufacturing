"""Evaluation framework: extraction accuracy, retrieval quality, coverage."""

import re
import logging
from typing import List, Dict
from collections import Counter

from .models import ProcessedChunk

logger = logging.getLogger(__name__)


def evaluate_extraction(ground_truth: str, extracted: str) -> Dict:
    """Evaluate extraction quality against ground truth."""
    from difflib import SequenceMatcher

    similarity = SequenceMatcher(None, ground_truth, extracted).ratio()

    # Number recall (critical for financial/scientific docs)
    gt_nums = set(re.findall(r"\d+\.?\d*", ground_truth))
    ext_nums = set(re.findall(r"\d+\.?\d*", extracted))
    num_recall = len(gt_nums & ext_nums) / max(len(gt_nums), 1)

    # Noise check
    noise_patterns = [r"page \d+ of \d+", r"confidential", r"draft"]
    noise_count = sum(len(re.findall(p, extracted, re.IGNORECASE)) for p in noise_patterns)

    return {
        "text_similarity": round(similarity, 3),
        "number_recall": round(num_recall, 3),
        "noise_count": noise_count,
        "length_ratio": round(len(extracted) / max(len(ground_truth), 1), 3),
    }


def evaluate_retrieval(
    questions: List[str], expected: List[str],
    retrieved: List[List[str]], k: int = 5,
) -> Dict:
    """Evaluate retrieval quality: recall@k and MRR."""
    from difflib import SequenceMatcher

    hits, mrr_sum = 0, 0
    for q, exp, ret in zip(questions, expected, retrieved):
        for rank, chunk in enumerate(ret[:k], 1):
            if exp in chunk or SequenceMatcher(None, exp, chunk).ratio() > 0.8:
                hits += 1
                mrr_sum += 1 / rank
                break

    n = max(len(questions), 1)
    return {"recall_at_k": round(hits / n, 3), "mrr": round(mrr_sum / n, 3)}


def pipeline_coverage_report(chunks: List[ProcessedChunk]) -> Dict:
    """Report which data types were detected and processed."""
    types = Counter(c.content_type for c in chunks)
    quality_flags = Counter()
    for c in chunks:
        for flag in ["low_quality", "short_chunk", "repetitive"]:
            if c.metadata.get(flag):
                quality_flags[flag] += 1

    return {
        "total_chunks": len(chunks),
        "type_distribution": dict(types),
        "has_tables": types.get("table", 0) > 0,
        "has_forms": types.get("form", 0) > 0,
        "has_charts": types.get("chart", 0) > 0,
        "has_redactions": any(c.metadata.get("has_redactions") for c in chunks),
        "quality_flags": dict(quality_flags),
        "avg_chunk_length": round(
            sum(len(c.content) for c in chunks) / max(len(chunks), 1)
        ),
        "pages_covered": sorted(set(
            c.metadata.get("page", 0) for c in chunks if c.metadata.get("page")
        )),
    }
