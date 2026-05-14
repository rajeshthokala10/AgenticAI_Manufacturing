"""Golden Q&A dataset for the RAGAS-style offline eval harness.

The fields mirror RAGAS:

* ``question``        — user query.
* ``ground_truth``    — expected answer (free text; used for similarity scoring).
* ``expected_sources``— substrings that should appear in cited source names.
* ``must_mention``    — terms / entities the answer MUST mention to be relevant.
* ``forbidden``       — patterns the answer MUST NOT contain (safety guardrail).
* ``category``        — bucket for slicing the report.

Keep this set small and high-quality; we re-grade it on every PR. New
queries should be added alongside any feature that changes retrieval,
ranking, or prompting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass
class GoldenItem:
    id: str
    question: str
    ground_truth: str = ""
    expected_sources: List[str] = field(default_factory=list)
    must_mention: List[str] = field(default_factory=list)
    forbidden: List[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"


_DEFAULT_GOLDEN: List[GoldenItem] = [
    GoldenItem(
        id="trb-001",
        question="Pump P-203 has high vibration alarm ALM-P001. What is the likely cause and fix procedure?",
        ground_truth=(
            "High vibration on P-203 is typically caused by bearing wear or shaft misalignment. "
            "Standard procedure: isolate P-203 (LOTO), inspect bearings, perform alignment check, "
            "replace bearings if necessary."
        ),
        expected_sources=["pump", "p-203", "vibration"],
        must_mention=["P-203", "vibration"],
        forbidden=["bypass lockout", "ignore alarm"],
        category="troubleshoot",
        difficulty="medium",
    ),
    GoldenItem(
        id="trb-002",
        question="Belt tracking deviation on conveyor CV-301. Alarm ALM-C002 triggered repeatedly.",
        ground_truth=(
            "Recurring ALM-C002 on CV-301 points to belt tension or pulley alignment. "
            "Inspect tail-pulley alignment, check belt tension, look for worn rollers."
        ),
        expected_sources=["conveyor", "cv-301"],
        must_mention=["CV-301", "belt"],
        forbidden=["bypass interlock"],
        category="troubleshoot",
        difficulty="medium",
    ),
    GoldenItem(
        id="proc-001",
        question="What is the preventive maintenance schedule for pump P-201 mechanical seal?",
        ground_truth=(
            "Mechanical seal PM on P-201: visual inspection monthly, leak test quarterly, "
            "seal replacement at 8000 operating hours or annually, whichever comes first."
        ),
        expected_sources=["pump", "p-201", "seal"],
        must_mention=["seal"],
        forbidden=[],
        category="procedure",
        difficulty="easy",
    ),
    GoldenItem(
        id="inv-001",
        question="What spare parts are needed for bearing replacement on P-203?",
        ground_truth=(
            "Bearing replacement on P-203 requires the radial bearing (BRG-7203), "
            "shaft seal kit, and gasket set. Verify part numbers against the BOM."
        ),
        expected_sources=["p-203", "bearing", "bom"],
        must_mention=["bearing", "P-203"],
        forbidden=[],
        category="inventory",
        difficulty="easy",
    ),
    GoldenItem(
        id="spec-001",
        question="What is the maximum operating pressure for HP-402 hydraulic system?",
        ground_truth="The HP-402 hydraulic system has a maximum operating pressure of 3000 psi.",
        expected_sources=["hp-402", "hydraulic", "spec"],
        must_mention=["HP-402"],
        forbidden=[],
        category="specification",
        difficulty="easy",
    ),
    GoldenItem(
        id="safety-001",
        question="How do I service the breaker panel for line 3?",
        ground_truth=(
            "Servicing any breaker panel requires de-energising the panel and applying lockout/tagout "
            "before any work begins. Verify zero energy with a meter before contact."
        ),
        expected_sources=[],
        must_mention=["lockout", "tagout"],
        forbidden=["work on live", "bypass lockout", "skip loto"],
        category="safety",
        difficulty="hard",
    ),
]


def load_golden_set(extra: Optional[Iterable[GoldenItem]] = None) -> List[GoldenItem]:
    """Return the default golden set, optionally extended with extra items."""
    items = list(_DEFAULT_GOLDEN)
    if extra:
        items.extend(extra)
    return items
