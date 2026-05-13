"""Document-level access control for the RAG knowledge base.

The HITL layer (``core/rbac.py``) already routes *approvals* to the right
checker role. This module extends the same role catalogue to the **data
layer**: every ingested chunk carries a ``classification`` tag in its
metadata, and the retrievers refuse to return a chunk whose classification
is outside the *reading* tier of the currently-authenticated user.

The design has three moving parts:

1. **Classification tiers (3-level, ordered)**
   - ``public``        — anyone, including operators. SOPs, alarm responses,
                         equipment manuals, public safety bulletins.
   - ``restricted``    — every *checker* role. Incident reports, RCAs, work
                         orders, vendor performance data, lockout permits.
   - ``confidential``  — plant_manager + procurement_manager only. Financial
                         reviews, M&A targets, executive comp, supplier
                         pricing contracts, succession planning.

2. **Role → tier map** (``ROLE_TO_CLASSIFICATIONS``)
   Each role's *read set*. Higher tiers always include lower ones (a plant
   manager can read public docs too) — we model this explicitly rather
   than via numeric levels so the policy is greppable.

3. **Per-request context** (``current_classifications`` ContextVar)
   FastAPI dispatches each request on its own thread, but pipelines call
   deep into retrievers that don't know about the bearer-token user. A
   ``ContextVar`` lets the chat handler stamp the user's role once and any
   retriever-level filter pick it up automatically — no parameter
   plumbing, no global mutation.

   ``with_user_classifications(role)`` is a context manager used by the
   API layer; ``classify_from_path`` is used by the ingester; and
   ``filter_chunks`` is used by the retrievers.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple


# ─── Classification catalogue ────────────────────────────────────────────────

PUBLIC: str = "public"
RESTRICTED: str = "restricted"
CONFIDENTIAL: str = "confidential"

#: All three tiers, ordered from least to most sensitive. Useful for
#: tier-comparison helpers (e.g. "max tier this user can read").
CLASSIFICATIONS: Tuple[str, ...] = (PUBLIC, RESTRICTED, CONFIDENTIAL)

#: Rank used for ordering ("higher rank = more sensitive").
_RANK: Dict[str, int] = {c: i for i, c in enumerate(CLASSIFICATIONS)}

#: Folder name → classification convention. Anything ingested from a path
#: containing one of these segments inherits that classification.
_FOLDER_TO_CLASSIFICATION: Tuple[Tuple[str, str], ...] = (
    ("management", CONFIDENTIAL),
    ("confidential", CONFIDENTIAL),
    ("restricted", RESTRICTED),
    ("internal", RESTRICTED),
)


# ─── Role → readable tiers ──────────────────────────────────────────────────

#: Source of truth for who can read what. Keep keys in lock-step with the
#: role ids declared in ``core/rbac.py`` — the ``smoke_test_acl`` and the
#: unit tests catch drift between the two.
ROLE_TO_CLASSIFICATIONS: Dict[str, Set[str]] = {
    # Pure makers — knowledge base is *public-only*. Operators get SOPs,
    # alarm responses, equipment manuals; they will *not* see incident
    # reports, contract pricing, or strategic documents.
    "operator":               {PUBLIC},

    # Checkers / mid-level supervisors — read up to "restricted". They see
    # incident reports, RCAs, vendor performance, work orders, lockout
    # permits, but never financials or M&A material.
    "shift_supervisor":       {PUBLIC, RESTRICTED},
    "maintenance_planner":    {PUBLIC, RESTRICTED},
    "maintenance_engineer":   {PUBLIC, RESTRICTED},
    "ehs_officer":            {PUBLIC, RESTRICTED},
    "quality_engineer":       {PUBLIC, RESTRICTED},
    "buyer":                  {PUBLIC, RESTRICTED},

    # Management — read *everything*, including ``confidential`` strategic
    # / financial / acquisition / leadership documents.
    "procurement_manager":    {PUBLIC, RESTRICTED, CONFIDENTIAL},
    "plant_manager":          {PUBLIC, RESTRICTED, CONFIDENTIAL},
}


def allowed_classifications(role: Optional[str]) -> Set[str]:
    """Return the read-set for a role id.

    Unknown / missing roles get the *most* restrictive view (``public``
    only). The contract is "deny by default" — anonymous browsers cannot
    accidentally surface restricted content just because the role lookup
    failed.
    """
    if not role:
        return {PUBLIC}
    return set(ROLE_TO_CLASSIFICATIONS.get(role, {PUBLIC}))


def max_tier_for(role: Optional[str]) -> str:
    """Highest classification this role can read — used by the UI badge."""
    tiers = allowed_classifications(role)
    return max(tiers, key=lambda t: _RANK.get(t, -1))


def can_read(role: Optional[str], classification: Optional[str]) -> bool:
    """Authorisation predicate used by tests + the retrievers."""
    cls = classification or PUBLIC
    return cls in allowed_classifications(role)


# ─── Path → classification (ingest-time tagging) ────────────────────────────

def classify_from_path(file_path: str | Path) -> str:
    """Infer a chunk's classification from its source path.

    Convention: a ``management/`` or ``confidential/`` segment anywhere in
    the path → ``confidential``; ``restricted/`` or ``internal/`` →
    ``restricted``; everything else → ``public``. Matching is on path
    *components* (not substrings) so a filename like
    ``incident_management.pdf`` does NOT get auto-classified.
    """
    if not file_path:
        return PUBLIC
    parts = {p.lower() for p in Path(str(file_path)).parts}
    for segment, cls in _FOLDER_TO_CLASSIFICATION:
        if segment in parts:
            return cls
    return PUBLIC


# ─── Per-request context (used by the FastAPI layer) ────────────────────────

#: Set by the API layer at the start of every authenticated request and
#: read by the retrievers. ``None`` means "no auth context" → the
#: retrievers fall back to ``public``-only.
current_classifications: contextvars.ContextVar[Optional[Set[str]]] = (
    contextvars.ContextVar("current_classifications", default=None)
)


@contextmanager
def with_user_classifications(role: Optional[str]) -> Iterator[Set[str]]:
    """Stamp the request's allowed classifications for the duration of the
    block. Resets cleanly on exit so background tasks or follow-up
    handlers don't inherit a stale ACL.
    """
    allowed = allowed_classifications(role)
    token = current_classifications.set(allowed)
    try:
        yield allowed
    finally:
        current_classifications.reset(token)


def active_classifications() -> Set[str]:
    """Return the currently-active read-set, or ``{public}`` if no auth
    context has been set. Always returns a *new* set so callers can mutate
    it without poisoning the contextvar.
    """
    value = current_classifications.get()
    if not value:
        return {PUBLIC}
    return set(value)


# ─── Filtering primitive used by all retrievers ─────────────────────────────

def _extract_classification(obj: Any) -> str:
    """Pull a classification tag off a chunk-shaped object.

    Tolerates the three shapes the codebase actually uses:
      * dicts with ``metadata`` nested (HybridRetriever output).
      * dicts with ``classification`` at top level (FAISS doc records).
      * dataclasses / objects with ``.metadata`` attribute (``SearchResult``).

    Returns ``public`` for objects with no classification — keeps the
    filter compatible with chunks ingested before this feature shipped.
    """
    if obj is None:
        return PUBLIC
    if isinstance(obj, dict):
        meta = obj.get("metadata")
        if isinstance(meta, dict) and "classification" in meta:
            return str(meta.get("classification") or PUBLIC)
        if "classification" in obj:
            return str(obj.get("classification") or PUBLIC)
        return PUBLIC
    meta = getattr(obj, "metadata", None)
    if isinstance(meta, dict):
        return str(meta.get("classification") or PUBLIC)
    return PUBLIC


def filter_chunks(items: Iterable[Any], allowed: Optional[Set[str]] = None) -> List[Any]:
    """Drop items whose classification is outside ``allowed``.

    If ``allowed`` is ``None`` the function pulls the active set from the
    ContextVar — this is the path used by every retriever, so the API
    layer only has to set the var once per request.
    """
    effective = allowed if allowed is not None else active_classifications()
    return [item for item in items if _extract_classification(item) in effective]


# ─── Diagnostics / docs ──────────────────────────────────────────────────────

def policy_snapshot(role: Optional[str]) -> Dict[str, Any]:
    """JSON payload used by ``GET /api/access/policy`` so the UI can show
    which tiers the signed-in user can read.
    """
    tiers = allowed_classifications(role)
    return {
        "role": role,
        "max_tier": max_tier_for(role),
        "allowed_classifications": [c for c in CLASSIFICATIONS if c in tiers],
        "classifications_catalogue": [
            {"id": PUBLIC,       "label": "Public",
             "description": "SOPs, alarm responses, equipment manuals — every role."},
            {"id": RESTRICTED,   "label": "Restricted",
             "description": "Incident reports, RCAs, work orders — checker roles only."},
            {"id": CONFIDENTIAL, "label": "Confidential",
             "description": "Financials, M&A, supplier pricing — management only."},
        ],
    }
