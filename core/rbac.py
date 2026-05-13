"""Role-based access control (RBAC) for HITL approvals.

This module is the single source of truth for:

1. **The role catalogue** — the set of personas relevant to a manufacturing
   plant, modelled after real maker / checker workflows. Every signed-up user
   carries exactly one role; the catalogue is intentionally finite so the
   approval-routing logic stays auditable.

2. **The driver → required-roles map** — given the list of ``drivers`` the
   ``criticality_classifier`` attached to a pending approval, returns the set
   of roles that are authorised to resolve it. The set has **OR** semantics
   (any user with one of these roles can approve) so we never block on a
   single missing person; a stricter AND policy can be layered on later for
   regulated workflows (e.g. fatality-class incidents).

3. **Helpers** for the API layer to check ``can_approve(role, required)`` and
   enforce the **maker-cannot-be-checker** rule.

Why a separate module instead of inlining into ``criticality_classifier``?
The classifier scores risk; this module *routes* the resulting risk to the
right approver. Keeping them separate means we can swap the classifier (e.g.
plug in a learned model) without rewriting the routing policy, and vice
versa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# ─── Role catalogue ──────────────────────────────────────────────────────────
#
# Each role has:
#   * ``id``          — stable machine identifier used in the user store and
#                       the audit log. Never rename — migrate instead.
#   * ``label``       — human-readable name shown in the UIs.
#   * ``description`` — one-line summary of what this role is allowed to do.
#   * ``is_maker``    — operators are the *only* makers in this iteration;
#                       all other roles are checkers. A maker can submit
#                       queries but cannot approve their own escalations.

@dataclass(frozen=True)
class Role:
    id: str
    label: str
    description: str
    is_maker: bool = False
    is_checker: bool = True


ROLES: Tuple[Role, ...] = (
    Role(
        id="operator",
        label="Operator",
        description=(
            "Line / control-room operator. Submits diagnostic queries and "
            "purchase requests. Cannot approve any escalation."
        ),
        is_maker=True,
        is_checker=False,
    ),
    Role(
        id="shift_supervisor",
        label="Shift Supervisor",
        description=(
            "Approves routine PM work, low-confidence diagnostic answers, "
            "and minor procedure deviations during the shift."
        ),
    ),
    Role(
        id="maintenance_planner",
        label="Maintenance Planner",
        description=(
            "Approves PM schedule changes, work-order release, and routine "
            "spare-part picks against existing plans."
        ),
    ),
    Role(
        id="maintenance_engineer",
        label="Maintenance Engineer",
        description=(
            "Approves equipment troubleshooting steps, lockout/tagout for "
            "routine maintenance, and Class-A equipment work."
        ),
    ),
    Role(
        id="ehs_officer",
        label="EHS Officer",
        description=(
            "Environment-Health-Safety officer. Mandatory approver for any "
            "safety-keyword escalation: lockout, hot work, confined space, "
            "permit-to-work, fire, H2S, arc-flash, injury / fatality."
        ),
    ),
    Role(
        id="quality_engineer",
        label="Quality Engineer",
        description=(
            "Approves SOP deviations, NCR closures, COA changes, and "
            "customer-facing quality decisions."
        ),
    ),
    Role(
        id="buyer",
        label="Buyer",
        description=(
            "Approves purchase requests up to the small-PO threshold "
            "(default ≤ $10,000). Larger POs route to Procurement Manager."
        ),
    ),
    Role(
        id="procurement_manager",
        label="Procurement Manager",
        description=(
            "Approves purchase requests above the buyer ceiling, "
            "single-source vendors, and long-lead-time items."
        ),
    ),
    Role(
        id="plant_manager",
        label="Plant Manager",
        description=(
            "Top-level escalation. Approves anything > $100k, multi-week "
            "downtime, regulatory incidents, and fatality-class events."
        ),
    ),
)

ROLES_BY_ID: Dict[str, Role] = {r.id: r for r in ROLES}
ROLE_IDS: Tuple[str, ...] = tuple(r.id for r in ROLES)
MAKER_ROLE_IDS: Set[str] = {r.id for r in ROLES if r.is_maker}
CHECKER_ROLE_IDS: Set[str] = {r.id for r in ROLES if r.is_checker}


def get_role(role_id: str) -> Optional[Role]:
    """Return the :class:`Role` with this id, or ``None`` if unknown."""
    return ROLES_BY_ID.get(role_id)


def list_roles_public() -> List[Dict[str, Any]]:
    """JSON-friendly catalogue for the UIs (signup screen / role badge)."""
    return [
        {
            "id": r.id,
            "label": r.label,
            "description": r.description,
            "is_maker": r.is_maker,
            "is_checker": r.is_checker,
        }
        for r in ROLES
    ]


# ─── Driver → required roles routing ────────────────────────────────────────
#
# The classifier emits *drivers* as short strings (``safety_keyword:lockout``,
# ``purchase_value=$5,000>=$2,000``, ``low_critic_confidence:0.42`` …). The
# map below is a list of ``(predicate, roles)`` pairs evaluated in order;
# every matching predicate contributes its role-set to the union. We never
# bail early — a single pending may need (e.g.) ``ehs_officer`` *and* allow
# ``maintenance_engineer`` to also approve, so we collect everything.
#
# Predicates are *prefix matches* on the driver string to stay forgiving of
# the live values the classifier reports (e.g.
# ``"safety_keyword:hot work"`` matches both ``safety_keyword:`` and
# ``safety_keyword:hot work``). Tier dollars are extracted by ``_dollar_tier``.

# Keywords that mandate the EHS officer (or a higher-authority escalation).
_SAFETY_EHS_KEYWORDS: Tuple[str, ...] = (
    "lockout", "tagout", "hot work", "fire", "explosion", "h2s",
    "arc flash", "confined space", "toxic", "asphyxiation", "radiation",
    "permit-to-work", "permit to work",
)
# Plant-manager-mandatory keywords (life-safety / regulatory exposure).
_LIFE_SAFETY_KEYWORDS: Tuple[str, ...] = (
    "fatal", "fatality", "injury", "death",
)
# Keywords where Maintenance Engineering co-approves with EHS.
_MAINT_COAPPROVE_KEYWORDS: Tuple[str, ...] = ("lockout", "tagout", "shutdown")
# Emergency / unplanned shutdown — supervisor + EHS + plant manager.
_EMERGENCY_KEYWORDS: Tuple[str, ...] = ("emergency", "shutdown")

# Purchase-value tiers (USD). The numbers track the README defaults.
PURCHASE_TIER_BUYER_MAX: float = 10_000.0
PURCHASE_TIER_PROC_MAX: float = 100_000.0


def _dollar_value(driver: str) -> Optional[float]:
    """Extract the dollar value from a ``purchase_value=$5,000>=$2,000`` driver."""
    if not driver.startswith("purchase_value"):
        return None
    # Format: ``purchase_value=$5,000>=$2,000``  — first $-number is the actual.
    after_eq = driver.split("=", 1)[-1]
    head = after_eq.split(">=")[0]
    raw = head.replace("$", "").replace(",", "").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def required_roles_for(
    drivers: Iterable[str],
    purchase_request: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return the de-duplicated list of role-ids allowed to resolve this approval.

    Empty drivers list → the safe fallback ``shift_supervisor`` only (an
    approval should never be totally un-routable; "any supervisor" is a
    sensible default for the inconclusive band).
    """
    required: Set[str] = set()

    for raw in drivers or []:
        d = (raw or "").lower()

        # ── Safety / EHS ─────────────────────────────────────────────────
        if any(kw in d for kw in _SAFETY_EHS_KEYWORDS):
            required.add("ehs_officer")
        if any(kw in d for kw in _LIFE_SAFETY_KEYWORDS):
            required.add("ehs_officer")
            required.add("plant_manager")
        if any(kw in d for kw in _MAINT_COAPPROVE_KEYWORDS):
            required.add("maintenance_engineer")
        if any(kw in d for kw in _EMERGENCY_KEYWORDS):
            required.add("shift_supervisor")
            required.add("ehs_officer")
            required.add("plant_manager")

        # ── Intent-based ─────────────────────────────────────────────────
        if d.startswith("high_risk_intent"):
            # Routine high-risk procedural intents (lockout_tagout, shutdown,
            # emergency, permit_to_work) all go through EHS.
            required.add("ehs_officer")

        # ── Critic confidence ────────────────────────────────────────────
        if d.startswith("low_critic_confidence"):
            required.add("shift_supervisor")

        # ── Purchase value tier ──────────────────────────────────────────
        if d.startswith("purchase_value"):
            val = _dollar_value(raw)
            if val is None:
                # Defensive: tier unknown → procurement manager
                required.add("procurement_manager")
            elif val <= PURCHASE_TIER_BUYER_MAX:
                required.add("buyer")
            elif val <= PURCHASE_TIER_PROC_MAX:
                required.add("procurement_manager")
            else:
                required.add("procurement_manager")
                required.add("plant_manager")

        # ── Purchase-request side-conditions ────────────────────────────
        if d == "single_source_vendor":
            required.add("procurement_manager")
        if d.startswith("long_lead_time"):
            required.add("procurement_manager")
        if d == "class_a_equipment":
            required.add("maintenance_engineer")
            required.add("reliability_engineer")  # may not exist; harmless

        # ── LLM grader fallback ──────────────────────────────────────────
        if d.startswith("llm_grader"):
            required.add("shift_supervisor")

    # Purchase-request payload can carry signals the drivers don't (e.g. the
    # equipment_criticality field). Honour them too.
    pr = purchase_request or {}
    if (pr.get("equipment_criticality") or "").upper() == "A":
        required.add("maintenance_engineer")
    if pr.get("single_source"):
        required.add("procurement_manager")
    if pr.get("lead_time_days") is not None and pr["lead_time_days"] > 7:
        required.add("procurement_manager")

    # Safety net: never return an empty set.
    if not required:
        required.add("shift_supervisor")

    # Drop any role-ids that aren't in our catalogue (forward-compat).
    valid = [r for r in required if r in ROLES_BY_ID]
    # Stable, catalogue-order output for predictable UIs.
    return [r.id for r in ROLES if r.id in valid]


# ─── Authorisation checks ────────────────────────────────────────────────────

def can_approve(
    user_role: Optional[str],
    required_roles: Iterable[str],
) -> bool:
    """Does ``user_role`` belong to the set of roles allowed to approve?"""
    if not user_role:
        return False
    role = ROLES_BY_ID.get(user_role)
    if role is None or not role.is_checker:
        return False
    return user_role in set(required_roles or [])


def is_maker_locked(
    approver_user_id: Optional[str],
    maker_user_id: Optional[str],
) -> bool:
    """True when the approver is the same person who submitted the request.

    The HITL gate exists precisely to enforce *segregation of duties*; the
    request submitter cannot rubber-stamp their own escalation regardless of
    role. Both IDs are normalised to lowercase before comparison so
    ``alice@plant.local`` and ``Alice@Plant.Local`` are treated as one
    identity.
    """
    if not approver_user_id or not maker_user_id:
        return False
    return approver_user_id.strip().lower() == maker_user_id.strip().lower()


# ─── Use-case matrix for docs / tests ────────────────────────────────────────
#
# A small fixed set of *named scenarios* the integration tests + the README
# can both reference. Keeping the strings in one place means a regression in
# the routing rule fails the smoke test before it ships.

USE_CASES: Tuple[Dict[str, Any], ...] = (
    {
        "id": "lockout_tagout",
        "label": "Lockout/Tagout procedure",
        "example_query": "What is the lockout/tagout procedure for pump P-203?",
        "expected_required_roles": ["maintenance_engineer", "ehs_officer"],
        "notes": "Routine LOTO — either Maintenance Engineering or EHS can sign.",
    },
    {
        "id": "hot_work_permit",
        "label": "Hot-work permit",
        "example_query": "Hot work permit for tank T-9 — emergency shutdown.",
        "expected_required_roles": [
            "shift_supervisor", "maintenance_engineer", "ehs_officer", "plant_manager",
        ],
        "notes": "Hot-work always needs EHS; the emergency keyword adds Plant Manager.",
    },
    {
        "id": "small_po",
        "label": "Small spare-part PO (< $10k)",
        "example_query": "Please raise a PO for 5 BRG-7203 bearings at $5000 from Vendor SKF urgent.",
        "expected_required_roles": ["buyer"],
        "notes": "Below the buyer ceiling — Buyer can clear it solo.",
    },
    {
        "id": "mid_po",
        "label": "Mid-tier PO ($10k–$100k)",
        "example_query": "PO for replacement servo drive: $35,000 from Siemens lead time 12 days.",
        "expected_required_roles": ["procurement_manager"],
        "notes": "Above buyer ceiling but below plant-manager threshold.",
    },
    {
        "id": "large_po",
        "label": "Capital PO (> $100k)",
        "example_query": "Capex PO for $150000 to replace CNC spindle from SKF single source.",
        "expected_required_roles": ["procurement_manager", "plant_manager"],
        "notes": "Single-source flag + tier-3 dollars → dual approval.",
    },
    {
        "id": "fatality_incident",
        "label": "Injury / fatality report",
        "example_query": "Operator injury during permit-to-work on H2S vessel — recommend response.",
        "expected_required_roles": ["ehs_officer", "plant_manager"],
        "notes": "Life-safety keyword forces EHS + Plant Manager.",
    },
)
