"""
Clarifier Agent — pre-processes user queries before RAG retrieval.

Three-stage analysis:
1. Intent Classification — maps the query to a manufacturing intent
   (lookup, comparison, troubleshooting, compliance, metric, procedure, trend, status)
2. Entity Extraction — pulls out equipment IDs, part numbers, suppliers,
   metrics, departments, plants, standards, dates, materials, severities
3. Slot Filling — checks whether all required slots for the detected intent
   are present; generates clarification prompts for missing slots
"""

import re
from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════
# 1.  INTENT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════

class Intent(Enum):
    LOOKUP          = "lookup"
    COMPARISON      = "comparison"
    TROUBLESHOOTING = "troubleshooting"
    COMPLIANCE      = "compliance"
    METRIC_QUERY    = "metric_query"
    PROCEDURE       = "procedure"
    TREND           = "trend"
    STATUS          = "status"
    ROOT_CAUSE      = "root_cause"
    UNKNOWN         = "unknown"


INTENT_PATTERNS: list[tuple[Intent, list[str], float]] = [
    # (intent, keyword/phrase patterns, base confidence boost)
    (Intent.TROUBLESHOOTING, [
        r"\bwhy\b.*\b(fail|broke|shut ?down|stop|alarm|error|issue|problem)\b",
        r"\bwhat\b.*\b(caus|happen|went wrong|issue|failure)\b",
        r"\b(diagnos|troubleshoot|debug|investig)\b",
        r"\b(root cause|rca|8d|fishbone|ishikawa)\b",
    ], 0.90),

    (Intent.ROOT_CAUSE, [
        r"\b(root cause|rca|5[- ]?why|8d|fault tree|fishbone)\b",
        r"\bwhy\b.*\b(keep|recur|repeat|again)\b",
    ], 0.92),

    (Intent.COMPARISON, [
        r"\b(compar|versus|vs\.?|differ|better|worse)\b",
        r"\b(which|between)\b.*\b(and|or)\b",
        r"\b(rank|top|bottom|highest|lowest|best|worst)\b",
        r"\bhow\b.*\b(compar|stack up|measure up)\b",
    ], 0.85),

    (Intent.TREND, [
        r"\b(trend|over time|quarter.over.quarter|year.over.year|improv|declin)\b",
        r"\bhow\b.*\b(chang|evolv|progress|improv)\b",
        r"\b(increas|decreas|grow|shrink|ris|fall)\b.*\b(over|since|from)\b",
        r"\b(q[1-4]|month|week|year)\b.*\bto\b.*\b(q[1-4]|month|week|year)\b",
    ], 0.85),

    (Intent.COMPLIANCE, [
        r"\b(complian|osha|epa|iso|iatf|neshap|rcra|regulation|audit|citation)\b",
        r"\b(permit|emission|discharge|violation|certificate)\b",
        r"\bare\b.*\b(compli|meet|pass|conform)\b",
    ], 0.88),

    (Intent.METRIC_QUERY, [
        r"\b(oee|mtbf|mttr|cpk|ppm|trir|dart|scrap rate|yield|throughput)\b",
        r"\bwhat\b.*\b(rate|score|percentage|metric|kpi|target)\b",
        r"\bhow\b.*\b(much|many|often|long|high|low)\b",
    ], 0.87),

    (Intent.PROCEDURE, [
        r"\bhow\b.*\b(do|perform|execute|run|conduct|operate|change|set ?up)\b",
        r"\bwhat\b.*\b(step|procedure|process|instruction|protocol)\b",
        r"\b(sop|standard operating|checklist|guideline|work instruction)\b",
        r"\bprocedure\b.*\bfor\b",
    ], 0.86),

    (Intent.STATUS, [
        r"\b(status|current|state|where\b.*\bstand|update)\b",
        r"\bis\b.*\b(running|active|open|closed|pending|overdue)\b",
        r"\bwhat\b.*\b(status|state|condition)\b",
    ], 0.83),

    (Intent.LOOKUP, [
        r"\b(what|where|who|which|tell|show|find|get|list|give)\b",
        r"\b(detail|information|info|about|describe|explain|define)\b",
    ], 0.70),
]


def _schema_intent_extras(domain: str | None) -> list[tuple[Intent, list[str], float]]:
    """Pull ``clarifier.intent_patterns`` from ``schemas/<domain>.yaml``.

    Each YAML entry has shape::

        - intent: TROUBLESHOOTING        # any Intent enum name (case-insensitive)
          patterns: [r"\\bmag.?drop\\b"]
          boost: 0.92                    # optional, default 0.85

    Unknown intent names are dropped (logged inline). Returned in the same
    ``(Intent, [patterns], boost)`` shape as ``INTENT_PATTERNS`` so we can
    just concatenate the two lists.
    """
    if not domain:
        return []
    raw = _schema_clarifier(domain) or {}
    extras_raw = raw.get("intent_patterns") or []
    out: list[tuple[Intent, list[str], float]] = []
    valid = {i.name.lower(): i for i in Intent}
    for entry in extras_raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("intent", "")).strip().lower()
        intent = valid.get(name)
        if intent is None:
            continue
        patterns = [str(p) for p in (entry.get("patterns") or []) if p]
        if not patterns:
            continue
        try:
            boost = float(entry.get("boost", 0.85))
        except (TypeError, ValueError):
            boost = 0.85
        out.append((intent, patterns, max(0.0, min(boost, 0.99))))
    return out


class IntentClassifier:
    """Rule-based intent classifier.

    Defaults to the manufacturing-flavoured ``INTENT_PATTERNS``. When a
    ``domain`` is provided, additional patterns from
    ``schemas/<domain>.yaml`` → ``clarifier.intent_patterns`` are layered on
    top so domain-specific queries (e.g. aviation "mag drop on run-up")
    score correctly without touching this module.
    """

    def __init__(self, domain: str | None = None):
        self.domain = domain
        self.patterns: list[tuple[Intent, list[str], float]] = (
            list(INTENT_PATTERNS) + _schema_intent_extras(domain)
        )

    def classify(self, query: str) -> tuple[Intent, float]:
        query_lower = query.lower().strip()
        scores: dict[Intent, float] = {}

        for intent, patterns, base_conf in self.patterns:
            match_count = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    match_count += 1

            if match_count > 0:
                conf = min(base_conf + 0.03 * (match_count - 1), 0.98)
                scores[intent] = max(scores.get(intent, 0), conf)

        if not scores:
            return Intent.UNKNOWN, 0.3

        best_intent = max(scores, key=scores.get)
        return best_intent, scores[best_intent]


# ═══════════════════════════════════════════════════════════════════════
# 2.  ENTITY EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedEntity:
    entity_type: str
    value: str
    span: tuple[int, int] = (0, 0)
    normalized: str = ""
    confidence: float = 1.0


EQUIPMENT_PATTERNS = [
    (r"\b(CNC[- ]?[A-Z][- ]?\d{3})\b", "equipment_id"),
    (r"\b(STAMP[- ]?[A-Z][- ]?\d{3})\b", "equipment_id"),
    (r"\b(WELD[- ]?[A-Z][- ]?\d{3})\b", "equipment_id"),
    (r"\b(HT[- ]?[A-Z][- ]?\d{3})\b", "equipment_id"),
    (r"\b(COAT[- ]?[A-Z][- ]?\d{3})\b", "equipment_id"),
    (r"\b(CNC\s+Line\s+\d+)\b", "equipment_line"),
    (r"\b(Mori Seiki\s+\w+)\b", "equipment_name"),
    (r"\b(DMG Mori\s+\w+)\b", "equipment_name"),
    (r"\b(Komatsu\s+\w+)\b", "equipment_name"),
    (r"\b(Fanuc\s+ArcMate\s*\w*)\b", "equipment_name"),
    (r"\b(Ipsen\s+TurboTreater)\b", "equipment_name"),
    (r"\b(Nordson\s+\w+)\b", "equipment_name"),
    (r"\b(Renishaw\s+\w+)\b", "equipment_name"),
]

PART_NUMBER_PATTERNS = [
    (r"\b(TH[- ]?\d{4})\b", "part_number"),
    (r"\b(BRK[- ]?\d{4})\b", "part_number"),
    (r"\b(SFT[- ]?\d{4})\b", "part_number"),
    (r"\b(HSG[- ]?\d{4})\b", "part_number"),
    (r"\b(GR[- ]?\d{4})\b", "part_number"),
    (r"\b(Part\s*#?\s*[A-Z]{2,4}[- ]?\d{3,5})\b", "part_number"),
]

SUPPLIER_NAMES = {
    "nippon steel": "Nippon Steel Corp",
    "arcelormittal": "ArcelorMittal",
    "steel warehouse": "Steel Warehouse Inc",
    "alcoa": "Alcoa Corporation",
    "novelis": "Novelis Inc",
    "sandvik": "Sandvik Coromant",
    "sandvik coromant": "Sandvik Coromant",
    "kennametal": "Kennametal",
    "iscar": "Iscar",
    "skf": "SKF",
    "timken": "Timken",
    "parker": "Parker Hannifin",
    "parker hannifin": "Parker Hannifin",
    "festo": "Festo",
    "ecotransport": "EcoTransport LLC",
}

METRIC_NAMES = {
    "oee": "Overall Equipment Effectiveness",
    "mtbf": "Mean Time Between Failures",
    "mttr": "Mean Time To Repair",
    "cpk": "Process Capability Index",
    "ppm": "Parts Per Million Defects",
    "trir": "Total Recordable Incident Rate",
    "dart": "Days Away Restricted Transferred",
    "lti": "Lost Time Injury",
    "scrap rate": "Scrap Rate",
    "yield": "Production Yield",
    "throughput": "Throughput",
    "cycle time": "Cycle Time",
    "takt time": "Takt Time",
    "lead time": "Lead Time",
    "changeover time": "Changeover Time",
    "on-time delivery": "On-Time Delivery",
    "capacity utilization": "Capacity Utilization",
    "inventory turns": "Inventory Turns",
    "pm compliance": "PM Compliance Rate",
    "downtime": "Downtime",
    "scrap": "Scrap Rate",
}

DEPARTMENT_NAMES = {
    "cnc machining": "CNC Machining",
    "cnc": "CNC Machining",
    "machining": "CNC Machining",
    "stamping": "Stamping",
    "welding": "Welding",
    "assembly": "Assembly",
    "heat treatment": "Heat Treatment",
    "heat treat": "Heat Treatment",
    "finishing": "Finishing/Coating",
    "coating": "Finishing/Coating",
    "painting": "Finishing/Coating",
    "quality": "Quality Control",
    "quality control": "Quality Control",
    "maintenance": "Maintenance",
    "procurement": "Procurement",
    "supply chain": "Supply Chain",
    "warehouse": "Warehouse",
    "shipping": "Shipping/Logistics",
    "logistics": "Shipping/Logistics",
}

PLANT_PATTERNS = [
    (r"\b(Plant\s*[A-Z])\b", "plant"),
    (r"\b(plant\s*[a-z])\b", "plant"),
]

STANDARD_PATTERNS = [
    (r"\b(ISO\s*\d{4,5}(?::\d{4})?)\b", "standard"),
    (r"\b(IATF\s*\d{5})\b", "standard"),
    (r"\b(ASTM\s*[A-Z]?\d+)\b", "standard"),
    (r"\b(ANSI[/ ]?\w+\s*\w*)\b", "standard"),
    (r"\b(ASME\s*\w[\d.]+(?:-\d{4})?)\b", "standard"),
    (r"\b(OSHA)\b", "standard"),
    (r"\b(29\s*CFR\s*\d{4}\.\d+)\b", "regulation"),
    (r"\b(NESHAP|NPDES|RCRA)\b", "regulation"),
    (r"\b(AMS\s*\d{4})\b", "standard"),
    (r"\b(AS\d{4})\b", "standard"),
]

DATE_PATTERNS = [
    (r"\b(Q[1-4]\s*20\d{2})\b", "time_period"),
    (r"\b(Q[1-4])\b", "time_period"),
    (r"\b(20\d{2}[- ]\d{2}[- ]\d{2})\b", "date"),
    (r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s*(20\d{2})?\b", "time_period"),
    (r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(20\d{2})?\b", "time_period"),
    (r"\b(last|this|next)\s+(week|month|quarter|year)\b", "relative_time"),
]

MATERIAL_PATTERNS = [
    (r"\b(steel|stainless\s*(?:steel)?|carbon\s*steel|high[- ]tensile\s*steel)\b", "material"),
    (r"\b(aluminum|aluminium)\b", "material"),
    (r"\b(titanium|Ti\s*Gr\s*\d+)\b", "material"),
    (r"\b(copper|brass|bronze)\b", "material"),
    (r"\b(chromium|nickel|zinc)\b", "material"),
    (r"\b(alloy\s*\d*)\b", "material"),
    (r"\b(6061[- ]?T6)\b", "material_grade"),
    (r"\b(304|316|A36)\b", "material_grade"),
]

SEVERITY_PATTERNS = [
    (r"\b(critical|major|minor)\b", "severity"),
    (r"\b(high|medium|low)\s*(priority|severity|risk)\b", "severity"),
]


def _norm_upper_dash(s: str) -> str:
    return s.upper().replace(" ", "-")


def _norm_upper_nospace(s: str) -> str:
    return s.upper().replace(" ", "")


def _norm_title(s: str) -> str:
    return s.title()


def _norm_upper(s: str) -> str:
    return s.upper()


def _norm_strip(s: str) -> str:
    return s.strip()


# Each pattern group: (regex_patterns, normalizer, use_full_match)
#  - use_full_match=False → entity value is m.group(1)
#  - use_full_match=True  → entity value is m.group(0)
_PATTERN_GROUPS: list[tuple[list[tuple[str, str]], callable, bool]] = [
    (EQUIPMENT_PATTERNS,    _norm_upper_dash,    False),
    (PART_NUMBER_PATTERNS,  _norm_upper_nospace, False),
    (PLANT_PATTERNS,        _norm_title,         False),
    (STANDARD_PATTERNS,     _norm_upper,         False),
    (DATE_PATTERNS,         _norm_strip,         True),
    (MATERIAL_PATTERNS,     _norm_title,         True),
    (SEVERITY_PATTERNS,     _norm_title,         True),
]


def _schema_clarifier(domain: str | None) -> dict | None:
    """Pull the ``clarifier:`` block (if any) from ``schemas/<domain>.yaml``.

    Lets a domain ship its own equipment regexes / supplier / metric /
    department dictionaries without touching this file.
    """
    if not domain:
        return None
    try:
        from config import schema_path
        import yaml as _yaml
        raw = _yaml.safe_load(schema_path(domain).read_text()) or {}
    except Exception:
        return None
    return raw.get("clarifier") if isinstance(raw, dict) else None


class EntityExtractor:
    """Regex + dictionary-based entity extractor. Manufacturing defaults
    plus optional per-domain extras from ``schemas/<domain>.yaml`` →
    ``clarifier`` block."""

    def __init__(self, domain: str | None = None):
        # Start from the manufacturing defaults so today's behaviour stays
        # identical. Schema entries are appended / merged below.
        self.equipment_patterns = list(EQUIPMENT_PATTERNS)
        self.part_number_patterns = list(PART_NUMBER_PATTERNS)
        self.supplier_names = dict(SUPPLIER_NAMES)
        self.metric_names = dict(METRIC_NAMES)
        self.department_names = dict(DEPARTMENT_NAMES)

        extras = _schema_clarifier(domain)
        if extras:
            for entry in extras.get("equipment_patterns") or []:
                # Each entry is ``{pattern, type}`` — re-uses the same
                # (regex, entity_type) tuple shape EntityExtractor uses.
                if isinstance(entry, dict) and entry.get("pattern"):
                    self.equipment_patterns.append(
                        (str(entry["pattern"]), str(entry.get("type", "equipment_id"))),
                    )
            for entry in extras.get("part_number_patterns") or []:
                if isinstance(entry, dict) and entry.get("pattern"):
                    self.part_number_patterns.append(
                        (str(entry["pattern"]), str(entry.get("type", "part_number"))),
                    )
            for k, v in (extras.get("supplier_names") or {}).items():
                self.supplier_names[str(k).lower()] = str(v)
            for k, v in (extras.get("metric_names") or {}).items():
                self.metric_names[str(k).lower()] = str(v)
            for k, v in (extras.get("department_names") or {}).items():
                self.department_names[str(k).lower()] = str(v)

        # Per-instance pattern groups (mirrors the module-level list).
        self._pattern_groups = [
            (self.equipment_patterns,   _norm_upper_dash,    False),
            (self.part_number_patterns, _norm_upper_nospace, False),
            (PLANT_PATTERNS,            _norm_title,         False),
            (STANDARD_PATTERNS,         _norm_upper,         False),
            (DATE_PATTERNS,             _norm_strip,         True),
            (MATERIAL_PATTERNS,         _norm_title,         True),
            (SEVERITY_PATTERNS,         _norm_title,         True),
        ]

    def extract(self, query: str) -> list[ExtractedEntity]:
        entities: list[ExtractedEntity] = []

        for patterns, normalizer, use_full_match in self._pattern_groups:
            entities.extend(self._extract_patterns(query, patterns, normalizer, use_full_match))

        entities.extend(self._extract_dictionary(
            query, self.supplier_names, entity_type="supplier", word_boundary=False,
        ))
        entities.extend(self._extract_dictionary(
            query, self.metric_names, entity_type="metric", word_boundary=True,
        ))
        entities.extend(self._extract_dictionary(
            query, self.department_names, entity_type="department",
            word_boundary=True, first_only=True,
        ))

        return self._deduplicate(entities)

    @staticmethod
    def _extract_patterns(
        query: str,
        patterns: list[tuple[str, str]],
        normalize,
        use_full_match: bool,
    ) -> list[ExtractedEntity]:
        out: list[ExtractedEntity] = []
        for pattern, etype in patterns:
            for m in re.finditer(pattern, query, re.IGNORECASE):
                value = m.group(0) if use_full_match else m.group(1)
                value = value.strip()
                out.append(ExtractedEntity(
                    entity_type=etype,
                    value=value,
                    span=(m.start(), m.end()),
                    normalized=normalize(value),
                ))
        return out

    @staticmethod
    def _extract_dictionary(
        query: str,
        mapping: dict[str, str],
        entity_type: str,
        word_boundary: bool,
        first_only: bool = False,
    ) -> list[ExtractedEntity]:
        out: list[ExtractedEntity] = []
        query_lower = query.lower()

        for key, normalized in mapping.items():
            if word_boundary:
                pattern = r"\b" + re.escape(key) + r"\b"
                iterator = re.finditer(pattern, query_lower)
            else:
                idx = query_lower.find(key)
                if idx < 0:
                    continue

                class _M:
                    def __init__(self, s, e, t): self._s, self._e, self._t = s, e, t
                    def start(self): return self._s
                    def end(self): return self._e
                    def group(self, n): return self._t
                iterator = iter([_M(idx, idx + len(key), query[idx:idx + len(key)])])

            for m in iterator:
                out.append(ExtractedEntity(
                    entity_type=entity_type,
                    value=m.group(0),
                    span=(m.start(), m.end()),
                    normalized=normalized,
                ))
                if first_only:
                    break
        return out

    @staticmethod
    def _deduplicate(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        seen: set[tuple[str, str]] = set()
        unique: list[ExtractedEntity] = []
        for e in entities:
            key = (e.entity_type, e.normalized)
            if key in seen:
                continue
            seen.add(key)
            unique.append(e)
        return unique


# ═══════════════════════════════════════════════════════════════════════
# 3.  SLOT FILLING
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Slot:
    name: str
    entity_types: list[str]
    required: bool = True
    filled: bool = False
    value: str = ""
    prompt: str = ""


INTENT_SLOT_TEMPLATES: dict[Intent, list[dict]] = {
    Intent.METRIC_QUERY: [
        {"name": "metric",      "entity_types": ["metric"],                    "required": True,
         "prompt": "Which metric would you like? (e.g., OEE, MTBF, scrap rate, CPK, PPM)"},
        {"name": "time_period", "entity_types": ["time_period", "date", "relative_time"], "required": False,
         "prompt": "For which time period? (e.g., Q1 2026, January, last month)"},
        {"name": "department",  "entity_types": ["department"],                "required": False,
         "prompt": "For which department? (e.g., CNC Machining, Stamping, Welding)"},
        {"name": "plant",       "entity_types": ["plant"],                     "required": False,
         "prompt": "For which plant? (e.g., Plant A, Plant B)"},
    ],

    Intent.TROUBLESHOOTING: [
        {"name": "equipment",   "entity_types": ["equipment_id", "equipment_line", "equipment_name"], "required": True,
         "prompt": "Which equipment or line? (e.g., CNC-A-004, CNC Line 4, Stamping Press #1)"},
        {"name": "symptom",     "entity_types": [],                            "required": False,
         "prompt": "What symptom or error are you seeing?"},
        {"name": "time_period", "entity_types": ["time_period", "date"],       "required": False,
         "prompt": "When did this occur?"},
    ],

    Intent.ROOT_CAUSE: [
        {"name": "issue",       "entity_types": [],                            "required": True,
         "prompt": "What specific issue do you need the root cause for?"},
        {"name": "equipment",   "entity_types": ["equipment_id", "equipment_line", "equipment_name"], "required": False,
         "prompt": "Which equipment is affected?"},
        {"name": "severity",    "entity_types": ["severity"],                  "required": False,
         "prompt": "What is the severity? (Critical, Major, Minor)"},
    ],

    Intent.COMPARISON: [
        {"name": "entity_a",    "entity_types": ["supplier", "equipment_id", "department", "plant", "material"], "required": True,
         "prompt": "What is the first item to compare?"},
        {"name": "entity_b",    "entity_types": ["supplier", "equipment_id", "department", "plant", "material"], "required": False,
         "prompt": "What is the second item to compare?"},
        {"name": "metric",      "entity_types": ["metric"],                    "required": False,
         "prompt": "Which metric should be compared? (e.g., OEE, quality score, cost)"},
    ],

    Intent.COMPLIANCE: [
        {"name": "standard",    "entity_types": ["standard", "regulation"],    "required": False,
         "prompt": "Which standard or regulation? (e.g., ISO 9001, OSHA, EPA, NESHAP)"},
        {"name": "area",        "entity_types": ["department"],                "required": False,
         "prompt": "Which area or department?"},
        {"name": "time_period", "entity_types": ["time_period", "date"],       "required": False,
         "prompt": "For which time period?"},
    ],

    Intent.PROCEDURE: [
        {"name": "process",     "entity_types": [],                            "required": True,
         "prompt": "Which procedure or process do you need? (e.g., tool change, LOTO, first article inspection)"},
        {"name": "equipment",   "entity_types": ["equipment_id", "equipment_name"], "required": False,
         "prompt": "For which equipment?"},
    ],

    Intent.TREND: [
        {"name": "metric",      "entity_types": ["metric"],                    "required": True,
         "prompt": "Which metric's trend? (e.g., OEE, MTBF, scrap rate)"},
        {"name": "time_range",  "entity_types": ["time_period", "date", "relative_time"], "required": False,
         "prompt": "Over what time range? (e.g., Q4 2025 to Q1 2026)"},
        {"name": "scope",       "entity_types": ["department", "plant", "equipment_id"], "required": False,
         "prompt": "For which scope? (department, plant, or equipment)"},
    ],

    Intent.STATUS: [
        {"name": "subject",     "entity_types": ["equipment_id", "equipment_name", "part_number"], "required": False,
         "prompt": "Status of what? (equipment, work order, NCR, etc.)"},
        {"name": "department",  "entity_types": ["department"],                "required": False,
         "prompt": "Which department?"},
    ],

    Intent.LOOKUP: [
        {"name": "topic",       "entity_types": [],                            "required": True,
         "prompt": "What information are you looking for?"},
    ],

    Intent.UNKNOWN: [
        {"name": "topic",       "entity_types": [],                            "required": True,
         "prompt": "Could you clarify what you're looking for?"},
    ],
}


def _schema_slot_templates(domain: str | None) -> dict[Intent, list[dict]]:
    """Pull ``clarifier.slot_templates`` from ``schemas/<domain>.yaml``.

    Each YAML key is an Intent enum name; the value is a list of slot
    definitions with the same ``{name, entity_types, required, prompt}``
    shape as ``INTENT_SLOT_TEMPLATES``. Schema entries fully replace the
    manufacturing defaults for the given intent — partial overlays would
    be ambiguous, and a domain that wants to extend rather than replace
    can just copy the defaults into its YAML.
    """
    if not domain:
        return {}
    raw = _schema_clarifier(domain) or {}
    tmpl_raw = raw.get("slot_templates") or {}
    valid = {i.name.lower(): i for i in Intent}
    out: dict[Intent, list[dict]] = {}
    for name, slots in tmpl_raw.items():
        intent = valid.get(str(name).strip().lower())
        if intent is None or not isinstance(slots, list):
            continue
        clean: list[dict] = []
        for s in slots:
            if not isinstance(s, dict) or not s.get("name") or not s.get("prompt"):
                continue
            clean.append({
                "name": str(s["name"]),
                "entity_types": [str(t) for t in (s.get("entity_types") or [])],
                "required": bool(s.get("required", False)),
                "prompt": str(s["prompt"]),
            })
        if clean:
            out[intent] = clean
    return out


class SlotFiller:
    """Checks extracted entities against intent-specific slot templates."""

    def __init__(self, domain: str | None = None):
        self.domain = domain
        # Per-instance templates: manufacturing defaults, with the schema's
        # slot_templates block layered on top (full per-intent replacement).
        self.templates: dict[Intent, list[dict]] = dict(INTENT_SLOT_TEMPLATES)
        self.templates.update(_schema_slot_templates(domain))

    def fill_slots(self, intent: Intent,
                   entities: list[ExtractedEntity],
                   query: str) -> list[Slot]:
        template = self.templates.get(intent, self.templates[Intent.UNKNOWN])
        slots = []

        for slot_def in template:
            slot = Slot(
                name=slot_def["name"],
                entity_types=slot_def["entity_types"],
                required=slot_def["required"],
                prompt=slot_def["prompt"],
            )

            if slot.entity_types:
                for entity in entities:
                    if entity.entity_type in slot.entity_types:
                        slot.filled = True
                        slot.value = entity.normalized
                        break
            else:
                slot.filled = self._infer_slot_from_query(slot.name, query)
                if slot.filled:
                    slot.value = "(inferred from query)"

            slots.append(slot)

        return slots

    def _infer_slot_from_query(self, slot_name: str, query: str) -> bool:
        q = query.lower()
        if slot_name == "topic":
            return len(query.split()) >= 3
        if slot_name == "process":
            procedure_words = [
                "tool change", "loto", "lockout", "changeover", "setup",
                "warm-up", "warmup", "calibration", "inspection", "first article",
                "fai", "spc", "audit", "receiving", "shipping", "welding",
            ]
            return any(w in q for w in procedure_words)
        if slot_name == "symptom":
            symptom_words = [
                "alarm", "error", "noise", "vibration", "leak", "overheat",
                "chatter", "drift", "failure", "smoke", "stuck", "jam",
            ]
            return any(w in q for w in symptom_words)
        if slot_name == "issue":
            return len(query.split()) >= 4
        if slot_name == "subject":
            return len(query.split()) >= 3
        return False


# ═══════════════════════════════════════════════════════════════════════
# 4.  CLARIFIER AGENT  (orchestrator)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ClarifierResult:
    original_query: str
    intent: Intent
    intent_confidence: float
    entities: list[ExtractedEntity]
    slots: list[Slot]
    missing_required_slots: list[Slot]
    missing_optional_slots: list[Slot]
    is_complete: bool
    enriched_query: str
    clarification_prompt: str
    summary: str


class ClarifierAgent:
    """
    Orchestrates intent classification, entity extraction, and slot filling.

    Usage:
        agent = ClarifierAgent()
        result = agent.analyze("What is the OEE for Plant A in Q1?")
        # result.is_complete → True
        # result.intent → Intent.METRIC_QUERY
        # result.entities → [metric: OEE, plant: Plant A, time_period: Q1]
    """

    def __init__(self, domain: str | None = None):
        self.domain = domain
        self.intent_classifier = IntentClassifier(domain=domain)
        self.entity_extractor = EntityExtractor(domain=domain)
        self.slot_filler = SlotFiller(domain=domain)

    def analyze(self, query: str) -> ClarifierResult:
        intent, confidence = self.intent_classifier.classify(query)
        entities = self.entity_extractor.extract(query)
        slots = self.slot_filler.fill_slots(intent, entities, query)

        missing_required = [s for s in slots if s.required and not s.filled]
        missing_optional = [s for s in slots if not s.required and not s.filled]
        is_complete = len(missing_required) == 0

        enriched_query = self._build_enriched_query(query, intent, entities, slots)
        clarification_prompt = self._build_clarification(missing_required, missing_optional)
        summary = self._build_summary(intent, confidence, entities, slots, is_complete)

        return ClarifierResult(
            original_query=query,
            intent=intent,
            intent_confidence=confidence,
            entities=entities,
            slots=slots,
            missing_required_slots=missing_required,
            missing_optional_slots=missing_optional,
            is_complete=is_complete,
            enriched_query=enriched_query,
            clarification_prompt=clarification_prompt,
            summary=summary,
        )

    def _build_enriched_query(self, query: str, intent: Intent,
                               entities: list[ExtractedEntity],
                               slots: list[Slot]) -> str:
        parts = [query]

        entity_context = []
        for e in entities:
            if e.normalized and e.normalized.lower() not in query.lower():
                entity_context.append(f"{e.entity_type}:{e.normalized}")

        if entity_context:
            parts.append(f"[entities: {', '.join(entity_context)}]")

        filled_context = []
        for s in slots:
            if s.filled and s.value and s.value != "(inferred from query)":
                filled_context.append(f"{s.name}={s.value}")

        if filled_context:
            parts.append(f"[slots: {', '.join(filled_context)}]")

        if intent != Intent.UNKNOWN:
            parts.append(f"[intent: {intent.value}]")

        return " ".join(parts)

    def _build_clarification(self, missing_required: list[Slot],
                              missing_optional: list[Slot]) -> str:
        if not missing_required and not missing_optional:
            return ""

        lines = []
        if missing_required:
            lines.append("To give you the best answer, I need:")
            for slot in missing_required:
                lines.append(f"  * {slot.prompt}")

        if missing_optional and len(missing_optional) <= 3:
            lines.append("\nOptionally, you can also specify:")
            for slot in missing_optional:
                lines.append(f"  - {slot.prompt}")

        return "\n".join(lines)

    def _build_summary(self, intent: Intent, confidence: float,
                        entities: list[ExtractedEntity], slots: list[Slot],
                        is_complete: bool) -> str:
        lines = []
        lines.append(f"Intent: {intent.value} (confidence: {confidence:.0%})")

        if entities:
            ent_strs = [f"{e.entity_type}=\"{e.normalized}\"" for e in entities]
            lines.append(f"Entities: {', '.join(ent_strs)}")

        filled = [s for s in slots if s.filled]
        unfilled = [s for s in slots if not s.filled]
        lines.append(f"Slots: {len(filled)} filled, {len(unfilled)} unfilled")

        if is_complete:
            lines.append("Status: COMPLETE — ready for retrieval")
        else:
            lines.append("Status: INCOMPLETE — clarification needed")

        return " | ".join(lines)

    def format_analysis(self, result: ClarifierResult) -> str:
        lines = []
        lines.append("┌─ CLARIFIER AGENT " + "─" * 52 + "┐")
        lines.append(f"│  Query: {result.original_query}")
        lines.append(f"│  Intent: {result.intent.value.upper()} "
                     f"(confidence: {result.intent_confidence:.0%})")

        if result.entities:
            lines.append("│  Entities:")
            for e in result.entities:
                lines.append(f"│    [{e.entity_type}] {e.value} → {e.normalized}")

        lines.append("│  Slots:")
        for s in result.slots:
            status = "✓" if s.filled else "✗"
            req = "required" if s.required else "optional"
            val = f" = {s.value}" if s.filled else ""
            lines.append(f"│    {status} {s.name} ({req}){val}")

        if result.is_complete:
            lines.append("│  Status: ✓ COMPLETE — all required slots filled")
        else:
            lines.append("│  Status: ✗ INCOMPLETE — needs clarification")
            if result.clarification_prompt:
                for line in result.clarification_prompt.split("\n"):
                    lines.append(f"│  {line}")

        lines.append("│  Enriched: " + result.enriched_query[:65] +
                     ("..." if len(result.enriched_query) > 65 else ""))
        lines.append("└" + "─" * 71 + "┘")
        return "\n".join(lines)
