"""
Query Auto-Correction Engine for manufacturing business context.

Handles three types of correction:
1. Spelling correction using a manufacturing domain dictionary + edit distance
2. Acronym/abbreviation expansion (OEE, MTBF, SPC, CAPA, etc.)
3. Semantic query enhancement — adds domain-relevant terms to improve retrieval
"""

import re
from difflib import get_close_matches
from dataclasses import dataclass


@dataclass
class CorrectedQuery:
    original: str
    corrected: str
    expanded: str
    corrections_applied: list[str]
    confidence: float


MANUFACTURING_VOCABULARY = {
    "oee", "mtbf", "mttr", "spc", "capa", "ncr", "mrb", "fai", "cmm",
    "cnc", "plc", "hmi", "mes", "erp", "bom", "mrp", "aps", "wms",
    "smed", "tpm", "rcm", "fmea", "ppap", "apqp", "sop",
    "machining", "milling", "turning", "grinding", "stamping", "welding",
    "forging", "casting", "heat treatment", "tempering", "annealing",
    "quenching", "hardening", "nitriding", "carburizing",
    "tolerance", "dimension", "specification", "calibration",
    "tensile", "yield", "hardness", "fatigue", "impact", "elongation",
    "roughness", "profilometer", "micrometer", "caliper", "gauge",
    "iso", "astm", "ansi", "asme", "osha", "neshap", "epa", "rcra",
    "iatf", "nadcap", "aiag",
    "scrap", "rework", "defect", "reject", "nonconformance",
    "downtime", "changeover", "setup", "throughput", "cycle time",
    "takt time", "lead time", "bottleneck", "capacity", "utilization",
    "preventive", "predictive", "corrective", "maintenance",
    "vibration", "thermography", "ultrasonic", "oil analysis",
    "coolant", "lubricant", "hydraulic", "pneumatic",
    "supplier", "vendor", "procurement", "inventory", "kanban",
    "fifo", "safety stock", "reorder point", "consignment",
    "ppm", "cpk", "dpmo", "trir", "dart", "lti",
    "spindle", "bearing", "gearbox", "servo", "encoder", "actuator",
    "fixture", "jig", "die", "mold", "tooling",
    "compliance", "audit", "inspection", "certification", "traceability",
    "aluminum", "titanium", "stainless", "alloy", "copper", "chromium",
}

ACRONYM_EXPANSIONS = {
    "oee": "Overall Equipment Effectiveness (OEE)",
    "mtbf": "Mean Time Between Failures (MTBF)",
    "mttr": "Mean Time To Repair (MTTR)",
    "spc": "Statistical Process Control (SPC)",
    "capa": "Corrective and Preventive Actions (CAPA)",
    "ncr": "Non-Conformance Report (NCR)",
    "mrb": "Material Review Board (MRB)",
    "fai": "First Article Inspection (FAI)",
    "cmm": "Coordinate Measuring Machine (CMM)",
    "cnc": "Computer Numerical Control (CNC) machining",
    "plc": "Programmable Logic Controller (PLC)",
    "hmi": "Human-Machine Interface (HMI)",
    "mes": "Manufacturing Execution System (MES)",
    "erp": "Enterprise Resource Planning (ERP)",
    "bom": "Bill of Materials (BOM)",
    "mrp": "Material Requirements Planning (MRP)",
    "aps": "Advanced Planning and Scheduling (APS)",
    "wms": "Warehouse Management System (WMS)",
    "smed": "Single-Minute Exchange of Die (SMED) changeover",
    "tpm": "Total Productive Maintenance (TPM)",
    "rcm": "Reliability-Centered Maintenance (RCM)",
    "fmea": "Failure Mode and Effects Analysis (FMEA)",
    "ppap": "Production Part Approval Process (PPAP)",
    "apqp": "Advanced Product Quality Planning (APQP)",
    "sop": "Standard Operating Procedure (SOP)",
    "loto": "Lockout/Tagout (LOTO) safety procedure",
    "ppe": "Personal Protective Equipment (PPE)",
    "ppm": "Parts Per Million defect rate (PPM)",
    "cpk": "Process Capability Index (Cpk)",
    "trir": "Total Recordable Incident Rate (TRIR)",
    "dart": "Days Away Restricted or Transferred (DART) rate",
    "lti": "Lost Time Injury (LTI)",
    "avl": "Approved Vendor List (AVL)",
    "pdm": "Predictive Maintenance (PdM)",
    "pm": "Preventive Maintenance (PM)",
    "gd&t": "Geometric Dimensioning and Tolerancing (GD&T)",
    "neshap": "National Emission Standards for Hazardous Air Pollutants (NESHAP)",
    "voc": "Volatile Organic Compounds (VOC) emissions",
    "tss": "Total Suspended Solids (TSS)",
    "npdes": "National Pollutant Discharge Elimination System (NPDES)",
    "aql": "Acceptance Quality Level (AQL)",
    "asn": "Advance Shipping Notice (ASN)",
    "edi": "Electronic Data Interchange (EDI)",
}

DOMAIN_SYNONYMS = {
    "quality": ["quality control", "inspection", "defect", "non-conformance", "NCR", "CAPA"],
    "machine": ["equipment", "CNC", "machining center", "stamping press", "asset"],
    "breakdown": ["failure", "downtime", "unplanned", "MTBF", "MTTR"],
    "safety": ["OSHA", "incident", "TRIR", "PPE", "hazard", "compliance"],
    "cost": ["budget", "variance", "spend", "cost analysis", "CAPEX"],
    "supplier": ["vendor", "procurement", "AVL", "supply chain", "scorecard"],
    "efficiency": ["OEE", "utilization", "capacity", "throughput", "productivity"],
    "environment": ["emissions", "VOC", "wastewater", "EPA", "NESHAP", "RCRA"],
    "schedule": ["planning", "scheduling", "MRP", "APS", "production plan"],
    "parts": ["components", "raw materials", "inventory", "BOM", "spare parts"],
    "repair": ["maintenance", "corrective", "rework", "MTTR", "work order"],
    "testing": ["inspection", "measurement", "CMM", "tensile", "hardness"],
    "scrap": ["waste", "reject", "defect rate", "PPM", "yield loss"],
    "welding": ["weld", "arc welding", "MIG", "TIG", "welding robot"],
    "steel": ["steel coil", "stainless", "carbon steel", "alloy steel", "ASTM"],
    "training": ["certification", "operator training", "safety training", "qualification"],
}

COMMON_MISSPELLINGS = {
    "maintanance": "maintenance", "maintainance": "maintenance",
    "maintenence": "maintenance", "mantenance": "maintenance",
    "tolerence": "tolerance", "toleranse": "tolerance",
    "calender": "calendar", "calandar": "calendar",
    "specificaiton": "specification", "specfication": "specification",
    "defective": "defective", "deffect": "defect",
    "recieved": "received", "recived": "received",
    "equiment": "equipment", "equipement": "equipment",
    "inspectionn": "inspection", "inpsection": "inspection",
    "compliace": "compliance", "complience": "compliance",
    "machning": "machining", "machinng": "machining",
    "qualtiy": "quality", "qualitiy": "quality",
    "procudre": "procedure", "procedur": "procedure",
    "shiping": "shipping", "shippment": "shipment",
    "inventry": "inventory", "inventroy": "inventory",
    "safty": "safety", "saftey": "safety",
    "productin": "production", "prodction": "production",
    "schedul": "schedule", "shedule": "schedule",
    "aluminium": "aluminum", "alluminum": "aluminum",
    "titanum": "titanium", "titanim": "titanium",
    "spinle": "spindle", "spindel": "spindle",
    "hydralic": "hydraulic", "hydraulc": "hydraulic",
    "pnuematic": "pneumatic", "pneumtic": "pneumatic",
    "calibraton": "calibration", "calibraiton": "calibration",
    "thermogrphy": "thermography", "thermograpy": "thermography",
    "vibation": "vibration", "vibraiton": "vibration",
    "scorcard": "scorecard", "scorecard": "scorecard",
    "changover": "changeover", "changerover": "changeover",
    "throughpout": "throughput", "thoughput": "throughput",
}


COMMON_ENGLISH_WORDS = {
    "work", "works", "working", "worker", "workers",
    "training", "train", "trained", "trainer",
    "vendor", "vendors", "score", "scored", "scores",
    "card", "cards", "board", "boards",
    "what", "when", "where", "which", "while", "who", "whom", "whose",
    "how", "why", "the", "this", "that", "these", "those",
    "for", "from", "with", "about", "into", "through",
    "does", "did", "done", "doing", "have", "has", "had",
    "will", "would", "could", "should", "shall", "might", "must",
    "can", "may", "need", "want", "like", "make", "made",
    "show", "tell", "find", "give", "take", "help", "know",
    "get", "got", "set", "put", "run", "let", "say", "said",
    "new", "old", "good", "bad", "high", "low", "long", "short",
    "time", "year", "day", "week", "month", "hour", "minute",
    "line", "list", "rate", "data", "plan", "test", "stop",
    "report", "system", "process", "status", "level", "value",
    "total", "number", "first", "last", "next", "other", "each",
    "all", "any", "both", "more", "most", "some", "such",
    "analysis", "bearing", "bearings", "testing", "handling",
    "plant", "part", "parts", "team", "area", "zone", "unit", "units",
    "use", "used", "using", "after", "before", "between", "during",
    "only", "also", "just", "still", "even", "much", "very",
    # Troubleshooting verbs/nouns commonly used in manufacturing queries —
    # protected from being "auto-corrected" into similarly-spelled acronyms
    # (e.g. fail → FAI, lock → LTI, etc.).
    "fail", "fails", "failed", "failing", "failure", "failures",
    "break", "breaks", "broke", "broken", "breaking", "breakdown", "breakdowns",
    "start", "starts", "started", "starting",
    "stop", "stops", "stopped", "stopping",
    "shut", "shutdown", "shutdowns",
    "error", "errors", "fault", "faults",
    "alarm", "alarms", "alert", "alerts",
    "leak", "leaks", "leaking", "leaked",
    "smoke", "smoking", "smoked",
    "crack", "cracks", "cracked", "cracking",
    "jam", "jams", "jammed", "jamming",
    "stuck", "loose",
    "noise", "noisy",
    "happen", "happens", "happened", "happening",
    "occur", "occurs", "occurred", "occurring",
    "issue", "issues", "problem", "problems", "trouble",
    "cause", "causes", "caused", "causing", "reason", "reasons", "fix", "fixes", "fixed",
    "pump", "pumps", "press", "presses", "motor", "motors",
    "alarm", "alarms", "code", "codes",
    "loto", "tag", "tags", "tagged",
    "ago", "since", "when", "today", "yesterday",
    "low", "high", "hot", "cold", "warm",
    "ok", "okay", "yes", "no",
}


class QueryCorrector:
    """Corrects and enhances user queries using manufacturing domain knowledge."""

    def __init__(self, custom_vocabulary: set[str] | None = None):
        self.vocabulary = MANUFACTURING_VOCABULARY.copy()
        if custom_vocabulary:
            self.vocabulary.update(custom_vocabulary)
        single_word_vocab = {w for w in self.vocabulary if " " not in w}
        self.vocab_list = sorted(single_word_vocab)
        self.protected_words = COMMON_ENGLISH_WORDS

    def correct(self, query: str) -> CorrectedQuery:
        corrections = []
        confidence = 1.0

        step1, spelling_fixes = self._fix_spelling(query)
        corrections.extend(spelling_fixes)
        if spelling_fixes:
            confidence *= 0.9

        step2, acronym_fixes = self._expand_acronyms(step1)
        corrections.extend(acronym_fixes)

        step3, enhancements = self._enhance_query(step2)
        corrections.extend(enhancements)

        return CorrectedQuery(
            original=query,
            corrected=step2,
            expanded=step3,
            corrections_applied=corrections,
            confidence=confidence,
        )

    def _fix_spelling(self, query: str) -> tuple[str, list[str]]:
        words = query.split()
        corrected = []
        fixes = []

        for word in words:
            lower = word.lower().strip(",.;:!?")

            if lower in COMMON_MISSPELLINGS:
                replacement = COMMON_MISSPELLINGS[lower]
                corrected.append(replacement)
                fixes.append(f"spelling: '{word}' → '{replacement}'")
                continue

            if lower in self.vocabulary or lower in self.protected_words or len(lower) <= 2:
                corrected.append(word)
                continue

            matches = get_close_matches(lower, self.vocab_list, n=1, cutoff=0.8)
            if matches and matches[0] != lower and lower not in self.protected_words:
                candidate = matches[0]
                # Don't "correct" a real word into a much shorter acronym (e.g.
                # fail → fai). Only allow shortening to candidates of length ≥ 5.
                if len(candidate) < len(lower) and len(candidate) < 5:
                    corrected.append(word)
                else:
                    corrected.append(candidate)
                    fixes.append(f"spelling: '{word}' → '{candidate}'")
            else:
                corrected.append(word)

        return " ".join(corrected), fixes

    def _expand_acronyms(self, query: str) -> tuple[str, list[str]]:
        words = query.split()
        fixes = []
        expanded_parts = []

        for word in words:
            lower = word.lower().strip(",.;:!?")
            if lower in ACRONYM_EXPANSIONS:
                expansion = ACRONYM_EXPANSIONS[lower]
                expanded_parts.append(word)
                if expansion.lower() not in query.lower():
                    fixes.append(f"acronym: {word} = {expansion}")
            else:
                expanded_parts.append(word)

        result = " ".join(expanded_parts)
        if fixes:
            expansions = "; ".join(f.split("= ")[1] for f in fixes)
            result = f"{result} [{expansions}]"

        return result, fixes

    def _enhance_query(self, query: str) -> tuple[str, list[str]]:
        enhancements = []
        added_terms = set()
        query_lower = query.lower()

        for keyword, synonyms in DOMAIN_SYNONYMS.items():
            if keyword in query_lower:
                for syn in synonyms:
                    if syn.lower() not in query_lower and syn.lower() not in added_terms:
                        added_terms.add(syn.lower())

        if added_terms:
            top_terms = list(added_terms)[:5]
            enhanced = f"{query} (related: {', '.join(top_terms)})"
            enhancements.append(f"enhanced with: {', '.join(top_terms)}")
            return enhanced, enhancements

        return query, enhancements
