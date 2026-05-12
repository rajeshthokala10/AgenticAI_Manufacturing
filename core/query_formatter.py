import re
from typing import Dict, List, Optional


MANUFACTURING_ABBREVIATIONS = {
    "vib": "vibration",
    "temp": "temperature",
    "press": "pressure",
    "mech": "mechanical",
    "hyd": "hydraulic",
    "elec": "electrical",
    "instr": "instrumentation",
    "maint": "maintenance",
    "assy": "assembly",
    "brg": "bearing",
    "vlv": "valve",
    "cyl": "cylinder",
    "accum": "accumulator",
    "xfmr": "transformer",
    "ctrl": "controller",
    "sw": "switch",
    "freq": "frequency",
    "rpm": "revolutions per minute",
    "psi": "pounds per square inch",
    "gpm": "gallons per minute",
    "plc": "programmable logic controller",
    "hmi": "human machine interface",
    "vfd": "variable frequency drive",
    "dcs": "distributed control system",
    "rca": "root cause analysis",
    "pm": "preventive maintenance",
    "cm": "corrective maintenance",
    "mtbf": "mean time between failures",
    "mttr": "mean time to repair",
    "oee": "overall equipment effectiveness",
    "sop": "standard operating procedure",
    "loto": "lockout tagout",
}

INTENT_PATTERNS = {
    "troubleshoot": [
        r"(?:what|why|how).+(?:fail|error|fault|alarm|trip|stop|down)",
        r"(?:troubleshoot|diagnose|investigate|fix|resolve|repair)",
        r"(?:root cause|rca|failure analysis)",
        r"(?:not working|broken|malfunction|defect)",
    ],
    "procedure": [
        r"(?:how to|steps to|procedure for|process for|instructions)",
        r"(?:replace|install|remove|adjust|calibrate|align|lubricate)",
        r"(?:maintenance|inspection|overhaul|rebuild)",
    ],
    "specification": [
        r"(?:what is the|specification|spec|rating|capacity|tolerance)",
        r"(?:part number|model|serial|dimension|clearance|torque)",
    ],
    "alarm": [
        r"(?:alarm|alert|warning|fault code|error code)",
        r"ALM-[A-Z]\d{3}",
        r"FC-\d{3}",
    ],
    "inventory": [
        r"(?:spare|part|inventory|stock|available|order|lead time)",
        r"SP-\d{4}",
    ],
}


def format_query(raw_query: str) -> Dict:
    normalized = _normalize_text(raw_query)
    entities = _extract_entities(normalized)
    intent = _classify_intent(normalized)
    expanded = _expand_abbreviations(normalized)
    structured = _build_structured_query(expanded, entities, intent)

    return {
        "original": raw_query,
        "normalized": normalized,
        "expanded": expanded,
        "entities": entities,
        "intent": intent,
        "structured_query": structured,
        "search_terms": _extract_search_terms(expanded, entities),
    }


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    return text


def _expand_abbreviations(text: str) -> str:
    words = text.split()
    expanded = []
    for word in words:
        clean = word.lower().strip(".,;:!?")
        if clean in MANUFACTURING_ABBREVIATIONS:
            expanded.append(MANUFACTURING_ABBREVIATIONS[clean])
        else:
            expanded.append(word)
    return " ".join(expanded)


def _extract_entities(text: str) -> Dict[str, List[str]]:
    entities = {}
    patterns = {
        "equipment_ids": r'(?:P-\d{3}|CV-\d{3}|HP-\d{3})',
        "alarm_codes": r'ALM-[A-Z]\d{3}',
        "part_numbers": r'SP-\d{4}',
        "fault_codes": r'FC-\d{3}',
        "work_orders": r'WO-\d{4}-\d{3}',
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            entities[key] = [m.upper() for m in matches]
    return entities


def _classify_intent(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text_lower):
                score += 1
        scores[intent] = score

    if max(scores.values(), default=0) == 0:
        return "general"
    return max(scores, key=scores.get)


def _build_structured_query(text: str, entities: Dict, intent: str) -> str:
    parts = []

    intent_prefixes = {
        "troubleshoot": "Diagnose and provide root cause analysis for:",
        "procedure": "Provide step-by-step procedure for:",
        "specification": "Look up technical specifications for:",
        "alarm": "Explain alarm/fault code and recommended actions for:",
        "inventory": "Check spare parts information for:",
        "general": "Find relevant information about:",
    }
    parts.append(intent_prefixes.get(intent, intent_prefixes["general"]))
    parts.append(text)

    if entities:
        context_parts = []
        if "equipment_ids" in entities:
            context_parts.append(f"Equipment: {', '.join(entities['equipment_ids'])}")
        if "alarm_codes" in entities:
            context_parts.append(f"Alarms: {', '.join(entities['alarm_codes'])}")
        if "part_numbers" in entities:
            context_parts.append(f"Parts: {', '.join(entities['part_numbers'])}")
        if "fault_codes" in entities:
            context_parts.append(f"Fault codes: {', '.join(entities['fault_codes'])}")
        if context_parts:
            parts.append("Context: " + " | ".join(context_parts))

    return " ".join(parts)


def _extract_search_terms(text: str, entities: Dict) -> List[str]:
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "about", "between", "through", "during", "before",
        "after", "above", "below", "up", "down", "out", "off", "over",
        "under", "again", "further", "then", "once", "what", "why",
        "how", "which", "who", "when", "where", "that", "this", "it",
        "and", "but", "or", "nor", "not", "so", "if", "than", "too",
        "very", "just", "i", "me", "my", "we", "our",
    }

    words = re.findall(r'\b\w+\b', text.lower())
    terms = [w for w in words if w not in stop_words and len(w) > 2]

    for entity_list in entities.values():
        terms.extend([e.lower() for e in entity_list])

    seen = set()
    unique = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique
