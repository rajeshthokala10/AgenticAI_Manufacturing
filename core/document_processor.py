import os
import re
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

from config import PDF_DIR, EXCEL_DIR, PROCESSED_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def generate_chunk_id(source: str, index: int) -> str:
    raw = f"{source}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    sentences = re.split(r'(?<=[.!?\n])\s+', text)
    chunks, current, current_len = [], [], 0

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_words = []
            total = 0
            for s in reversed(current):
                s_words = s.split()
                if total + len(s_words) > overlap:
                    break
                overlap_words.insert(0, s)
                total += len(s_words)
            current = overlap_words
            current_len = total
        current.append(sentence)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


def extract_section_metadata(text: str) -> Dict[str, str]:
    metadata = {}
    equipment_pattern = r'(?:P-\d{3}|CV-\d{3}|HP-\d{3})'
    alarm_pattern = r'ALM-[A-Z]\d{3}'
    part_pattern = r'SP-\d{4}'
    fault_pattern = r'FC-\d{3}'

    equipment = re.findall(equipment_pattern, text)
    alarms = re.findall(alarm_pattern, text)
    parts = re.findall(part_pattern, text)
    faults = re.findall(fault_pattern, text)

    if equipment:
        metadata["equipment_ids"] = list(set(equipment))
    if alarms:
        metadata["alarm_codes"] = list(set(alarms))
    if parts:
        metadata["part_numbers"] = list(set(parts))
    if faults:
        metadata["fault_codes"] = list(set(faults))

    section_match = re.search(r'^#+\s*(.+)', text, re.MULTILINE)
    if section_match:
        metadata["section_title"] = section_match.group(1).strip()

    return metadata


def process_text_file(filepath: Path) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    source_name = filepath.stem
    chunks = chunk_text(content)
    documents = []

    for i, chunk in enumerate(chunks):
        chunk_id = generate_chunk_id(source_name, i)
        metadata = extract_section_metadata(chunk)
        metadata.update({
            "source": source_name,
            "source_file": filepath.name,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "doc_type": "manual"
        })
        documents.append({
            "chunk_id": chunk_id,
            "text": chunk,
            "metadata": metadata
        })

    return documents


def process_excel_file(filepath: Path) -> List[Dict]:
    df = pd.read_excel(filepath, engine="openpyxl")
    source_name = filepath.stem
    documents = []

    for i, row in df.iterrows():
        row_text_parts = []
        row_meta = {"source": source_name, "source_file": filepath.name, "row_index": i}

        for col in df.columns:
            val = row[col]
            if pd.notna(val):
                row_text_parts.append(f"{col}: {val}")
                if isinstance(val, str):
                    extracted = extract_section_metadata(val)
                    for k, v in extracted.items():
                        existing = row_meta.get(k, [])
                        if isinstance(existing, list) and isinstance(v, list):
                            row_meta[k] = list(set(existing + v))
                        else:
                            row_meta[k] = v

        row_text = " | ".join(row_text_parts)
        chunk_id = generate_chunk_id(source_name, i)

        doc_type_map = {
            "work_orders": "work_order",
            "alarm_history": "alarm_event",
            "spare_parts_inventory": "spare_part",
        }
        row_meta["doc_type"] = doc_type_map.get(source_name, "tabular")

        documents.append({
            "chunk_id": chunk_id,
            "text": row_text,
            "metadata": row_meta
        })

    return documents


def process_all_documents() -> List[Dict]:
    all_docs = []

    for txt_file in sorted(PDF_DIR.glob("*.txt")):
        docs = process_text_file(txt_file)
        all_docs.extend(docs)
        print(f"  Processed {txt_file.name}: {len(docs)} chunks")

    for xlsx_file in sorted(EXCEL_DIR.glob("*.xlsx")):
        docs = process_excel_file(xlsx_file)
        all_docs.extend(docs)
        print(f"  Processed {xlsx_file.name}: {len(docs)} records")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "all_chunks.json"
    with open(output_path, "w") as f:
        json.dump(all_docs, f, indent=2, default=str)
    print(f"  Total documents: {len(all_docs)} saved to {output_path}")

    return all_docs
