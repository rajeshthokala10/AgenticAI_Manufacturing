import re
import json
import pickle
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

import networkx as nx

from config import GRAPH_PATH, PROCESSED_DIR, DOMAIN_ONTOLOGY


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._entity_index: Dict[str, Set[str]] = {}

    def build_from_documents(self, documents: List[Dict]) -> None:
        for doc in documents:
            self._extract_and_add_entities(doc)
        self._build_entity_index()
        self._compute_edge_priors(documents)
        self.save()

    def _extract_and_add_entities(self, doc: Dict) -> None:
        text = doc["text"]
        chunk_id = doc["chunk_id"]
        meta = doc.get("metadata", {})

        equipment_ids = meta.get("equipment_ids", [])
        alarm_codes = meta.get("alarm_codes", [])
        part_numbers = meta.get("part_numbers", [])
        fault_codes = meta.get("fault_codes", [])

        for eq_id in equipment_ids:
            self._add_entity(eq_id, "Equipment", chunk_id)

        for alarm in alarm_codes:
            self._add_entity(alarm, "Alarm", chunk_id)

        for part in part_numbers:
            self._add_entity(part, "SparePart", chunk_id)

        for fc in fault_codes:
            self._add_entity(fc, "FailureMode", chunk_id)

        symptoms = self._extract_symptoms(text)
        for symptom in symptoms:
            sym_id = f"SYM:{symptom[:50]}"
            self._add_entity(sym_id, "Symptom", chunk_id)
            for eq_id in equipment_ids:
                self._add_relation(eq_id, sym_id, "HAS_SYMPTOM", chunk_id)

        procedures = self._extract_procedures(text)
        for proc in procedures:
            proc_id = f"PROC:{proc[:50]}"
            self._add_entity(proc_id, "Procedure", chunk_id)

        for eq_id in equipment_ids:
            for alarm in alarm_codes:
                self._add_relation(eq_id, alarm, "TRIGGERS_ALARM", chunk_id)
            for part in part_numbers:
                self._add_relation(eq_id, part, "REQUIRES_PART", chunk_id)
            for fc in fault_codes:
                self._add_relation(eq_id, fc, "CAUSES_FAILURE", chunk_id)

        for alarm in alarm_codes:
            for fc in fault_codes:
                self._add_relation(alarm, fc, "CAUSES_FAILURE", chunk_id)

        components = self._extract_components(text)
        for comp in components:
            comp_id = f"COMP:{comp}"
            self._add_entity(comp_id, "Component", chunk_id)
            for eq_id in equipment_ids:
                self._add_relation(eq_id, comp_id, "HAS_COMPONENT", chunk_id)

    def _extract_symptoms(self, text: str) -> List[str]:
        patterns = [
            r'(?:symptom|indication|sign|observed|detected|reported)[:\s]+([^.]+)',
            r'(?:high|low|excessive|abnormal|unexpected)\s+\w+(?:\s+\w+){0,3}',
        ]
        symptoms = []
        for pat in patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            symptoms.extend(matches[:3])
        return symptoms

    def _extract_procedures(self, text: str) -> List[str]:
        patterns = [
            r'(?:procedure|step|action|resolution)[:\s]+([^.]+)',
            r'(?:replace|inspect|check|verify|adjust|clean|lubricate|tighten)\s+(?:the\s+)?(\w+(?:\s+\w+){0,4})',
        ]
        procedures = []
        for pat in patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches[:3]:
                procedures.append(m if isinstance(m, str) else m[0] if m else "")
        return [p for p in procedures if p]

    def _extract_components(self, text: str) -> List[str]:
        component_keywords = [
            "impeller", "seal", "bearing", "coupling", "motor", "shaft",
            "belt", "roller", "tensioner", "sensor", "vfd", "gearbox",
            "cylinder", "valve", "accumulator", "platen", "pump",
            "filter", "gasket", "o-ring", "bushing", "encoder"
        ]
        found = []
        text_lower = text.lower()
        for kw in component_keywords:
            if kw in text_lower:
                found.append(kw)
        return found

    def _add_entity(self, entity_id: str, entity_type: str, chunk_id: str) -> None:
        if self.graph.has_node(entity_id):
            self.graph.nodes[entity_id]["chunk_ids"].add(chunk_id)
        else:
            self.graph.add_node(
                entity_id,
                entity_type=entity_type,
                chunk_ids={chunk_id},
            )

    def _add_relation(self, source: str, target: str, relation: str, chunk_id: str) -> None:
        if self.graph.has_edge(source, target):
            self.graph[source][target]["chunk_ids"].add(chunk_id)
            self.graph[source][target]["weight"] += 1
        else:
            self.graph.add_edge(
                source, target,
                relation=relation,
                chunk_ids={chunk_id},
                weight=1,
            )

    def _build_entity_index(self) -> None:
        self._entity_index = {}
        for node, data in self.graph.nodes(data=True):
            etype = data.get("entity_type", "Unknown")
            if etype not in self._entity_index:
                self._entity_index[etype] = set()
            self._entity_index[etype].add(node)

    def _compute_edge_priors(self, documents: List[Dict]) -> None:
        total_docs = len(documents)
        for u, v, data in self.graph.edges(data=True):
            co_occurrence = len(data.get("chunk_ids", set()))
            data["prior"] = min(co_occurrence / max(total_docs * 0.01, 1), 1.0)

    def get_allow_list(self, query: str) -> Set[str]:
        query_entities = self._match_query_to_entities(query)
        allowed_chunks: Set[str] = set()

        for entity_id in query_entities:
            if entity_id in self.graph:
                allowed_chunks.update(self.graph.nodes[entity_id].get("chunk_ids", set()))
                for neighbor in self.graph.neighbors(entity_id):
                    allowed_chunks.update(self.graph.nodes[neighbor].get("chunk_ids", set()))
                    for second_hop in self.graph.neighbors(neighbor):
                        allowed_chunks.update(self.graph.nodes[second_hop].get("chunk_ids", set()))

        for entity_id in query_entities:
            if entity_id in self.graph:
                for predecessor in self.graph.predecessors(entity_id):
                    allowed_chunks.update(self.graph.nodes[predecessor].get("chunk_ids", set()))

        return allowed_chunks

    def get_edge_priors(self, chunk_ids: Set[str]) -> Dict[str, float]:
        priors = {}
        for u, v, data in self.graph.edges(data=True):
            edge_chunks = data.get("chunk_ids", set())
            overlap = edge_chunks & chunk_ids
            if overlap:
                prior = data.get("prior", 0.0)
                for cid in overlap:
                    priors[cid] = max(priors.get(cid, 0.0), prior)
        return priors

    def get_subgraph_for_query(self, query: str) -> Dict:
        entities = self._match_query_to_entities(query)
        nodes = []
        edges = []

        visited = set()
        for eid in entities:
            if eid in self.graph and eid not in visited:
                visited.add(eid)
                node_data = self.graph.nodes[eid]
                nodes.append({
                    "id": eid,
                    "type": node_data.get("entity_type", "Unknown"),
                    "chunks": len(node_data.get("chunk_ids", set()))
                })
                for neighbor in list(self.graph.neighbors(eid)) + list(self.graph.predecessors(eid)):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        n_data = self.graph.nodes[neighbor]
                        nodes.append({
                            "id": neighbor,
                            "type": n_data.get("entity_type", "Unknown"),
                            "chunks": len(n_data.get("chunk_ids", set()))
                        })

                for neighbor in self.graph.neighbors(eid):
                    edge_data = self.graph[eid][neighbor]
                    edges.append({
                        "source": eid,
                        "target": neighbor,
                        "relation": edge_data.get("relation", "RELATED"),
                        "weight": edge_data.get("weight", 1)
                    })
                for predecessor in self.graph.predecessors(eid):
                    edge_data = self.graph[predecessor][eid]
                    edges.append({
                        "source": predecessor,
                        "target": eid,
                        "relation": edge_data.get("relation", "RELATED"),
                        "weight": edge_data.get("weight", 1)
                    })

        return {"nodes": nodes, "edges": edges}

    def _match_query_to_entities(self, query: str) -> Set[str]:
        matched = set()
        equipment_ids = re.findall(r'(?:P-\d{3}|CV-\d{3}|HP-\d{3})', query)
        alarm_codes = re.findall(r'ALM-[A-Z]\d{3}', query)
        part_numbers = re.findall(r'SP-\d{4}', query)
        fault_codes = re.findall(r'FC-\d{3}', query)

        matched.update(equipment_ids)
        matched.update(alarm_codes)
        matched.update(part_numbers)
        matched.update(fault_codes)

        query_lower = query.lower()
        symptom_keywords = {
            "vibration": "high vibration", "leak": "seal leakage",
            "overheat": "overheating", "cavitation": "cavitation",
            "noise": "abnormal noise", "pressure": "pressure",
            "tracking": "belt tracking", "speed": "speed deviation",
            "temperature": "high temperature", "flow": "low flow",
        }
        for keyword, symptom in symptom_keywords.items():
            if keyword in query_lower:
                for node in self.graph.nodes():
                    if node.startswith("SYM:") and keyword in node.lower():
                        matched.add(node)

        component_keywords = [
            "impeller", "seal", "bearing", "coupling", "motor",
            "belt", "roller", "tensioner", "sensor", "vfd",
            "cylinder", "valve", "accumulator", "platen", "pump",
            "filter", "encoder", "gearbox"
        ]
        for comp in component_keywords:
            if comp in query_lower:
                comp_id = f"COMP:{comp}"
                if comp_id in self.graph:
                    matched.add(comp_id)

        return matched

    def get_stats(self) -> Dict:
        type_counts = {}
        for _, data in self.graph.nodes(data=True):
            etype = data.get("entity_type", "Unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1

        rel_counts = {}
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relation", "RELATED")
            rel_counts[rel] = rel_counts.get(rel, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "entity_types": type_counts,
            "relation_types": rel_counts,
        }

    def save(self) -> None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        save_data = {
            "nodes": {},
            "edges": []
        }
        for node, data in self.graph.nodes(data=True):
            save_data["nodes"][node] = {
                "entity_type": data.get("entity_type", "Unknown"),
                "chunk_ids": list(data.get("chunk_ids", set()))
            }
        for u, v, data in self.graph.edges(data=True):
            save_data["edges"].append({
                "source": u,
                "target": v,
                "relation": data.get("relation", "RELATED"),
                "weight": data.get("weight", 1),
                "prior": data.get("prior", 0.0),
                "chunk_ids": list(data.get("chunk_ids", set()))
            })
        with open(GRAPH_PATH, "w") as f:
            json.dump(save_data, f, indent=2)

    def load(self) -> bool:
        if not Path(GRAPH_PATH).exists():
            return False
        with open(GRAPH_PATH, "r") as f:
            save_data = json.load(f)
        self.graph = nx.DiGraph()
        for node_id, data in save_data["nodes"].items():
            self.graph.add_node(
                node_id,
                entity_type=data["entity_type"],
                chunk_ids=set(data["chunk_ids"])
            )
        for edge in save_data["edges"]:
            self.graph.add_edge(
                edge["source"], edge["target"],
                relation=edge["relation"],
                weight=edge["weight"],
                prior=edge.get("prior", 0.0),
                chunk_ids=set(edge["chunk_ids"])
            )
        self._build_entity_index()
        return True
