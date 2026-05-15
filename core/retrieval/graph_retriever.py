from typing import List, Dict, Set, Optional

from core.knowledge_graph import KnowledgeGraph


class GraphRetriever:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph

    def get_allow_list(
        self,
        query: str,
        *,
        min_confidence: float = 0.0,
    ) -> Set[str]:
        return self.kg.get_allow_list(query, min_confidence=min_confidence)

    def get_edge_priors(self, chunk_ids: Set[str]) -> Dict[str, float]:
        return self.kg.get_edge_priors(chunk_ids)

    def retrieve_by_entity(self, query: str, top_k: int = 10) -> List[Dict]:
        entities = self.kg._match_query_to_entities(query)
        chunk_scores: Dict[str, float] = {}

        for entity_id in entities:
            if entity_id not in self.kg.graph:
                continue
            node_data = self.kg.graph.nodes[entity_id]
            for cid in node_data.get("chunk_ids", set()):
                chunk_scores[cid] = chunk_scores.get(cid, 0) + 1.0

            for neighbor in self.kg.graph.neighbors(entity_id):
                edge_data = self.kg.graph[entity_id][neighbor]
                weight = edge_data.get("weight", 1)
                for cid in self.kg.graph.nodes[neighbor].get("chunk_ids", set()):
                    chunk_scores[cid] = chunk_scores.get(cid, 0) + 0.5 * weight

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"chunk_id": cid, "graph_score": score} for cid, score in sorted_chunks[:top_k]]
