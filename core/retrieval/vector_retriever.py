from typing import List, Dict, Optional, Set

import chromadb

from config import CHROMA_DIR, CHROMA_COLLECTION


class VectorRetriever:
    def __init__(self):
        self._documents: List[Dict] = []
        self._collection = None
        self._client = None

    def build_index(self, documents: List[Dict]) -> None:
        self._documents = documents
        self._client = chromadb.Client()

        try:
            self._client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass

        self._collection = self._client.create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self._collection.add(
                ids=[doc["chunk_id"] for doc in batch],
                documents=[doc["text"] for doc in batch],
                metadatas=[{k: str(v) for k, v in doc.get("metadata", {}).items()} for doc in batch],
            )

    def retrieve(self, query: str, top_k: int = 10, allow_list: Optional[Set[str]] = None) -> List[Dict]:
        if not self._collection:
            return []

        fetch_k = top_k * 3 if allow_list else top_k

        results = self._collection.query(
            query_texts=[query],
            n_results=min(fetch_k, self._collection.count()),
        )

        scored_docs = []
        if results and results["ids"]:
            for i, chunk_id in enumerate(results["ids"][0]):
                if allow_list and chunk_id not in allow_list:
                    continue

                distance = results["distances"][0][i] if results.get("distances") else 0
                similarity = 1.0 - distance

                scored_docs.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "vector_score": float(similarity),
                })

        scored_docs.sort(key=lambda x: x["vector_score"], reverse=True)
        return scored_docs[:top_k]
