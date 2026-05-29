import pickle
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    def __init__(self, kb_dir: str = "./knowledge_base"):
        kb_path = Path(kb_dir)

        with open(kb_path / "config.json") as f:
            config = json.load(f)

        self.model = SentenceTransformer(config["embed_model"])
        self.index = faiss.read_index(str(kb_path / "index.faiss"))

        with open(kb_path / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            results.append({
                "text": doc["text"],
                "meta": doc.get("meta", {}),
                "source": doc.get("source", ""),
                "score": float(score),
            })

        return results

    def format_context(self, results: list[dict]) -> str:
        return "\n\n".join(
            f"[Source {i+1}] {r['text']}"
            for i, r in enumerate(results)
        )
