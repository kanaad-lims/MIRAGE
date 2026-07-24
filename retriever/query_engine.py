import ast
import numpy as np
from config import TOP_N_CANDIDATES, CAPTION_WEIGHT, CLIP_WEIGHT
from indexer.embedder import embed_caption, embed_text_clip
from indexer.vector_store import VectorStore

def _cosine(a, b):
    # Computing cosine similarity
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class QueryEngine:
    def __init__(self):
        self.store = VectorStore()
    
    def search(self, query: str, k: int = 5):
            query_caption_emb = embed_caption(query)
            query_clip_emb = embed_text_clip(query)

            results = self.store.query(query_caption_emb, n_results=TOP_N_CANDIDATES)

            candidates = []
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for i in range(len(ids)):
                caption_score = 1 - (distances[i] / 2)  # chroma returns distance; convert to similarity
                clip_image_emb = ast.literal_eval(metas[i]["clip_embedding"])
                clip_score = _cosine(query_clip_emb, clip_image_emb)

                final_score = CAPTION_WEIGHT * caption_score + CLIP_WEIGHT * clip_score
                candidates.append({
                    "id": ids[i],
                    "caption": docs[i],
                    "image_path": metas[i]["image_path"],
                    "score": final_score,
                })

            candidates.sort(key=lambda x: x["score"], reverse=True)
            return candidates[:k]