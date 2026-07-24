"""
Query Engine for MIRAGE (Multimodal Image Retrieval & Attribute-Grounded Engine).

This module implements a Dual-Stage Search Pipeline:
1. Stage 1 (Text Filtering): Coarse-filters the image database using semantic 
   captions with SentenceTransformers (all-MiniLM-L6-v2) and ChromaDB.
2. Stage 2 (Visual Grounding): Reranks the top-N candidates using standard 
   CLIP (clip-vit-base-patch32) by computing visual-text similarity scores.
3. Score Fusion: Combines the text and visual scores using a weighted sum 
   defined in the configurations.
"""

import ast
import numpy as np
from config import TOP_N_CANDIDATES, CAPTION_WEIGHT, CLIP_WEIGHT
from indexer.embedder import embed_caption, embed_query_text
from indexer.vector_store import VectorStore

def _cosine(a, b):
    # Computing cosine similarity
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class QueryEngine:
    def __init__(self):
        self.store = VectorStore()
    
    def search_images(self, query: str, k: int = 5):
            query_caption_emb = embed_caption(query) # Represents search query in its semantic form
            query_text_clip_emb = embed_query_text(query) # Represents search query in CLIP space 

            # 1. First stage filtering (we get top_n (20) candidates after this)
            results = self.store.query(query_caption_emb, n_results=TOP_N_CANDIDATES)

            candidates = []
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            # 2. Applying reranking on top_n images
            # cosine -> Image embeddings and CLIP text embeddings (not semantic query embedding)
            for i in range(len(ids)):
                caption_score = 1 - (distances[i] / 2)  # chroma returns distance; convert to similarity
                clip_image_emb = ast.literal_eval(metas[i]["clip_embedding"])
                clip_score = _cosine(query_text_clip_emb, clip_image_emb)

                final_score = CAPTION_WEIGHT * caption_score + CLIP_WEIGHT * clip_score
                candidates.append({
                    "id": ids[i],
                    "caption": docs[i],
                    "image_path": metas[i]["image_path"],
                    "score": final_score,
                })

            candidates.sort(key=lambda x: x["score"], reverse=True)
            return candidates[:k]