"""Thin wrapper around ChromaDB for storing caption + clip embeddings."""
import chromadb
from config import CHROMA_DIR, COLLECTION_NAME

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)

    def add(self, image_id: str, image_path: str, caption: str, caption_embedding: list,
            clip_embedding: list):
        self.collection.add(
            ids=[image_id],
            embeddings=[caption_embedding],
            documents=[caption],
            metadatas=[{"image_path": image_path, "clip_embedding": str(clip_embedding)}],
        )

    def query(self, query_embedding: list, n_results: int):
        return self.collection.query(query_embeddings=[query_embedding], n_results=n_results)

    def get_existing_ids(self) -> set:
        results = self.collection.get(include=[])
        return set(results["ids"])
    