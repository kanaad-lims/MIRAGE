# build_index.py - Ingests images, generates captions/embeddings, and saves to ChromaDB.

import os
import time
from tqdm import tqdm
from config import IMAGES_DIR, CHROMA_DIR
from indexer.caption_generator import generate_caption
from indexer.embedder import embed_caption, embed_image_clip
from indexer.vector_store import VectorStore

def build_index():
    # Auto-create data directories if they don't exist
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Friendly check for empty dataset directory
    if not image_files:
        print("\n" + "="*80)
        print(" Welcome to Mirage!")
        print("="*80)
        print(f"Created dataset folder at: {IMAGES_DIR}")
        print("Please place some images (.jpg, .png, .jpeg) in that folder and run this script again.")
        print("="*80 + "\n")
        return

    store = VectorStore()
    print(f"Total images found in folder: {len(image_files)}")

    existing_ids = store.get_existing_ids()
    print(f"Found {len(existing_ids)} already indexed images in database.")

    # Filter to process only new images (skip/resume)
    to_index = [img for img in image_files if os.path.splitext(img)[0] not in existing_ids]
    print(f"Indexing {len(to_index)} new images...")

    if not to_index:
        print("Database is up to date. No new images to index.")
        return

    for img in tqdm(to_index, desc="Indexing images"):
        image_path = os.path.join(IMAGES_DIR, img)
        image_id = os.path.splitext(img)[0]

        try:
            # 1. Generate text caption via VLM
            caption = generate_caption(image_path=image_path)
        except Exception as e:
            print(f"Error generating caption for {img}: {e}")
            continue

        # 2. Extract embeddings
        caption_embedding = embed_caption(caption)
        image_embedding = embed_image_clip(image_path=image_path)

        # 3. Store in Vector Database
        store.add(image_id, image_path, caption, caption_embedding, image_embedding)
        time.sleep(0.5)  # Rate limiting safety sleep
    
    print("Indexing complete.")

if __name__ == "__main__":
    build_index()
