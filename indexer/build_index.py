# this is the main pipeline for building indexes for the images in the dataset.

import os
import time
from tqdm import tqdm
from config import IMAGES_DIR
from indexer.caption_generator import generate_caption
from indexer.embedder import embed_caption, embed_image_clip
from indexer.vector_store import VectorStore

def build_index():
    store = VectorStore()
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Number of image files to be indexed: {len(image_files)}")

    existing_ids = store.get_existing_ids()
    print(f"Found {len(existing_ids)} already indexed images in database.")

    for img in tqdm(image_files, desc="Indexing images"):
        image_path = os.path.join(IMAGES_DIR, img)
        image_id = os.path.splitext(img)[0]

        if image_id in existing_ids:
            continue

        try:
            caption = generate_caption(image_path=image_path)
        except Exception as e:
            print(f"Error generating caption for {img}: {e}")
            continue

        caption_embedding = embed_caption(caption)
        image_embedding = embed_image_clip(image_path=image_path)

        store.add(image_id, image_path, caption, caption_embedding, image_embedding)
        time.sleep(0.5)
    
    print("Indexing complete.")

if __name__ == "__main__":
    build_index()