import os
import sys
import time
from PIL import Image
from dotenv import load_dotenv

# Incase of HF outage we use cached embedder and fashion-clip
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Add parent directory to path so we can import config and indexer modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
config.TEXT_EMBED_MODEL="all-MiniLM-L6-v2"

# Override configuration directories for the test
config.IMAGES_DIR = r"\test_images"
config.CHROMA_DIR = r"\test_chroma_dir"
config.COLLECTION_NAME = "test_fashion_captions"

from indexer.caption_generator import generate_caption
from indexer.embedder import embed_caption, embed_image_clip
from indexer.vector_store import VectorStore

def run_test_indexer():
    print("Initializing VectorStore (test Chroma DB)...")
    store = VectorStore()
    
    # Clean up the collection if it exists to ensure a fresh test
    try:
        store.client.delete_collection(config.COLLECTION_NAME)
        store.collection = store.client.create_collection(config.COLLECTION_NAME)
        print("Cleared previous test collection.")
    except Exception:
        pass

    # Ensure test directory exists
    if not os.path.exists(config.IMAGES_DIR):
        print(f"Error: {config.IMAGES_DIR} directory not found.")
        return

    # List image files in the test directory
    image_files = [f for f in os.listdir(config.IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in {config.IMAGES_DIR}: {image_files}")

    if not image_files:
        print("No images found to test indexing.")
        return

    for img in image_files:
        image_path = os.path.join(config.IMAGES_DIR, img)
        image_id = os.path.splitext(img)[0]
        
        print(f"\n--- Indexing {img} ---")
        
        # 1. Generate Caption
        #print("Step 1: Generating caption via Moondream VLM...")
        try:
            caption = generate_caption(image_path)
            #print(f"  > Caption: {caption}")
        except Exception as e:
            print(f"  > Error generating caption: {e}")
            continue

        # 2. Embed Caption
        #print("Step 2: Generating text embedding for caption...")
        try:
            caption_embed = embed_caption(caption)
            #print(f"  > Caption embed size: {len(caption_embed)}")
        except Exception as e:
            print(f"  > Error embedding caption: {e}")
            continue

        # 3. Embed Image
        #print("Step 3: Generating FashionCLIP image embedding...")
        try:
            image_embed = embed_image_clip(image_path)
            #print(f"  > Image embed size: {len(image_embed)}")
        except Exception as e:
            print(f"  > Error embedding image via CLIP: {e}")
            continue

        # 4. Store in ChromaDB
        #print("Step 4: Storing in ChromaDB...")
        try:
            store.add(image_id, image_path, caption, caption_embed, image_embed)
            print("  > Stored successfully!")
        except Exception as e:
            print(f"  > Error storing in ChromaDB: {e}")
            continue
            
        time.sleep(0.5)

    print("\n=================================")
    print("Testing collection querying...")
    try:
        results = store.collection.get()
        print(f"Successfully indexed {len(results['ids'])} items in the database:")
        for idx, item_id in enumerate(results['ids']):
            print(f" - {item_id}: {results['documents'][idx][:60]}...")
    except Exception as e:
        print(f"Error reading from test collection: {e}")

if __name__ == "__main__":
    load_dotenv()
    run_test_indexer()
