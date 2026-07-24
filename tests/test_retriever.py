import os
import sys
from dotenv import load_dotenv

# Incase of HF downtime, use cached models (embedder and fashion-clip).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Add parent directory to path so we can import config and retriever modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
config.TEXT_EMBED_MODEL="all-MiniLM-L6-v2"

# Override configuration directories to use the test collection
config.IMAGES_DIR = r"D:\Kanaad\Glance-Fashion\test_images"
config.CHROMA_DIR = r"data/test_chroma_db"
config.COLLECTION_NAME = "test_fashion_captions"

from retriever.query_engine import QueryEngine

def run_test_retriever():
    engine = QueryEngine()
    
    # Test queries designed to test different features of our index
    test_queries = [
        "a woman in a light brown trench coat and black boots",
        "a man in a gray overcoat and white trousers on a city street",
        "gold oversized coat and pleated skirt",
        "knit blue sweater and short denim skirt",
        "all black outfit with wide-leg pants"
    ]

    single_test_query = "red shoes"

    # print("=== RUNNING RETRIEVER TESTS ===\n")
    # for q in test_queries:
    #     print(f"Query: '{q}'")
    #     print("-" * 50)
    #     try:
    #         results = engine.search(q, k=3)
    #         for rank, r in enumerate(results, 1):
    #             print(f"  {rank}. {r['image_path']} (score: {r['score']:.4f})")
    #             print(f"     Caption: {r['caption']}\n")
    #     except Exception as e:
    #         print(f"  Error running query: {e}\n")
    #     print("=" * 60 + "\n")
    
    # print("\n" + "="*60)
    print(f"SINGLE QUERY TEST: '{single_test_query}'")
    print("="*60)
    try:
        results = engine.search(single_test_query, k=3)
        for rank, r in enumerate(results, 1):
            filename = os.path.basename(r['image_path'])
            print(f"\n[Rank {rank}] Filename: {filename}")
            print(f"  Score:   {r['score']:.4f}")
            print(f"  Path:    {r['image_path']}")
            print(f"  Caption: {r['caption']}")
            print("-" * 60)
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    load_dotenv()
    run_test_retriever()
