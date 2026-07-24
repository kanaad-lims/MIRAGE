import os
from dotenv import load_dotenv

load_dotenv()

MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY")
TEXT_EMBED_MODEL = "all-MiniLM-L6-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
COLLECTION_NAME = "mirage_image_collection"

TOP_N_CANDIDATES = 20   # candidates pulled before rerank
CAPTION_WEIGHT = 0.4
CLIP_WEIGHT = 0.7