import os
from dotenv import load_dotenv

load_dotenv()

MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY")
TEXT_EMBED_MODEL = "all-MiniLM-L6-v2"
FASHION_CLIP_MODEL = "patrickjohncyh/fashion-clip"

IMAGES_DIR = r"\images_directory"
CHROMA_DIR = r"\chromadb"
COLLECTION_NAME = "image_captions"

TOP_N_CANDIDATES = 20   # candidates pulled before rerank
CAPTION_WEIGHT = 0.4
CLIP_WEIGHT = 0.7