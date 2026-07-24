"""
Contains all the embedding functions used in indexer and in retriever (for embedding the query)
"""

import os
# Force Hugging Face to run in offline mode using local cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from config import TEXT_EMBED_MODEL, FASHION_CLIP_MODEL

_text_model = SentenceTransformer(TEXT_EMBED_MODEL)
_clip_model = CLIPModel.from_pretrained(FASHION_CLIP_MODEL)
_clip_processor = CLIPProcessor.from_pretrained(FASHION_CLIP_MODEL)

def embed_caption(text: str) -> list[float]:
    return _text_model.encode(text, normalize_embeddings=True).tolist()

def embed_image_clip(image_path: str) -> list[float]:
    img = Image.open(image_path).convert("RGB")
    inputs = _clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        feats = _clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze().tolist()

def embed_text_clip(text: str) -> list[float]:
    inputs = _clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        feats = _clip_model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze().tolist()