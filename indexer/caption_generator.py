"""Generates detailed fashion+context captions using Moondream VLM (API)."""
import os
import moondream as md
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MOONDREAM_API_KEY")
model = md.vl(api_key=api_key)

CAPTION_PROMPT = CAPTION_PROMPT = """Describe this image in detail for an open-vocabulary semantic search engine. Write a single, cohesive, natural-flowing paragraph covering:

1. **Subjects & Actions:** Detail the primary subjects, objects, people, and any actions or events taking place in the image.
2. **Colors & Textures:** Describe the prominent colors, materials, patterns, and visual details of the main subjects.
3. **Setting & Location:** Describe the environment, background, location, lighting, and weather conditions (e.g. 'cozy modern living room', 'sunny beach', 'rainy city street at night', 'mountain trail').
4. **Style & Vibe:** Capture the overall mood, aesthetic style, or context of the scene (e.g. 'minimalist', 'vibrant', 'adventurous', 'professional').

Do NOT describe facial expressions, poses, or specific physical details of people unless relevant to the action. Keep it highly descriptive and natural."""



def generate_caption(image_path: str) -> str:
    image = Image.open(image_path)
    result = model.query(image, CAPTION_PROMPT)
    return result["answer"].strip()