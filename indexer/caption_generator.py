"""Generates detailed fashion+context captions using Moondream VLM (API)."""
import os
import moondream as md
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MOONDREAM_API_KEY")
model = md.vl(api_key=api_key)

CAPTION_PROMPT = """Describe this image in detail for a fashion search engine. Write a single, cohesive, natural-flowing paragraph covering:
1. **Garments & Colors:** Detail every visible garment, accessory, and shoe, including their exact color, pattern, and texture (e.g. 'navy blue blazer', 'white button-down shirt', 'grey pinstripe trousers'). Ensure colors are explicitly bound to their nouns.
2. **Occasion & Appropriateness (The Vibe):** Describe the overall style/vibe and explicitly list the occasions or locations where this outfit is appropriate (e.g. 'suitable for a party', 'appropriate for a casual beach walk', 'perfect for a night out', 'good for a gym workout').
3. **Formalwear Rule:** If the outfit contains a blazer, suit jacket, tie, dress shirt, slacks, pencil skirt, or formal dress shoes, you MUST explicitly label the style/vibe as 'professional wear', 'office wear', or 'corporate business formal'.
4. **Setting:** Describe the background environment (e.g. 'modern office', 'city street', 'outdoor park', 'corporate lobby').

Do NOT describe the model's pose, facial expressions, or physical features. Use specific, rich fashion terminology."""


def generate_caption(image_path: str) -> str:
    image = Image.open(image_path)
    result = model.query(image, CAPTION_PROMPT)
    return result["answer"].strip()