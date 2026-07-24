import os
import moondream as md
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Retrieve the key from the environment variables loaded by load_dotenv()
api_key = os.getenv("MOONDREAM_API_KEY")
model = md.vl(api_key=api_key)

image = Image.open(r"/image_path")

prompt = """Describe this image for a fashion search engine. Include, in one flowing paragraph:
1. Every visible garment and its exact color (e.g. "red tie", "white button-down shirt").
2. The style/vibe of the outfit (formal, casual, streetwear, business, athleisure, etc.).
3. The setting/environment (office, park, street, home, studio, etc.).
Do NOT describe the model's pose or facial expression(eg smile, frown etc).
Be specific and use natural descriptive language, not a bullet list."""

result = model.query(image, prompt)
print(result["answer"])
