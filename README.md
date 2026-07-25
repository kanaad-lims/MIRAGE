
# 🌌 MIRAGE: Multimodal Image Retrieval & Attribute-Grounded Engine

MIRAGE is an intelligent, open-vocabulary zero-shot image search engine built to retrieve specific images from a database using natural language queries. The system successfully solves the **compositional binding problem** (e.g., distinguishing "a red cup and a blue plate" from "a blue cup and a red plate") and matches visual assets based on objects, colors, locations, styling, and abstract vibes.

---

## 🚀 Key Features
*   **Compositional Attribute Binding:** Uses a Vision-Language Model (VLM) at indexing time to correctly associate colors, textures, and patterns with their specific objects/subjects.
*   **Context & Location Awareness:** Successfully filters and ranks images based on background environments, lighting, and weather conditions (e.g. *"rainy city street at night"*, *"sunny beach"*).
*   **Abstract Style & Vibe Inference:** Retrieves image matches based on mood or style (e.g. *"cozy living room"*, *"minimalist architecture"*).
*   **Domain-Agnostic Retrieval:** Grounded on general-purpose models, allowing you to index and search any collection of images (landscapes, products, vehicles, fashion, etc.).
*   **Domain-Specific Visual Reranking:** Leverages **OpenAI CLIP** to visually ground the candidates retrieved by the text search engine.
*   **Interactive CLI with Visual Grid Output:** Search directly from the command line and view the top retrieved images side-by-side in a horizontal grid window.

---

## 📐 System Architecture

MIRAGE implements a dual-stage retrieval pipeline:

```
[User Query]
    │
    ├──► (Text Encoder) ──► Caption Search (ChromaDB) ──► Top-N Candidates (Filter)
    │                                                          │
    └──► (CLIP Encoder) ──► standard CLIP Rerank (Visuals) ◄───┘
                                   │
                                   ▼
                           [Weighted Fusion] ──► Top-K Results (Grid Display)
```

1.  **Part A — The Indexer (Offline):** Generates detailed captions for each image using the Moondream VLM, embeds them using `all-MiniLM-L6-v2`, extracts visual feature vectors using standard `CLIP`, and stores everything in a persistent `ChromaDB` collection.
2.  **Part B — The Retriever (Online):** Computes text embeddings for the query, retrieves the top-N candidates from ChromaDB, reranks them using CLIP visual-text similarity, fuses the scores, and displays the top-k images in a horizontal grid.

---

## 🛠️ Installation & Setup

### 1. Clone the repository and navigate inside:
```bash
git clone https://github.com/kanaad-lims/mirage.git
cd mirage
```

### 2. Set up your environment and install dependencies:
Ensure you have PyTorch and Python 3.10+ installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configure your environment variables:
Create a `.env` file in the root directory and add your Moondream API Key:
```env
MOONDREAM_API_KEY=your_moondream_api_key_here
```

---

## 💻 How to Run

### 1. Build the Index (Indexer)
Place any set of images you want to search (e.g. products, travel photos, landscapes) inside the `data/images/` folder. Run the indexer to process and store them in ChromaDB. The indexer includes automatic **skip/resume** logic in case it is interrupted:
```bash
python -m indexer.build_index
```
*(If the folders do not exist, the script will automatically initialize them for you on the first run).*

### 2. Search the Database (Retriever)
You can search the database using natural language queries. 

**Option A (Interactive Prompt):**
```bash
python -m retriever.search
```
*It will prompt you: `Enter query: `*

**Option B (One-liner CLI Argument):**
```bash
python -m retriever.search a red car parked on a sunny city street
```

---

## 📂 Project Structure
```
├── data/
│   ├── images/               # Raw image dataset (ignored by git)
│   └── chroma_db/            # Persistent ChromaDB database files (ignored by git)
├── indexer/
│   ├── caption_generator.py  # Connects to VLM API (Moondream)
│   ├── embedder.py           # Computes text & standard CLIP embeddings
│   ├── vector_store.py       # Wrapper interface around ChromaDB
│   └── build_index.py        # Orchestrates the indexing pipeline
├── retriever/
│   ├── query_engine.py       # Search logic, standard CLIP reranking, and score fusion
│   └── search.py             # CLI entry point and Pillow-based visual grid rendering
├── tests/
│   ├── test_images/          # Cohort of 18 images used for testing
│   ├── test_indexer.py       # Isolated indexer test script
│   └── test_retriever.py     # Isolated retriever query test script
├── config.py                 # Hyperparameters, model configurations, and file paths
└── requirements.txt          # External Python dependencies
```

---

## 📝 Model Hyperparameters & Score Fusion
Weights and candidates count can be easily tuned in `config.py`:
*   `TOP_N_CANDIDATES = 20` (Number of images retrieved by the text filter before reranking)
*   `CAPTION_WEIGHT = 0.3` (Weight given to the semantic VLM description similarity)
*   `CLIP_WEIGHT = 0.7` (Weight given to the visual CLIP similarity)
