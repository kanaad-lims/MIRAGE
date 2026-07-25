import os
import streamlit as st
from PIL import Image
from retriever.query_engine import QueryEngine
import config

st.set_page_config(page_title="MIRAGE Search", page_icon="🌌", layout="wide")

st.title("🌌 MIRAGE: Multimodal Image Search Engine")
st.markdown("### Interactive playground for VLM dense captions fused with OpenAI CLIP visual reranking.")

@st.cache_resource
def load_engine():
    return QueryEngine()

engine = load_engine()

with st.sidebar:
    st.header("⚙️ Search Configuration")
    query = st.text_input("Search Query", placeholder="e.g., a yellow taxi in the city")
    caption_weight = st.slider("Caption Semantic Weight", 0.0, 1.0, 0.35, 0.05)
    clip_weight = st.slider("CLIP Visual Weight", 0.0, 1.0, 0.65, 0.05)
    k = st.slider("Number of Matches (K)", 1, 5, 3)
    search_btn = st.button("🔍 Search Database", type="primary", use_container_width=True)

if search_btn and query:
    config.CAPTION_WEIGHT = caption_weight
    config.CLIP_WEIGHT = clip_weight

    with st.spinner("Searching..."):
        results = engine.search_images(query, k=k)

    if not results:
        st.warning("No results found. Try a different query!")
    else:
        st.success(f"Found {len(results)} matches!")
        cols = st.columns(len(results))
        for i, (col, r) in enumerate(zip(cols, results)):
            with col:
                img = Image.open(r['image_path'])
                st.image(img, use_column_width=True)
                st.caption(f"**Rank {i+1}** | Score: `{r['score']:.3f}`\n\n`{os.path.basename(r['image_path'])}`")
elif search_btn and not query:
    st.error("Please enter a search query!")
else:
    st.info("👈 Enter a query in the sidebar and click **Search Database** to get started!")
