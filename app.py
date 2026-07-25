import spaces
import os
import gradio as gr
from retriever.query_engine import QueryEngine
import config

# Initialize query engine
engine = QueryEngine()

@spaces.GPU
def dummy_gpu_function():
    return None

# Custom CSS to force a unified dark theme and fix contrast issues
custom_css = """
body, .gradio-container { 
    background-color: #0b0f19 !important; 
    color: #f8fafc !important; 
}
h1, h2, h3, p, span, label, .readable-text { 
    color: #f8fafc !important; 
}
.gradio-container {
    border-radius: 12px;
    padding: 20px;
}
/* Force inputs and panels to have a dark background */
.input-background, textarea, input, .form, .gr-box {
    background-color: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
}
.gr-slider-input {
    background-color: #1e293b !important;
    color: #f8fafc !important;
}
/* Style the accordions and sliders */
.gr-accordion, .gr-collapse {
    border: 1px solid #334155 !important;
    background-color: #111827 !important;
}
"""

def search_interface(query, caption_weight, clip_weight, k):
    # Apply selected weights to configuration
    config.CAPTION_WEIGHT = float(caption_weight)
    config.CLIP_WEIGHT = float(clip_weight)
    
    # Query the pre-built database
    results = engine.search_images(query, k=int(k))
    
    # Format results as a list of (image_path, caption_label) for Gradio Gallery
    gallery_items = []
    for i, r in enumerate(results):
        # Extract just the filename to make the path relative and platform-independent
        filename = os.path.basename(r['image_path'])
        relative_path = os.path.join("data", "images", filename)
        
        label = f"Rank {i+1} | Score: {r['score']:.3f} | {filename}"
        gallery_items.append((relative_path, label))
        
    return gallery_items

# Custom interface with CSS injection and forced dark mode class
with gr.Blocks(theme=gr.themes.Soft(primary_hue="amber", neutral_hue="slate"), css=custom_css) as demo:
    # Force dark mode class on the document body on load
    demo.load(js="""
    () => {
        document.body.classList.add('dark');
        document.querySelector('.gradio-container').classList.add('dark');
    }
    """)
    
    gr.HTML(
        """
        <h1 style='font-size: 2.2em; font-weight: 800; text-align: center; color: #f8fafc; margin-top: 20px;'>🌌 MIRAGE: Multimodal Image Search Engine</h1>
        <p style='font-size: 1.1em; text-align: center; color: #94a3b8; margin-bottom: 20px;'>Interactive playground for VLM dense captions fused with OpenAI CLIP visual reranking.</p>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            q_input = gr.Textbox(
                label="Search Query", 
                placeholder="e.g., a yellow taxi in the city", 
                lines=2,
                elem_classes="input-background"
            )
            
            with gr.Accordion("Search Weights & Tuning", open=True):
                w_cap = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.35, step=0.05, 
                    label="Caption Semantic Weight"
                )
                w_clip = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.65, step=0.05, 
                    label="CLIP Visual Weight"
                )
                k_results = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1, 
                    label="Number of Matches (K)"
                )
                
            btn = gr.Button("🔍 Search Database", variant="primary")
            
        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="Retrieved Results", 
                columns=2, 
                rows=2, 
                height="auto"
            )
            
    btn.click(search_interface, [q_input, w_cap, w_clip, k_results], gallery)

# Launch app
if __name__ == "__main__":
    demo.launch()
