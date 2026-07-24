import os
import gradio as gr
from retriever.query_engine import QueryEngine
import config
import spaces

# Initialize query engine
engine = QueryEngine()

# Adding ZeroGPU decorator for HF spaces
@spaces.GPU

def search_interface(query, caption_weight, clip_weight, k):
    # Apply selected weights to configuration
    config.CAPTION_WEIGHT = float(caption_weight)
    config.CLIP_WEIGHT = float(clip_weight)
    
    # Query the pre-built database
    results = engine.search_images(query, k=int(k))
    
    # Format results as a list of (image_path, caption_label) for Gradio Gallery
    gallery_items = []
    for i, r in enumerate(results):
        filename = os.path.basename(r['image_path'])
        label = f"Rank {i+1} | Score: {r['score']:.3f} | {filename}"
        gallery_items.append((r['image_path'], label))
        
    return gallery_items

# Custom interface with sliders for score fusion weights
with gr.Blocks(theme=gr.themes.Soft(primary_hue="amber", neutral_hue="slate")) as demo:
    gr.Markdown(
        """
        # 🌌 MIRAGE: Multimodal Image Search Engine
        ### Interactive playground for VLM dense captions fused with OpenAI CLIP visual reranking.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            q_input = gr.Textbox(
                label="Search Query", 
                placeholder="e.g., a yellow taxi in the city", 
                lines=2
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
                columns=[2], 
                rows=[2], 
                height="auto"
            )
            
    btn.click(search_interface, [q_input, w_cap, w_clip, k_results], gallery)

# Launch app
if __name__ == "__main__":
    demo.launch()
