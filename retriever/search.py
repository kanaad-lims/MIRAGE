"""CLI entry point for running a query."""
import sys
import os
from PIL import Image
from retriever.query_engine import QueryEngine

def show_results_grid(results):
    images = []
    for r in results:
        try:
            img = Image.open(r['image_path']).convert("RGB")
            # Resize for consistent grid display
            img.thumbnail((300, 300))
            images.append(img)
        except Exception as e:
            print(f"   (Could not load image file {r['image_path']}: {e})")
            
    if not images:
        return
        
    # Stitch images horizontally
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    
    grid_img = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        grid_img.paste(img, (x_offset, (max_height - img.size[1]) // 2))
        x_offset += img.size[0]
        
    print("Opening results image grid...")
    grid_img.show()

def main():
    query = " ".join(sys.argv[1:]) or input("Enter query: ")
    engine = QueryEngine()
    
    try:
        results = engine.search(query, k=5)
    except Exception as e:
        print(f"Error during search: {e}")
        return

    print("\nSearch Results:")
    print("="*60)
    for rank, r in enumerate(results, 1):
        filename = os.path.basename(r['image_path'])
        print(f"{rank}. {filename} (score={r['score']:.3f})")
        print(f"   caption: {r['caption']}\n")
    print("="*60)

    # Show the visual grid
    show_results_grid(results)

if __name__ == "__main__":
    main()