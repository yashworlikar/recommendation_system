import gradio as gr
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class EcommerceRecommendationSystem:
    def __init__(self):
        # Load available embeddings
        self.embedding_files = {
            "Product Features": "ecommerce_embeddings.npy",
            "Product Names Only": "ecommerce_embeddings_nameOnly.npy",
        }
        
        # Load base data
        self.metadata = pd.read_csv("ecommerce_data.csv")
        
        # Initialize sentence transformer with a fixed model
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = np.load(self.embedding_files["Product Features"])  # Default embedding
        
    def vector_search(self, query, top_n=5, intimate_option='none', 
                      embedding_choice='Product Features', 
                      similarity_threshold=0.4, 
                      hide_embeddings=False):
        # Validate inputs
        if not query.strip():
            return "<p style='color: #888; text-align: center;'>Start typing to see fashion recommendations.</p>"
        
        # Load the selected embedding file
        self.embeddings = np.load(self.embedding_files[embedding_choice])
        
        # Compute query embeddings
        query_embedding = self.sentence_model.encode(query)
        
        # Compute similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Filter by similarity threshold
        valid_indices = np.where(similarities >= similarity_threshold)[0]
        
        # If no products meet the threshold, return a message
        if len(valid_indices) == 0:
            return "<p style='color: #888; text-align: center;'>No products found matching your preferences. Try a broader search.</p>"
        
        # Sort valid indices by similarity in descending order
        sorted_indices = valid_indices[similarities[valid_indices].argsort()[::-1]]
        
        # Get top N results
        top_indices = sorted_indices[:top_n]
        results = self.metadata.iloc[top_indices]
        
        # Intimate wear filtering
        if intimate_option == 'none':
            results = results[results['is_intimate_wear'] == False]
        
        # Prepare result HTML with improved visibility
        html_content = """
        <style>
            .product-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 20px; 
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            .product-card {
                border-radius: 12px;
                overflow: hidden;
                transition: all 0.3s ease;
                display: flex;
                flex-direction: column;
                box-shadow: 0 6px 12px rgba(255,107,53,0.2);
            }
            .product-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 15px 30px rgba(255,107,53,0.3);
            }
            .product-image {
                width: 100%; 
                height: 250px; 
                object-fit: cover;
                filter: saturate(0.9);
                transition: filter 0.3s ease;
            }
            .product-card:hover .product-image {
                filter: saturate(1.1);
            }
            .product-details {
                padding: 15px;
                text-align: center;
                flex-grow: 1;
                display: flex;
                flex-direction: column;
                background: linear-gradient(to bottom, white 50%, var(--secondary-color) 50%);
                background-size: 100% 200%;
                transition: background-position 0.3s;
                font-size: 0.9em;
            }
            .product-details h3 {
                font-size: 1.2em; 
            }
            .product-details p {
                color: rgba(0, 0, 0, 0.8); 
                margin: 5px 0;
            }
            .product-card:hover .product-details {
                background-position: 0 100%;
            }
            .product-link {
                display: inline-block;
                color: white;
                padding: 10px 20px;
                border-radius: 25px;
                text-decoration: none;
                margin-top: auto;
                align-self: center;
                transition: transform 0.3s, background 0.3s;
            }
            .product-link:hover {
                transform: scale(1.05);
            }
            .similarity-info {
                font-size: 0.7em;
                margin-top: 5px;
                opacity: 0.7;
            }
        </style>
        <div class="product-grid">
        """
        
        # Main results
        for idx, row in results.iterrows():
            # Calculate similarity percentage for display
            similarity_percentage = int(similarities[idx] * 100)
            
            # Optionally hide embedding details
            similarity_display = f'<div class="similarity-info">Similarity: {similarity_percentage}%</div>' if not hide_embeddings else ''
            
            html_content += f"""
            <div class="product-card">
                <img src="{row['link']}" alt="Product Image" class="product-image">
                <div class="product-details">
                    <h3>{row['productDisplayName'][:35]}...</h3>
                    <p>Category: {row['masterCategory']} | {row['subCategory']}</p>
                    <p>Color: {row['baseColour']} | Season: {row['season']} {row['year']}</p>
                    <a href="{row['link']}" target="_blank" class="product-link">View Product</a>
                    {similarity_display}
                </div>
            </div>
            """
        html_content += "</div>"
        return html_content
    
    def get_embedding_files(self):
        return list(self.embedding_files.keys())

# Initialize the recommendation system
recommender = EcommerceRecommendationSystem()

# Custom Gradio Theme
orange_theme = gr.Theme(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.yellow,
)

# Gradio Interface
with gr.Blocks(theme=orange_theme) as demo:
    gr.Markdown("# Fashion Recommendation")
    
    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Describe Your Fashion Preferences",
                placeholder="e.g., 'Casual black leather jacket for winter'",
                lines=2
            )
        
        with gr.Column(scale=1):
            result_count = gr.Slider(
                minimum=5, maximum=25, value=10, step=5, 
                label="Number of Recommendations"
            )
    
    with gr.Row():
        with gr.Column():
            settings_accordion = gr.Accordion("Advanced Settings", open=False)
            
            with settings_accordion:
                gr.Markdown("### Fine-tune Your Search")
                
                with gr.Row():
                    with gr.Column():
                        similarity_threshold = gr.Slider(
                            minimum=0, maximum=1, value=0.5, step=0.1, 
                            label="Similarity Threshold"
                        )
                    
                    with gr.Column():
                        hide_embeddings = gr.Checkbox(
                            label="Hide Similarity Details", 
                            value=False
                        )
                
                intimate_dropdown = gr.Dropdown(
                    choices=['none', 'include'], 
                    value='none', 
                    label="Intimate Wear Options"
                )
                
                embedding_dropdown = gr.Dropdown(
                    choices=recommender.get_embedding_files(), 
                    value='Product Features', 
                    label="Embedding Strategy"
                )
                
    
    output = gr.HTML(label="Your Fashion Recommendations")
    
    # Event listeners
    text_inputs = [
        text_input, 
        result_count, 
        intimate_dropdown, 
        embedding_dropdown,
        similarity_threshold,
        hide_embeddings
    ]

    search_triggers = [
        text_input.submit,
        text_input.change,
        result_count.change,
        intimate_dropdown.change,
        embedding_dropdown.change,
        similarity_threshold.change,
        hide_embeddings.change
    ]

    for trigger in search_triggers:
        trigger(
            fn=recommender.vector_search, 
            inputs=text_inputs, 
            outputs=output
        )
demo.queue(default_concurrency_limit=6)
demo.launch()
