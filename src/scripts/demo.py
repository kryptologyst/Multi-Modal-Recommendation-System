"""
Demo script for the multi-modal recommendation system.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from transformers import CLIPProcessor
import streamlit as st
import logging
from PIL import Image
import numpy as np

from ..data import ProductDataset
from ..models import create_model
from ..utils import set_seed, get_device, move_to_device

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> nn.Module:
    """Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        config: Model configuration.
        device: Device to load model on.
        
    Returns:
        Loaded model.
    """
    # Create model
    model = create_model(
        model_type="clip_based",
        model_name=config["model"]["backbone"],
        embedding_dim=config["model"]["embedding_dim"],
        temperature=config["model"]["temperature"],
        use_hard_negatives=config["model"]["use_hard_negatives"]
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


@st.cache_resource
def load_model_cached(checkpoint_path: str, config: dict, device: torch.device) -> nn.Module:
    """Cached version of model loading for Streamlit."""
    return load_model(checkpoint_path, config, device)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Demo multi-modal recommendation system")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--port", type=int, default=8501,
                       help="Port for Streamlit app")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config["device"]["seed"])
    
    # Get device
    device = get_device()
    
    # Load CLIP processor
    processor = CLIPProcessor.from_pretrained(config["model"]["backbone"])
    
    # Load model
    if os.path.exists(args.checkpoint):
        model = load_model_cached(args.checkpoint, config, device)
        model_loaded = True
    else:
        logger.warning(f"Checkpoint not found: {args.checkpoint}")
        model_loaded = False
    
    # Load products
    try:
        product_dataset = ProductDataset(args.data_path, processor, "test", generate_synthetic=True)
        products = []
        for i in range(min(20, len(product_dataset))):  # Limit to 20 products for demo
            item = product_dataset[i]
            products.append({
                "id": item["id"],
                "text": item["text"],
                "category": item["category"],
                "price": item["price"],
                "rating": item["rating"],
                "brand": item["brand"]
            })
    except Exception as e:
        logger.error(f"Error loading products: {e}")
        products = []
    
    # Streamlit app
    st.set_page_config(
        page_title="Multi-Modal Recommendation System",
        page_icon="🛍️",
        layout="wide"
    )
    
    st.title("🛍️ Multi-Modal Recommendation System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    if model_loaded:
        st.sidebar.success("✅ Model loaded successfully")
    else:
        st.sidebar.error("❌ Model not loaded")
    
    st.sidebar.markdown(f"**Device:** {device}")
    st.sidebar.markdown(f"**Products loaded:** {len(products)}")
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("User Query")
        
        # Text input
        user_query = st.text_area(
            "Describe what you're looking for:",
            value="I need comfortable running shoes for marathon training",
            height=100
        )
        
        # Image upload (optional)
        uploaded_image = st.file_uploader(
            "Upload an image (optional):",
            type=['png', 'jpg', 'jpeg']
        )
        
        # Parameters
        st.subheader("Parameters")
        top_k = st.slider("Number of recommendations:", 1, 10, 5)
        temperature = st.slider("Temperature:", 0.01, 1.0, 0.07, 0.01)
        
        # Generate button
        generate_button = st.button("🔍 Generate Recommendations", type="primary")
    
    with col2:
        st.header("Recommendations")
        
        if generate_button and model_loaded and products:
            with st.spinner("Generating recommendations..."):
                try:
                    # Generate recommendations
                    recommendations = model.get_recommendations(
                        user_query=user_query,
                        products=products,
                        processor=processor,
                        top_k=top_k
                    )
                    
                    # Display recommendations
                    for i, rec in enumerate(recommendations):
                        with st.container():
                            st.markdown(f"### {i+1}. {rec['text']}")
                            
                            col2a, col2b = st.columns([2, 1])
                            
                            with col2a:
                                st.markdown(f"**Category:** {rec['category']}")
                                st.markdown(f"**Brand:** {rec['brand']}")
                                st.markdown(f"**Price:** ${rec['price']:.2f}")
                                st.markdown(f"**Rating:** {rec['rating']:.1f}/5.0")
                            
                            with col2b:
                                st.metric(
                                    "Similarity Score",
                                    f"{rec['similarity_score']:.3f}",
                                    delta=None
                                )
                            
                            st.markdown("---")
                    
                    # Show metrics
                    st.subheader("Recommendation Quality")
                    avg_score = np.mean([r['similarity_score'] for r in recommendations])
                    st.metric("Average Similarity", f"{avg_score:.3f}")
                    
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
        
        elif not model_loaded:
            st.warning("Please load a trained model to generate recommendations.")
        
        elif not products:
            st.warning("No products available. Please check the data path.")
        
        else:
            st.info("Click 'Generate Recommendations' to see results.")
    
    # Footer
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This is a multi-modal recommendation system that uses CLIP embeddings to match user preferences 
    with product descriptions. The system can process both text queries and images to provide 
    personalized recommendations.
    
    **Features:**
    - Text-based product search
    - Image-based product matching (coming soon)
    - Real-time recommendation generation
    - Similarity scoring
    """)
    
    # Safety disclaimer
    st.markdown("### Disclaimer")
    st.markdown("""
    ⚠️ **Important:** This is a demonstration system for educational purposes. 
    The recommendations are based on text similarity and may not reflect actual product quality or suitability.
    Always verify product information before making purchasing decisions.
    """)


if __name__ == "__main__":
    main()
