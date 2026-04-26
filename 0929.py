"""
Project 929: Multi-Modal Recommendation System

A modern, research-ready multi-modal recommendation system that leverages 
vision-language models (CLIP) to provide personalized product recommendations 
based on user preferences and product descriptions.

This is a simplified demonstration script. For the full implementation,
see the src/ directory with proper training, evaluation, and demo scripts.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Any
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import set_seed, get_device, format_metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleRecommender:
    """Simple CLIP-based recommendation system for demonstration."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize the recommender.
        
        Args:
            model_name: Name of the CLIP model to use.
        """
        self.model_name = model_name
        self.device = get_device()
        
        # Load pre-trained CLIP model and processor
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        logger.info(f"Model loaded on device: {self.device}")
    
    def recommend_products(self, 
                          user_query: str, 
                          products: List[Dict[str, Any]], 
                          top_n: int = 3) -> List[Dict[str, Any]]:
        """Recommend products based on user query.
        
        Args:
            user_query: User's text query.
            products: List of product dictionaries.
            top_n: Number of top recommendations.
            
        Returns:
            List of recommended products with similarity scores.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Process user query
            query_inputs = self.processor(
                text=[user_query], 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            # Get query embedding
            query_embedding = self.model.get_text_features(**query_inputs)
            
            # Process products
            product_texts = [p["text"] for p in products]
            product_inputs = self.processor(
                text=product_texts, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            # Get product embeddings
            product_embeddings = self.model.get_text_features(**product_inputs)
            
            # Compute cosine similarity
            similarity = torch.cosine_similarity(
                query_embedding, 
                product_embeddings, 
                dim=1
            )
            
            # Get top-k recommendations
            top_indices = torch.topk(similarity, top_n).indices.cpu().numpy()
            
            recommendations = []
            for idx in top_indices:
                product = products[idx].copy()
                product["similarity_score"] = similarity[idx].item()
                recommendations.append(product)
            
            return recommendations


def create_sample_products() -> List[Dict[str, Any]]:
    """Create sample product data for demonstration.
    
    Returns:
        List of sample product dictionaries.
    """
    return [
        {
            "id": "product_001",
            "text": "High-performance running shoes with excellent cushioning and breathable mesh upper.",
            "category": "footwear",
            "price": 129.99,
            "rating": 4.5,
            "brand": "SportMax"
        },
        {
            "id": "product_002", 
            "text": "Comfortable cotton t-shirt in various colors, perfect for casual wear.",
            "category": "clothing",
            "price": 24.99,
            "rating": 4.2,
            "brand": "ComfortWear"
        },
        {
            "id": "product_003",
            "text": "Warm winter jacket with water-resistant outer shell and cozy inner lining.",
            "category": "outerwear",
            "price": 199.99,
            "rating": 4.7,
            "brand": "WinterGear"
        },
        {
            "id": "product_004",
            "text": "Smartwatch with fitness tracking, heart rate monitor, and smartphone connectivity.",
            "category": "electronics",
            "price": 299.99,
            "rating": 4.4,
            "brand": "TechWear"
        },
        {
            "id": "product_005",
            "text": "Wireless noise-cancelling headphones with premium sound quality and long battery life.",
            "category": "electronics", 
            "price": 179.99,
            "rating": 4.6,
            "brand": "AudioPro"
        },
        {
            "id": "product_006",
            "text": "Durable hiking backpack with multiple compartments and ergonomic shoulder straps.",
            "category": "accessories",
            "price": 89.99,
            "rating": 4.3,
            "brand": "AdventureGear"
        }
    ]


def main():
    """Main demonstration function."""
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create sample products
    products = create_sample_products()
    logger.info(f"Created {len(products)} sample products")
    
    # Initialize recommender
    recommender = SimpleRecommender()
    
    # Sample user queries
    user_queries = [
        "I need comfortable running shoes for marathon training",
        "Looking for a warm jacket for winter hiking",
        "Want wireless headphones for music and calls",
        "Need a smartwatch to track my fitness goals"
    ]
    
    # Generate recommendations for each query
    for query in user_queries:
        logger.info(f"\nQuery: {query}")
        logger.info("-" * 50)
        
        recommendations = recommender.recommend_products(
            user_query=query,
            products=products,
            top_n=3
        )
        
        for i, rec in enumerate(recommendations):
            logger.info(f"{i+1}. {rec['text']}")
            logger.info(f"   Category: {rec['category']} | Price: ${rec['price']:.2f} | Rating: {rec['rating']:.1f}/5.0")
            logger.info(f"   Similarity Score: {rec['similarity_score']:.3f}")
            logger.info("")
    
    logger.info("Demonstration completed successfully!")
    logger.info("\nFor the full implementation with training, evaluation, and demo:")
    logger.info("1. Run: python -m src.scripts.train --config configs/default.yaml")
    logger.info("2. Run: python -m src.scripts.evaluate --checkpoint checkpoints/best_model.pth")
    logger.info("3. Run: streamlit run src/scripts/demo.py")


if __name__ == "__main__":
    main()

