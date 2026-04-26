#!/usr/bin/env python3
"""
Quick demo script for the Multi-Modal Recommendation System.

This script demonstrates the basic functionality of the recommendation system
without requiring training or complex setup.
"""

import os
import sys
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_sample_products() -> List[Dict[str, Any]]:
    """Create sample product data for demonstration."""
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
        },
        {
            "id": "product_007",
            "text": "Automatic coffee maker with programmable settings and thermal carafe.",
            "category": "appliances",
            "price": 149.99,
            "rating": 4.5,
            "brand": "BrewMaster"
        },
        {
            "id": "product_008",
            "text": "Non-slip yoga mat with extra cushioning and carrying strap included.",
            "category": "fitness",
            "price": 39.99,
            "rating": 4.4,
            "brand": "FlexiFit"
        }
    ]


def recommend_products(user_query: str, products: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    """Recommend products based on user query using CLIP.
    
    Args:
        user_query: User's text query.
        products: List of product dictionaries.
        top_n: Number of top recommendations.
        
    Returns:
        List of recommended products with similarity scores.
    """
    device = get_device()
    
    # Load CLIP model
    logger.info("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    model.eval()
    
    with torch.no_grad():
        # Process user query
        query_inputs = processor(
            text=[user_query], 
            return_tensors="pt", 
            padding=True
        ).to(device)
        
        # Get query embedding
        query_embedding = model.get_text_features(**query_inputs)
        
        # Process products
        product_texts = [p["text"] for p in products]
        product_inputs = processor(
            text=product_texts, 
            return_tensors="pt", 
            padding=True
        ).to(device)
        
        # Get product embeddings
        product_embeddings = model.get_text_features(**product_inputs)
        
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


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("🛍️  Multi-Modal Recommendation System Demo")
    print("=" * 60)
    
    # Create sample products
    products = create_sample_products()
    logger.info(f"Created {len(products)} sample products")
    
    # Sample user queries
    user_queries = [
        "I need comfortable running shoes for marathon training",
        "Looking for a warm jacket for winter hiking",
        "Want wireless headphones for music and calls",
        "Need a smartwatch to track my fitness goals",
        "Looking for a good coffee maker for home use"
    ]
    
    # Generate recommendations for each query
    for i, query in enumerate(user_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 50)
        
        try:
            recommendations = recommend_products(
                user_query=query,
                products=products,
                top_n=3
            )
            
            for j, rec in enumerate(recommendations, 1):
                print(f"   {j}. {rec['text']}")
                print(f"      Category: {rec['category']} | Price: ${rec['price']:.2f} | Rating: {rec['rating']:.1f}/5.0")
                print(f"      Similarity Score: {rec['similarity_score']:.3f}")
                print()
        
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            print(f"   Error: {e}")
    
    print("=" * 60)
    print("✅ Demo completed successfully!")
    print("\nFor the full implementation with training, evaluation, and web demo:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Train model: python -m src.scripts.train --config configs/default.yaml")
    print("3. Evaluate model: python -m src.scripts.evaluate --checkpoint checkpoints/best_model.pth")
    print("4. Run web demo: streamlit run src/scripts/demo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
