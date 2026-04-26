"""
Data handling utilities for the multi-modal recommendation system.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import CLIPProcessor
import logging

logger = logging.getLogger(__name__)


class ProductDataset(Dataset):
    """Dataset class for product recommendation data.
    
    This dataset handles both real product data and synthetic data generation.
    """
    
    def __init__(self, 
                 data_path: str,
                 processor: CLIPProcessor,
                 split: str = "train",
                 max_samples: Optional[int] = None,
                 generate_synthetic: bool = False):
        """Initialize the dataset.
        
        Args:
            data_path: Path to the data directory.
            processor: CLIP processor for text and image preprocessing.
            split: Dataset split ("train", "val", "test").
            max_samples: Maximum number of samples to load.
            generate_synthetic: Whether to generate synthetic data if real data is missing.
        """
        self.data_path = data_path
        self.processor = processor
        self.split = split
        self.max_samples = max_samples
        self.generate_synthetic = generate_synthetic
        
        self.products = self._load_data()
        
        if self.max_samples:
            self.products = self.products[:self.max_samples]
        
        logger.info(f"Loaded {len(self.products)} products for {split} split")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load product data from files or generate synthetic data.
        
        Returns:
            List of product dictionaries.
        """
        annotations_path = os.path.join(self.data_path, "annotations.json")
        
        if os.path.exists(annotations_path):
            with open(annotations_path, 'r') as f:
                data = json.load(f)
            return data.get(self.split, [])
        elif self.generate_synthetic:
            logger.info("Generating synthetic product data")
            return self._generate_synthetic_data()
        else:
            logger.warning("No data found and synthetic generation disabled")
            return []
    
    def _generate_synthetic_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic product data for testing.
        
        Returns:
            List of synthetic product dictionaries.
        """
        synthetic_products = [
            {
                "id": "product_001",
                "image_path": "synthetic_running_shoes.jpg",
                "text": "High-performance running shoes with excellent cushioning and breathable mesh upper.",
                "category": "footwear",
                "price": 129.99,
                "rating": 4.5,
                "brand": "SportMax"
            },
            {
                "id": "product_002", 
                "image_path": "synthetic_cotton_tshirt.jpg",
                "text": "Comfortable cotton t-shirt in various colors, perfect for casual wear.",
                "category": "clothing",
                "price": 24.99,
                "rating": 4.2,
                "brand": "ComfortWear"
            },
            {
                "id": "product_003",
                "image_path": "synthetic_winter_jacket.jpg", 
                "text": "Warm winter jacket with water-resistant outer shell and cozy inner lining.",
                "category": "outerwear",
                "price": 199.99,
                "rating": 4.7,
                "brand": "WinterGear"
            },
            {
                "id": "product_004",
                "image_path": "synthetic_smartwatch.jpg",
                "text": "Smartwatch with fitness tracking, heart rate monitor, and smartphone connectivity.",
                "category": "electronics",
                "price": 299.99,
                "rating": 4.4,
                "brand": "TechWear"
            },
            {
                "id": "product_005",
                "image_path": "synthetic_headphones.jpg",
                "text": "Wireless noise-cancelling headphones with premium sound quality and long battery life.",
                "category": "electronics", 
                "price": 179.99,
                "rating": 4.6,
                "brand": "AudioPro"
            },
            {
                "id": "product_006",
                "image_path": "synthetic_backpack.jpg",
                "text": "Durable hiking backpack with multiple compartments and ergonomic shoulder straps.",
                "category": "accessories",
                "price": 89.99,
                "rating": 4.3,
                "brand": "AdventureGear"
            },
            {
                "id": "product_007",
                "image_path": "synthetic_coffee_maker.jpg",
                "text": "Automatic coffee maker with programmable settings and thermal carafe.",
                "category": "appliances",
                "price": 149.99,
                "rating": 4.5,
                "brand": "BrewMaster"
            },
            {
                "id": "product_008",
                "image_path": "synthetic_yoga_mat.jpg",
                "text": "Non-slip yoga mat with extra cushioning and carrying strap included.",
                "category": "fitness",
                "price": 39.99,
                "rating": 4.4,
                "brand": "FlexiFit"
            }
        ]
        
        # Create synthetic images directory if it doesn't exist
        images_dir = os.path.join(self.data_path, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Generate synthetic images
        for product in synthetic_products:
            image_path = os.path.join(images_dir, product["image_path"])
            if not os.path.exists(image_path):
                self._create_synthetic_image(image_path, product["category"])
        
        return synthetic_products
    
    def _create_synthetic_image(self, image_path: str, category: str) -> None:
        """Create a synthetic image for testing purposes.
        
        Args:
            image_path: Path where to save the synthetic image.
            category: Product category to determine image characteristics.
        """
        # Create a simple colored rectangle as a placeholder
        colors = {
            "footwear": (100, 50, 0),      # Brown
            "clothing": (0, 100, 200),     # Blue  
            "outerwear": (50, 50, 50),     # Dark gray
            "electronics": (200, 200, 0),   # Yellow
            "accessories": (150, 75, 0),    # Orange
            "appliances": (128, 128, 128),  # Gray
            "fitness": (0, 150, 0)          # Green
        }
        
        color = colors.get(category, (128, 128, 128))
        
        # Create a simple image
        img = Image.new('RGB', (224, 224), color)
        img.save(image_path)
    
    def __len__(self) -> int:
        """Return the number of products in the dataset."""
        return len(self.products)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a product item by index.
        
        Args:
            idx: Index of the product.
            
        Returns:
            Dictionary containing product data and processed features.
        """
        product = self.products[idx]
        
        # Load image
        image_path = os.path.join(self.data_path, "images", product["image_path"])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Could not load image {image_path}: {e}")
            # Create a placeholder image
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        # Process text and image
        inputs = self.processor(
            text=product["text"],
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Add metadata
        item = {
            "id": product["id"],
            "text": product["text"],
            "category": product["category"],
            "price": product.get("price", 0.0),
            "rating": product.get("rating", 0.0),
            "brand": product.get("brand", "Unknown"),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0)
        }
        
        return item


class UserPreferenceDataset(Dataset):
    """Dataset class for user preferences and queries."""
    
    def __init__(self, 
                 preferences: List[Dict[str, Any]],
                 processor: CLIPProcessor):
        """Initialize the user preference dataset.
        
        Args:
            preferences: List of user preference dictionaries.
            processor: CLIP processor for text and image preprocessing.
        """
        self.preferences = preferences
        self.processor = processor
    
    def __len__(self) -> int:
        """Return the number of user preferences."""
        return len(self.preferences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a user preference item by index.
        
        Args:
            idx: Index of the preference.
            
        Returns:
            Dictionary containing user preference data.
        """
        preference = self.preferences[idx]
        
        # Process text query
        inputs = self.processor(
            text=preference["query"],
            return_tensors="pt",
            padding=True
        )
        
        item = {
            "user_id": preference.get("user_id", f"user_{idx}"),
            "query": preference["query"],
            "preferred_category": preference.get("category", "general"),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }
        
        # Add image if available
        if "image_path" in preference:
            try:
                image = Image.open(preference["image_path"]).convert('RGB')
                image_inputs = self.processor(
                    images=image,
                    return_tensors="pt"
                )
                item["pixel_values"] = image_inputs["pixel_values"].squeeze(0)
            except Exception as e:
                logger.warning(f"Could not load user image {preference['image_path']}: {e}")
        
        return item


def create_data_loaders(data_path: str,
                       processor: CLIPProcessor,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       train_split: float = 0.8,
                       val_split: float = 0.1,
                       test_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        data_path: Path to the data directory.
        processor: CLIP processor for preprocessing.
        batch_size: Batch size for data loaders.
        num_workers: Number of worker processes.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation.
        test_split: Fraction of data for testing.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Create datasets for each split
    train_dataset = ProductDataset(data_path, processor, "train", generate_synthetic=True)
    val_dataset = ProductDataset(data_path, processor, "val", generate_synthetic=True)
    test_dataset = ProductDataset(data_path, processor, "test", generate_synthetic=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
