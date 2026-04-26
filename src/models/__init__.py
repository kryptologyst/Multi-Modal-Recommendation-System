"""
Model implementations for the multi-modal recommendation system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from transformers import CLIPModel, CLIPProcessor
import logging

logger = logging.getLogger(__name__)


class CLIPBasedRecommender(nn.Module):
    """CLIP-based recommendation model with contrastive learning.
    
    This model uses CLIP embeddings for both products and user preferences,
    then computes similarity scores for recommendation.
    """
    
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 embedding_dim: int = 512,
                 temperature: float = 0.07,
                 use_hard_negatives: bool = True):
        """Initialize the CLIP-based recommender.
        
        Args:
            model_name: Name of the CLIP model to use.
            embedding_dim: Dimension of the embedding space.
            temperature: Temperature for contrastive learning.
            use_hard_negatives: Whether to use hard negative mining.
        """
        super().__init__()
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_model.eval()  # Freeze CLIP parameters initially
        
        # Projection layers for fine-tuning
        self.text_projection = nn.Linear(embedding_dim, embedding_dim)
        self.image_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Initialize projections
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.xavier_uniform_(self.image_projection.weight)
        
        logger.info(f"Initialized CLIP-based recommender with {model_name}")
    
    def forward(self, 
                text_inputs: Dict[str, torch.Tensor],
                image_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            text_inputs: Dictionary containing text input tensors.
            image_inputs: Dictionary containing image input tensors.
            
        Returns:
            Tuple of (text_embeddings, image_embeddings).
        """
        # Get CLIP embeddings
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
            image_features = self.clip_model.get_image_features(**image_inputs)
        
        # Apply projections
        text_embeddings = self.text_projection(text_features)
        image_embeddings = self.image_projection(image_features)
        
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        
        return text_embeddings, image_embeddings
    
    def compute_similarity(self, 
                          query_embeddings: torch.Tensor,
                          candidate_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity between query and candidate embeddings.
        
        Args:
            query_embeddings: Query embeddings.
            candidate_embeddings: Candidate embeddings.
            
        Returns:
            Similarity matrix.
        """
        # Compute cosine similarity
        similarity = torch.mm(query_embeddings, candidate_embeddings.t())
        
        # Apply temperature scaling
        similarity = similarity / self.temperature
        
        return similarity
    
    def get_recommendations(self, 
                           user_query: str,
                           products: list,
                           processor: CLIPProcessor,
                           top_k: int = 5) -> list:
        """Get product recommendations for a user query.
        
        Args:
            user_query: User's text query.
            products: List of product dictionaries.
            processor: CLIP processor for preprocessing.
            top_k: Number of top recommendations to return.
            
        Returns:
            List of recommended products with scores.
        """
        self.eval()
        
        with torch.no_grad():
            # Process user query
            query_inputs = processor(text=[user_query], return_tensors="pt", padding=True)
            query_inputs = {k: v.to(next(self.parameters()).device) for k, v in query_inputs.items()}
            
            # Get query embedding
            query_embedding = self.clip_model.get_text_features(**query_inputs)
            query_embedding = self.text_projection(query_embedding)
            query_embedding = F.normalize(query_embedding, p=2, dim=-1)
            
            # Process products
            product_texts = [p["text"] for p in products]
            product_inputs = processor(text=product_texts, return_tensors="pt", padding=True)
            product_inputs = {k: v.to(next(self.parameters()).device) for k, v in product_inputs.items()}
            
            # Get product embeddings
            product_embeddings = self.clip_model.get_text_features(**product_inputs)
            product_embeddings = self.text_projection(product_embeddings)
            product_embeddings = F.normalize(product_embeddings, p=2, dim=-1)
            
            # Compute similarities
            similarities = torch.mm(query_embedding, product_embeddings.t()).squeeze(0)
            
            # Get top-k recommendations
            top_indices = torch.topk(similarities, top_k).indices.cpu().numpy()
            
            recommendations = []
            for idx in top_indices:
                product = products[idx].copy()
                product["similarity_score"] = similarities[idx].item()
                recommendations.append(product)
            
            return recommendations


class AdvancedRecommender(nn.Module):
    """Advanced recommendation model with multi-modal fusion and attention.
    
    This model extends the basic CLIP approach with additional features
    like attention mechanisms and multi-modal fusion.
    """
    
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 embedding_dim: int = 512,
                 hidden_dim: int = 256,
                 num_attention_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """Initialize the advanced recommender.
        
        Args:
            model_name: Name of the CLIP model to use.
            embedding_dim: Dimension of the embedding space.
            hidden_dim: Hidden dimension for attention layers.
            num_attention_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_model.eval()  # Freeze CLIP parameters
        
        # Multi-modal fusion layers
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer layers for refinement
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projections
        self.text_projection = nn.Linear(embedding_dim, embedding_dim)
        self.image_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"Initialized advanced recommender with {num_layers} transformer layers")
    
    def forward(self, 
                text_inputs: Dict[str, torch.Tensor],
                image_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the advanced model.
        
        Args:
            text_inputs: Dictionary containing text input tensors.
            image_inputs: Dictionary containing image input tensors.
            
        Returns:
            Tuple of (text_embeddings, image_embeddings).
        """
        batch_size = text_inputs["input_ids"].size(0)
        
        # Get CLIP embeddings
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
            image_features = self.clip_model.get_image_features(**image_inputs)
        
        # Reshape for attention
        text_features = text_features.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        image_features = image_features.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # Multi-modal fusion
        fused_features, _ = self.fusion_layer(
            text_features, image_features, image_features
        )
        
        # Apply transformer layers
        refined_features = self.transformer_encoder(fused_features)
        
        # Apply projections
        text_embeddings = self.text_projection(refined_features.squeeze(1))
        image_embeddings = self.image_projection(refined_features.squeeze(1))
        
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        
        return text_embeddings, image_embeddings


class ContrastiveLoss(nn.Module):
    """Contrastive loss for training the recommendation model."""
    
    def __init__(self, temperature: float = 0.07):
        """Initialize the contrastive loss.
        
        Args:
            temperature: Temperature parameter for scaling.
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, 
                text_embeddings: torch.Tensor,
                image_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            text_embeddings: Text embeddings.
            image_embeddings: Image embeddings.
            
        Returns:
            Contrastive loss value.
        """
        batch_size = text_embeddings.size(0)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(text_embeddings, image_embeddings.t()) / self.temperature
        
        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size, device=text_embeddings.device)
        
        # Compute cross-entropy loss
        loss_text_to_image = F.cross_entropy(similarity_matrix, labels)
        loss_image_to_text = F.cross_entropy(similarity_matrix.t(), labels)
        
        # Total loss
        total_loss = (loss_text_to_image + loss_image_to_text) / 2
        
        return total_loss


def create_model(model_type: str = "clip_based", **kwargs) -> nn.Module:
    """Create a recommendation model.
    
    Args:
        model_type: Type of model to create.
        **kwargs: Additional arguments for model initialization.
        
    Returns:
        Initialized model.
    """
    if model_type == "clip_based":
        return CLIPBasedRecommender(**kwargs)
    elif model_type == "advanced":
        return AdvancedRecommender(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
