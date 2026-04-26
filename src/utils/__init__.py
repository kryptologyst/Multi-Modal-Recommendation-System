"""
Core utilities for the multi-modal recommendation system.
"""

import random
import numpy as np
import torch
from typing import Optional, Union, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set random seed to {seed}")


def get_device() -> torch.device:
    """Auto-detect and return the best available device.
    
    Returns:
        torch.device: The best available device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def move_to_device(data: Union[torch.Tensor, Dict[str, Any], List[Any]], 
                  device: torch.device) -> Union[torch.Tensor, Dict[str, Any], List[Any]]:
    """Move data to the specified device.
    
    Args:
        data: Data to move to device.
        device: Target device.
        
    Returns:
        Data moved to the target device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data


def compute_cosine_similarity(embeddings1: torch.Tensor, 
                             embeddings2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings.
        embeddings2: Second set of embeddings.
        
    Returns:
        Cosine similarity matrix.
    """
    # Normalize embeddings
    embeddings1_norm = torch.nn.functional.normalize(embeddings1, p=2, dim=-1)
    embeddings2_norm = torch.nn.functional.normalize(embeddings2, p=2, dim=-1)
    
    # Compute cosine similarity
    similarity = torch.mm(embeddings1_norm, embeddings2_norm.t())
    
    return similarity


def compute_recall_at_k(similarity_matrix: torch.Tensor, 
                       k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """Compute Recall@K metrics.
    
    Args:
        similarity_matrix: Similarity matrix between queries and candidates.
        k_values: List of K values to compute recall for.
        
    Returns:
        Dictionary containing Recall@K scores.
    """
    # Get top-k indices for each query
    _, top_k_indices = torch.topk(similarity_matrix, max(k_values), dim=1)
    
    # Create ground truth (diagonal elements are relevant)
    batch_size = similarity_matrix.size(0)
    ground_truth = torch.arange(batch_size, device=similarity_matrix.device)
    
    results = {}
    for k in k_values:
        # Check if ground truth is in top-k
        top_k_preds = top_k_indices[:, :k]
        hits = (top_k_preds == ground_truth.unsqueeze(1)).any(dim=1)
        recall_at_k = hits.float().mean().item()
        results[f"recall@{k}"] = recall_at_k
    
    return results


def compute_map_score(similarity_matrix: torch.Tensor, 
                     k: int = 10) -> float:
    """Compute Mean Average Precision (MAP) score.
    
    Args:
        similarity_matrix: Similarity matrix between queries and candidates.
        k: Number of top results to consider.
        
    Returns:
        MAP score.
    """
    batch_size = similarity_matrix.size(0)
    ground_truth = torch.arange(batch_size, device=similarity_matrix.device)
    
    # Get top-k indices
    _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
    
    map_scores = []
    for i in range(batch_size):
        pred_indices = top_k_indices[i]
        relevant_mask = (pred_indices == ground_truth[i])
        
        if relevant_mask.any():
            # Compute precision at each position
            precision_at_k = []
            for j in range(k):
                if pred_indices[j] == ground_truth[i]:
                    precision_at_k.append(relevant_mask[:j+1].sum().item() / (j + 1))
            
            if precision_at_k:
                map_scores.append(np.mean(precision_at_k))
            else:
                map_scores.append(0.0)
        else:
            map_scores.append(0.0)
    
    return np.mean(map_scores)


def compute_ndcg_score(similarity_matrix: torch.Tensor, 
                      k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG) score.
    
    Args:
        similarity_matrix: Similarity matrix between queries and candidates.
        k: Number of top results to consider.
        
    Returns:
        NDCG score.
    """
    batch_size = similarity_matrix.size(0)
    ground_truth = torch.arange(batch_size, device=similarity_matrix.device)
    
    # Get top-k indices
    _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
    
    ndcg_scores = []
    for i in range(batch_size):
        pred_indices = top_k_indices[i]
        
        # Compute DCG
        dcg = 0.0
        for j in range(k):
            if pred_indices[j] == ground_truth[i]:
                dcg += 1.0 / np.log2(j + 2)  # j+2 because log2(1) = 0
        
        # Compute IDCG (ideal DCG)
        idcg = 1.0 / np.log2(2)  # Only one relevant item at position 1
        
        # Compute NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary into a readable string.
    
    Args:
        metrics: Dictionary of metric names and values.
        
    Returns:
        Formatted metrics string.
    """
    formatted = []
    for name, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{name}: {value:.4f}")
        else:
            formatted.append(f"{name}: {value}")
    
    return " | ".join(formatted)
