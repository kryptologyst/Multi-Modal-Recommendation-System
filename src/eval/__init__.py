"""
Evaluation utilities for the multi-modal recommendation system.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import ndcg_score
import logging

from ..utils import compute_recall_at_k, compute_map_score, compute_ndcg_score

logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """Evaluator for recommendation models."""
    
    def __init__(self, 
                 metrics: List[str] = None,
                 top_k_values: List[int] = None,
                 cross_modal_eval: bool = True):
        """Initialize the evaluator.
        
        Args:
            metrics: List of metrics to compute.
            top_k_values: List of K values for top-K metrics.
            cross_modal_eval: Whether to perform cross-modal evaluation.
        """
        self.metrics = metrics or ["recall@1", "recall@5", "recall@10", "map", "ndcg"]
        self.top_k_values = top_k_values or [1, 5, 10, 20]
        self.cross_modal_eval = cross_modal_eval
        
        logger.info(f"Initialized evaluator with metrics: {self.metrics}")
    
    def evaluate(self, 
                model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                device: torch.device) -> Dict[str, float]:
        """Evaluate the model on a dataset.
        
        Args:
            model: The model to evaluate.
            data_loader: Data loader for evaluation.
            device: Device to run evaluation on.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        model.eval()
        
        all_text_embeddings = []
        all_image_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get embeddings
                text_embeddings, image_embeddings = model(
                    {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
                    {"pixel_values": batch["pixel_values"]}
                )
                
                all_text_embeddings.append(text_embeddings)
                all_image_embeddings.append(image_embeddings)
                
                # Create labels (assuming batch order corresponds to ground truth)
                batch_size = text_embeddings.size(0)
                labels = torch.arange(batch_size, device=device)
                all_labels.append(labels)
        
        # Concatenate all embeddings
        text_embeddings = torch.cat(all_text_embeddings, dim=0)
        image_embeddings = torch.cat(all_image_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Compute metrics
        metrics = self._compute_metrics(text_embeddings, image_embeddings, labels)
        
        return metrics
    
    def _compute_metrics(self, 
                        text_embeddings: torch.Tensor,
                        image_embeddings: torch.Tensor,
                        labels: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            text_embeddings: Text embeddings.
            image_embeddings: Image embeddings.
            labels: Ground truth labels.
            
        Returns:
            Dictionary of computed metrics.
        """
        metrics = {}
        
        # Compute similarity matrices
        text_to_image_sim = torch.mm(text_embeddings, image_embeddings.t())
        image_to_text_sim = torch.mm(image_embeddings, text_embeddings.t())
        
        # Text-to-Image metrics
        if "recall@1" in self.metrics or any(f"recall@{k}" in self.metrics for k in self.top_k_values):
            recall_metrics = compute_recall_at_k(text_to_image_sim, self.top_k_values)
            metrics.update({f"text_to_image_{k}": v for k, v in recall_metrics.items()})
        
        if "map" in self.metrics:
            map_score = compute_map_score(text_to_image_sim, max(self.top_k_values))
            metrics["text_to_image_map"] = map_score
        
        if "ndcg" in self.metrics:
            ndcg_score = compute_ndcg_score(text_to_image_sim, max(self.top_k_values))
            metrics["text_to_image_ndcg"] = ndcg_score
        
        # Image-to-Text metrics (if cross-modal evaluation is enabled)
        if self.cross_modal_eval:
            if "recall@1" in self.metrics or any(f"recall@{k}" in self.metrics for k in self.top_k_values):
                recall_metrics = compute_recall_at_k(image_to_text_sim, self.top_k_values)
                metrics.update({f"image_to_text_{k}": v for k, v in recall_metrics.items()})
            
            if "map" in self.metrics:
                map_score = compute_map_score(image_to_text_sim, max(self.top_k_values))
                metrics["image_to_text_map"] = map_score
            
            if "ndcg" in self.metrics:
                ndcg_score = compute_ndcg_score(image_to_text_sim, max(self.top_k_values))
                metrics["image_to_text_ndcg"] = ndcg_score
        
        # Average metrics across modalities
        if self.cross_modal_eval:
            for metric_name in ["recall@1", "recall@5", "recall@10", "map", "ndcg"]:
                text_key = f"text_to_image_{metric_name}"
                image_key = f"image_to_text_{metric_name}"
                if text_key in metrics and image_key in metrics:
                    avg_key = f"avg_{metric_name}"
                    metrics[avg_key] = (metrics[text_key] + metrics[image_key]) / 2
        
        return metrics
    
    def evaluate_recommendations(self, 
                                recommendations: List[Dict[str, Any]],
                                ground_truth: List[str],
                                top_k: int = 10) -> Dict[str, float]:
        """Evaluate recommendation quality.
        
        Args:
            recommendations: List of recommended items.
            ground_truth: List of ground truth relevant items.
            top_k: Number of top recommendations to consider.
            
        Returns:
            Dictionary of recommendation metrics.
        """
        metrics = {}
        
        # Hit Rate
        recommended_ids = [r["id"] for r in recommendations[:top_k]]
        hits = sum(1 for item_id in recommended_ids if item_id in ground_truth)
        hit_rate = hits / len(ground_truth) if ground_truth else 0.0
        metrics[f"hit_rate@{top_k}"] = hit_rate
        
        # Precision
        precision = hits / top_k if top_k > 0 else 0.0
        metrics[f"precision@{top_k}"] = precision
        
        # Recall
        recall = hits / len(ground_truth) if ground_truth else 0.0
        metrics[f"recall@{top_k}"] = recall
        
        # F1 Score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        metrics[f"f1@{top_k}"] = f1
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for i, item_id in enumerate(recommended_ids):
            if item_id in ground_truth:
                mrr = 1.0 / (i + 1)
                break
        metrics[f"mrr@{top_k}"] = mrr
        
        return metrics


class Leaderboard:
    """Leaderboard for tracking model performance."""
    
    def __init__(self, 
                 metrics: List[str] = None,
                 save_path: Optional[str] = None):
        """Initialize the leaderboard.
        
        Args:
            metrics: List of metrics to track.
            save_path: Path to save leaderboard results.
        """
        self.metrics = metrics or ["recall@1", "recall@5", "recall@10", "map", "ndcg"]
        self.save_path = save_path
        self.results = []
        
        logger.info(f"Initialized leaderboard with metrics: {self.metrics}")
    
    def add_result(self, 
                   model_name: str,
                   metrics: Dict[str, float],
                   config: Dict[str, Any] = None) -> None:
        """Add a result to the leaderboard.
        
        Args:
            model_name: Name of the model.
            metrics: Dictionary of metrics.
            config: Model configuration.
        """
        result = {
            "model_name": model_name,
            "metrics": metrics,
            "config": config or {},
            "timestamp": torch.datetime.now().isoformat()
        }
        
        self.results.append(result)
        
        # Sort by primary metric (recall@1)
        if "recall@1" in metrics:
            self.results.sort(key=lambda x: x["metrics"].get("recall@1", 0), reverse=True)
        
        logger.info(f"Added result for {model_name}: {metrics}")
    
    def get_top_results(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top-k results from the leaderboard.
        
        Args:
            top_k: Number of top results to return.
            
        Returns:
            List of top results.
        """
        return self.results[:top_k]
    
    def print_leaderboard(self, top_k: int = 10) -> None:
        """Print the leaderboard.
        
        Args:
            top_k: Number of top results to display.
        """
        print("\n" + "="*80)
        print("RECOMMENDATION SYSTEM LEADERBOARD")
        print("="*80)
        
        for i, result in enumerate(self.results[:top_k]):
            print(f"\n{i+1}. {result['model_name']}")
            print("-" * 40)
            
            for metric in self.metrics:
                if metric in result["metrics"]:
                    print(f"{metric:15}: {result['metrics'][metric]:.4f}")
        
        print("\n" + "="*80)
    
    def save_results(self) -> None:
        """Save leaderboard results to file."""
        if self.save_path:
            import json
            with open(self.save_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Saved leaderboard results to {self.save_path}")


def create_evaluator(config: Dict[str, Any]) -> RecommendationEvaluator:
    """Create an evaluator from configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Initialized evaluator.
    """
    eval_config = config.get("evaluation", {})
    
    return RecommendationEvaluator(
        metrics=eval_config.get("metrics", ["recall@1", "recall@5", "recall@10", "map", "ndcg"]),
        top_k_values=eval_config.get("top_k_values", [1, 5, 10, 20]),
        cross_modal_eval=eval_config.get("cross_modal_eval", True)
    )
