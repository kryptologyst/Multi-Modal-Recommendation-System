"""
Evaluation script for the multi-modal recommendation system.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from transformers import CLIPProcessor
import logging
from omegaconf import OmegaConf

from ..data import create_data_loaders
from ..models import create_model
from ..eval import create_evaluator, Leaderboard
from ..viz import create_visualizer
from ..utils import set_seed, get_device, move_to_device, format_metrics

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation.log')
        ]
    )


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
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Model trained for {checkpoint['epoch']} epochs")
    
    return model


def evaluate_model(model: nn.Module,
                   test_loader,
                   evaluator,
                   device: torch.device) -> dict:
    """Evaluate the model on test set.
    
    Args:
        model: The model to evaluate.
        test_loader: Test data loader.
        evaluator: Model evaluator.
        device: Device to evaluate on.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info("Evaluating model on test set...")
    
    # Evaluate model
    metrics = evaluator.evaluate(model, test_loader, device)
    
    logger.info(f"Test Metrics: {format_metrics(metrics)}")
    
    return metrics


def generate_recommendations(model: nn.Module,
                           processor: CLIPProcessor,
                           products: list,
                           user_queries: list,
                           device: torch.device,
                           top_k: int = 5) -> dict:
    """Generate recommendations for user queries.
    
    Args:
        model: The trained model.
        processor: CLIP processor.
        products: List of available products.
        user_queries: List of user queries.
        device: Device to run inference on.
        top_k: Number of top recommendations.
        
    Returns:
        Dictionary mapping queries to recommendations.
    """
    model.eval()
    recommendations = {}
    
    logger.info(f"Generating recommendations for {len(user_queries)} queries...")
    
    with torch.no_grad():
        for query in user_queries:
            # Get recommendations
            recs = model.get_recommendations(
                user_query=query,
                products=products,
                processor=processor,
                top_k=top_k
            )
            
            recommendations[query] = recs
            
            logger.info(f"Query: {query}")
            for i, rec in enumerate(recs):
                logger.info(f"  {i+1}. {rec['text']} (Score: {rec['similarity_score']:.3f})")
    
    return recommendations


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate multi-modal recommendation system")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Logging level")
    parser.add_argument("--generate_recs", action="store_true",
                       help="Generate recommendation examples")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set random seed
    set_seed(config["device"]["seed"])
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load CLIP processor
    processor = CLIPProcessor.from_pretrained(config["model"]["backbone"])
    logger.info(f"Loaded CLIP processor: {config['model']['backbone']}")
    
    # Create test data loader
    _, _, test_loader = create_data_loaders(
        data_path=args.data_path,
        processor=processor,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"]
    )
    
    logger.info(f"Created test data loader with {len(test_loader)} batches")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Create evaluator
    evaluator = create_evaluator(config)
    
    # Create visualizer
    visualizer = create_visualizer({"save_dir": args.output_dir})
    
    # Evaluate model
    test_metrics = evaluate_model(model, test_loader, evaluator, device)
    
    # Generate recommendation examples if requested
    if args.generate_recs:
        logger.info("Generating recommendation examples...")
        
        # Sample products for demonstration
        sample_products = [
            {"id": "prod_1", "text": "High-performance running shoes with excellent cushioning", "category": "footwear", "price": 129.99, "rating": 4.5},
            {"id": "prod_2", "text": "Comfortable cotton t-shirt in various colors", "category": "clothing", "price": 24.99, "rating": 4.2},
            {"id": "prod_3", "text": "Warm winter jacket with water-resistant shell", "category": "outerwear", "price": 199.99, "rating": 4.7},
            {"id": "prod_4", "text": "Smartwatch with fitness tracking and heart rate monitor", "category": "electronics", "price": 299.99, "rating": 4.4},
            {"id": "prod_5", "text": "Wireless noise-cancelling headphones", "category": "electronics", "price": 179.99, "rating": 4.6}
        ]
        
        # Sample user queries
        user_queries = [
            "I need comfortable running shoes for marathon training",
            "Looking for a warm jacket for winter hiking",
            "Want wireless headphones for music and calls",
            "Need a smartwatch to track my fitness goals"
        ]
        
        # Generate recommendations
        recommendations = generate_recommendations(
            model=model,
            processor=processor,
            products=sample_products,
            user_queries=user_queries,
            device=device,
            top_k=3
        )
        
        # Visualize recommendations
        for query, recs in recommendations.items():
            visualizer.plot_recommendation_examples(
                query=query,
                recommendations=recs,
                title=f"Recommendations for: {query}",
                save_name=f"recommendations_{query.replace(' ', '_')}.png"
            )
    
    # Save results
    import json
    results = {
        "test_metrics": test_metrics,
        "config": config,
        "checkpoint_path": args.checkpoint
    }
    
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved evaluation results to {results_path}")
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
