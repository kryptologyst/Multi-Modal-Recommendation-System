"""
Training script for the multi-modal recommendation system.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from tqdm import tqdm
import logging
from omegaconf import OmegaConf

from ..data import create_data_loaders
from ..models import create_model, ContrastiveLoss
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
            logging.FileHandler('training.log')
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


def train_epoch(model: nn.Module, 
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epoch: int) -> float:
    """Train the model for one epoch.
    
    Args:
        model: The model to train.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        epoch: Current epoch number.
        
    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = move_to_device(batch, device)
        
        # Forward pass
        text_embeddings, image_embeddings = model(
            {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
            {"pixel_values": batch["pixel_values"]}
        )
        
        # Compute loss
        loss = criterion(text_embeddings, image_embeddings)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
    
    return total_loss / num_batches


def validate_epoch(model: nn.Module,
                   val_loader: DataLoader,
                   criterion: nn.Module,
                   evaluator,
                   device: torch.device,
                   epoch: int) -> tuple:
    """Validate the model for one epoch.
    
    Args:
        model: The model to validate.
        val_loader: Validation data loader.
        criterion: Loss function.
        evaluator: Model evaluator.
        device: Device to validate on.
        epoch: Current epoch number.
        
    Returns:
        Tuple of (validation loss, validation metrics).
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = move_to_device(batch, device)
            
            # Forward pass
            text_embeddings, image_embeddings = model(
                {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
                {"pixel_values": batch["pixel_values"]}
            )
            
            # Compute loss
            loss = criterion(text_embeddings, image_embeddings)
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
    
    # Compute validation metrics
    val_metrics = evaluator.evaluate(model, val_loader, device)
    
    return total_loss / num_batches, val_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train multi-modal recommendation system")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_path", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--log_level", type=str, default="INFO",
                       help="Logging level")
    
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
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_path=args.data_path,
        processor=processor,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"]
    )
    
    logger.info(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    # Create model
    model = create_model(
        model_type="clip_based",
        model_name=config["model"]["backbone"],
        embedding_dim=config["model"]["embedding_dim"],
        temperature=config["model"]["temperature"],
        use_hard_negatives=config["model"]["use_hard_negatives"]
    )
    model = model.to(device)
    
    # Create loss function
    criterion = ContrastiveLoss(temperature=config["model"]["temperature"])
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"]
    )
    
    # Create evaluator
    evaluator = create_evaluator(config)
    
    # Create visualizer
    visualizer = create_visualizer({"save_dir": "assets"})
    
    # Create leaderboard
    leaderboard = Leaderboard(
        metrics=config["evaluation"]["metrics"],
        save_path="assets/leaderboard.json"
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_metrics_history = {}
    val_metrics_history = {}
    
    logger.info("Starting training...")
    
    for epoch in range(1, config["training"]["epochs"] + 1):
        # Train epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # Validate epoch
        if epoch % config["training"]["eval_every_n_epochs"] == 0:
            val_loss, val_metrics = validate_epoch(model, val_loader, criterion, evaluator, device, epoch)
            val_losses.append(val_loss)
            
            # Store metrics history
            for metric, value in val_metrics.items():
                if metric not in val_metrics_history:
                    val_metrics_history[metric] = []
                val_metrics_history[metric].append(value)
            
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Validation Metrics: {format_metrics(val_metrics)}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'config': config
                }
                torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
                logger.info(f"Saved best model at epoch {epoch}")
        
        # Update scheduler
        scheduler.step()
        
        # Save checkpoint
        if epoch % config["training"]["save_every_n_epochs"] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'config': config
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluator.evaluate(model, test_loader, device)
    logger.info(f"Test Metrics: {format_metrics(test_metrics)}")
    
    # Add to leaderboard
    leaderboard.add_result(
        model_name=f"clip_based_epoch_{config['training']['epochs']}",
        metrics=test_metrics,
        config=config
    )
    
    # Plot training curves
    visualizer.plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        val_metrics=val_metrics_history,
        title="Training Progress",
        save_name="training_curves.png"
    )
    
    # Print leaderboard
    leaderboard.print_leaderboard()
    leaderboard.save_results()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
