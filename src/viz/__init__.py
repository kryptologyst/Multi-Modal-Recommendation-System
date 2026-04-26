"""
Visualization utilities for the multi-modal recommendation system.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


class RecommendationVisualizer:
    """Visualizer for recommendation results and model analysis."""
    
    def __init__(self, 
                 save_dir: str = "assets",
                 style: str = "seaborn-v0_8"):
        """Initialize the visualizer.
        
        Args:
            save_dir: Directory to save visualizations.
            style: Matplotlib style to use.
        """
        self.save_dir = save_dir
        plt.style.use(style)
        
        # Create save directory
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Initialized visualizer with save directory: {save_dir}")
    
    def plot_embedding_space(self, 
                            embeddings: torch.Tensor,
                            labels: List[str],
                            title: str = "Embedding Space Visualization",
                            method: str = "tsne",
                            save_name: str = "embedding_space.png") -> None:
        """Plot 2D visualization of embedding space.
        
        Args:
            embeddings: Embedding vectors.
            labels: Labels for each embedding.
            title: Plot title.
            method: Dimensionality reduction method ("tsne" or "pca").
            save_name: Name to save the plot.
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # Convert to numpy
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Dimensionality reduction
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "pca":
            reducer = PCA(n_components=2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embeddings_2d = reducer.fit_transform(embeddings_np)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Color by unique labels
        unique_labels = list(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = [l == label for l in labels]
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=label, alpha=0.7, s=50)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(f"{method.upper()} Component 1", fontsize=12)
        plt.ylabel(f"{method.upper()} Component 2", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        save_path = f"{self.save_dir}/{save_name}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Saved embedding visualization to {save_path}")
    
    def plot_similarity_matrix(self, 
                              similarity_matrix: torch.Tensor,
                              labels: List[str],
                              title: str = "Similarity Matrix",
                              save_name: str = "similarity_matrix.png") -> None:
        """Plot similarity matrix heatmap.
        
        Args:
            similarity_matrix: Similarity matrix.
            labels: Labels for rows/columns.
            title: Plot title.
            save_name: Name to save the plot.
        """
        # Convert to numpy
        sim_matrix = similarity_matrix.detach().cpu().numpy()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        
        # Truncate labels if too long
        display_labels = [label[:20] + "..." if len(label) > 20 else label 
                         for label in labels]
        
        sns.heatmap(sim_matrix, 
                   xticklabels=display_labels,
                   yticklabels=display_labels,
                   cmap='viridis',
                   cbar_kws={'label': 'Similarity Score'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("Products", fontsize=12)
        plt.ylabel("Products", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        save_path = f"{self.save_dir}/{save_name}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Saved similarity matrix to {save_path}")
    
    def plot_metrics_comparison(self, 
                               metrics_data: Dict[str, Dict[str, float]],
                               title: str = "Model Performance Comparison",
                               save_name: str = "metrics_comparison.png") -> None:
        """Plot comparison of metrics across models.
        
        Args:
            metrics_data: Dictionary mapping model names to metrics.
            title: Plot title.
            save_name: Name to save the plot.
        """
        # Extract metrics and models
        models = list(metrics_data.keys())
        metrics = list(next(iter(metrics_data.values())).keys())
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            values = [metrics_data[model][metric] for model in models]
            
            bars = ax.bar(models, values, alpha=0.7)
            ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        save_path = f"{self.save_dir}/{save_name}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Saved metrics comparison to {save_path}")
    
    def plot_recommendation_examples(self, 
                                   query: str,
                                   recommendations: List[Dict[str, Any]],
                                   images: Optional[List[Any]] = None,
                                   title: str = "Recommendation Examples",
                                   save_name: str = "recommendation_examples.png") -> None:
        """Plot recommendation examples with images and scores.
        
        Args:
            query: User query.
            recommendations: List of recommended products.
            images: Optional list of product images.
            title: Plot title.
            save_name: Name to save the plot.
        """
        n_recommendations = len(recommendations)
        
        if images:
            # Create subplot with images
            fig, axes = plt.subplots(2, n_recommendations, figsize=(4*n_recommendations, 8))
            if n_recommendations == 1:
                axes = axes.reshape(2, 1)
        else:
            # Create subplot without images
            fig, axes = plt.subplots(1, n_recommendations, figsize=(4*n_recommendations, 4))
            if n_recommendations == 1:
                axes = [axes]
        
        for i, rec in enumerate(recommendations):
            if images:
                # Plot image
                axes[0, i].imshow(images[i])
                axes[0, i].set_title(f"Rank {i+1}", fontweight='bold')
                axes[0, i].axis('off')
                
                # Plot text and score
                ax = axes[1, i]
            else:
                ax = axes[i]
            
            # Display product information
            text = f"Score: {rec.get('similarity_score', 0):.3f}\n"
            text += f"Category: {rec.get('category', 'Unknown')}\n"
            text += f"Price: ${rec.get('price', 0):.2f}\n"
            text += f"Rating: {rec.get('rating', 0):.1f}/5.0"
            
            ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="lightblue", alpha=0.7))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        plt.suptitle(f"{title}\nQuery: {query}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        save_path = f"{self.save_dir}/{save_name}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Saved recommendation examples to {save_path}")
    
    def create_interactive_plot(self, 
                               metrics_data: Dict[str, Dict[str, float]],
                               title: str = "Interactive Model Comparison") -> go.Figure:
        """Create an interactive plotly visualization.
        
        Args:
            metrics_data: Dictionary mapping model names to metrics.
            title: Plot title.
            
        Returns:
            Plotly figure object.
        """
        models = list(metrics_data.keys())
        metrics = list(next(iter(metrics_data.values())).keys())
        
        # Create radar chart
        fig = go.Figure()
        
        for model in models:
            values = [metrics_data[model][metric] for metric in metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=model,
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=title,
            font=dict(size=12)
        )
        
        return fig
    
    def plot_training_curves(self, 
                            train_losses: List[float],
                            val_losses: List[float],
                            train_metrics: Optional[Dict[str, List[float]]] = None,
                            val_metrics: Optional[Dict[str, List[float]]] = None],
                            title: str = "Training Progress",
                            save_name: str = "training_curves.png") -> None:
        """Plot training curves.
        
        Args:
            train_losses: Training loss values.
            val_losses: Validation loss values.
            train_metrics: Training metrics.
            val_metrics: Validation metrics.
            title: Plot title.
            save_name: Name to save the plot.
        """
        epochs = range(1, len(train_losses) + 1)
        
        # Determine number of subplots
        n_plots = 1  # Loss plot
        if train_metrics:
            n_plots += len(train_metrics)
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        # Plot loss
        ax = axes[0]
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax.set_title('Loss', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot metrics
        if train_metrics and val_metrics:
            for i, metric in enumerate(train_metrics.keys()):
                ax = axes[i + 1]
                ax.plot(epochs, train_metrics[metric], 'b-', 
                       label=f'Training {metric}', linewidth=2)
                ax.plot(epochs, val_metrics[metric], 'r-', 
                       label=f'Validation {metric}', linewidth=2)
                ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Score')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        save_path = f"{self.save_dir}/{save_name}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Saved training curves to {save_path}")


def create_visualizer(config: Dict[str, Any]) -> RecommendationVisualizer:
    """Create a visualizer from configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Initialized visualizer.
    """
    return RecommendationVisualizer(
        save_dir=config.get("save_dir", "assets"),
        style=config.get("style", "seaborn-v0_8")
    )
