"""
Simple test for the multi-modal recommendation system.
"""

import pytest
import torch
import numpy as np
from src.utils import set_seed, get_device, compute_cosine_similarity
from src.models import create_model, ContrastiveLoss


def test_seed_setting():
    """Test that seed setting works correctly."""
    set_seed(42)
    a = torch.randn(10)
    set_seed(42)
    b = torch.randn(10)
    assert torch.allclose(a, b)


def test_device_detection():
    """Test device detection."""
    device = get_device()
    assert isinstance(device, torch.device)


def test_cosine_similarity():
    """Test cosine similarity computation."""
    emb1 = torch.randn(5, 10)
    emb2 = torch.randn(5, 10)
    sim = compute_cosine_similarity(emb1, emb2)
    assert sim.shape == (5, 5)
    assert torch.allclose(torch.diag(sim), torch.ones(5), atol=1e-6)


def test_model_creation():
    """Test model creation."""
    model = create_model("clip_based")
    assert hasattr(model, 'forward')
    assert hasattr(model, 'get_recommendations')


def test_contrastive_loss():
    """Test contrastive loss computation."""
    loss_fn = ContrastiveLoss(temperature=0.07)
    text_emb = torch.randn(4, 512)
    image_emb = torch.randn(4, 512)
    loss = loss_fn(text_emb, image_emb)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__])
