#!/usr/bin/env python3
"""
Setup script for the Multi-Modal Recommendation System.

This script helps set up the environment and install dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.10 or higher.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_pip():
    """Check if pip is available."""
    try:
        import pip
        print("✅ pip is available")
        return True
    except ImportError:
        print("❌ pip is not available. Please install pip first.")
        return False


def install_dependencies():
    """Install project dependencies."""
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    return True


def setup_pre_commit():
    """Setup pre-commit hooks."""
    if not run_command("pip install pre-commit", "Installing pre-commit"):
        return False
    
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        return False
    
    return True


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/images",
        "data/text", 
        "data/annotations",
        "checkpoints",
        "logs",
        "assets",
        "results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def test_installation():
    """Test if the installation works."""
    print("🧪 Testing installation...")
    
    try:
        # Test basic imports
        import torch
        import transformers
        import numpy as np
        import pandas as pd
        import matplotlib
        import streamlit
        
        print("✅ All core dependencies imported successfully")
        
        # Test device detection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ PyTorch device detection: {device}")
        
        # Test CLIP model loading
        from transformers import CLIPProcessor
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("✅ CLIP processor loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("🚀 Multi-Modal Recommendation System Setup")
    print("=" * 60)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating project directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup pre-commit (optional)
    print("\n🔧 Setting up development tools...")
    setup_pre_commit()
    
    # Test installation
    print("\n🧪 Testing installation...")
    if not test_installation():
        print("❌ Installation test failed")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    
    print("\n📋 Next steps:")
    print("1. Run quick demo: python demo/quick_demo.py")
    print("2. Train model: python -m src.scripts.train --config configs/default.yaml")
    print("3. Evaluate model: python -m src.scripts.evaluate --checkpoint checkpoints/best_model.pth")
    print("4. Run web demo: streamlit run src/scripts/demo.py")
    
    print("\n📚 Documentation:")
    print("- README.md: Complete project documentation")
    print("- configs/: Configuration files")
    print("- src/: Source code")
    print("- demo/: Demo scripts")
    
    print("\n⚠️  Important Notes:")
    print("- This is a demonstration system for educational purposes")
    print("- Recommendations are based on text similarity")
    print("- Always verify product information before purchasing")
    print("- For production use, consider additional safety measures")


if __name__ == "__main__":
    main()
