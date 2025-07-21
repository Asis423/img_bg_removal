#!/usr/bin/env python3
"""
Main training script for background removal model
Run this file to start training
"""

import torch
import os
import sys
from config import Config
from src.model import UNet
from src.dataset import get_data_loaders
from src.train import Trainer
from src.utils import ensure_dir


def main():
    # Initialize configuration
    config = Config()
    
    print("="*60)
    print("Background Removal Model Training")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"Input size: {config.INPUT_SIZE}x{config.INPUT_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("="*60)
    
    # Check if data directories exist
    required_dirs = [
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        config.TEST_IMG_DIR,
        config.TEST_MASK_DIR
    ]
    
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    if missing_dirs:
        print("Error: Missing data directories:")
        for d in missing_dirs:
            print(f"  - {d}")
        print("\nPlease ensure DUTS dataset is properly extracted.")
        return
    
    # Create necessary directories
    ensure_dir(os.path.dirname(config.SAVE_MODEL_PATH))
    ensure_dir(config.RESULTS_DIR)
    
    try:
        # Load datasets
        print("Loading datasets...")
        train_loader, val_loader = get_data_loaders(config)
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        # Initialize model
        print("\nInitializing model...")
        model = UNet().to(config.DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Initialize trainer
        trainer = Trainer(model, train_loader, val_loader, config)
        
        # Start training
        print("\nStarting training...")
        train_losses, val_losses, train_ious, val_ious = trainer.train()
        
        # Print final results
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        print(f"Best validation loss: {min(val_losses):.4f}")
        print(f"Best validation IoU: {max(val_ious):.4f}")
        print(f"Model saved at: {config.SAVE_MODEL_PATH}")
        print(f"Training curves saved at: {config.RESULTS_DIR}/training_curves.png")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()