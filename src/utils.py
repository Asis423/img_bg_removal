import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union (IoU)"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def calculate_accuracy(pred, target, threshold=0.5):
    """Calculate pixel-wise accuracy"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    correct = (pred_binary == target_binary).float().sum()
    total = target_binary.numel()
    
    return correct / total

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")
    return epoch, loss

def plot_training_curves(train_losses, val_losses, train_ious, val_ious, save_path):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # IoU curves
    ax2.plot(train_ious, label='Training IoU', color='green')
    ax2.plot(val_ious, label='Validation IoU', color='orange')
    ax2.set_title('Training and Validation IoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved: {save_path}")

def apply_morphological_operations(mask, kernel_size=3):
    """Apply morphological operations to smooth mask"""
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
    
    # Convert to uint8
    mask_np = (mask_np * 255).astype(np.uint8)
    
    # Create kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply closing to fill holes
    mask_closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    
    # Apply opening to remove noise
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    
    # Convert back to float
    return mask_clean.astype(np.float32) / 255.0

def create_transparent_background(image, mask, blur_edges=True, blur_radius=1):
    """Create image with transparent background"""
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    # Ensure image is in correct format (H, W, C)
    if image.shape[0] == 3:  # If channels first
        image = np.transpose(image, (1, 2, 0))
    
    # Ensure mask is 2D
    if mask.shape[0] == 1:  # If channels first
        mask = mask.squeeze(0)
    
    # Normalize image to 0-255
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Smooth mask edges if requested
    if blur_edges:
        mask = cv2.GaussianBlur(mask, (blur_radius*2+1, blur_radius*2+1), 0)
    
    # Create RGBA image
    rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgba_image[:, :, :3] = image  # RGB channels
    rgba_image[:, :, 3] = (mask * 255).astype(np.uint8)  # Alpha channel
    
    return rgba_image

def save_prediction_comparison(image, mask, pred_mask, save_path):
    """Save comparison of original, ground truth, and prediction"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if torch.is_tensor(image):
        img_np = image.cpu().numpy()
        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
    else:
        img_np = image
    
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    if torch.is_tensor(mask):
        mask_np = mask.cpu().squeeze().numpy()
    else:
        mask_np = mask
    
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Predicted mask
    if torch.is_tensor(pred_mask):
        pred_np = pred_mask.cpu().squeeze().numpy()
    else:
        pred_np = pred_mask
    
    axes[2].imshow(pred_np, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")