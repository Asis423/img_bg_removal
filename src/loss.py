import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    """Custom Binary Cross Entropy Loss implementation from scratch"""
    def __init__(self, weight=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.eps = 1e-7  # Small epsilon to avoid log(0)
    
    def forward(self, pred, target):
        # Clamp predictions to avoid numerical instability
        pred = torch.clamp(pred, self.eps, 1 - self.eps)
        
        # Calculate BCE manually: -[y*log(p) + (1-y)*log(1-p)]
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        
        if self.weight is not None:
            bce = bce * self.weight
        
        if self.reduction == 'mean':
            return torch.mean(bce)
        elif self.reduction == 'sum':
            return torch.sum(bce)
        else:
            return bce

class DiceLoss(nn.Module):
    """Dice Loss for better boundary detection"""
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        
        # Dice coefficient
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        # Return dice loss (1 - dice coefficient)
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined BCE and Dice Loss"""
    def __init__(self, bce_weight=0.7, dice_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = BCELoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-7
    
    def forward(self, pred, target):
        pred = torch.clamp(pred, self.eps, 1 - self.eps)
        
        # Calculate cross entropy
        ce_loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        
        # Calculate p_t
        p_t = pred * target + (1 - pred) * (1 - target)
        
        # Calculate focal weight
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

# Test loss functions
if __name__ == "__main__":
    # Test data
    pred = torch.sigmoid(torch.randn(4, 1, 320, 320))
    target = torch.randint(0, 2, (4, 1, 320, 320)).float()
    
    # Test different losses
    bce_loss = BCELoss()
    dice_loss = DiceLoss()
    combined_loss = CombinedLoss()
    focal_loss = FocalLoss()
    
    print(f"BCE Loss: {bce_loss(pred, target):.4f}")
    print(f"Dice Loss: {dice_loss(pred, target):.4f}")
    print(f"Combined Loss: {combined_loss(pred, target):.4f}")
    print(f"Focal Loss: {focal_loss(pred, target):.4f}")