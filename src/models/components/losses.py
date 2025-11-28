"""Loss functions for segmentation."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice Loss."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.flatten(1)
        target = target.flatten(1)
        
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss."""
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()  # Safe for autocast/mixed precision
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid for Dice loss (expects probabilities)
        pred_sigmoid = torch.sigmoid(pred)
        dice_loss = self.dice(pred_sigmoid, target)
        
        # BCE with logits (expects raw logits)
        bce_loss = self.bce(pred, target)
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss