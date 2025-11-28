"""Lightning module for MK-Spatial-Mamba U-Net."""

import torch
from lightning import LightningModule
from torchmetrics import Metric, JaccardIndex

from src.models.components.mk_spatial_mamba_unet import MKSpatialMambaUNet


class DiceMetric(Metric):
    """Custom Dice metric for binary segmentation."""
    
    def __init__(self):
        super().__init__()
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update metric state."""
        # Ensure binary predictions
        if preds.dtype == torch.float:
            preds = (preds > 0.5).float()
        else:
            preds = preds.float()
        
        target = target.float()
        
        # Flatten
        preds = preds.view(-1)
        target = target.view(-1)
        
        # Calculate intersection and union
        intersection = (preds * target).sum()
        union = preds.sum() + target.sum()
        
        self.intersection += intersection
        self.union += union
    
    def compute(self):
        """Compute final metric value."""
        return (2.0 * self.intersection + 1e-6) / (self.union + 1e-6)


class MKSpatialMambaLitModule(LightningModule):
    """
    LightningModule for MK-Spatial-Mamba U-Net.
    
    Supports:
    - Hybrid (Mamba + CNN)
    - Pure CNN (no Mamba)
    - With/without spatial coordinates
    - With/without CBAM attention
    
    Metrics:
    - Dice Score (primary)
    - IoU / Jaccard Index
    """
    
    def __init__(
        self,
        net: dict = None,
        optimizer: dict = None,
        criterion: dict = None,
        # Legacy flat parameters (for backward compatibility)
        in_channels: int = 6,
        num_classes: int = 1,
        channels: list = [16, 32, 64, 96, 160],
        use_spatial: bool = True,
        use_mamba: bool = True,
        use_cbam: bool = True,
        optimizer_lr: float = 0.001,
        optimizer_weight_decay: float = 0.0001,
        dice_weight: float = 0.6,
        bce_weight: float = 0.4,
    ):
        super().__init__()
        
        # Check if net is already instantiated by Hydra
        if isinstance(net, MKSpatialMambaUNet):
            # Network already instantiated - use it directly
            print("✅ Network already instantiated by Hydra")
            self.net = net
        elif isinstance(net, dict):
            # Network is a config dict - instantiate it
            print("✅ Instantiating network from config dict")
            net_params = {k: v for k, v in net.items() if k != '_target_'}
            in_channels = net_params.get('in_channels', in_channels)
            num_classes = net_params.get('num_classes', num_classes)
            channels = net_params.get('channels', channels)
            use_spatial = net_params.get('use_spatial', use_spatial)
            use_mamba = net_params.get('use_mamba', use_mamba)
            use_cbam = net_params.get('use_cbam', use_cbam)
            
            self.net = MKSpatialMambaUNet(
                in_channels=in_channels,
                num_classes=num_classes,
                channels=channels,
                use_spatial=use_spatial,
                use_mamba=use_mamba,
                use_cbam=use_cbam,
            )
        elif net is None:
            # No net provided - use legacy parameters
            print("✅ Using legacy flat parameters")
            self.net = MKSpatialMambaUNet(
                in_channels=in_channels,
                num_classes=num_classes,
                channels=channels,
                use_spatial=use_spatial,
                use_mamba=use_mamba,
                use_cbam=use_cbam,
            )
        else:
            raise ValueError(
                f"Invalid 'net' parameter type: {type(net)}. "
                "Expected MKSpatialMambaUNet instance, dict, or None."
            )
        
        # Handle optimizer config
        if isinstance(optimizer, dict):
            optimizer_lr = optimizer.get('lr', optimizer_lr)
            optimizer_weight_decay = optimizer.get('weight_decay', optimizer_weight_decay)
        
        # Handle criterion config
        if isinstance(criterion, dict):
            dice_weight = criterion.get('dice_weight', dice_weight)
            bce_weight = criterion.get('bce_weight', bce_weight)
        
        # Save hyperparameters (exclude the network itself)
        self.save_hyperparameters(ignore=['net'])
        
        # Store optimizer params for configure_optimizers
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        
        # Loss function
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        
        # Metrics - Dice Score
        self.train_dice = DiceMetric()
        self.val_dice = DiceMetric()
        self.test_dice = DiceMetric()
        
        # Metrics - IoU (Jaccard Index)
        self.train_iou = JaccardIndex(task='binary', num_classes=2)
        self.val_iou = JaccardIndex(task='binary', num_classes=2)
        self.test_iou = JaccardIndex(task='binary', num_classes=2)
        
        # Track best validation metrics
        self.val_dice_best = 0.0
        self.val_iou_best = 0.0
    
    def forward(self, x):
        """Forward pass."""
        return self.net(x)
    
    def dice_loss(self, pred, target, smooth=1e-6):
        """Compute Dice loss."""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice
    
    def compute_loss(self, pred, target):
        """
        Compute combined Dice + BCE loss.
        
        Args:
            pred: Logits [B, 1, H, W]
            target: Ground truth [B, 1, H, W]
        """
        # Sigmoid for Dice loss (needs probabilities)
        pred_sigmoid = torch.sigmoid(pred)
        dice = self.dice_loss(pred_sigmoid, target)
        
        # BCEWithLogitsLoss expects logits (applies sigmoid internally)
        bce = self.bce_loss(pred, target)
        
        loss = self.dice_weight * dice + self.bce_weight * bce
        return loss, dice, bce
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        images, masks = batch
        
        # Forward pass (returns logits)
        logits = self.forward(images)
        
        # Compute loss
        loss, dice, bce = self.compute_loss(logits, masks)
        
        # Convert to probabilities for metrics
        preds = torch.sigmoid(logits)
        
        # Update metrics
        self.train_dice.update(preds, masks.long())
        
        preds_binary = (preds > 0.5).long()
        self.train_iou.update(preds_binary, masks.long())
        
        # Log
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice_loss", dice, on_step=False, on_epoch=True)
        self.log("train/bce_loss", bce, on_step=False, on_epoch=True)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, masks = batch
        
        # Forward pass (returns logits)
        logits = self.forward(images)
        
        # Compute loss
        loss, dice, bce = self.compute_loss(logits, masks)
        
        # Convert to probabilities for metrics
        preds = torch.sigmoid(logits)
        
        # Update metrics
        self.val_dice.update(preds, masks.long())
        
        preds_binary = (preds > 0.5).long()
        self.val_iou.update(preds_binary, masks.long())
        
        # Log
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice_loss", dice, on_step=False, on_epoch=True)
        self.log("val/bce_loss", bce, on_step=False, on_epoch=True)
        self.log("val/dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Track best validation metrics."""
        current_dice = self.val_dice.compute()
        current_iou = self.val_iou.compute()
        
        if current_dice > self.val_dice_best:
            self.val_dice_best = current_dice
        
        if current_iou > self.val_iou_best:
            self.val_iou_best = current_iou
        
        self.log("val/dice_best", self.val_dice_best, prog_bar=True)
        self.log("val/iou_best", self.val_iou_best, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        images, masks = batch
        
        # Forward pass (returns logits)
        logits = self.forward(images)
        
        # Compute loss
        loss, dice, bce = self.compute_loss(logits, masks)
        
        # Convert to probabilities for metrics
        preds = torch.sigmoid(logits)
        
        # Update metrics
        self.test_dice.update(preds, masks.long())
        
        preds_binary = (preds > 0.5).long()
        self.test_iou.update(preds_binary, masks.long())
        
        # Log
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/dice_loss", dice, on_step=False, on_epoch=True)
        self.log("test/bce_loss", bce, on_step=False, on_epoch=True)
        self.log("test/dice", self.test_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/iou", self.test_iou, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/dice",
            },
        }