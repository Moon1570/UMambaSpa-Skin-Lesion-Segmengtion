"""Lightning module for enhanced MK U-Net experiments."""

import torch
from lightning import LightningModule
from torchmetrics import Metric, JaccardIndex

from src.models.components.mk_enhanced_unet import MKEnhancedUNet


class DiceMetric(Metric):
    """Custom Dice metric for binary segmentation."""
    
    def __init__(self):
        super().__init__()
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update metric state."""
        if preds.dtype == torch.float:
            preds = (preds > 0.5).float()
        else:
            preds = preds.float()
        
        target = target.float()
        preds = preds.view(-1)
        target = target.view(-1)
        
        intersection = (preds * target).sum()
        union = preds.sum() + target.sum()
        
        self.intersection += intersection
        self.union += union
    
    def compute(self):
        """Compute final metric value."""
        return (2.0 * self.intersection) / (self.union + 1e-6)


class MKEnhancedLitModule(LightningModule):
    """
    Lightning module for enhanced MK U-Net.
    
    Supports:
    - Deep Supervision with weighted auxiliary losses
    - Squeeze-and-Excitation attention
    - Multiple loss components (Dice + BCE)
    """
    
    def __init__(
        self,
        net: dict = None,
        optimizer: dict = None,
        criterion: dict = None,
        # Legacy parameters
        in_channels: int = 6,
        num_classes: int = 1,
        channels: list = [16, 32, 64, 96],
        use_spatial: bool = True,
        use_se: bool = False,
        deep_supervision: bool = False,
        optimizer_lr: float = 0.001,
        optimizer_weight_decay: float = 0.0001,
        dice_weight: float = 0.6,
        bce_weight: float = 0.4,
        aux_weight: float = 0.4,  # Weight for auxiliary losses in deep supervision
    ):
        super().__init__()
        
        # Validate config
        if net is None:
            raise ValueError(
                "Config error: 'net' parameter is required but was None. "
                "Please check your model config file."
            )
        
        # Check if net is already instantiated or a dict config
        if isinstance(net, dict):
            # Extract network parameters from dict config
            net_params = {k: v for k, v in net.items() if k != '_target_'}
            in_channels = net_params.get('in_channels', in_channels)
            num_classes = net_params.get('num_classes', num_classes)
            channels = net_params.get('channels', channels)
            use_spatial = net_params.get('use_spatial', use_spatial)
            use_se = net_params.get('use_se', use_se)
            deep_supervision = net_params.get('deep_supervision', deep_supervision)
        else:
            # Net is already instantiated, extract params from the instance
            if hasattr(net, 'in_channels'):
                in_channels = net.in_channels
            if hasattr(net, 'num_classes'):
                num_classes = net.num_classes
            if hasattr(net, 'use_spatial'):
                use_spatial = net.use_spatial
            if hasattr(net, 'use_se'):
                use_se = net.use_se
            if hasattr(net, 'deep_supervision'):
                deep_supervision = net.deep_supervision
        
        # Extract optimizer parameters
        if optimizer is not None:
            optimizer_lr = optimizer.get('lr', optimizer_lr)
            optimizer_weight_decay = optimizer.get('weight_decay', optimizer_weight_decay)
        
        # Extract loss parameters
        if criterion is not None:
            dice_weight = criterion.get('dice_weight', dice_weight)
            bce_weight = criterion.get('bce_weight', bce_weight)
            aux_weight = criterion.get('aux_weight', aux_weight)
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Store params
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.deep_supervision = deep_supervision
        self.aux_weight = aux_weight
        
        # Model - use provided instance or create new one
        if isinstance(net, dict):
            # Create new instance from config
            self.net = MKEnhancedUNet(
                in_channels=in_channels,
                num_classes=num_classes,
                channels=channels,
                use_spatial=use_spatial,
                use_se=use_se,
                deep_supervision=deep_supervision,
            )
        else:
            # Use the already instantiated model
            self.net = net
        
        # Loss functions
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce_loss = torch.nn.BCEWithLogitsLoss()  # Safe for mixed precision
        
        # Metrics
        self.train_dice = DiceMetric()
        self.val_dice = DiceMetric()
        self.test_dice = DiceMetric()
        
        self.train_iou = JaccardIndex(task='binary', num_classes=2)
        self.val_iou = JaccardIndex(task='binary', num_classes=2)
        self.test_iou = JaccardIndex(task='binary', num_classes=2)
        
        self.val_dice_best = 0.0
        self.val_iou_best = 0.0
    
    def forward(self, x):
        """Forward pass."""
        return self.net(x)
    
    def dice_loss(self, pred, target, smooth=1e-6):
        """Compute Dice loss from logits."""
        # Apply sigmoid to convert logits to probabilities
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice
    
    def compute_loss(self, pred, target, aux_preds=None):
        """
        Compute combined loss.
        
        If deep supervision is enabled and aux_preds are provided,
        applies weighted loss to auxiliary outputs.
        """
        # Main output loss
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        main_loss = self.dice_weight * dice + self.bce_weight * bce
        
        # Deep supervision auxiliary losses
        if self.deep_supervision and aux_preds is not None:
            aux_loss = 0
            for aux_pred in aux_preds:
                aux_dice = self.dice_loss(aux_pred, target)
                aux_bce = self.bce_loss(aux_pred, target)
                aux_loss += self.dice_weight * aux_dice + self.bce_weight * aux_bce
            
            # Average auxiliary loss and combine with main loss
            aux_loss = aux_loss / len(aux_preds)
            total_loss = (1 - self.aux_weight) * main_loss + self.aux_weight * aux_loss
            
            return total_loss, main_loss, aux_loss
        
        return main_loss, dice, bce
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        images, masks = batch
        
        # Forward pass
        output = self.forward(images)
        
        # Handle deep supervision outputs
        if self.deep_supervision and isinstance(output, tuple):
            preds, aux_preds = output
            loss, main_loss, aux_loss = self.compute_loss(preds, masks, aux_preds)
            self.log("train/main_loss", main_loss, on_step=False, on_epoch=True)
            self.log("train/aux_loss", aux_loss, on_step=False, on_epoch=True)
        else:
            preds = output
            loss, dice, bce = self.compute_loss(preds, masks)
            self.log("train/dice_loss", dice, on_step=False, on_epoch=True)
            self.log("train/bce_loss", bce, on_step=False, on_epoch=True)
        
        # Update metrics (apply sigmoid to logits)
        preds_prob = torch.sigmoid(preds)
        self.train_dice.update(preds_prob, masks.long())
        preds_binary = (preds_prob > 0.5).long()
        self.train_iou.update(preds_binary, masks.long())
        
        # Log
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, masks = batch
        
        # Forward pass (no aux outputs during validation)
        output = self.forward(images)
        if isinstance(output, tuple):
            preds = output[0]
        else:
            preds = output
        
        # Compute loss
        loss, dice, bce = self.compute_loss(preds, masks)
        
        # Update metrics (apply sigmoid to logits)
        preds_prob = torch.sigmoid(preds)
        self.val_dice.update(preds_prob, masks.long())
        preds_binary = (preds_prob > 0.5).long()
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
        
        # Forward pass
        output = self.forward(images)
        if isinstance(output, tuple):
            preds = output[0]
        else:
            preds = output
        
        # Compute loss
        loss, dice, bce = self.compute_loss(preds, masks)
        
        # Update metrics (apply sigmoid to logits)
        preds_prob = torch.sigmoid(preds)
        self.test_dice.update(preds_prob, masks.long())
        preds_binary = (preds_prob > 0.5).long()
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
