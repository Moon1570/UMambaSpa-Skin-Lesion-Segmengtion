"""LightningModule for Spatial Mamba U-Net."""

from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MaxMetric
from torchmetrics import Metric

from src.models.components.spatial_mamba_unet import SpatialMambaUNet


class DiceMetric(Metric):
    """Custom Dice metric for binary segmentation."""
    
    def __init__(self):
        super().__init__()
        self.add_state("dice_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update metric state."""
        # Binarize predictions
        preds = (preds > 0.5).float()
        target = target.float()
        
        # Flatten
        preds = preds.view(-1)
        target = target.view(-1)
        
        # Compute Dice
        intersection = (preds * target).sum()
        union = preds.sum() + target.sum()
        
        dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
        
        self.dice_sum += dice
        self.total += 1
    
    def compute(self):
        """Compute final metric value."""
        return self.dice_sum / self.total


class SpatialMambaLitModule(LightningModule):
    """
    Lightning Module for training Spatial Mamba U-Net.
    """
    
    def __init__(
        self,
        net: SpatialMambaUNet,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        compile: bool = False,
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])
        
        self.net = net
        self.criterion = criterion
        
        # Metrics - use custom Dice metric
        self.train_dice = DiceMetric()
        self.val_dice = DiceMetric()
        self.test_dice = DiceMetric()
        
        # Best validation metric
        self.val_dice_best = MaxMetric()
        
        # Loss tracking
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def on_train_start(self):
        """Reset metrics at start of training."""
        self.val_dice_best.reset()
    
    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        """Perform a single model step."""
        images, masks = batch
        preds = self.forward(images)
        loss = self.criterion(preds, masks)
        return loss, preds, masks
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Training step."""
        loss, preds, masks = self.model_step(batch)
        
        # Update metrics
        self.train_loss(loss)
        self.train_dice(preds, masks)
        
        # Log
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Validation step."""
        loss, preds, masks = self.model_step(batch)
        
        # Update metrics
        self.val_loss(loss)
        self.val_dice(preds, masks)
        
        # Log
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self):
        """Called at end of validation."""
        dice = self.val_dice.compute()
        self.val_dice_best(dice)
        self.log("val/dice_best", self.val_dice_best.compute(), sync_dist=True, prog_bar=True)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Test step."""
        loss, preds, masks = self.model_step(batch)
        
        # Update metrics
        self.test_loss(loss)
        self.test_dice(preds, masks)
        
        # Log
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dice", self.test_dice, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = self.hparams.optimizer(params=self.parameters())
        
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}