"""LightningDataModule for ISIC datasets."""

from typing import Optional, Tuple
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data.components.dataset import ISICDataset


class ISICDataModule(LightningDataModule):
    """
    LightningDataModule for ISIC datasets.
    
    Handles train/val/test splits and dataloaders.
    """
    
    def __init__(
        self,
        data_dir: str = "data/ISIC2017",
        train_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        spatial_mode: str = "rgb",
        image_size: Tuple[int, int] = (256, 256),
        batch_size: int = 12,
        num_workers: int = 4,
        pin_memory: bool = True,
        augmentation: dict = None,
        fold: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[ISICDataset] = None
        self.data_val: Optional[ISICDataset] = None
        self.data_test: Optional[ISICDataset] = None
    
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        
        if stage == "fit" or stage is None:
            # Training dataset
            if self.hparams.train_dir:
                self.data_train = ISICDataset(
                    image_dir=f"{self.hparams.train_dir}/images",
                    mask_dir=f"{self.hparams.train_dir}/masks",
                    image_size=self.hparams.image_size,
                    spatial_mode=self.hparams.spatial_mode,
                    transform=self._get_train_transform()
                )
            
            # Validation dataset
            if self.hparams.val_dir:
                self.data_val = ISICDataset(
                    image_dir=f"{self.hparams.val_dir}/images",
                    mask_dir=f"{self.hparams.val_dir}/masks",
                    image_size=self.hparams.image_size,
                    spatial_mode=self.hparams.spatial_mode,
                    transform=self._get_val_transform()
                )
        
        if stage == "test" or stage is None:
            # Test dataset
            if self.hparams.test_dir:
                self.data_test = ISICDataset(
                    image_dir=f"{self.hparams.test_dir}/images",
                    mask_dir=f"{self.hparams.test_dir}/masks",
                    image_size=self.hparams.image_size,
                    spatial_mode=self.hparams.spatial_mode,
                    transform=self._get_val_transform()
                )
    
    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False
        )
    
    def _get_train_transform(self):
        """Training augmentations."""
        augmentation = self.hparams.augmentation or {}
        aug_config = augmentation.get('train', {})
        
        return A.Compose([
            A.Resize(*self.hparams.image_size),
            A.HorizontalFlip(p=aug_config.get('horizontal_flip_prob', 0.5)),
            A.VerticalFlip(p=aug_config.get('vertical_flip_prob', 0.5)),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=aug_config.get('scale_limit', 0.2),
                rotate_limit=aug_config.get('rotate_limit', 30),
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config.get('brightness_limit', 0.2),
                contrast_limit=aug_config.get('contrast_limit', 0.2),
                p=0.5
            ),
            A.GaussNoise(p=0.2),
            ToTensorV2()
        ])
    
    def _get_val_transform(self):
        """Validation/test transforms."""
        return A.Compose([
            A.Resize(*self.hparams.image_size),
            ToTensorV2()
        ])