"""ISIC Dataset with spatial encoding."""

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable

from src.data.spatial_encoding import SpatialEncoder


class ISICDataset(Dataset):
    """ISIC skin lesion dataset."""
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        spatial_mode: str = "rgb",
        transform: Optional[Callable] = None,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.transform = transform
        
        # Get image paths
        self.image_paths = sorted(
            list(self.image_dir.glob("*.jpg")) + 
            list(self.image_dir.glob("*.png"))
        )
        
        # Spatial encoder
        self.spatial_encoder = SpatialEncoder(
            mode=spatial_mode,
            image_size=image_size
        )
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self._find_mask_path(img_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Normalize to [0, 1]
        image = image.float() / 255.0
        mask = mask.float() / 255.0
        mask = mask.unsqueeze(0)
        
        # Add spatial encoding
        image = self.spatial_encoder.encode(image)
        
        return image, mask
    
    def _find_mask_path(self, img_path: Path) -> Path:
        """Find corresponding mask file."""
        # Try different naming conventions
        possible_names = [
            img_path.stem + "_segmentation.png",
            img_path.stem + "_mask.png",
            img_path.stem + ".png",
        ]
        
        for name in possible_names:
            mask_path = self.mask_dir / name
            if mask_path.exists():
                return mask_path
        
        raise FileNotFoundError(f"No mask found for {img_path.name}")