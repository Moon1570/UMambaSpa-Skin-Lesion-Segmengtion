# Quick data check: check_data.py
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.isic_datamodule import ISICDataModule
import matplotlib.pyplot as plt
import torch

# Load data
dm = ISICDataModule(
    data_dir="data/ISIC2017",
    train_dir="data/ISIC2017/train",
    val_dir="data/ISIC2017/val",
    test_dir="data/ISIC2017/test",
    spatial_mode="rgb",
    batch_size=4
)

dm.setup()

# Get a batch
batch = next(iter(dm.train_dataloader()))
images, masks = batch

# Visualize
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(4):
    # Image
    img = images[i].permute(1, 2, 0).numpy()
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"Image {i}")
    axes[0, i].axis('off')
    
    # Mask
    mask = masks[i, 0].numpy()
    axes[1, i].imshow(mask, cmap='gray')
    axes[1, i].set_title(f"Mask {i}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('data_check.png')
print("✅ Saved data_check.png")
print(f"Image shape: {images.shape}")
print(f"Mask shape: {masks.shape}")
print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
print(f"Mask range: [{masks.min():.2f}, {masks.max():.2f}]")
print(f"Mask unique values: {torch.unique(masks)}")