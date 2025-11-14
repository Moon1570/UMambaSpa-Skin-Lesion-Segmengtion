"""Test that augmentation is actually being applied."""

import matplotlib.pyplot as plt
from src.data.isic_datamodule import ISICDataModule

# Create datamodule
dm = ISICDataModule(
    data_dir="data/ISIC2017",
    train_dir="data/ISIC2017/train",
    val_dir="data/ISIC2017/val",
    spatial_mode="rgb",
    batch_size=1,
    augmentation={
        'train': {
            'horizontal_flip_prob': 0.5,
            'vertical_flip_prob': 0.5,
            'rotate_limit': 30,
            'brightness_limit': 0.2,
            'contrast_limit': 0.2,
            'scale_limit': 0.2,
        }
    }
)

dm.setup()

# Get the same image multiple times (should look different!)
train_dataset = dm.data_train

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Same Image with Different Augmentations', fontsize=16)

# Get first image
for i in range(5):
    img, mask = train_dataset[0]  # Same index, different augmentations!
    
    # Image
    img_np = img[:3].permute(1, 2, 0).numpy()  # Take RGB only
    axes[0, i].imshow(img_np)
    axes[0, i].set_title(f"Augmented {i+1}")
    axes[0, i].axis('off')
    
    # Mask
    mask_np = mask[0].numpy()
    axes[1, i].imshow(mask_np, cmap='gray')
    axes[1, i].set_title(f"Mask {i+1}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('augmentation_test.png', dpi=150)
print("✅ Saved augmentation_test.png")
print("⚠️  If all images look identical, augmentation is NOT working!")
print("✅ If images look different, augmentation IS working!")