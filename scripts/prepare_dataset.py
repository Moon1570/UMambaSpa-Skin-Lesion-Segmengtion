"""
Organize ISIC 2017 into train/val/test structure.
Handles the actual ISIC 2017 Challenge folder structure.

Run: python scripts/prepare_dataset.py
"""

import shutil
from pathlib import Path
from tqdm import tqdm
import os


def find_image_directory(base_path: Path) -> Path:
    """
    Find the actual images directory (may be nested).
    Handles structures like: train_images/ISIC-2017_Training_Data/
    """
    if not base_path.exists():
        return None
    
    # If direct image files exist
    if list(base_path.glob("*.jpg")) or list(base_path.glob("*.png")):
        return base_path
    
    # Look in subdirectories
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            if list(subdir.glob("*.jpg")) or list(subdir.glob("*.png")):
                return subdir
    
    return None


def find_mask_directory(base_path: Path) -> Path:
    """
    Find the actual masks directory (may be nested).
    Handles structures like: train_masks/ISIC-2017_Training_Part1_GroundTruth/
    """
    if not base_path.exists():
        return None
    
    # If direct mask files exist
    if list(base_path.glob("*.png")):
        return base_path
    
    # Look in subdirectories
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            if list(subdir.glob("*.png")):
                return subdir
    
    return None


def should_skip_file(filename: str) -> bool:
    """
    Check if file should be skipped.
    Skip superpixel files and other non-essential files.
    """
    skip_patterns = [
        'superpixel',
        'Superpixel',
        'thumbnail',
        'Thumbnail',
        '.DS_Store',
    ]
    
    return any(pattern in filename for pattern in skip_patterns)


def organize_split(
    source_images_base: Path,
    source_masks_base: Path,
    dest_images: Path,
    dest_masks: Path,
    split_name: str
):
    """Organize one split (train/val/test)."""
    
    print(f"\n📂 Organizing {split_name} split...")
    
    # Find actual directories (may be nested)
    source_images = find_image_directory(source_images_base)
    source_masks = find_mask_directory(source_masks_base)
    
    if source_images is None:
        print(f"   ❌ Could not find images in {source_images_base}")
        print(f"      Available subdirs: {list(source_images_base.iterdir()) if source_images_base.exists() else 'None'}")
        return
    
    if source_masks is None:
        print(f"   ❌ Could not find masks in {source_masks_base}")
        print(f"      Available subdirs: {list(source_masks_base.iterdir()) if source_masks_base.exists() else 'None'}")
        return
    
    print(f"   📁 Images found in: {source_images}")
    print(f"   📁 Masks found in: {source_masks}")
    
    # Create destination directories
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_masks.mkdir(parents=True, exist_ok=True)
    
    # Get all images (exclude superpixel files)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        for img in source_images.glob(ext):
            if not should_skip_file(img.name):
                image_files.append(img)
    
    image_files = sorted(image_files)
    
    print(f"   Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print(f"   ⚠️  No images found!")
        return
    
    # Copy files
    copied_images = 0
    copied_masks = 0
    missing_masks = []
    
    for img_path in tqdm(image_files, desc=f"Copying {split_name}"):
        # Copy image
        dest_img = dest_images / img_path.name
        if not dest_img.exists():
            shutil.copy2(img_path, dest_img)
            copied_images += 1
        
        # Find and copy mask
        # ISIC uses this naming: ISIC_0000000.jpg → ISIC_0000000_segmentation.png
        mask_patterns = [
            img_path.stem + "_segmentation.png",
            img_path.stem + "_Segmentation.png",
            img_path.stem + "_seg.png",
            img_path.stem + ".png",
        ]
        
        mask_found = False
        for pattern in mask_patterns:
            mask_path = source_masks / pattern
            if mask_path.exists():
                # Always save as: ISIC_XXXXXXX_segmentation.png
                dest_mask = dest_masks / f"{img_path.stem}_segmentation.png"
                if not dest_mask.exists():
                    shutil.copy2(mask_path, dest_mask)
                    copied_masks += 1
                mask_found = True
                break
        
        if not mask_found:
            missing_masks.append(img_path.name)
    
    # Report
    print(f"\n   ✅ Copied {copied_images} images")
    print(f"   ✅ Copied {copied_masks} masks")
    
    if missing_masks:
        print(f"   ⚠️  {len(missing_masks)} images missing masks:")
        for name in missing_masks[:5]:
            print(f"      - {name}")
        if len(missing_masks) > 5:
            print(f"      ... and {len(missing_masks) - 5} more")
    else:
        print(f"   ✅ All images have masks!")
    
    # Final count
    final_images = len(list(dest_images.glob('*')))
    final_masks = len(list(dest_masks.glob('*.png')))
    print(f"   📊 Final count: {final_images} images, {final_masks} masks")


def main():
    # Paths
    raw_dir = Path("data/ISIC2017_raw")
    output_dir = Path("data/ISIC2017")
    
    print("🔧 Preparing ISIC 2017 dataset...")
    print(f"📂 Reading from: {raw_dir}")
    print(f"📂 Writing to: {output_dir}")
    
    if not raw_dir.exists():
        print(f"\n❌ Raw data not found at {raw_dir}!")
        print("   Please ensure your data is in data/ISIC2017_raw/")
        print("\n   Expected structure:")
        print("   data/ISIC2017_raw/")
        print("   ├── train_images/ISIC-2017_Training_Data/")
        print("   ├── train_masks/ISIC-2017_Training_Part1_GroundTruth/")
        print("   ├── val_images/ISIC-2017_Validation_Data/")
        print("   ├── val_masks/ISIC-2017_Validation_Part1_GroundTruth/")
        print("   ├── test_images/")
        print("   └── test_masks/")
        return
    
    # Train split
    organize_split(
        source_images_base=raw_dir / "train_images",
        source_masks_base=raw_dir / "train_masks",
        dest_images=output_dir / "train" / "images",
        dest_masks=output_dir / "train" / "masks",
        split_name="train"
    )
    
    # Validation split
    organize_split(
        source_images_base=raw_dir / "val_images",
        source_masks_base=raw_dir / "val_masks",
        dest_images=output_dir / "val" / "images",
        dest_masks=output_dir / "val" / "masks",
        split_name="val"
    )
    
    # Test split
    organize_split(
        source_images_base=raw_dir / "test_images",
        source_masks_base=raw_dir / "test_masks",
        dest_images=output_dir / "test" / "images",
        dest_masks=output_dir / "test" / "masks",
        split_name="test"
    )
    
    print("\n" + "="*60)
    print("✅ Dataset preparation complete!")
    print(f"📁 Dataset ready at: {output_dir}")
    print("\n📊 Summary:")
    
    # Count files in each split
    for split in ['train', 'val', 'test']:
        img_dir = output_dir / split / "images"
        mask_dir = output_dir / split / "masks"
        
        if img_dir.exists():
            n_images = len(list(img_dir.glob('*')))
            n_masks = len(list(mask_dir.glob('*.png')))
            print(f"   {split.capitalize():12s}: {n_images:4d} images, {n_masks:4d} masks")
    
    print("\n⏭️  Next: Run 'python scripts/verify_dataset.py' to verify")


if __name__ == "__main__":
    main()