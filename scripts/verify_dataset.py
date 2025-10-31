"""
Verify ISIC 2017 dataset structure and integrity.
Run: python scripts/verify_dataset.py
"""

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def verify_split(images_dir: Path, masks_dir: Path, split_name: str) -> bool:
    """Verify one split."""
    
    print(f"\n🔍 Verifying {split_name} split...")
    
    # Check directories exist
    if not images_dir.exists():
        print(f"   ❌ Images directory not found: {images_dir}")
        return False
    
    if not masks_dir.exists():
        print(f"   ❌ Masks directory not found: {masks_dir}")
        return False
    
    # Get files
    image_files = sorted(
        list(images_dir.glob("*.jpg")) + 
        list(images_dir.glob("*.jpeg")) +
        list(images_dir.glob("*.png"))
    )
    mask_files = sorted(masks_dir.glob("*.png"))
    
    print(f"   📊 Found {len(image_files)} images, {len(mask_files)} masks")
    
    if len(image_files) == 0:
        print(f"   ❌ No images found!")
        return False
    
    # Check each image has a mask
    missing_masks = []
    invalid_images = []
    invalid_masks = []
    size_mismatches = []
    
    for img_path in tqdm(image_files, desc=f"Checking {split_name}"):
        # Check image is valid
        img = cv2.imread(str(img_path))
        if img is None:
            invalid_images.append(img_path.name)
            continue
        
        # Find corresponding mask
        mask_patterns = [
            img_path.stem + "_segmentation.png",
            img_path.stem + "_Segmentation.png",
            img_path.stem + ".png",
        ]
        
        mask_found = False
        for pattern in mask_patterns:
            mask_path = masks_dir / pattern
            if mask_path.exists():
                mask_found = True
                
                # Verify mask is valid
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    invalid_masks.append(img_path.name)
                elif mask.shape[0] < 10 or mask.shape[1] < 10:
                    invalid_masks.append(img_path.name)
                else:
                    # Check if mask and image have same dimensions
                    if img.shape[:2] != mask.shape[:2]:
                        size_mismatches.append(
                            f"{img_path.name}: img{img.shape[:2]} vs mask{mask.shape[:2]}"
                        )
                
                break
        
        if not mask_found:
            missing_masks.append(img_path.name)
    
    # Report issues
    all_ok = True
    
    if invalid_images:
        print(f"   ❌ {len(invalid_images)} invalid images:")
        for name in invalid_images[:5]:
            print(f"      - {name}")
        all_ok = False
    
    if missing_masks:
        print(f"   ❌ {len(missing_masks)} images missing masks:")
        for name in missing_masks[:5]:
            print(f"      - {name}")
        if len(missing_masks) > 5:
            print(f"      ... and {len(missing_masks) - 5} more")
        all_ok = False
    
    if invalid_masks:
        print(f"   ❌ {len(invalid_masks)} invalid masks:")
        for name in invalid_masks[:5]:
            print(f"      - {name}")
        all_ok = False
    
    if size_mismatches:
        print(f"   ⚠️  {len(size_mismatches)} size mismatches (will be resized during training):")
        for mismatch in size_mismatches[:3]:
            print(f"      - {mismatch}")
    
    if all_ok:
        print(f"   ✅ All images have valid masks!")
    
    return all_ok


def analyze_dataset(images_dir: Path):
    """Analyze dataset statistics."""
    print(f"\n📊 Analyzing dataset...")
    
    # Sample images to get statistics
    image_files = list(images_dir.glob("*.jpg"))[:100]
    
    if not image_files:
        return
    
    sizes = []
    for img_path in tqdm(image_files, desc="Sampling images"):
        img = cv2.imread(str(img_path))
        if img is not None:
            sizes.append(img.shape)
    
    if sizes:
        heights = [s[0] for s in sizes]
        widths = [s[1] for s in sizes]
        channels = [s[2] for s in sizes]
        
        print(f"\n   📏 Image size statistics (sampled {len(sizes)} images):")
        print(f"      Height:   min={min(heights):4d}, max={max(heights):4d}, mean={np.mean(heights):6.1f}")
        print(f"      Width:    min={min(widths):4d}, max={max(widths):4d}, mean={np.mean(widths):6.1f}")
        print(f"      Channels: {set(channels)}")
        
        # Find most common sizes
        size_counts = defaultdict(int)
        for h, w, c in sizes:
            size_counts[(h, w)] += 1
        
        print(f"\n   📊 Most common image sizes:")
        for (h, w), count in sorted(size_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {h}x{w}: {count} images")


def check_naming_convention(images_dir: Path, masks_dir: Path):
    """Check if naming convention is consistent."""
    print(f"\n🔤 Checking naming convention...")
    
    image_files = list(images_dir.glob("*.jpg"))[:10]
    
    for img_path in image_files:
        mask_path = masks_dir / f"{img_path.stem}_segmentation.png"
        if mask_path.exists():
            print(f"   ✅ {img_path.name} → {mask_path.name}")
        else:
            print(f"   ❌ {img_path.name} → mask not found")


def main():
    dataset_dir = Path("data/ISIC2017")
    
    if not dataset_dir.exists():
        print("❌ Dataset not found!")
        print("   Run 'python scripts/prepare_dataset.py' first")
        return
    
    print("🔍 Verifying ISIC 2017 dataset structure...")
    print(f"📂 Dataset directory: {dataset_dir}")
    
    # Verify each split
    train_ok = verify_split(
        dataset_dir / "train" / "images",
        dataset_dir / "train" / "masks",
        "train"
    )
    
    val_ok = verify_split(
        dataset_dir / "val" / "images",
        dataset_dir / "val" / "masks",
        "val"
    )
    
    test_ok = verify_split(
        dataset_dir / "test" / "images",
        dataset_dir / "test" / "masks",
        "test"
    )
    
    # Analyze dataset
    analyze_dataset(dataset_dir / "train" / "images")
    
    # Check naming convention
    check_naming_convention(
        dataset_dir / "train" / "images",
        dataset_dir / "train" / "masks"
    )
    
    # Final verdict
    print("\n" + "="*60)
    if train_ok and val_ok and test_ok:
        print("✅ Dataset verification PASSED!")
        print("\n📊 Dataset Statistics:")
        for split in ['train', 'val', 'test']:
            img_dir = dataset_dir / split / "images"
            if img_dir.exists():
                n_images = len(list(img_dir.glob('*')))
                print(f"   {split.capitalize():12s}: {n_images:4d} images")
        
        print("\n⏭️  Ready to train! Run:")
        print("   python src/train.py trainer.fast_dev_run=True")
    else:
        print("⚠️  Dataset verification found issues")
        print("   Please fix the issues above before training")
        print("\n   Common fixes:")
        print("   1. Re-run: python scripts/prepare_dataset.py")
        print("   2. Check raw data structure in data/ISIC2017_raw/")


if __name__ == "__main__":
    main()