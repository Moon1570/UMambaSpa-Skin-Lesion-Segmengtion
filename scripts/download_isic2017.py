"""
Download ISIC 2017 dataset from official source.
Run: python scripts/download_isic2017.py
"""

import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import os


def download_file(url: str, dest: Path):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dest, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip file."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extracted to {extract_to}")


def main():
    # Define paths
    data_root = Path("data")
    raw_dir = data_root / "raw"
    isic2017_dir = data_root / "ISIC2017_raw"  # Temporary extraction
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # ISIC 2017 Challenge URLs
    # Note: You need to register at https://challenge.isic-archive.com/
    # These are example URLs - replace with actual download links
    
    urls = {
        'train_images': 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip',
        'train_masks': 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip',
        'val_images': 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip',
        'val_masks': 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part1_GroundTruth.zip',
        'test_images': 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip',
        'test_masks': 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip',
    }
    
    print("📥 Downloading ISIC 2017 dataset...")
    print("⚠️  This may take a while (several GB)")
    print()
    
    # Download all files
    for name, url in urls.items():
        dest = raw_dir / f"{name}.zip"
        
        if dest.exists():
            print(f"✅ {name}.zip already exists, skipping download")
        else:
            print(f"📥 Downloading {name}...")
            try:
                download_file(url, dest)
            except Exception as e:
                print(f"❌ Failed to download {name}: {e}")
                print(f"   Please download manually from:")
                print(f"   https://challenge.isic-archive.com/data/")
                continue
        
        # Extract
        if not (isic2017_dir / name).exists():
            extract_zip(dest, isic2017_dir / name)
    
    print("\n✅ Download complete!")
    print(f"📁 Files extracted to: {isic2017_dir}")
    print("\n⏭️  Next step: Run 'python scripts/prepare_dataset.py'")


if __name__ == "__main__":
    main()