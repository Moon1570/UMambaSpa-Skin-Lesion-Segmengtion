from pathlib import Path

import pytest
import torch

from src.data.isic_datamodule import ISICDataModule


@pytest.mark.requires_data
@pytest.mark.parametrize("batch_size", [2, 4])
def test_isic_datamodule_basic(batch_size: int) -> None:
    """Tests `ISICDataModule` to verify that dataloaders work correctly and that dtypes and batch
    sizes correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/ISIC2017"
    
    # Skip test if data directory doesn't exist
    if not Path(data_dir).exists():
        pytest.skip(f"Data directory {data_dir} not found")
    
    # Check if train/val directories exist
    train_dir = Path(f"{data_dir}/train/images")
    val_dir = Path(f"{data_dir}/val/images")
    if not train_dir.exists() or not val_dir.exists():
        pytest.skip(f"Train or val directories not found in {data_dir}")

    dm = ISICDataModule(
        data_dir=data_dir,
        train_dir=f"{data_dir}/train",
        val_dir=f"{data_dir}/val",
        test_dir=f"{data_dir}/test",
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        augmentation={"train": {}, "val": {}},
    )

    # Before setup, datasets should be None
    assert dm.data_train is None
    assert dm.data_val is None
    assert dm.data_test is None

    # Setup for training
    dm.setup(stage="fit")
    assert dm.data_train is not None
    assert dm.data_val is not None
    assert dm.train_dataloader() is not None
    assert dm.val_dataloader() is not None

    # Test batch from train loader
    batch = next(iter(dm.train_dataloader()))
    
    # Dataset returns (image, mask) tuple
    assert isinstance(batch, (tuple, list))
    assert len(batch) == 2
    
    image, mask = batch
    
    assert image is not None
    assert mask is not None
    assert len(image) <= batch_size  # Could be less if last batch
    assert len(mask) <= batch_size
    assert image.dtype == torch.float32
    assert mask.dtype in [torch.float32, torch.int64, torch.long]
