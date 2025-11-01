"""Tests for ISICDataModule."""
import pytest
import torch
from pathlib import Path

from src.data.isic_datamodule import ISICDataModule


@pytest.mark.requires_data
def test_isic_datamodule_initialization():
    """Test ISICDataModule initialization."""
    data_dir = "data/ISIC2017"
    
    # Skip test if data directory doesn't exist
    if not Path(data_dir).exists():
        pytest.skip(f"Data directory {data_dir} not found")
    
    dm = ISICDataModule(
        data_dir=data_dir,
        train_dir=f"{data_dir}/train",
        val_dir=f"{data_dir}/val",
        test_dir=f"{data_dir}/test",
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        augmentation={"train": {}, "val": {}},
    )
    
    # Test basic attributes stored in hparams
    assert dm.hparams.batch_size == 2
    assert dm.hparams.num_workers == 0
    assert dm.hparams.pin_memory is False
    
    # Check that datasets are not loaded yet
    assert dm.data_train is None
    assert dm.data_val is None


@pytest.mark.requires_data
def test_isic_datamodule_setup():
    """Test ISICDataModule setup method."""
    data_dir = "data/ISIC2017"
    
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
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        augmentation={"train": {}, "val": {}},
    )
    
    # Setup for training
    dm.setup(stage="fit")
    
    # Check datasets are loaded
    assert dm.data_train is not None
    assert dm.data_val is not None
    assert len(dm.data_train) > 0
    assert len(dm.data_val) > 0


@pytest.mark.requires_data
def test_isic_datamodule_dataloaders():
    """Test ISICDataModule dataloaders."""
    data_dir = "data/ISIC2017"
    
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
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        augmentation={"train": {}, "val": {}},
    )
    
    dm.setup(stage="fit")
    
    # Test train dataloader
    train_loader = dm.train_dataloader()
    assert train_loader is not None
    assert len(train_loader) > 0
    
    # Test validation dataloader
    val_loader = dm.val_dataloader()
    assert val_loader is not None
    assert len(val_loader) > 0
    
    # Test batch from train loader
    batch = next(iter(train_loader))
    assert isinstance(batch, (dict, tuple, list))
    
    # If batch is a dict, check for image and mask keys
    if isinstance(batch, dict):
        assert "image" in batch or "img" in batch
        assert "mask" in batch or "label" in batch or "target" in batch
        
        # Get image tensor
        image = batch.get("image") or batch.get("img")
        assert image is not None
        assert isinstance(image, torch.Tensor)
        assert image.shape[0] == 2  # batch_size
        
        # Get mask tensor
        mask = batch.get("mask") or batch.get("label") or batch.get("target")
        assert mask is not None
        assert isinstance(mask, torch.Tensor)
        assert mask.shape[0] == 2  # batch_size
    
    # If batch is a tuple/list
    elif isinstance(batch, (tuple, list)):
        assert len(batch) >= 2
        image, mask = batch[0], batch[1]
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert image.shape[0] == 2  # batch_size
        assert mask.shape[0] == 2  # batch_size


@pytest.mark.requires_data
def test_isic_datamodule_test_setup():
    """Test ISICDataModule test setup."""
    data_dir = "data/ISIC2017"
    
    if not Path(data_dir).exists():
        pytest.skip(f"Data directory {data_dir} not found")
    
    # Check if test directory exists
    test_dir = Path(f"{data_dir}/test/images")
    if not test_dir.exists():
        pytest.skip(f"Test directory not found in {data_dir}")
    
    dm = ISICDataModule(
        data_dir=data_dir,
        test_dir=f"{data_dir}/test",
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        augmentation={"train": {}, "val": {}},
    )
    
    # Setup for testing
    dm.setup(stage="test")
    
    # Check test dataset is loaded
    assert hasattr(dm, 'data_test')
    assert dm.data_test is not None
    
    # Test test dataloader
    test_loader = dm.test_dataloader()
    assert test_loader is not None
    assert len(test_loader) > 0


@pytest.mark.requires_data
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_isic_datamodule_batch_sizes(batch_size):
    """Test ISICDataModule with different batch sizes."""
    data_dir = "data/ISIC2017"
    
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
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        augmentation={"train": {}, "val": {}},
    )
    
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    
    batch = next(iter(train_loader))
    
    # Get the actual batch tensor
    if isinstance(batch, dict):
        image = batch.get("image") or batch.get("img")
    else:
        image = batch[0]
    
    # Check batch size (last batch might be smaller)
    assert image.shape[0] <= batch_size