"""Tests for model components."""
import pytest
import torch

from src.models.components.spatial_mamba_unet import SpatialMambaUNet


def test_unet_instantiation():
    """Test that the U-Net can be instantiated."""
    net = SpatialMambaUNet(
        in_channels=3,
        out_channels=1,
        base_channels=16,
        use_mamba=False,  # Use False for faster testing
    )
    
    assert net is not None
    assert hasattr(net, 'forward')


def test_unet_forward_pass():
    """Test forward pass with dummy data."""
    net = SpatialMambaUNet(
        in_channels=3,
        out_channels=1,
        base_channels=16,
        use_mamba=False,
    )
    
    # Create dummy batch
    batch_size = 2
    image = torch.rand(batch_size, 3, 256, 256)
    
    # Forward pass
    output = net(image)
    
    assert output.shape == (batch_size, 1, 256, 256)


@pytest.mark.parametrize("in_channels", [3, 5, 6])
def test_unet_different_input_channels(in_channels):
    """Test U-Net with different input channel counts."""
    net = SpatialMambaUNet(
        in_channels=in_channels,
        out_channels=1,
        base_channels=16,
        use_mamba=False,
    )
    
    # Create dummy input
    image = torch.rand(2, in_channels, 128, 128)
    
    # Forward pass
    output = net(image)
    
    assert output.shape == (2, 1, 128, 128)


def test_unet_output_shape():
    """Test that output shape matches input spatial dimensions."""
    net = SpatialMambaUNet(
        in_channels=3,
        out_channels=1,
        base_channels=16,
        use_mamba=False,
    )
    
    # Test different input sizes
    for size in [64, 128, 256]:
        image = torch.rand(1, 3, size, size)
        output = net(image)
        assert output.shape == (1, 1, size, size)


def test_unet_batch_processing():
    """Test U-Net processes batches correctly."""
    net = SpatialMambaUNet(
        in_channels=3,
        out_channels=1,
        base_channels=16,
        use_mamba=False,
    )
    
    # Test different batch sizes
    for batch_size in [1, 2, 4]:
        image = torch.rand(batch_size, 3, 128, 128)
        output = net(image)
        assert output.shape[0] == batch_size


def test_unet_output_range():
    """Test that output values are in valid range after sigmoid."""
    net = SpatialMambaUNet(
        in_channels=3,
        out_channels=1,
        base_channels=16,
        use_mamba=False,
    )
    
    image = torch.rand(2, 3, 128, 128)
    output = net(image)
    
    # Output should be in [0, 1] range after sigmoid
    assert output.min() >= 0.0
    assert output.max() <= 1.0
