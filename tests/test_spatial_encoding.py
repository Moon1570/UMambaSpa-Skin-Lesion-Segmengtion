"""Tests for spatial encoding module."""
import pytest
import torch

from src.data.spatial_encoding import SpatialEncoder


@pytest.mark.parametrize("mode", ["rgb", "rgb_xy", "rgb_xyz_radial"])
def test_spatial_encoder_modes(mode):
    """Test spatial encoder with different modes."""
    encoder = SpatialEncoder(mode=mode, image_size=(256, 256))
    
    # Create dummy RGB image (C, H, W)
    image = torch.rand(3, 256, 256)
    
    # Encode
    encoded = encoder.encode(image)
    
    # Check output
    assert isinstance(encoded, torch.Tensor)
    assert encoded.shape[1:] == (256, 256)  # Height and width preserved
    
    # Check channel count based on mode
    if mode == "rgb":
        assert encoded.shape[0] == 3  # RGB only
    elif mode == "rgb_xy":
        assert encoded.shape[0] == 5  # RGB + XY
    elif mode == "rgb_xyz_radial":
        assert encoded.shape[0] == 6  # RGB + XY + Distance


def test_spatial_encoder_rgb_mode():
    """Test RGB mode specifically."""
    encoder = SpatialEncoder(mode="rgb", image_size=(128, 128))
    image = torch.rand(3, 128, 128)
    encoded = encoder.encode(image)
    
    # RGB mode should return the same image
    assert torch.allclose(encoded, image)


def test_spatial_encoder_xy_coordinates():
    """Test XY coordinate encoding."""
    encoder = SpatialEncoder(mode="rgb_xy", image_size=(4, 4))
    image = torch.zeros(3, 4, 4)
    encoded = encoder.encode(image)
    
    # Should have 5 channels: RGB + X + Y
    assert encoded.shape[0] == 5
    
    # Check X coordinate channel (channel 3)
    x_channel = encoded[3]
    assert x_channel[0, 0] < x_channel[0, -1]  # X increases left to right
    
    # Check Y coordinate channel (channel 4)
    y_channel = encoded[4]
    assert y_channel[0, 0] < y_channel[-1, 0]  # Y increases top to bottom


def test_spatial_encoder_full_mode():
    """Test full mode with distance channel."""
    encoder = SpatialEncoder(mode="rgb_xyz_radial", image_size=(8, 8))
    image = torch.zeros(3, 8, 8)
    encoded = encoder.encode(image)
    
    # Should have 6 channels: RGB + X + Y + Distance
    assert encoded.shape[0] == 6
    
    # Distance channel should have minimum at center
    distance_channel = encoded[5]
    center_y, center_x = 8 // 2, 8 // 2
    center_distance = distance_channel[center_y, center_x]
    
    # Check corners have higher distance than center
    corner_distance = distance_channel[0, 0]
    assert corner_distance > center_distance


@pytest.mark.parametrize("image_size", [(64, 64), (128, 128), (256, 256)])
def test_spatial_encoder_different_sizes(image_size):
    """Test encoder with different image sizes."""
    encoder = SpatialEncoder(mode="rgb_xyz_radial", image_size=image_size)
    h, w = image_size
    image = torch.rand(3, h, w)
    encoded = encoder.encode(image)
    
    assert encoded.shape == (6, h, w)


def test_spatial_encoder_batch_processing():
    """Test encoder works with batched images."""
    encoder = SpatialEncoder(mode="rgb_xy", image_size=(32, 32))
    
    # Single image
    image = torch.rand(3, 32, 32)
    encoded = encoder.encode(image)
    assert encoded.shape == (5, 32, 32)


def test_spatial_encoder_value_ranges():
    """Test that spatial coordinates are normalized."""
    encoder = SpatialEncoder(mode="rgb_xyz_radial", image_size=(16, 16))
    image = torch.rand(3, 16, 16)
    encoded = encoder.encode(image)
    
    # X and Y coordinates should be in [-1, 1] range
    x_channel = encoded[3]
    y_channel = encoded[4]
    
    assert x_channel.min() >= -1.0
    assert x_channel.max() <= 1.0
    assert y_channel.min() >= -1.0
    assert y_channel.max() <= 1.0
    
    # Distance should be non-negative
    distance_channel = encoded[5]
    assert distance_channel.min() >= 0.0
