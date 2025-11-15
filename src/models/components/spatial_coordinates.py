"""
Spatial Coordinate Generator for position-aware features.

Generates X, Y, and radial distance coordinate channels.
"""

import torch
import torch.nn as nn
import numpy as np


class SpatialCoordinateGenerator(nn.Module):
    """
    Generate spatial coordinate channels (X, Y, Radial distance).
    
    These explicit coordinate channels help the network understand
    positional relationships in the image.
    
    Channels:
    - X: Horizontal position (0 to 1, left to right)
    - Y: Vertical position (0 to 1, top to bottom)
    - Radial: Distance from center (0 at center, 1 at corners)
    """
    
    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: If True, normalize coordinates to [0, 1] range
        """
        super().__init__()
        self.normalize = normalize
        self.register_buffer('coords_cache', None)
        self.cache_size = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate spatial coordinate channels.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            coords: Coordinate channels [B, 3, H, W]
                - Channel 0: X coordinates
                - Channel 1: Y coordinates  
                - Channel 2: Radial distance from center
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Use cached coordinates if same size
        if self.cache_size == (H, W) and self.coords_cache is not None:
            coords = self.coords_cache
        else:
            # Generate coordinate grids
            if self.normalize:
                # Normalized coordinates [0, 1]
                y_coords = torch.linspace(0, 1, H, device=device)
                x_coords = torch.linspace(0, 1, W, device=device)
            else:
                # Pixel coordinates
                y_coords = torch.arange(H, dtype=torch.float32, device=device)
                x_coords = torch.arange(W, dtype=torch.float32, device=device)
            
            # Create meshgrid
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # X coordinate channel (horizontal position)
            x_channel = xx.unsqueeze(0)  # [1, H, W]
            
            # Y coordinate channel (vertical position)
            y_channel = yy.unsqueeze(0)  # [1, H, W]
            
            # Radial distance from center
            if self.normalize:
                center_y, center_x = 0.5, 0.5
                max_dist = np.sqrt(0.5**2 + 0.5**2)  # Distance to corner
            else:
                center_y, center_x = H / 2, W / 2
                max_dist = np.sqrt((H/2)**2 + (W/2)**2)
            
            # Compute distance from center
            dist = torch.sqrt((yy - center_y)**2 + (xx - center_x)**2)
            
            # Normalize to [0, 1]
            radial_channel = (dist / max_dist).unsqueeze(0)  # [1, H, W]
            
            # Stack all coordinate channels
            coords = torch.stack([x_channel, y_channel, radial_channel], dim=0).squeeze(1)  # [3, H, W]
            
            # Cache for future use
            self.coords_cache = coords
            self.cache_size = (H, W)
        
        # Expand to batch size
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 3, H, W]
        
        return coords


class SpatialAwareModule(nn.Module):
    """
    Wrapper that adds spatial coordinates to input images.
    
    Converts RGB (3 channels) to RGB+Spatial (6 channels).
    """
    
    def __init__(self, spatial_mode: str = "rgb_xyz_radial", normalize: bool = True):
        """
        Args:
            spatial_mode: Type of spatial encoding
                - "rgb": RGB only (3 channels)
                - "rgb_xy": RGB + X + Y (5 channels)
                - "rgb_xyz_radial": RGB + X + Y + Radial (6 channels)
            normalize: Normalize coordinates to [0, 1]
        """
        super().__init__()
        
        self.spatial_mode = spatial_mode
        self.spatial_gen = SpatialCoordinateGenerator(normalize=normalize)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB image [B, 3, H, W]
            
        Returns:
            out: RGB + Spatial coordinates [B, 3-6, H, W]
        """
        if self.spatial_mode == "rgb":
            # No spatial coordinates
            return x
        
        # Generate spatial coordinates
        coords = self.spatial_gen(x)  # [B, 3, H, W]: X, Y, Radial
        
        if self.spatial_mode == "rgb_xy":
            # RGB + X + Y (5 channels)
            return torch.cat([x, coords[:, :2]], dim=1)
        
        elif self.spatial_mode == "rgb_xyz_radial":
            # RGB + X + Y + Radial (6 channels)
            return torch.cat([x, coords], dim=1)
        
        else:
            raise ValueError(f"Unknown spatial_mode: {self.spatial_mode}")
    
    def get_output_channels(self) -> int:
        """Get number of output channels."""
        if self.spatial_mode == "rgb":
            return 3
        elif self.spatial_mode == "rgb_xy":
            return 5
        elif self.spatial_mode == "rgb_xyz_radial":
            return 6
        else:
            raise ValueError(f"Unknown spatial_mode: {self.spatial_mode}")


# Quick test
if __name__ == "__main__":
    print("Testing SpatialCoordinateGenerator...")
    
    # Create generator
    gen = SpatialCoordinateGenerator(normalize=True)
    
    # Test with sample input
    x = torch.randn(2, 3, 256, 256)
    coords = gen(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Coords shape: {coords.shape}")
    print(f"X range: [{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}]")
    print(f"Y range: [{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]")
    print(f"Radial range: [{coords[:, 2].min():.3f}, {coords[:, 2].max():.3f}]")
    
    # Test wrapper
    print("\nTesting SpatialAwareModule...")
    wrapper = SpatialAwareModule(spatial_mode="rgb_xyz_radial")
    out = wrapper(x)
    print(f"Output shape: {out.shape}")  # Should be [2, 6, 256, 256]
    
    print("\n✅ Spatial coordinates test passed!")