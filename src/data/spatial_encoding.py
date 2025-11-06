"""Spatial coordinate encoding for position-aware features."""

import torch
import numpy as np
from typing import Tuple


class SpatialEncoder:
    """
    Generate spatial coordinate channels.
    
    Modes:
    - 'rgb': Just RGB (3 channels)
    - 'rgb_xy': RGB + X,Y coordinates (5 channels)
    - 'rgb_xyz_radial': RGB + X,Y + radial distance (6 channels)
    """
    
    def __init__(
        self,
        mode: str = "rgb",
        normalize: bool = True,
        image_size: Tuple[int, int] = (256, 256)
    ):
        self.mode = mode
        self.normalize = normalize
        self.image_size = image_size
        
        # Pre-compute coordinate grids
        self.coord_grids = self._generate_coordinate_grids()
    
    def _generate_coordinate_grids(self) -> dict:
        """Pre-compute normalized coordinate grids."""
        H, W = self.image_size
        
        # Create meshgrid [0, 1]
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        xv, yv = np.meshgrid(x, y)
        
        # Radial distance from center
        center_x, center_y = 0.5, 0.5
        radial = np.sqrt((xv - center_x)**2 + (yv - center_y)**2)
        
        # Normalize radial to [0, 1]
        if self.normalize:
            radial = radial / radial.max()
        
        return {
            'x': torch.FloatTensor(xv),
            'y': torch.FloatTensor(yv),
            'radial': torch.FloatTensor(radial)
        }
    
    def get_num_channels(self) -> int:
        """Return number of output channels based on mode."""
        channel_map = {
            'rgb': 3,
            'rgb_xy': 5,
            'rgb_xyz_radial': 6
        }
        return channel_map[self.mode]
    
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Add spatial channels to image.
        
        Args:
            image: (C, H, W) RGB image
            
        Returns:
            (C', H, W) with spatial channels
        """
        if self.mode == "rgb":
            return image
        
        C, H, W = image.shape
        device = image.device
        
        # Resize grids if needed
        if (H, W) != self.image_size:
            x_grid = torch.nn.functional.interpolate(
                self.coord_grids['x'].unsqueeze(0).unsqueeze(0),
                size=(H, W), mode='bilinear', align_corners=True
            ).squeeze()
            y_grid = torch.nn.functional.interpolate(
                self.coord_grids['y'].unsqueeze(0).unsqueeze(0),
                size=(H, W), mode='bilinear', align_corners=True
            ).squeeze()
            radial_grid = torch.nn.functional.interpolate(
                self.coord_grids['radial'].unsqueeze(0).unsqueeze(0),
                size=(H, W), mode='bilinear', align_corners=True
            ).squeeze()
        else:
            x_grid = self.coord_grids['x']
            y_grid = self.coord_grids['y']
            radial_grid = self.coord_grids['radial']
        
        # Move to device
        x_grid = x_grid.to(device)
        y_grid = y_grid.to(device)
        radial_grid = radial_grid.to(device)
        
        if self.mode == "rgb_xy":
            spatial_features = torch.stack([x_grid, y_grid], dim=0)
            return torch.cat([image, spatial_features], dim=0)
        
        elif self.mode == "rgb_xyz_radial":
            spatial_features = torch.stack([x_grid, y_grid, radial_grid], dim=0)
            return torch.cat([image, spatial_features], dim=0)
        
        else:
            raise ValueError(f"Unknown spatial mode: {self.mode}")