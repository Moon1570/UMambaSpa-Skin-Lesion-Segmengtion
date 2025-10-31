"""Mamba block with auto-detection of real vs mock implementation."""

import torch
import torch.nn as nn
from typing import Optional


# Use our factory function instead of direct import
from src.models.components.mamba_mock import Mamba


class MambaBlock2D(nn.Module):
    """
    Mamba block for 2D feature maps.
    Processes spatial features sequentially using Mamba SSM.
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        
        self.dim = dim
        
        # Mamba block for sequential processing
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) tensor
        Returns:
            (B, C, H, W) tensor
        """
        B, C, H, W = x.shape
        
        # Reshape to (B, H*W, C) for sequential processing
        # Use contiguous() to ensure memory layout is contiguous
        x_flat = x.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
        
        # Process through Mamba
        x = self.mamba(x_flat)
        
        # Reshape back to (B, C, H, W)
        # Use contiguous() after reshape and permute
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        # Apply normalization (on spatial dimensions)
        # Normalize along channel dimension for each spatial location
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        
        return x


class MambaConvBlock(nn.Module):
    """
    Combined convolutional and Mamba block.
    Similar to standard UNet conv block but with Mamba for global context.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_mamba: bool = True,  # Added parameter
    ):
        super().__init__()
        
        self.use_mamba = use_mamba
        
        # Standard convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)  # Changed to inplace=False for better compatibility
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Mamba block for global context (only if use_mamba is True)
        if self.use_mamba:
            self.mamba = MambaBlock2D(
                dim=out_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) tensor
        Returns:
            (B, C_out, H, W) tensor
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Mamba block for global context (if enabled)
        if self.use_mamba and self.mamba is not None:
            # Ensure contiguous before Mamba
            x = x.contiguous()
            
            # Mamba block for global context
            x = self.mamba(x)
            
            # Ensure contiguous output
            x = x.contiguous()
        
        return x