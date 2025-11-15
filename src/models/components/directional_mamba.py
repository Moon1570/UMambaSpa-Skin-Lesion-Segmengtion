"""
Directional Mamba for proper 2D spatial scanning.
Based on VMamba (ECCV 2024) and Vision Mamba principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("⚠️  mamba-ssm not available, will use mock implementation")


class MambaMock(nn.Module):
    """Mock Mamba using GRU for local testing."""
    
    def __init__(self, d_model: int, d_state: int = 16, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        print("⚠️  Using MOCK Mamba (GRU-based)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D]"""
        identity = x
        x_seq, _ = self.gru(x)
        out = x_seq + identity
        return self.norm(out)


def get_mamba(d_model: int, d_state: int = 16):
    """Factory function for Mamba."""
    if MAMBA_AVAILABLE:
        return Mamba(d_model=d_model, d_state=d_state)
    else:
        return MambaMock(d_model=d_model, d_state=d_state)


class DirectionalMamba2D(nn.Module):
    """
    Directional Mamba for 2D feature maps.
    
    Scans in 4 directions to preserve spatial relationships:
    - Horizontal: left→right, right→left
    - Vertical: top→bottom, bottom→top
    
    This properly handles 2D spatial data unlike naive flattening.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        num_directions: int = 4,
        use_bidirectional: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_directions = num_directions
        self.use_bidirectional = use_bidirectional
        
        # Separate Mamba for each direction
        self.mamba_h = get_mamba(d_model, d_state)  # Horizontal
        self.mamba_v = get_mamba(d_model, d_state)  # Vertical
        
        # Merge features from all directions
        merge_channels = d_model * num_directions
        self.merge = nn.Sequential(
            nn.Conv2d(merge_channels, d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
    
    def scan_horizontal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scan horizontally (left-to-right and right-to-left).
        
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Left-to-right scan
        # Reshape: [B, C, H, W] → [B, H, W, C] → [B*H, W, C]
        x_h = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x_h = x_h.view(B * H, W, C)  # Treat each row as a sequence
        
        # Apply Mamba
        out_lr = self.mamba_h(x_h)  # [B*H, W, C]
        out_lr = out_lr.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        if not self.use_bidirectional:
            return out_lr
        
        # Right-to-left scan (flip, process, flip back)
        x_h_flip = torch.flip(x, [3])  # Flip along width
        x_h_flip = x_h_flip.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
        
        out_rl = self.mamba_h(x_h_flip)
        out_rl = out_rl.view(B, H, W, C).permute(0, 3, 1, 2)
        out_rl = torch.flip(out_rl, [3])  # Flip back
        
        # Average both directions
        return (out_lr + out_rl) / 2
    
    def scan_vertical(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scan vertically (top-to-bottom and bottom-to-top).
        
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Top-to-bottom scan
        # Reshape: [B, C, H, W] → [B, W, H, C] → [B*W, H, C]
        x_v = x.permute(0, 3, 2, 1).contiguous()  # [B, W, H, C]
        x_v = x_v.view(B * W, H, C)  # Treat each column as a sequence
        
        # Apply Mamba
        out_tb = self.mamba_v(x_v)  # [B*W, H, C]
        out_tb = out_tb.view(B, W, H, C).permute(0, 3, 2, 1)  # [B, C, H, W]
        
        if not self.use_bidirectional:
            return out_tb
        
        # Bottom-to-top scan (flip, process, flip back)
        x_v_flip = torch.flip(x, [2])  # Flip along height
        x_v_flip = x_v_flip.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)
        
        out_bt = self.mamba_v(x_v_flip)
        out_bt = out_bt.view(B, W, H, C).permute(0, 3, 2, 1)
        out_bt = torch.flip(out_bt, [2])  # Flip back
        
        # Average both directions
        return (out_tb + out_bt) / 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply directional Mamba scanning.
        
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        identity = x
        
        # Scan in multiple directions
        out_h = self.scan_horizontal(x)  # ←→
        out_v = self.scan_vertical(x)    # ↑↓
        
        # Concatenate features from all directions
        out = torch.cat([out_h, out_v], dim=1)  # [B, C*2, H, W]
        
        # If 4 directions, add diagonal scans
        if self.num_directions == 4:
            # For now, use horizontal and vertical twice
            # Full diagonal scanning is complex, can be added later
            out = torch.cat([out, out_h, out_v], dim=1)  # [B, C*4, H, W]
        
        # Merge all directions
        out = self.merge(out)  # [B, C, H, W]
        
        # Residual connection
        out = out + identity
        
        # Layer normalization (channel-wise)
        out = out.permute(0, 2, 3, 1)  # [B, H, W, C]
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return out


class DirectionalMambaBlock(nn.Module):
    """
    Mamba block with pre-normalization and residual connection.
    """
    
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = DirectionalMamba2D(d_model, d_state, num_directions=4)
        
        # MLP
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        """
        # Mamba block
        identity = x
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.mamba(x)
        x = x + identity
        
        # MLP block
        identity = x
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, C]
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        x = x + identity
        
        return x