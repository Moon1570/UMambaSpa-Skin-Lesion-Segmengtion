"""
MK-Spatial-Mamba U-Net: Hybrid architecture combining:
- MK-UNet's lightweight multi-kernel blocks (encoder/decoder)
- Directional Mamba (bottleneck only)
- Spatial coordinate encoding
- CBAM attention on skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.directional_mamba import DirectionalMambaBlock
from src.models.components.spatial_coordinates import SpatialCoordinateGenerator


class MKIR(nn.Module):
    """
    Multi-Kernel Inverted Residual block (from MK-UNet).
    Lightweight feature extraction using depth-wise convolutions.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernels=[1, 3, 5], expansion=2):
        super().__init__()
        
        hidden_channels = in_channels * expansion
        
        # Point-wise expansion
        self.pw1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ReLU6(inplace=True)
        
        # Multi-kernel depth-wise convolution
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, k, padding=k//2, groups=hidden_channels)
            for k in kernels
        ])
        self.dw_bns = nn.ModuleList([nn.BatchNorm2d(hidden_channels) for _ in kernels])
        
        # Point-wise projection
        self.pw2 = nn.Conv2d(hidden_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Residual
        self.residual = in_channels == out_channels
    
    def forward(self, x):
        identity = x
        
        # Expansion
        x = self.act(self.bn1(self.pw1(x)))
        
        # Multi-kernel depth-wise conv
        dw_outs = []
        for dw, bn in zip(self.dw_convs, self.dw_bns):
            dw_out = self.act(bn(dw(x)))
            dw_outs.append(dw_out)
        
        # Sum multi-kernel outputs
        x = sum(dw_outs) / len(dw_outs)
        
        # Projection
        x = self.bn2(self.pw2(x))
        
        # Residual
        if self.residual:
            x = x + identity
        
        return x


class CBAM(nn.Module):
    """Channel and Spatial Attention Module."""
    
    def __init__(self, channels: int, reduction=16):
        super().__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x


class MKSpatialMambaUNet(nn.Module):
    """
    Hybrid architecture: MK-UNet + Directional Mamba + Spatial Features
    
    Architecture:
    - Encoder: Lightweight MKIR blocks
    - Bottleneck: Directional Mamba (proper 2D scanning)
    - Decoder: MKIR blocks with CBAM skip connections
    - Input: RGB + Spatial coordinates (6 channels)
    """
    
    def __init__(
        self,
        in_channels: int = 6,  # RGB + X + Y + Radial
        num_classes: int = 1,
        channels: list = [16, 32, 64, 96, 160],
        use_spatial: bool = True,
        use_mamba: bool = True,  # Can disable for ablation
        use_cbam: bool = True
    ):
        super().__init__()
        
        self.use_spatial = use_spatial
        self.use_mamba = use_mamba
        self.use_cbam = use_cbam
        
        # Spatial coordinate generator
        if use_spatial:
            self.spatial_gen = SpatialCoordinateGenerator(normalize=True)
            actual_in_channels = 6  # RGB + X + Y + Radial
        else:
            actual_in_channels = 3  # RGB only
        
        # Encoder (lightweight MK blocks)
        self.encoder1 = MKIR(actual_in_channels, channels[0])
        self.encoder2 = MKIR(channels[0], channels[1])
        self.encoder3 = MKIR(channels[1], channels[2])
        self.encoder4 = MKIR(channels[2], channels[3])
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        if use_mamba:
            print("✅ Using Directional Mamba in bottleneck")
            self.bottleneck = nn.Sequential(
                MKIR(channels[3], channels[4]),
                DirectionalMambaBlock(channels[4], d_state=16),
                MKIR(channels[4], channels[4])
            )
        else:
            print("✅ Using pure CNN in bottleneck (no Mamba)")
            self.bottleneck = nn.Sequential(
                MKIR(channels[3], channels[4]),
                MKIR(channels[4], channels[4]),
                MKIR(channels[4], channels[4])
            )
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(channels[4], channels[3], 2, stride=2)
        self.decoder4 = MKIR(channels[3] * 2, channels[3])  # Skip concat doubles channels
        
        self.upconv3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.decoder3 = MKIR(channels[2] * 2, channels[2])
        
        self.upconv2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.decoder2 = MKIR(channels[1] * 2, channels[1])
        
        self.upconv1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.decoder1 = MKIR(channels[0] * 2, channels[0])
        
        # CBAM attention on skip connections
        if use_cbam:
            print("✅ Using CBAM attention on skip connections")
            self.cbam1 = CBAM(channels[0])
            self.cbam2 = CBAM(channels[1])
            self.cbam3 = CBAM(channels[2])
            self.cbam4 = CBAM(channels[3])
        
        # Output
        self.output = nn.Conv2d(channels[0], num_classes, 1)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - C=3 (RGB) or C=6 (RGB+spatial)
        Returns:
            out: [B, 1, H, W] segmentation mask
        """
        # Add spatial coordinates only if input is RGB (3 channels)
        # If input already has 6 channels, dataloader added spatial coords
        if self.use_spatial and x.shape[1] == 3:
            spatial_coords = self.spatial_gen(x)  # [B, 3, H, W]
            x = torch.cat([x, spatial_coords], dim=1)  # [B, 6, H, W]
        
        # Encoder
        e1 = self.encoder1(x)              # [B, 16, H, W]
        e2 = self.encoder2(self.pool(e1))  # [B, 32, H/2, W/2]
        e3 = self.encoder3(self.pool(e2))  # [B, 64, H/4, W/4]
        e4 = self.encoder4(self.pool(e3))  # [B, 96, H/8, W/8]
        
        # Bottleneck (with or without Mamba)
        b = self.bottleneck(self.pool(e4))  # [B, 160, H/16, W/16]
        
        # Decoder with skip connections
        d4 = self.upconv4(b)  # [B, 96, H/8, W/8]
        if self.use_cbam:
            e4 = self.cbam4(e4)
        d4 = torch.cat([d4, e4], dim=1)  # [B, 192, H/8, W/8]
        d4 = self.decoder4(d4)  # [B, 96, H/8, W/8]
        
        d3 = self.upconv3(d4)
        if self.use_cbam:
            e3 = self.cbam3(e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        if self.use_cbam:
            e2 = self.cbam2(e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        if self.use_cbam:
            e1 = self.cbam1(e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        # Output
        out = self.output(d1)
        # out = torch.sigmoid(out) # Commented out for Mixed precision error
        
        return out