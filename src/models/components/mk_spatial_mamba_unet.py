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
    
    def __init__(self, in_channels: int, out_channels: int, kernels=[1, 3, 5], expansion=4):
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
        
        # Residual connection
        self.residual = None
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
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
        if self.residual is not None:
            identity = self.residual(identity)
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
    
    Architecture (3 stages):
    - Stem: Conv 6→32
    - Encoder: MKIR 32→64→128 (2 stages)
    - Bottleneck: MKIR + Mamba + MKIR (128→128)
    - Decoder: MKIR 128→64→32 (2 stages)
    - Output: Conv 32→1
    
    Total params: ~2.5-3M
    """
    
    def __init__(
        self,
        in_channels=6,
        num_classes=1,
        channels=[32, 64, 128],
        use_spatial=True,
        use_mamba=True,
        use_cbam=True,
        mamba_d_state=16  # Mamba state dimension
    ):
        super().__init__()
        
        self.use_spatial = use_spatial
        self.use_mamba = use_mamba
        self.use_cbam = use_cbam
        
        # Spatial coordinate generator
        if use_spatial:
            self.spatial_gen = SpatialCoordinateGenerator()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder (2 stages)
        self.enc1 = MKIR(channels[0], channels[1])  # 32→64
        self.enc2 = MKIR(channels[1], channels[2])  # 64→128
        
        # Bottleneck (with optional Mamba)
        if use_mamba:
            self.bottleneck_pre = MKIR(channels[2], channels[2])  # 128→128
            self.mamba_block = DirectionalMambaBlock(
                d_model=channels[2],     # ← FIXED!
                d_state=mamba_d_state    # ← FIXED!
            )
            self.bottleneck_post = MKIR(channels[2], channels[2])  # 128→128
        else:
            self.bottleneck = MKIR(channels[2], channels[2])
        
        # Decoder (2 stages)
        self.dec2 = MKIR(channels[2] * 2, channels[1])  # (128+128)→64
        self.dec1 = MKIR(channels[1] * 2, channels[0])  # (64+64)→32
        
        # Upsampling
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # CBAM for skip connections
        if use_cbam:
            self.cbam_enc1 = CBAM(channels[1])  # For enc1 (64 channels)
            self.cbam_enc2 = CBAM(channels[2])  # For enc2 (128 channels)
        
        # Output head
        self.output = nn.Conv2d(channels[0], num_classes, 1)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - C=3 (RGB) or C=6 (RGB+spatial)
        Returns:
            out: [B, 1, H, W] segmentation mask
        """
        # Add spatial coordinates if input is RGB only
        if self.use_spatial and x.shape[1] == 3:
            spatial_coords = self.spatial_gen(x)  # [B, 3, H, W]
            x = torch.cat([x, spatial_coords], dim=1)  # [B, 6, H, W]
        
        # Stem
        x = self.stem(x)  # [B, 32, H, W]
        
        # Encoder
        e1 = self.enc1(self.pool(x))   # [B, 64, H/2, W/2]
        e2 = self.enc2(self.pool(e1))  # [B, 128, H/4, W/4]
        
        # Bottleneck
        if self.use_mamba:
            b = self.pool(e2)              # [B, 128, H/8, W/8]
            b = self.bottleneck_pre(b)     # [B, 128, H/8, W/8]
            b = self.mamba_block(b)        # [B, 128, H/8, W/8]
            b = self.bottleneck_post(b)    # [B, 128, H/8, W/8]
        else:
            b = self.bottleneck(self.pool(e2))  # [B, 128, H/8, W/8]
        
        # Decoder with skip connections
        # Stage 2: 128 → 64
        d2 = self.up2(b)  # [B, 128, H/4, W/4]
        if self.use_cbam:
            e2 = self.cbam_enc2(e2)
        d2 = torch.cat([d2, e2], dim=1)  # [B, 256, H/4, W/4]
        d2 = self.dec2(d2)  # [B, 64, H/4, W/4]
        
        # Stage 1: 64 → 32
        d1 = self.up1(d2)  # [B, 64, H/2, W/2]
        if self.use_cbam:
            e1 = self.cbam_enc1(e1)
        d1 = torch.cat([d1, e1], dim=1)  # [B, 128, H/2, W/2]
        d1 = self.dec1(d1)  # [B, 32, H/2, W/2]
        
        # Final upsample to original size
        d1 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 32, H, W]
        
        # Output
        out = self.output(d1)  # [B, 1, H, W]
        
        return out