"""Enhanced MK-Spatial U-Net with modular improvements."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class MKIR_SE(nn.Module):
    """Multi-Kernel Inverted Residual with Squeeze-and-Excitation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 2,
        kernels: List[int] = [3, 5, 7],
        use_se: bool = True,
    ):
        super().__init__()
        
        hidden = in_channels * expansion
        self.residual = in_channels == out_channels
        
        # Pointwise expansion
        self.pw1 = nn.Conv2d(in_channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act1 = nn.ReLU6(inplace=True)
        
        # Multi-kernel depthwise
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(hidden, hidden, k, padding=k//2, groups=hidden, bias=False)
            for k in kernels
        ])
        self.dw_bns = nn.ModuleList([nn.BatchNorm2d(hidden) for _ in kernels])
        
        # Pointwise projection
        self.pw2 = nn.Conv2d(hidden, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Squeeze-and-Excitation
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(out_channels, reduction=4)
    
    def forward(self, x):
        identity = x
        
        # Expansion
        x = self.act1(self.bn1(self.pw1(x)))
        
        # Multi-kernel depthwise (average fusion)
        dw_outs = []
        for conv, bn in zip(self.dw_convs, self.dw_bns):
            dw_outs.append(bn(conv(x)))
        x = sum(dw_outs) / len(dw_outs)
        
        # Projection
        x = self.bn2(self.pw2(x))
        
        # SE attention
        if self.use_se:
            x = self.se(x)
        
        # Residual
        if self.residual:
            x = x + identity
        
        return x


class SpatialCoordinateGenerator(nn.Module):
    """Generate spatial coordinate features (X, Y, Radial)."""
    
    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        
        # Normalized coordinates [-1, 1]
        y_coords = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        
        # Radial distance from center
        radial = torch.sqrt(x_coords ** 2 + y_coords ** 2)
        
        return torch.cat([x_coords, y_coords, radial], dim=1)


class MKEnhancedUNet(nn.Module):
    """
    Enhanced Multi-Kernel U-Net with modular improvements.
    
    Supports:
    - Squeeze-and-Excitation (SE) attention
    - Deep Supervision (DS)
    - Boundary-aware loss (via boundary maps)
    - Spatial coordinate encoding
    """
    
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 1,
        channels: List[int] = [16, 32, 64, 96],  # Reduced for speed
        use_spatial: bool = True,
        use_se: bool = False,
        deep_supervision: bool = False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_spatial = use_spatial
        self.use_se = use_se
        self.deep_supervision = deep_supervision
        
        # Spatial coordinate generator
        if use_spatial:
            self.spatial_gen = SpatialCoordinateGenerator()
            actual_in = 6  # RGB + spatial coords (X, Y, Radial)
        else:
            actual_in = in_channels
        
        # Encoder
        self.enc1 = MKIR_SE(actual_in, channels[0], use_se=use_se)
        self.enc2 = MKIR_SE(channels[0], channels[1], use_se=use_se)
        self.enc3 = MKIR_SE(channels[1], channels[2], use_se=use_se)
        self.enc4 = MKIR_SE(channels[2], channels[3], use_se=use_se)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck (no Mamba for speed)
        self.bottleneck = nn.Sequential(
            MKIR_SE(channels[3], channels[3] * 2, use_se=use_se),
            MKIR_SE(channels[3] * 2, channels[3] * 2, use_se=use_se),
        )
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(channels[3] * 2, channels[3], 2, stride=2)
        self.dec4 = MKIR_SE(channels[3] * 2, channels[3], use_se=use_se)
        
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3 = MKIR_SE(channels[2] * 2, channels[2], use_se=use_se)
        
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2 = MKIR_SE(channels[1] * 2, channels[1], use_se=use_se)
        
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.dec1 = MKIR_SE(channels[0] * 2, channels[0], use_se=use_se)
        
        # Output heads (no sigmoid - will use BCEWithLogitsLoss)
        self.output = nn.Conv2d(channels[0], num_classes, 1)
        
        # Deep supervision auxiliary outputs
        if deep_supervision:
            self.aux_out4 = nn.Conv2d(channels[3], num_classes, 1)
            self.aux_out3 = nn.Conv2d(channels[2], num_classes, 1)
            self.aux_out2 = nn.Conv2d(channels[1], num_classes, 1)
        
        self._log_architecture()
    
    def _log_architecture(self):
        """Log architecture configuration."""
        if self.use_spatial:
            print("✅ Using spatial coordinate encoding (X, Y, Radial)")
        if self.use_se:
            print("✅ Using Squeeze-and-Excitation attention")
        if self.deep_supervision:
            print("✅ Using Deep Supervision (4 output levels)")
        print("✅ Pure CNN architecture (no Mamba for fast training)")
    
    def forward(self, x):
        # Add spatial coordinates if enabled
        if self.use_spatial and x.shape[1] == 3:
            spatial_coords = self.spatial_gen(x)
            x = torch.cat([x, spatial_coords], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        # Main output
        out = self.output(d1)
        
        # Deep supervision outputs
        if self.training and self.deep_supervision:
            aux4 = F.interpolate(self.aux_out4(d4), size=x.shape[2:], 
                                mode='bilinear', align_corners=False)
            aux3 = F.interpolate(self.aux_out3(d3), size=x.shape[2:], 
                                mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_out2(d2), size=x.shape[2:], 
                                mode='bilinear', align_corners=False)
            return out, [aux4, aux3, aux2]
        
        return out


if __name__ == "__main__":
    # Test the enhanced model
    print("Testing MKEnhancedUNet configurations...\n")
    
    # Test 1: Base model
    print("1. Base Model (no enhancements):")
    model1 = MKEnhancedUNet(use_se=False, deep_supervision=False)
    x = torch.randn(2, 3, 256, 256)
    y = model1(x)
    params1 = sum(p.numel() for p in model1.parameters()) / 1e6
    print(f"   Params: {params1:.2f}M\n")
    
    # Test 2: With SE
    print("2. With Squeeze-and-Excitation:")
    model2 = MKEnhancedUNet(use_se=True, deep_supervision=False)
    y = model2(x)
    params2 = sum(p.numel() for p in model2.parameters()) / 1e6
    print(f"   Params: {params2:.2f}M\n")
    
    # Test 3: With Deep Supervision
    print("3. With Deep Supervision:")
    model3 = MKEnhancedUNet(use_se=False, deep_supervision=True)
    model3.train()
    out = model3(x)
    params3 = sum(p.numel() for p in model3.parameters()) / 1e6
    print(f"   Main output: {out[0].shape}")
    print(f"   Aux outputs: {[aux.shape for aux in out[1]]}")
    print(f"   Params: {params3:.2f}M\n")
    
    # Test 4: Full model
    print("4. Full Model (SE + DS):")
    model4 = MKEnhancedUNet(use_se=True, deep_supervision=True)
    model4.train()
    out = model4(x)
    params4 = sum(p.numel() for p in model4.parameters()) / 1e6
    print(f"   Params: {params4:.2f}M\n")
    
    print("✅ All configurations working!")
