"""Spatial-Aware Mamba U-Net architecture."""

import torch
import torch.nn as nn
from typing import List

from src.models.components.mamba_block import MambaConvBlock


class SpatialMambaUNet(nn.Module):
    """
    U-Net with Mamba blocks and spatial encoding support.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        use_mamba: bool = True,
        mamba_config: dict = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Channel progression
        channels = [base_channels * (2 ** i) for i in range(5)]
        # [64, 128, 256, 512, 1024]
        
        # ENCODER
        self.encoder1 = self._make_encoder_block(in_channels, channels[0])
        self.encoder2 = self._make_encoder_block(channels[0], channels[1])
        self.encoder3 = self._make_encoder_block(channels[1], channels[2])
        self.encoder4 = self._make_encoder_block(channels[2], channels[3])
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # BOTTLENECK
        self.bottleneck = MambaConvBlock(
            channels[3], channels[4], use_mamba=use_mamba
        )
        
        # DECODER
        self.upconv4 = nn.ConvTranspose2d(channels[4], channels[3], 2, stride=2)
        self.decoder4 = MambaConvBlock(channels[4], channels[3], use_mamba=use_mamba)
        
        self.upconv3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.decoder3 = MambaConvBlock(channels[3], channels[2], use_mamba=use_mamba)
        
        self.upconv2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.decoder2 = MambaConvBlock(channels[2], channels[1], use_mamba=use_mamba)
        
        self.upconv1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.decoder1 = MambaConvBlock(channels[1], channels[0], use_mamba=use_mamba)
        
        # OUTPUT
        self.output_conv = nn.Conv2d(channels[0], out_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def _make_encoder_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ENCODER
        enc1 = self.encoder1(x)
        x = self.pool(enc1)
        
        enc2 = self.encoder2(x)
        x = self.pool(enc2)
        
        enc3 = self.encoder3(x)
        x = self.pool(enc3)
        
        enc4 = self.encoder4(x)
        x = self.pool(enc4)
        
        # BOTTLENECK
        x = self.bottleneck(x)
        
        # DECODER
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.decoder4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.decoder3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder1(x)
        
        # OUTPUT
        x = self.output_conv(x)
        x = self.sigmoid(x)
        
        return x