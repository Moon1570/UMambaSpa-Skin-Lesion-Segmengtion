"""
Mock Mamba implementation for local testing without CUDA.
SOLUTION 1: Disable cuDNN for GRU operations only.
"""

import torch
import torch.nn as nn
import warnings
from typing import Optional


class MambaMock(nn.Module):
    """Mock Mamba implementation using GRU for testing without mamba-ssm."""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, **kwargs):
        super().__init__()
        warnings.warn("Using MOCK Mamba (for testing without mamba-ssm)")
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        
        # Use GRU as a sequential processor (similar to Mamba's SSM)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) tensor
        Returns:
            (B, L, D) tensor
        """
        B, L, D = x.shape
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # SOLUTION 1: Disable cuDNN for GRU forward pass
        # This fixes CUDNN_STATUS_NOT_SUPPORTED error
        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        
        try:
            # Process through GRU (without cuDNN)
            x_seq, _ = self.gru(x)
        finally:
            # Restore cuDNN state
            torch.backends.cudnn.enabled = cudnn_enabled
        
        # Ensure output is contiguous
        x_seq = x_seq.contiguous()
        
        # Apply layer norm
        x_out = self.norm(x_seq)
        
        return x_out


# Auto-detect which Mamba to use
try:
    from mamba_ssm import Mamba as MambaReal
    MAMBA_AVAILABLE = True
    print("✅ Real Mamba (mamba-ssm) is available")
except ImportError:
    MambaReal = None
    MAMBA_AVAILABLE = False
    print("⚠️  mamba-ssm not available, will use mock implementation")


def Mamba(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    use_mock: bool = False,
    **kwargs
):
    """
    Factory function that returns real or mock Mamba.
    
    Args:
        use_mock: Force use of mock implementation (for testing)
        
    Returns:
        Real Mamba if available and not forced to mock, otherwise Mock Mamba
    """
    if use_mock or not MAMBA_AVAILABLE:
        return MambaMock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs
        )
    else:
        return MambaReal(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs
        )


