"""Quick test to verify architecture works."""

import torch
from src.models.components.mk_spatial_mamba_unet import MKSpatialMambaUNet
from src.models.mk_spatial_mamba_module import MKSpatialMambaLitModule

# Test with mock Mamba (for local GTX 1070)
print("=" * 60)
print("Testing MKSpatialMambaUNet (Base Model)")
print("=" * 60)

# Test hybrid (with Mamba)
print("\n1. Testing Hybrid (Mamba + CNN):")
model_hybrid = MKSpatialMambaUNet(
    in_channels=6,
    num_classes=1,
    channels=[16, 32, 64, 96, 160],
    use_spatial=True,
    use_mamba=True,
    use_cbam=True
)

x = torch.randn(2, 3, 256, 256)  # RGB input
y = model_hybrid(x)
print(f"   Input: {x.shape}")
print(f"   Output: {y.shape}")
print(f"   Params: {sum(p.numel() for p in model_hybrid.parameters()) / 1e6:.2f}M")

# Test pure CNN (no Mamba)
print("\n2. Testing Pure CNN (no Mamba):")
model_cnn = MKSpatialMambaUNet(
    in_channels=6,
    num_classes=1,
    channels=[16, 32, 64, 96, 160],
    use_spatial=True,
    use_mamba=False,  # No Mamba
    use_cbam=True
)

y = model_cnn(x)
print(f"   Input: {x.shape}")
print(f"   Output: {y.shape}")
print(f"   Params: {sum(p.numel() for p in model_cnn.parameters()) / 1e6:.2f}M")

print("\n✅ Base model test passed!")

# Test Lightning Module
print("\n" + "=" * 60)
print("Testing MKSpatialMambaLitModule (Lightning Module)")
print("=" * 60)

print("\n3. Testing Lightning Module (Hybrid):")
lit_module = MKSpatialMambaLitModule(
    in_channels=6,
    num_classes=1,
    channels=[16, 32, 64, 96, 160],
    use_spatial=True,
    use_mamba=True,
    use_cbam=True,
    optimizer_lr=0.001,
    dice_weight=0.6,
    bce_weight=0.4,
)

# Test forward pass
x = torch.randn(2, 3, 256, 256)
y = lit_module(x)
print(f"   Input: {x.shape}")
print(f"   Output: {y.shape}")
print(f"   Params: {sum(p.numel() for p in lit_module.parameters()) / 1e6:.2f}M")

# Test training step
print("\n4. Testing training_step:")
# Create valid batch with masks in [0, 1] range
images = torch.randn(2, 3, 256, 256)
masks = torch.randint(0, 2, (2, 1, 256, 256)).float()  # Binary masks (0 or 1)
batch = (images, masks)
loss = lit_module.training_step(batch, 0)
print(f"   Loss: {loss.item():.4f}")
print(f"   Loss shape: {loss.shape}")

# Test validation step
print("\n5. Testing validation_step:")
val_loss = lit_module.validation_step(batch, 0)
print(f"   Val Loss: {val_loss.item():.4f}")

# Test optimizer configuration
print("\n6. Testing configure_optimizers:")
opt_config = lit_module.configure_optimizers()
print(f"   Optimizer: {type(opt_config['optimizer']).__name__}")
print(f"   Scheduler: {type(opt_config['lr_scheduler']['scheduler']).__name__}")
print(f"   Monitor metric: {opt_config['lr_scheduler']['monitor']}")

print("\n✅ Lightning module test passed!")
print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)