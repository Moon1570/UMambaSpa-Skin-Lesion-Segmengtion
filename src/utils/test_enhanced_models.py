"""Test script for enhanced MK U-Net experiments."""

import torch
from src.models.components.mk_enhanced_unet import MKEnhancedUNet
from src.models.mk_enhanced_module import MKEnhancedLitModule

print("=" * 70)
print("Testing Enhanced MK U-Net Configurations")
print("=" * 70)

# Test configurations
configs = [
    ("Baseline (no enhancements)", {"use_se": False, "deep_supervision": False}),
    ("Deep Supervision only", {"use_se": False, "deep_supervision": True}),
    ("Squeeze-and-Excitation only", {"use_se": True, "deep_supervision": False}),
    ("DS + SE (Full)", {"use_se": True, "deep_supervision": True}),
]

x = torch.randn(2, 3, 256, 256)

print("\n📊 Model Configurations:")
print("-" * 70)

for name, config in configs:
    print(f"\n{name}:")
    model = MKEnhancedUNet(
        in_channels=6,
        channels=[16, 32, 64, 96],
        use_spatial=True,
        **config
    )
    
    # Test forward pass
    model.train()
    output = model(x)
    
    # Handle deep supervision outputs
    if isinstance(output, tuple):
        main_out, aux_outs = output
        print(f"  Main output (logits): {main_out.shape}")
        print(f"  Aux outputs (logits): {[aux.shape for aux in aux_outs]}")
    else:
        print(f"  Output (logits): {output.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {params:.2f}M")

# Test Lightning module
print("\n" + "=" * 70)
print("Testing Lightning Module")
print("=" * 70)

lit_config = {
    "net": {
        "in_channels": 6,
        "num_classes": 1,
        "channels": [16, 32, 64, 96],
        "use_spatial": True,
        "use_se": True,
        "deep_supervision": True,
    },
    "optimizer": {
        "lr": 0.001,
        "weight_decay": 0.0001,
    },
    "criterion": {
        "dice_weight": 0.6,
        "bce_weight": 0.4,
        "aux_weight": 0.4,
    }
}

print("\n✅ Creating Lightning module with full config...")
lit_module = MKEnhancedLitModule(**lit_config)

# Test forward pass
print("\n✅ Testing forward pass...")
y = lit_module(x)
if isinstance(y, tuple):
    print(f"  Main output: {y[0].shape}")
    print(f"  Aux outputs: {[aux.shape for aux in y[1]]}")
else:
    print(f"  Output: {y.shape}")

# Test training step
print("\n✅ Testing training step...")
batch = (torch.randn(2, 3, 256, 256), torch.randint(0, 2, (2, 1, 256, 256)).float())
loss = lit_module.training_step(batch, 0)
print(f"  Training loss: {loss.item():.4f}")

# Test validation step
print("\n✅ Testing validation step...")
val_loss = lit_module.validation_step(batch, 0)
print(f"  Validation loss: {val_loss.item():.4f}")

# Test optimizer config
print("\n✅ Testing optimizer configuration...")
opt_config = lit_module.configure_optimizers()
print(f"  Optimizer: {type(opt_config['optimizer']).__name__}")
print(f"  Scheduler: {type(opt_config['lr_scheduler']['scheduler']).__name__}")
print(f"  Monitor: {opt_config['lr_scheduler']['monitor']}")

print("\n" + "=" * 70)
print("✅ All tests passed successfully!")
print("=" * 70)

print("\n📋 Experiment Configurations Created:")
print("-" * 70)
experiments = [
    ("exp100_baseline", "Baseline - no enhancements"),
    ("exp101_deep_supervision", "Deep Supervision only"),
    ("exp102_squeeze_excitation", "Squeeze-and-Excitation only"),
    ("exp103_spatial_encoding", "Spatial encoding only"),
    ("exp104_ds_se_combo", "DS + SE combination"),
    ("exp105_spatial_se", "Spatial + SE combination"),
    ("exp106_spatial_ds", "Spatial + DS combination"),
    ("exp1XX_full_enhanced", "FULL MODEL - All enhancements"),
]

for exp_name, description in experiments:
    print(f"  ✅ {exp_name:30s} - {description}")

print("\n🚀 Ready to train! Example commands:")
print("-" * 70)
print("  # Baseline")
print("  python src/train.py experiment=exp100_baseline")
print("\n  # Individual enhancements")
print("  python src/train.py experiment=exp101_deep_supervision")
print("  python src/train.py experiment=exp102_squeeze_excitation")
print("\n  # Full model")
print("  python src/train.py experiment=exp1XX_full_enhanced")
print("\n  # Quick test")
print("  python src/train.py experiment=exp100_baseline trainer.fast_dev_run=True")
print("=" * 70)
