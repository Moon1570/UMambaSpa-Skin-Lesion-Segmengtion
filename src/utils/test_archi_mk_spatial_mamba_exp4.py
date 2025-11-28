# test_module.py
import torch
from src.models.mk_spatial_mamba_module import MKSpatialMambaLitModule, DiceMetric

# Test DiceMetric
print("Testing DiceMetric...")
dice_metric = DiceMetric()

# Perfect prediction
preds = torch.ones(2, 1, 256, 256)
target = torch.ones(2, 1, 256, 256)
dice_metric.update(preds, target)
score = dice_metric.compute()
print(f"Perfect match: {score:.4f} (should be ~1.0)")
assert score > 0.99, "Perfect prediction should give Dice ~1.0"

# Reset
dice_metric.reset()

# No overlap
preds = torch.ones(2, 1, 256, 256)
target = torch.zeros(2, 1, 256, 256)
dice_metric.update(preds, target)
score = dice_metric.compute()
print(f"No overlap: {score:.4f} (should be ~0.0)")
assert score < 0.01, "No overlap should give Dice ~0.0"

# Reset
dice_metric.reset()

# 50% overlap
preds = torch.cat([torch.ones(2, 1, 128, 256), torch.zeros(2, 1, 128, 256)], dim=2)
target = torch.cat([torch.ones(2, 1, 128, 256), torch.zeros(2, 1, 128, 256)], dim=2)
dice_metric.update(preds, target)
score = dice_metric.compute()
print(f"Perfect 50% overlap: {score:.4f} (should be ~1.0)")

print("\n✅ DiceMetric tests passed!")

# Test module instantiation
print("\nTesting MKSpatialMambaLitModule instantiation...")
net_config = {
    'in_channels': 6,
    'num_classes': 1,
    'channels': [16, 32, 64, 96, 160],
    'use_spatial': True,
    'use_mamba': False,  # Use CNN for quick test
    'use_cbam': True,
}

optimizer_config = {
    'lr': 0.001,
    'weight_decay': 0.0001,
}

criterion_config = {
    'dice_weight': 0.6,
    'bce_weight': 0.4,
}

module = MKSpatialMambaLitModule(
    net=net_config,
    optimizer=optimizer_config,
    criterion=criterion_config,
)

print(f"Module created successfully!")
print(f"Params: {sum(p.numel() for p in module.parameters()) / 1e6:.2f}M")

# Test forward pass
x = torch.randn(2, 3, 256, 256)
y = module(x)
print(f"Input: {x.shape}")
print(f"Output: {y.shape}")

print("\n✅ Module tests passed!")
