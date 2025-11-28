"""Estimate GPU memory requirements for MKSpatialMambaUNet."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models.components.mk_spatial_mamba_unet import MKSpatialMambaUNet


def format_bytes(bytes_val):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def count_parameters(model):
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def estimate_memory(model, batch_size=4, image_size=(256, 256), in_channels=6):
    """
    Estimate GPU memory usage for training.
    
    Memory breakdown:
    1. Model parameters (weights + gradients + optimizer states)
    2. Forward activations
    3. Backward gradients
    4. Input batch
    """
    
    print("=" * 70)
    print("GPU MEMORY ESTIMATION FOR MKSpatialMambaUNet")
    print(f"Batch size: {batch_size}, Image size: {image_size}, In channels: {in_channels}")
    print("=" * 70)
    
    # 1. Model Parameters
    total_params, trainable_params = count_parameters(model)
    
    # FP32: 4 bytes per parameter
    # Gradients: 4 bytes per parameter (same as weights)
    # Adam optimizer states: 8 bytes per parameter (momentum + variance)
    params_memory = total_params * 4  # weights
    gradients_memory = trainable_params * 4  # gradients
    optimizer_memory = trainable_params * 8  # Adam states
    
    model_total_memory = params_memory + gradients_memory + optimizer_memory
    
    print(f"\n📊 MODEL STATISTICS:")
    print(f"  Total parameters: {total_params:,} ({format_bytes(total_params * 4)})")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    print(f"\n💾 MODEL MEMORY (FP32):")
    print(f"  Weights:           {format_bytes(params_memory)}")
    print(f"  Gradients:         {format_bytes(gradients_memory)}")
    print(f"  Optimizer states:  {format_bytes(optimizer_memory)}")
    print(f"  ─────────────────────────────────")
    print(f"  Total model:       {format_bytes(model_total_memory)}")
    
    # 2. Input Batch Memory
    input_elements = batch_size * in_channels * image_size[0] * image_size[1]
    mask_elements = batch_size * 1 * image_size[0] * image_size[1]
    
    input_memory = input_elements * 4  # FP32
    mask_memory = mask_elements * 4
    batch_memory = input_memory + mask_memory
    
    print(f"\n📥 INPUT BATCH MEMORY:")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}")
    print(f"  Input channels: {in_channels}")
    print(f"  Images:  {format_bytes(input_memory)}")
    print(f"  Masks:   {format_bytes(mask_memory)}")
    print(f"  ─────────────────────────────────")
    print(f"  Total batch: {format_bytes(batch_memory)}")
    
    # 3. Activation Memory (rough estimate)
    # Encoder: e1, e2, e3, e4 + bottleneck
    # Decoder: d4, d3, d2, d1
    # Each layer stores feature maps for backprop
    
    h, w = image_size
    channels = [16, 32, 64, 96, 160]
    
    # Encoder activations
    e1_mem = batch_size * channels[0] * h * w * 4
    e2_mem = batch_size * channels[1] * (h//2) * (w//2) * 4
    e3_mem = batch_size * channels[2] * (h//4) * (w//4) * 4
    e4_mem = batch_size * channels[3] * (h//8) * (w//8) * 4
    b_mem = batch_size * channels[4] * (h//16) * (w//16) * 4
    
    # Decoder activations (similar sizes)
    d4_mem = batch_size * channels[3] * (h//8) * (w//8) * 4
    d3_mem = batch_size * channels[2] * (h//4) * (w//4) * 4
    d2_mem = batch_size * channels[1] * (h//2) * (w//2) * 4
    d1_mem = batch_size * channels[0] * h * w * 4
    
    # Concatenated features (decoder input after skip connections)
    concat_mem = (
        batch_size * channels[3] * 2 * (h//8) * (w//8) * 4 +
        batch_size * channels[2] * 2 * (h//4) * (w//4) * 4 +
        batch_size * channels[1] * 2 * (h//2) * (w//2) * 4 +
        batch_size * channels[0] * 2 * h * w * 4
    )
    
    activation_memory = (e1_mem + e2_mem + e3_mem + e4_mem + b_mem +
                        d4_mem + d3_mem + d2_mem + d1_mem + concat_mem)
    
    print(f"\n🔥 ACTIVATION MEMORY (Forward Pass):")
    print(f"  Encoder activations: {format_bytes(e1_mem + e2_mem + e3_mem + e4_mem)}")
    print(f"  Bottleneck:          {format_bytes(b_mem)}")
    print(f"  Decoder activations: {format_bytes(d4_mem + d3_mem + d2_mem + d1_mem)}")
    print(f"  Skip connections:    {format_bytes(concat_mem)}")
    print(f"  ─────────────────────────────────")
    print(f"  Total activations:   {format_bytes(activation_memory)}")
    
    # 4. Backward pass (gradient tensors for activations)
    # Roughly same as forward activations
    backward_memory = activation_memory
    
    print(f"\n⬅️  BACKWARD PASS MEMORY:")
    print(f"  Gradient tensors:    {format_bytes(backward_memory)}")
    
    # 5. Total Memory
    total_memory = model_total_memory + batch_memory + activation_memory + backward_memory
    
    # Add 20% overhead for PyTorch internal buffers, workspace, etc.
    overhead = total_memory * 0.2
    total_with_overhead = total_memory + overhead
    
    print(f"\n" + "=" * 70)
    print(f"📊 TOTAL GPU MEMORY ESTIMATE:")
    print(f"=" * 70)
    print(f"  Model (weights + gradients + optimizer): {format_bytes(model_total_memory)}")
    print(f"  Input batch:                             {format_bytes(batch_memory)}")
    print(f"  Forward activations:                     {format_bytes(activation_memory)}")
    print(f"  Backward gradients:                      {format_bytes(backward_memory)}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Subtotal:                                {format_bytes(total_memory)}")
    print(f"  PyTorch overhead (20%):                  {format_bytes(overhead)}")
    print(f"  ═════════════════════════════════════════════════════")
    print(f"  TOTAL ESTIMATED:                         {format_bytes(total_with_overhead)}")
    print(f"=" * 70)
    
    # GPU Recommendations
    print(f"\n💡 GPU RECOMMENDATIONS:")
    total_gb = total_with_overhead / (1024**3)
    
    if total_gb < 4:
        print(f"  ✅ Can run on: GTX 1650 (4GB), RTX 3050 (4GB)")
        print(f"  ✅ Comfortable on: RTX 3060 (12GB), RTX 4060 (8GB)")
    elif total_gb < 6:
        print(f"  ⚠️  Minimum: RTX 3060 (12GB), RTX 4060 (8GB)")
        print(f"  ✅ Comfortable on: RTX 3070 (8GB), RTX 4070 (12GB)")
    elif total_gb < 8:
        print(f"  ⚠️  Minimum: RTX 3070 (8GB), RTX 4070 (12GB)")
        print(f"  ✅ Comfortable on: RTX 3080 (10GB), RTX 4080 (16GB)")
    elif total_gb < 12:
        print(f"  ⚠️  Minimum: RTX 3080 (10GB), RTX 3090 (24GB)")
        print(f"  ✅ Comfortable on: RTX 4090 (24GB), A100 (40GB)")
    else:
        print(f"  ⚠️  Requires: RTX 3090 (24GB), A100 (40GB)")
        print(f"  ⚠️  Consider reducing batch size")
    
    print(f"\n📝 NOTES:")
    print(f"  • This is an ESTIMATE based on batch_size={batch_size}")
    print(f"  • Actual usage may vary ±20%")
    print(f"  • Mixed precision (FP16) can reduce memory by ~40%")
    print(f"  • Gradient accumulation can reduce batch size requirements")
    print(f"  • fast_dev_run uses smaller batches (should be fine)")
    
    return total_with_overhead


if __name__ == "__main__":
    # Configuration from exp4_hybrid_mamba
    batch_size = 4
    image_size = (256, 256)
    in_channels = 6  # RGB + X + Y + Radial
    channels = [16, 32, 64, 96, 160]
    
    print("\n🔧 Building model...")
    model = MKSpatialMambaUNet(
        in_channels=in_channels,
        num_classes=1,
        channels=channels,
        use_spatial=True,
        use_mamba=True,
        use_cbam=True
    )
    
    print("✅ Model built successfully!\n")
    
    # Estimate memory
    total_memory = estimate_memory(
        model=model,
        batch_size=batch_size,
        image_size=image_size,
        in_channels=in_channels
    )
    
    print("\n" + "=" * 70)
