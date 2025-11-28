#!/bin/bash
# Quick training script for all enhanced experiments

echo "================================"
echo "Enhanced MK U-Net Experiments"
echo "================================"
echo ""

# Activate environment
source .ubuvenv/bin/activate

echo "🧪 Available Experiments:"
echo "  exp100 - Baseline (no enhancements)"
echo "  exp101 - Deep Supervision"
echo "  exp102 - Squeeze-and-Excitation"
echo "  exp103 - Spatial Encoding"
echo "  exp104 - DS + SE"
echo "  exp105 - Spatial + SE"
echo "  exp106 - Spatial + DS"
echo "  exp1XX - FULL MODEL (all enhancements)"
echo ""

# Parse command line argument
if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/train_enhanced.sh <experiment_number>"
    echo "Example: ./scripts/train_enhanced.sh 100"
    echo "Example: ./scripts/train_enhanced.sh 1XX"
    exit 1
fi

EXP_NUM=$1

case $EXP_NUM in
    100)
        echo "🚀 Training exp100_baseline..."
        python src/train.py experiment=exp100_baseline
        ;;
    101)
        echo "🚀 Training exp101_deep_supervision..."
        python src/train.py experiment=exp101_deep_supervision
        ;;
    102)
        echo "🚀 Training exp102_squeeze_excitation..."
        python src/train.py experiment=exp102_squeeze_excitation
        ;;
    103)
        echo "🚀 Training exp103_spatial_encoding..."
        python src/train.py experiment=exp103_spatial_encoding
        ;;
    104)
        echo "🚀 Training exp104_ds_se_combo..."
        python src/train.py experiment=exp104_ds_se_combo
        ;;
    105)
        echo "🚀 Training exp105_spatial_se..."
        python src/train.py experiment=exp105_spatial_se
        ;;
    106)
        echo "🚀 Training exp106_spatial_ds..."
        python src/train.py experiment=exp106_spatial_ds
        ;;
    1XX|1xx)
        echo "🚀 Training exp1XX_full_enhanced (ALL FEATURES)..."
        python src/train.py experiment=exp1XX_full_enhanced
        ;;
    *)
        echo "❌ Unknown experiment: $EXP_NUM"
        echo "Valid options: 100, 101, 102, 103, 104, 105, 106, 1XX"
        exit 1
        ;;
esac
