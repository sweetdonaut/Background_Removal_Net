#!/bin/bash

# Example: Visualize feature maps for a specific patch

python visualize_feature_maps.py \
    --model_path ./checkpoints/4channel/BgRemoval_lr0.001_ep30_bs16_128x128_strip.pth \
    --image_path ./MVTec_AD_dataset/grid_stripe_4channel/test/bright_spots/250.tiff \
    --output_dir ./feature_maps \
    --patch_idx 3 1 \
    --img_format tiff \
    --image_type strip \
    --reduction_method all \
    --gpu_id 0

# Options:
# --patch_idx: Y_index X_index (e.g., 0 0 for first patch, 4 1 for middle-ish patch)
#              Strip: Y can be 0-8 (9 patches), X can be 0-1 (2 patches)
#              Square: Y can be 0-3 (4 patches), X can be 0-3 (4 patches)
# --reduction_method: mean, max, std, top_variance, or all
#   - mean: Average all channels (smooth, general view)
#   - max: Maximum across channels (highlights strongest activations)
#   - std: Standard deviation across channels (shows variation)
#   - top_variance: Select most informative channel (single channel view)
#   - all: Generate all four methods for comparison
