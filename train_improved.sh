#!/bin/bash

# Training script with improved dataloader (edge negative samples)
echo "Training with improved dataloader (edge negative samples)..."
echo "============================================"

python trainer.py \
    --bs 16 \
    --lr 0.0001 \
    --epochs 30 \
    --gpu_id 0 \
    --checkpoint_path ./checkpoints/improved_edge_neg \
    --image_type strip \
    --training_dataset_path ./MVTec_AD_dataset/grid_stripe/train/good/ \
    --img_format tiff \
    --use_mask True \
    --num_defects_range 3 8 \
    --seed 42 \
    --cache_size 200

echo ""
echo "Training completed!"
echo "Model saved to: ./checkpoints/improved_edge_neg/"
echo ""
echo "Next steps:"
echo "1. Run inference with the new model:"
echo "   python inference.py --model_path ./checkpoints/improved_edge_neg/BgRemoval_*.pth ..."
echo "2. Compare with diagnostic test:"
echo "   python compare_models.py"