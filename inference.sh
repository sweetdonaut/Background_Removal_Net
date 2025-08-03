python inference.py \
    --model_path ./checkpoints/stripe_experiment/BgRemoval_lr0.0001_ep100_bs16_128x128_strip.pth \
    --test_path ./MVTec_AD_dataset/grid_stripe/test/ \
    --output_dir ./output/stripe_experiment \
    --gpu_id 0 \
    --img_format tiff \
    --image_type strip \
    --use_ground_truth_mask False