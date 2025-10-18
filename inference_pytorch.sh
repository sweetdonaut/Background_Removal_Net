python inference_pytorch.py \
    --model_path ./checkpoints/4channel/BgRemoval_lr0.001_ep30_bs16_128x128_strip.pth \
    --test_path ./MVTec_AD_dataset/grid_stripe_4channel/test/ \
    --output_dir ./output/pytorch \
    --img_format tiff \
    --image_type strip
