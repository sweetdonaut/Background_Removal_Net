python inference_pytorch.py \
    --model_path ./checkpoints/square_3channel/BgRemoval_lr0.001_ep30_bs16_128x128_square.pth \
    --test_path ./MVTec_AD_dataset/grid_align_3channel/test/ \
    --output_dir ./output/pytorch_square \
    --img_format tiff \
    --image_type square
