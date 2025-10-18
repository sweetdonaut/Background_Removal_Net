python inference_onnx.py \
    --model_path ./onnx_models/background_removal_gamma_schedule_fullimage.onnx \
    --test_path ./MVTec_AD_dataset/grid_stripe_4channel/test/ \
    --output_dir ./output/onnx \
    --img_format tiff \
    --image_type strip
